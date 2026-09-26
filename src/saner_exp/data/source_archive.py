"""Stream an author tar.gz into an audited, selected-version staging directory."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tarfile
import tempfile
import time


def member_path(name: str) -> PurePosixPath:
    while name.startswith("./"):
        name = name[2:]
    name = name.rstrip("/")
    if not name or name.startswith("/") or "\\" in name or "\0" in name:
        raise ValueError(f"unsafe archive path: {name!r}")
    if any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"noncanonical archive path: {name!r}")
    path = PurePosixPath(name)
    if not re.fullmatch(r"linux-[0-9][0-9.a-z-]*", path.parts[0]):
        raise ValueError(f"unexpected top-level source directory: {path.parts[0]!r}")
    return path


def link_target(path: PurePosixPath, target: str, *, hardlink: bool) -> PurePosixPath:
    if not target or target.startswith("/") or "\\" in target or "\0" in target:
        raise ValueError(f"unsafe link target at {path}")
    parts = [] if hardlink else list(path.parent.parts)
    for part in target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"link escapes archive at {path}")
            parts.pop()
        else:
            parts.append(part)
    resolved = PurePosixPath(*parts)
    if not parts or parts[0] != path.parts[0]:
        raise ValueError(f"link escapes version at {path}")
    return resolved


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data

    def tell(self):
        return self.count


class ChunkedArchive:
    """Read completed, checksummed download ranges in order as they arrive."""

    def __init__(self, directory: Path, size: int):
        self.directory = directory
        identity = json.loads((directory / "identity.json").read_bytes())
        if identity["size"] != size:
            raise ValueError("download-part size identity mismatch")
        self.size = size
        self.chunk_size = identity["chunk_bytes"]

    def open(self, mode):
        if mode != "rb":
            raise ValueError("download parts are read-only")
        return ChunkReader(self)


class ChunkReader:
    def __init__(self, source):
        self.source = source
        self.position = 0
        self.stream = None
        self.chunk_digest = None
        self.expected_digest = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.close()

    def read(self, size):
        if size < 0:
            raise ValueError("chunk stream requires bounded reads")
        data = bytearray()
        while len(data) < size and self.position < self.source.size:
            index = self.position // self.source.chunk_size
            chunk_end = min((index + 1) * self.source.chunk_size, self.source.size)
            if self.stream is None:
                path = self.source.directory / f"{index:04d}.chunk"
                checksum = self.source.directory / f"{index:04d}.sha256"
                if not path.exists():
                    print(f"Waiting for completed download range {index}", flush=True)
                deadline = time.monotonic() + 7200
                while not path.exists() or not checksum.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"download range {index} did not arrive")
                    time.sleep(1)
                if path.stat().st_size != chunk_end - index * self.source.chunk_size:
                    raise ValueError(f"wrong size for completed range {index}")
                self.expected_digest = checksum.read_text().strip()
                self.chunk_digest = hashlib.sha256()
                self.stream = path.open("rb")
            block = self.stream.read(min(size-len(data), chunk_end-self.position))
            if not block:
                raise ValueError("truncated completed download range")
            self.chunk_digest.update(block)
            data.extend(block)
            self.position += len(block)
            if self.position == chunk_end:
                self.stream.close()
                self.stream = None
                if self.chunk_digest.hexdigest() != self.expected_digest:
                    raise ValueError(f"completed download range {index} hash mismatch")
        return bytes(data)


def parent_directories(root: Path, path: PurePosixPath):
    current = root
    for part in path.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"archive member traverses a symlink: {path}")
        current.mkdir(exist_ok=True)


def makefile_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict")
    values = dict(re.findall(r"^(VERSION|PATCHLEVEL|SUBLEVEL|EXTRAVERSION)[ \t]*=[ \t]*([^\r\n]*)", text, re.M))
    if any(key not in values for key in ("VERSION", "PATCHLEVEL", "SUBLEVEL", "EXTRAVERSION")):
        raise ValueError("incomplete root Makefile version fields")
    major, minor, sub = (int(values[k].strip()) for k in ("VERSION", "PATCHLEVEL", "SUBLEVEL"))
    base = f"{major}.{minor}.{sub}" if major == 2 or sub != 0 else f"{major}.{minor}"
    return base + values["EXTRAVERSION"].strip()


def prepare_archive(archive: Path | ChunkedArchive, receipt: dict, inventory: dict, destination: Path,
                    *, completed_receipt_path: Path | None = None,
                    partition: str = "preexperiment") -> dict:
    """Never execute archived content; publish only after full stream verification.

    Run on a Linux filesystem. SQLite bounds memory for millions of paths.
    Failed attempts remain under a uniquely named staging directory for inspection.
    """
    if os.name != "posix":
        raise ValueError("source extraction requires the checked Linux container")
    if partition not in ("preexperiment", "main"):
        raise ValueError("partition must be preexperiment or main")
    expected = {"linux-" + e["kernel_version"] for e in inventory["versions"]}
    task_field = f"{partition}_task_ids"
    selected = {
        "linux-" + e["kernel_version"] for e in inventory["versions"]
        if e[task_field]
    }
    if not selected:
        raise ValueError(f"no {partition} versions selected")
    destination.mkdir(parents=True, exist_ok=True)
    prefix = "author-" if partition == "preexperiment" else "author-main-"
    final = destination / (prefix + receipt["sha256"][:16]) if receipt.get("sha256") else None
    if final is not None and final.exists():
        raise ValueError("published source directory already exists; no overwrite")
    stage = Path(tempfile.mkdtemp(prefix=".author-staging-", dir=destination))
    versions_root = stage / "versions"
    versions_root.mkdir()
    print(f"Staging at {stage}", flush=True)
    db = sqlite3.connect(stage / "catalog.sqlite")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("CREATE TABLE members (path TEXT PRIMARY KEY, version TEXT, type TEXT, size INTEGER, mode INTEGER, link TEXT, sha256 TEXT)")
    stats = {}
    types = Counter()
    count = total_size = excluded_git = 0
    last_report = time.monotonic()
    deferred_links = []
    directory_modes = {}
    try:
        with archive.open("rb") as raw:
            hashed = HashingReader(raw)
            with gzip.GzipFile(fileobj=hashed, mode="rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|", ignore_zeros=True) as tar:
                    for member in tar:
                        path = member_path(member.name)
                        version = path.parts[0]
                        kind = member.type.decode("ascii", errors="replace")
                        if member.size < 0:
                            raise ValueError("negative archive member size")
                        try:
                            db.execute("INSERT INTO members VALUES (?,?,?,?,?,?,NULL)",
                                       (str(path), version, kind, member.size, member.mode, member.linkname))
                        except sqlite3.IntegrityError:
                            raise ValueError(f"duplicate normalized archive member: {path}") from None
                        count += 1
                        types[kind] += 1
                        s = stats.setdefault(version, {"members": 0, "regular_bytes": 0, "selected": version in selected})
                        s["members"] += 1
                        if member.isfile():
                            s["regular_bytes"] += member.size
                            total_size += member.size
                        if total_size > 300 * 1024**3 or count > 12_000_000:
                            raise ValueError("archive exceeds preparation limits")
                        target = None
                        if member.issym() or member.islnk():
                            target = link_target(path, member.linkname, hardlink=member.islnk())
                        if ".git" in path.parts:
                            excluded_git += 1
                        elif version in selected:
                            parent_directories(versions_root, path)
                            output = versions_root.joinpath(*path.parts)
                            if member.isdir():
                                output.mkdir(exist_ok=True)
                                directory_modes[str(path)] = member.mode & 0o777
                            elif member.isfile():
                                if member.sparse is not None:
                                    raise ValueError(f"sparse source member needs explicit handling: {path}")
                                content = tar.extractfile(member)
                                h = hashlib.sha256()
                                written = 0
                                with output.open("xb") as out:
                                    while block := content.read(1024 * 1024):
                                        out.write(block)
                                        h.update(block)
                                        written += len(block)
                                if written != member.size:
                                    raise ValueError(f"truncated source member: {path}")
                                output.chmod(member.mode & 0o777)
                                db.execute("UPDATE members SET sha256=? WHERE path=?", (h.hexdigest(), str(path)))
                            elif member.issym():
                                # Linking is safe only within the same version; reads never follow links during extraction.
                                output.symlink_to(member.linkname)
                            elif member.islnk():
                                deferred_links.append((path, target))
                            else:
                                raise ValueError(f"unsupported source member type {kind} at {path}")
                        if count % 20000 == 0:
                            db.commit()
                        now = time.monotonic()
                        if now - last_report >= 25:
                            print(f"Scanned {count} members; {len(stats)} versions; {total_size/1024**3:.2f} GiB regular data; compressed {hashed.count/receipt['size_bytes']:.1%}", flush=True)
                            last_report = now
                        tar.members.clear()
                # Consume through the gzip trailer(s), verifying CRC and ISIZE.
                while compressed.read(1024 * 1024):
                    pass
            if not receipt.get("sha256"):
                if completed_receipt_path is None:
                    raise ValueError("complete archive receipt is required before publication")
                print("Full stream scanned; waiting for the independently assembled download receipt", flush=True)
                deadline = time.monotonic() + 600
                while True:
                    try:
                        complete_receipt = json.loads(completed_receipt_path.read_bytes())
                        break
                    except (FileNotFoundError, json.JSONDecodeError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("complete download receipt unavailable")
                        time.sleep(1)
                if complete_receipt["size_bytes"] != receipt["size_bytes"]:
                    raise ValueError("completed receipt size changed")
                receipt = complete_receipt
                final = destination / (prefix + receipt["sha256"][:16])
                if final.exists():
                    raise ValueError("published source directory already exists")
            if hashed.count != receipt["size_bytes"] or hashed.digest.hexdigest() != receipt["sha256"]:
                raise ValueError("full compressed archive does not match download receipt")
        missing = expected - set(stats)
        if selected - set(stats):
            raise ValueError(f"missing selected versions: {sorted(selected-set(stats))}")
        for path, target in deferred_links:
            source = versions_root.joinpath(*target.parts)
            parent_directories(versions_root, target)
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"hardlink target is not an extracted regular file: {path}")
            os.link(source, versions_root.joinpath(*path.parts))
        for path, mode in sorted(directory_modes.items(), key=lambda item: item[0].count("/"), reverse=True):
            versions_root.joinpath(*PurePosixPath(path).parts).chmod(mode)
        snapshots = []
        version_identity_mismatches = []
        for version in sorted(selected):
            actual = makefile_version(versions_root / version / "Makefile")
            expected_version = version.removeprefix("linux-")
            if actual != expected_version:
                version_identity_mismatches.append({
                    "directory_kernel_version": expected_version,
                    "makefile_kernel_version": actual,
                    "policy": "retain_verbatim_author_package_tree_and_record_mismatch",
                })
            tree = hashlib.sha256()
            for row in db.execute("SELECT path,type,size,mode,link,sha256 FROM members WHERE version=? ORDER BY path", (version,)):
                if ".git" in PurePosixPath(row[0]).parts:
                    continue
                line = [str(PurePosixPath(row[0]).relative_to(version)), *row[1:]]
                tree.update((json.dumps(line, ensure_ascii=True, separators=(",", ":")) + "\n").encode())
            snapshots.append({"kernel_version": expected_version,
                              "makefile_kernel_version": actual,
                              "relative_path": f"versions/{version}",
                              "members_in_author_archive": stats[version]["members"],
                              "regular_bytes": stats[version]["regular_bytes"],
                              "catalog_tree_sha256": tree.hexdigest(),
                              "upstream_equivalence": "not_checked"})
        db.commit()
        db.close()
        record = {
            "record_kind": "author_source_preparation",
            "partition": partition,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": receipt["sha256"], "archive_size_bytes": receipt["size_bytes"],
            "gzip_crc_and_size_verified": True, "complete_stream_scanned": True,
            "member_count": count, "member_types": dict(types), "regular_bytes": total_size,
            "archive_version_count": len(stats), "archive_versions": dict(sorted(stats.items())),
            "missing_expected_versions": sorted(missing), "extra_versions": sorted(set(stats)-expected),
            "excluded_git_metadata_members": excluded_git,
            "snapshot_count": len(snapshots), "snapshots": snapshots,
            "version_identity_mismatches": version_identity_mismatches,
            "source_root": str(final), "catalog": "catalog.sqlite",
            "catalog_hash_encoding": "sorted relative-path,type,size,original-mode,link,regular-file-sha256 arrays as compact ASCII JSON plus LF; .git members excluded",
            "mode_policy": "preserve permission bits 0777; do not restore owner or special permission bits",
            "status": (
                "author_package_prepared_with_recorded_directory_version_mismatches"
                if version_identity_mismatches
                else "author_package_prepared_upstream_comparison_pending"
            ),
        }
        (stage / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        stage.rename(final)
        return record
    except Exception:
        db.close()
        print(f"Preparation failed. Unpublished staging retained at {stage}", flush=True)
        raise

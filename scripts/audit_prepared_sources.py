"""Read-only verification of prepared Linux sources against their archive catalog."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from saner_exp.data.source_archive import link_target, makefile_version, member_path


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(source_root):
    if os.name != "posix":
        raise ValueError("source audit requires a Linux filesystem")
    record = json.loads((source_root / "preparation.json").read_bytes())
    recorded_mismatches = {
        item["directory_kernel_version"]: item
        for item in record.get("version_identity_mismatches", [])
    }
    selected = {"linux-" + snapshot["kernel_version"] for snapshot in record["snapshots"]}
    if len(selected) != record["snapshot_count"] or {p.name for p in (source_root / "versions").iterdir()} != selected:
        raise ValueError("prepared version directory set mismatch")
    catalog = source_root / "catalog.sqlite"
    db = sqlite3.connect(catalog.as_uri() + "?mode=ro", uri=True)
    results = []
    try:
        for snapshot in record["snapshots"]:
            version = "linux-" + snapshot["kernel_version"]
            root = source_root / "versions" / version
            rows = list(db.execute(
                "SELECT path,type,size,mode,link,sha256 FROM members WHERE version=? ORDER BY path",
                (version,)))
            rows = [row for row in rows if ".git" not in PurePosixPath(row[0]).parts]
            expected = {"."}
            for row in rows:
                relative = member_path(row[0]).relative_to(version)
                expected.add(str(relative))
                expected.update(str(parent) for parent in relative.parents)
            actual = {".": root.lstat().st_mode}
            for directory, dirs, files in os.walk(root, followlinks=False):
                for name in dirs + files:
                    path = Path(directory) / name
                    actual[str(path.relative_to(root))] = path.lstat().st_mode
            if set(actual) != expected:
                raise ValueError(f"source path set mismatch: {version}; missing={len(expected-set(actual))}, extra={len(set(actual)-expected)}")
            explicit = {str(PurePosixPath(row[0]).relative_to(version)) for row in rows}
            for relative in expected - explicit:
                if not stat.S_ISDIR(actual[relative]):
                    raise ValueError(f"implicit source parent is not a directory: {version}/{relative}")
            tree = hashlib.sha256()
            regular_files = regular_bytes = 0
            for path, kind, size, mode, target, digest in rows:
                relative = str(PurePosixPath(path).relative_to(version))
                stored = root / relative
                stored_mode = actual[relative]
                valid_type = ((kind in ("0", "\0", "1") and stat.S_ISREG(stored_mode))
                              or (kind == "5" and stat.S_ISDIR(stored_mode))
                              or (kind == "2" and stat.S_ISLNK(stored_mode)))
                if not valid_type:
                    raise ValueError(f"source member type mismatch: {path}")
                if kind in ("0", "\0"):
                    if stored.stat().st_size != size or file_sha256(stored) != digest:
                        raise ValueError(f"source file checksum mismatch: {path}")
                    regular_files += 1
                    regular_bytes += size
                elif kind == "2":
                    if os.readlink(stored) != target:
                        raise ValueError(f"source symlink mismatch: {path}")
                elif kind == "1":
                    resolved = link_target(PurePosixPath(path), target, hardlink=True)
                    if not stored.samefile(source_root / "versions" / str(resolved)):
                        raise ValueError(f"source hardlink mismatch: {path}")
                if kind in ("0", "\0", "5") and stat.S_IMODE(stored_mode) != mode & 0o777:
                    raise ValueError(f"source permission mismatch: {path}")
                tree.update((json.dumps([relative, kind, size, mode, target, digest],
                                       ensure_ascii=True, separators=(",", ":")) + "\n").encode())
            if tree.hexdigest() != snapshot["catalog_tree_sha256"]:
                raise ValueError(f"source catalog tree mismatch: {version}")
            actual_makefile_version = makefile_version(root / "Makefile")
            expected_makefile_version = snapshot.get(
                "makefile_kernel_version", snapshot["kernel_version"],
            )
            if actual_makefile_version != expected_makefile_version:
                raise ValueError(f"source Makefile identity differs from preparation record: {version}")
            if actual_makefile_version != snapshot["kernel_version"]:
                mismatch = recorded_mismatches.get(snapshot["kernel_version"])
                if not mismatch or mismatch.get("makefile_kernel_version") != actual_makefile_version:
                    raise ValueError(f"unrecorded source Makefile mismatch: {version}")
            results.append({"kernel_version": snapshot["kernel_version"],
                            "makefile_kernel_version": actual_makefile_version,
                            "regular_files": regular_files,
                            "regular_bytes": regular_bytes, "catalog_tree_sha256": tree.hexdigest()})
            print(f"Verified {version}: {regular_files} regular files", file=sys.stderr, flush=True)
    finally:
        db.close()
    if len(results) != record["snapshot_count"]:
        raise ValueError("snapshot count mismatch")
    return {"record_kind": "prepared_source_storage_audit", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root), "archive_sha256": record["archive_sha256"],
            "version_identity_mismatches": list(recorded_mismatches.values()),
            "preparation_sha256": file_sha256(source_root / "preparation.json"),
            "catalog_sha256": file_sha256(catalog), "snapshot_count": len(results), "snapshots": results,
            "status": "stored_sources_match_preparation_catalog", "upstream_equivalence": "not_checked"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source_root.resolve()), indent=2))


if __name__ == "__main__":
    main()

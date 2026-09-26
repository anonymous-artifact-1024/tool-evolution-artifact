"""Build a deterministic C-function index from an exact upstream Git tree export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saner_exp.data.kernel_versions import parse_version
from saner_exp.tools.function_index import make_index_record


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_files(archive: Path):
    previous = None
    selected = 0
    with tarfile.open(archive, "r:") as tree:
        for member in tree:
            if not member.isfile() or not member.name.endswith((".c", ".h")):
                continue
            path = PurePosixPath(member.name)
            if (not member.name or member.name.startswith("/") or "\\" in member.name
                    or any(part in ("", ".", "..", ".git") for part in path.parts)):
                raise ValueError(f"unsafe or noncanonical Git tree path: {member.name!r}")
            normalized = path.as_posix()
            if previous is not None and normalized <= previous:
                raise ValueError("Git tree members are not unique and path-sorted")
            previous = normalized
            stream = tree.extractfile(member)
            content = stream.read()
            if len(content) != member.size:
                raise ValueError(f"truncated archive member: {normalized}")
            selected += 1
            if selected % 1000 == 0:
                print(f"Read {selected} C/header files; latest={normalized}", flush=True)
            yield normalized, content


def build(archive: Path, version: str) -> dict:
    parse_version(version)
    identity = {
        "kind": "exact_upstream_git_tree_export",
        "kernel_version": version,
        "tag": "v" + version,
        "archive_name": archive.name,
        "archive_sha256": archive_sha256(archive),
        "tree_sitter_version": importlib.metadata.version("tree-sitter"),
        "tree_sitter_c_version": importlib.metadata.version("tree-sitter-c"),
    }
    record = make_index_record(source_files(archive), source_identity=identity)
    definition_count = sum(len(item["definitions"]) for item in record["files"].values())
    record["file_count"] = len(record["files"])
    record["definition_count"] = definition_count
    record["complete_definition_count"] = sum(
        definition["complete"] for item in record["files"].values() for definition in item["definitions"]
    )
    record["unsupported_function_node_count"] = sum(
        item["unsupported_function_nodes"] for item in record["files"].values()
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--kernel-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = build(args.archive, args.kernel_version)
    encoded = (json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if args.check:
        if args.output.read_bytes() != encoded:
            raise SystemExit("function index differs from the deterministic rebuild")
        print(f"Verified {args.output}: {record['file_count']} files, {record['definition_count']} definitions")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing index: {args.output}")
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    temporary.replace(args.output)
    print(f"Created {args.output}: {record['file_count']} files, {record['definition_count']} definitions")


if __name__ == "__main__":
    main()

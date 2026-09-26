"""Create or check the recorded seed-2027 20/230 split; never overwrite it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saner_exp.data import DatasetValidationError, LinuxFLBenchDataset
from saner_exp.data.split import (
    ID_ORDER, RANDOM_ALGORITHM, SPLIT_ALGORITHM_VERSION, split_task_ids,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def id_list_sha256(ids: tuple[str, ...]) -> str:
    """Canonical list encoding: UTF-8, one ID per line, including final LF."""
    return sha256(("\n".join(ids) + "\n").encode("utf-8"))


def build_record(dataset_path: Path, manifest_path: Path) -> dict:
    dataset = LinuxFLBenchDataset.load(dataset_path, manifest_path=manifest_path)
    if len(dataset) != 250:
        raise DatasetValidationError("this protocol requires exactly 250 tasks")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = split_task_ids(dataset.task_ids, seed=2027, preexperiment_size=20)
    pre, main = partition.preexperiment, partition.main
    if len(pre) != 20 or len(main) != 230 or set(pre) & set(main):
        raise ValueError("invalid partition sizes or overlap")
    if set(pre) | set(main) != set(dataset.task_ids):
        raise ValueError("partition does not cover the complete dataset")

    source_files = [
        "scripts/generate_split.py",
        "src/saner_exp/data/split.py",
        "src/saner_exp/data/linuxflbench.py",
        "src/saner_exp/data/task.py",
    ]
    return {
        "schema_version": 1,
        "record_kind": "task_partition",
        "protocol_status": "planned_not_formally_frozen",
        "dataset": {
            "dataset_name": manifest["dataset_name"],
            "upstream_repository": manifest["upstream_repository"],
            "git_commit_recorded_in_dataset_manifest": manifest["git_commit"],
            "dataset_file": manifest["dataset_file"],
            "sha256": dataset.sha256,
            "task_count": len(dataset),
            "dataset_manifest_sha256": sha256(manifest_path.read_bytes()),
        },
        "method": {
            "version": SPLIT_ALGORITHM_VERSION,
            "seed": 2027,
            "id_order": ID_ORDER,
            "random_algorithm": RANDOM_ALGORITHM,
            "assignment": "first 20 shuffled IDs: preexperiment; remaining 230: main",
            "stratification": "none",
            "list_order": "shuffled partition order; not episode execution order",
            "id_list_hash_encoding": "UTF-8; one ID per line; LF; final LF included",
        },
        "implementation": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "random_module_sha256": sha256(Path(random.__file__).read_bytes()),
            "source_sha256": {
                name: sha256((PROJECT_ROOT / name).read_bytes()) for name in source_files
            },
        },
        "partitions": {
            "preexperiment": {
                "task_count": len(pre), "task_ids": list(pre),
                "id_list_sha256": id_list_sha256(pre),
            },
            "main": {
                "task_count": len(main), "task_ids": list(main),
                "id_list_sha256": id_list_sha256(main),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=PROJECT_ROOT.parent / "datasets/LinuxFLBench/dataset/LINUXFLBENCH_dataset.jsonl",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "data/provenance/dataset_manifest.json",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/tasks/task_split.json",
    )
    parser.add_argument("--check", action="store_true", help="verify existing record without writing")
    args = parser.parse_args()
    try:
        record = build_record(args.dataset, args.manifest)
        expected = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if args.check:
            if args.output.read_bytes() != expected:
                raise ValueError("existing split or its provenance differs; file left unchanged")
            status = "verified"
        else:
            # Exclusive creation prevents an accidental or concurrent overwrite.
            try:
                with args.output.open("xb") as stream:
                    stream.write(expected)
                status = "created"
            except FileExistsError:
                if args.output.read_bytes() != expected:
                    raise ValueError("existing split or its provenance differs; file left unchanged")
                status = "verified_existing"
        print(json.dumps({
            "status": status, "record": str(args.output.resolve()),
            "preexperiment_tasks": 20, "main_tasks": 230,
            "overlap": 0, "total_tasks": 250, "seed": 2027,
        }, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Task split failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

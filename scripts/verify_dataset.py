"""Verify the pinned LinuxFLBench input without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

EXPECTED_COMMIT = "de65c3958f264137835eb23cacf3591fb89fc09b"
EXPECTED_ROWS = 250
EXPECTED_BYTES = 1_587_676
EXPECTED_SHA256 = "cc60e91b37a78d1cfb85f12c0bf884e41362afe798ac7ca2fa83e4b27bde974c"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    root = args.repository.resolve()
    dataset = root / "dataset" / "LINUXFLBENCH_dataset.jsonl"
    if not dataset.is_file():
        raise SystemExit(f"missing dataset: {dataset}")
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    checks = {
        "commit": commit == EXPECTED_COMMIT,
        "rows": len(rows) == EXPECTED_ROWS,
        "bytes": dataset.stat().st_size == EXPECTED_BYTES,
        "sha256": digest == EXPECTED_SHA256,
        "unique_task_ids": len({str(row["id"]) for row in rows}) == EXPECTED_ROWS,
    }
    report = {"status": "passed" if all(checks.values()) else "failed",
              "checks": checks, "commit": commit, "sha256": digest}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the compact release data and its internal consistency."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONDITIONS = {"A", "B1", "C1", "D1", "B2", "C2", "D2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(path: Path) -> tuple[int, list[str]]:
    checked = 0
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = ROOT / relative
        checked += 1
        if not target.is_file() or sha256(target) != expected:
            failures.append(relative)
    return checked, failures


def main() -> int:
    episodes = read_csv(DATA / "results" / "episode_results.csv")
    blocks = read_csv(DATA / "results" / "block_results.csv")
    contrasts = read_csv(DATA / "results" / "primary_contrasts.csv")
    log_manifest = read_csv(DATA / "provenance" / "raw_logs_manifest.csv")
    results = json.loads((DATA / "results" / "main_results.json").read_text(encoding="utf-8"))
    episode_keys = {(r["task_id"], r["model"], r["repetition"], r["condition"]) for r in episodes}
    block_keys = {(r["task_id"], r["model"], r["repetition"]) for r in blocks}
    missing = [r for r in episodes if r["status"] == "missing_evidence"]
    logs_by_suffix = {
        row["log_path"].replace("\\", "/").split("saner-main-v1/", 1)[1]: row
        for row in log_manifest
    }
    sample_logs = list((DATA / "sample_logs").rglob("*.jsonl"))
    sample_log_checks = []
    for path in sample_logs:
        suffix = path.relative_to(DATA / "sample_logs").as_posix()
        record = logs_by_suffix.get(suffix)
        sample_log_checks.append(
            record is not None
            and path.stat().st_size == int(record["bytes"])
            and sha256(path) == record["sha256"]
        )
    checks = {
        "main_task_catalog_rows": jsonl_count(DATA / "tasks" / "main_tasks.jsonl") == 230,
        "preexperiment_task_catalog_rows": jsonl_count(DATA / "tasks" / "preexperiment_tasks.jsonl") == 20,
        "episode_rows": len(episodes) == 9_660,
        "unique_episode_cells": len(episode_keys) == 9_660,
        "block_rows": len(blocks) == 1_380,
        "unique_blocks": len(block_keys) == 1_380,
        "primary_contrasts": len(contrasts) == 20,
        "task_count": len({r["task_id"] for r in episodes}) == 230,
        "model_count": len({r["model"] for r in episodes}) == 2,
        "condition_set": {r["condition"] for r in episodes} == CONDITIONS,
        "missing_evidence_cells": len(missing) == 3,
        "missing_metrics_are_empty": all(not r["reciprocal_rank"] for r in missing),
        "analysis_eligibility": results["eligibility"] == "verified_main_evidence",
        "complete_common_tasks": results["primary"]["complete_common_task_count"] == 227,
        "reported_episode_count": results["completeness"]["scheduled_episodes"] == 9_660,
        "raw_log_manifest_rows": len(log_manifest) == 9_660,
        "sample_logs_match_manifest": len(sample_logs) == 14 and all(sample_log_checks),
    }
    checksum_count, failures = verify_checksums(DATA / "provenance" / "checksums.sha256")
    checks["curated_file_checksums"] = not failures and checksum_count > 0
    report = {"status": "passed" if all(checks.values()) else "failed",
              "checks": checks, "checksum_files_checked": checksum_count,
              "checksum_failures": failures}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

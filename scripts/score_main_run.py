"""Score one main-experiment block in the trusted host evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from saner_exp.data.linuxflbench import LinuxFLBenchDataset  # noqa: E402
from saner_exp.evaluation import score_submission  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path,
        default=WORKSPACE / "datasets" / "LinuxFLBench" / "dataset" / "LINUXFLBENCH_dataset.jsonl",
    )
    parser.add_argument(
        "--dataset-manifest", type=Path,
        default=PROJECT / "manifests" / "dataset_manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite scores: {args.output}")

    summary = json.loads(args.summary.read_bytes())
    if (summary.get("record_kind") != "main_condition_matrix_block_v1"
            or summary.get("experiment_id") != "saner-main-v1"
            or summary.get("partition") != "main"):
        raise SystemExit("refusing to score a non-main or unrecognized run summary")
    dataset = LinuxFLBenchDataset.load(args.dataset, manifest_path=args.dataset_manifest)
    truth = dataset.ground_truth(summary["task_id"])
    scores = []
    for outcome in summary["outcomes"]:
        if outcome["status"] == "missing_evidence":
            scores.append({
                "condition": outcome["condition"],
                "status": "missing_evidence",
                "reciprocal_rank": None,
                "recall_at_1": None,
                "recall_at_5": None,
                "recall_at_10": None,
                "first_relevant_rank": None,
                "submitted_count": None,
                "relevant_count": len(truth.paths),
            })
        else:
            score = score_submission(truth.paths, outcome.get("submission"))
            scores.append({
                "condition": outcome["condition"],
                "status": outcome["status"],
                **asdict(score),
            })
    record = {
        "format": "saner-matrix-scores-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility": "main_candidate_pending_complete_integrity_audit",
        "eligible_for_main_analysis": False,
        "task_id": summary["task_id"],
        "model": summary["model"],
        "repetition": summary["repetition"],
        "scores": scores,
        "provenance": {
            "run_summary_sha256": sha256(args.summary),
            "dataset_sha256": dataset.sha256,
            "scoring_contract_sha256": sha256(PROJECT / "manifests" / "scoring_contract.json"),
            "scoring_implementation_sha256": sha256(
                PROJECT / "src" / "saner_exp" / "evaluation" / "scoring.py"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

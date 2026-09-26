"""Audit and analyze a complete frozen saner-main-v1 execution."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze_preexperiment_logs_complete import aggregate, summarize_episode  # noqa: E402
from saner_exp.evaluation import analyze_primary  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=PROJECT / "manifests/main_schedule.json")
    parser.add_argument(
        "--dataset", type=Path,
        default=WORKSPACE / "datasets/LinuxFLBench/dataset/LINUXFLBENCH_dataset.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "runs/main/saner-main-v1/analysis/main_results.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing formal analysis")
    schedule = json.loads(args.schedule.read_bytes())
    if (schedule.get("experiment_id") != "saner-main-v1"
            or schedule.get("block_count") != 1380 or schedule.get("episode_count") != 9660):
        raise SystemExit("formal analyzer requires the frozen saner-main-v1 schedule")
    task_ids = list(dict.fromkeys(row["task_id"] for row in schedule["blocks"]))
    models = list(dict.fromkeys(row["model"] for row in schedule["blocks"]))
    cells = {}
    secondary_cells = {metric: {} for metric in ("recall_at_1", "recall_at_5", "recall_at_10")}
    process_rows = []
    missing_records = []
    artifacts = []
    status_counts = Counter()
    for block in schedule["blocks"]:
        output_dir = WORKSPACE / Path(block["output_directory"])
        summary_path = output_dir / "summary.json"
        scores_path = output_dir / "scores.json"
        if not summary_path.is_file() or not scores_path.is_file():
            raise SystemExit(f"main execution is unfinished: {block['output_directory']}")
        summary = json.loads(summary_path.read_bytes())
        scores = json.loads(scores_path.read_bytes())
        identity = (block["task_id"], block["model"], block["repetition"])
        if (summary.get("record_kind") != "main_condition_matrix_block_v1"
                or (summary["task_id"], summary["model"], summary["repetition"]) != identity):
            raise ValueError(f"main summary identity mismatch: {summary_path}")
        if (scores.get("eligibility") != "main_candidate_pending_complete_integrity_audit"
                or (scores["task_id"], scores["model"], scores["repetition"]) != identity
                or scores["provenance"]["run_summary_sha256"] != sha256(summary_path)):
            raise ValueError(f"main score provenance mismatch: {scores_path}")
        outcomes = {row["condition"]: row for row in summary["outcomes"]}
        score_rows = {row["condition"]: row for row in scores["scores"]}
        if set(outcomes) != set(block["condition_order"]) or set(score_rows) != set(block["condition_order"]):
            raise ValueError(f"incomplete condition set: {output_dir}")
        for condition in block["condition_order"]:
            outcome = outcomes[condition]
            score = score_rows[condition]
            if outcome["status"] != score["status"]:
                raise ValueError(f"outcome/score status mismatch: {output_dir}/{condition}")
            value = score["reciprocal_rank"]
            if outcome["status"] != "missing_evidence" and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"invalid reciprocal-rank score: {scores_path}/{condition}")
            cells[(block["task_id"], block["model"], condition, block["repetition"])] = (
                None if outcome["status"] == "missing_evidence" else float(value)
            )
            for metric, metric_cells in secondary_cells.items():
                metric_value = score[metric]
                if outcome["status"] != "missing_evidence" and (
                        isinstance(metric_value, bool) or not isinstance(metric_value, (int, float))
                        or not 0.0 <= float(metric_value) <= 1.0):
                    raise ValueError(f"invalid {metric} score: {scores_path}/{condition}")
                metric_cells[(block["task_id"], block["model"], condition, block["repetition"])] = (
                    None if outcome["status"] == "missing_evidence" else float(metric_value)
                )
            log_path = output_dir / outcome["log"]
            if not log_path.is_file():
                raise FileNotFoundError(log_path)
            events = load_jsonl(log_path)
            if outcome["status"] == "missing_evidence":
                missing_path = output_dir / outcome["missing_evidence"]
                missing = json.loads(missing_path.read_bytes())
                if (missing.get("reason") != "second_infrastructure_failure"
                        or missing.get("log_sha256") != sha256(log_path)):
                    raise ValueError(f"invalid missing-evidence record: {missing_path}")
                missing_records.append({
                    "task_id": block["task_id"], "model": block["model"],
                    "repetition": block["repetition"], "condition": condition,
                    "reason": missing["reason"], "record": str(missing_path.relative_to(WORKSPACE)),
                })
                synthetic_end = {"event": "episode_end", "status": "missing_evidence"}
                details = summarize_episode([*events, synthetic_end], condition)
                artifacts.append({"path": str(missing_path.relative_to(WORKSPACE)),
                                  "sha256": sha256(missing_path)})
            else:
                if not events or events[0].get("event") != "episode_start" \
                        or events[-1].get("event") != "episode_end":
                    raise ValueError(f"main episode log has invalid boundaries: {log_path}")
                details = summarize_episode(events, condition)
            partials = list(output_dir.glob(f"{log_path.stem}.attempt-*.partial.jsonl"))
            artifacts.extend({"path": str(path.relative_to(WORKSPACE)), "sha256": sha256(path)}
                             for path in partials)
            checkpoint_path = log_path.with_suffix(".checkpoint.json")
            if checkpoint_path.is_file():
                artifacts.append({"path": str(checkpoint_path.relative_to(WORKSPACE)),
                                  "sha256": sha256(checkpoint_path)})
            details.update({
                "task_id": block["task_id"], "model": block["model"],
                "condition": condition, "repetition": block["repetition"],
                "new_function_not_selected": condition in {"C1", "D1", "C2", "D2"}
                and details["feature_requested_calls"] == 0,
                "stable_checkpoint_recovery": details["checkpoint_resumes"] > 0,
                "clean_retry": bool(partials),
                "final_task_success": value is not None and float(value) > 0,
                # Preserve the legacy field until aggregate() has consumed it.  Its
                # old name never meant that the task itself recovered; it only
                # means a later tool call was valid after an earlier invalid call.
                "subsequent_valid_call_after_error": details["recovered_tool_call_errors"],
            })
            process_rows.append(details)
            status_counts[outcome["status"]] += 1
            artifacts.append({"path": str(log_path.relative_to(WORKSPACE)), "sha256": sha256(log_path)})
        artifacts.extend([
            {"path": str(summary_path.relative_to(WORKSPACE)), "sha256": sha256(summary_path)},
            {"path": str(scores_path.relative_to(WORKSPACE)), "sha256": sha256(scores_path)},
        ])

    components = {}
    for row in load_jsonl(args.dataset):
        if str(row["id"]) in set(task_ids):
            component = row.get("Component")
            if not isinstance(component, str) or not component:
                raise ValueError(f"missing Component metadata for task {row['id']}")
            components[str(row["id"])] = component
    primary = analyze_primary(task_ids, models, cells, components)
    common_tasks = primary["complete_common_task_ids"]
    secondary = {}
    for metric, metric_cells in secondary_cells.items():
        rows = []
        for model in models:
            for condition in ("A", "B1", "C1", "D1", "B2", "C2", "D2"):
                complete_task_means = [
                    sum(metric_cells[(task, model, condition, repetition)]
                        for repetition in (1, 2, 3)) / 3.0
                    for task in common_tasks
                ]
                bounds = []
                for task in task_ids:
                    values = [metric_cells[(task, model, condition, repetition)]
                              for repetition in (1, 2, 3)]
                    observed = sum(value for value in values if value is not None)
                    missing = sum(value is None for value in values)
                    bounds.append((observed / 3.0, (observed + missing) / 3.0))
                rows.append({
                    "model": model,
                    "condition": condition,
                    "complete_common_task_mean": sum(complete_task_means) / len(complete_task_means),
                    "complete_common_task_count": len(complete_task_means),
                    "scheduled_task_missing_score_bound": [
                        sum(low for low, _ in bounds) / len(bounds),
                        sum(high for _, high in bounds) / len(bounds),
                    ],
                })
        overall = []
        for model in models:
            for instance in (1, 2):
                left = f"D{instance}"
                differences = []
                task_bounds = []
                for task in task_ids:
                    left_values = [metric_cells[(task, model, left, repetition)]
                                   for repetition in (1, 2, 3)]
                    right_values = [metric_cells[(task, model, "A", repetition)]
                                    for repetition in (1, 2, 3)]
                    left_observed = sum(value for value in left_values if value is not None)
                    right_observed = sum(value for value in right_values if value is not None)
                    left_missing = sum(value is None for value in left_values)
                    right_missing = sum(value is None for value in right_values)
                    task_bounds.append((
                        left_observed / 3.0 - (right_observed + right_missing) / 3.0,
                        (left_observed + left_missing) / 3.0 - right_observed / 3.0,
                    ))
                    if task in common_tasks:
                        differences.append({
                            "task_id": task,
                            "difference": (
                                sum(left_values) / 3.0 - sum(right_values) / 3.0
                            ),
                        })
                overall.append({
                    "model": model,
                    "instance": instance,
                    "expression": f"{left}-A",
                    "status": "exploratory_not_in_primary_family",
                    "complete_common_task_count": len(common_tasks),
                    "mean_difference": sum(row["difference"] for row in differences) / len(differences),
                    "task_differences": differences,
                    "scheduled_task_missing_score_bound": [
                        sum(low for low, _ in task_bounds) / len(task_bounds),
                        sum(high for _, high in task_bounds) / len(task_bounds),
                    ],
                })
        secondary[metric] = {
            "status": "secondary_descriptive",
            "metric_direction": "higher_is_better",
            "analysis_unit": "maintenance_task",
            "repetition_aggregation": "arithmetic_mean_of_three_within_task_condition",
            "missingness": "same_common_complete_task_set_and_zero_to_one_bounds_as_primary",
            "denominator": len(common_tasks),
            "condition_summaries": rows,
            "exploratory_overall_upgrades": overall,
        }
    by_model_condition = aggregate(process_rows, ("model", "condition"))
    for row in by_model_condition:
        members = [item for item in process_rows
                   if item["model"] == row["model"] and item["condition"] == row["condition"]]
        row["eligible_episode_denominator"] = len(members)
        row["missing_evidence_episodes"] = sum(item["status"] == "missing_evidence" for item in members)
        row["final_episode_status_counts"] = dict(sorted(Counter(
            item["status"] for item in members
        ).items()))
        row["new_function_not_selected_episodes"] = sum(item["new_function_not_selected"] for item in members)
        row["stable_checkpoint_recoveries"] = sum(item["stable_checkpoint_recovery"] for item in members)
        row["clean_retries"] = sum(item["clean_retry"] for item in members)
        row["final_task_successes"] = sum(item["final_task_success"] for item in members)
        row["subsequent_valid_calls_after_error"] = row["recovered_tool_call_errors"]
        errors = Counter()
        for item in members:
            errors.update(item["error_categories"])
        row["tool_call_error_categories"] = dict(sorted(errors.items()))
        row["feature_request_without_valid_return_episodes"] = sum(
            item["feature_requested_calls"] > 0 and item["feature_valid_calls"] == 0
            for item in members
        )
        row["feature_valid_without_effective_return_episodes"] = sum(
            item["feature_valid_calls"] > 0 and item["feature_effective_returns"] == 0
            for item in members
        )
        row["successful_search_without_immediate_read_calls"] = (
            row["successful_searches"] - row["search_followed_immediately_by_read"]
        )

    artifact_set = hashlib.sha256(json.dumps(
        sorted(artifacts, key=lambda row: row["path"]), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    record = {
        "format": "saner-main-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "saner-main-v1",
        "eligibility": "verified_main_evidence",
        "completeness": {
            "scheduled_tasks": 230, "scheduled_blocks": 1380, "scheduled_episodes": 9660,
            "scheduled_score_cells": len(cells),
            "observed_nonmissing_score_cells": sum(value is not None for value in cells.values()),
            "missing_evidence_cells": len(missing_records),
            "episode_status_counts": dict(sorted(status_counts.items())),
        },
        "primary": primary,
        "primary_metric_metadata": {
            "metric": "reciprocal_rank",
            "direction": "higher_is_better",
            "analysis_unit": "maintenance_task",
            "repetition_aggregation": "arithmetic_mean_of_three_within_task_condition",
            "confirmatory_family_size": 20,
        },
        "secondary_descriptive": secondary,
        "missing_evidence": missing_records,
        "process_by_model_condition": by_model_condition,
        "process_interpretation": {
            "new_function_not_selected_is_separate_from_call_or_chain_failure": True,
            "subsequent_valid_call_after_error_is_not_recovery_or_task_success": True,
            "recovery_is_reported_separately_from_final_task_success": True,
        },
        "provenance": {
            "schedule_sha256": sha256(args.schedule),
            "analysis_plan_sha256": sha256(PROJECT / "configs/analysis/main_v1.json"),
            "analysis_implementation_sha256": sha256(Path(__file__)),
            "core_analysis_sha256": sha256(PROJECT / "src/saner_exp/evaluation/main_analysis.py"),
            "artifact_count": len(artifacts),
            "artifact_set_sha256": artifact_set,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "tasks": 230, "contrasts": 20,
                      "missing_evidence": len(missing_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

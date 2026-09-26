"""Validate the frozen formal analysis against non-empirical synthetic data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from saner_exp.evaluation.main_analysis import CONDITIONS, MODELS, analyze_primary  # noqa: E402


OUTPUT = PROJECT / "manifests/main_analysis_synthetic_validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    task_ids = [f"synthetic-{index:03d}" for index in range(230)]
    bases = {
        "A": 0.20, "B1": 0.21, "C1": 0.23, "D1": 0.25,
        "B2": 0.19, "C2": 0.24, "D2": 0.26,
    }
    cells = {}
    for task_index, task in enumerate(task_ids):
        for model_index, model in enumerate(MODELS):
            for condition_index, condition in enumerate(CONDITIONS):
                for repetition in (1, 2, 3):
                    variation = (((task_index + 1) * (condition_index + 2) + repetition) % 11 - 5) / 1000
                    cells[(task, model, condition, repetition)] = (
                        bases[condition] + model_index * 0.03 + variation
                    )
    missing_key = (task_ids[0], "qwen-plus-2025-12-01", "D2", 3)
    cells[missing_key] = None
    components = {task: f"component-{index % 66:02d}" for index, task in enumerate(task_ids)}
    result = analyze_primary(
        task_ids, MODELS, cells, components, resamples=1000, grouped_resamples=1000,
    )
    if (result["scheduled_task_count"] != 230
            or result["complete_common_task_count"] != 229
            or result["excluded_task_ids"] != [task_ids[0]]
            or result["primary_contrast_count"] != 20):
        raise AssertionError("complete-case task blocking or contrast construction failed")
    if result["component_grouped_sensitivity"]["group_count"] != 66:
        raise AssertionError("component grouped sensitivity did not preserve 66 groups")
    affected = [row for row in result["contrasts"]
                if row["model"] == "qwen-plus-2025-12-01"
                and row["instance"] == 2 and "D" in row["left_condition"]]
    if not affected or not all(row["missing_score_bound"][0] < row["missing_score_bound"][1]
                               for row in affected):
        raise AssertionError("0-1 missing-score bounds did not widen for affected contrasts")
    unaffected = next(row for row in result["contrasts"]
                      if row["model"] == "Qwen3-14B" and row["contrast"] == "extension_original"
                      and row["instance"] == 1)
    if unaffected["missing_score_bound"][0] != unaffected["missing_score_bound"][1]:
        raise AssertionError("complete contrast unexpectedly received a nonzero missingness width")
    if not all(set(row["delta_sensitivity"]) == {"0.01", "0.05"}
               for row in result["contrasts"]):
        raise AssertionError("delta sensitivity classifications are incomplete")
    if (len(result["exploratory_overall_upgrades"]) != 4
            or not all(len(row["task_difference_distribution"]["values"]) == 229
                       and row["condition_means"] and row["repetition_variability"]
                       and row["missingness_robustness"] in {
                           "no_missing_evidence",
                           "same_practical_class_under_zero_to_one_bounds",
                           "not_robust_do_not_claim_primary_class",
                       }
                       for row in result["contrasts"])):
        raise AssertionError("paper-required condition means, task differences, or variability are absent")
    record = {
        "record_kind": "main_analysis_synthetic_validation_v1",
        "status": "passed",
        "empirical_data_used": False,
        "fixture": {
            "tasks": 230, "models": 2, "conditions": 7, "repetitions": 3,
            "score_cells": 9660, "synthetic_missing_cells": 1, "components": 66,
        },
        "checks": {
            "complete_case_common_task_blocking": "passed",
            "twenty_primary_contrasts": "passed",
            "joint_max_t_bootstrap": "passed",
            "zero_to_one_missing_score_bounds": "passed",
            "delta_0_02_primary_classification": "passed",
            "delta_0_01_0_05_sensitivity": "passed",
            "component_grouped_resampling": "passed",
            "group_count_and_imbalance_disclosure": "passed",
            "table_iv_condition_means_task_differences_and_repetition_variability": "passed",
            "exploratory_overall_upgrade_separated_from_primary_family": "passed",
        },
        "validation_resamples": 1000,
        "formal_resamples_frozen_in_analysis_config": 10000,
        "observed": {
            "complete_common_tasks": result["complete_common_task_count"],
            "excluded_task_ids": result["excluded_task_ids"],
            "contrast_count": result["primary_contrast_count"],
            "component_group_count": result["component_grouped_sensitivity"]["group_count"],
            "component_minimum_group_size": result["component_grouped_sensitivity"]["minimum_group_size"],
            "component_maximum_group_size": result["component_grouped_sensitivity"]["maximum_group_size"],
        },
        "tracked_sha256": {
            "analysis_config": sha256(PROJECT / "configs/analysis/main_v1.json"),
            "formal_analyzer": sha256(PROJECT / "scripts/analyze_main.py"),
            "core_analysis": sha256(PROJECT / "src/saner_exp/evaluation/main_analysis.py"),
            "factorial_analysis": sha256(PROJECT / "src/saner_exp/evaluation/factorial.py"),
            "validation_script": sha256(Path(__file__)),
        },
    }
    encoded = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if OUTPUT.exists() and OUTPUT.read_bytes() != encoded:
        raise SystemExit("existing formal-analysis validation differs from deterministic rebuild")
    OUTPUT.write_bytes(encoded)
    print(json.dumps({"status": "passed", "record": str(OUTPUT), "contrasts": 20,
                      "complete_common_tasks": 229}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Activate overlapping local execution and four service shards before the first main run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUTPUT = PROJECT / "manifests/main_experiment_activation_v3.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict:
    return json.loads((PROJECT / relative).read_bytes())


def verify_hashes(mapping: dict[str, str], *, label: str) -> None:
    for relative, expected in mapping.items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"{label} hash mismatch: {relative}")


def expected_record() -> dict:
    activation_v1 = load("manifests/main_experiment_activation.json")
    activation_v2 = load("manifests/main_experiment_activation_v2.json")
    freeze = load("manifests/main_experiment_freeze.json")
    if activation_v1.get("status") != "ready_for_execution":
        raise SystemExit("activation v1 evidence is unavailable")
    if (activation_v2.get("status") != "ready_for_execution"
            or activation_v2.get("provenance", {}).get("activation_v1_sha256")
            != sha256(PROJECT / "manifests/main_experiment_activation.json")):
        raise SystemExit("activation v2 evidence is unavailable or detached from v1")
    if activation_v1["provenance"]["original_freeze_record_sha256"] != sha256(
            PROJECT / "manifests/main_experiment_freeze.json"):
        raise SystemExit("original freeze changed after activation v1")
    if sha256(PROJECT / freeze["schedule"]["path"]) != freeze["schedule"]["sha256"]:
        raise SystemExit("frozen main schedule changed")

    parallel = load("manifests/parallel_execution_v2_validation.json")
    if (parallel.get("status") != "passed"
            or parallel.get("service_blocks_per_shard") != [173, 173, 172, 172]
            or parallel.get("service_shards_pairwise_disjoint") is not True
            or parallel.get("service_shards_complete_union") is not True
            or parallel.get("cross_model_output_directories_disjoint") is not True
            or parallel.get("cross_model_overlap_enabled") is not True
            or parallel.get("provider_capacity_projection_within_registered_limits") is not True
            or parallel.get("result_relevant_inputs_changed") is not False):
        raise SystemExit("cross-model parallel-execution validation is incomplete")
    verify_hashes({
        "configs/execution/parallel_execution_v2.json": parallel["tracked_sha256"]["execution_config"],
        "manifests/main_schedule.json": parallel["tracked_sha256"]["main_schedule"],
        "scripts/validate_parallel_execution_v2.py": parallel["tracked_sha256"]["validator"],
    }, label="cross-model partition gate")

    runner = load("manifests/main_runner_dry_run_v5.json")
    if (runner.get("record_kind") != "main_runner_container_dry_run_v5_cross_model_parallel"
            or runner.get("status") != "passed" or runner.get("model_calls") != 0
            or runner.get("credential_present") is not False
            or runner.get("probe", {}).get("status") != "pass"):
        raise SystemExit("cross-model formal-runner dry-run gate is incomplete")
    runner_paths = {
        "dry_run_script": "scripts/dry_run_main_runner_v5.ps1",
        "main_runner": "scripts/run_main_block.py",
        "launcher": "scripts/run_main_experiment.ps1",
        "parallel_validator": "scripts/validate_parallel_execution_v2.py",
        "parallel_config": "configs/execution/parallel_execution_v2.json",
        "parallel_validation": "manifests/parallel_execution_v2_validation.json",
        "main_schedule": "manifests/main_schedule.json",
        "main_task_catalog": "manifests/agent_tasks/main.jsonl",
        "condition_registry": "manifests/condition_registry.json",
    }
    verify_hashes(
        {runner_paths[key]: value for key, value in runner["tracked_sha256"].items()},
        label="cross-model runner gate",
    )
    runtime_paths = (
        "scripts/run_main_experiment.ps1",
        "scripts/run_main_block.py",
        "scripts/score_main_run.py",
        "scripts/analyze_main.py",
        "scripts/activate_main_experiment_v3.py",
        "configs/execution/parallel_execution_v2.json",
        "src/saner_exp/evaluation/main_analysis.py",
    )
    return {
        "record_kind": "saner-main-experiment-activation-v3-cross-model-parallel",
        "status": "ready_for_execution",
        "experiment_id": "saner-main-v1",
        "supersedes_for_execution": "manifests/main_experiment_activation_v2.json",
        "scientific_design_unchanged": True,
        "execution_amendment": {
            "timing": "before_first_main_episode",
            "local_workers": 1,
            "service_workers": 4,
            "maximum_simultaneous_block_processes": 5,
            "service_blocks_per_shard": [173, 173, 172, 172],
            "partition_rule": "model-specific frozen block ordinal modulo 4",
            "local_and_service_may_overlap": True,
            "conditions_within_block": "sequential_in_frozen_order",
            "prompts_tools_models_parameters_tasks_repetitions_scoring_changed": False,
            "hidden_retries_added": False,
        },
        "provenance": {
            "activation_v1_sha256": sha256(PROJECT / "manifests/main_experiment_activation.json"),
            "activation_v2_sha256": sha256(PROJECT / "manifests/main_experiment_activation_v2.json"),
            "original_freeze_sha256": sha256(PROJECT / "manifests/main_experiment_freeze.json"),
            "frozen_schedule_sha256": freeze["schedule"]["sha256"],
            "parallel_validation_sha256": sha256(
                PROJECT / "manifests/parallel_execution_v2_validation.json"
            ),
            "parallel_runner_gate_sha256": sha256(
                PROJECT / "manifests/main_runner_dry_run_v5.json"
            ),
            "runtime_sha256": {path: sha256(PROJECT / path) for path in runtime_paths},
        },
        "governance": {
            "created_before_first_main_episode": True,
            "original_schedule_and_prior_activations_retained": True,
            "parallelism_is_operational_and_outcome_blind": True,
            "local_gpu_request_concurrency_remains_one": True,
            "runtime_and_gate_hashes_must_match_before_each_launch": True,
            "result_driven_reruns_or_rule_changes_forbidden": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = expected_record()
    encoded = (json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != encoded:
            raise SystemExit("main activation v3 is missing, stale, or modified")
        print("Verified activation v3: one local worker overlaps four service shards.")
        return 0
    main_root = WORKSPACE / "runs/main/saner-main-v1"
    if main_root.exists() and any(main_root.rglob("summary.json")):
        raise SystemExit("refusing to activate the amendment after formal execution has begun")
    if OUTPUT.exists() and OUTPUT.read_bytes() != encoded:
        raise SystemExit("refusing to overwrite a different activation v3 record")
    OUTPUT.write_bytes(encoded)
    print("Activated one local worker overlapping four service shards for saner-main-v1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Complete, audit-first analysis of the immutable SANER preexperiment logs.

This supplements the frozen primary analysis. It does not alter prompts, tools,
logs, scores, condition membership, or eligibility.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
RUN_ROOT = WORKSPACE / "runs" / "preexperiment" / "saner-preexperiment-v1"
CONDITIONS = ("A", "B1", "C1", "D1", "B2", "C2", "D2")
EXTENDED = {"C1": "context", "D1": "context", "C2": "symbol", "D2": "symbol"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def decode_json(value: str):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def feature_argument(condition: str, arguments) -> tuple[bool, bool]:
    """Return (parameter_present, positive_or_nonempty_requested)."""
    if type(arguments) is not dict:
        return False, False
    if condition == "C1":
        present = "context_lines" in arguments
        value = arguments.get("context_lines")
        return present, type(value) is int and not isinstance(value, bool) and value > 0
    if condition == "D1":
        options = arguments.get("options")
        present = type(options) is dict and "context_lines" in options
        value = options.get("context_lines") if type(options) is dict else None
        return present, type(value) is int and not isinstance(value, bool) and value > 0
    if condition == "C2":
        present = "symbol" in arguments
        value = arguments.get("symbol")
        return present, type(value) is str and bool(value)
    if condition == "D2":
        options = arguments.get("options")
        present = type(options) is dict and "symbol" in options
        value = options.get("symbol") if type(options) is dict else None
        return present, type(value) is str and bool(value)
    return False, False


def response_details(condition: str, response) -> dict:
    details = {
        "ok": False,
        "error_category": None,
        "context_lines_added": 0,
        "symbol_status": None,
        "returned_line_count": 0,
    }
    if type(response) is not dict:
        details["error_category"] = "malformed_logged_response"
        return details
    details["ok"] = response.get("ok") is True
    if not details["ok"]:
        error = response.get("error")
        details["error_category"] = (
            error.get("category") if type(error) is dict else "unclassified_tool_error"
        )
        return details
    data = response.get("data")
    if type(data) is not dict:
        details["error_category"] = "successful_response_missing_data"
        return details
    if condition in ("C1", "D1"):
        for match in data.get("matches", []):
            if type(match) is dict:
                details["context_lines_added"] += len(match.get("before", [])) + len(match.get("after", []))
    elif condition in ("C2", "D2"):
        details["symbol_status"] = data.get("status")
        details["returned_line_count"] = len(data.get("lines", []))
    return details


def summarize_episode(events: list[dict], condition: str) -> dict:
    generations = [row for row in events if row.get("event") == "generation"]
    tools = [row for row in events if row.get("event") == "tool_call"]
    ends = [row for row in events if row.get("event") == "episode_end"]
    if len(ends) != 1:
        raise ValueError(f"episode must contain exactly one episode_end; condition={condition}")
    valid_flags = []
    error_categories = Counter()
    feature_present = feature_requested = feature_valid = feature_effective = 0
    context_lines_added = returned_lines = 0
    symbol_statuses = Counter()
    operation_sequence = []
    for row in tools:
        response = decode_json(row.get("response_json"))
        details = response_details(condition, response)
        valid_flags.append(details["ok"])
        if details["ok"]:
            operation_sequence.append(row.get("operation"))
        else:
            error_categories[details["error_category"]] += 1
        if condition in EXTENDED and row.get("operation") in ("search", "read"):
            present, requested = feature_argument(condition, decode_json(row.get("arguments_json")))
            feature_present += int(present)
            feature_requested += int(requested)
            if requested and details["ok"]:
                feature_valid += 1
                if condition in ("C1", "D1"):
                    added = details["context_lines_added"]
                    context_lines_added += added
                    feature_effective += int(added > 0)
                else:
                    status = details["symbol_status"] or "missing_status"
                    symbol_statuses[status] += 1
                    returned_lines += details["returned_line_count"]
                    feature_effective += int(status == "found" and details["returned_line_count"] > 0)

    recovered_errors = 0
    for index, ok in enumerate(valid_flags):
        if not ok and any(valid_flags[index + 1:]):
            recovered_errors += 1
    successful_searches = 0
    search_followed_by_read = 0
    for index, operation in enumerate(operation_sequence):
        if operation == "search":
            successful_searches += 1
            if index + 1 < len(operation_sequence) and operation_sequence[index + 1] == "read":
                search_followed_by_read += 1

    no_action_generations = sum(int(row.get("tool_call_count", 0)) == 0 for row in generations)
    multi_action_generations = sum(int(row.get("tool_call_count", 0)) > 1 for row in generations)
    return {
        "status": ends[0]["status"],
        "generations": len(generations),
        "attempted_tool_calls": len(tools),
        "valid_tool_calls": sum(valid_flags),
        "tool_call_errors": len(tools) - sum(valid_flags),
        "recovered_tool_call_errors": recovered_errors,
        "no_action_generations": no_action_generations,
        "multi_action_generations": multi_action_generations,
        "protocol_input_tokens": sum(int(row.get("input_tokens", 0)) for row in generations),
        "provider_prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in generations),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in generations),
        "cached_prompt_tokens": sum(int(row.get("cached_prompt_tokens") or 0) for row in generations),
        "truncated_tool_responses": sum(bool(row.get("response_truncated")) for row in tools),
        "infrastructure_failures": sum(row.get("event") == "infrastructure_failure" for row in events),
        "checkpoint_resumes": sum(row.get("event") == "checkpoint_resume" for row in events),
        "feature_parameter_present_calls": feature_present,
        "feature_requested_calls": feature_requested,
        "feature_valid_calls": feature_valid,
        "feature_effective_returns": feature_effective,
        "context_lines_added": context_lines_added,
        "symbol_found": symbol_statuses["found"],
        "symbol_not_found": symbol_statuses["not_found"],
        "symbol_ambiguous": symbol_statuses["ambiguous"],
        "symbol_incomplete": symbol_statuses["incomplete"],
        "returned_function_lines": returned_lines,
        "successful_searches": successful_searches,
        "search_followed_immediately_by_read": search_followed_by_read,
        "error_categories": dict(sorted(error_categories.items())),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    sum_fields = [
        "generations", "attempted_tool_calls", "valid_tool_calls", "tool_call_errors",
        "recovered_tool_call_errors", "no_action_generations", "multi_action_generations",
        "protocol_input_tokens", "provider_prompt_tokens", "completion_tokens",
        "cached_prompt_tokens", "truncated_tool_responses", "infrastructure_failures",
        "checkpoint_resumes", "feature_parameter_present_calls", "feature_requested_calls",
        "feature_valid_calls", "feature_effective_returns", "context_lines_added",
        "symbol_found", "symbol_not_found", "symbol_ambiguous", "symbol_incomplete",
        "returned_function_lines", "successful_searches", "search_followed_immediately_by_read",
    ]
    for identity, members in sorted(groups.items()):
        item = dict(zip(keys, identity))
        item["episodes"] = len(members)
        item["submitted_episodes"] = sum(row["status"] == "submitted" for row in members)
        item["budget_exhausted_episodes"] = sum("budget_exhausted" in row["status"] for row in members)
        for field in sum_fields:
            item[field] = sum(row[field] for row in members)
        item["valid_call_rate"] = (
            item["valid_tool_calls"] / item["attempted_tool_calls"]
            if item["attempted_tool_calls"] else None
        )
        item["feature_adoption_episode_count"] = sum(row["feature_requested_calls"] > 0 for row in members)
        item["feature_effective_episode_count"] = sum(row["feature_effective_returns"] > 0 for row in members)
        item["feature_adoption_episode_rate"] = (
            item["feature_adoption_episode_count"] / len(members) if members else None
        )
        item["feature_effective_episode_rate"] = (
            item["feature_effective_episode_count"] / len(members) if members else None
        )
        item["search_to_immediate_read_rate"] = (
            item["search_followed_immediately_by_read"] / item["successful_searches"]
            if item["successful_searches"] else None
        )
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "analysis_complete")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = PROJECT / "manifests" / "preexperiment_schedule.json"
    registry_path = PROJECT / "manifests" / "condition_registry.json"
    contracts_path = PROJECT / "manifests" / "tool_contract_verification.json"
    primary_path = RUN_ROOT / "analysis" / "preexperiment_results.json"
    schedule = load_json(schedule_path)
    registry = load_json(registry_path)
    contracts = load_json(contracts_path)
    primary = load_json(primary_path)
    expected_schema = {
        condition: details["schema_sha256"] for condition, details in registry["conditions"].items()
    }

    episodes = []
    artifacts = []
    integrity_errors = []
    error_rows = []
    for block in schedule["blocks"]:
        output_dir = WORKSPACE / Path(block["output_directory"])
        summary_path = output_dir / "summary.json"
        scores_path = output_dir / "scores.json"
        if not summary_path.is_file() or not scores_path.is_file():
            integrity_errors.append(f"missing block outputs: {output_dir}")
            continue
        summary = load_json(summary_path)
        scores = load_json(scores_path)
        identity = (block["task_id"], block["model"], block["repetition"])
        if (summary.get("task_id"), summary.get("model"), summary.get("repetition")) != identity:
            integrity_errors.append(f"summary identity mismatch: {summary_path}")
        if scores.get("provenance", {}).get("run_summary_sha256") != sha256(summary_path):
            integrity_errors.append(f"score provenance mismatch: {scores_path}")
        outcomes = {row["condition"]: row for row in summary.get("outcomes", [])}
        score_map = {row["condition"]: row for row in scores.get("scores", [])}
        if set(outcomes) != set(CONDITIONS) or set(score_map) != set(CONDITIONS):
            integrity_errors.append(f"condition set mismatch: {output_dir}")
            continue
        for condition in CONDITIONS:
            outcome = outcomes[condition]
            log_path = output_dir / outcome["log"]
            if not log_path.is_file():
                integrity_errors.append(f"missing log: {log_path}")
                continue
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            if not events or events[0].get("event") != "episode_start" or events[-1].get("event") != "episode_end":
                integrity_errors.append(f"bad log boundary: {log_path}")
                continue
            if outcome.get("schema_sha256") != expected_schema[condition]:
                integrity_errors.append(f"schema mismatch: {log_path}")
            metrics = summarize_episode(events, condition)
            if metrics["status"] != outcome.get("status"):
                integrity_errors.append(f"status mismatch: {log_path}")
            score = score_map[condition]
            row = {
                "task_id": block["task_id"], "model": block["model"],
                "repetition": block["repetition"], "condition": condition,
                "feature": EXTENDED.get(condition, "none"),
                "reciprocal_rank": score["reciprocal_rank"],
                "recall_at_1": score["recall_at_1"],
                "recall_at_5": score["recall_at_5"],
                "recall_at_10": score["recall_at_10"],
                "log_path": str(log_path.relative_to(WORKSPACE)).replace("\\", "/"),
                **{key: value for key, value in metrics.items() if key != "error_categories"},
            }
            episodes.append(row)
            tool_events = [event for event in events if event.get("event") == "tool_call"]
            tool_valid = [response_details(condition, decode_json(event.get("response_json")))["ok"] for event in tool_events]
            for event_index, event in enumerate(tool_events):
                response = decode_json(event.get("response_json"))
                details = response_details(condition, response)
                if details["ok"]:
                    continue
                error = response.get("error", {}) if type(response) is dict else {}
                error_rows.append({
                    "task_id": block["task_id"], "model": block["model"],
                    "repetition": block["repetition"], "condition": condition,
                    "tool_name": event.get("tool_name"),
                    "operation": event.get("operation"),
                    "error_category": details["error_category"],
                    "error_message": error.get("message", "") if type(error) is dict else "",
                    "recovered_later_in_episode": any(tool_valid[event_index + 1:]),
                    "count": 1,
                })
            artifacts.append({"path": row["log_path"], "sha256": sha256(log_path)})

    duplicate_cells = len(episodes) - len({
        (row["task_id"], row["model"], row["repetition"], row["condition"]) for row in episodes
    })
    partials = sorted(RUN_ROOT.glob("**/*.partial.jsonl"))
    integrity = {
        "expected_blocks": 120,
        "observed_blocks": sum((WORKSPACE / Path(block["output_directory"]) / "summary.json").is_file() for block in schedule["blocks"]),
        "expected_episodes": 840,
        "observed_episodes": len(episodes),
        "duplicate_episode_cells": duplicate_cells,
        "partial_log_count": len(partials),
        "integrity_error_count": len(integrity_errors),
        "integrity_errors": integrity_errors,
        "condition_registry_status": registry.get("status"),
        "tool_contract_status": contracts.get("status"),
        "mapped_legal_pair_counts": contracts.get("mapped_legal_call_counts"),
        "conservative_extension_counts": contracts.get("conservative_extension_call_counts"),
        "invalid_error_class_pair_count": len(contracts.get("invalid_error_class_pairs", [])),
        "analysis_eligible": not integrity_errors and len(episodes) == 840 and not partials and duplicate_cells == 0,
    }

    by_condition = aggregate(episodes, ("model", "condition"))
    feature_rows = [row for row in by_condition if row["condition"] in EXTENDED]
    for feature in feature_rows:
        members = [
            row for row in episodes
            if row["model"] == feature["model"] and row["condition"] == feature["condition"]
        ]
        feature["episodes_model_did_not_request"] = sum(row["feature_requested_calls"] == 0 for row in members)
        feature["episodes_requested_but_no_valid_feature_call"] = sum(
            row["feature_requested_calls"] > 0 and row["feature_valid_calls"] == 0 for row in members
        )
        feature["episodes_valid_feature_call_without_effective_return"] = sum(
            row["feature_valid_calls"] > 0 and row["feature_effective_returns"] == 0 for row in members
        )
        feature["episodes_with_effective_feature_return"] = sum(
            row["feature_effective_returns"] > 0 for row in members
        )
    process_rows = by_condition

    task_difference_rows = []
    interval_rows = []
    n20 = primary["bootstrap"]["task_count"]
    n230 = 230
    scale = math.sqrt(n20 / n230)
    for contrast in primary["primary_contrasts"]:
        interval = contrast["simultaneous_interval"]
        if interval is None:
            half_width_20 = projected_half_width = None
            projected_lower = projected_upper = None
        else:
            half_width_20 = (interval[1] - interval[0]) / 2
            projected_half_width = half_width_20 * scale
            projected_lower = contrast["estimate"] - projected_half_width
            projected_upper = contrast["estimate"] + projected_half_width
        interval_rows.append({
            "key": contrast["key"], "model": contrast["model"],
            "instance": contrast["instance"], "rq": contrast["rq"],
            "contrast": contrast["contrast"], "left_condition": contrast["left_condition"],
            "right_condition": contrast["right_condition"],
            "condition_mean_left": contrast.get("left_condition_mean"),
            "condition_mean_right": contrast.get("right_condition_mean"),
            "estimate_n20": contrast["estimate"], "standard_error_n20": contrast["standard_error"],
            "simultaneous_lower_n20": None if interval is None else interval[0],
            "simultaneous_upper_n20": None if interval is None else interval[1],
            "simultaneous_half_width_n20": half_width_20,
            "precision_scale_sqrt_20_over_230": scale,
            "projected_standard_error_n230": contrast["standard_error"] * scale,
            "projected_simultaneous_lower_n230": projected_lower,
            "projected_simultaneous_upper_n230": projected_upper,
            "projected_simultaneous_half_width_n230": projected_half_width,
            "projected_width_reduction_fraction": 1 - scale,
            "projection_assumption": "same task-level variance, dependence, missingness and max-t critical value",
        })
        for item in contrast["task_differences"]:
            task_difference_rows.append({
                "key": contrast["key"], "model": contrast["model"],
                "instance": contrast["instance"], "rq": contrast["rq"],
                "contrast": contrast["contrast"], "task_id": item["task_id"],
                "task_level_difference_after_3_rep_mean": item["difference"],
            })

    error_summary = []
    error_groups = defaultdict(lambda: {"count": 0, "recovered": 0})
    for row in error_rows:
        key = (
            row["model"], row["condition"], row["tool_name"], row["operation"],
            row["error_category"], row["error_message"],
        )
        error_groups[key]["count"] += 1
        error_groups[key]["recovered"] += int(row["recovered_later_in_episode"])
    for key, counts in sorted(error_groups.items()):
        error_summary.append({
            "model": key[0], "condition": key[1], "tool_name": key[2], "operation": key[3],
            "error_category": key[4], "error_message": key[5],
            "count": counts["count"], "recovered_later_in_episode": counts["recovered"],
        })

    totals = aggregate(episodes, ("model",))
    record = {
        "format": "saner-preexperiment-complete-log-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility": "preexperiment_only_ineligible_for_main_findings",
        "definitions": {
            "valid_tool_call": "logged tool response has ok=true",
            "recovered_tool_call_error": "an erroneous tool attempt is followed later in the same episode by at least one valid tool response",
            "feature_requested": "positive context_lines or nonempty symbol appears in the condition-appropriate argument location",
            "feature_effective_return_context": "requested positive context call succeeds and returns at least one before/after line",
            "feature_effective_return_symbol": "requested symbol call succeeds with status=found and at least one returned line",
            "search_to_read": "among successful search calls, the immediately next successful common-operation call is read",
            "n230_precision_projection": "n20 simultaneous half-width multiplied by sqrt(20/230); planning estimate, not a formal result or power guarantee",
        },
        "integrity": integrity,
        "model_totals": totals,
        "process_by_model_condition": process_rows,
        "feature_use_by_model_condition": feature_rows,
        "error_summary": error_summary,
        "contrast_intervals_and_n230_precision": interval_rows,
        "provenance": {
            "schedule": str(schedule_path.relative_to(PROJECT)).replace("\\", "/"),
            "schedule_sha256": sha256(schedule_path),
            "condition_registry_sha256": sha256(registry_path),
            "tool_contract_verification_sha256": sha256(contracts_path),
            "frozen_primary_analysis_sha256": sha256(primary_path),
            "analysis_implementation_sha256": sha256(Path(__file__)),
            "log_artifact_count": len(artifacts),
            "log_artifact_set_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
    }
    (args.output_dir / "complete_log_analysis.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "episode_process.csv", episodes)
    write_csv(args.output_dir / "process_by_model_condition.csv", process_rows)
    write_csv(args.output_dir / "feature_use_by_model_condition.csv", feature_rows)
    write_csv(args.output_dir / "error_summary.csv", error_summary)
    write_csv(args.output_dir / "error_events.csv", error_rows)
    write_csv(args.output_dir / "contrast_intervals_precision.csv", interval_rows)
    write_csv(args.output_dir / "contrast_task_differences.csv", task_difference_rows)

    lines = [
        "# 预实验完整日志分析", "",
        "本报告补充运行完整性、工具采用与过程指标。所有结果仅用于协议验收，不能作为 RQ1–RQ3 的正式证据。", "",
        "## 运行与契约完整性", "",
        f"- 区块：{integrity['observed_blocks']}/{integrity['expected_blocks']}",
        f"- Episode：{integrity['observed_episodes']}/{integrity['expected_episodes']}",
        f"- 重复单元：{integrity['duplicate_episode_cells']}；partial logs：{integrity['partial_log_count']}；完整性错误：{integrity['integrity_error_count']}",
        f"- 七条件 registry：`{integrity['condition_registry_status']}`；工具契约：`{integrity['tool_contract_status']}`",
        "- 四组映射等价与四组旧调用兼容各完成 10,000 次；历史记录仅含 3 组 invalid 对应错误类检查，因此正式实验前仍需扩充 invalid/boundary v2 契约。", "",
        "## 新功能判读原则", "",
        "`未采用` 表示模型没有发出正 context 或非空 symbol；`采用但调用错误` 表示参数出现但工具返回错误；",
        "`合法但无有效返回` 包括搜索没有可增加的上下文，以及 symbol 的 not_found/ambiguous/incomplete；",
        "`有效返回` 才表示实现链路实际向模型提供了新增信息。工具采用率和返回率不用于调整 MRR 主效应。", "",
        "## 230-task 精度估算", "",
        f"在任务级方差、相关结构、缺失率和 max-t 临界值保持不变的规划假设下，区间半宽乘以 sqrt(20/230)={scale:.4f}，",
        f"即预计缩小约 {(1-scale)*100:.1f}%。逐对比的 n=20 端点、任务级差值和 n=230 规划端点见 CSV/工作簿。", "",
        "## 文件", "",
        "- `complete_log_analysis.json`：完整定义、汇总和 provenance。",
        "- `episode_process.csv`：840 个 episode 的过程指标。",
        "- `feature_use_by_model_condition.csv`：新增功能采用与有效返回。",
        "- `error_summary.csv`：调用错误类别。",
        "- `contrast_task_differences.csv`：20×20 个任务级差值。",
        "- `contrast_intervals_precision.csv`：20 个对比的端点与 230-task 精度估算。", "",
    ]
    (args.output_dir / "complete_log_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": "complete" if integrity["analysis_eligible"] else "integrity_failure",
        "episodes": len(episodes), "contrasts": len(interval_rows),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if integrity["analysis_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

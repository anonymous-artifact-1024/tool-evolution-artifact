"""Run one frozen main-experiment task/model/repetition block."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE = PROJECT / "manifests" / "main_schedule.json"
sys.path.insert(0, str(PROJECT / "src"))

from saner_exp.agent import (  # noqa: E402
    AgentLoop, ChatCompletionTokenCounter, DashScopeTokenCounter, EndpointConfig,
    JsonlEventSink, ModelProtocolError, ModelTransportError, OffsetTokenCounter,
    OpenAICompatibleClient, VLLMTokenCounter,
    build_initial_messages,
)
from saner_exp.conditions import load_conditions  # noqa: E402
from saner_exp.data import load_public_tasks  # noqa: E402
from saner_exp.experiment import condition_order  # noqa: E402
from saner_exp.tools import (  # noqa: E402
    FunctionIndex, RestrictedCommandRunner, SourceWorkspace, SubmissionBox,
    build_tool_registry,
)


def completed_outcome(log_path: Path, name: str, position: int) -> dict | None:
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    end = records[-1] if records and records[-1].get("event") == "episode_end" else None
    complete_statuses = {
        "submitted", "generation_budget_exhausted", "tool_call_budget_exhausted",
        "configuration_failure",
    }
    if end is None or end.get("status") not in complete_statuses:
        return None
    start = records[0]
    return {
        "position": position,
        "condition": name,
        "status": end["status"],
        "generations": end["generations"],
        "attempted_tool_calls": end["attempted_tool_calls"],
        "removed_history_groups": end["removed_history_groups"],
        "submission": end["submission"],
        "log": log_path.name,
        "schema_sha256": start["tool_schema_sha256"],
    }


def retain_partial(log_path: Path) -> str:
    attempt = 1
    while True:
        partial = log_path.with_name(f"{log_path.stem}.attempt-{attempt}.partial.jsonl")
        if not partial.exists():
            break
        attempt += 1
    log_path.rename(partial)
    return partial.name


def checkpoint_from_stable_log(
    log_path: Path, checkpoint_path: Path, initial_messages, registry,
) -> bool:
    """Migrate a legacy partial log only when its final model turn is complete."""
    initial_messages = [dict(message) for message in initial_messages]
    records = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("event") != "episode_start":
        return False
    generations = [
        (position, row) for position, row in enumerate(records)
        if row.get("event") == "generation"
    ]
    if not generations:
        groups = []
        generation_count = attempted = removed = 0
        status = "ready"
        outcome_status = submission = None
    else:
        last_position, generation = generations[-1]
        trailing_tools = [
            row for row in records[last_position + 1:]
            if row.get("event") == "tool_call"
            and row.get("generation") == generation.get("generation")
        ]
        call_count = generation.get("tool_call_count")
        if type(call_count) is not int or (
            call_count > 0 and len(trailing_tools) != call_count
        ) or (call_count == 0 and trailing_tools):
            return False
        messages = json.loads(generation["request_messages_json"])
        if messages[:len(initial_messages)] != initial_messages:
            return False
        groups = []
        for message in messages[len(initial_messages):]:
            if message.get("role") == "assistant":
                groups.append([message])
            elif not groups:
                return False
            else:
                groups[-1].append(message)
        if call_count == 0:
            groups.append([{
                "role": "assistant",
                "content": generation.get("content"),
            }, {
                "role": "user",
                "content": (
                    "Continue the investigation using exactly one tool call; "
                    "finish only with the result-submission tool."
                ),
            }])
        else:
            calls = []
            tool_messages = []
            for tool in trailing_tools:
                arguments_json = tool["arguments_json"]
                try:
                    decoded = json.loads(arguments_json)
                except json.JSONDecodeError:
                    decoded = None
                if type(decoded) is not dict:
                    arguments_json = "{}"
                calls.append({
                    "id": tool["call_id"],
                    "type": "function",
                    "function": {
                        "name": tool["tool_name"],
                        "arguments": arguments_json,
                    },
                })
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool["call_id"],
                    "name": tool["tool_name"],
                    "content": tool["response_json"],
                })
            groups.append([{
                "role": "assistant",
                "content": generation.get("content"),
                "tool_calls": calls,
            }, *tool_messages])
        generation_count = generation["generation"]
        attempts = [
            row["attempt"] for row in records if row.get("event") == "tool_call"
        ]
        attempted = max(attempts, default=0)
        removed = generation_count - len(groups)
        if removed < 0:
            return False
        terminal = trailing_tools[-1] if trailing_tools and trailing_tools[-1]["terminal"] else None
        if terminal:
            status = "complete"
            outcome_status = "submitted"
            submission = json.loads(terminal["response_json"])["data"]["paths"]
        else:
            status = "ready"
            outcome_status = submission = None
    canonical = lambda value: json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    checkpoint = {
        "format": "saner-agent-checkpoint-v1",
        "status": status,
        "outcome_status": outcome_status,
        "initial_messages_sha256": hashlib.sha256(canonical(initial_messages)).hexdigest(),
        "tool_schema_sha256": registry.schema_sha256,
        "budgets": {
            "max_generations": 100,
            "max_tool_calls": 100,
            "max_input_tokens": 30_720,
            "max_output_tokens": 2_048,
        },
        "prefix": initial_messages,
        "groups": groups,
        "generations": generation_count,
        "attempted_tool_calls": attempted,
        "removed_history_groups": removed,
        "resume_count": 0,
        "submission": submission,
        "migrated_from_log": log_path.name,
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary.write_bytes(canonical(checkpoint) + b"\n")
    temporary.replace(checkpoint_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("local", "service"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--function-index", type=Path, required=True)
    parser.add_argument("--task-catalog", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--token-counter",
        choices=("vllm", "paid-exact", "dashscope-native", "local-calibrated", "offline-reference"),
    )
    parser.add_argument("--tokenizer-base-url")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-policy", choices=("disabled", "resume-once"), default="resume-once",
    )
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    tokenizer_base_url = None if args.tokenizer_base_url in (None, "-") else args.tokenizer_base_url
    tokenizer_model = None if args.tokenizer_model in (None, "-") else args.tokenizer_model

    schedule = json.loads(args.schedule.read_bytes())
    if (schedule.get("format") != "saner-main-schedule-v1"
            or schedule.get("experiment_id") != "saner-main-v1"
            or schedule.get("task_count") != 230
            or schedule.get("block_count") != 1380
            or schedule.get("episode_count") != 9660):
        raise SystemExit("main schedule is not the frozen 230-task design")
    matches = [
        row for row in schedule["blocks"]
        if row["task_id"] == args.task_id and row["model"] == args.model
        and row["repetition"] == args.repetition
    ]
    if len(matches) != 1:
        raise SystemExit("requested task/model/repetition is absent or duplicated in the main schedule")
    scheduled_block = matches[0]
    expected_model = {"local": "Qwen3-14B", "service": "qwen-plus-2025-12-01"}[args.profile]
    if args.model != expected_model:
        raise SystemExit(f"{args.profile} profile requires frozen model {expected_model}")

    api_key = os.environ.get(f"SANER_{args.profile.upper()}_API_KEY", "")
    if args.profile == "service" and not api_key:
        raise SystemExit("SANER_SERVICE_API_KEY is empty")
    tasks = load_public_tasks(args.task_catalog)
    if len(tasks) != 230:
        raise SystemExit("main runner requires the 230-task label-free catalog")
    try:
        task = tasks[args.task_id]
    except KeyError:
        raise SystemExit("task id is absent from the label-free task catalog") from None
    if task.kernel_version != scheduled_block["kernel_version"]:
        raise SystemExit("task kernel version differs from the frozen main schedule")
    workspace = SourceWorkspace(args.source)
    raw_index = json.loads(args.function_index.read_bytes())
    if raw_index.get("source_identity", {}).get("kernel_version") != task.kernel_version:
        raise SystemExit("function index version differs from the scheduled task source")
    index = FunctionIndex.load(args.function_index)
    conditions = load_conditions(PROJECT / "configs" / "conditions")
    config = EndpointConfig(
        args.base_url, args.model, api_key,
        enable_thinking=False if args.profile == "service" else None,
    )
    if args.profile == "service":
        if args.token_counter == "paid-exact":
            if tokenizer_base_url or tokenizer_model:
                raise SystemExit("paid-exact does not accept a separate tokenizer")
            counter = ChatCompletionTokenCounter(config)
            input_token_accounting = "one_token_chat_completions_metered_preflight"
        elif args.token_counter == "dashscope-native":
            if tokenizer_base_url or not tokenizer_model:
                raise SystemExit("dashscope-native requires only --tokenizer-model")
            validation_path = PROJECT / "manifests" / "service_tokenizer_alias_validation.json"
            if not validation_path.exists():
                raise SystemExit("dashscope-native requires an exhaustive equivalence validation record")
            validation = json.loads(validation_path.read_bytes())
            if not (
                validation.get("generation_model") == args.model
                and validation.get("tokenizer_model") == tokenizer_model
                and validation.get("all_counts_exact") is True
                and validation.get("exhaustive") is True
                and validation.get("request_count") == validation.get("available_request_count")
            ):
                raise SystemExit("dashscope-native equivalence validation is absent, partial, or failed")
            tokenizer_config = EndpointConfig(
                args.base_url, tokenizer_model, api_key, enable_thinking=False,
            )
            counter = DashScopeTokenCounter(tokenizer_config)
            input_token_accounting = f"dashscope_native_tokenizer:{tokenizer_model}"
        elif args.token_counter == "local-calibrated":
            if not tokenizer_base_url or not tokenizer_model:
                raise SystemExit("local-calibrated requires a tokenizer endpoint and model")
            tokenizer_config = EndpointConfig(tokenizer_base_url, tokenizer_model, "")
            counter = OffsetTokenCounter(VLLMTokenCounter(tokenizer_config), 512)
            input_token_accounting = "local_qwen3_vllm_tokenize_plus_512"
        elif args.token_counter == "offline-reference":
            if not tokenizer_base_url or not tokenizer_model:
                raise SystemExit("offline-reference requires a tokenizer endpoint and model")
            tokenizer_config = EndpointConfig(tokenizer_base_url, tokenizer_model, "")
            counter = VLLMTokenCounter(tokenizer_config)
            input_token_accounting = (
                "offline_reference_qwen3_14b_revision_"
                "40c069824f4251a91eefaf281ebe4c544efd3e18"
            )
        else:
            raise SystemExit("service profile requires an explicit --token-counter mode")
    else:
        if args.token_counter in (None, "vllm"):
            if tokenizer_base_url or tokenizer_model:
                raise SystemExit("local vllm counter does not accept a separate tokenizer")
            counter = VLLMTokenCounter(config)
            input_token_accounting = "vllm_tokenize_chat_request"
        elif args.token_counter == "offline-reference":
            if not tokenizer_base_url or not tokenizer_model:
                raise SystemExit("offline-reference requires a tokenizer endpoint and model")
            tokenizer_config = EndpointConfig(tokenizer_base_url, tokenizer_model, "")
            counter = VLLMTokenCounter(tokenizer_config)
            input_token_accounting = (
                "offline_reference_qwen3_14b_revision_"
                "40c069824f4251a91eefaf281ebe4c544efd3e18"
            )
        else:
            raise SystemExit("local profile accepts only vllm or offline-reference counting")
    summary_path = args.output_dir / "summary.json"
    generated_order = condition_order(args.task_id, args.model, args.repetition, seed=args.seed)
    if list(generated_order) != scheduled_block["condition_order"]:
        raise SystemExit("condition order differs from the frozen main schedule")
    order = tuple(scheduled_block["condition_order"])
    planned = [
        args.output_dir / f"{position:02d}-{name}.jsonl"
        for position, name in enumerate(order, start=1)
    ]
    if summary_path.exists():
        raise SystemExit("block is already complete")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outcomes = []
    retained_partial_logs = []
    initial_messages = build_initial_messages(task)
    for position, (name, log_path) in enumerate(zip(order, planned), start=1):
        checkpoint_path = log_path.with_suffix(".checkpoint.json")
        missing_path = log_path.with_suffix(".missing-evidence.json")
        if missing_path.exists():
            missing = json.loads(missing_path.read_bytes())
            if (missing.get("status") != "missing_evidence"
                    or missing.get("condition") != name
                    or missing.get("task_id") != args.task_id
                    or missing.get("model") != args.model
                    or missing.get("repetition") != args.repetition):
                raise SystemExit(f"invalid missing-evidence record: {missing_path.name}")
            outcomes.append(missing["outcome"])
            print(f"[{position}/7] condition {name}: recorded missing evidence", file=sys.stderr, flush=True)
            continue
        prior_partial_logs = list(
            log_path.parent.glob(f"{log_path.stem}.attempt-*.partial.jsonl")
        )
        if len(prior_partial_logs) > 1:
            raise SystemExit(
                f"{log_path.name} has more than one retained infrastructure attempt"
            )
        registry = None
        if log_path.exists():
            outcome = completed_outcome(log_path, name, position)
            if outcome is not None:
                outcomes.append(outcome)
                print(f"[{position}/7] condition {name}: already complete", file=sys.stderr, flush=True)
                continue
            if checkpoint_path.exists() and args.checkpoint_policy == "resume-once":
                print(
                    f"[{position}/7] condition {name}: resuming from stable checkpoint",
                    file=sys.stderr, flush=True,
                )
            elif args.checkpoint_policy == "resume-once":
                registry = build_tool_registry(
                    conditions[name], workspace, RestrictedCommandRunner(workspace),
                    SubmissionBox(workspace), index,
                )
                if checkpoint_from_stable_log(
                    log_path, checkpoint_path, initial_messages, registry,
                ):
                    print(
                        f"[{position}/7] condition {name}: migrated stable log checkpoint",
                        file=sys.stderr, flush=True,
                    )
                else:
                    partial_name = retain_partial(log_path)
                    retained_partial_logs.append(partial_name)
                    print(
                        f"[{position}/7] condition {name}: unstable partial retained as {partial_name}; "
                        "starting one clean retry",
                        file=sys.stderr, flush=True,
                    )
            elif checkpoint_path.exists():
                raise SystemExit(
                    f"{checkpoint_path.name} exists; resume requires checkpoint-policy resume-once"
                )
            else:
                partial_name = retain_partial(log_path)
                retained_partial_logs.append(partial_name)
                print(
                    f"[{position}/7] condition {name}: retained partial log as {partial_name}",
                    file=sys.stderr, flush=True,
                )
        print(f"[{position}/7] condition {name}: running", file=sys.stderr, flush=True)
        if registry is None:
            registry = build_tool_registry(
                conditions[name], workspace, RestrictedCommandRunner(workspace),
                SubmissionBox(workspace), index,
            )
        loop = AgentLoop(
            OpenAICompatibleClient(config), counter,
            event_sink=JsonlEventSink(
                log_path,
                f"main-{args.profile}-{args.task_id}-r{args.repetition}-{name}",
            ),
            checkpoint_path=(
                checkpoint_path if args.checkpoint_policy == "resume-once" else None
            ),
            prior_infrastructure_failures=(
                len(list(log_path.parent.glob(
                    f"{log_path.stem}.attempt-*.partial.jsonl"
                )))
                if args.checkpoint_policy == "resume-once" else 0
            ),
        )
        try:
            result = loop.run(initial_messages, registry)
        except (ModelTransportError, ModelProtocolError, RuntimeError) as error:
            checkpoint = json.loads(checkpoint_path.read_bytes()) if checkpoint_path.exists() else {}
            second_resume_guard = (
                isinstance(error, RuntimeError)
                and str(error) == (
                    "checkpoint already resumed once; second infrastructure failure "
                    "is missing evidence"
                )
            )
            if (not isinstance(error, (ModelTransportError, ModelProtocolError))
                    and not second_resume_guard) \
                    or int(checkpoint.get("resume_count", 0)) < 1:
                raise
            outcome = {
                "position": position,
                "condition": name,
                "status": "missing_evidence",
                "generations": int(checkpoint.get("generations", 0)),
                "attempted_tool_calls": int(checkpoint.get("attempted_tool_calls", 0)),
                "removed_history_groups": int(checkpoint.get("removed_history_groups", 0)),
                "submission": None,
                "log": log_path.name,
                "schema_sha256": registry.schema_sha256,
                "missing_evidence": missing_path.name,
            }
            missing_record = {
                "format": "saner-main-missing-evidence-v1",
                "status": "missing_evidence",
                "reason": "second_infrastructure_failure",
                "experiment_id": "saner-main-v1",
                "task_id": args.task_id,
                "model": args.model,
                "repetition": args.repetition,
                "condition": name,
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
                "checkpoint_resume_count": checkpoint.get("resume_count"),
                "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "outcome": outcome,
            }
            temporary = missing_path.with_suffix(missing_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(missing_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8", newline="\n",
            )
            temporary.replace(missing_path)
            outcomes.append(outcome)
            print(
                f"[{position}/7] condition {name}: second infrastructure failure; "
                "recorded missing evidence",
                file=sys.stderr, flush=True,
            )
            continue
        outcomes.append({
            "position": position,
            "condition": name,
            "status": result.status,
            "generations": result.generations,
            "attempted_tool_calls": result.attempted_tool_calls,
            "removed_history_groups": result.removed_history_groups,
            "submission": result.submission,
            "log": log_path.name,
            "schema_sha256": registry.schema_sha256,
        })
        print(
            f"[{position}/7] condition {name}: {result.status} "
            f"({result.generations} generations)",
            file=sys.stderr, flush=True,
        )

    record = {
        "record_kind": "main_condition_matrix_block_v1",
        "experiment_id": "saner-main-v1",
        "partition": "main",
        "planned_eligibility": "main_analysis_after_complete_integrity_audit",
        "eligible_for_main_analysis": False,
        "scoring": "deferred_to_trusted_host_evaluator",
        "ground_truth_mounted_in_agent_container": False,
        "profile": args.profile,
        "task_id": args.task_id,
        "kernel_version": task.kernel_version,
        "model": args.model,
        "thinking": False,
        "input_token_accounting": input_token_accounting,
        "checkpoint_policy": args.checkpoint_policy,
        "seed": args.seed,
        "repetition": args.repetition,
        "condition_order": order,
        "outcomes": outcomes,
        "retained_partial_logs": retained_partial_logs,
        "provenance": {
            "schedule_sha256": hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
            "function_index_sha256": hashlib.sha256(args.function_index.read_bytes()).hexdigest(),
            "task_catalog_sha256": hashlib.sha256(args.task_catalog.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    summary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic single-action control loop around a model and tool registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol, Sequence

from saner_exp.tools.registry import ToolRegistry

from .model_client import AssistantTurn, ModelClient, ModelTransportError, ToolCall


class EventSink(Protocol):
    def write(self, event) -> None: ...


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    event: str
    recorded_at_utc: str
    generation: int
    input_tokens: int
    request_messages_json: str
    tool_schema_sha256: str
    finish_reason: str | None
    content: str | None
    tool_call_count: int
    prompt_tokens: int | None
    completion_tokens: int | None
    response_id: str | None
    created: int | None
    system_fingerprint: str | None
    cached_prompt_tokens: int | None


@dataclass(frozen=True, slots=True)
class ToolEvent:
    event: str
    recorded_at_utc: str
    generation: int
    attempt: int
    call_id: str
    tool_name: str
    arguments_json: str
    operation: str | None
    response_json: str
    response_bytes: int
    core_source_bytes: int
    response_truncated: bool
    terminal: bool


@dataclass(frozen=True, slots=True)
class EpisodeStartEvent:
    event: str
    recorded_at_utc: str
    initial_messages_json: str
    tool_schema_sha256: str
    max_generations: int
    max_tool_calls: int
    max_input_tokens: int
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class EpisodeEndEvent:
    event: str
    recorded_at_utc: str
    status: str
    generations: int
    attempted_tool_calls: int
    removed_history_groups: int
    submission: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class InfrastructureFailureEvent:
    event: str
    recorded_at_utc: str
    stage: str
    category: str
    next_generation: int
    completed_generations: int
    attempted_tool_calls: int
    removed_history_groups: int
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class CheckpointResumeEvent:
    event: str
    recorded_at_utc: str
    completed_generations: int
    attempted_tool_calls: int
    removed_history_groups: int
    infrastructure_failure_count: int


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    status: str
    generations: int
    attempted_tool_calls: int
    removed_history_groups: int
    submission: tuple[str, ...] | None
    events: tuple[
        EpisodeStartEvent | GenerationEvent | ToolEvent | InfrastructureFailureEvent
        | CheckpointResumeEvent | EpisodeEndEvent, ...
    ]


class JsonlEventSink:
    """Append events without ever receiving endpoint configuration or API credentials."""

    def __init__(self, path: Path | str, episode_id: str):
        self.path = Path(path)
        self.episode_id = episode_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event) -> None:
        record = {"episode_id": self.episode_id, **asdict(event)}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


class AgentLoop:
    def __init__(
        self,
        client: ModelClient,
        input_token_counter: Callable[[Sequence[dict], Sequence[dict]], int],
        *,
        max_generations: int = 100,
        max_tool_calls: int = 100,
        max_input_tokens: int = 30_720,
        max_output_tokens: int = 2_048,
        event_sink: EventSink | None = None,
        checkpoint_path: Path | str | None = None,
        prior_infrastructure_failures: int = 0,
    ):
        if min(max_generations, max_tool_calls, max_input_tokens, max_output_tokens) < 1:
            raise ValueError("all episode budgets must be positive")
        if max_output_tokens > 2_048:
            raise ValueError("the protocol allows at most 2048 output tokens per generation")
        self.client = client
        self.input_token_counter = input_token_counter
        self.max_generations = max_generations
        self.max_tool_calls = max_tool_calls
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.event_sink = event_sink
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        if type(prior_infrastructure_failures) is not int or not 0 <= prior_infrastructure_failures <= 1:
            raise ValueError("prior_infrastructure_failures must be zero or one")
        self.prior_infrastructure_failures = prior_infrastructure_failures

    def _emit(self, event, events: list) -> None:
        events.append(event)
        if self.event_sink is not None:
            self.event_sink.write(event)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _history_message(turn: AssistantTurn) -> dict:
        """Keep provider-facing tool arguments valid while retaining raw arguments in events."""
        message = turn.as_message()
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                decoded = None
            if type(decoded) is not dict:
                call["function"]["arguments"] = "{}"
        return message

    def _finish(self, status: str, generations: int, attempted: int, removed: int,
                submission: tuple[str, ...] | None, events: list,
                prefix: list[dict], groups: list[list[dict]], registry: ToolRegistry,
                resume_count: int) -> EpisodeResult:
        self._emit(EpisodeEndEvent(
            "episode_end", self._now(), status, generations, attempted, removed, submission,
        ), events)
        self._write_checkpoint(
            prefix, groups, registry, generations, attempted, removed, resume_count,
            status="complete", outcome_status=status, submission=submission,
        )
        return EpisodeResult(status, generations, attempted, removed, submission, tuple(events))

    @staticmethod
    def _failure_category(error: ModelTransportError) -> str:
        message = str(error)
        if "HTTP 429" in message:
            return "service_throttling"
        if "HTTP 403" in message:
            return "service_access_interruption"
        if "connection failed" in message:
            return "connection_failure"
        return "model_transport_failure"

    def _record_transport_failure(
        self, error: ModelTransportError, stage: str, generations: int,
        attempted: int, removed: int, events: list,
    ) -> None:
        self._emit(InfrastructureFailureEvent(
            "infrastructure_failure", self._now(), stage,
            self._failure_category(error), generations + 1, generations,
            attempted, removed, type(error).__name__, str(error)[:500],
        ), events)

    @staticmethod
    def _canonical(value) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()

    def _write_checkpoint(
        self, prefix: list[dict], groups: list[list[dict]], registry: ToolRegistry,
        generations: int, attempted: int, removed: int, resume_count: int,
        *, status: str = "ready", outcome_status: str | None = None,
        submission: tuple[str, ...] | None = None,
    ) -> None:
        if self.checkpoint_path is None:
            return
        record = {
            "format": "saner-agent-checkpoint-v1",
            "status": status,
            "outcome_status": outcome_status,
            "initial_messages_sha256": hashlib.sha256(self._canonical(prefix)).hexdigest(),
            "tool_schema_sha256": registry.schema_sha256,
            "budgets": {
                "max_generations": self.max_generations,
                "max_tool_calls": self.max_tool_calls,
                "max_input_tokens": self.max_input_tokens,
                "max_output_tokens": self.max_output_tokens,
            },
            "prefix": prefix,
            "groups": groups,
            "generations": generations,
            "attempted_tool_calls": attempted,
            "removed_history_groups": removed,
            "resume_count": resume_count,
            "submission": list(submission) if submission is not None else None,
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        temporary.write_bytes(self._canonical(record) + b"\n")
        temporary.replace(self.checkpoint_path)

    def _load_checkpoint(
        self, initial_messages: Sequence[dict], registry: ToolRegistry,
    ) -> tuple[list[dict], list[list[dict]], int, int, int, int, EpisodeResult | None]:
        prefix = [dict(message) for message in initial_messages]
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return prefix, [], 0, 0, 0, self.prior_infrastructure_failures, None
        record = json.loads(self.checkpoint_path.read_bytes())
        expected_budgets = {
            "max_generations": self.max_generations,
            "max_tool_calls": self.max_tool_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
        }
        if (
            record.get("format") != "saner-agent-checkpoint-v1"
            or record.get("initial_messages_sha256")
            != hashlib.sha256(self._canonical(prefix)).hexdigest()
            or record.get("tool_schema_sha256") != registry.schema_sha256
            or record.get("budgets") != expected_budgets
            or record.get("prefix") != prefix
        ):
            raise ValueError("checkpoint does not match the current episode protocol")
        generations = record["generations"]
        attempted = record["attempted_tool_calls"]
        removed = record["removed_history_groups"]
        resume_count = record["resume_count"]
        groups = record["groups"]
        if not all(type(value) is int and value >= 0 for value in (
            generations, attempted, removed, resume_count,
        )):
            raise ValueError("checkpoint counters are invalid")
        if record.get("status") == "complete":
            submission = record.get("submission")
            result = EpisodeResult(
                record.get("outcome_status") or (
                    "submitted" if submission is not None else "generation_budget_exhausted"
                ),
                generations, attempted, removed,
                tuple(submission) if submission is not None else None, (),
            )
            return prefix, groups, generations, attempted, removed, resume_count, result
        if record.get("status") != "ready":
            raise ValueError("checkpoint status is invalid")
        if resume_count >= 1:
            raise RuntimeError("checkpoint already resumed once; second infrastructure failure is missing evidence")
        resume_count += 1
        self._write_checkpoint(
            prefix, groups, registry, generations, attempted, removed, resume_count,
        )
        return prefix, groups, generations, attempted, removed, resume_count, None

    def run(self, initial_messages: Sequence[dict], registry: ToolRegistry) -> EpisodeResult:
        events = []
        tools = list(registry.schemas)
        checkpoint_existed = (
            self.checkpoint_path is not None and self.checkpoint_path.exists()
        )
        prefix, groups, generations, attempted, removed, resume_count, completed = (
            self._load_checkpoint(initial_messages, registry)
        )
        if completed is not None:
            return completed
        if checkpoint_existed:
            self._emit(CheckpointResumeEvent(
                "checkpoint_resume", self._now(), generations, attempted, removed,
                resume_count,
            ), events)
        if not checkpoint_existed:
            self._emit(EpisodeStartEvent(
                "episode_start", self._now(),
                json.dumps(prefix, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                registry.schema_sha256, self.max_generations, self.max_tool_calls,
                self.max_input_tokens, self.max_output_tokens,
            ), events)
            self._write_checkpoint(prefix, groups, registry, 0, 0, 0, resume_count)

        while generations < self.max_generations:
            messages = prefix + [message for group in groups for message in group]
            try:
                input_tokens = self.input_token_counter(messages, tools)
            except ModelTransportError as error:
                self._record_transport_failure(
                    error, "input_token_count", generations, attempted, removed, events,
                )
                raise
            if type(input_tokens) is not int or input_tokens < 0:
                raise ValueError("input token counter must return a non-negative integer")
            while input_tokens > self.max_input_tokens and groups:
                groups.pop(0)
                removed += 1
                messages = prefix + [message for group in groups for message in group]
                input_tokens = self.input_token_counter(messages, tools)
                if type(input_tokens) is not int or input_tokens < 0:
                    raise ValueError("input token counter must return a non-negative integer")
            if input_tokens > self.max_input_tokens:
                return self._finish(
                    "configuration_failure", generations, attempted, removed, None, events,
                    prefix, groups, registry, resume_count,
                )

            try:
                turn = self.client.complete(messages, tools, max_tokens=self.max_output_tokens)
            except ModelTransportError as error:
                self._record_transport_failure(
                    error, "model_generation", generations, attempted, removed, events,
                )
                raise
            generations += 1
            self._emit(GenerationEvent(
                "generation", self._now(), generations, input_tokens,
                json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                registry.schema_sha256, turn.finish_reason, turn.content,
                len(turn.tool_calls), turn.prompt_tokens, turn.completion_tokens, turn.response_id,
                turn.created, turn.system_fingerprint,
                turn.cached_prompt_tokens,
            ), events)

            if not turn.tool_calls:
                groups.append([turn.as_message(), {
                    "role": "user",
                    "content": "Continue the investigation using exactly one tool call; finish only with the result-submission tool.",
                }])
                self._write_checkpoint(
                    prefix, groups, registry, generations, attempted, removed, resume_count,
                )
                continue

            if attempted + len(turn.tool_calls) > self.max_tool_calls:
                return self._finish(
                    "tool_call_budget_exhausted", generations, attempted, removed, None, events,
                    prefix, groups, registry, resume_count,
                )

            if len(turn.tool_calls) != 1:
                group = [self._history_message(turn)]
                for call in turn.tool_calls:
                    attempted += 1
                    response = json.dumps({
                        "ok": False,
                        "error": {"category": "multiple_actions", "message": "one tool call is allowed per turn"},
                    }, sort_keys=True, separators=(",", ":"))
                    group.append({"role": "tool", "tool_call_id": call.call_id,
                                  "name": call.name, "content": response})
                    self._emit(ToolEvent(
                        "tool_call", self._now(), generations, attempted, call.call_id, call.name,
                        call.arguments_json, None, response, len(response.encode()), 0, False, False,
                    ), events)
                groups.append(group)
                self._write_checkpoint(
                    prefix, groups, registry, generations, attempted, removed, resume_count,
                )
                continue

            call = turn.tool_calls[0]
            attempted += 1
            try:
                arguments = json.loads(call.arguments_json)
            except json.JSONDecodeError:
                arguments = call.arguments_json
            observation = registry.call(call.name, arguments)
            self._emit(ToolEvent(
                "tool_call", self._now(), generations, attempted, call.call_id, call.name,
                call.arguments_json, observation.operation, observation.payload.content,
                observation.payload.size_bytes, observation.payload.core_source_bytes,
                observation.payload.truncated, observation.terminal,
            ), events)
            groups.append([self._history_message(turn), {
                "role": "tool",
                "tool_call_id": call.call_id,
                "name": call.name,
                "content": observation.payload.content,
            }])
            self._write_checkpoint(
                prefix, groups, registry, generations, attempted, removed, resume_count,
            )
            if observation.terminal:
                decoded = json.loads(observation.payload.content)
                return self._finish(
                    "submitted", generations, attempted, removed,
                    tuple(decoded["data"]["paths"]), events,
                    prefix, groups, registry, resume_count,
                )

        return self._finish(
            "generation_budget_exhausted", generations, attempted, removed, None, events,
            prefix, groups, registry, resume_count,
        )

import json
from pathlib import Path

import pytest

from saner_exp.agent import (
    AgentLoop, AssistantTurn, JsonlEventSink, ModelTransportError, ToolCall,
)
from saner_exp.conditions import load_conditions
from saner_exp.tools import CommandResult, SourceWorkspace, SubmissionBox, build_tool_registry


PROJECT = Path(__file__).resolve().parents[1]


class FakeRunner:
    def run(self, argv, timeout_seconds):
        return CommandResult(tuple(argv), 0, "", "")


class ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, messages, tools, *, max_tokens):
        self.requests.append((list(messages), list(tools), max_tokens))
        return self.turns.pop(0)


def turn(*calls, content=None):
    return AssistantTurn(content, tuple(calls), "tool_calls" if calls else "stop",
                         10, 5, None, None, None)


def registry(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.c").write_text("needle\n", encoding="utf-8")
    workspace = SourceWorkspace(source)
    condition = load_conditions(PROJECT / "configs" / "conditions")["A"]
    return build_tool_registry(condition, workspace, FakeRunner(), SubmissionBox(workspace))


def count_tokens(messages, tools):
    return len(json.dumps([messages, tools])) // 4 + 1


def test_loop_executes_tool_observation_and_first_legal_submission(tmp_path):
    model = ScriptedModel([
        turn(ToolCall("1", "search_text", '{"pattern":"needle"}')),
        turn(ToolCall("2", "submit_result", '{"paths":["a.c"]}')),
    ])
    log = tmp_path / "events.jsonl"
    result = AgentLoop(model, count_tokens, event_sink=JsonlEventSink(log, "episode-1")).run(
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}],
        registry(tmp_path),
    )
    assert result.status == "submitted"
    assert result.submission == ("a.c",)
    assert result.generations == 2 and result.attempted_tool_calls == 2
    second_messages = model.requests[1][0]
    assert second_messages[-1]["role"] == "tool"
    assert json.loads(second_messages[-1]["content"])["ok"] is True
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "episode_start", "generation", "tool_call", "generation", "tool_call", "episode_end",
    ]
    assert records[-1]["status"] == "submitted"
    assert all(record["recorded_at_utc"].endswith("+00:00") for record in records)


def test_loop_reports_multiple_actions_without_executing_them(tmp_path):
    model = ScriptedModel([
        turn(ToolCall("1", "submit_result", '{"paths":["a.c"]}'),
             ToolCall("2", "submit_result", '{"paths":["a.c"]}')),
        turn(ToolCall("3", "submit_result", '{"paths":["a.c"]}')),
    ])
    result = AgentLoop(model, count_tokens).run([{"role": "user", "content": "task"}], registry(tmp_path))
    assert result.status == "submitted"
    assert result.attempted_tool_calls == 3
    assert json.loads(model.requests[1][0][-1]["content"])["error"]["category"] == "multiple_actions"


def test_plain_text_and_invalid_arguments_do_not_finish_episode(tmp_path):
    model = ScriptedModel([
        turn(content="a.c"),
        turn(ToolCall("1", "submit_result", "not-json")),
        turn(ToolCall("2", "submit_result", '{"paths":["a.c"]}')),
    ])
    result = AgentLoop(model, count_tokens).run([{"role": "user", "content": "task"}], registry(tmp_path))
    assert result.status == "submitted"
    assert result.generations == 3 and result.attempted_tool_calls == 2
    invalid = [event for event in result.events if getattr(event, "event") == "tool_call"][0]
    assert json.loads(invalid.response_json)["error"]["category"] == "invalid_arguments"
    assert invalid.arguments_json == "not-json"
    assert model.requests[2][0][-2]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_history_removes_earliest_complete_groups(tmp_path):
    model = ScriptedModel([
        turn(ToolCall("1", "search_text", '{"pattern":"missing"}')),
        turn(ToolCall("2", "submit_result", '{"paths":["a.c"]}')),
    ])
    calls = 0

    def counter(messages, tools):
        nonlocal calls
        calls += 1
        return 100 if any(message.get("role") == "tool" for message in messages) else 10

    result = AgentLoop(model, counter, max_input_tokens=50).run(
        [{"role": "user", "content": "task"}], registry(tmp_path),
    )
    assert result.status == "submitted"
    assert result.removed_history_groups == 1
    assert all(message["role"] != "tool" for message in model.requests[1][0])
    assert calls >= 3


def test_nontrimmable_prefix_is_configuration_failure(tmp_path):
    model = ScriptedModel([])
    result = AgentLoop(model, lambda messages, tools: 99, max_input_tokens=10).run(
        [{"role": "user", "content": "too large"}], registry(tmp_path),
    )
    assert result.status == "configuration_failure"
    assert result.generations == 0


def test_checkpoint_resumes_after_last_complete_tool_round(tmp_path):
    class InterruptAfterFirstTurn(ScriptedModel):
        def complete(self, messages, tools, *, max_tokens):
            if not self.turns:
                raise ConnectionError("simulated interruption")
            return super().complete(messages, tools, max_tokens=max_tokens)

    tools = registry(tmp_path)
    checkpoint = tmp_path / "episode.checkpoint.json"
    first_model = InterruptAfterFirstTurn([
        turn(ToolCall("1", "search_text", '{"pattern":"needle"}')),
    ])
    with pytest.raises(ConnectionError):
        AgentLoop(first_model, count_tokens, checkpoint_path=checkpoint).run(
            [{"role": "user", "content": "task"}], tools,
        )

    resumed_model = ScriptedModel([
        turn(ToolCall("2", "submit_result", '{"paths":["a.c"]}')),
    ])
    result = AgentLoop(
        resumed_model, count_tokens, checkpoint_path=checkpoint,
    ).run([{"role": "user", "content": "task"}], tools)

    assert result.status == "submitted"
    assert result.generations == 2
    assert result.attempted_tool_calls == 2
    assert resumed_model.requests[0][0][-1]["role"] == "tool"
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["outcome_status"] == "submitted"
    assert saved["resume_count"] == 1


def test_checkpoint_refuses_a_second_resume(tmp_path):
    class AlwaysInterrupted:
        def complete(self, messages, tools, *, max_tokens):
            raise ConnectionError("simulated interruption")

    tools = registry(tmp_path)
    checkpoint = tmp_path / "episode.checkpoint.json"
    initial = [{"role": "user", "content": "task"}]
    with pytest.raises(ConnectionError):
        AgentLoop(AlwaysInterrupted(), count_tokens, checkpoint_path=checkpoint).run(initial, tools)
    with pytest.raises(ConnectionError):
        AgentLoop(AlwaysInterrupted(), count_tokens, checkpoint_path=checkpoint).run(initial, tools)
    with pytest.raises(RuntimeError, match="second infrastructure failure"):
        AgentLoop(AlwaysInterrupted(), count_tokens, checkpoint_path=checkpoint).run(initial, tools)


def test_checkpoint_counts_a_prior_clean_retry_against_failure_limit(tmp_path):
    class AlwaysInterrupted:
        def complete(self, messages, tools, *, max_tokens):
            raise ConnectionError("simulated interruption")

    tools = registry(tmp_path)
    checkpoint = tmp_path / "episode.checkpoint.json"
    initial = [{"role": "user", "content": "task"}]
    with pytest.raises(ConnectionError):
        AgentLoop(
            AlwaysInterrupted(), count_tokens, checkpoint_path=checkpoint,
            prior_infrastructure_failures=1,
        ).run(initial, tools)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["resume_count"] == 1
    with pytest.raises(RuntimeError, match="second infrastructure failure"):
        AgentLoop(AlwaysInterrupted(), count_tokens, checkpoint_path=checkpoint).run(initial, tools)


def test_transport_failure_and_checkpoint_resume_are_audited(tmp_path):
    class BalanceInterrupted:
        def complete(self, messages, tools, *, max_tokens):
            raise ModelTransportError("model service returned HTTP 403")

    tools = registry(tmp_path)
    checkpoint = tmp_path / "episode.checkpoint.json"
    log = tmp_path / "episode.jsonl"
    initial = [{"role": "user", "content": "task"}]
    with pytest.raises(ModelTransportError):
        AgentLoop(
            BalanceInterrupted(), count_tokens,
            event_sink=JsonlEventSink(log, "episode"),
            checkpoint_path=checkpoint,
        ).run(initial, tools)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "infrastructure_failure"
    assert records[-1]["category"] == "service_access_interruption"

    resumed = ScriptedModel([
        turn(ToolCall("1", "submit_result", '{"paths":["a.c"]}')),
    ])
    result = AgentLoop(
        resumed, count_tokens, event_sink=JsonlEventSink(log, "episode"),
        checkpoint_path=checkpoint,
    ).run(initial, tools)
    assert result.status == "submitted"
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(record["event"] == "checkpoint_resume" for record in records)

import json
from types import SimpleNamespace

from scripts.run_preexperiment_block import checkpoint_from_stable_log


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_migrates_legacy_log_at_complete_tool_boundary(tmp_path):
    initial = [{"role": "user", "content": "task"}]
    log = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "episode.checkpoint.json"
    write_jsonl(log, [
        {"event": "episode_start"},
        {
            "event": "generation",
            "generation": 1,
            "request_messages_json": json.dumps(initial),
            "tool_call_count": 1,
            "content": None,
        },
        {
            "event": "tool_call",
            "generation": 1,
            "attempt": 1,
            "call_id": "call-1",
            "tool_name": "search_text",
            "arguments_json": '{"pattern":"x"}',
            "response_json": '{"ok":true}',
            "terminal": False,
        },
    ])
    assert checkpoint_from_stable_log(
        log, checkpoint, initial, SimpleNamespace(schema_sha256="schema"),
    )
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "ready"
    assert saved["generations"] == 1
    assert saved["attempted_tool_calls"] == 1
    assert saved["groups"][0][-1]["content"] == '{"ok":true}'


def test_rejects_legacy_log_between_generation_and_tool_result(tmp_path):
    initial = [{"role": "user", "content": "task"}]
    log = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "episode.checkpoint.json"
    write_jsonl(log, [
        {"event": "episode_start"},
        {
            "event": "generation",
            "generation": 1,
            "request_messages_json": json.dumps(initial),
            "tool_call_count": 1,
            "content": None,
        },
    ])
    assert not checkpoint_from_stable_log(
        log, checkpoint, initial, SimpleNamespace(schema_sha256="schema"),
    )
    assert not checkpoint.exists()

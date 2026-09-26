import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_preexperiment_logs_complete.py"
SPEC = importlib.util.spec_from_file_location("complete_log_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_feature_argument_distinguishes_presence_from_actual_adoption():
    assert MODULE.feature_argument("C1", {"context_lines": 0}) == (True, False)
    assert MODULE.feature_argument("C1", {"context_lines": 3}) == (True, True)
    assert MODULE.feature_argument("D1", {"options": {"context_lines": 2}}) == (True, True)
    assert MODULE.feature_argument("C2", {"symbol": ""}) == (True, False)
    assert MODULE.feature_argument("D2", {"options": {"symbol": "alpha"}}) == (True, True)
    assert MODULE.feature_argument("D2", {"location": {"file_path": "a.c"}}) == (False, False)


def test_response_details_require_actual_added_context_or_found_range():
    context = MODULE.response_details("C1", {
        "ok": True,
        "data": {"matches": [{"before": [{"line_number": 1}], "after": [{"line_number": 3}]}]},
    })
    assert context["ok"] is True
    assert context["context_lines_added"] == 2
    symbol = MODULE.response_details("D2", {
        "ok": True, "data": {"status": "found", "lines": [{}, {}, {}]},
    })
    assert symbol["symbol_status"] == "found"
    assert symbol["returned_line_count"] == 3


def test_episode_summary_separates_error_recovery_and_feature_return():
    events = [
        {"event": "generation", "tool_call_count": 1, "input_tokens": 10,
         "prompt_tokens": 11, "completion_tokens": 2},
        {"event": "tool_call", "operation": "search", "arguments_json": "{}",
         "response_json": json.dumps({"ok": False, "error": {"category": "invalid_arguments"}}),
         "response_truncated": False},
        {"event": "generation", "tool_call_count": 1, "input_tokens": 20,
         "prompt_tokens": 21, "completion_tokens": 3},
        {"event": "tool_call", "operation": "search",
         "arguments_json": json.dumps({"context_lines": 2}),
         "response_json": json.dumps({"ok": True, "data": {
             "matches": [{"before": [{}], "after": [{}]}]
         }}), "response_truncated": False},
        {"event": "generation", "tool_call_count": 1, "input_tokens": 30,
         "prompt_tokens": 31, "completion_tokens": 4},
        {"event": "tool_call", "operation": "read", "arguments_json": "{}",
         "response_json": json.dumps({"ok": True, "data": {"lines": []}}),
         "response_truncated": True},
        {"event": "episode_end", "status": "submitted"},
    ]
    result = MODULE.summarize_episode(events, "C1")
    assert result["tool_call_errors"] == 1
    assert result["recovered_tool_call_errors"] == 1
    assert result["feature_requested_calls"] == 1
    assert result["feature_effective_returns"] == 1
    assert result["context_lines_added"] == 2
    assert result["search_followed_immediately_by_read"] == 1
    assert result["truncated_tool_responses"] == 1

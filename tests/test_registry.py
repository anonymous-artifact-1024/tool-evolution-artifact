import json
from pathlib import Path

import pytest

from saner_exp.conditions import load_conditions
from saner_exp.tools import (
    CORE_SOURCE_BYTES,
    TOTAL_RESPONSE_BYTES,
    CommandResult,
    FunctionIndex,
    ReadResult,
    SourceWorkspace,
    SubmissionBox,
    ToolInputError,
    build_tool_registry,
)
from saner_exp.tools.backend import ContextSearchMatch, ContextSearchResult, FileLine
from saner_exp.tools.serialization import serialize_success


PROJECT = Path(__file__).resolve().parents[1]


class FakeRunner:
    def run(self, argv, timeout_seconds):
        return CommandResult(tuple(argv), 0, "ok\n", "")


@pytest.fixture
def registries(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "sample.c").write_text("needle\nsecond\n", encoding="utf-8")
    workspace = SourceWorkspace(root)
    conditions = load_conditions(PROJECT / "configs" / "conditions")
    return {
        name: build_tool_registry(
            spec, workspace, FakeRunner(), SubmissionBox(workspace), FunctionIndex({}),
        )
        for name, spec in conditions.items()
    }


def schema_names(registry):
    return [schema["function"]["name"] for schema in registry.schemas]


def test_registry_exposes_exact_five_tool_combinations(registries):
    assert schema_names(registries["A"]) == [
        "list_directory", "search_text", "read_file", "run_command", "submit_result",
    ]
    assert schema_names(registries["B1"])[1:3] == ["query_repository_text", "read_file"]
    assert schema_names(registries["C1"])[1:3] == ["search_text", "read_file"]
    assert schema_names(registries["D1"])[1:3] == ["query_repository_text", "read_file"]
    assert schema_names(registries["B2"])[1:3] == ["search_text", "fetch_file_slice"]
    assert schema_names(registries["C2"])[1:3] == ["search_text", "read_file"]
    assert schema_names(registries["D2"])[1:3] == ["search_text", "fetch_file_slice"]
    assert len({registry.schema_sha256 for registry in registries.values()}) == 7


def test_only_the_intended_schema_changes_between_factorial_cells(registries):
    schemas = {name: list(registry.schemas) for name, registry in registries.items()}
    for left, right, changed_index in [
        ("A", "B1", 1), ("A", "C1", 1), ("B1", "D1", 1),
        ("A", "B2", 2), ("A", "C2", 2), ("B2", "D2", 2),
    ]:
        changed = [i for i, (a, b) in enumerate(zip(schemas[left], schemas[right])) if a != b]
        assert changed == [changed_index]


def test_dispatch_rejects_unknown_and_invalid_calls_without_ending_episode(registries):
    registry = registries["A"]
    unknown = registry.call("read_anything", {})
    assert unknown.operation is None and unknown.terminal is False
    assert json.loads(unknown.payload.content)["error"]["category"] == "unknown_tool"

    invalid = registry.call("search_text", {"query": "needle"})
    assert invalid.operation == "search" and invalid.terminal is False
    assert json.loads(invalid.payload.content)["error"]["category"] == "invalid_arguments"

    submit = registry.call("submit_result", {"paths": ["sample.c"]})
    assert submit.operation == "submit" and submit.terminal is True
    assert json.loads(submit.payload.content)["data"]["accepted"] is True


def test_response_serializer_enforces_both_byte_budgets():
    lines = tuple(FileLine(number, "界" * 1000) for number in range(1, 101))
    payload = serialize_success(ReadResult("huge.c", 1, 100, 100, lines))
    assert payload.size_bytes <= TOTAL_RESPONSE_BYTES
    assert payload.core_source_bytes <= CORE_SOURCE_BYTES
    assert payload.truncated is True
    decoded = json.loads(payload.content)
    assert decoded["ok"] is True
    assert decoded["data"]["response_truncated"] is True


def test_context_and_command_output_are_trimmed_to_valid_json():
    context = tuple(FileLine(number, "context" * 500) for number in range(1, 11))
    matches = (ContextSearchMatch("a.c", 11, "core", context, context),)
    search_payload = serialize_success(ContextSearchResult("core", ".", 10, matches, False))
    assert search_payload.size_bytes <= TOTAL_RESPONSE_BYTES
    assert json.loads(search_payload.content)["ok"] is True

    command_payload = serialize_success(CommandResult(("grep", "x"), 0, "z" * 100_000, ""))
    assert command_payload.size_bytes <= TOTAL_RESPONSE_BYTES
    assert command_payload.truncated is True
    assert json.loads(command_payload.content)["data"]["response_truncated"] is True

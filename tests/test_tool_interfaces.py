import hashlib
from pathlib import Path

import pytest

from saner_exp.tools import (
    ContextSearchResult,
    FileReadToolInterface,
    FunctionDefinition,
    FunctionIndex,
    FunctionReadResult,
    IndexedFile,
    ReadResult,
    SearchResult,
    SearchToolInterface,
    SourceWorkspace,
    ToolInputError,
)
from saner_exp.tools.function_index import parse_c_functions


@pytest.fixture
def setup(tmp_path: Path):
    root = tmp_path / "linux-3.3.3"
    root.mkdir()
    content = (
        b"// heading\n"
        b"static int alpha(int x)\n{\n    return x + 1;\n}\n"
        b"// needle\n"
        b"int duplicate(void) { return 1; }\n"
        b"int duplicate(int x) { return x; }\n"
        b"// tail\n"
    )
    (root / "sample.c").write_bytes(content)
    indexed = IndexedFile(hashlib.sha256(content).hexdigest(), parse_c_functions(content))
    return SourceWorkspace(root), FunctionIndex({"sample.c": indexed})


@pytest.mark.parametrize("limit", [1, 20])
def test_search_refactoring_is_response_equivalent(setup, limit: int):
    workspace, _ = setup
    original = SearchToolInterface(workspace, refactored=False, extended=False)
    refactored = SearchToolInterface(workspace, refactored=True, extended=False)
    left = original.call({"pattern": "int", "root_path": ".", "max_results": limit})
    right = refactored.call({"target": {"query": "int", "root_path": "."},
                             "options": {"result_limit": limit}})
    assert original.name == "search_text"
    assert refactored.name == "query_repository_text"
    assert left == right


def test_search_extension_is_conservative_and_adds_clipped_context(setup):
    workspace, _ = setup
    base = SearchToolInterface(workspace, refactored=False, extended=False)
    extended = SearchToolInterface(workspace, refactored=False, extended=True)
    old = base.call({"pattern": "heading"})
    assert isinstance(old, SearchResult)
    assert extended.call({"pattern": "heading"}) == old
    assert extended.call({"pattern": "heading", "context_lines": 0}) == old
    result = extended.call({"pattern": "heading", "context_lines": 2})
    assert isinstance(result, ContextSearchResult)
    assert result.matches[0].before == ()
    assert [(line.line_number, line.text) for line in result.matches[0].after] == [
        (2, "static int alpha(int x)"), (3, "{")
    ]


def test_extended_search_refactoring_is_response_equivalent(setup):
    workspace, _ = setup
    original = SearchToolInterface(workspace, refactored=False, extended=True)
    refactored = SearchToolInterface(workspace, refactored=True, extended=True)
    assert original.call({"pattern": "needle", "context_lines": 1, "max_results": 20}) == refactored.call({
        "target": {"query": "needle"}, "options": {"context_lines": 1, "result_limit": 20}
    })


def test_file_read_refactoring_and_defaults_are_response_equivalent(setup):
    workspace, _ = setup
    original = FileReadToolInterface(workspace, refactored=False, extended=False)
    refactored = FileReadToolInterface(workspace, refactored=True, extended=False)
    left = original.call({"file_path": "sample.c"})
    right = refactored.call({"location": {"file_path": "sample.c"}})
    assert original.name == "read_file"
    assert refactored.name == "fetch_file_slice"
    assert isinstance(left, ReadResult)
    assert left == right
    assert left.start_line == 1
    assert left.end_line == left.total_lines


def test_file_extension_is_conservative_and_exact_symbol_lookup(setup):
    workspace, index = setup
    base = FileReadToolInterface(workspace, refactored=False, extended=False)
    extended = FileReadToolInterface(workspace, index, refactored=False, extended=True)
    assert extended.call({"file_path": "sample.c", "start_line": 2, "end_line": 4}) == base.call(
        {"file_path": "sample.c", "start_line": 2, "end_line": 4}
    )
    result = extended.call({"file_path": "sample.c", "start_line": 999, "end_line": 1,
                            "symbol": "alpha"})
    assert isinstance(result, FunctionReadResult)
    assert result.status == "found"
    assert (result.selected_start_line, result.selected_end_line) == (2, 5)
    assert result.returned_start_line == 2
    assert result.returned_end_line == 5


def test_extended_file_refactoring_is_response_equivalent(setup):
    workspace, index = setup
    original = FileReadToolInterface(workspace, index, refactored=False, extended=True)
    refactored = FileReadToolInterface(workspace, index, refactored=True, extended=True)
    left = original.call({"file_path": "sample.c", "symbol": "alpha"})
    right = refactored.call({"location": {"file_path": "sample.c"},
                             "options": {"symbol": "alpha"}})
    assert left == right


def test_symbol_lookup_reports_not_found_ambiguity_and_incomplete(setup):
    workspace, index = setup
    tool = FileReadToolInterface(workspace, index, refactored=False, extended=True)
    assert tool.call({"file_path": "sample.c", "symbol": "missing"}).status == "not_found"
    ambiguous = tool.call({"file_path": "sample.c", "symbol": "duplicate"})
    assert ambiguous.status == "ambiguous"
    assert ambiguous.candidate_count == 2

    content = (workspace.root / "sample.c").read_bytes()
    incomplete_index = FunctionIndex({"sample.c": IndexedFile(
        hashlib.sha256(content).hexdigest(), (FunctionDefinition("broken", 1, 2, False),)
    )})
    incomplete_tool = FileReadToolInterface(workspace, incomplete_index, refactored=False, extended=True)
    assert incomplete_tool.call({"file_path": "sample.c", "symbol": "broken"}).status == "incomplete"


@pytest.mark.parametrize("arguments", [
    {"pattern": "needle", "max_results": 0},
    {"pattern": "needle", "max_results": 21},
    {"pattern": "needle", "max_results": "20"},
    {"pattern": "needle", "context_lines": 11},
    {"pattern": "needle", "unknown": 1},
    {"pattern": None},
])
def test_search_interface_rejects_invalid_json_arguments(setup, arguments):
    workspace, _ = setup
    with pytest.raises(ToolInputError):
        SearchToolInterface(workspace, refactored=False, extended=True).call(arguments)


def test_base_interfaces_reject_extension_fields(setup):
    workspace, _ = setup
    with pytest.raises(ToolInputError):
        SearchToolInterface(workspace, refactored=False, extended=False).call(
            {"pattern": "needle", "context_lines": 0}
        )
    with pytest.raises(ToolInputError):
        FileReadToolInterface(workspace, refactored=True, extended=False).call(
            {"location": {"file_path": "sample.c"}, "options": {"symbol": "alpha"}}
        )

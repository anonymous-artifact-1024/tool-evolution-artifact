import os
from pathlib import Path

import pytest

from saner_exp.data.task import AgentTask
from saner_exp.tools import SourceWorkspace, ToolInputError, WorkspacePathError


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    version = tmp_path / "versions" / "linux-3.3.3"
    (version / "drivers").mkdir(parents=True)
    (version / "drivers" / "a.c").write_bytes(b"first\nneedle alpha\nlast\n")
    (version / "drivers" / "b.c").write_bytes(b"needle beta\n")
    (version / "binary.bin").write_bytes(b"needle\0hidden\n")
    (version / ".git").mkdir()
    (version / ".git" / "config").write_text("needle secret\n", encoding="utf-8")
    return tmp_path


def workspace(source_root: Path) -> SourceWorkspace:
    task = AgentTask("7", "title", "description", "3.3.3")
    return SourceWorkspace.for_task(source_root, task)


def test_task_binding_and_literal_search_are_deterministic(source_root: Path):
    ws = workspace(source_root)
    result = ws.search_text("needle")
    assert ws.root.name == "linux-3.3.3"
    assert [(m.path, m.line_number, m.line) for m in result.matches] == [
        ("drivers/a.c", 2, "needle alpha"),
        ("drivers/b.c", 1, "needle beta"),
    ]
    assert result.scope == "."
    assert result.truncated is False


def test_task_binding_rejects_unsupported_version(source_root: Path):
    task = AgentTask("7", "title", "description", "../3.3.3")
    with pytest.raises(ToolInputError):
        SourceWorkspace.for_task(source_root, task)


def test_search_scope_limit_and_binary_handling(source_root: Path):
    ws = workspace(source_root)
    result = ws.search_text("needle", "drivers", max_results=1)
    assert [m.path for m in result.matches] == ["drivers/a.c"]
    assert result.truncated is True
    assert ws.search_text("hidden").matches == ()


def test_read_file_uses_inclusive_one_based_ranges(source_root: Path):
    result = workspace(source_root).read_file("drivers/a.c", 2, 9)
    assert result.path == "drivers/a.c"
    assert result.total_lines == 3
    assert result.start_line == 2
    assert result.end_line == 3
    assert [(line.line_number, line.text) for line in result.lines] == [
        (2, "needle alpha"), (3, "last")
    ]
    past_end = workspace(source_root).read_file("drivers/a.c", 9, 10)
    assert past_end.lines == ()
    assert past_end.end_line == 8


@pytest.mark.parametrize("bad", ["", "/etc/passwd", "../outside", "drivers/../binary.bin", "drivers\\a.c", ".git/config"])
def test_unsafe_paths_are_rejected(source_root: Path, bad: str):
    with pytest.raises(WorkspacePathError):
        workspace(source_root).read_file(bad, 1, 1)


def test_invalid_requests_are_rejected(source_root: Path):
    ws = workspace(source_root)
    with pytest.raises(ToolInputError):
        ws.search_text("")
    with pytest.raises(ToolInputError):
        ws.read_file("drivers/a.c", 0, 2)
    with pytest.raises(ToolInputError):
        ws.read_file("binary.bin", 1, 2)
    with pytest.raises(ToolInputError):
        ws.read_file("drivers", 1, 2)
    with pytest.raises(WorkspacePathError):
        ws.read_file("missing.c", 1, 2)


def test_symlink_escape_is_rejected_when_supported(source_root: Path, tmp_path: Path):
    link = source_root / "versions" / "linux-3.3.3" / "escape"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(WorkspacePathError):
        workspace(source_root).read_file("escape", 1, 1)

from pathlib import Path

import pytest

from saner_exp.tools import (
    CommandResult,
    CommandToolInterface,
    DirectoryToolInterface,
    RestrictedCommandRunner,
    SourceWorkspace,
    SubmissionBox,
    ToolInputError,
)


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, argv, timeout_seconds):
        self.calls.append((tuple(argv), timeout_seconds))
        return CommandResult(tuple(argv), 0, "match\n", "")


@pytest.fixture
def workspace(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("hidden", encoding="utf-8")
    (root / "b.c").write_text("b", encoding="utf-8")
    (root / "a.c").write_text("alpha", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "c.c").write_text("c", encoding="utf-8")
    return SourceWorkspace(root)


def test_directory_listing_is_sorted_bounded_and_hides_git(workspace):
    result = DirectoryToolInterface(workspace).call({"max_entries": 2})
    assert [entry.name for entry in result.entries] == ["a.c", "b.c"]
    assert result.truncated is True
    assert ".git" not in [entry.name for entry in result.entries]
    nested = DirectoryToolInterface(workspace).call({"path": "sub"})
    assert nested.entries[0].path == "sub/c.c"


def test_command_interface_uses_argv_without_a_shell(workspace):
    runner = RecordingRunner()
    interface = CommandToolInterface(runner)
    result = interface.call({"argv": ["grep", "-n", "alpha", "a.c"], "timeout_seconds": 9})
    assert result.stdout == "match\n"
    assert runner.calls == [(('grep', '-n', 'alpha', 'a.c'), 9)]

    invalid = [
        ["sh", "-c", "cat a.c"],
        ["find", ".", "-delete"],
        ["sed", "--in-place=.bak", "a.c"],
        ["cat", "../outside"],
        ["cat", "C:\\outside"],
    ]
    for argv in invalid:
        with pytest.raises(ToolInputError):
            interface.call({"argv": argv})
    assert len(runner.calls) == 1


def test_command_arguments_have_a_total_input_bound():
    with pytest.raises(ToolInputError, match="1024 bytes"):
        RestrictedCommandRunner.validate_argv(["grep", "x" * 1025])


def test_first_legal_submission_is_terminal_state(workspace):
    box = SubmissionBox(workspace)
    with pytest.raises(ToolInputError):
        box.submit(["missing.c"])
    assert box.result is None

    result = box.submit(["./a.c", "a.c", "sub/c.c"])
    assert result.paths == ("a.c", "sub/c.c")
    assert box.result == result
    with pytest.raises(ToolInputError, match="already has"):
        box.submit(["b.c"])

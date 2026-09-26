"""Tools whose functionality and interface are identical in all seven conditions."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import PurePosixPath
import subprocess
from typing import Protocol, Sequence

from .backend import SourceWorkspace, ToolInputError
from .interfaces import _integer, _object, _string


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    paths: tuple[str, ...]
    accepted: bool = True


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], timeout_seconds: int) -> CommandResult: ...


class RestrictedCommandRunner:
    """Run a small read-only command set without a shell inside the task snapshot."""

    ALLOWED = frozenset({"cat", "find", "grep", "head", "tail", "wc"})
    FORBIDDEN_ARGUMENTS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir",
                                     "-fprint", "-fprintf", "-fls", "-i", "--in-place"})

    def __init__(self, workspace: SourceWorkspace):
        self.workspace = workspace

    @classmethod
    def validate_argv(cls, argv: Sequence[str]) -> tuple[str, ...]:
        if type(argv) not in (list, tuple) or not 1 <= len(argv) <= 64:
            raise ToolInputError("argv must contain from 1 through 64 strings")
        if any(type(arg) is not str or not arg or "\0" in arg or "\n" in arg or "\r" in arg for arg in argv):
            raise ToolInputError("every argv item must be one non-empty line")
        if any(len(arg.encode("utf-8")) > 1024 for arg in argv):
            raise ToolInputError("every argv item must be at most 1024 bytes")
        if sum(len(arg.encode("utf-8")) for arg in argv) > 8192:
            raise ToolInputError("argv exceeds the 8192-byte input limit")
        command = argv[0]
        if command not in cls.ALLOWED:
            raise ToolInputError(f"command is not allowed: {command!r}")
        for argument in argv[1:]:
            if argument in cls.FORBIDDEN_ARGUMENTS or argument.startswith("--in-place="):
                raise ToolInputError(f"command argument is not allowed: {argument!r}")
            if argument.startswith("/") or "\\" in argument:
                raise ToolInputError("absolute and Windows-style command paths are not allowed")
            if not argument.startswith("-") and ".." in PurePosixPath(argument).parts:
                raise ToolInputError("command paths cannot traverse outside the source snapshot")
        return tuple(argv)

    def run(self, argv: Sequence[str], timeout_seconds: int) -> CommandResult:
        checked = self.validate_argv(argv)
        safe_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        environment = {
            "PATH": safe_path if os.name == "posix" else os.environ.get("PATH", ""),
            "LANG": "C",
            "LC_ALL": "C",
        }
        try:
            completed = subprocess.run(
                checked, cwd=self.workspace.root, env=environment, stdin=subprocess.DEVNULL,
                capture_output=True, timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            raise ToolInputError(f"command exceeded {timeout_seconds} seconds") from None
        return CommandResult(checked, completed.returncode,
                             completed.stdout.decode("utf-8", errors="replace"),
                             completed.stderr.decode("utf-8", errors="replace"))


class DirectoryToolInterface:
    name = "list_directory"

    def __init__(self, workspace: SourceWorkspace):
        self.workspace = workspace

    def call(self, arguments: dict):
        values = _object(arguments, "arguments", {"path", "max_entries"})
        path = _string(values.get("path", "."), "path")
        limit = _integer(values.get("max_entries", 200), "max_entries", 1, 200)
        return self.workspace.list_directory(path, max_entries=limit)


class CommandToolInterface:
    name = "run_command"

    def __init__(self, runner: CommandRunner):
        self.runner = runner

    def call(self, arguments: dict):
        values = _object(arguments, "arguments", {"argv", "timeout_seconds"}, {"argv"})
        argv = values["argv"]
        if type(argv) is not list:
            raise ToolInputError("argv must be a JSON array")
        RestrictedCommandRunner.validate_argv(argv)
        timeout = _integer(values.get("timeout_seconds", 300), "timeout_seconds", 1, 300)
        return self.runner.run(argv, timeout)


class SubmissionBox:
    """Accept the first legal ranked-file submission and then close the episode."""

    def __init__(self, workspace: SourceWorkspace):
        self.workspace = workspace
        self._result: SubmissionResult | None = None

    @property
    def result(self) -> SubmissionResult | None:
        return self._result

    def submit(self, paths) -> SubmissionResult:
        if self._result is not None:
            raise ToolInputError("the episode already has a legal submission")
        if type(paths) is not list or not 1 <= len(paths) <= 10:
            raise ToolInputError("paths must be a JSON array containing 1 through 10 entries")
        normalized = []
        seen = set()
        for raw_path in paths:
            path = _string(raw_path, "paths item").removeprefix("./")
            path = self.workspace.validate_file(path)
            if path not in seen:
                seen.add(path)
                normalized.append(path)
        self._result = SubmissionResult(tuple(normalized))
        return self._result


class SubmitToolInterface:
    name = "submit_result"

    def __init__(self, box: SubmissionBox):
        self.box = box

    def call(self, arguments: dict):
        values = _object(arguments, "arguments", {"paths"}, {"paths"})
        return self.box.submit(values["paths"])

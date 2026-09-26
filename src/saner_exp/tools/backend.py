"""Deterministic, read-only file operations for an isolated source snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Iterator

from saner_exp.data.kernel_versions import parse_version
from saner_exp.data.task import AgentTask


class ToolInputError(ValueError):
    """A tool request has invalid arguments or targets unsupported content."""


class WorkspacePathError(ToolInputError):
    """A requested path is outside the task's isolated source workspace."""


@dataclass(frozen=True, slots=True)
class FileLine:
    line_number: int
    text: str


@dataclass(frozen=True, slots=True)
class SearchMatch:
    path: str
    line_number: int
    line: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    scope: str
    matches: tuple[SearchMatch, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class ContextSearchMatch:
    path: str
    line_number: int
    line: str
    before: tuple[FileLine, ...]
    after: tuple[FileLine, ...]


@dataclass(frozen=True, slots=True)
class ContextSearchResult:
    query: str
    scope: str
    context_lines: int
    matches: tuple[ContextSearchMatch, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class ReadResult:
    path: str
    start_line: int
    end_line: int
    total_lines: int
    lines: tuple[FileLine, ...]


@dataclass(frozen=True, slots=True)
class FunctionCandidate:
    start_line: int
    end_line: int
    complete: bool


@dataclass(frozen=True, slots=True)
class FunctionReadResult:
    path: str
    symbol: str
    status: str
    candidate_count: int
    candidates: tuple[FunctionCandidate, ...]
    selected_start_line: int | None
    selected_end_line: int | None
    returned_start_line: int | None
    returned_end_line: int | None
    total_lines: int
    lines: tuple[FileLine, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    name: str
    path: str
    kind: str
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class DirectoryResult:
    path: str
    entries: tuple[DirectoryEntry, ...]
    truncated: bool


class SourceWorkspace:
    """Expose only read operations rooted at one task's kernel snapshot."""

    def __init__(self, root: Path | str):
        candidate = Path(root)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if not candidate.is_dir():
            raise NotADirectoryError(candidate)
        self._root = candidate.resolve(strict=True)

    @classmethod
    def for_task(cls, prepared_source_root: Path | str, task: AgentTask) -> "SourceWorkspace":
        """Bind a task to exactly ``versions/linux-<kernel_version>``."""
        version = task.kernel_version
        try:
            parse_version(version)
        except (TypeError, ValueError):
            raise ToolInputError(f"invalid kernel version: {version!r}") from None
        return cls(Path(prepared_source_root) / "versions" / f"linux-{version}")

    @property
    def root(self) -> Path:
        """Resolved host path for orchestration and auditing; never shown to the agent."""
        return self._root

    @staticmethod
    def _relative_path(raw_path: str, *, allow_root: bool) -> PurePosixPath:
        if not isinstance(raw_path, str) or "\0" in raw_path or "\\" in raw_path:
            raise WorkspacePathError("path must be a relative POSIX path")
        if raw_path == "." and allow_root:
            return PurePosixPath(".")
        if not raw_path or raw_path.startswith("/"):
            raise WorkspacePathError("path must be relative to the task source root")
        parts = raw_path.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise WorkspacePathError("path must be canonical and cannot traverse directories")
        if ".git" in parts:
            raise WorkspacePathError("Git metadata is outside the tool workspace")
        return PurePosixPath(*parts)

    def _resolve(self, raw_path: str, *, allow_root: bool = False) -> tuple[PurePosixPath, Path]:
        relative = self._relative_path(raw_path, allow_root=allow_root)
        try:
            candidate = self._root.joinpath(*relative.parts).resolve(strict=True)
        except FileNotFoundError:
            raise WorkspacePathError("path does not exist in the task source root") from None
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise WorkspacePathError("path resolves outside the task source root") from None
        return relative, candidate

    def validate_file(self, raw_path: str) -> str:
        """Return a canonical path for an existing, non-symlink regular file."""
        relative = self._relative_path(raw_path, allow_root=False)
        unresolved = self._root.joinpath(*relative.parts)
        if unresolved.is_symlink():
            raise WorkspacePathError("symbolic links cannot be submitted as source files")
        relative, candidate = self._resolve(raw_path)
        if not candidate.is_file():
            raise WorkspacePathError("path must identify a regular source file")
        return relative.as_posix()

    def list_directory(self, path: str = ".", *, max_entries: int = 200) -> DirectoryResult:
        """List one directory without following links or exposing Git metadata."""
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not 1 <= max_entries <= 200:
            raise ToolInputError("max_entries must be an integer from 1 through 200")
        relative, directory = self._resolve(path, allow_root=True)
        if not directory.is_dir():
            raise ToolInputError("list_directory path must identify a directory")
        entries = []
        truncated = False
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.name == ".git":
                continue
            if len(entries) == max_entries:
                truncated = True
                break
            display = child.relative_to(self._root).as_posix()
            if child.is_symlink():
                kind, size = "symlink", None
            elif child.is_dir():
                kind, size = "directory", None
            elif child.is_file():
                kind, size = "file", child.stat().st_size
            else:
                kind, size = "other", None
            entries.append(DirectoryEntry(child.name, display, kind, size))
        display_path = "." if relative == PurePosixPath(".") else relative.as_posix()
        return DirectoryResult(display_path, tuple(entries), truncated)

    @staticmethod
    def _text_lines(path: Path) -> list[str]:
        content = path.read_bytes()
        if b"\0" in content:
            raise ToolInputError("binary files cannot be read with text tools")
        return content.decode("utf-8", errors="replace").splitlines()

    def _files(self, scope_path: Path) -> Iterator[Path]:
        if scope_path.is_file():
            yield scope_path
            return
        if not scope_path.is_dir():
            raise ToolInputError("search scope must be a file or directory")
        for directory, dirnames, filenames in os.walk(scope_path, followlinks=False):
            current = Path(directory)
            dirnames[:] = sorted(
                name for name in dirnames
                if name != ".git" and not (current / name).is_symlink()
            )
            for name in sorted(filenames):
                path = current / name
                if path.is_symlink():
                    continue
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(self._root)
                except ValueError:
                    continue
                if resolved.is_file():
                    yield resolved

    def search_text(
        self, query: str, path: str = ".", *, max_results: int = 100, context_lines: int = 0
    ) -> SearchResult | ContextSearchResult:
        """Return deterministic, case-sensitive literal matches in path/line order."""
        if not isinstance(query, str) or not query or "\n" in query or "\r" in query or "\0" in query:
            raise ToolInputError("query must be one non-empty line of text")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 1000:
            raise ToolInputError("max_results must be an integer from 1 through 1000")
        if isinstance(context_lines, bool) or not isinstance(context_lines, int) or not 0 <= context_lines <= 10:
            raise ToolInputError("context_lines must be an integer from 0 through 10")
        relative, scope_path = self._resolve(path, allow_root=True)
        matches: list[SearchMatch] = []
        truncated = False
        for file_path in self._files(scope_path):
            try:
                lines = self._text_lines(file_path)
            except ToolInputError:
                continue
            display_path = file_path.relative_to(self._root).as_posix()
            for number, line in enumerate(lines, start=1):
                if query in line:
                    if len(matches) == max_results:
                        truncated = True
                        break
                    matches.append(SearchMatch(display_path, number, line))
            if truncated:
                break
        scope = "." if relative == PurePosixPath(".") else relative.as_posix()
        base_result = SearchResult(query, scope, tuple(matches), truncated)
        if context_lines == 0:
            return base_result
        contextual = []
        cached_lines: dict[str, list[str]] = {}
        for match in matches:
            if match.path not in cached_lines:
                cached_lines[match.path] = self._text_lines(
                    self._root.joinpath(*PurePosixPath(match.path).parts)
                )
            file_lines = cached_lines[match.path]
            before_start = max(1, match.line_number - context_lines)
            after_end = min(len(file_lines), match.line_number + context_lines)
            before = tuple(FileLine(n, file_lines[n - 1]) for n in range(before_start, match.line_number))
            after = tuple(FileLine(n, file_lines[n - 1]) for n in range(match.line_number + 1, after_end + 1))
            contextual.append(ContextSearchMatch(match.path, match.line_number, match.line, before, after))
        return ContextSearchResult(query, scope, context_lines, tuple(contextual), truncated)

    def read_file(self, path: str, start_line: int, end_line: int) -> ReadResult:
        """Read a 1-based inclusive line range from one text file."""
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start_line, end_line)):
            raise ToolInputError("line numbers must be integers")
        if start_line < 1 or end_line < start_line:
            raise ToolInputError("line range must satisfy 1 <= start_line <= end_line")
        if end_line - start_line + 1 > 1000:
            raise ToolInputError("one read_file call may request at most 1000 lines")
        relative, file_path = self._resolve(path)
        if not file_path.is_file():
            raise ToolInputError("read_file path must identify a file")
        lines = self._text_lines(file_path)
        selected = tuple(
            FileLine(number, lines[number - 1])
            for number in range(start_line, min(end_line, len(lines)) + 1)
        )
        actual_end = selected[-1].line_number if selected else start_line - 1
        return ReadResult(relative.as_posix(), start_line, actual_end, len(lines), selected)

    def read_function(self, path: str, symbol: str, index, *, max_lines: int = 1000) -> FunctionReadResult:
        """Read one exact indexed C function from an already selected file."""
        if not isinstance(symbol, str) or not symbol:
            raise ToolInputError("symbol must be a non-empty string")
        relative, file_path = self._resolve(path)
        if not file_path.is_file():
            raise ToolInputError("function lookup path must identify a file")
        try:
            entry = index.file(relative.as_posix())
        except KeyError:
            raise ToolInputError("selected file is absent from the function index") from None
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if entry.sha256 != digest:
            raise ToolInputError("function index does not match the selected source file")
        definitions = tuple(d for d in entry.definitions if d.symbol == symbol)
        candidates = tuple(FunctionCandidate(d.start_line, d.end_line, d.complete) for d in definitions)
        file_lines = self._text_lines(file_path)
        common = dict(path=relative.as_posix(), symbol=symbol, candidate_count=len(candidates),
                      candidates=candidates, total_lines=len(file_lines))
        if not candidates:
            return FunctionReadResult(status="not_found", selected_start_line=None, selected_end_line=None,
                                      returned_start_line=None, returned_end_line=None, lines=(), truncated=False,
                                      **common)
        if len(candidates) != 1:
            return FunctionReadResult(status="ambiguous", selected_start_line=None, selected_end_line=None,
                                      returned_start_line=None, returned_end_line=None, lines=(), truncated=False,
                                      **common)
        candidate = candidates[0]
        if not candidate.complete:
            return FunctionReadResult(status="incomplete", selected_start_line=None, selected_end_line=None,
                                      returned_start_line=None, returned_end_line=None, lines=(), truncated=False,
                                      **common)
        returned_end = min(candidate.end_line, candidate.start_line + max_lines - 1)
        lines = tuple(FileLine(n, file_lines[n - 1]) for n in range(candidate.start_line, returned_end + 1))
        return FunctionReadResult(
            status="found", selected_start_line=candidate.start_line, selected_end_line=candidate.end_line,
            returned_start_line=candidate.start_line, returned_end_line=returned_end, lines=lines,
            truncated=returned_end < candidate.end_line, **common
        )

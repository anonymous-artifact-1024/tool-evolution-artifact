"""Strict original and refactored JSON interfaces over the shared backend."""

from __future__ import annotations

from typing import Any

from .backend import SourceWorkspace, ToolInputError


def _object(value: Any, label: str, allowed: set[str], required: set[str] = frozenset()) -> dict:
    if type(value) is not dict:
        raise ToolInputError(f"{label} must be a JSON object")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ToolInputError(f"unknown {label} fields: {sorted(unknown)}")
    if missing:
        raise ToolInputError(f"missing {label} fields: {sorted(missing)}")
    if any(item is None for item in value.values()):
        raise ToolInputError(f"explicit null is not allowed in {label}")
    return value


def _string(value: Any, label: str, *, nonempty: bool = True, max_bytes: int = 4096) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ToolInputError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > max_bytes:
        raise ToolInputError(f"{label} exceeds the {max_bytes}-byte input limit")
    return value


def _integer(value: Any, label: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        suffix = f" through {high}" if high is not None else " or greater"
        raise ToolInputError(f"{label} must be an integer from {low}{suffix}")
    return value


class SearchToolInterface:
    def __init__(self, workspace: SourceWorkspace, *, refactored: bool, extended: bool):
        self.workspace = workspace
        self.refactored = refactored
        self.extended = extended

    @property
    def name(self) -> str:
        return "query_repository_text" if self.refactored else "search_text"

    def call(self, arguments: dict):
        if self.refactored:
            top = _object(arguments, "arguments", {"target", "options"}, {"target"})
            target = _object(top["target"], "target", {"query", "root_path"}, {"query"})
            allowed_options = {"result_limit"}
            if self.extended:
                allowed_options.add("context_lines")
            options = _object(top.get("options", {}), "options", allowed_options)
            query = _string(target["query"], "target.query")
            root_path = _string(target.get("root_path", "."), "target.root_path")
            max_results = _integer(options.get("result_limit", 20), "options.result_limit", 1, 20)
            context = _integer(options.get("context_lines", 0), "options.context_lines", 0, 10)
        else:
            allowed = {"pattern", "root_path", "max_results"}
            if self.extended:
                allowed.add("context_lines")
            values = _object(arguments, "arguments", allowed, {"pattern"})
            query = _string(values["pattern"], "pattern")
            root_path = _string(values.get("root_path", "."), "root_path")
            max_results = _integer(values.get("max_results", 20), "max_results", 1, 20)
            context = _integer(values.get("context_lines", 0), "context_lines", 0, 10)
        return self.workspace.search_text(query, root_path, max_results=max_results, context_lines=context)


class FileReadToolInterface:
    def __init__(self, workspace: SourceWorkspace, function_index=None, *, refactored: bool, extended: bool):
        if extended and function_index is None:
            raise ValueError("extended file reading requires a function index")
        self.workspace = workspace
        self.function_index = function_index
        self.refactored = refactored
        self.extended = extended

    @property
    def name(self) -> str:
        return "fetch_file_slice" if self.refactored else "read_file"

    def call(self, arguments: dict):
        if self.refactored:
            top = _object(arguments, "arguments", {"location", "options"}, {"location"})
            location = _object(top["location"], "location", {"file_path", "line_start", "line_end"},
                               {"file_path"})
            allowed_options = {"symbol"} if self.extended else set()
            options = _object(top.get("options", {}), "options", allowed_options)
            path = _string(location["file_path"], "location.file_path")
            start = _integer(location.get("line_start", 1), "location.line_start", 1)
            end = _integer(location.get("line_end", start + 200), "location.line_end", 1)
            symbol = _string(options["symbol"], "options.symbol") if "symbol" in options else None
        else:
            allowed = {"file_path", "start_line", "end_line"}
            if self.extended:
                allowed.add("symbol")
            values = _object(arguments, "arguments", allowed, {"file_path"})
            path = _string(values["file_path"], "file_path")
            start = _integer(values.get("start_line", 1), "start_line", 1)
            end = _integer(values.get("end_line", start + 200), "end_line", 1)
            symbol = _string(values["symbol"], "symbol") if "symbol" in values else None
        if symbol is None:
            return self.workspace.read_file(path, start, end)
        return self.workspace.read_function(path, symbol, self.function_index)

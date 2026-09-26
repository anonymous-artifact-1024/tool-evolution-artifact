"""Exact tool registration and dispatch for a frozen experiment condition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from saner_exp.conditions import ConditionSpec

from .backend import SourceWorkspace, ToolInputError
from .common import (
    CommandRunner, CommandToolInterface, DirectoryToolInterface, SubmissionBox, SubmitToolInterface,
)
from .interfaces import FileReadToolInterface, SearchToolInterface
from .schemas import common_schemas, read_schema, search_schema
from .serialization import SerializedPayload, serialize_error, serialize_success


@dataclass(frozen=True, slots=True)
class ToolObservation:
    tool_name: str
    operation: str | None
    payload: SerializedPayload
    terminal: bool


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    operation: str
    interface: object
    schema: dict
    terminal: bool = False


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool]):
        self._tools = {tool.interface.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("duplicate visible tool name")

    @property
    def schemas(self) -> tuple[dict, ...]:
        return tuple(deepcopy(tool.schema) for tool in self._tools.values())

    @property
    def schema_sha256(self) -> str:
        encoded = json.dumps(
            self.schemas, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def call(self, tool_name: str, arguments) -> ToolObservation:
        tool = self._tools.get(tool_name)
        if tool is None:
            payload = serialize_error("unknown_tool", f"unknown tool: {tool_name!r}")
            return ToolObservation(tool_name, None, payload, False)
        try:
            result = tool.interface.call(arguments)
            payload = serialize_success(result)
        except ToolInputError as error:
            payload = serialize_error("invalid_arguments", str(error))
            return ToolObservation(tool_name, tool.operation, payload, False)
        return ToolObservation(tool_name, tool.operation, payload, tool.terminal)


def build_tool_registry(
    condition: ConditionSpec,
    workspace: SourceWorkspace,
    command_runner: CommandRunner,
    submission_box: SubmissionBox,
    function_index=None,
) -> ToolRegistry:
    if condition.read.extended and function_index is None:
        raise ValueError(f"condition {condition.condition} requires a function index")
    search = SearchToolInterface(
        workspace, refactored=condition.search.refactored, extended=condition.search.extended,
    )
    read = FileReadToolInterface(
        workspace, function_index, refactored=condition.read.refactored, extended=condition.read.extended,
    )
    common = common_schemas()
    return ToolRegistry([
        RegisteredTool("directory", DirectoryToolInterface(workspace), common["list_directory"]),
        RegisteredTool(
            "search", search,
            search_schema(refactored=condition.search.refactored, extended=condition.search.extended),
        ),
        RegisteredTool(
            "read", read,
            read_schema(refactored=condition.read.refactored, extended=condition.read.extended),
        ),
        RegisteredTool("command", CommandToolInterface(command_runner), common["run_command"]),
        RegisteredTool(
            "submit", SubmitToolInterface(submission_box), common["submit_result"], terminal=True,
        ),
    ])

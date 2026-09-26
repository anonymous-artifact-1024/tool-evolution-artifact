"""Visible OpenAI-compatible function schemas for the experimental tools."""

from __future__ import annotations


def function_schema(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def search_schema(*, refactored: bool, extended: bool) -> dict:
    description = "Search source files for case-sensitive literal text matches."
    if extended:
        description += " Optionally include nearby source lines around each core match."
    if not refactored:
        properties = {
            "pattern": {"type": "string", "minLength": 1, "maxLength": 4096},
            "root_path": {"type": "string", "minLength": 1, "maxLength": 4096, "default": "."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 20},
        }
        if extended:
            properties["context_lines"] = {
                "type": "integer", "minimum": 0, "maximum": 10, "default": 0,
            }
        return function_schema("search_text", description, {
            "type": "object", "properties": properties, "required": ["pattern"],
            "additionalProperties": False,
        })
    option_properties = {
        "result_limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 20},
    }
    if extended:
        option_properties["context_lines"] = {
            "type": "integer", "minimum": 0, "maximum": 10, "default": 0,
        }
    return function_schema("query_repository_text", description, {
        "type": "object",
        "properties": {
            "target": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "root_path": {"type": "string", "minLength": 1, "maxLength": 4096, "default": "."},
                },
                "required": ["query"], "additionalProperties": False,
            },
            "options": {
                "type": "object", "properties": option_properties, "additionalProperties": False,
            },
        },
        "required": ["target"], "additionalProperties": False,
    })


def read_schema(*, refactored: bool, extended: bool) -> dict:
    description = "Read an inclusive line range from an existing source file."
    if extended:
        description += " An exact function symbol may select its indexed definition range within that file."
    if not refactored:
        properties = {
            "file_path": {"type": "string", "minLength": 1, "maxLength": 4096},
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "end_line": {
                "type": "integer", "minimum": 1,
                "description": "Defaults to start_line + 200.",
            },
        }
        if extended:
            properties["symbol"] = {"type": "string", "minLength": 1, "maxLength": 4096}
        return function_schema("read_file", description, {
            "type": "object", "properties": properties, "required": ["file_path"],
            "additionalProperties": False,
        })
    options = {"symbol": {"type": "string", "minLength": 1, "maxLength": 4096}} if extended else {}
    return function_schema("fetch_file_slice", description, {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "line_start": {"type": "integer", "minimum": 1, "default": 1},
                    "line_end": {
                        "type": "integer", "minimum": 1,
                        "description": "Defaults to line_start + 200.",
                    },
                },
                "required": ["file_path"], "additionalProperties": False,
            },
            "options": {"type": "object", "properties": options, "additionalProperties": False},
        },
        "required": ["location"], "additionalProperties": False,
    })


def common_schemas() -> dict[str, dict]:
    return {
        "list_directory": function_schema(
            "list_directory", "List entries in one source directory without following links.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 4096, "default": "."},
                    "max_entries": {
                        "type": "integer", "minimum": 1, "maximum": 200, "default": 200,
                    },
                },
                "additionalProperties": False,
            },
        ),
        "run_command": function_schema(
            "run_command", "Run one allowed read-only source inspection command without a shell.",
            {
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 1024},
                        "minItems": 1, "maxItems": 64,
                    },
                    "timeout_seconds": {
                        "type": "integer", "minimum": 1, "maximum": 300, "default": 300,
                    },
                },
                "required": ["argv"], "additionalProperties": False,
            },
        ),
        "submit_result": function_schema(
            "submit_result", "Submit a ranked list of up to ten suspected source files and end the episode.",
            {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 4096},
                        "minItems": 1, "maxItems": 10,
                    },
                },
                "required": ["paths"], "additionalProperties": False,
            },
        ),
    }

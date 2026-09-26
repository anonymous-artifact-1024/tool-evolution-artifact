"""Serializable exact C-function ranges for symbol-based file reading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class FunctionDefinition:
    symbol: str
    start_line: int
    end_line: int
    complete: bool


@dataclass(frozen=True, slots=True)
class IndexedFile:
    sha256: str
    definitions: tuple[FunctionDefinition, ...]


class FunctionIndex:
    def __init__(self, files: dict[str, IndexedFile]):
        self._files = dict(files)

    def file(self, path: str) -> IndexedFile:
        try:
            return self._files[path]
        except KeyError:
            raise KeyError(f"file is absent from the function index: {path}") from None

    @classmethod
    def load(cls, path: Path | str) -> "FunctionIndex":
        record = json.loads(Path(path).read_bytes())
        if record.get("format") != "saner-c-function-index-v1":
            raise ValueError("unsupported function index format")
        files = {}
        for file_path, value in record["files"].items():
            definitions = tuple(FunctionDefinition(**item) for item in value["definitions"])
            if list(definitions) != sorted(definitions, key=lambda d: (d.start_line, d.end_line, d.symbol)):
                raise ValueError(f"unordered function definitions for {file_path}")
            files[file_path] = IndexedFile(value["sha256"], definitions)
        return cls(files)


@lru_cache(maxsize=1)
def _c_language():
    from tree_sitter import Language
    import tree_sitter_c

    return Language(tree_sitter_c.language())


@lru_cache(maxsize=1)
def _c_parser():
    from tree_sitter import Parser

    return Parser(_c_language())


def parse_c_functions_with_diagnostics(content: bytes) -> tuple[tuple[FunctionDefinition, ...], int]:
    """Parse function-definition nodes with pinned tree-sitter C grammar semantics."""
    tree = _c_parser().parse(content)
    definitions = []
    unsupported = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            start = node.start_point
            end = node.end_point
            declarator = node.child_by_field_name("declarator")
            for _ in range(32):
                if declarator is None or declarator.type == "identifier":
                    break
                declarator = declarator.child_by_field_name("declarator")
            if declarator is None or declarator.type != "identifier":
                unsupported += 1
                stack.extend(reversed(node.children))
                continue
            name_start = declarator.start_byte
            name_end = declarator.end_byte
            symbol = content[name_start:name_end].decode("utf-8", errors="strict")
            body = node.child_by_field_name("body")
            body_start = body.start_byte
            body_end = body.end_byte
            complete = content[body_start:body_end].rstrip().endswith(b"}")
            definitions.append(FunctionDefinition(symbol, start.row + 1, end.row + 1, complete))
        stack.extend(reversed(node.children))
    return tuple(sorted(definitions, key=lambda d: (d.start_line, d.end_line, d.symbol))), unsupported


def parse_c_functions(content: bytes) -> tuple[FunctionDefinition, ...]:
    return parse_c_functions_with_diagnostics(content)[0]


def make_index_record(files: Iterable[tuple[str, bytes]], *, source_identity: dict) -> dict:
    indexed = {}
    previous = None
    for path, content in files:
        if previous is not None and path <= previous:
            raise ValueError("function-index inputs must be unique and sorted by POSIX path")
        previous = path
        if not path.endswith((".c", ".h")):
            continue
        definitions, unsupported = parse_c_functions_with_diagnostics(content)
        indexed[path] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "unsupported_function_nodes": unsupported,
            "definitions": [
                {"symbol": d.symbol, "start_line": d.start_line, "end_line": d.end_line,
                 "complete": d.complete}
                for d in definitions
            ],
        }
    return {"format": "saner-c-function-index-v1", "source_identity": source_identity, "files": indexed}

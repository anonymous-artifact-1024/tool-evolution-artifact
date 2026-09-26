"""Authoritative formal-experiment tool-contract verification."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from saner_exp.tools import (  # noqa: E402
    FileReadToolInterface, FunctionDefinition, FunctionIndex, IndexedFile,
    RestrictedCommandRunner, SearchToolInterface, SourceWorkspace, ToolInputError,
)
from saner_exp.tools.function_index import parse_c_functions_with_diagnostics  # noqa: E402
from saner_exp.tools.serialization import serialize_success  # noqa: E402


SEED = 2027
CALLS_PER_CATEGORY_PER_PAIR = 10_000
OUTPUT = PROJECT / "manifests/tool_contract_verification_v2.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def error_class(call) -> str:
    try:
        call()
    except Exception as error:
        return type(error).__name__
    return "no_error"


def assert_same(left, right, label: str) -> None:
    if left != right:
        raise AssertionError(f"mapped calls differ: {label}")


def search_args(*, refactored: bool, query: str, path: str = ".", limit: int = 20,
                context: int | None = None) -> dict:
    if refactored:
        value = {"target": {"query": query, "root_path": path},
                 "options": {"result_limit": limit}}
        if context is not None:
            value["options"]["context_lines"] = context
        return value
    value = {"pattern": query, "root_path": path, "max_results": limit}
    if context is not None:
        value["context_lines"] = context
    return value


def read_args(*, refactored: bool, path: str, start: int = 1, end: int = 1,
              symbol: str | None = None) -> dict:
    if refactored:
        value = {"location": {"file_path": path, "line_start": start, "line_end": end}}
        if symbol is not None:
            value["options"] = {"symbol": symbol}
        return value
    value = {"file_path": path, "start_line": start, "end_line": end}
    if symbol is not None:
        value["symbol"] = symbol
    return value


def invalid_search_pair(case: int, extended: bool) -> tuple[dict, dict]:
    cases = [
        ({"pattern": ""}, {"target": {"query": ""}}),
        ({"pattern": "x", "max_results": 0},
         {"target": {"query": "x"}, "options": {"result_limit": 0}}),
        ({"pattern": "x", "max_results": 21},
         {"target": {"query": "x"}, "options": {"result_limit": 21}}),
        ({"pattern": "x", "max_results": True},
         {"target": {"query": "x"}, "options": {"result_limit": True}}),
        ({"pattern": "x", "root_path": "../outside"},
         {"target": {"query": "x", "root_path": "../outside"}}),
        ({"pattern": "x", "unknown": 1},
         {"target": {"query": "x"}, "unknown": 1}),
        ([], []),
    ]
    if extended:
        cases.extend([
            ({"pattern": "x", "context_lines": -1},
             {"target": {"query": "x"}, "options": {"context_lines": -1}}),
            ({"pattern": "x", "context_lines": 11},
             {"target": {"query": "x"}, "options": {"context_lines": 11}}),
            ({"pattern": "x", "context_lines": True},
             {"target": {"query": "x"}, "options": {"context_lines": True}}),
        ])
    return cases[case % len(cases)]


def invalid_read_pair(case: int, extended: bool) -> tuple[dict, dict]:
    cases = [
        ({"file_path": ""}, {"location": {"file_path": ""}}),
        ({"file_path": "sample.c", "start_line": 0},
         {"location": {"file_path": "sample.c", "line_start": 0}}),
        ({"file_path": "sample.c", "start_line": True},
         {"location": {"file_path": "sample.c", "line_start": True}}),
        ({"file_path": "sample.c", "start_line": 5, "end_line": 4},
         {"location": {"file_path": "sample.c", "line_start": 5, "line_end": 4}}),
        ({"file_path": "../sample.c"}, {"location": {"file_path": "../sample.c"}}),
        ({"file_path": "sample.c", "unknown": 1},
         {"location": {"file_path": "sample.c"}, "unknown": 1}),
        (None, None),
    ]
    if extended:
        cases.extend([
            ({"file_path": "sample.c", "symbol": ""},
             {"location": {"file_path": "sample.c"}, "options": {"symbol": ""}}),
            ({"file_path": "sample.c", "symbol": None},
             {"location": {"file_path": "sample.c"}, "options": {"symbol": None}}),
        ])
    return cases[case % len(cases)]


def verify() -> dict:
    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        content = (
            "// fixture\nstatic int alpha(int x)\n{\n    return x + 1;\n}\n"
            "// needle\nint duplicate(void) { return 1; }\n"
            "int duplicate(int x) { return x; }\n// tail\n"
        ).encode()
        (root / "sample.c").write_bytes(content)
        (root / "other.h").write_text("needle\n" * 40, encoding="utf-8")
        workspace = SourceWorkspace(root)
        index = FunctionIndex({"sample.c": IndexedFile(hashlib.sha256(content).hexdigest(), (
            FunctionDefinition("alpha", 2, 5, True),
            FunctionDefinition("duplicate", 7, 7, True),
            FunctionDefinition("duplicate", 8, 8, True),
        ))})
        initial_hashes = {path.name: sha256(path) for path in root.iterdir() if path.is_file()}
        mapped = {}

        for extended in (False, True):
            left = SearchToolInterface(workspace, refactored=False, extended=extended)
            right = SearchToolInterface(workspace, refactored=True, extended=extended)
            key = f"search_functionality_{int(extended)}"
            for i in range(CALLS_PER_CATEGORY_PER_PAIR):
                query = rng.choice(("int", "needle", "tail", "absent"))
                limit = rng.randint(2, 19)
                context = rng.randint(1, 9) if extended else None
                assert_same(left.call(search_args(refactored=False, query=query, limit=limit,
                                                  context=context)),
                            right.call(search_args(refactored=True, query=query, limit=limit,
                                                   context=context)), f"{key}/normal/{i}")
            for i in range(CALLS_PER_CATEGORY_PER_PAIR):
                limit = 1 if i % 2 == 0 else 20
                context = (0 if i % 2 == 0 else 10) if extended else None
                assert_same(left.call(search_args(refactored=False, query="needle", limit=limit,
                                                  context=context)),
                            right.call(search_args(refactored=True, query="needle", limit=limit,
                                                   context=context)), f"{key}/boundary/{i}")
            for i in range(CALLS_PER_CATEGORY_PER_PAIR):
                a, b = invalid_search_pair(i, extended)
                classes = (error_class(lambda a=a: left.call(a)),
                           error_class(lambda b=b: right.call(b)))
                if classes[0] != classes[1] or classes[0] == "no_error":
                    raise AssertionError(f"invalid mapped search class differs: {classes}")
            mapped[key] = {"normal": 10000, "boundary": 10000, "invalid": 10000}

        for extended in (False, True):
            left = FileReadToolInterface(workspace, index if extended else None,
                                         refactored=False, extended=extended)
            right = FileReadToolInterface(workspace, index if extended else None,
                                          refactored=True, extended=extended)
            key = f"file_functionality_{int(extended)}"
            for i in range(CALLS_PER_CATEGORY_PER_PAIR):
                start = rng.randint(1, 7)
                end = rng.randint(start, 9)
                symbol = rng.choice(("alpha", "duplicate", "missing")) if extended and i % 2 else None
                assert_same(left.call(read_args(refactored=False, path="sample.c", start=start,
                                                end=end, symbol=symbol)),
                            right.call(read_args(refactored=True, path="sample.c", start=start,
                                                 end=end, symbol=symbol)), f"{key}/normal/{i}")
            for i in range(CALLS_PER_CATEGORY_PER_PAIR):
                start, end = ((1, 1) if i % 2 == 0 else (1, 9))
                symbol = ("missing" if i % 3 == 0 else "duplicate") if extended else None
                assert_same(left.call(read_args(refactored=False, path="sample.c", start=start,
                                                end=end, symbol=symbol)),
                            right.call(read_args(refactored=True, path="sample.c", start=start,
                                                 end=end, symbol=symbol)), f"{key}/boundary/{i}")
            for i in range(CALLS_PER_CATEGORY_PER_PAIR):
                a, b = invalid_read_pair(i, extended)
                classes = (error_class(lambda a=a: left.call(a)),
                           error_class(lambda b=b: right.call(b)))
                if classes[0] != classes[1] or classes[0] == "no_error":
                    raise AssertionError(f"invalid mapped file class differs: {classes}")
            mapped[key] = {"normal": 10000, "boundary": 10000, "invalid": 10000}

        conservative = {}
        for refactored in (False, True):
            base_search = SearchToolInterface(workspace, refactored=refactored, extended=False)
            ext_search = SearchToolInterface(workspace, refactored=refactored, extended=True)
            base_read = FileReadToolInterface(workspace, refactored=refactored, extended=False)
            ext_read = FileReadToolInterface(workspace, index, refactored=refactored, extended=True)
            for i in range(CALLS_PER_CATEGORY_PER_PAIR):
                search = search_args(refactored=refactored, query=("needle" if i % 2 else "int"),
                                     limit=1 + i % 20)
                read = read_args(refactored=refactored, path="sample.c", start=1, end=1 + i % 9)
                assert_same(base_search.call(search), ext_search.call(search), "old search call")
                assert_same(base_read.call(read), ext_read.call(read), "old file call")
            conservative[f"style_{int(refactored)}_search"] = 10000
            conservative[f"style_{int(refactored)}_file"] = 10000

        conditional = b"#if ENABLE_A\nint conditional_a(void) { return 1; }\n#else\nint conditional_b(void) { return 2; }\n#endif\n"
        definitions, unsupported = parse_c_functions_with_diagnostics(conditional)
        if {item.symbol for item in definitions} != {"conditional_a", "conditional_b"}:
            raise AssertionError("conditional-compilation fixture lost a function definition")

        long_lines = [f"int line_{i};" for i in range(1, 1206)]
        long_path = root / "long.c"
        long_path.write_text("\n".join(long_lines) + "\n", encoding="utf-8")
        long_index = FunctionIndex({"long.c": IndexedFile(sha256(long_path), (
            FunctionDefinition("huge", 1, 1205, True),
        ))})
        long_result = workspace.read_function("long.c", "huge", long_index)
        if not (long_result.status == "found" and long_result.truncated
                and len(long_result.lines) == 1000 and long_result.returned_end_line == 1000):
            raise AssertionError("function range truncation contract failed")
        payload = serialize_success(long_result)
        if payload.size_bytes > 32768 or payload.core_source_bytes > 24576 or not payload.truncated:
            raise AssertionError("serialized response budgets were not enforced")
        ambiguous = workspace.read_function("sample.c", "duplicate", index)
        missing = workspace.read_function("sample.c", "missing", index)
        if ambiguous.status != "ambiguous" or missing.status != "not_found":
            raise AssertionError("function ambiguity/not-found statuses changed")

        isolation_cases = {
            "workspace_parent_traversal": error_class(
                lambda: workspace.search_text("x", "../outside")
            ),
            "workspace_absolute_path": error_class(
                lambda: workspace.read_file(root.resolve().as_posix(), 1, 1)
            ),
            "command_parent_traversal": error_class(
                lambda: RestrictedCommandRunner.validate_argv(["cat", "../outside"])
            ),
            "command_forbidden_exec": error_class(
                lambda: RestrictedCommandRunner.validate_argv(["find", ".", "-exec", "cat"])
            ),
            "command_shell_rejected": error_class(
                lambda: RestrictedCommandRunner.validate_argv(["sh", "-c", "id"])
            ),
        }
        if set(isolation_cases.values()) != {"WorkspacePathError", "ToolInputError"}:
            raise AssertionError(f"sandbox isolation checks changed: {isolation_cases}")
        symlink_result = "not_supported_by_host"
        outside = root.parent / f"saner-contract-outside-{os.getpid()}.txt"
        try:
            outside.write_text("outside", encoding="utf-8")
            link = root / "outside-link"
            try:
                link.symlink_to(outside)
                symlink_result = error_class(lambda: workspace.read_file("outside-link", 1, 1))
                if symlink_result != "WorkspacePathError":
                    raise AssertionError("workspace followed an escaping symlink")
            except OSError:
                pass
        finally:
            outside.unlink(missing_ok=True)

        final_hashes = {path.name: sha256(path) for path in root.iterdir()
                        if path.is_file() and path.name in initial_hashes}
        if initial_hashes != final_hashes:
            raise AssertionError("tool verification mutated its source fixtures")

    tracked = [
        PROJECT / "src/saner_exp/tools/backend.py",
        PROJECT / "src/saner_exp/tools/common.py",
        PROJECT / "src/saner_exp/tools/function_index.py",
        PROJECT / "src/saner_exp/tools/interfaces.py",
        PROJECT / "src/saner_exp/tools/registry.py",
        PROJECT / "src/saner_exp/tools/serialization.py",
        Path(__file__),
    ]
    return {
        "record_kind": "formal_tool_contract_verification_v2",
        "status": "passed",
        "seed": SEED,
        "calls_per_category_per_mapped_pair": CALLS_PER_CATEGORY_PER_PAIR,
        "mapped_pair_counts": mapped,
        "conservative_extension_counts": conservative,
        "explicit_checks": {
            "zero_context": "passed",
            "omitted_symbol": "passed",
            "function_index_ranges": "passed",
            "multiple_definitions": "passed",
            "conditional_compilation": {"status": "passed", "unsupported_nodes": unsupported},
            "error_class_mapping": "passed",
            "response_truncation": "passed",
            "response_total_bytes_limit": 32768,
            "response_core_source_bytes_limit": 24576,
            "sandbox_isolation": isolation_cases,
            "escaping_symlink": symlink_result,
            "fixture_unchanged": True,
        },
        "python_version": sys.version,
        "tree_sitter_version": importlib.metadata.version("tree-sitter"),
        "tree_sitter_c_version": importlib.metadata.version("tree-sitter-c"),
        "source_sha256": {path.relative_to(PROJECT).as_posix(): sha256(path) for path in tracked},
    }


def main() -> int:
    record = verify()
    encoded = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if OUTPUT.exists() and OUTPUT.read_bytes() != encoded:
        raise SystemExit("existing v2 tool verification differs from the deterministic rebuild")
    OUTPUT.write_bytes(encoded)
    print(json.dumps({
        "status": "passed",
        "mapped_pairs": len(record["mapped_pair_counts"]),
        "generated_mapped_calls": 4 * 3 * CALLS_PER_CATEGORY_PER_PAIR,
        "conservative_calls": sum(record["conservative_extension_counts"].values()),
        "record": str(OUTPUT),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

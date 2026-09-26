"""Deterministic JSON observations with the protocol's response-size budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json


TOTAL_RESPONSE_BYTES = 32 * 1024
CORE_SOURCE_BYTES = 24 * 1024


@dataclass(frozen=True, slots=True)
class SerializedPayload:
    content: str
    size_bytes: int
    core_source_bytes: int
    truncated: bool


def _encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _prefix(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text
    return raw[:budget].decode("utf-8", errors="ignore")


def _core_text_slots(data: dict):
    if "matches" in data:
        for match in data["matches"]:
            if "line" in match:
                yield match, "line"
    if "lines" in data:
        for line in data["lines"]:
            if "text" in line:
                yield line, "text"


def _trim_core(data: dict) -> tuple[int, bool]:
    used = 0
    truncated = False
    for owner, key in _core_text_slots(data):
        text = owner[key]
        remaining = CORE_SOURCE_BYTES - used
        clipped = _prefix(text, remaining)
        if clipped != text:
            truncated = True
        owner[key] = clipped
        used += len(clipped.encode("utf-8"))
    return used, truncated


def _remove_context(data: dict) -> bool:
    for match in reversed(data.get("matches", [])):
        if match.get("after"):
            match["after"].pop()
            return True
        if match.get("before"):
            match["before"].pop(0)
            return True
    return False


def _remove_bulk_item(data: dict) -> bool:
    for key in ("entries", "candidates", "lines", "matches"):
        values = data.get(key)
        if values:
            values.pop()
            if key == "candidates":
                data["candidates_truncated"] = True
            if key in ("lines", "matches") and "truncated" in data:
                data["truncated"] = True
            return True
    if len(data.get("argv", [])) > 1:
        data["argv"].pop()
        data["argv_truncated"] = True
        return True
    return False


def _measure_core(data: dict) -> int:
    return sum(len(owner[key].encode("utf-8")) for owner, key in _core_text_slots(data))


def serialize_success(result) -> SerializedPayload:
    if not is_dataclass(result):
        raise TypeError("tool result must be a dataclass instance")
    # Work on a mutable JSON-shaped tree: asdict deliberately preserves tuples.
    data = json.loads(json.dumps(asdict(result), ensure_ascii=False))
    _, truncated = _trim_core(data)
    if truncated and "truncated" in data:
        data["truncated"] = True
    data["response_truncated"] = truncated
    wrapper = {"ok": True, "data": data}
    while len(_encoded(wrapper)) > TOTAL_RESPONSE_BYTES:
        if _remove_context(data) or _remove_bulk_item(data):
            truncated = True
            data["response_truncated"] = True
            continue
        changed = False
        for key in ("stdout", "stderr"):
            if data.get(key):
                data[key] = _prefix(data[key], len(data[key].encode("utf-8")) // 2)
                changed = truncated = True
                data["response_truncated"] = True
                break
        if not changed:
            raise ValueError("tool response metadata exceeds the 32 KiB protocol limit")
    content = _encoded(wrapper).decode("utf-8")
    return SerializedPayload(content, len(content.encode("utf-8")), _measure_core(data), truncated)


def serialize_error(category: str, message: str) -> SerializedPayload:
    error = {"ok": False, "error": {"category": category, "message": _prefix(message, 4096)}}
    content = _encoded(error).decode("utf-8")
    return SerializedPayload(content, len(content.encode("utf-8")), 0, False)

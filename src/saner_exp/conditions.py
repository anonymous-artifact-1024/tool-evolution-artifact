"""Load and validate the seven factorial-condition configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


EXPECTED = {
    "A": (("base", "original"), ("base", "original")),
    "B1": (("base", "refactored"), ("base", "original")),
    "C1": (("extended", "original"), ("base", "original")),
    "D1": (("extended", "refactored"), ("base", "original")),
    "B2": (("base", "original"), ("base", "refactored")),
    "C2": (("base", "original"), ("extended", "original")),
    "D2": (("base", "original"), ("extended", "refactored")),
}


@dataclass(frozen=True, slots=True)
class ToolVariant:
    functionality: str
    interface: str

    @property
    def extended(self) -> bool:
        return self.functionality == "extended"

    @property
    def refactored(self) -> bool:
        return self.interface == "refactored"


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    condition: str
    search: ToolVariant
    read: ToolVariant


def load_conditions(directory: Path | str) -> dict[str, ConditionSpec]:
    directory = Path(directory)
    specs = {}
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_bytes())
        if set(value) != {"condition", "search", "read"}:
            raise ValueError(f"unexpected condition fields: {path}")
        condition = value["condition"]
        if condition in specs or path.stem != condition:
            raise ValueError(f"duplicate or mismatched condition file: {path}")
        variants = []
        for key in ("search", "read"):
            item = value[key]
            if type(item) is not dict or set(item) != {"functionality", "interface"}:
                raise ValueError(f"invalid {key} variant in {path}")
            variant = ToolVariant(item["functionality"], item["interface"])
            if variant.functionality not in {"base", "extended"} or variant.interface not in {"original", "refactored"}:
                raise ValueError(f"unsupported {key} variant in {path}")
            variants.append(variant)
        specs[condition] = ConditionSpec(condition, *variants)
    if set(specs) != set(EXPECTED):
        raise ValueError(f"condition set differs from the seven planned conditions: {sorted(specs)}")
    for name, expected in EXPECTED.items():
        actual = ((specs[name].search.functionality, specs[name].search.interface),
                  (specs[name].read.functionality, specs[name].read.interface))
        if actual != expected:
            raise ValueError(f"condition {name} differs from the planned factorial cell")
    return specs

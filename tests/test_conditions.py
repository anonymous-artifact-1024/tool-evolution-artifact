import json
from pathlib import Path
import shutil

import pytest

from saner_exp.conditions import EXPECTED, load_conditions


PROJECT = Path(__file__).resolve().parents[1]


def test_condition_files_are_exact_factorial_cells():
    conditions = load_conditions(PROJECT / "configs" / "conditions")
    assert set(conditions) == set(EXPECTED)
    for name, expected in EXPECTED.items():
        spec = conditions[name]
        assert ((spec.search.functionality, spec.search.interface),
                (spec.read.functionality, spec.read.interface)) == expected


def test_condition_loader_rejects_missing_or_changed_cell(tmp_path: Path):
    source = PROJECT / "configs" / "conditions"
    target = tmp_path / "conditions"
    shutil.copytree(source, target)
    (target / "D2.json").unlink()
    with pytest.raises(ValueError, match="condition set differs"):
        load_conditions(target)

    shutil.copy(source / "D2.json", target / "D2.json")
    changed = json.loads((target / "D2.json").read_text(encoding="utf-8"))
    changed["read"]["interface"] = "original"
    (target / "D2.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the planned factorial cell"):
        load_conditions(target)

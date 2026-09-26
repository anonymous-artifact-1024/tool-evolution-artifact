import json

import pytest

from saner_exp.data import DatasetValidationError, load_public_tasks


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_public_catalog_loads_only_agent_fields(tmp_path):
    path = tmp_path / "tasks.jsonl"
    write_rows(path, [{
        "id": "7", "title": "bug", "description": "details", "kernel_version": "2.6.22",
    }])
    assert load_public_tasks(path)["7"].title == "bug"


@pytest.mark.parametrize("extra", [{"paths": ["secret.c"]}, {"patch": "secret"}, {"methods": ["f"]}])
def test_public_catalog_rejects_ground_truth_fields(tmp_path, extra):
    path = tmp_path / "tasks.jsonl"
    row = {"id": "7", "title": "bug", "description": "details", "kernel_version": "2.6.22"}
    row.update(extra)
    write_rows(path, [row])
    with pytest.raises(DatasetValidationError):
        load_public_tasks(path)

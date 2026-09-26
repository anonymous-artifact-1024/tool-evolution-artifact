import hashlib
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from saner_exp.data import DatasetValidationError, LinuxFLBenchDataset


@pytest.fixture
def record():
    # Synthetic fixture: never evidence for the paper.
    return {
        "id": "0007", "title": "Example bug", "description": "  official text\n",
        "Kernel Version": "5.6.7", "patch": ["SECRET_PATCH"],
        "paths": ["secret/file.c"], "methods": ["secret_function"],
        "Bisected commit-id": "SECRET_COMMIT", "URL": "SECRET_URL",
    }


def write_dataset(tmp_path, records):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def write_manifest(tmp_path, dataset_path, **overrides):
    manifest = {
        "dataset_name": "LinuxFLBench", "task_count": 1,
        "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    manifest.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_allowlist_preserves_text_and_excludes_labels(tmp_path, record):
    dataset = LinuxFLBenchDataset.load(write_dataset(tmp_path, [record]))
    task = dataset.agent_task("0007")
    assert asdict(task) == {
        "task_id": "0007", "title": record["title"],
        "description": record["description"], "kernel_version": "5.6.7",
    }
    assert "SECRET" not in json.dumps(asdict(task))
    with pytest.raises(FrozenInstanceError):
        task.title = "changed"
    truth = dataset.ground_truth("0007")
    assert truth.patch == ("SECRET_PATCH",)
    assert truth.paths == ("secret/file.c",)
    assert truth.methods == ("secret_function",)
    assert "SECRET_PATCH" not in repr(truth)
    with pytest.raises(TypeError):
        truth.paths[0] = "changed"
    with pytest.raises(KeyError):
        dataset.agent_task(7)  # Never silently coerce identifiers.


@pytest.mark.parametrize("field,value", [
    ("id", 7), ("title", None), ("description", ""), ("Kernel Version", []),
    ("patch", "not a list"), ("paths", []), ("methods", [1]),
])
def test_rejects_invalid_fields(tmp_path, record, field, value):
    record[field] = value
    with pytest.raises(DatasetValidationError, match=f"line 1: {field}"):
        LinuxFLBenchDataset.load(write_dataset(tmp_path, [record]))


def test_missing_field_and_duplicate_id(tmp_path, record):
    missing = dict(record)
    del missing["patch"]
    with pytest.raises(DatasetValidationError, match="patch"):
        LinuxFLBenchDataset.load(write_dataset(tmp_path, [missing]))
    with pytest.raises(DatasetValidationError, match="line 2: duplicate task id"):
        LinuxFLBenchDataset.load(write_dataset(tmp_path, [record, record]))


@pytest.mark.parametrize("content,message", [
    ("\n", "no tasks"), ("{bad json}", "line 1: invalid JSON"),
    ("[]", "record must be an object"),
    ('{"id":"1","id":"2"}', "duplicate JSON field"),
])
def test_rejects_malformed_jsonl(tmp_path, content, message):
    path = tmp_path / "invalid.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        LinuxFLBenchDataset.load(path)


def test_manifest_accepts_exact_bytes_and_detects_tampering(tmp_path, record):
    path = write_dataset(tmp_path, [record])
    manifest = write_manifest(tmp_path, path)
    assert len(LinuxFLBenchDataset.load(path, manifest_path=manifest)) == 1
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DatasetValidationError, match="SHA-256"):
        LinuxFLBenchDataset.load(path, manifest_path=manifest)


@pytest.mark.parametrize("overrides,message", [
    ({"task_count": 2}, "task count"),
    ({"task_count": True}, "positive integer"),
    ({"dataset_name": "other"}, "dataset_name"),
])
def test_manifest_metadata_checks(tmp_path, record, overrides, message):
    path = write_dataset(tmp_path, [record])
    manifest = write_manifest(tmp_path, path, **overrides)
    with pytest.raises(DatasetValidationError, match=message):
        LinuxFLBenchDataset.load(path, manifest_path=manifest)


def test_upstream_dataset_matches_recorded_identity():
    project = Path(__file__).resolve().parents[1]
    path = project.parent / "datasets/LinuxFLBench/dataset/LINUXFLBENCH_dataset.jsonl"
    if not path.exists():
        pytest.skip("External LinuxFLBench checkout is not installed")
    dataset = LinuxFLBenchDataset.load(
        path, manifest_path=project / "data/provenance/dataset_manifest.json"
    )
    assert len(dataset) == len(set(dataset.task_ids)) == 250
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert dataset.task_ids == tuple(row["id"] for row in rows)
    for row in rows:
        task = dataset.agent_task(row["id"])
        assert asdict(task) == {
            "task_id": row["id"], "title": row["title"],
            "description": row["description"], "kernel_version": row["Kernel Version"],
        }
        truth = dataset.ground_truth(row["id"])
        assert truth.patch == tuple(row["patch"])
        assert truth.paths == tuple(row["paths"])
        assert truth.methods == tuple(row["methods"])

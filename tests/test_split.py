import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from saner_exp.data.split import split_task_ids


def test_fixed_reference_permutation_uses_numeric_id_order():
    # Synthetic reference vector guards the chosen shuffle and ordering contract.
    result = split_task_ids([str(x) for x in range(10, 0, -1)], preexperiment_size=3)
    assert result.preexperiment == ("9", "7", "5")
    assert result.main == ("4", "3", "6", "1", "10", "8", "2")


def test_complete_disjoint_repeatable_and_independent_of_input_order():
    ids = [str(x) for x in range(1, 251)]
    expected = split_task_ids(ids)
    assert len(expected.preexperiment) == len(set(expected.preexperiment)) == 20
    assert len(expected.main) == len(set(expected.main)) == 230
    assert not set(expected.preexperiment) & set(expected.main)
    assert set(expected.preexperiment) | set(expected.main) == set(ids)
    assert split_task_ids(reversed(ids)) == expected
    shuffled = list(ids)
    random.Random(42).shuffle(shuffled)
    assert split_task_ids(shuffled) == expected


def test_does_not_read_or_modify_global_rng_state():
    state = random.getstate()
    try:
        ids = [str(x) for x in range(30)]
        expected = split_task_ids(ids)
        assert random.getstate() == state
        random.seed(12345)
        new_state = random.getstate()
        assert split_task_ids(ids) == expected
        assert random.getstate() == new_state
    finally:
        random.setstate(state)


def test_zero_padded_ids_are_preserved_and_ties_are_deterministic():
    ids = ["1", "01", "2", "002"]
    result = split_task_ids(ids, preexperiment_size=1)
    assert result == split_task_ids(reversed(ids), preexperiment_size=1)
    assert set(result.preexperiment + result.main) == set(ids)


@pytest.mark.parametrize("ids", [
    ["1", "1"], [1, "2"], ["", "2"], [" 1", "2"], ["-1", "2"], ["１", "2"],
])
def test_invalid_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        split_task_ids(ids, preexperiment_size=1)


@pytest.mark.parametrize("size", [0, -1, 3, 4, True, 1.5])
def test_invalid_partition_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="both partitions nonempty"):
        split_task_ids(["1", "2", "3"], preexperiment_size=size)


@pytest.mark.parametrize("seed", [True, "2027", 2027.0])
def test_seed_must_be_an_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        split_task_ids(["1", "2"], preexperiment_size=1, seed=seed)


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli():
    dataset = PROJECT.parent / "datasets/LinuxFLBench/dataset/LINUXFLBENCH_dataset.jsonl"
    if not dataset.exists():
        pytest.skip("External LinuxFLBench checkout is not installed")

    def run(output, *args):
        return subprocess.run(
            [sys.executable, str(PROJECT / "scripts/generate_split.py"),
             "--output", str(output), *args],
            cwd=output.parent, capture_output=True, text=True, encoding="utf-8",
        )
    return run


def test_cli_creates_once_then_checks_identical_bytes(tmp_path, cli):
    output = tmp_path / "split.json"
    first = cli(output)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["status"] == "created"
    before = output.read_bytes()
    stat = output.stat()
    second = cli(output)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["status"] == "verified_existing"
    assert output.stat().st_mtime_ns == stat.st_mtime_ns
    assert output.read_bytes() == before
    check = cli(output, "--check")
    assert check.returncode == 0, check.stderr
    record = json.loads(before)
    assert record["method"]["seed"] == 2027
    assert record["implementation"]["python_version"]
    assert record["implementation"]["source_sha256"]
    assert len(record["partitions"]["preexperiment"]["task_ids"]) == 20
    assert len(record["partitions"]["main"]["task_ids"]) == 230


def test_cli_refuses_changed_record_and_never_overwrites(tmp_path, cli):
    output = tmp_path / "split.json"
    assert cli(output).returncode == 0
    record = json.loads(output.read_bytes())
    record["partitions"]["preexperiment"]["task_ids"][0] = "UNKNOWN"
    output.write_text(json.dumps(record), encoding="utf-8")
    tampered = output.read_bytes()
    for args in [(), ("--check",)]:
        result = cli(output, *args)
        assert result.returncode != 0
        assert "file left unchanged" in result.stderr
        assert output.read_bytes() == tampered


def test_check_missing_record_is_read_only(tmp_path, cli):
    output = tmp_path / "absent.json"
    assert cli(output, "--check").returncode != 0
    assert not output.exists()


def test_changed_dataset_is_rejected_before_creating_split(tmp_path, cli):
    dataset = PROJECT.parent / "datasets/LinuxFLBench/dataset/LINUXFLBENCH_dataset.jsonl"
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(dataset.read_bytes() + b"\n")
    output = tmp_path / "split.json"
    result = cli(output, "--dataset", str(changed))
    assert result.returncode != 0
    assert "SHA-256" in result.stderr
    assert not output.exists()

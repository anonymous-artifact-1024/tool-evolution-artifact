from pathlib import Path
import subprocess
import sys
import json

import pytest

from saner_exp.data.kernel_versions import parse_checksums, parse_tag_refs, parse_version, version_order


@pytest.mark.parametrize("text,historical,tag", [
    ("2.4.32", True, "v2.4.32"), ("2.5.67", True, "v2.5.67"),
    ("2.6.11", True, "v2.6.11"), ("2.6.12", False, "v2.6.12"),
    ("2.6.18-rc3", False, "v2.6.18-rc3"), ("5.6.7", False, "v5.6.7"),
])
def test_exact_versions(text, historical, tag):
    version = parse_version(text)
    assert version.original == text
    assert version.historical_archive == historical
    assert version.tag == tag


@pytest.mark.parametrize("text", ["5.6.7-custom", "5.6-rc0", "5.6.7 ", "v5.6.7", "../5.6.7", ""])
def test_unsupported_versions_are_not_silently_normalized(text):
    with pytest.raises(ValueError):
        parse_version(text)


def test_release_candidate_order():
    assert sorted(["2.6.18", "2.6.18-rc3", "2.6.18-rc2", "2.6.9"], key=version_order) == [
        "2.6.9", "2.6.18-rc2", "2.6.18-rc3", "2.6.18",
    ]


def test_tag_objects_are_not_mislabeled_as_commits():
    refs = parse_tag_refs(f"{'a'*40}\trefs/tags/v5.6.7\n{'b'*40}\trefs/tags/v5.6.7^{{}}\n")
    assert refs == {"v5.6.7": {"ref_object_id": "a"*40, "peeled_object_id": "b"*40}}
    assert parse_tag_refs(f"{'c'*40}\trefs/tags/v1\n")["v1"] == {"ref_object_id": "c"*40}


@pytest.mark.parametrize("text", ["", "bad refs/tags/v1", f"{'a'*40}\trefs/tags/v1^{{}}",
    f"{'a'*40}\trefs/tags/v1\n{'b'*40}\trefs/tags/v1"])
def test_invalid_tag_advertisements_are_rejected(text):
    with pytest.raises(ValueError):
        parse_tag_refs(text)


def test_checksum_listing_with_pgp_wrapper():
    text = f"-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA256\n\n{'a'*64}  linux-2.5.67.tar.xz\n-----BEGIN PGP SIGNATURE-----\n"
    assert parse_checksums(text) == {"linux-2.5.67.tar.xz": "a"*64}
    with pytest.raises(ValueError):
        parse_checksums("not a checksum listing")
    with pytest.raises(ValueError):
        parse_checksums(text + f"{'b'*64}  linux-2.5.67.tar.xz\n")


def test_offline_inventory_has_complete_task_mapping(tmp_path):
    project = Path(__file__).resolve().parents[1]
    if not (project / "manifests/kernel_source_evidence/metadata.json").exists():
        pytest.skip("Public source metadata has not been fetched")
    output = tmp_path / "inventory.json"
    cmd = [sys.executable, str(project / "scripts/audit_kernel_versions.py"), "--output", str(output)]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_bytes())
    assert record["summary"]["unique_versions"] == 120
    assert len({e["kernel_version"] for e in record["versions"]}) == 120
    all_ids = [i for e in record["versions"] for i in e["task_ids"]]
    assert len(all_ids) == len(set(all_ids)) == 250
    assert sum(len(e["preexperiment_task_ids"]) for e in record["versions"]) == 20
    assert sum(len(e["main_task_ids"]) for e in record["versions"]) == 230
    for entry in record["versions"]:
        assert entry["benchmark_source_equivalence"] == "not_checked"
        assert entry["snapshot_status"] == "not_acquired"
    before = output.read_bytes()
    assert subprocess.run(cmd + ["--check"], capture_output=True).returncode == 0
    assert output.read_bytes() == before

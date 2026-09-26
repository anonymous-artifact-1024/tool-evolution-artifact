import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saner_exp.data.source_archive import ChunkedArchive, member_path, link_target, prepare_archive
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_prepared_sources import audit


class PathTests(unittest.TestCase):
    def test_normal_paths(self):
        self.assertEqual(str(member_path("./linux-3.0/kernel/file.c")), "linux-3.0/kernel/file.c")

    def test_unsafe_paths(self):
        for name in ("/linux-3.0/file", "linux-3.0/../escape", "linux-3.0//file", "linux-3.0/a\\b", "other/file"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                member_path(name)

    def test_link_scope(self):
        path = PurePosixPath("linux-3.0/a/link")
        self.assertEqual(str(link_target(path, "../file", hardlink=False)), "linux-3.0/file")
        for target in ("../../escape", "/etc/passwd", "../../linux-4.0/file"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                link_target(path, target, hardlink=False)


@unittest.skipUnless(os.name == "posix", "Extraction tests run inside the Linux container")
class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inventory = {"versions": [
            {"kernel_version": "3.0", "preexperiment_task_ids": ["1"], "main_task_ids": []},
            {"kernel_version": "4.0", "preexperiment_task_ids": [], "main_task_ids": ["2"]},
        ]}

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, extras=(), makefile=None):
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode="w:") as tar:
            files = [("linux-3.0/Makefile", makefile or b"VERSION = 3\nPATCHLEVEL = 0\nSUBLEVEL = 0\nEXTRAVERSION =\nNAME = Synthetic fixture\n"),
                     ("linux-3.0/a.c", b"original code\n"),
                     ("linux-4.0/Makefile", b"VERSION = 4\nPATCHLEVEL = 0\nSUBLEVEL = 0\nEXTRAVERSION =\nNAME = Synthetic fixture\n"),
                     ("linux-4.0/a.c", b"unselected\n"), *extras]
            for name, data in files:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o644
                tar.addfile(member, io.BytesIO(data))
        path = self.root / "source.tar.gz"
        path.write_bytes(gzip.compress(content.getvalue()))
        raw = path.read_bytes()
        self.receipt = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return path

    def prepare(self, archive):
        return prepare_archive(archive, self.receipt, self.inventory, self.root / "sources")

    def assert_no_publication(self):
        self.assertFalse(any((self.root / "sources").glob("author-*")))

    def test_selected_sources_and_complete_catalog(self):
        result = self.prepare(self.archive())
        root = Path(result["source_root"])
        self.assertEqual(result["snapshot_count"], 1)
        self.assertEqual(result["archive_version_count"], 2)
        self.assertTrue(result["gzip_crc_and_size_verified"])
        self.assertEqual((root / "versions/linux-3.0/a.c").read_bytes(), b"original code\n")
        self.assertFalse((root / "versions/linux-4.0").exists())
        self.assertTrue((root / "catalog.sqlite").exists())
        self.assertEqual(json.loads((root / "preparation.json").read_bytes())["archive_sha256"], self.receipt["sha256"])
        self.assertEqual(audit(root)["snapshot_count"], 1)

    def test_main_partition_selects_only_main_sources(self):
        result = prepare_archive(
            self.archive(), self.receipt, self.inventory, self.root / "main-sources",
            partition="main",
        )
        root = Path(result["source_root"])
        self.assertEqual(result["partition"], "main")
        self.assertEqual(result["snapshot_count"], 1)
        self.assertTrue((root / "versions/linux-4.0/a.c").is_file())
        self.assertFalse((root / "versions/linux-3.0").exists())

    def test_storage_audit_detects_changed_content(self):
        result = self.prepare(self.archive())
        root = Path(result["source_root"])
        (root / "versions/linux-3.0/a.c").write_bytes(b"modified code\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            audit(root)

    def test_storage_audit_detects_unexpected_file(self):
        result = self.prepare(self.archive())
        root = Path(result["source_root"])
        (root / "versions/linux-3.0/extra.c").write_bytes(b"extra\n")
        with self.assertRaisesRegex(ValueError, "path set"):
            audit(root)

    def test_checksummed_range_pipeline_matches_complete_archive(self):
        archive = self.archive()
        content = archive.read_bytes()
        parts = self.root / "parts"
        parts.mkdir()
        chunk_size = 71
        (parts / "identity.json").write_text(json.dumps({"size": len(content), "chunk_bytes": chunk_size}))
        for index, start in enumerate(range(0, len(content), chunk_size)):
            piece = content[start:start+chunk_size]
            (parts / f"{index:04d}.chunk").write_bytes(piece)
            (parts / f"{index:04d}.sha256").write_text(hashlib.sha256(piece).hexdigest())
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps(self.receipt))
        record = prepare_archive(ChunkedArchive(parts, len(content)), {"size_bytes": len(content), "sha256": None},
                                 self.inventory, self.root / "sources", completed_receipt_path=receipt_path)
        self.assertEqual(record["archive_sha256"], self.receipt["sha256"])
        self.assertTrue(record["complete_stream_scanned"])

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare(self.archive([("linux-3.0/a.c", b"replacement")]))
        self.assert_no_publication()

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(self.archive([("linux-3.0/../../escaped", b"bad")]))
        self.assertFalse((self.root / "escaped").exists())
        self.assert_no_publication()

    def test_full_hash_mismatch_rejected(self):
        archive = self.archive()
        self.receipt["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.prepare(archive)
        self.assert_no_publication()

    def test_gzip_corruption_rejected(self):
        archive = self.archive()
        raw = bytearray(archive.read_bytes())
        raw[-8] ^= 1
        archive.write_bytes(raw)
        self.receipt["sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(gzip.BadGzipFile):
            self.prepare(archive)
        self.assert_no_publication()

    def test_author_makefile_mismatch_is_recorded_without_relabeling_directory(self):
        archive = self.archive(makefile=b"VERSION = 4\nPATCHLEVEL = 0\nSUBLEVEL = 0\nEXTRAVERSION =\n")
        result = self.prepare(archive)
        self.assertEqual(result["snapshots"][0]["kernel_version"], "3.0")
        self.assertEqual(result["snapshots"][0]["makefile_kernel_version"], "4.0")
        self.assertEqual(result["version_identity_mismatches"], [{
            "directory_kernel_version": "3.0",
            "makefile_kernel_version": "4.0",
            "policy": "retain_verbatim_author_package_tree_and_record_mismatch",
        }])


if __name__ == "__main__":
    unittest.main()

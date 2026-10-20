"""Small download-integrity tests; no large downloads or optional packages."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "qhap_data_download", Path(__file__).resolve().parents[1] / "data/download.py")
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parts = []
        for index, content in enumerate((b"first", b"second"), 1):
            path = self.root / f"input.part{index:02d}"
            path.write_bytes(content)
            self.parts.append({
                "name": path.name, "url": path.as_uri(), "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()})
        self.record = {
            "file": "HiFi/chr22.bam", "bytes": 11,
            "sha256": hashlib.sha256(b"firstsecond").hexdigest(),
            "parts": self.parts}
        self.output = self.root / "output"

    def test_multipart_order_and_cleanup(self):
        target = DOWNLOAD.restore(self.record, self.output)
        self.assertEqual(target.read_bytes(), b"firstsecond")
        self.assertFalse(list((target.parent / ".downloads").glob("*.part*")))

    def test_verified_target_needs_no_network(self):
        target = DOWNLOAD.restore(self.record, self.output)
        with patch.object(DOWNLOAD.urllib.request, "urlopen",
                          side_effect=AssertionError("unexpected network")):
            self.assertEqual(DOWNLOAD.restore(self.record, self.output), target)

    def test_verified_parts_can_be_reused(self):
        target = DOWNLOAD.restore(self.record, self.output, keep_parts=True)
        target.unlink()
        with patch.object(DOWNLOAD.urllib.request, "urlopen",
                          side_effect=AssertionError("unexpected network")):
            DOWNLOAD.restore(self.record, self.output, keep_parts=True)
        self.assertEqual(target.read_bytes(), b"firstsecond")

    def test_bad_part_does_not_replace_existing_target(self):
        target = self.output / self.record["file"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"previous file")
        self.parts[0]["sha256"] = "0" * 64
        with patch.object(DOWNLOAD.time, "sleep"):
            with self.assertRaisesRegex(ValueError, "Checksum"):
                DOWNLOAD.restore(self.record, self.output)
        self.assertEqual(target.read_bytes(), b"previous file")

    def test_bad_complete_hash_does_not_publish_file(self):
        self.record["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Restored checksum"):
            DOWNLOAD.restore(self.record, self.output)
        self.assertFalse((self.output / self.record["file"]).exists())

    def test_single_part_is_not_truncated(self):
        part = self.parts[0]
        self.record.update(bytes=part["bytes"], sha256=part["sha256"], parts=[part])
        target = DOWNLOAD.restore(self.record, self.output)
        self.assertEqual(target.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()

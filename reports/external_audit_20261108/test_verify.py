"""Synthetic byte-only regression tests; never read project data or import it."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


AUDIT = "reports/external_audit_20261108"
SOURCE = Path(__file__).with_name("verify.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), "byte-only verifier must exist")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audit = self.root / AUDIT
        (self.audit / "inputs").mkdir(parents=True)
        shutil.copyfile(SOURCE, self.audit / "verify.py")
        (self.root / "frozen.txt").write_bytes(b"frozen bytes\n")
        (self.audit / "notes.txt").write_bytes(b"audit notes\n")
        self.snapshot = {"baseline_commit": "a" * 40,
                         "files": [self.entry("frozen.txt")]}
        self.delivery = {"files": [self.entry(AUDIT + "/notes.txt")]}
        self.pack = AUDIT + "/inputs/pack.tar"
        self.member = "objects/" + digest(b"synthetic input\n")
        self.index = {"packs": [], "objects": [{
            "sha256": digest(b"synthetic input\n"), "size": 16,
            "pack": self.pack, "member": self.member,
            "aliases": ["fixtures/input.bin"]}]}
        self.write_pack([(self.member, b"synthetic input\n", tarfile.REGTYPE)])

    def entry(self, path):
        data = (self.root / path).read_bytes()
        return {"path": path, "sha256": digest(data), "size": len(data)}

    def write_pack(self, members):
        with tarfile.open(self.root / self.pack, "w") as archive:
            for name, data, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(data) if kind == tarfile.REGTYPE else 0
                info.linkname = "../../outside" if kind == tarfile.SYMTYPE else ""
                archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
        self.index["packs"] = [self.entry(self.pack)]

    def run_verify(self, *args):
        for name, data in (("snapshot.json", self.snapshot),
                           ("delivery.json", self.delivery),
                           ("inputs/index.json", self.index)):
            (self.audit / name).write_text(json.dumps(data), encoding="utf-8")
        before = {p.relative_to(self.root): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        proc = subprocess.run([sys.executable, "-B", str(self.audit / "verify.py"),
                               *args], cwd=self.root, capture_output=True,
                              text=True, check=False)
        after = {p.relative_to(self.root): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after, "verification must not write files")
        self.assertEqual(proc.stderr, "")
        try:
            report = json.loads(proc.stdout)
        except ValueError:
            self.fail("verifier must return JSON: " + proc.stdout)
        return proc.returncode, report

    def assert_failed(self, needle):
        code, report = self.run_verify()
        self.assertNotEqual(code, 0)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn(needle.lower(), json.dumps(report).lower())

    def test_valid_archive_passes_without_writes(self):
        code, report = self.run_verify()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["objects"], 1)

    def test_one_byte_frozen_file_tamper_fails(self):
        (self.root / "frozen.txt").write_bytes(b"Frozen bytes\n")
        self.assert_failed("frozen.txt")

    def test_delivery_tamper_fails(self):
        (self.audit / "notes.txt").write_bytes(b"changed\n")
        self.assert_failed("notes.txt")

    def test_pack_tamper_fails(self):
        path = self.root / self.pack
        content = bytearray(path.read_bytes())
        content[520] ^= 1
        path.write_bytes(content)
        self.assert_failed("pack.tar")

    def test_unmaterialized_lfs_pointer_is_explicit(self):
        item = self.snapshot["files"][0]
        item["lfs_oid"] = item["sha256"]
        (self.root / item["path"]).write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + item["sha256"] + "\nsize 13\n")
        self.assert_failed("LFS pointer")

    def test_file_path_traversal_fails(self):
        self.snapshot["files"][0]["path"] = "../outside"
        self.assert_failed("unsafe path")

    def test_filesystem_symlink_escape_fails(self):
        (self.root / "frozen.txt").unlink()
        (self.root / "frozen.txt").symlink_to("/etc/hostname")
        self.assert_failed("unsafe path")

    def test_unsafe_tar_path_fails(self):
        self.index["objects"][0]["member"] = "../outside"
        self.write_pack([("../outside", b"synthetic input\n", tarfile.REGTYPE)])
        self.assert_failed("unsafe path")

    def test_tar_link_fails(self):
        self.write_pack([(self.member, b"", tarfile.SYMTYPE)])
        self.assert_failed("non-regular")

    def test_missing_member_fails(self):
        self.write_pack([])
        self.assert_failed("missing member")

    def test_unexpected_member_fails(self):
        self.write_pack([(self.member, b"synthetic input\n", tarfile.REGTYPE),
                         ("extra", b"extra", tarfile.REGTYPE)])
        self.assert_failed("unexpected member")

    def test_duplicate_member_fails(self):
        self.write_pack([(self.member, b"synthetic input\n", tarfile.REGTYPE)] * 2)
        self.assert_failed("duplicate member")

    def test_member_hash_checked_even_when_pack_hash_matches(self):
        self.write_pack([(self.member, b"Synthetic input\n", tarfile.REGTYPE)])
        self.assert_failed(self.member)

    def test_unknown_pack_mapping_fails(self):
        self.index["objects"][0]["pack"] = AUDIT + "/inputs/unknown.tar"
        self.assert_failed("unknown pack")

    def test_duplicate_object_mapping_fails(self):
        self.index["objects"] *= 2
        self.assert_failed("duplicate")

    def test_metadata_only_is_partial_with_missing_inputs(self):
        self.snapshot["files"][0]["lfs_oid"] = self.snapshot["files"][0]["sha256"]
        (self.root / "frozen.txt").unlink()
        (self.root / self.pack).unlink()
        code, report = self.run_verify("--metadata-only")
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PARTIAL")
        self.assertEqual(report["counts"]["objects"], 0)

    def test_partial_still_checks_delivery(self):
        (self.audit / "notes.txt").write_bytes(b"bad")
        code, report = self.run_verify("--metadata-only")
        self.assertNotEqual(code, 0)
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()

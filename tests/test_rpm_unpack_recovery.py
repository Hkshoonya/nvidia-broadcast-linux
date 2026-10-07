"""Recovery preserves known RPM fragments and refuses unrecognized contents."""

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rpm_unpack_recovery", ROOT / "packaging/native-prototype/recover_rpm_unpacked.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RpmUnpackRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rpm recovery ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "installed/runtime"
        self.runtime.mkdir(parents=True)
        (self.runtime.parent / ".transaction").write_text("prototype.cuda\n")
        self.payload = self.root / "verified-payload"
        self.payload.mkdir()
        self.quarantine = self.root / "recovery"
        self.tid = 1_791_388_800
        owner = mock.patch.object(recovery, "ROOT_OWNER", (os.getuid(), os.getgid()))
        owner.start()
        self.addCleanup(owner.stop)
        self.manifest = {}

    def file(self, name="lib/example.so", partial=b"known", content=b"known payload"):
        source = self.payload / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        self.manifest[name] = {"mode": 0o644, "size": len(content),
                               "sha256": hashlib.sha256(content).hexdigest()}
        fragment = self.runtime / (name + f";{self.tid:08x}")
        fragment.parent.mkdir(parents=True, exist_ok=True)
        fragment.write_bytes(partial)
        return source, fragment

    def recover(self):
        return recovery.recover(self.runtime, self.payload, self.manifest,
                                {"lib": {"mode": 0o755}}, self.tid - 1, self.tid + 1, self.quarantine)

    def test_partial_and_complete_payloads_are_preserved_with_audit_record(self):
        for name, data in (("lib/example.so", b"known"), ("lib/complete.so", b"known payload")):
            _, fragment = self.file(name, partial=data)
            fragment.chmod(0o600)
        report = self.recover()
        self.assertEqual(report["status"], "preserved")
        self.assertEqual(len(report["preserved"]), 2)
        for entry in report["preserved"]:
            saved = self.quarantine / "files" / entry["original"]
            self.assertEqual(recovery.sha(saved), entry["sha256"])
            self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
            self.assertFalse((self.runtime / entry["original"]).exists())
        self.assertTrue((self.runtime.parent / ".transaction").exists())

    def test_unknown_file_stops_the_entire_plan_and_preserves_every_file(self):
        _, fragment = self.file()
        unknown = self.runtime / "user-note"
        unknown.write_text("administrator data")
        with self.assertRaisesRegex(ValueError, "unrecognized file"):
            self.recover()
        self.assertEqual(fragment.read_bytes(), b"known")
        self.assertEqual(unknown.read_text(), "administrator data")
        self.assertFalse(self.quarantine.exists())

    def test_wrong_bytes_old_tid_and_unlisted_stem_are_rejected(self):
        _, fragment = self.file()
        for problem in ("wrong bytes", "old transaction", "unlisted file"):
            with self.subTest(problem=problem):
                actual = fragment
                if problem == "wrong bytes":
                    fragment.write_bytes(b"other")
                elif problem == "old transaction":
                    actual = fragment.with_name("example.so;00000001")
                    fragment.rename(actual)
                else:
                    actual = fragment.with_name(f"unlisted.so;{self.tid:08x}")
                    fragment.rename(actual)
                with self.assertRaises(ValueError):
                    self.recover()
                self.assertTrue(actual.exists())
                self.assertFalse(self.quarantine.exists())
                actual.rename(fragment)
                fragment.write_bytes(b"known")

    def test_verified_source_change_is_rejected(self):
        source, fragment = self.file()
        source.write_bytes(b"tampered source")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.recover()
        self.assertTrue(fragment.exists())

    def test_unexpected_owner_or_hardlinked_fragment_is_rejected(self):
        _, fragment = self.file()
        with mock.patch.object(recovery, "ROOT_OWNER", (os.getuid() + 1, os.getgid())):
            with self.assertRaisesRegex(ValueError, "owner or hardlinks"):
                self.recover()
        preserved = self.root / "administrator-link"
        os.link(fragment, preserved)
        with self.assertRaisesRegex(ValueError, "owner or hardlinks"):
            self.recover()
        self.assertEqual(preserved.read_bytes(), b"known")
        self.assertTrue(fragment.exists())

    def test_known_temporary_symlink_is_preserved_without_following_it(self):
        name = "lib/example.so"
        (self.payload / "lib").mkdir()
        (self.payload / name).symlink_to("target.so")
        self.manifest[name] = {"mode": 0o777, "symlink": "target.so"}
        (self.runtime / "lib").mkdir()
        fragment = self.runtime / (name + f";{self.tid:08x}")
        fragment.symlink_to("target.so")
        report = self.recover()
        self.assertEqual(os.readlink(self.quarantine / "files" / report["preserved"][0]["original"]), "target.so")

    def test_redirected_runtime_and_quarantine_are_rejected(self):
        _, fragment = self.file()
        link = self.root / "redirected"
        link.symlink_to(self.runtime, target_is_directory=True)
        for runtime, quarantine in ((link, self.quarantine), (self.runtime, link / "recovery"),
                                    (self.runtime, self.runtime / "lib/../recovery")):
            with self.subTest(runtime=runtime, quarantine=quarantine):
                with self.assertRaises(ValueError):
                    recovery.recover(runtime, self.payload, self.manifest, {"lib": {}},
                                     self.tid, self.tid, quarantine)
        self.assertTrue(fragment.exists())

    def test_marker_and_new_recovery_directory_are_required(self):
        _, fragment = self.file()
        marker = self.runtime.parent / ".transaction"
        marker.unlink()
        with self.assertRaisesRegex(ValueError, "marker required"):
            self.recover()
        marker.touch()
        self.quarantine.mkdir()
        (self.quarantine / "keep").write_text("earlier evidence")
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.recover()
        self.assertTrue(fragment.exists())
        self.assertEqual((self.quarantine / "keep").read_text(), "earlier evidence")

    def native_files(self):
        source_root = self.root / "native-source"
        manifest = {str(self.runtime): {"mode": 0o755}}
        fragments = []
        for original in (self.runtime.parent / "check-packages", self.root / "system-bin/nvbroadcast"):
            original.parent.mkdir(parents=True, exist_ok=True)
            source = source_root / str(original).lstrip("/")
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"known native adapter\n")
            manifest[str(original)] = {"mode": 0o755, "sha256": recovery.sha(source)}
            fragment = original.with_name(original.name + f";{self.tid:08x}")
            fragment.write_bytes(b"known native")
            fragments.append(fragment)
        return source_root, manifest, fragments

    def test_known_native_stems_are_preserved_and_unrelated_scripts_remain(self):
        source, manifest, fragments = self.native_files()
        unrelated = fragments[1].parent / f"user-script;{self.tid:08x}"
        unrelated.write_text("unrelated user file")
        report = recovery.recover(self.runtime, self.payload, {}, {}, self.tid, self.tid,
                                  self.quarantine, source, manifest, {})
        self.assertEqual(len(report["preserved"]), 2)
        for entry in report["preserved"]:
            self.assertEqual(entry["scope"], "native")
            saved = self.quarantine / "files" / entry["stored_as"]
            self.assertTrue(saved.is_relative_to(self.quarantine))
            self.assertEqual(saved.read_bytes(), b"known native")
            self.assertFalse(Path(entry["original"]).exists())
        self.assertEqual(unrelated.read_text(), "unrelated user file")

    def test_unknown_private_prefix_file_stops_native_preservation(self):
        source, manifest, fragments = self.native_files()
        unknown = self.runtime.parent / "administrator-note"
        unknown.write_text("preserve local data")
        with self.assertRaisesRegex(ValueError, "unrecognized file"):
            recovery.recover(self.runtime, self.payload, {}, {}, self.tid, self.tid,
                             self.quarantine, source, manifest, {})
        self.assertTrue(all(p.exists() for p in fragments))
        self.assertFalse(self.quarantine.exists())
        self.assertEqual(unknown.read_text(), "preserve local data")

    def test_native_source_tampering_and_manifest_path_escape_are_rejected(self):
        source, manifest, fragments = self.native_files()
        original = str(fragments[0]).rsplit(";", 1)[0]
        (source / original.lstrip("/")).write_bytes(b"modified adapter")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            recovery.recover(self.runtime, self.payload, {}, {}, self.tid, self.tid,
                             self.quarantine, source, manifest, {})
        manifest["/tmp/../escape"] = {"sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "manifest path"):
            recovery.recover(self.runtime, self.payload, {}, {}, self.tid, self.tid,
                             self.quarantine, source, manifest, {})
        self.assertTrue(all(p.exists() for p in fragments))


if __name__ == "__main__":
    unittest.main()

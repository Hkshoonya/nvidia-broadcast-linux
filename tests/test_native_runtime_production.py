"""Package payload identity and fail-closed production build boundaries."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "packaging/native-runtime"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load("native_production_build", "build.py")
validator = load("native_production_validator", "validate_install.py")


class NativeProductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.file = self.runtime / "module.py"
        self.file.write_bytes(b"package contents")
        self.file.chmod(0o644)
        self.manifest = {"module.py": {"mode": 0o644, "size": 16,
            "sha256": hashlib.sha256(b"package contents").hexdigest()}}
        self.manifest["module.py"]["size"] = self.file.stat().st_size

    def validate(self):
        validator.validate(self.runtime, self.manifest, owner=os.getuid())

    def test_exact_payload_passes_without_mutation(self):
        before = self.file.stat()
        self.validate()
        self.assertEqual(self.file.stat().st_mtime_ns, before.st_mtime_ns)

    def test_changed_bytes_missing_extra_and_permissions_are_rejected(self):
        self.file.write_bytes(b"tampered payload")
        with self.assertRaises(ValueError):
            self.validate()
        self.file.unlink()
        with self.assertRaisesRegex(ValueError, "file set"):
            self.validate()
        self.file.write_bytes(b"package contents")
        self.file.chmod(0o666)
        with self.assertRaisesRegex(ValueError, "permissions"):
            self.validate()
        self.file.chmod(0o644)
        (self.runtime / "unowned").touch()
        with self.assertRaisesRegex(ValueError, "file set"):
            self.validate()

    def test_same_byte_link_cannot_replace_regular_file(self):
        outside = self.root / "outside"
        self.file.rename(outside)
        self.file.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.validate()

    def test_redirected_runtime_root_is_rejected(self):
        link = self.root / "redirected"
        link.symlink_to(self.runtime, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "real directory"):
            validator.validate(link, self.manifest, owner=os.getuid())

    def test_unreviewed_builder_fails_before_invocation(self):
        pins = self.root / "builders.json"
        pins.write_text(json.dumps({"deb": {"image": "sha256:" + "a" * 64}}))
        with mock.patch.object(build.subprocess, "check_output") as invoke:
            with self.assertRaisesRegex(ValueError, "reviewed builder"):
                build.verified_images(pins, {"deb": "sha256:" + "b" * 64})
            invoke.assert_not_called()

    def test_native_hooks_have_no_destination_resolution(self):
        for family in ("deb", "rpm"):
            for action in ("prepare", "finish", "remove"):
                with self.subTest(family=family, action=action):
                    script = build.hook(action, "1.5.3-2.cpu", "cpu", family)
                    self.assertNotIn("@VERSION@", script)
                    self.assertNotIn("@VARIANT@", script)
                    self.assertNotIn("pip install", script)
                    self.assertNotIn("uv pip", script)
                    self.assertNotIn("curl ", script)
                    self.assertNotIn("wget ", script)
                    self.assertIn('"$base/check-packages" --configure', script)

    def test_production_locks_have_exactly_one_variant_owner(self):
        import tomllib
        for variant, owner in (("cpu", "onnxruntime"), ("cuda", "onnxruntime-gpu")):
            lock = tomllib.loads((HERE / f"pylock.linux-x86_64-cp313-{variant}.toml").read_text())
            packages = {p["name"] for p in lock["packages"]}
            self.assertEqual(packages & {"onnxruntime", "onnxruntime-gpu"}, {owner})
            self.assertNotIn("nvbroadcast", packages)


if __name__ == "__main__":
    unittest.main()

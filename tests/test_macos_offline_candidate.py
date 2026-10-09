"""Portable negative tests; actual macOS runtime is qualified by its own CI job."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "packaging/macos-runtime-prototype"
spec = importlib.util.spec_from_file_location("mac_offline_runtime", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
spec = importlib.util.spec_from_file_location("mac_offline_builder", HERE / "build.py")
builder = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"runtime": runtime}):
    spec.loader.exec_module(builder)
spec = importlib.util.spec_from_file_location("mac_offline_target", HERE / "wheel_target.py")
wheel_target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_target)


class MacOfflineCandidateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # macOS tempfile uses /var, whose system symlink points at /private/var.
        # Resolve the fixture itself; runtime symlink rejection stays exercised.
        self.root = Path(temporary.name).resolve()
        self.bundle = self.root / "payload"
        self.wheels = self.bundle / "wheels"
        self.wheels.mkdir(parents=True)
        for name in ("nvbroadcast", "onnxruntime", "faster-whisper", "pip", "setuptools", "wheel", "packaging"):
            self.add_wheel(name)
        self.manifest()
        self.runtime = self.root / "user runtime"

    def add_wheel(self, name, version="1.0"):
        normalized = name.replace("-", "_")
        path = self.wheels / f"{normalized}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{normalized}-{version}.dist-info/METADATA",
                             f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n")
        return path

    def manifest(self):
        packages = sorted((runtime.wheel_record(path) for path in self.wheels.iterdir()), key=lambda item: item["name"])
        manifest = {"schema_version": 1, "target": runtime.TARGET, "packages": packages}
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        (self.bundle / "requirements.txt").write_text(runtime.requirements(packages))
        (self.bundle / "macos-runtime-id").write_text(runtime.tree_digest(self.bundle) + "\n")

    def test_exact_bundle_is_accepted(self):
        result = runtime.validate_bundle(self.bundle)
        self.assertEqual(len(result["packages"]), 7)

    def test_vendored_dependency_metadata_is_not_a_second_wheel_owner(self):
        path = self.wheels / "pip-1.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("pip/_vendor/packaging-26.0.dist-info/METADATA",
                             "Metadata-Version: 2.3\nName: packaging\nVersion: 26.0\n")
        self.assertEqual(runtime.wheel_record(path)["name"], "pip")
        self.manifest()
        runtime.validate_bundle(self.bundle)

    def test_wheel_tampering_and_undeclared_wheel_are_rejected(self):
        path = next(self.wheels.iterdir())
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "bytes/metadata differ"):
            runtime.validate_bundle(self.bundle)
        path.write_bytes(original)
        self.add_wheel("unexpected")
        with self.assertRaisesRegex(ValueError, "bytes/metadata differ"):
            runtime.validate_bundle(self.bundle)

    def test_missing_wheel_is_not_resolved_from_network(self):
        next(self.wheels.iterdir()).unlink()
        with self.assertRaisesRegex(ValueError, "bytes/metadata differ"):
            runtime.validate_bundle(self.bundle)

    def test_injected_requirements_are_rejected(self):
        with (self.bundle / "requirements.txt").open("a") as stream:
            stream.write("--index-url https://invalid.example/\n")
        with self.assertRaisesRegex(ValueError, "requirements differ"):
            runtime.validate_bundle(self.bundle)

    def test_duplicate_and_cuda_owners_are_rejected(self):
        extra = self.add_wheel("onnxruntime", "2.0")
        self.manifest()
        with self.assertRaisesRegex(ValueError, "duplicate wheel owners"):
            runtime.validate_bundle(self.bundle)
        extra.unlink()
        self.add_wheel("onnxruntime-gpu")
        self.manifest()
        with self.assertRaisesRegex(ValueError, "CPU ONNX"):
            runtime.validate_bundle(self.bundle)

    def test_redirected_wheel_and_payload_are_rejected(self):
        path = next(self.wheels.iterdir())
        external = self.root / path.name
        path.rename(external)
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "regular wheels"):
            runtime.validate_bundle(self.bundle)
        with self.assertRaisesRegex(ValueError, "Redirected package"):
            runtime.tree_digest(self.bundle)

    def test_root_never_probes_homebrew(self):
        with mock.patch.object(runtime.os, "geteuid", return_value=0), mock.patch.object(runtime, "native_probe") as probe:
            with self.assertRaisesRegex(RuntimeError, "without sudo"):
                runtime.install(self.bundle, self.runtime)
        probe.assert_not_called()
        self.assertFalse(self.runtime.exists())

    def test_failed_wheel_install_never_marks_ready_or_touches_old_runtime(self):
        previous = self.root / "previous-runtime"
        previous.mkdir()
        (previous / "runtime-ready").write_text("previous successful runtime")
        with mock.patch.object(runtime.os, "geteuid", return_value=501), \
                mock.patch.object(runtime, "native_probe", return_value={}), \
                mock.patch.object(runtime, "run"), \
                mock.patch.object(runtime, "pip_install", side_effect=RuntimeError("install failed")):
            with self.assertRaisesRegex(RuntimeError, "install failed"):
                runtime.install(self.bundle, self.runtime)
        self.assertFalse((self.runtime / "runtime-ready").exists())
        self.assertEqual((previous / "runtime-ready").read_text(), "previous successful runtime")

    def test_incomplete_runtime_is_not_replaced(self):
        self.runtime.mkdir()
        (self.runtime / "keep.txt").write_text("inspect this failure")
        with mock.patch.object(runtime.os, "geteuid", return_value=501), \
                mock.patch.object(runtime, "native_probe", return_value={}), \
                mock.patch.object(runtime, "pip_install") as install:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                runtime.install(self.bundle, self.runtime)
        install.assert_not_called()
        self.assertEqual((self.runtime / "keep.txt").read_text(), "inspect this failure")

    def test_changed_payload_is_rejected_before_environment_creation(self):
        (self.bundle / "extra-script.py").write_text("changed payload")
        with mock.patch.object(runtime.os, "geteuid", return_value=501), \
                mock.patch.object(runtime, "native_probe", return_value={}):
            with self.assertRaisesRegex(ValueError, "identity"):
                runtime.install(self.bundle, self.runtime)
        self.assertFalse(self.runtime.exists())

    def test_redirected_user_runtime_is_rejected_without_touching_target(self):
        target = self.root / "separate-target"
        target.mkdir()
        self.runtime.symlink_to(target, target_is_directory=True)
        with mock.patch.object(runtime.os, "geteuid", return_value=501), \
                mock.patch.object(runtime, "native_probe", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "redirected user runtime"):
                runtime.install(self.bundle, self.runtime)
        self.assertEqual(list(target.iterdir()), [])

    def test_pip_is_offline_hash_checked_and_does_not_resolve(self):
        with mock.patch.object(runtime, "run") as run:
            runtime.pip_install(Path("/user/.venv/bin/python"), self.bundle, self.bundle / "requirements.txt")
        arguments = run.call_args.args
        for flag in ("--isolated", "--no-index", "--no-deps", "--require-hashes", "--only-binary=:all:",
                     "--ignore-installed", "--disable-pip-version-check", "--no-cache-dir"):
            self.assertIn(flag, arguments)
        self.assertNotIn("--upgrade", arguments)

    def test_candidate_preinstall_reuses_native_privilege_policy(self):
        script = builder.root_preinstall()
        for phrase in ("admin-owned", "ACL requires administrator review", "Refusing symlinked",
                       "running system volume", "Apple Silicon", "macOS 13"):
            self.assertIn(phrase, script)
        self.assertIn("/usr/bin/find -P /opt/nvbroadcast-offline-candidate -print", script)
        self.assertNotIn("/usr/local/bin/nvbroadcast;", script)
        self.assertNotIn("/opt/nvbroadcast/", script)
        result = subprocess.run(["/bin/bash", "-n"], input=script, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checked_in_supplier_hashes_have_valid_length(self):
        pins = json.loads((HERE / "inputs.json").read_text())
        for item in [pins["uv"], *pins["bootstrap"]]:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(item["url"].startswith("https://"))

    def test_target_wheel_selection_cannot_raise_deployment_floor_to_runner_os(self):
        old = "sample-1.0-cp313-cp313-macosx_13_0_arm64.whl"
        newer = "sample-1.0-cp313-cp313-macosx_15_0_arm64.whl"
        intel = "sample-1.0-cp313-cp313-macosx_13_0_x86_64.whl"
        universal = "sample-1.0-cp312-abi3-macosx_11_0_universal2.whl"
        other_abi = "sample-1.0-cp314-cp314-macosx_13_0_arm64.whl"
        ranks = wheel_target.rank_wheels([old, newer, intel, universal, other_abi])
        self.assertIsNotNone(ranks[old])
        self.assertIsNone(ranks[newer])
        self.assertIsNone(ranks[intel])
        self.assertIsNone(ranks[other_abi])
        self.assertIsNotNone(ranks[universal])
        self.assertLess(ranks[old], ranks[universal])


if __name__ == "__main__":
    unittest.main()

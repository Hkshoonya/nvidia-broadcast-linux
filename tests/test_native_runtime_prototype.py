"""Input integrity and launcher state boundaries for the native prototype.

Actual APT/DNF transactions run separately in disposable offline containers.
"""

import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load("native_prototype_build", ROOT / "packaging/native-prototype/build.py")
lifecycle = load("native_prototype_lifecycle", ROOT / "packaging/native-prototype/run_lifecycle.py")
verifier = load("native_prototype_verifier", ROOT / "packaging/native-prototype/verify.py")


class NativeRuntimePrototypeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def application(self):
        site = self.root / "lib/python3.13/site-packages"
        app = site / "nvbroadcast/__init__.py"
        app.parent.mkdir(parents=True)
        app.write_text("VERSION = '1.5.2'\n")
        record = site / "nvbroadcast-1.5.2.dist-info/RECORD"
        record.parent.mkdir()
        hashed = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(app.read_bytes()).digest()).rstrip(b"=").decode()
        with record.open("w", newline="") as stream:
            csv.writer(stream).writerows([["nvbroadcast/__init__.py", hashed, str(app.stat().st_size)],
                                          ["nvbroadcast-1.5.2.dist-info/RECORD", "", ""]])
        return app, record

    def test_record_partition_is_disjoint_and_exhaustive_for_files(self):
        app, record = self.application()
        dependency = app.parent.parent / "dependency.py"
        dependency.write_text("third party\n")
        manifest = build.inventory(self.root)
        selected = build.app_files(self.root, manifest)
        self.assertEqual(selected, {str(p.relative_to(self.root)) for p in (app, record)})
        self.assertNotIn(str(dependency.relative_to(self.root)), selected)

    def test_record_tampering_and_missing_app_files_are_rejected(self):
        app, record = self.application()
        app.write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
            build.app_files(self.root, build.inventory(self.root))
        app.write_text("VERSION = '1.5.2'\n")
        (app.parent / "untracked.py").write_text("untracked\n")
        with self.assertRaisesRegex(ValueError, "missing from RECORD"):
            build.app_files(self.root, build.inventory(self.root))

    def test_record_escape_and_duplicate_are_rejected(self):
        app, record = self.application()
        original = record.read_text()
        for row in (original.splitlines()[0], "/etc/passwd,,", "../../../../../../etc/passwd,,"):
            with self.subTest(row=row):
                record.write_text(original + row + "\n")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    build.app_files(self.root, build.inventory(self.root))

    def package_set(self):
        records = []
        for family in ("deb", "rpm"):
            for kind in ("self", "app", "runtime"):
                for revision in (1, 2):
                    path = self.root / f"{family}-{kind}-{revision}"
                    path.write_text(f"{family} {kind} {revision}\n")
                    records.append({"family": family, "kind": kind, "revision": revision,
                                    "artifact": path.name, "sha256": lifecycle.digest(path)})
        (self.root / "packages.json").write_text(json.dumps(records))
        return records

    def test_exact_package_set_is_verified_before_running_docker(self):
        records = self.package_set()
        self.assertEqual(lifecycle.checked_packages(self.root), records)
        (self.root / records[0]["artifact"]).write_text("corrupted archive")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            lifecycle.checked_packages(self.root)

    def test_regular_native_file_rejects_same_byte_symlink_substitution(self):
        prefix = self.root / "private/runtime"
        prefix.mkdir(parents=True)
        launcher = self.root / "system-bin/nvbroadcast"
        launcher.parent.mkdir()
        launcher.write_bytes(b"#!/bin/sh\nexit 0\n")
        launcher.chmod(0o755)
        target = self.root / "unowned-target"
        target.write_bytes(launcher.read_bytes())
        target.chmod(0o755)
        contents = {str(prefix.parent): {"mode": 0o755}, str(prefix): {"mode": 0o755},
                    str(launcher): {"mode": 0o755, "sha256": verifier.sha(launcher)}}
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        (artifacts / "content.json").write_text(json.dumps(contents))
        (artifacts / "packages.json").write_text(json.dumps([
            {"family": "deb", "kind": "self", "revision": 1, "name": "nvbroadcast-cpu",
             "content": "content.json"}]))
        manifest = self.root / "manifest.json"
        manifest.write_text("{}")
        original_lstat, original_exists = Path.lstat, Path.exists

        def root_metadata(path):
            actual = original_lstat(path)
            return SimpleNamespace(st_mode=actual.st_mode, st_uid=0, st_gid=0)

        def fixture_exists(path):
            return False if path == Path("/usr/lib/nvbroadcast/.transaction") else original_exists(path)

        # Only package database/process calls and fixture UID/GID are simulated;
        # both checks inspect real file/link types, bytes and permissions.
        with mock.patch.object(verifier, "PREFIX", prefix), \
                mock.patch.object(verifier.subprocess, "check_output", return_value="\n".join(contents)), \
                mock.patch.object(verifier.subprocess, "run"), \
                mock.patch.object(Path, "lstat", root_metadata), \
                mock.patch.object(Path, "exists", fixture_exists):
            self.assertEqual(verifier.verify("deb", "self", 1, artifacts, manifest)["status"], "pass")
            launcher.unlink()
            launcher.symlink_to(target)
            self.assertEqual(verifier.sha(launcher), contents[str(launcher)]["sha256"])
            with self.assertRaisesRegex(AssertionError, "expected regular file"):
                verifier.verify("deb", "self", 1, artifacts, manifest)

    def test_package_identity_duplicate_missing_and_escape_are_rejected(self):
        records = self.package_set()
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_text("outside")
        self.addCleanup(outside.unlink)
        cases = [records[:-1], records + [records[0]],
                 [{**records[0], "artifact": str(outside), "sha256": lifecycle.digest(outside)}, *records[1:]]]
        for changed in cases:
            with self.subTest(changed=changed[0]):
                (self.root / "packages.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    lifecycle.checked_packages(self.root)

    def check_versions(self, family, kind, app, runtime, configuring=False):
        fake = self.root / ("dpkg-query" if family == "deb" else "rpm")
        fake.write_text('#!/bin/sh\ncase "$*" in *nvbroadcast-runtime-cpu*) printf "%s" "$TEST_RUNTIME";; '
                        '*) printf "%s" "$TEST_APP";; esac\n')
        fake.chmod(0o755)
        version = build.versions(family, 1)
        script = self.root / "check-packages"
        script.write_text(build.version_check(family, kind, version))
        script.chmod(0o755)
        env = {**os.environ, "PATH": str(self.root) + os.pathsep + os.environ.get("PATH", ""),
               "TEST_APP": app, "TEST_RUNTIME": runtime}
        return subprocess.run([str(script), *(["--configure"] if configuring else [])], env=env,
                              capture_output=True, text=True).returncode

    def test_split_launcher_rejects_mismatched_runtime_even_if_app_is_current(self):
        for family in ("deb", "rpm"):
            with self.subTest(family=family):
                prefix = "install ok installed " if family == "deb" else ""
                correct = prefix + build.versions(family, 1)
                wrong = prefix + build.versions(family, 2)
                self.assertEqual(self.check_versions(family, "app", correct, correct), 0)
                self.assertEqual(self.check_versions(family, "app", correct, wrong), 78)
                self.assertEqual(self.check_versions(family, "app", correct, ""), 78)

    def test_dpkg_configure_exception_only_accepts_current_application(self):
        version = build.versions("deb", 1)
        installed = "install ok installed " + version
        pending = "install ok half-configured " + version
        self.assertEqual(self.check_versions("deb", "app", pending, installed), 78)
        self.assertEqual(self.check_versions("deb", "app", pending, installed, configuring=True), 0)
        self.assertEqual(self.check_versions("deb", "app", installed, pending, configuring=True), 78)

    def test_legacy_cleanup_preserves_redirected_prefix_and_adjacent_files(self):
        base = self.root / "new-prefix"
        base.mkdir()
        check = base / "check-packages"
        check.write_text("#!/bin/sh\nexit 0\n")
        check.chmod(0o755)
        external = self.root / "administrator-data"
        (external / ".venv").mkdir(parents=True)
        sentinel = external / ".venv/keep"
        sentinel.write_text("preserve redirected data")
        legacy = self.root / "legacy"
        legacy.symlink_to(external, target_is_directory=True)
        script = build.hooks("deb", "app", build.versions("deb", 1), "finish")
        # Relocate the actual hook into an ordinary-user temporary fixture.
        # No production paths can be touched by this test.
        script = script.replace("/usr/lib/nvbroadcast", str(base)).replace("/opt/nvbroadcast", str(legacy))
        self.assertNotIn("/opt/nvbroadcast", script)
        self.assertNotIn("/usr/lib/nvbroadcast", script)
        for redirected in (True, False):
            with self.subTest(redirected=redirected):
                (base / ".legacy-runtime").touch()
                (base / ".transaction").touch()
                if not redirected:
                    legacy.unlink()
                    (legacy / ".venv").mkdir(parents=True)
                    (legacy / ".venv/old-runtime").write_text("remove obsolete runtime")
                    (legacy / "keep-adjacent").write_text("unrelated administrator file")
                subprocess.run(["sh", "-c", script], check=True)
                self.assertEqual(sentinel.read_text(), "preserve redirected data")
                self.assertFalse((base / ".transaction").exists())
                if not redirected:
                    self.assertFalse((legacy / ".venv").exists())
                    self.assertTrue((legacy / "keep-adjacent").exists())


if __name__ == "__main__":
    unittest.main()

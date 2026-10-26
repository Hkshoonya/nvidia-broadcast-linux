"""Package payload identity and fail-closed production build boundaries."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from importlib import metadata


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "packaging/native-runtime"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load("native_production_build", "build.py")
validator = load("native_production_validator", "validate_install.py")
qualifier = load("native_production_qualifier", "qualify.py")


class CandidateInterpreterTests(unittest.TestCase):
    def test_each_candidate_uses_its_authenticated_interpreter_pin(self):
        for version in ("3.13.16", "3.13.18"):
            data = json.dumps({"python": {"version": version}}).encode()
            package = {"adapter": {"revision": "a" * 40, "files": {
                "packaging/runtime-prototype/inputs.json": hashlib.sha256(data).hexdigest()}}}
            with self.subTest(version=version), mock.patch.object(qualifier.subprocess, "check_output", return_value=data):
                self.assertEqual(qualifier.expected_python_version(package), version)
                package["adapter"]["files"]["packaging/runtime-prototype/inputs.json"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "differs from recorded adapter"):
                    qualifier.expected_python_version(package)


class NativeArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        (self.runtime / "module.py").write_text("fixture payload\n")
        self.source = {"revision": "a" * 40, "version": "2.7.11", "source_date_epoch": 1700000000}
        provenance = self.runtime / "share/nvbroadcast-runtime-provenance/inputs.json"
        provenance.parent.mkdir(parents=True)
        provenance.write_text(json.dumps({"application": self.source}))
        (self.root / "manifest.json").write_text(json.dumps(build.inventory(self.runtime)))
        # These tests exercise archive construction, not application installation.
        # Keep real inventory, staging, hooks, recipe rendering and size checks.
        for target, name in ((build.native, "verify_owner"), (build.native, "app_files"),
                             (build, "verify_application_source")):
            patch = mock.patch.object(target, name)
            patch.start()
            self.addCleanup(patch.stop)

    def arguments(self, variant):
        return SimpleNamespace(variant=variant, package_revision="17", adapter={"revision": "b" * 40},
            deb_image="sha256:" + "c" * 64, rpm_image="sha256:" + "d" * 64)

    def archive(self, output, family, variant):
        version = f"2.7.11-17.{variant}"
        relative = (f"deb/nvbroadcast_{version}_amd64.deb" if family == "deb" else
                    f"rpm/build/RPMS/x86_64/nvbroadcast-{version}.x86_64.rpm")
        return output / relative

    def builder(self, archive, size=23):
        def invoke(image, mounts, command, log, epoch):
            archive.parent.mkdir(parents=True, exist_ok=True)
            with archive.open("wb") as stream:
                stream.truncate(size)
            log.write_text("fixture builder completed\n")
        return invoke

    def test_release_asset_limit_is_exclusive_without_reading_gigabytes(self):
        archive = self.root / "boundary.deb"
        # Sparse files exercise real filesystem sizes without allocating or
        # reading a multi-gigabyte archive. The public limit is exactly 2 GiB.
        limit = 2 * 1024**3
        for size in (1, limit - 1, limit, limit + 1, 0):
            with self.subTest(size=size):
                with archive.open("wb") as stream:
                    stream.truncate(size)
                if 0 < size < limit:
                    self.assertEqual(build.release_asset_bytes(archive), size)
                else:
                    with self.assertRaisesRegex(ValueError, "smaller than 2147483648 bytes"):
                        build.release_asset_bytes(archive)

    def test_missing_directory_symlink_and_fifo_are_not_release_assets(self):
        with self.assertRaises(FileNotFoundError):
            build.release_asset_bytes(self.root / "missing")
        target = self.root / "archive.deb"
        target.write_bytes(b"nonempty")
        link = self.root / "link.deb"
        link.symlink_to(target)
        fifo = self.root / "pipe.deb"
        os.mkfifo(fifo)
        for path in (self.runtime, link, fifo):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                build.release_asset_bytes(path)

    def test_wrappers_use_variant_compression_and_bind_actual_archive_bytes(self):
        for family in ("deb", "rpm"):
            for variant, level in (("cpu", 8), ("cuda", 10)):
                with self.subTest(family=family, variant=variant):
                    output = self.root / f"{family}-{variant}"
                    archive = self.archive(output, family, variant)
                    args = self.arguments(variant)
                    with mock.patch.object(build, "docker", side_effect=self.builder(archive)) as invoke:
                        record = build.wrap(args, self.runtime, output, self.source, family)
                    command = invoke.call_args.args[2]
                    if family == "deb":
                        self.assertEqual([arg for arg in command if arg.startswith("-z")], [f"-z{level}"])
                        self.assertIn("-Zzstd", command)
                    else:
                        spec = (output / "rpm/package.spec").read_text()
                        payload = [line for line in spec.splitlines() if line.startswith("%global _binary_payload ")]
                        self.assertEqual(payload, [f"%global _binary_payload w{level}.zstdio"])
                    self.assertEqual(record["bytes"], 23)
                    self.assertEqual(record["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
                    self.assertEqual(record["source"], self.source)
                    self.assertEqual(record["adapter"], args.adapter)
                    self.assertEqual(record["variant"], variant)
                    self.assertEqual(record["version"], f"2.7.11-17.{variant}")

    def test_oversized_wrappers_fail_before_hashing_or_accepting_archive(self):
        digest = build.prepare.digest
        for family in ("deb", "rpm"):
            for variant in ("cpu", "cuda"):
                with self.subTest(family=family, variant=variant):
                    output = self.root / f"oversized-{family}-{variant}"
                    archive = self.archive(output, family, variant)
                    def guarded_digest(path):
                        self.assertNotEqual(path, archive, "oversized archive must be rejected before hashing")
                        return digest(path)
                    with mock.patch.object(build, "docker", side_effect=self.builder(archive, 2 * 1024**3)), \
                            mock.patch.object(build.prepare, "digest", side_effect=guarded_digest):
                        with self.assertRaisesRegex(ValueError, "native release asset"):
                            build.wrap(self.arguments(variant), self.runtime, output, self.source, family)
                    self.assertTrue(archive.is_file())
                    self.assertEqual((output / family / "build.log").read_text(), "fixture builder completed\n")

    def test_failed_archive_is_not_written_to_packages_manifest(self):
        output = self.root / "failed-build"
        args = self.arguments("cuda")
        archive = self.archive(output, "deb", "cuda")
        argv = ["build.py", "--variant", "cuda", "--package-revision", "17", "--families", "deb",
                "--cache", str(self.root / "cache"), "--output", str(output),
                "--bindings-image", "sha256:" + "e" * 64,
                "--deb-image", args.deb_image, "--rpm-image", args.rpm_image]
        old_umask = os.umask(0o022)
        self.addCleanup(os.umask, old_umask)
        with mock.patch.object(build.sys, "argv", argv), \
                mock.patch.object(build, "source_identity", return_value=self.source), \
                mock.patch.object(build, "recipe_identity", return_value=args.adapter), \
                mock.patch.object(build, "verified_images"), \
                mock.patch.object(build, "assemble", return_value=self.runtime), \
                mock.patch.object(build.os, "getuid", return_value=1000), \
                mock.patch.object(build, "docker", side_effect=self.builder(archive, 2 * 1024**3)):
            with self.assertRaisesRegex(ValueError, "native release asset"):
                build.main()
        self.assertFalse((output / "packages.json").exists())
        self.assertTrue((output / "build-provenance.json").is_file())
        self.assertTrue(archive.is_file())

    def test_uncommitted_recipe_is_rejected_before_identity_is_recorded(self):
        with mock.patch.object(build.subprocess, "check_output",
                return_value=" M packaging/native-runtime/build.py\n"), \
                mock.patch.object(build.prepare, "digest") as digest:
            with self.assertRaisesRegex(ValueError, "commit the executable adapter"):
                build.recipe_identity()
            digest.assert_not_called()

    def test_recipe_identity_binds_compression_and_size_policy(self):
        with mock.patch.object(build.subprocess, "check_output", side_effect=["", "a" * 40, "b" * 40]):
            identity = build.recipe_identity()
        self.assertEqual(identity["files"]["packaging/native-runtime/build.py"],
                         hashlib.sha256((HERE / "build.py").read_bytes()).hexdigest())


class NativeProductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.runtime.chmod(0o755)
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

    def test_writable_or_wrong_owner_runtime_root_is_rejected(self):
        self.runtime.chmod(0o777)
        with self.assertRaisesRegex(ValueError, "runtime root"):
            self.validate()
        self.runtime.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "runtime root"):
            validator.validate(self.runtime, self.manifest, owner=os.getuid() + 1)

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

    def test_rpm_recovery_is_standalone_and_binds_the_reviewed_public_key(self):
        stage = self.root / "stage"
        build.recovery_interface(stage)
        helper = stage / "usr/bin/nvbroadcast-recover-rpm"
        self.assertEqual(helper.stat().st_mode & 0o777, 0o755)
        self.assertTrue(helper.read_text().startswith("#!/usr/bin/python3 -IB\n"))
        namespace = {"__name__": "packaged_recovery"}
        exec(compile(helper.read_bytes(), str(helper), "exec"), namespace)
        self.assertEqual(hashlib.sha256(namespace["PUBLIC_ARMOR"]).hexdigest(), namespace["PUBLIC_SHA256"])
        self.assertEqual(namespace["PUBLIC_ARMOR"], (ROOT / "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc").read_bytes())
        self.assertFalse((stage / "usr/lib/nvbroadcast/runtime").exists())

    def test_production_locks_have_exactly_one_variant_owner(self):
        import tomllib
        for variant, owner in (("cpu", "onnxruntime"), ("cuda", "onnxruntime-gpu")):
            lock = tomllib.loads((HERE / f"pylock.linux-x86_64-cp313-{variant}.toml").read_text())
            packages = {p["name"] for p in lock["packages"]}
            self.assertEqual(packages & {"onnxruntime", "onnxruntime-gpu"}, {owner})
            self.assertNotIn("nvbroadcast", packages)

    def environment(self, packages):
        from nvbroadcast.runtime.artifact import ArtifactEnvironment
        from packaging.markers import default_environment
        distributions = []
        installed = {}
        for name, version, requirements in packages:
            directory = self.root / f"{name}-{version}.dist-info"
            directory.mkdir()
            data = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            data += "".join("Requires-Dist: " + raw + "\n" for raw in requirements)
            (directory / "METADATA").write_text(data)
            distributions.append(metadata.PathDistribution(directory))
            installed[name] = (version,)
        return ArtifactEnvironment((), tuple(distributions), installed, default_environment())

    def test_installed_extra_version_bounds_are_enforced(self):
        environment = self.environment([
            ("nvbroadcast", "1.5.3", ['onnxruntime>=1.24.4; extra == "cpu"']),
            ("faster-whisper", "1.2.1", []), ("onnxruntime", "1.23.0", [])])
        with self.assertRaisesRegex(ValueError, "onnxruntime>=1.24.4"):
            validator.validate_dependencies(environment, "cpu")

    def test_cuda_cannot_substitute_an_unreviewed_selected_extra(self):
        environment = self.environment([
            ("nvbroadcast", "1.5.3", ['onnxruntime>=1.24.4; extra == "cuda"']),
            ("faster-whisper", "1.2.1", []), ("onnxruntime-gpu", "1.24.4", [])])
        with self.assertRaisesRegex(ValueError, "unreviewed CUDA"):
            validator.validate_dependencies(environment, "cuda")

    def test_missing_managed_meeting_backend_is_rejected(self):
        environment = self.environment([("nvbroadcast", "1.5.3", []), ("onnxruntime", "1.24.4", [])])
        with self.assertRaisesRegex(ValueError, "managed faster-whisper"):
            validator.validate_dependencies(environment, "cpu")


if __name__ == "__main__":
    unittest.main()

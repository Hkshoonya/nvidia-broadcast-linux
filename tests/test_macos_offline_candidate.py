"""Portable negative tests; actual macOS runtime is qualified by its own CI job."""

import copy
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import struct
import sys
import tempfile
import tarfile
import tomllib
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
spec = importlib.util.spec_from_file_location("mac_binding_prepare", HERE / "prepare_bindings.py")
binding_prepare = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"runtime": runtime}):
    spec.loader.exec_module(binding_prepare)
spec = importlib.util.spec_from_file_location("mac_offline_target", HERE / "wheel_target.py")
wheel_target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_target)

spec = importlib.util.spec_from_file_location("mac_supplier_lock", HERE / "supplier_lock.py")
supplier = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"wheel_target": wheel_target, "runtime": runtime}):
    spec.loader.exec_module(supplier)


spec = importlib.util.spec_from_file_location("mac_media_validator", ROOT / "scripts/validate_macos_runtime.py")
media_validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media_validator)


def native_fixture(version=(14, 0, 0), cpu=0x0100000c):
    """A Mach-O header fixture, never an executable native extension."""
    floor = (version[0] << 16) | (version[1] << 8) | version[2]
    return struct.pack("<8I", 0xfeedfacf, cpu, 0, 8, 1, 24, 0, 0) + struct.pack("<6I", 0x32, 24, 1, floor, floor, 0)


def write_binding_fixture(wheels, source, *, tag="cp313-cp313-macosx_14_0_arm64", floor=(14, 0, 0)):
    name = runtime.canonical(source["name"])
    version = source["version"]
    package = "cairo" if name == "pycairo" else "gi"
    info = f"{name}-{version}.dist-info"
    licence = (f"{name}-{version}/COPYING-LGPL-2.1" if name == "pycairo"
               else f"{name}-{version}/LICENSES/LGPL-2.1-or-later.txt")
    notice = b"Synthetic licence text for portable tests only.\n"
    licence_file = info + "/licenses/nvbroadcast-source/" + licence
    source_licenses = {licence: {"file": licence_file, "sha256": hashlib.sha256(notice).hexdigest()}}
    metadata = (f"Metadata-Version: 2.3\nName: {source['name']}\nVersion: {version}\nRequires-Python: >=3.9\n"
                + ("Requires-Dist: pycairo>=1.16.0\n" if name == "pygobject" else ""))
    payloads = {f"{package}/__init__.py": b"# synthetic package; not imported\n",
                f"{info}/METADATA": metadata.encode(),
                f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n".encode(),
                licence_file: notice}
    for module in (("_cairo",) if name == "pycairo" else ("_gi", "_gi_cairo")):
        payloads[f"{package}/{module}.cpython-313-darwin.so"] = native_fixture(floor)
    payloads[f"{info}/RECORD"] = b""  # Preparation regenerates this record when retaining source licences.
    path = wheels / f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for relative, payload in payloads.items():
            archive.writestr(relative, payload)
    return path, source_licenses


def write_binding_inputs(bundle):
    source_path = ROOT / "packaging/macos-runtime/inputs.json"
    (bundle / runtime.BINDING_INPUT_FILE).write_bytes(source_path.read_bytes())
    sources = json.loads(source_path.read_text())
    records = []
    for source in sources["bindings"]:
        path, licenses = write_binding_fixture(bundle / "wheels", source)
        records.append(runtime.validate_binding_wheel(path, source, licenses))
    (bundle / runtime.BINDING_INVENTORY_FILE).write_text(json.dumps({"scope": "synthetic portable fixture"}))
    receipt = {"schema_version": 1, "target": runtime.TARGET, "source_revision": "a" * 40,
               "source_inputs": {"file": runtime.BINDING_INPUT_FILE, "sha256": runtime.digest(bundle / runtime.BINDING_INPUT_FILE)},
               "build_inventory": {"file": runtime.BINDING_INVENTORY_FILE, "sha256": runtime.digest(bundle / runtime.BINDING_INVENTORY_FILE)},
               "wheels": sorted(records, key=lambda item: item["name"])}
    path = bundle / runtime.BINDING_MANIFEST_FILE
    path.write_text(json.dumps(receipt))
    return receipt, runtime.digest(path)


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
        self.bindings, self.binding_sha256 = write_binding_inputs(self.bundle)
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
        manifest = {"schema_version": 2, "target": runtime.TARGET,
                    "minimum_macos": runtime.MINIMUM_MACOS, "packages": packages,
                    "source_revision": self.bindings["source_revision"],
                    "native_bindings_manifest_sha256": self.binding_sha256, "native_bindings": self.bindings}
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        (self.bundle / "requirements.txt").write_text(runtime.requirements(packages))
        (self.bundle / "macos-runtime-id").write_text(runtime.tree_digest(self.bundle) + "\n")

    def test_exact_bundle_is_accepted(self):
        result = runtime.validate_bundle(self.bundle)
        self.assertEqual(len(result["packages"]), 9)

    def test_binding_provenance_must_match_application_source(self):
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_revision"] = "b" * 40
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            runtime.validate_bundle(self.bundle)

    def test_base_without_cairo_can_install_private_wheels_before_native_verification(self):
        with (mock.patch.object(runtime.os, "geteuid", return_value=501),
              mock.patch.object(runtime, "base_probe", return_value={}) as base,
              mock.patch.object(runtime, "native_probe", side_effect=AssertionError("Bare Python must not import Cairo")),
              mock.patch.object(runtime, "pip_install") as install,
              mock.patch.object(runtime, "run") as run):
            runtime.install(self.bundle, self.runtime)
        base.assert_called_once()
        install.assert_called_once()
        self.assertEqual(run.call_args_list[0].args, (sys.executable, "-m", "venv", self.runtime / ".venv"))
        self.assertIn("verify", run.call_args_list[1].args)
        self.assertTrue((self.runtime / "runtime-ready").is_file())

    def test_failed_private_native_verification_never_marks_ready(self):
        with (mock.patch.object(runtime.os, "geteuid", return_value=501),
              mock.patch.object(runtime, "base_probe", return_value={}),
              mock.patch.object(runtime, "pip_install"),
              mock.patch.object(runtime, "run", side_effect=[mock.Mock(), subprocess.CalledProcessError(1, "native verify")])):
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.install(self.bundle, self.runtime)
        self.assertFalse((self.runtime / "runtime-ready").exists())

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
        with mock.patch.object(runtime.os, "geteuid", return_value=0), mock.patch.object(runtime, "base_probe") as probe:
            with self.assertRaisesRegex(RuntimeError, "without sudo"):
                runtime.install(self.bundle, self.runtime)
        probe.assert_not_called()
        self.assertFalse(self.runtime.exists())

    def test_failed_wheel_install_never_marks_ready_or_touches_old_runtime(self):
        previous = self.root / "previous-runtime"
        previous.mkdir()
        (previous / "runtime-ready").write_text("previous successful runtime")
        with mock.patch.object(runtime.os, "geteuid", return_value=501), \
                mock.patch.object(runtime, "base_probe", return_value={}), \
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
                mock.patch.object(runtime, "base_probe", return_value={}), \
                mock.patch.object(runtime, "pip_install") as install:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                runtime.install(self.bundle, self.runtime)
        install.assert_not_called()
        self.assertEqual((self.runtime / "keep.txt").read_text(), "inspect this failure")

    def test_changed_payload_is_rejected_before_environment_creation(self):
        (self.bundle / "extra-script.py").write_text("changed payload")
        with mock.patch.object(runtime.os, "geteuid", return_value=501), \
                mock.patch.object(runtime, "base_probe", return_value={}):
            with self.assertRaisesRegex(ValueError, "identity"):
                runtime.install(self.bundle, self.runtime)
        self.assertFalse(self.runtime.exists())

    def test_redirected_user_runtime_is_rejected_without_touching_target(self):
        target = self.root / "separate-target"
        target.mkdir()
        self.runtime.symlink_to(target, target_is_directory=True)
        with mock.patch.object(runtime.os, "geteuid", return_value=501), \
                mock.patch.object(runtime, "base_probe", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "redirected user runtime"):
                runtime.install(self.bundle, self.runtime)
        self.assertEqual(list(target.iterdir()), [])

    def test_pip_is_offline_hash_checked_and_does_not_resolve(self):
        with mock.patch.object(runtime, "run") as run:
            runtime.pip_install(Path("/user/.venv/bin/python"), self.bundle, self.bundle / "requirements.txt")
        arguments = run.call_args.args
        for flag in ("--isolated", "--no-index", "--no-deps", "--require-hashes", "--only-binary=:all:",
                     "--force-reinstall", "--disable-pip-version-check", "--no-cache-dir"):
            self.assertIn(flag, arguments)
        self.assertNotIn("--upgrade", arguments)

    def test_offline_install_replaces_bootstrap_metadata_without_duplicate_owners(self):
        environment = self.root / "real-venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True,
                       capture_output=True, text=True)
        python = environment / "bin/python"
        requirement = self.root / "fixture-requirement.txt"
        for version in ("1.0", "2.0"):
            name = "nvb_offline_fixture"
            info = f"{name}-{version}.dist-info"
            wheel = self.wheels / f"{name}-{version}-py3-none-any.whl"
            contents = {
                f"{name}.py": f"VERSION = '{version}'\n",
                f"{info}/METADATA": f"Metadata-Version: 2.3\nName: nvb-offline-fixture\nVersion: {version}\n",
                f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            }
            contents[f"{info}/RECORD"] = "".join(f"{path},,\n" for path in [*contents, f"{info}/RECORD"])
            with zipfile.ZipFile(wheel, "w") as archive:
                for path, content in contents.items():
                    archive.writestr(path, content)
            requirement.write_text(runtime.requirements([runtime.wheel_record(wheel)]))
            runtime.pip_install(python, self.bundle, requirement)
        result = subprocess.run([str(python), "-c", '''
from importlib import metadata
import nvb_offline_fixture
assert nvb_offline_fixture.VERSION == '2.0'
versions = [d.version for d in metadata.distributions() if d.metadata['Name'] == 'nvb-offline-fixture']
assert versions == ['2.0'], versions
'''], check=True, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_local_metadata_owner_must_be_unique_even_if_first_record_matches(self):
        prefix = self.root / "metadata-env"
        local = prefix / "lib/python3.13/site-packages"
        external = self.root / "external-native-site"
        for directory, version in ((local, "1.0"), (external, "9.0")):
            record = directory / f"fixture-{version}.dist-info"
            record.mkdir(parents=True)
            (record / "METADATA").write_text(f"Metadata-Version: 2.3\nName: fixture\nVersion: {version}\n")
        packages = [{"name": "fixture", "version": "1.0"}]
        paths = [str(local), str(local), str(external)]
        self.assertEqual(runtime.pinned_distribution_problems(packages, prefix, paths), [])
        stale = local / "fixture-0.9.dist-info"
        stale.mkdir()
        (stale / "METADATA").write_text("Metadata-Version: 2.3\nName: fixture\nVersion: 0.9\n")
        problems = runtime.pinned_distribution_problems(packages, prefix, paths)
        self.assertEqual(len(problems), 1)
        self.assertIn("Expected one local fixture==1.0 owner", problems[0])

    def test_candidate_preinstall_reuses_native_privilege_policy(self):
        script = builder.root_preinstall()
        for phrase in ("admin-owned", "ACL requires administrator review", "Refusing symlinked",
                       "running system volume", "Apple Silicon", "macOS 14"):
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
        ranks = wheel_target.rank_wheels([old, newer, intel, universal, other_abi], minimum_macos=(13, 0))
        self.assertIsNotNone(ranks[old])
        self.assertIsNone(ranks[newer])
        self.assertIsNone(ranks[intel])
        self.assertIsNone(ranks[other_abi])
        self.assertIsNotNone(ranks[universal])
        self.assertLess(ranks[old], ranks[universal])

    def test_candidate_rejects_unpatched_python_before_native_imports(self):
        for version in ((3, 13, 0), (3, 13, 15), (3, 13, 17)):
            with self.subTest(version=version), mock.patch.object(runtime.sys, "version_info", version):
                with self.assertRaisesRegex(RuntimeError, "CPython 3.13.18 or newer"):
                    runtime.native_probe()
        for version in ((3, 13, 18), (3, 13, 19)):
            runtime.validate_python_version(version)
        for version in ((3, 12, 16), (3, 14, 10)):
            with self.assertRaisesRegex(RuntimeError, "matching native bindings"):
                runtime.validate_python_version(version)
        pins = json.loads((HERE / "inputs.json").read_text())
        self.assertEqual(pins["minimum_python"], ".".join(map(str, runtime.MINIMUM_PYTHON)))
        self.assertEqual(supplier.PYTHON_VERSION, pins["minimum_python"])

    @unittest.skipIf(os.geteuid() == 0, "candidate launcher refuses Installer root")
    def test_launcher_does_not_start_app_when_interpreter_policy_fails(self):
        bundle = self.root / "launch-bundle"
        bundle.mkdir()
        identity = "a" * 64
        (bundle / "macos-runtime-id").write_text(identity + "\n")
        user_home = self.root / "user-home"
        installed = user_home / "Library/Application Support/NVBroadcast Offline Candidate" / identity
        (installed / ".venv/bin").mkdir(parents=True)
        (installed / "runtime-ready").write_text(identity + "\n")
        python = installed / ".venv/bin/python"
        python.write_text('''#!/bin/bash
if [[ "$2" == python-policy ]]; then
    echo 'interpreter policy rejected' >&2
    exit 23
fi
echo application-started >> "$NVB_TEST_LOG"
''')
        python.chmod(0o755)
        launcher = self.root / "candidate-launcher"
        launcher.write_text((HERE / "launcher.sh").read_text().replace(
            "bundle=/opt/nvbroadcast-offline-candidate", "bundle=" + shlex.quote(str(bundle))))
        app_log = self.root / "app.log"
        result = subprocess.run(["bash", str(launcher), "--help"], capture_output=True, text=True,
                                env={**os.environ, "HOME": str(user_home), "NVB_TEST_LOG": str(app_log)})
        self.assertEqual(result.returncode, 23)
        self.assertIn("interpreter policy rejected", result.stderr)
        self.assertFalse(app_log.exists())

    def test_candidate_floor_matches_pinned_pyav_and_rejects_higher_tags(self):
        pinned = "av-16.1.0-cp313-cp313-macosx_14_0_arm64.whl"
        newer = "av-16.1.0-cp313-cp313-macosx_15_0_arm64.whl"
        self.assertIsNotNone(wheel_target.rank_wheels([pinned])[pinned])
        self.assertIsNone(wheel_target.rank_wheels([newer])[newer])
        pins = json.loads((HERE / "inputs.json").read_text())
        self.assertEqual(pins["minimum_macos"], runtime.MINIMUM_MACOS)
        self.assertEqual(runtime.MINIMUM_MACOS, "14.0")


class MacPrivateBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = self.root / "bindings"
        (self.bundle / "wheels").mkdir(parents=True)
        self.receipt, self.expected = write_binding_inputs(self.bundle)
        self.sources = json.loads((self.bundle / runtime.BINDING_INPUT_FILE).read_text())
        self.source = next(item for item in self.sources["bindings"] if item["name"] == "pycairo")

    def seal(self):
        path = self.bundle / runtime.BINDING_MANIFEST_FILE
        path.write_text(json.dumps(self.receipt))
        self.expected = runtime.digest(path)

    def test_exact_prepared_binding_inputs_are_accepted(self):
        result = runtime.validate_binding_inputs(self.bundle, self.expected)
        self.assertEqual({item["name"] for item in result["wheels"]}, runtime.BINDING_NAMES)
        for item in result["wheels"]:
            self.assertTrue(item["source_licenses"])
            self.assertTrue(item["native_objects"])

    def test_manifest_source_inventory_and_wheel_tampering_are_rejected(self):
        paths = [self.bundle / name for name in (
            runtime.BINDING_MANIFEST_FILE, runtime.BINDING_INPUT_FILE, runtime.BINDING_INVENTORY_FILE)]
        paths += [self.bundle / "wheels" / self.receipt["wheels"][0]["file"]]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    with self.assertRaises(ValueError):
                        runtime.validate_binding_inputs(self.bundle, self.expected)
                finally:
                    path.write_bytes(original)

    def test_missing_extra_duplicate_and_redirected_binding_inputs_fail(self):
        original = copy.deepcopy(self.receipt)
        for change in ("missing", "duplicate", "owner", "path", "target"):
            with self.subTest(change=change):
                self.receipt = copy.deepcopy(original)
                if change == "missing":
                    self.receipt["wheels"].pop()
                elif change == "duplicate":
                    self.receipt["wheels"].append(self.receipt["wheels"][0])
                elif change == "owner":
                    self.receipt["wheels"][0]["name"] = "unreviewed"
                elif change == "path":
                    self.receipt["source_inputs"]["file"] = "../other.json"
                else:
                    self.receipt["target"] = "macos-arm64-cp314-cpu"
                self.seal()
                with self.assertRaises(ValueError):
                    runtime.validate_binding_inputs(self.bundle, self.expected)
        self.receipt = original
        self.seal()
        extra = self.bundle / "wheels/unexpected.whl"
        extra.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "undeclared"):
            runtime.validate_binding_inputs(self.bundle, self.expected)
        extra.unlink()
        path = self.bundle / "wheels" / original["wheels"][0]["file"]
        outside = self.root / path.name
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "regular wheel"):
            runtime.validate_binding_inputs(self.bundle, self.expected)

    def test_python_architecture_and_exact_deployment_floor_are_checked(self):
        for tag in ("cp314-cp314-macosx_14_0_arm64", "cp313-cp313-macosx_14_0_x86_64",
                    "cp313-cp313-macosx_15_0_arm64", "py3-none-any"):
            with self.subTest(tag=tag):
                path, licenses = write_binding_fixture(self.bundle / "wheels", self.source, tag=tag)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    runtime.validate_binding_wheel(path, self.source, licenses)
        for floor in ((14, 0, 1), (14, 1, 0), (15, 0, 0)):
            with self.subTest(floor=floor):
                path, licenses = write_binding_fixture(self.bundle / "wheels", self.source, floor=floor)
                with self.assertRaisesRegex(ValueError, "deployment floor"):
                    runtime.validate_binding_wheel(path, self.source, licenses)
        with self.assertRaisesRegex(ValueError, "architecture"):
            runtime._native_object(native_fixture(cpu=0x01000007))
        with self.assertRaisesRegex(ValueError, "Mach-O"):
            runtime._native_object(b"not an extension")
        path, licenses = write_binding_fixture(self.bundle / "wheels", self.source,
                                               tag="cp313-cp313-macosx_11_0_arm64")
        with self.assertRaisesRegex(ValueError, "wheel tag"):
            runtime.validate_binding_wheel(path, self.source, licenses)

    def test_wheel_filename_and_wheel_metadata_are_bound_to_source(self):
        path, licenses = write_binding_fixture(self.bundle / "wheels", self.source)
        renamed = path.with_name(path.name.replace("pycairo-1.29.2-", "unrelated-9.0-"))
        path.rename(renamed)
        with self.assertRaisesRegex(ValueError, "filename"):
            runtime.validate_binding_wheel(renamed, self.source, licenses)
        renamed.rename(path)
        with zipfile.ZipFile(path) as archive:
            payloads = {item.filename: archive.read(item) for item in archive.infolist()}
        wheel_metadata = next(name for name in payloads if name.endswith(".dist-info/WHEEL"))
        payloads[wheel_metadata] = payloads[wheel_metadata].replace(b"cp313-cp313-", b"cp314-cp314-")
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in payloads.items():
                archive.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "tags differ"):
            runtime.validate_binding_wheel(path, self.source, licenses)

    def test_private_module_origins_and_actual_installed_bytes_are_checked(self):
        prefix = self.root / "private"
        site = prefix / "lib/python3.13/site-packages"
        paths = {}
        for name in ("gi/__init__.py", "gi/_gi.so", "cairo/__init__.py", "cairo/_cairo.so"):
            path = site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            paths[name] = path
        expected = {name: runtime.digest(path) for name, path in paths.items()}
        runtime.verify_binding_modules(paths, prefix, site, expected)
        paths["gi/_gi.so"].write_bytes(b"changed extension")
        with self.assertRaisesRegex(RuntimeError, "packaged wheel"):
            runtime.verify_binding_modules(paths, prefix, site, expected)
        outside = self.root / "external.so"
        outside.write_bytes(b"external")
        mixed = {**paths, "gi/_gi.so": outside}
        with self.assertRaisesRegex(RuntimeError, "private prefix"):
            runtime.verify_binding_modules(mixed, prefix, site, None)
        paths["gi/_gi.so"].unlink()
        paths["gi/_gi.so"].symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "private prefix"):
            runtime.verify_binding_modules(paths, prefix, site, None)

    def test_producer_digest_is_new_and_kept_outside_mutable_inputs(self):
        cache = self.root / "cache"
        cache.mkdir()
        output = self.root / "expected.sha256"
        binding_prepare.write_expected_digest(output, self.expected, self.bundle, cache)
        self.assertEqual(output.read_text().strip(), self.expected)
        with self.assertRaises(FileExistsError):
            binding_prepare.write_expected_digest(output, "b" * 64, self.bundle, cache)
        self.assertEqual(output.read_text().strip(), self.expected)
        for path in (cache / "expected.sha256", self.bundle / "expected.sha256"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                binding_prepare.write_expected_digest(path, self.expected, self.bundle, cache)
        link = self.root / "redirected.sha256"
        link.symlink_to(output)
        with self.assertRaises(FileExistsError):
            binding_prepare.write_expected_digest(link, self.expected, self.bundle, cache)

    def test_authenticated_source_notices_are_retained_and_record_is_regenerated(self):
        path, _ = write_binding_fixture(self.bundle / "wheels", self.source)
        with zipfile.ZipFile(path) as wheel:
            original_modules = {name: wheel.read(name) for name in wheel.namelist() if name.endswith(".so")}
        source_archive = self.root / "pycairo-1.29.2.tar.gz"
        entries = {"pycairo-1.29.2/COPYING": b"source notice\n",
                   "pycairo-1.29.2/LICENSES/MPL-1.1.txt": b"source licence text\n",
                   "pycairo-1.29.2/REUSE.toml": b"source attribution metadata\n",
                   "pycairo-1.29.2/README.md": b"ordinary documentation\n"}
        with tarfile.open(source_archive, "w:gz") as archive:
            for name, data in entries.items():
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        source = {**self.source, "sha256": runtime.digest(source_archive)}
        retained = binding_prepare.retain_source_licenses(path, source_archive, source)
        self.assertEqual(set(retained), set(entries) - {"pycairo-1.29.2/README.md"})
        runtime.validate_binding_wheel(path, source, retained)
        with zipfile.ZipFile(path) as wheel:
            for name, data in original_modules.items():
                self.assertEqual(wheel.read(name), data)
            for name, record in retained.items():
                self.assertEqual(wheel.read(record["file"]), entries[name])
            record_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/RECORD"))
            rows = list(csv.reader(io.StringIO(wheel.read(record_path).decode())))
            self.assertEqual({row[0] for row in rows}, set(wheel.namelist()))
            for name, expected_hash, expected_size in rows:
                if name == record_path:
                    self.assertEqual((expected_hash, expected_size), ("", ""))
                    continue
                data = wheel.read(name)
                expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                self.assertEqual((expected_hash, int(expected_size)), (expected, len(data)))

    def test_compilation_requires_actual_network_denial_before_any_install(self):
        client = mock.MagicMock()
        client.__enter__.return_value = client
        client.connect.return_value = None
        with (mock.patch.object(runtime.os, "geteuid", return_value=501),
              mock.patch.object(runtime, "base_probe", return_value={}),
              mock.patch.object(binding_prepare.socket, "socket", return_value=client),
              mock.patch.object(runtime, "run") as run):
            with self.assertRaisesRegex(RuntimeError, "enforced network denial"):
                binding_prepare.compile_bindings(self.bundle, self.root, "", "", "a" * 40, "1")
        run.assert_not_called()

    def test_builder_requires_explicit_expected_binding_digest(self):
        with (mock.patch.object(builder.os, "geteuid", return_value=501),
              mock.patch.object(runtime, "base_probe") as probe):
            with self.assertRaisesRegex(ValueError, "explicit expected SHA"):
                builder.build(self.root / "output", native_bindings=self.bundle)
        probe.assert_not_called()

    def test_private_binding_dependencies_are_checked_with_upstream_selection(self):
        upstream = self.root / "upstream"
        upstream.mkdir()
        app = self.root / "nvbroadcast-1.0-py3-none-any.whl"
        with zipfile.ZipFile(app, "w") as archive:
            archive.writestr("nvbroadcast-1.0.dist-info/METADATA",
                             "Metadata-Version: 2.3\nName: nvbroadcast\nVersion: 1.0\n")
        result = binding_prepare.verify_closure(self.bundle, self.expected, upstream, app)
        self.assertEqual(result["status"], "pass")
        record = next(item for item in self.receipt["wheels"] if item["name"] == "pygobject")
        wheel = self.bundle / "wheels" / record["file"]
        with zipfile.ZipFile(wheel) as archive:
            contents = {item.filename: archive.read(item) for item in archive.infolist()}
        metadata = next(name for name in contents if name.endswith(".dist-info/METADATA"))
        contents[metadata] = contents[metadata].replace(b"pycairo>=1.16.0", b"pycairo>=99")
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        source = next(item for item in self.sources["bindings"] if item["name"] == "PyGObject")
        updated = runtime.validate_binding_wheel(wheel, source, record["source_licenses"])
        self.receipt["wheels"] = [updated if item["name"] == "pygobject" else item for item in self.receipt["wheels"]]
        self.seal()
        with self.assertRaisesRegex(ValueError, "Unsatisfied private binding dependency"):
            binding_prepare.verify_closure(self.bundle, self.expected, upstream, app)


class MacFixedSupplierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.wheels = self.root / "supplier-wheels"
        self.wheels.mkdir()
        self.selection = []
        for name in sorted(supplier.REQUIRED):
            self.add_supplier(name)
        self.app = self.write_wheel(self.root, "nvbroadcast", requirements=[
            'onnxruntime>=1; extra == "cpu"', 'faster-whisper>=1; extra == "meeting-support"',
            'linux-only; sys_platform == "linux"'], extras=("cpu", "meeting-support"))
        self.pins = [{key: item[key] for key in ("name", "version", "url", "sha256")}
                     for item in self.selection if item["name"] in {"pip", "setuptools", "wheel", "packaging"}]

    def write_wheel(self, directory, name, version="1.0", requirements=(), extras=(), tag="py3-none-any"):
        normalized = name.replace("-", "_")
        path = directory / f"{normalized}-{version}-{tag}.whl"
        text = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\nRequires-Python: >=3.11\n"
        text += "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
        text += "".join(f"Provides-Extra: {extra}\n" for extra in extras)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{normalized}-{version}.dist-info/METADATA", text)
        return path

    def add_supplier(self, name, **kwargs):
        path = self.write_wheel(self.wheels, name, **kwargs)
        item = {"name": name, "version": kwargs.get("version", "1.0"), "file": path.name,
                "url": "https://supplier.example/" + path.name, "sha256": supplier.sha256(path),
                "size": path.stat().st_size}
        self.selection.append(item)
        return item

    def verify(self):
        return supplier.verify_wheelhouse(self.selection, self.wheels, self.app, "1.0", self.pins)

    def test_committed_selection_is_hash_pinned_and_matches_bootstrap(self):
        pins = json.loads((HERE / "inputs.json").read_text())
        lock = pins["supplier_lock"]
        selected = supplier.read_selection(HERE / lock["file"], lock["sha256"])
        self.assertEqual(len(selected), 54)
        owners = {item["name"]: item for item in selected}
        self.assertNotIn("nvbroadcast", owners)
        for item in pins["bootstrap"]:
            self.assertEqual({key: owners[item["name"]][key] for key in item}, item)

    def test_normal_selection_executes_no_resolver_and_rejects_changed_lock(self):
        pins = json.loads((HERE / "inputs.json").read_text())
        with mock.patch.object(builder.prepare, "fetch", side_effect=AssertionError("Unexpected resolver download")):
            selection, path, digest = builder.select_suppliers(Path(sys.executable), self.root, self.root,
                pins, self.app, [], "1.0", refresh_lock=False)
        self.assertEqual(len(selection), 54)
        self.assertEqual(digest, supplier.sha256(path))
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            supplier.read_selection(path, digest)

    def test_fixed_lock_rejects_missing_duplicate_local_and_wrong_target_entries(self):
        lock = tomllib.loads(supplier.render_lock(self.selection))
        cases = []
        missing = copy.deepcopy(lock); missing["packages"].pop(); cases.append(missing)
        duplicate = copy.deepcopy(lock); duplicate["packages"].append(duplicate["packages"][0]); cases.append(duplicate)
        local = copy.deepcopy(lock); local["packages"][0]["archive"] = {"path": "evil.whl"}; cases.append(local)
        for tag in ("cp314-cp314-macosx_14_0_arm64", "cp313-cp313-macosx_14_0_x86_64", "cp313-cp313-macosx_15_0_arm64"):
            wrong = copy.deepcopy(lock)
            wrong["packages"][0]["wheels"][0]["url"] = wrong["packages"][0]["wheels"][0]["url"].replace("py3-none-any", tag)
            cases.append(wrong)
        for invalid in cases:
            with self.subTest(invalid=invalid["packages"][0]["name"]):
                with self.assertRaises(ValueError):
                    supplier.supplier_selection(invalid)

    def test_supplier_download_never_repairs_missing_or_tampered_offline_input(self):
        item = self.selection[0]
        with mock.patch.object(builder.prepare.urllib.request, "urlopen", side_effect=AssertionError("Network access")):
            with self.assertRaises(FileNotFoundError):
                builder.prepare.fetch(self.root, item, offline=True)
            (self.root / item["file"]).write_bytes(b"tampered cache")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                builder.prepare.fetch(self.root, item, offline=True)

    def test_current_application_and_transitive_extra_closure_are_verified(self):
        self.add_supplier("parent", requirements=['child>=2; extra == "chosen"'], extras=("chosen",))
        self.add_supplier("child", version="2.0")
        self.app = self.write_wheel(self.root, "nvbroadcast", requirements=['parent[chosen]>=1'], extras=("cpu", "meeting-support"))
        self.assertEqual(self.verify()["dependency_closure"], "pass")
        self.app = self.write_wheel(self.root, "nvbroadcast", requirements=['parent[chosen]>=2'], extras=("cpu", "meeting-support"))
        with self.assertRaisesRegex(ValueError, "Unsatisfied locked dependency"):
            self.verify()

    def test_marker_evaluation_uses_mac_target_and_app_extras(self):
        self.assertEqual(self.verify()["status"], "pass")
        for requirement in ('new-package; sys_platform == "darwin"', 'new-package; extra == "cpu"',
                            'new-package; extra == "meeting-support"'):
            self.app = self.write_wheel(self.root, "nvbroadcast", requirements=[requirement], extras=("cpu", "meeting-support"))
            with self.subTest(requirement=requirement):
                with self.assertRaisesRegex(ValueError, "Unsatisfied locked dependency"):
                    self.verify()

    def test_active_direct_url_and_undefined_dependency_extra_are_rejected(self):
        for requirement, error in [('onnxruntime @ https://supplier.example/other.whl', "direct URL"),
                                    ('onnxruntime[not-provided]', "requested extras")]:
            self.app = self.write_wheel(self.root, "nvbroadcast", requirements=[requirement], extras=("cpu", "meeting-support"))
            with self.assertRaisesRegex(ValueError, error):
                self.verify()

    def test_missing_extra_tampered_and_redirected_wheelhouse_files_are_rejected(self):
        self.assertEqual(self.verify()["status"], "pass")
        path = self.wheels / self.selection[0]["file"]
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing, extra"):
            self.verify()
        path.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            self.verify()
        path.unlink()
        source = self.root / path.name
        source.write_bytes(original)
        path.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "hash/size"):
            self.verify()
        path.unlink(); path.write_bytes(original)
        (self.wheels / "extra.whl").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "missing, extra"):
            self.verify()

    def test_changed_managed_version_and_builder_bootstrap_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "faster-whisper"):
            supplier.verify_wheelhouse(self.selection, self.wheels, self.app, "2.0", self.pins)
        self.pins[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.verify()

    def test_refreshed_lock_requires_current_application_identity(self):
        resolved = supplier.render_lock(self.selection) + ('\n[[packages]]\nname = "nvbroadcast"\nversion = "1.0"\n'
            'archive = { path = ' + json.dumps(str(self.app)) + ', hashes = { sha256 = ' +
            json.dumps(supplier.sha256(self.app)) + ' } }\n')
        path = self.root / "resolved.toml"; path.write_text(resolved)
        output = self.root / "proposal.toml"
        supplier.propose_lock(path, self.app, output)
        self.assertEqual(supplier.supplier_selection(tomllib.loads(output.read_text())), self.selection)
        self.app.write_bytes(self.app.read_bytes() + b"new source")
        with self.assertRaisesRegex(ValueError, "rebuilt source wheel"):
            supplier.propose_lock(path, self.app, output)

    def test_refresh_cannot_hide_online_resolution_behind_offline_flag(self):
        with self.assertRaisesRegex(ValueError, "explicit online resolution"):
            builder.build(self.root / "output", refresh_lock=True, offline=True)


class MacGeneratedMediaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.retained = self.root / "diagnostics"
        self.recordings = []
        self.gst = mock.MagicMock()

        def encode(description):
            location = next(token.removeprefix("location=") for token in shlex.split(description)
                            if token.startswith("location="))
            recording = Path(location)
            recording.write_bytes(b"synthetic fixture only")
            self.recordings.append(recording)
            return mock.Mock()

        self.gst.parse_launch.side_effect = encode
        patcher = mock.patch.object(media_validator, "_complete", return_value={"status": "pass"})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def decoded(video=24, audio=49, status="pass"):
        return {"status": status, "decoded_buffers": {"video": video, "audio": audio},
                "decoder_factories": [{"factory": "fixture_decoder"}]}

    def check(self, reports, *, retain=True, repeats=1):
        with mock.patch.object(media_validator, "_decode_recording", side_effect=reports) as decode:
            result = media_validator.check_generated_media(
                self.gst, "fixture_video", "fixture_audio", self.retained if retain else None,
                diagnostic_repeats=repeats)
            return result, decode.call_count

    def evidence(self):
        paths = list(self.retained.glob("attempt-*/diagnostics.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def test_complete_software_cannot_rescue_truncated_automatic_recording(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete audiovisual"):
            self.check([self.decoded(video=3), self.decoded()])
        evidence = self.evidence()
        self.assertEqual(evidence["status"], "fail")
        self.assertEqual(evidence["automatic"]["decoded_buffers"]["video"], 3)
        self.assertEqual(evidence["software"]["decoded_buffers"]["video"], 24)
        self.assertEqual(self.recordings[0].read_bytes(), b"synthetic fixture only")

    def test_missing_software_plugins_cannot_rescue_automatic_failure(self):
        self.gst.ElementFactory.find.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Incomplete audiovisual"):
            self.check([self.decoded(status="fail")])
        self.assertEqual(self.evidence()["software"]["status"], "unavailable")

    def test_normal_checks_clean_up_recording_on_success_and_failure(self):
        result, count = self.check([self.decoded()], retain=False)
        self.assertEqual(count, 1)  # Normal success does not run a software comparison.
        self.assertFalse(result["decoder_diagnostics"]["retained"])
        self.assertFalse(self.recordings[-1].parent.exists())
        with self.assertRaisesRegex(RuntimeError, "Incomplete audiovisual"):
            self.check([self.decoded(video=3), self.decoded()], retain=False)
        self.assertFalse(self.recordings[-1].parent.exists())
        self.assertFalse(self.retained.exists())

    def test_later_automatic_repeat_failure_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete audiovisual"):
            self.check([self.decoded(), self.decoded(video=3), self.decoded(), self.decoded()], repeats=3)
        self.assertEqual(len(self.evidence()["automatic_repeats"]), 3)

    def test_diagnostics_require_complete_software_and_present_plugins(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete audiovisual"):
            self.check([self.decoded(), self.decoded(), self.decoded(video=3)], repeats=2)
        self.gst.ElementFactory.find.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Software comparison plugins are unavailable"):
            self.check([self.decoded(), self.decoded()], retain=False, repeats=2)

    def test_video_without_audio_remains_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete audiovisual"):
            self.check([self.decoded(audio=0), self.decoded()])

    def test_redirected_diagnostic_directory_is_rejected_before_encoding(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.retained.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.check([])
        self.assertFalse(self.recordings)
        self.assertEqual(list(outside.iterdir()), [])

    def test_changed_file_cannot_be_compared_as_the_same_recording(self):
        def mutate(_gst, recording, *, software):
            recording.write_bytes(b"changed recording")
            return self.decoded()
        with mock.patch.object(media_validator, "_decode_recording", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "Generated file changed"):
                media_validator.check_generated_media(self.gst, "video", "audio", self.retained)
        self.assertEqual(self.evidence()["status"], "fail")


if __name__ == "__main__":
    unittest.main()

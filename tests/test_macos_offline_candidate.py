"""Portable negative tests; actual macOS runtime is qualified by its own CI job."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
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
spec = importlib.util.spec_from_file_location("mac_offline_target", HERE / "wheel_target.py")
wheel_target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_target)

spec = importlib.util.spec_from_file_location("mac_supplier_lock", HERE / "supplier_lock.py")
supplier = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"wheel_target": wheel_target, "runtime": runtime}):
    spec.loader.exec_module(supplier)


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
        manifest = {"schema_version": 1, "target": runtime.TARGET,
                    "minimum_macos": runtime.MINIMUM_MACOS, "packages": packages}
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


if __name__ == "__main__":
    unittest.main()

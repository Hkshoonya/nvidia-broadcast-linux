"""Candidate boundaries and real subprocess probes, without Docker or installation."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "packaging/native-runtime"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load("candidate_qualification", HERE / "qualify_candidate.py")
recovery = candidate.recovery
smoke = load("candidate_smoke", HERE / "smoke_candidate.py")
fixture = load("candidate_fixture", HERE / "fixtures/build_fixture.py")


class QualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nvb candidate tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifact = self.root / "package.rpm"
        self.artifact.write_bytes(b"unit fixture, never install")
        self.hash = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        self.arguments = ["--family", "rpm", "--artifact", str(self.artifact),
            "--artifact-sha256", self.hash, "--source", "1" * 40,
            "--package-version", "2.7.11", "--package-revision", "12",
            "--runtime-manifest-sha256", "2" * 64, "--recovery-helper-sha256", "3" * 64,
            "--image", "sha256:" + "4" * 64, "--output", str(self.root / "qualification")]

    def test_plan_is_read_only_and_version_is_not_fixed_to_the_original_candidate(self):
        before = set(self.root.iterdir())
        output = io.StringIO()
        with mock.patch.object(candidate, "execute", side_effect=AssertionError("execution")), \
             mock.patch.object(subprocess, "Popen", side_effect=AssertionError("external command")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(candidate.main(self.arguments), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["status"], "planned-not-executed")
        self.assertEqual(plan["variant"], "cpu")
        self.assertEqual(plan["version"], "2.7.11-12.cpu")
        self.assertEqual(plan["scenarios"], ["smoke", "recovery"])
        self.assertEqual(set(self.root.iterdir()), before)
        args = candidate.parse_arguments(self.arguments)
        args.family = "deb"
        self.assertEqual(candidate.make_plan(args)["scenarios"], ["smoke"])

    def test_cuda_plan_binds_variant_and_passes_it_to_both_drivers(self):
        args = candidate.parse_arguments([*self.arguments, "--variant", "cuda"])
        plan = candidate.make_plan(args)
        self.assertEqual((plan["variant"], plan["version"]), ("cuda", "2.7.11-12.cuda"))

        def finish(arguments):
            self.assertEqual(arguments[arguments.index("--variant") + 1], "cuda")
            destination = Path(arguments[arguments.index("--output") + 1])
            destination.mkdir()
            (destination / "result.json").write_text('{"status":"pass","variant":"cuda"}\n')

        smoke_driver = SimpleNamespace(main=mock.Mock(side_effect=finish))
        with mock.patch.object(candidate, "load", return_value=smoke_driver), \
             mock.patch.object(recovery, "main", side_effect=finish) as recover:
            result = candidate.execute(args, plan)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(set(result["receipts"]), {"smoke/result.json", "recovery/result.json"})
        smoke_driver.main.assert_called_once()
        recover.assert_called_once()

    def test_cuda_recovery_plan_and_signed_binding_cannot_accept_cpu_payload(self):
        arguments = ["--rpm", str(self.artifact), "--sha256", self.hash,
            "--source-revision", "1" * 40, "--runtime-manifest-sha256", "2" * 64,
            "--helper-sha256", "3" * 64, "--package-version", "2.7.11",
            "--package-revision", "12", "--image", "sha256:" + "4" * 64,
            "--output", str(self.root / "recovery"), "--variant", "cuda"]
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("external command")):
            plan = recovery.make_plan(recovery.parse_arguments(arguments))
        self.assertEqual((plan["variant"], plan["package_version_release"]), ("cuda", "2.7.11-12.cuda"))
        record = {"package": {"schema_version": 1, "variant": "cuda", "version": "2.7.11-12.cuda",
                  "source": {"revision": "1" * 40, "version": "2.7.11"},
                  "adapter": {"revision": "1" * 40}, "runtime_manifest_sha256": "2" * 64},
                  "runtime_manifest_sha256": "2" * 64, "helper_sha256": "3" * 64}
        recovery.check_bindings(record, plan)
        for field, value in (("variant", "cpu"), ("version", "2.7.11-12.cpu")):
            swapped = copy.deepcopy(record)
            swapped["package"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.check_bindings(swapped, plan)
        self.assertFalse((self.root / "recovery").exists())

    def test_unbound_or_shell_shaped_inputs_cannot_reach_execution(self):
        for name, value in (("source", "main"), ("artifact_sha256", "0" * 64),
                            ("recovery_helper_sha256", "pending"), ("image", "fedora:44"),
                            ("image", "sha256:" + "0" * 64), ("package_version", "2.7.11; true"),
                            ("package_revision", "0"), ("package_revision", "12.cpu")):
            args = candidate.parse_arguments(self.arguments)
            setattr(args, name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                candidate.make_plan(args)

    def test_wrong_bytes_links_and_existing_output_fail_without_modification(self):
        args = candidate.parse_arguments(self.arguments)
        self.artifact.write_bytes(b"changed package")
        with self.assertRaisesRegex(ValueError, "input identity"):
            candidate.make_plan(args)
        self.artifact.write_bytes(b"unit fixture, never install")
        link = self.root / "link"
        link.symlink_to(self.artifact)
        args.artifact = link
        with self.assertRaises(OSError):
            candidate.make_plan(args)
        args.artifact = self.artifact
        args.output.mkdir()
        sentinel = args.output / "keep"
        sentinel.write_text("earlier evidence")
        with self.assertRaisesRegex(ValueError, "fresh qualification"):
            candidate.make_plan(args)
        self.assertEqual(sentinel.read_text(), "earlier evidence")

    def test_snapshot_copies_exact_regular_bytes_and_preserves_prior_files(self):
        target = self.root / "snapshot"
        recovery.snapshot(self.artifact, target, self.hash)
        self.assertEqual(recovery.digest(target), self.hash)
        self.assertNotEqual(target.stat().st_ino, self.artifact.stat().st_ino)
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            recovery.snapshot(self.artifact, target, self.hash)
        with self.assertRaisesRegex(ValueError, "reviewed input"):
            recovery.snapshot(self.artifact, self.root / "wrong", "f" * 64)
        link = self.root / "linked"
        link.symlink_to(self.artifact)
        with self.assertRaises(OSError):
            recovery.snapshot(link, self.root / "linked-copy", self.hash)

    def test_package_bounds_are_selected_before_hashing_or_container_execution(self):
        for variant, size in (("cpu", 1024**3 + 1), ("cuda", 2 * 1024**3)):
            with self.subTest(variant=variant):
                with self.artifact.open("wb") as stream:
                    stream.truncate(size)
                args = candidate.parse_arguments([*self.arguments, "--variant", variant])
                with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("external command")), \
                     self.assertRaisesRegex(ValueError, "bounded regular"):
                    candidate.make_plan(args)
                self.assertFalse(args.output.exists())
        self.artifact.write_bytes(b"")
        with self.assertRaises(ValueError):
            recovery.package_digest(self.artifact, "cuda")

    def test_streaming_digest_and_snapshot_enforce_exact_bounds_and_stability(self):
        size = self.artifact.stat().st_size
        self.assertEqual(recovery.digest(self.artifact, limit=size), self.hash)
        with self.assertRaisesRegex(ValueError, "bounded regular"):
            recovery.digest(self.artifact, limit=size - 1)
        with self.assertRaisesRegex(ValueError, "snapshot input"):
            recovery.snapshot(self.artifact, self.root / "too-small", self.hash, limit=size - 1)
        original = os.fstat
        calls = 0

        def changed(descriptor):
            nonlocal calls
            calls += 1
            info = original(descriptor)
            if calls == 2:
                return SimpleNamespace(**{key: getattr(info, key) for key in
                    ("st_dev", "st_size", "st_mtime_ns", "st_ctime_ns")}, st_ino=info.st_ino + 1)
            return info

        with mock.patch.object(recovery.os, "fstat", side_effect=changed), \
             self.assertRaisesRegex(ValueError, "changed while hashing"):
            recovery.digest(self.artifact)

    def test_cuda_smoke_requires_no_device_probe_and_exact_cpu_execution(self):
        self.assertNotIn("--cuda-unavailable", smoke.probe_command("cpu", "3.13.16"))
        command = smoke.probe_command("cuda", "3.13.16")
        self.assertEqual(command[command.index("--variant") + 1], "cuda")
        self.assertIn("--cuda-unavailable", command)
        self.assertEqual(recovery.no_device_arguments(),
                         ["--runtime=runc", "--env", "NVIDIA_VISIBLE_DEVICES=void"])
        app = {"app.py": "5" * 64}
        report = {"status": "pass", "variant": "cuda", "app_python_hashes": app,
                  "cpu_execution": {"provider": "cpu", "success": True,
                                    "executed_providers": ["CPUExecutionProvider"]},
                  "cuda_execution": {"provider": "cuda", "success": False,
                                     "executed_providers": [], "error": "no CUDA device"},
                  "cuda_unavailable_expected": True, "cupy_execution": None}
        encode = lambda value: "RESULT=" + json.dumps(value) + "\n"
        self.assertEqual(smoke.runtime_report(encode(report), "cuda", app), report)
        for fields in ({"variant": "cpu"}, {"app_python_hashes": {}},
                       {"cpu_execution": {"provider": "cpu", "success": True, "executed_providers": []}},
                       {"cuda_execution": {"provider": "cuda", "success": True,
                                           "executed_providers": ["CUDAExecutionProvider"]}},
                       {"cuda_unavailable_expected": False}, {"cupy_execution": {"device": "unexpected"}}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                smoke.runtime_report(encode({**report, **fields}), "cuda", app)
        with self.assertRaises(ValueError):
            smoke.runtime_report(encode(report) * 2, "cuda", app)

    def test_recovered_probe_calls_both_providers_without_cached_results(self):
        providers = SimpleNamespace(CPU="cpu", CUDA="cuda")
        def compute(provider, *, use_cache):
            self.assertIs(use_cache, False)
            return SimpleNamespace(to_payload=lambda: {"provider": provider, "success": provider == "cpu",
                "executed_providers": ["CPUExecutionProvider"] if provider == "cpu" else [],
                "error": "" if provider == "cpu" else "no device"})
        invoke = mock.Mock(side_effect=compute)
        module = SimpleNamespace(ProbeProvider=providers, probe_execution_provider=invoke)
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"nvbroadcast.runtime.probe": module}), contextlib.redirect_stdout(output):
            exec(recovery.RECOVERED_RUNTIME, {})
        invoke.assert_has_calls([mock.call("cpu", use_cache=False), mock.call("cuda", use_cache=False)])
        self.assertEqual(invoke.call_count, 2)
        report = json.loads(output.getvalue())
        recovery.policy.validate_runtime_probe(report, "cuda")
        report["cpu_execution"]["success"] = False
        with self.assertRaises(ValueError):
            recovery.policy.validate_runtime_probe(report, "cuda")

    def test_digest_only_or_mixed_failed_signature_is_rejected(self):
        valid = "Header V4 RSA/SHA256 Signature, key ID dbacbcb5: OK\n"
        recovery.require_project_signature(valid)
        for text in ("Header SHA256 digest: OK\nPayload SHA256 digest: OK\n",
                     valid + "Payload SHA256 digest: BAD\n", valid + "Signature: NOKEY\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                recovery.require_project_signature(text)

    def test_payload_source_version_runtime_and_rendered_helper_must_all_match(self):
        plan = {"package_version_release": "2.7.11-12.cpu", "source_revision": "1" * 40,
                "package_version": "2.7.11", "runtime_manifest_sha256": "2" * 64,
                "helper_sha256": "3" * 64}
        record = {"package": {"schema_version": 1, "variant": "cpu", "version": "2.7.11-12.cpu",
                   "source": {"revision": "1" * 40, "version": "2.7.11"},
                   "adapter": {"revision": "1" * 40}, "runtime_manifest_sha256": "2" * 64},
                  "runtime_manifest_sha256": "2" * 64, "helper_sha256": "3" * 64}
        recovery.check_bindings(record, plan)
        changes = [(('package', 'source', 'revision'), '4' * 40),
                   (('package', 'adapter', 'revision'), '4' * 40),
                   (('package', 'version'), '2.7.11-11.cpu'),
                   (('package', 'source', 'version'), '2.7.10'),
                   (('package', 'variant'), 'cuda'),
                   (('package', 'runtime_manifest_sha256'), '4' * 64),
                   (('runtime_manifest_sha256',), '4' * 64), (('helper_sha256',), '4' * 64)]
        for keys, value in changes:
            modified = copy.deepcopy(record)
            parent = modified
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                recovery.check_bindings(modified, plan)

    def test_capped_runner_keeps_both_streams_and_enforces_output_and_time_bounds(self):
        result = recovery.run_capped([sys.executable, "-I", "-c",
            "import sys; print('normal'); print('diagnostic',file=sys.stderr)"], timeout=10)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"normal\n", b"diagnostic\n"))
        log = self.root / "bounded.log"
        with self.assertRaisesRegex(RuntimeError, "output limit"):
            recovery.run_capped([sys.executable, "-I", "-c", "print('x'*100000)"],
                                timeout=10, limit=1024, log=log)
        self.assertLessEqual(log.stat().st_size, 1024)
        with self.assertRaises(subprocess.TimeoutExpired):
            recovery.run_capped([sys.executable, "-I", "-c", "import time; time.sleep(5)"], timeout=0.05)

    def test_image_must_match_exact_linux_amd64_id(self):
        image = "sha256:" + "4" * 64
        def inspect(record):
            return mock.patch.object(recovery, "run_capped", return_value=
                subprocess.CompletedProcess([], 0, json.dumps([record]).encode(), b""))
        valid = {"Id": image, "Os": "linux", "Architecture": "amd64"}
        with inspect(valid):
            recovery.inspect_image(image)
        for key, value in (("Id", "sha256:" + "5" * 64), ("Os", "windows"), ("Architecture", "arm64")):
            with self.subTest(key=key), inspect({**valid, key: value}), self.assertRaises(ValueError):
                recovery.inspect_image(image)

    def test_smoke_main_reaches_source_preflight_without_container_execution(self):
        arguments = self.arguments[:]
        with mock.patch.object(smoke.recovery, "inspect_image") as inspect, \
             mock.patch.object(smoke, "git_bytes", side_effect=RuntimeError("source preflight sentinel")), \
             mock.patch.object(subprocess, "Popen", side_effect=AssertionError("container execution")):
            with self.assertRaisesRegex(RuntimeError, "source preflight sentinel"):
                smoke.main(arguments)
        inspect.assert_called_once_with("sha256:" + "4" * 64)
        self.assertFalse((self.root / "qualification").exists())

    def test_probe_directory_is_traversable_under_private_umask(self):
        def source_bytes(source, path):
            if path == "pyproject.toml":
                return b'[project]\nversion = "2.7.11"\n'
            if path in smoke.PROBES.values():
                return b"public qualification probe\n"
            # Stop after creating the actual mounted probes, before any
            # package/container execution or further source preparation.
            raise RuntimeError("probe preparation complete")

        previous = os.umask(0o077)
        try:
            with mock.patch.object(smoke.recovery, "inspect_image"), \
                 mock.patch.object(smoke, "git_bytes", side_effect=source_bytes), \
                 mock.patch.object(subprocess, "check_output", return_value="1" * 40), \
                 mock.patch.object(subprocess, "Popen", side_effect=AssertionError("external execution")):
                with self.assertRaisesRegex(RuntimeError, "probe preparation complete"):
                    smoke.main(self.arguments)
        finally:
            os.umask(previous)
        output = self.root / "qualification"
        probes = output / "probes"
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((output / "package.rpm").stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(probes.stat().st_mode), 0o755)
        self.assertEqual({path.name for path in probes.iterdir()}, set(smoke.PROBES))
        for path in probes.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            self.assertEqual(path.read_bytes(), b"public qualification probe\n")

    def export_fixture(self):
        raw = self.root / "raw"
        smoke_dir = raw / "smoke"
        smoke_dir.mkdir(parents=True)
        log = smoke_dir / "00-launcher.log"
        log.write_bytes(b"verified launcher\n")
        (smoke_dir / "package.deb").write_bytes(b"must not be exported")
        (smoke_dir / "probes").mkdir()
        result = {"status": "pass", "steps": [{"log": log.name, "sha256": recovery.digest(log)}]}
        receipt = smoke_dir / "result.json"
        receipt.write_text(json.dumps(result))
        plan = {"output": str(raw), "scenarios": ["smoke"],
                "receipts": {"smoke/result.json": recovery.digest(receipt)}}
        return plan, result, receipt, log

    def test_public_export_excludes_packages_and_probes_and_verifies_logs(self):
        plan, result, receipt, log = self.export_fixture()
        target = self.root / "public"
        candidate.export_evidence(plan, target)
        self.assertEqual({str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()},
                         {"smoke/result.json", "smoke/00-launcher.log"})
        log.write_bytes(b"tampered evidence")
        with self.assertRaisesRegex(ValueError, "changed before export"):
            candidate.export_evidence(plan, self.root / "tampered")
        self.assertFalse((self.root / "tampered").exists())

    def test_public_export_rejects_path_traversal_and_changed_receipts(self):
        plan, result, receipt, _ = self.export_fixture()
        receipt.write_text("{}")
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            candidate.export_evidence(plan, self.root / "changed")
        result["steps"][0]["log"] = "../../outside"
        receipt.write_text(json.dumps(result))
        plan["receipts"]["smoke/result.json"] = recovery.digest(receipt)
        with self.assertRaisesRegex(ValueError, "log path"):
            candidate.export_evidence(plan, self.root / "traversal")
        self.assertFalse((self.root / "traversal").exists())

    def test_recovery_export_binds_preservation_and_manager_log_without_inputs(self):
        raw = self.root / "raw"
        directory = raw / "recovery"
        (directory / "inputs").mkdir(parents=True)
        (directory / "inputs/target.rpm").write_bytes(b"private snapshot")
        (directory / "inputs/public.asc").write_bytes(b"key snapshot")
        log = directory / "00-signature.log"
        log.write_bytes(b"Signature: OK\n")
        preservation = directory / "preservation.json"
        preservation.write_text('{"preserved": ["fragment"]}\n')
        manager = directory / "interrupted-manager.log"
        manager.write_bytes(b"Killed\n")
        receipt = directory / "result.json"
        receipt.write_text(json.dumps({"status": "pass",
            "steps": [{"log": log.name, "log_sha256": recovery.digest(log)}],
            "preservation_sha256": recovery.digest(preservation),
            "interruption": {"manager_log_sha256": recovery.digest(manager)}}))
        plan = {"output": str(raw), "scenarios": ["recovery"],
                "receipts": {"recovery/result.json": recovery.digest(receipt)}}
        target = self.root / "public"
        candidate.export_evidence(plan, target)
        self.assertEqual({str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()},
                         {"recovery/result.json", "recovery/00-signature.log", "recovery/preservation.json",
                          "recovery/interrupted-manager.log"})
        for index, path in enumerate((preservation, manager)):
            original = path.read_bytes()
            path.write_bytes(b"changed")
            destination = self.root / ("changed-" + str(index))
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "changed before export"):
                candidate.export_evidence(plan, destination)
            self.assertFalse(destination.exists())
            path.write_bytes(original)
        outside = self.root / "outside-log"
        log.rename(outside)
        log.symlink_to(outside)
        with self.assertRaises(OSError):
            candidate.export_evidence(plan, self.root / "symlink")
        self.assertFalse((self.root / "symlink").exists())

    def system_python(self, source, target):
        return subprocess.run(["sh", "-eu", "-c", source, "sentinel", str(target),
                               str(self.root / "python-state")], capture_output=True, timeout=10)

    def test_absent_system_python_stays_absent_and_dangling_links_are_not_absence(self):
        target = self.root / "no-python"
        self.assertEqual(self.system_python(smoke.SYSTEM_PYTHON_CAPTURE, target).returncode, 0)
        self.assertEqual(self.system_python(smoke.SYSTEM_PYTHON_VERIFY, target).returncode, 0)
        target.symlink_to(self.root / "missing")
        self.assertNotEqual(self.system_python(smoke.SYSTEM_PYTHON_VERIFY, target).returncode, 0)
        self.assertNotEqual(self.system_python(smoke.SYSTEM_PYTHON_CAPTURE, target).returncode, 0)

    def test_existing_system_python_hash_is_preserved(self):
        target = self.root / "python"
        target.write_bytes(b"existing system executable")
        self.assertEqual(self.system_python(smoke.SYSTEM_PYTHON_CAPTURE, target).returncode, 0)
        self.assertEqual(self.system_python(smoke.SYSTEM_PYTHON_VERIFY, target).returncode, 0)
        target.write_bytes(b"changed")
        self.assertNotEqual(self.system_python(smoke.SYSTEM_PYTHON_VERIFY, target).returncode, 0)

    def test_kill_is_limited_to_observed_rpm_group_and_transaction(self):
        version = "2.7.11-12.cpu"
        def path(value):
            obj = mock.Mock()
            obj.__str__ = mock.Mock(return_value=value)
            obj.read_text.return_value = "173" if value == "/tmp/manager.pid" else version + "\n"
            obj.read_bytes.return_value = b"rpm\0--upgrade\0/inputs/target.rpm\0"
            obj.lstat.return_value = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_size=2 * 1024**2)
            return obj
        for group, expected_calls in ((174, 0), (173, 1)):
            with mock.patch("pathlib.Path", side_effect=path), mock.patch("os.getpgid", return_value=group), \
                 mock.patch("os.killpg") as kill, mock.patch.object(sys, "argv", ["proof", "/fragment", version]), \
                 contextlib.redirect_stdout(io.StringIO()):
                if expected_calls:
                    exec(recovery.KILL_MANAGER, {})
                    kill.assert_called_once_with(173, signal.SIGKILL)
                else:
                    with self.assertRaises(AssertionError):
                        exec(recovery.KILL_MANAGER, {})
                    kill.assert_not_called()
        with mock.patch("pathlib.Path", side_effect=path), mock.patch("os.getpgid", return_value=173), \
             mock.patch("os.killpg") as kill, mock.patch.object(sys, "argv", ["proof", "/fragment", "wrong"]):
            with self.assertRaises(AssertionError):
                exec(recovery.KILL_MANAGER, {})
            kill.assert_not_called()

    def test_fixture_plan_uses_pinned_recipe_without_docker(self):
        for family in ("deb", "rpm"):
            captured = io.StringIO()
            with mock.patch.object(fixture, "build", side_effect=AssertionError("container build")), \
                 contextlib.redirect_stdout(captured):
                fixture.main(["--family", family, "--output", str(self.root / "fixture")])
            plan = json.loads(captured.getvalue())
            self.assertEqual(plan["status"], "planned-not-executed")
            self.assertEqual(plan["platform"], "linux/amd64")
            self.assertIn("@sha256:", plan["base"])
            self.assertFalse((self.root / "fixture").exists())

    def test_all_embedded_container_programs_compile(self):
        for name in ("PAYLOAD_BINDINGS", "INSTALLED_BINDINGS", "KILL_MANAGER", "PRESERVED_BYTES", "RECOVERED_RUNTIME"):
            compile(getattr(recovery, name), name, "exec")
        compile(smoke.INSTALLED_IDENTITY, "INSTALLED_IDENTITY", "exec")


if __name__ == "__main__":
    unittest.main()

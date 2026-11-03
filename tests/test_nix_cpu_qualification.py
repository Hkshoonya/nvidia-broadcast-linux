"""Failure tests for Nix preparation/qualification controls, without installing Nix."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "nix" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hosted = load("hosted_check")
runtime = load("qualify_runtime")
source_files = load("source_files")


def forecast_paths(count, suffix):
    return "".join(f"  /nix/store/{number:032d}-fixture{suffix}\n" for number in range(count))


class NixCpuQualificationTests(unittest.TestCase):
    def test_profile_selection_requires_an_exact_explicit_opt_in(self):
        for labels in ([], [{"name": "nix-full-rebuild-extra"}], [{"name": "other"}]):
            self.assertEqual(hosted.select_profile("pull_request", {"pull_request": {"labels": labels}}), "standard")
        for action in ("labeled", "synchronize"):
            self.assertEqual(hosted.select_profile("pull_request", {
                "action": action, "pull_request": {"labels": [{"name": "nix-full-rebuild"}]}}), "full-rebuild")
        self.assertEqual(hosted.select_profile("pull_request", {
            "action": "unlabeled", "label": {"name": "nix-full-rebuild"},
            "pull_request": {"labels": []}}), "standard")
        self.assertEqual(hosted.select_profile("workflow_dispatch", {}), "standard")
        self.assertEqual(hosted.select_profile("workflow_dispatch", {
            "inputs": {"profile": "full-rebuild", "operation": "qualify"}}), "full-rebuild")
        self.assertEqual(hosted.select_profile("workflow_dispatch", {
            "inputs": {"profile": "full-rebuild", "operation": "prepare-lock"}}), "standard")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            hosted.select_profile("workflow_dispatch", {"inputs": {"profile": "unlimited"}})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            hosted.select_profile("pull_request_target", {})

    def test_measured_full_plan_requires_separate_profile_and_keeps_standard_limits(self):
        measured = ("these 724 derivations will be built:\n" + forecast_paths(724, ".drv") +
                    "this path will be fetched (3.5 GiB download, 15.5 GiB unpacked):\n" + forecast_paths(1, "-cached"))
        with self.assertRaisesRegex(ValueError, "budget"):
            hosted.check_plan(measured, 85 * 1024**3)
        result = hosted.check_plan(measured, 85 * 1024**3, profile="full-rebuild")
        self.assertEqual(result["build_derivations"], 724)
        self.assertEqual(hosted.profile_limits("standard")["job_minutes"], 60)
        self.assertEqual(hosted.profile_limits("standard")["build_seconds"], 2400)
        self.assertEqual(result["limits"]["job_minutes"], 300)
        self.assertEqual(result["limits"]["build_seconds"], 270 * 60)

    def test_full_profile_refuses_excessive_build_download_unpack_and_scratch_costs(self):
        for forecast in (
            "these 901 derivations will be built:\n" + forecast_paths(901, ".drv"),
            "this path will be fetched (8.0 GiB download, 1.0 GiB unpacked):\n" + forecast_paths(1, "-cached"),
            "this path will be fetched (1.0 MiB download, 24.0 GiB unpacked):\n" + forecast_paths(1, "-cached"),
        ):
            with self.subTest(forecast=forecast[:60]), self.assertRaisesRegex(ValueError, "budget"):
                hosted.check_plan(forecast, 85 * 1024**3, profile="full-rebuild")
        with self.assertRaisesRegex(ValueError, "budget"):
            hosted.check_plan("this path will be fetched (1.0 MiB download, 15.5 GiB unpacked):\n" +
                              forecast_paths(1, "-cached"), 55 * 1024**3, profile="full-rebuild")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            hosted.check_plan("", 85 * 1024**3, verified_present=True, profile="unlimited")

    def test_full_profile_requires_hardware_floor_and_a_fixed_operation_growth_baseline(self):
        arguments = {"free_bytes": 75 * 1024**3, "cpus": 4, "memory_bytes": 14 * 1024**3}
        self.assertEqual(hosted.check_resources("full-rebuild", **arguments)["profile"], "full-rebuild")
        for key in arguments:
            reduced = {**arguments, key: arguments[key] - 1}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "too small"):
                hosted.check_resources("full-rebuild", **reduced)
        self.assertIsNone(hosted.check_live_storage("full-rebuild", 90 * 1024**3, 30 * 1024**3))
        self.assertIn("growth", hosted.check_live_storage("full-rebuild", 90 * 1024**3, 30 * 1024**3 - 1))
        self.assertIn("free-space", hosted.check_live_storage("full-rebuild", 75 * 1024**3, 10 * 1024**3 - 1))
        with self.assertRaisesRegex(ValueError, "fixed initial"):
            hosted.check_live_storage("full-rebuild", 0, 85 * 1024**3)

    def test_full_profile_live_growth_guard_terminates_the_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root, records = Path(directory), []
            with (mock.patch.object(hosted, "ROOT", root), mock.patch.object(hosted, "OUT", root),
                  mock.patch.object(hosted.shutil, "disk_usage", return_value=types.SimpleNamespace(free=29 * 1024**3))):
                with self.assertRaisesRegex(RuntimeError, "observed-growth"):
                    hosted.run("growth", [sys.executable, "-c", "import time; time.sleep(30)"], 5,
                               records, profile="full-rebuild", initial_free_bytes=90 * 1024**3)
            self.assertGreater(records[0]["maximum_observed_growth_bytes"], 60 * 1024**3)

    def test_full_profile_rejects_a_different_filesystem_baseline_and_stops_daemon(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            preflight = {"profile": "full-rebuild", "filesystem_device": -1,
                         "initial_free_bytes": 85 * 1024**3, "cpus": 4,
                         "memory_bytes": 16 * 1024**3}
            (output / "resource-preflight.json").write_text(json.dumps(preflight))
            environment = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted"}
            original_stat = os.stat
            with (mock.patch.dict(os.environ, environment), mock.patch.object(hosted, "OUT", output),
                  mock.patch.object(sys, "argv", ["hosted_check.py", "qualify"]),
                  mock.patch.object(hosted, "selected_profile", return_value="full-rebuild"),
                  mock.patch.object(hosted.shutil, "disk_usage", return_value=types.SimpleNamespace(free=85 * 1024**3)),
                  mock.patch.object(hosted.os, "stat", wraps=os.stat) as stat,
                  mock.patch.object(hosted.signal, "signal"), mock.patch.object(hosted, "run") as execute,
                  mock.patch.object(hosted, "stop_job_daemon", return_value={"states": "inactive"}) as cleanup):
                stat.side_effect = lambda path, *args, **kwargs: (
                    types.SimpleNamespace(st_dev=1) if path == "/nix" else original_stat(path, *args, **kwargs))
                with self.assertRaisesRegex(ValueError, "baseline on the Nix filesystem"):
                    hosted.main()
                execute.assert_not_called()
                cleanup.assert_called_once()
            self.assertEqual(json.loads((output / "receipt.json").read_text())["status"], "failed")

    def test_forecast_counts_transitive_builds_and_conservatively_bounds_sizes(self):
        result = hosted.check_plan(
            "these 12 derivations will be built:\n" + forecast_paths(12, ".drv") +
            "these 45 paths will be fetched (15.5 MiB download, 1.0 GiB unpacked):\n" + forecast_paths(45, "-cached"),
            20 * 1024**3,
        )
        self.assertEqual(result["build_derivations"], 12)
        self.assertGreater(result["unpacked_bytes_upper_bound"], 1024**3)
        self.assertGreater(result["download_bytes_upper_bound"], 15.5 * 1024**2)
        self.assertEqual(hosted.check_plan("this derivation will be built:\n" + forecast_paths(1, ".drv"),
                                          20 * 1024**3)["build_derivations"], 1)

    def test_forecast_refuses_excessive_or_unknown_work_before_build(self):
        for log, free in (
            ("these 201 derivations will be built:\n", 20 * 1024**3),
            ("these 3 paths will be fetched (8.0 GiB download, 1.0 GiB unpacked):\n", 20 * 1024**3),
            ("this path will be fetched (1.0 MiB download, 12.0 GiB unpacked):\n", 20 * 1024**3),
            ("this path will be fetched (1.0 MiB download, 1.0 GiB unpacked):\n", 9 * 1024**3),
            ("these 2 paths are not yet known to Nix:\n", 20 * 1024**3),
            ("don't know how to build these paths:\n", 20 * 1024**3),
            ("these 2 paths will be fetched (unknown download, unknown unpacked):\n", 20 * 1024**3),
            ("these 0 derivations will be built:\n", 20 * 1024**3),
            ("arbitrary unfamiliar forecast\n", 20 * 1024**3),
            ("", 20 * 1024**3),
        ):
            with self.subTest(log=log), self.assertRaises((ValueError, KeyError)):
                hosted.check_plan(log, free)

    def test_forecast_rejects_missing_paths_and_requires_verified_cache_hits(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            hosted.check_plan("these 2 derivations will be built:\n" + forecast_paths(1, ".drv"), 20 * 1024**3)
        self.assertEqual(hosted.check_plan("", 20 * 1024**3, verified_present=True)["build_derivations"], 0)
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            hosted.check_plan("not a valid forecast", 20 * 1024**3, verified_present=True)

    def test_forecast_gates_actual_download_build_and_storage_costs(self):
        cases = [
            "these 201 derivations will be built:\n" + forecast_paths(201, ".drv"),
            "this path will be fetched (8.0 GiB download, 1.0 GiB unpacked):\n" + forecast_paths(1, "-cached"),
            "this path will be fetched (1.0 MiB download, 12.0 GiB unpacked):\n" + forecast_paths(1, "-cached"),
        ]
        for log in cases:
            with self.subTest(log=log[:60]), self.assertRaisesRegex(ValueError, "budget"):
                hosted.check_plan(log, 30 * 1024**3)
        with self.assertRaisesRegex(ValueError, "budget"):
            hosted.check_plan("this path will be fetched (1.0 MiB download, 1.0 GiB unpacked):\n" +
                              forecast_paths(1, "-cached"), 9 * 1024**3)

    def test_output_budget_is_enforced_while_draining_even_a_fast_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            with (mock.patch.object(hosted, "ROOT", root), mock.patch.object(hosted, "OUT", root),
                  mock.patch.object(hosted, "MAX_LOG", 128),
                  mock.patch.object(hosted.shutil, "disk_usage", return_value=types.SimpleNamespace(free=100 * 1024**3))):
                with self.assertRaisesRegex(RuntimeError, "bounded log size"):
                    hosted.run("overflow", [sys.executable, "-c", "import sys; sys.stdout.write('x'*65536)"], 5, records)
            self.assertLessEqual((root / "overflow.json").stat().st_size, 128)
            self.assertLessEqual((root / "overflow.log").stat().st_size, 128)
            self.assertIn("bounded log size", records[0]["stop_reason"])

    def test_bootstrap_and_controller_refuse_self_hosted_execution(self):
        environment = dict(os.environ, GITHUB_ACTIONS="true", RUNNER_ENVIRONMENT="self-hosted",
                           RUNNER_OS="Linux", RUNNER_ARCH="X64")
        result = subprocess.run(["bash", str(ROOT / "nix/bootstrap_ci.sh")],
                                env=environment, text=True, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disposable", result.stderr)
        with mock.patch.dict(os.environ, environment), mock.patch.object(sys, "argv", ["hosted_check.py", "qualify"]):
            with self.assertRaisesRegex(ValueError, "hosted CI"):
                hosted.main()

    def test_abort_cleanup_targets_only_the_owned_daemon_and_checks_inactivity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bootstrap.json").write_text('{"version":"2.35.4"}')
            stopped = subprocess.CompletedProcess([], 0)
            inactive = subprocess.CompletedProcess([], 0, stdout="inactive\n")
            with mock.patch.object(hosted, "OUT", root), \
                    mock.patch.object(hosted.subprocess, "run", side_effect=[stopped, inactive, inactive]) as command:
                result = hosted.stop_job_daemon()
            self.assertEqual(result["states"], {"nix-daemon.socket": "inactive", "nix-daemon.service": "inactive"})
            self.assertEqual(command.call_args_list[0].args[0],
                             ["sudo", "systemctl", "stop", "nix-daemon.socket", "nix-daemon.service"])
            with mock.patch.object(hosted, "OUT", root), \
                    mock.patch.object(hosted.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["systemctl"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    hosted.stop_job_daemon()
            active = subprocess.CompletedProcess([], 0, stdout="active\n")
            with mock.patch.object(hosted, "OUT", root), \
                    mock.patch.object(hosted.subprocess, "run", side_effect=[stopped, active]):
                with self.assertRaisesRegex(RuntimeError, "did not stop"):
                    hosted.stop_job_daemon()

    def test_unexpected_monitor_error_still_kills_the_owned_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = subprocess.Popen
            with (mock.patch.object(hosted, "ROOT", root), mock.patch.object(hosted, "OUT", root),
                  mock.patch.object(hosted.subprocess, "Popen", wraps=original) as spawn,
                  mock.patch.object(hosted.shutil, "disk_usage", side_effect=RuntimeError("monitor failed"))):
                with self.assertRaisesRegex(RuntimeError, "monitor failed"):
                    hosted.run("monitor-error", [sys.executable, "-c", "import time; time.sleep(30)"], 1, [])
                self.assertEqual(spawn.call_count, 1)

    def test_tls_context_must_enforce_hostnames_certificates_and_exact_roots(self):
        digest = hashlib.sha256(b"root").hexdigest()
        context = types.SimpleNamespace(verify_mode=ssl.CERT_REQUIRED, check_hostname=True,
                                        get_ca_certs=lambda binary_form: [b"root"])
        self.assertEqual(runtime.check_context(context, {digest}, []), 1)
        for field, value in (("verify_mode", ssl.CERT_NONE), ("check_hostname", False)):
            previous = getattr(context, field)
            setattr(context, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "certificates and hostnames"):
                runtime.check_context(context, {digest}, [])
            setattr(context, field, previous)
        with self.assertRaisesRegex(ValueError, "rejected"):
            runtime.check_context(context, {digest}, [digest])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            runtime.check_context(context, {"other"}, [])

    def test_corrupt_tls_fixture_is_rejected_before_loading_any_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            path.write_bytes(b"bad archive")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runtime.check_spki_distrust(ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT), path, "0" * 64)

    def test_manifest_uses_tracked_inputs_and_rejects_ambiguous_newline_paths(self):
        with mock.patch.object(source_files.subprocess, "check_output", return_value=b"src/a.py\0nix/private\0flake.lock\0z.txt\0"):
            self.assertEqual(source_files.tracked_source(ROOT), ["src/a.py", "z.txt"])
        with mock.patch.object(source_files.subprocess, "check_output", return_value=b"bad\nname\0"):
            with self.assertRaisesRegex(ValueError, "control characters"):
                source_files.tracked_source(ROOT)

    def test_obsolete_source_substitution_fails_before_any_nix_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nix").mkdir()
            (root / "target").write_text("current source pattern")
            (root / "nix/package.nix").write_text(
                "\n  postPatch = ''\nsubstituteInPlace target --replace-fail 'current source' 'replacement'\n  '';\n")
            self.assertEqual(source_files.check_patches(root)[0]["matches"], 1)
            (root / "target").write_text("changed upstream input")
            with self.assertRaisesRegex(ValueError, "Obsolete"):
                source_files.check_patches(root)

    def test_preparation_is_explicitly_separate_from_cpu_qualification(self):
        workflow = (ROOT / ".github/workflows/nix-cpu.yml").read_text()
        self.assertIn("needs.lock-status.outputs.ready != 'true'", workflow)
        self.assertIn("needs.lock-status.outputs.ready == 'true'", workflow)
        self.assertIn("Prepare Nix input lock (no CPU qualification)", workflow)
        self.assertIn("types: [opened, synchronize, reopened, labeled, unlabeled]", workflow)
        self.assertIn("format('ignored-label-{0}', github.run_id)", workflow)
        self.assertIn("fromJSON(needs.lock-status.outputs.job_minutes)", workflow)
        self.assertIn("fromJSON(needs.lock-status.outputs.step_minutes)", workflow)
        self.assertLess(workflow.index("run: python3 nix/hosted_check.py preflight"),
                        workflow.rindex("run: bash nix/bootstrap_ci.sh"))
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("id-token:", workflow)
        self.assertNotIn("pull_request_target", workflow)
        pins = json.loads((ROOT / "nix/security-inputs.json").read_text())
        self.assertEqual(pins["python"]["version"], "3.13.18")
        self.assertTrue(pins["python"]["url"].startswith("https://www.python.org/ftp/python/"))
        self.assertEqual(len(pins["python"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

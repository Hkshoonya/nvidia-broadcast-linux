import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from nvbroadcast.runtime import __main__ as runtime_main
from nvbroadcast.runtime import probe
from nvbroadcast.runtime import process_scope
from nvbroadcast.runtime.process_scope import ProviderProbeCancelled, ProviderProbeScope, run_probe_process
from nvbroadcast.runtime.probe import (
    PROBE_EXPECTED_OUTPUT,
    PROBE_MODEL_SHA256,
    ProbeProvider,
    RuntimeProbeResult,
    clear_runtime_probe_cache,
    probe_execution_provider,
    probe_model_bytes,
    probe_observation_problems,
)


class RuntimeProbeTests(unittest.TestCase):
    def test_cuda_numeric_out_of_memory_code_is_retryable(self):
        for detail in (
            "CUDA error 2: allocation failed while creating the provider",
            "CUDA failure 2: allocation failed while creating the provider",
            "CUDNN_STATUS_ALLOC_FAILED",
            "cudaErrorMemoryAllocation",
        ):
            with self.subTest(detail=detail):
                result = RuntimeProbeResult.failure(
                    ProbeProvider.CUDA,
                    detail,
                )

                self.assertTrue(result.resource_exhausted)
                self.assertTrue(result.temporarily_unavailable)

    def test_cuda_error_with_code_starting_in_two_is_not_memory_pressure(self):
        for wording in ("error", "failure"):
            with self.subTest(wording=wording):
                result = RuntimeProbeResult.failure(
                    ProbeProvider.CUDA,
                    f"CUDA {wording} 209: no kernel image is available",
                )

                self.assertFalse(result.resource_exhausted)
                self.assertFalse(result.temporarily_unavailable)

    def tearDown(self):
        clear_runtime_probe_cache()

    def test_embedded_probe_model_matches_pinned_digest(self):
        model = probe_model_bytes()
        self.assertEqual(len(model), 144)
        self.assertEqual(hashlib.sha256(model).hexdigest(), PROBE_MODEL_SHA256)

    def test_valid_provider_observation_is_accepted(self):
        self.assertEqual(
            probe_observation_problems(
                ProbeProvider.CUDA,
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                ["CUDAExecutionProvider"],
                PROBE_EXPECTED_OUTPUT,
                [1, 4],
            ),
            [],
        )

    def test_cpu_fallback_is_rejected_even_with_cuda_registered(self):
        problems = probe_observation_problems(
            ProbeProvider.CUDA,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
            PROBE_EXPECTED_OUTPUT,
            [1, 4],
        )
        self.assertTrue(any("exclusively on CUDAExecutionProvider" in p for p in problems))

    def test_wrong_probe_output_and_shape_are_rejected(self):
        problems = probe_observation_problems(
            ProbeProvider.CPU,
            ["CPUExecutionProvider"],
            ["CPUExecutionProvider"],
            [0.0, 0.0],
            [1, 2],
        )
        self.assertTrue(any("output shape" in problem for problem in problems))
        self.assertTrue(any("probe output was" in problem for problem in problems))

    def test_parent_runs_isolated_interpreter_and_parses_result(self):
        child_result = RuntimeProbeResult(
            provider=ProbeProvider.CUDA,
            success=True,
            available_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            registered_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            executed_providers=("CUDAExecutionProvider",),
            output=PROBE_EXPECTED_OUTPUT,
            output_shape=(1, 4),
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                probe._RESULT_PREFIX
                + json.dumps(child_result.to_payload())
                + "\n"
            ),
            stderr="",
        )
        trusted_roots = (
            "/nix/store/numpy/lib/python/site-packages",
            "/nix/store/onnxruntime/lib/python/site-packages",
        )
        with mock.patch.object(
            probe,
            "_trusted_child_import_roots",
            return_value=trusted_roots,
        ), mock.patch.object(
            probe.subprocess, "run", return_value=completed
        ) as run:
            result = probe_execution_provider(
                ProbeProvider.CUDA, use_cache=False
            )

        self.assertTrue(result.success)
        command = run.call_args.args[0]
        probe_path = Path(probe.__file__).resolve()
        self.assertEqual(command[:3], [sys.executable, "-I", "-c"])
        self.assertIn("runpy.run_path", command[3])
        self.assertEqual(command[4], str(probe_path.parents[2]))
        self.assertEqual(command[5], str(probe_path))
        self.assertEqual(tuple(json.loads(command[6])), trusted_roots)
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
        self.assertNotIn("PYTHONHOME", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["timeout"], 120.0)

    def test_provider_defaults_allow_cold_gpu_execution_and_preserve_overrides(self):
        for provider, explicit, expected in (
            (ProbeProvider.CPU, None, 30.0),
            (ProbeProvider.CUDA, None, 120.0),
            (ProbeProvider.TENSORRT, None, 120.0),
            (ProbeProvider.CUDA, 3.0, 3.0),
            (ProbeProvider.TENSORRT, 5.0, 5.0),
        ):
            with self.subTest(provider=provider, timeout=explicit), mock.patch.object(
                probe, "_run_provider_probe"
            ) as run:
                probe_execution_provider(provider, timeout=explicit, use_cache=False)
                self.assertEqual(run.call_args.args[2], expected)

    def test_trusted_import_roots_support_nix_without_forwarding_arbitrary_paths(self):
        package_root = Path("/nix/store/nvbroadcast/lib/python/site-packages")
        nix_numpy = "/nix/store/numpy/lib/python/site-packages"
        nix_ort = "/nix/store/onnxruntime/lib/python/site-packages"
        with mock.patch.object(
            probe.sys,
            "path",
            ["", "/tmp/shadow", str(package_root), nix_numpy, nix_ort],
        ), mock.patch.object(
            probe.sys, "prefix", "/nix/store/python"
        ), mock.patch.object(
            probe.sys, "exec_prefix", "/nix/store/python"
        ), mock.patch.object(
            probe.sys, "base_prefix", "/nix/store/python"
        ), mock.patch.object(
            probe.sys, "base_exec_prefix", "/nix/store/python"
        ), mock.patch.object(
            probe.sys, "executable", "/nix/store/python/bin/python"
        ):
            roots = probe._trusted_child_import_roots(package_root)

        self.assertEqual(roots, (nix_numpy, nix_ort))
        self.assertNotIn("/tmp/shadow", roots)

    def test_nix_store_is_not_trusted_by_a_non_nix_interpreter(self):
        package_root = Path("/opt/nvbroadcast/lib/python/site-packages")
        with mock.patch.object(
            probe.sys,
            "path",
            [
                str(package_root),
                "/nix/store/untrusted/lib/python/site-packages",
                "/opt/python/lib/python/site-packages",
            ],
        ), mock.patch.object(
            probe.sys, "prefix", "/opt/python"
        ), mock.patch.object(
            probe.sys, "exec_prefix", "/opt/python"
        ), mock.patch.object(
            probe.sys, "base_prefix", "/opt/python"
        ), mock.patch.object(
            probe.sys, "base_exec_prefix", "/opt/python"
        ), mock.patch.object(
            probe.sys, "executable", "/opt/python/bin/python"
        ):
            roots = probe._trusted_child_import_roots(package_root)

        self.assertEqual(roots, ("/opt/python/lib/python/site-packages",))

    def test_native_stderr_and_child_error_are_retained(self):
        child_result = RuntimeProbeResult.failure(
            ProbeProvider.TENSORRT,
            "requested provider did not execute",
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=probe._RESULT_PREFIX + json.dumps(child_result.to_payload()),
            stderr="Failed to load libonnxruntime_providers_tensorrt.so: libnvinfer.so.10",
        )
        with mock.patch.object(probe.subprocess, "run", return_value=completed):
            result = probe_execution_provider(
                ProbeProvider.TENSORRT, use_cache=False
            )

        self.assertFalse(result.success)
        self.assertIn("requested provider did not execute", result.failure_detail)
        self.assertIn("libnvinfer.so.10", result.failure_detail)

    def test_result_marker_survives_oversized_child_diagnostics(self):
        child_result = RuntimeProbeResult(
            provider=ProbeProvider.CPU,
            success=True,
            registered_providers=("CPUExecutionProvider",),
            executed_providers=("CPUExecutionProvider",),
            output=PROBE_EXPECTED_OUTPUT,
            output_shape=(1, 4),
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                ("verbose output" * 6000)
                + "\n"
                + probe._RESULT_PREFIX
                + json.dumps(child_result.to_payload())
            ),
            stderr="",
        )
        with mock.patch.object(probe.subprocess, "run", return_value=completed):
            result = probe_execution_provider(
                ProbeProvider.CPU, use_cache=False
            )

        self.assertTrue(result.success)
        self.assertIn("characters omitted", result.diagnostics)

    def test_parent_rejects_success_payload_without_execution_evidence(self):
        child_result = RuntimeProbeResult(
            provider=ProbeProvider.CUDA,
            success=True,
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=probe._RESULT_PREFIX + json.dumps(child_result.to_payload()),
            stderr="",
        )
        with mock.patch.object(probe.subprocess, "run", return_value=completed):
            result = probe_execution_provider(
                ProbeProvider.CUDA, use_cache=False
            )

        self.assertFalse(result.success)
        self.assertIn("invalid successful probe result", result.failure_detail)
        self.assertIn("did not execute exclusively", result.failure_detail)

    def test_isolated_child_ignores_shadow_package_and_pythonpath(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            shadow_root = Path(tmp)
            shadow_runtime = shadow_root / "nvbroadcast" / "runtime"
            shadow_runtime.mkdir(parents=True)
            (shadow_root / "nvbroadcast" / "__init__.py").write_text("")
            (shadow_runtime / "__init__.py").write_text("")
            (shadow_runtime / "probe.py").write_text(
                "raise RuntimeError('shadow probe executed')\n"
            )
            try:
                os.chdir(shadow_root)
                with mock.patch.dict(
                    os.environ, {"PYTHONPATH": str(shadow_root)}, clear=False
                ):
                    result = probe_execution_provider(
                        ProbeProvider.CPU, timeout=15, use_cache=False
                    )
            finally:
                os.chdir(original_cwd)

        self.assertTrue(result.success, result.failure_detail)
        self.assertEqual(result.executed_providers, ("CPUExecutionProvider",))

    def test_child_crash_is_reported_with_exit_code(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=-11, stdout="", stderr="native crash"
        )
        with mock.patch.object(probe.subprocess, "run", return_value=completed):
            result = probe_execution_provider(
                ProbeProvider.CUDA, use_cache=False
            )

        self.assertFalse(result.success)
        self.assertIn("exited with code -11", result.failure_detail)
        self.assertIn("native crash", result.failure_detail)

    def test_probe_timeout_preserves_captured_output(self):
        timeout = subprocess.TimeoutExpired(
            cmd=[sys.executable],
            timeout=3,
            output="partial output",
            stderr="provider stalled",
        )
        with mock.patch.object(probe.subprocess, "run", side_effect=timeout):
            result = probe_execution_provider(
                ProbeProvider.CUDA, timeout=3, use_cache=False
            )

        self.assertFalse(result.success)
        self.assertIn("timed out after 3 seconds", result.failure_detail)
        self.assertIn("partial output", result.failure_detail)
        self.assertIn("provider stalled", result.failure_detail)

    def test_root_provider_error_is_promoted_from_fallback_chain(self):
        provider_error = RuntimeError("CUDA device initialization failed")
        fallback_error = ValueError("CPU fallback was disabled")
        fallback_error.__cause__ = provider_error

        self.assertIs(probe._root_exception(fallback_error), provider_error)

    def test_cuda_memory_pressure_is_classified_as_temporary_resource_exhaustion(self):
        for detail in (
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDA_ERROR_OUT_OF_MEMORY",
            "BFCArena::AllocateRawInternal: Available memory of 1024 is smaller than requested bytes of 2048",
        ):
            with self.subTest(detail=detail):
                result = RuntimeProbeResult.failure(ProbeProvider.CUDA, detail)
                self.assertTrue(result.resource_exhausted)
                self.assertTrue(result.temporarily_unavailable)

    def test_busy_or_timed_out_gpu_is_retryable_without_claiming_low_memory(self):
        for detail in (
            "provider execution probe timed out after 30 seconds",
            "cudaErrorDevicesUnavailable: CUDA-capable devices are busy or unavailable",
        ):
            with self.subTest(detail=detail):
                result = RuntimeProbeResult.failure(ProbeProvider.CUDA, detail)
                self.assertFalse(result.resource_exhausted)
                self.assertTrue(result.temporarily_unavailable)

    def test_cpu_probe_timeout_is_not_a_retryable_gpu_capacity_failure(self):
        result = RuntimeProbeResult.failure(
            ProbeProvider.CPU,
            "provider execution probe timed out after 30 seconds",
        )

        self.assertFalse(result.temporarily_unavailable)

    def test_missing_cuda_library_is_not_classified_as_resource_exhaustion(self):
        result = RuntimeProbeResult.failure(
            ProbeProvider.CUDA,
            "Failed to load libonnxruntime_providers_cuda.so: libcudnn.so.9 not found",
        )

        self.assertFalse(result.resource_exhausted)
        self.assertFalse(result.temporarily_unavailable)

    def test_probe_results_are_cached_until_environment_mutation(self):
        success = RuntimeProbeResult(
            provider=ProbeProvider.CPU,
            success=True,
        )
        with mock.patch.object(
            probe, "_run_provider_probe", return_value=success
        ) as run:
            probe_execution_provider(ProbeProvider.CPU)
            probe_execution_provider(ProbeProvider.CPU)
            self.assertEqual(run.call_count, 1)
            clear_runtime_probe_cache()
            probe_execution_provider(ProbeProvider.CPU)
            self.assertEqual(run.call_count, 2)

    def test_provider_probe_cache_is_separate_per_device(self):
        success = RuntimeProbeResult(
            provider=ProbeProvider.CUDA,
            success=True,
        )
        clear_runtime_probe_cache()
        with mock.patch.object(
            probe, "_run_provider_probe", return_value=success
        ) as run:
            probe_execution_provider(ProbeProvider.CUDA, device_id=0)
            probe_execution_provider(ProbeProvider.CUDA, device_id=1)
            probe_execution_provider(ProbeProvider.CUDA, device_id=0)

        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[1], 0)
        self.assertEqual(run.call_args_list[1].args[1], 1)

    def test_cache_clear_during_inflight_probe_cannot_restore_stale_result(self):
        started = threading.Event()
        release = threading.Event()
        stale = RuntimeProbeResult.failure(ProbeProvider.CUDA, "stale busy result")
        fresh = RuntimeProbeResult(provider=ProbeProvider.CUDA, success=True)

        def execute(*_args):
            if not started.is_set():
                started.set()
                self.assertTrue(release.wait(1.0))
                return stale
            return fresh

        clear_runtime_probe_cache()
        result = []
        with mock.patch.object(probe, "_run_provider_probe", side_effect=execute) as run:
            worker = threading.Thread(
                target=lambda: result.append(
                    probe_execution_provider(ProbeProvider.CUDA)
                )
            )
            worker.start()
            self.assertTrue(started.wait(1.0))
            clear_runtime_probe_cache()
            release.set()
            worker.join(1.0)

            self.assertIs(result[0], stale)
            self.assertIs(probe_execution_provider(ProbeProvider.CUDA), fresh)
            self.assertEqual(run.call_count, 2)

    def test_real_cpu_provider_executes_in_fresh_process(self):
        result = probe_execution_provider(
            ProbeProvider.CPU, timeout=15, use_cache=False
        )
        self.assertTrue(result.success, result.failure_detail)
        self.assertEqual(result.executed_providers, ("CPUExecutionProvider",))
        self.assertEqual(result.output, PROBE_EXPECTED_OUTPUT)

    def test_runtime_cli_surfaces_complete_probe_failure(self):
        failed = RuntimeProbeResult.failure(
            ProbeProvider.CUDA,
            "CUDA session creation failed",
            diagnostics="libcudnn.so.9 could not be loaded",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(runtime_main, "validate_current_runtime", return_value=[]), \
             mock.patch.object(runtime_main, "probe_execution_provider", return_value=failed), \
             mock.patch.object(sys, "argv", ["nvbroadcast.runtime", "--variant", "cuda"]), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            return_code = runtime_main.main()

        self.assertEqual(return_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("CUDA session creation failed", stderr.getvalue())
        self.assertIn("libcudnn.so.9", stderr.getvalue())

    def test_cpu_named_cuda_still_requires_successful_execution(self):
        runtime = mock.Mock()
        runtime.get_available_providers.return_value = [
            "CPUExecutionProvider", "CUDAExecutionProvider"
        ]
        failed = RuntimeProbeResult.failure(
            ProbeProvider.CUDA, "CUDA session creation failed"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch(
                "nvbroadcast.runtime.variants.current_distribution_inventory",
                return_value={"onnxruntime": ("1.29.0",)},
            ),
            mock.patch.dict(sys.modules, {"onnxruntime": runtime}),
            mock.patch.object(
                runtime_main, "probe_execution_provider", return_value=failed
            ) as execute,
            mock.patch.object(sys, "argv", ["nvbroadcast.runtime", "--variant", "cuda"]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            return_code = runtime_main.main()

        execute.assert_called_once_with(ProbeProvider.CUDA, use_cache=False)
        self.assertEqual(return_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("CUDA session creation failed", stderr.getvalue())


@unittest.skipUnless(os.name == "posix", "private process groups require POSIX")
class ProviderProbeScopeTests(unittest.TestCase):
    @staticmethod
    def _alive(pid):
        try:
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z":
                return False
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, FileNotFoundError):
            return False

    def _wait(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("controlled child did not reach the expected state")

    @staticmethod
    def _start(scope, script, *arguments):
        outcome = {}

        def worker():
            try:
                with scope.activate():
                    outcome["result"] = run_probe_process(
                        [sys.executable, "-I", "-c", script, *map(str, arguments)],
                        env={"PATH": os.defpath}, stdin=subprocess.DEVNULL,
                        check=False, capture_output=True, text=True, timeout=60,
                    )
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread, outcome

    def test_close_reaps_owned_parent_kills_inherited_pipe_descendant_and_is_idempotent(self):
        script = (
            "import json,os,pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)']); "
            "marker=pathlib.Path(sys.argv[1]); temporary=marker.with_suffix('.tmp'); "
            "temporary.write_text(json.dumps([os.getpid(),child.pid])); temporary.replace(marker)"
        )
        scope = ProviderProbeScope()
        unrelated = subprocess.Popen(
            [sys.executable, "-I", "-c", "import time;time.sleep(60)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                marker = Path(directory) / "owned-pids.json"
                thread, outcome = self._start(scope, script, marker)
                self._wait(marker.exists)
                parent, descendant = json.loads(marker.read_text())
                begin = time.monotonic()
                self.assertTrue(scope.close())
                self.assertTrue(scope.close())
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertLess(time.monotonic() - begin, 3)
                self.assertIsInstance(outcome.get("error"), ProviderProbeCancelled)
                self._wait(lambda: not self._alive(parent) and not self._alive(descendant))
                self.assertIsNone(unrelated.poll())
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    with scope.activate():
                        self.fail("closed owner accepted a new probe")
        finally:
            scope.close()
            unrelated.kill()
            unrelated.wait(timeout=2)

    def test_normal_return_also_disposes_of_detached_output_descendant(self):
        script = (
            "import subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)'],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(child.pid,flush=True)"
        )
        scope = ProviderProbeScope()
        try:
            thread, outcome = self._start(scope, script)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", outcome)
            self.assertEqual(outcome["result"].returncode, 0)
            descendant = int(outcome["result"].stdout.strip())
            self._wait(lambda: not self._alive(descendant))
        finally:
            scope.close()

    def test_dialog_scope_can_close_independently_and_app_close_cancels_nested_children(self):
        parent = ProviderProbeScope()
        script = (
            "import pathlib,sys,time; marker=pathlib.Path(sys.argv[1]); "
            "marker.write_text('ready'); time.sleep(60)"
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                first = parent.child()
                marker = Path(directory) / "first"
                thread, outcome = self._start(first, script, marker)
                self._wait(marker.exists)
                self.assertTrue(first.close())
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                with parent.activate():
                    result = run_probe_process(
                        [sys.executable, "-I", "-c", "print('parent still open')"],
                        env={"PATH": os.defpath}, stdin=subprocess.DEVNULL,
                        check=False, capture_output=True, text=True, timeout=2,
                    )
                self.assertEqual(result.stdout.strip(), "parent still open")
                second = parent.child()
                marker = Path(directory) / "second"
                nested_script = (
                    "import json,os,pathlib,subprocess,sys,time; "
                    "child=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)']); "
                    "marker=pathlib.Path(sys.argv[1]); temporary=marker.with_suffix('.tmp'); "
                    "temporary.write_text(json.dumps([os.getpid(),child.pid])); temporary.replace(marker); "
                    "time.sleep(60)"
                )
                thread, outcome = self._start(second, nested_script, marker)
                self._wait(marker.exists)
                nested_pids = json.loads(marker.read_text())
                self.assertTrue(parent.close())
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertIsInstance(outcome.get("error"), ProviderProbeCancelled)
                self._wait(lambda: not any(self._alive(pid) for pid in nested_pids))
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    with second.activate():
                        self.fail("application close left a dialog owner open")
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    parent.child()
        finally:
            parent.close()

    def test_cancellation_is_not_cached_and_open_owner_timeout_remains_a_real_failure(self):
        clear_runtime_probe_cache()
        scope = ProviderProbeScope()
        outcome = {}
        try:
            with tempfile.TemporaryDirectory() as directory:
                marker = Path(directory) / "probe-ready"
                bootstrap = (
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ready'); "
                    "print('partial',flush=True); import time; time.sleep(60)"
                )

                def worker():
                    try:
                        with scope.activate():
                            outcome["result"] = probe_execution_provider(ProbeProvider.CUDA)
                    except BaseException as error:
                        outcome["error"] = error

                with mock.patch.object(probe, "_CHILD_BOOTSTRAP", bootstrap):
                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    self._wait(marker.exists)
                    self.assertTrue(scope.close())
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())
                    self.assertIsInstance(outcome.get("error"), ProviderProbeCancelled)
                    fresh = RuntimeProbeResult.failure(ProbeProvider.CUDA, "fresh execution")
                    with mock.patch.object(probe, "_run_provider_probe", return_value=fresh) as run:
                        self.assertIs(probe_execution_provider(ProbeProvider.CUDA), fresh)
                        run.assert_called_once()
                    open_scope = ProviderProbeScope()
                    try:
                        with open_scope.activate():
                            timed_out = probe_execution_provider(
                                ProbeProvider.CUDA, timeout=0.5, use_cache=False,
                            )
                        self.assertFalse(timed_out.success)
                        self.assertTrue(timed_out.temporarily_unavailable)
                        self.assertIn("partial", timed_out.failure_detail)
                        self.assertIn("timed out after 0.5 seconds", timed_out.failure_detail)
                    finally:
                        open_scope.close()
        finally:
            scope.close()
            clear_runtime_probe_cache()

    def test_owned_timeout_preserves_partial_output_and_reaps_child(self):
        scope = ProviderProbeScope()
        try:
            with scope.activate(), self.assertRaises(subprocess.TimeoutExpired) as raised:
                run_probe_process(
                    [sys.executable, "-I", "-c", "import time;print('partial',flush=True);time.sleep(60)"],
                    env={"PATH": os.defpath}, stdin=subprocess.DEVNULL,
                    check=False, capture_output=True, text=True, timeout=0.5,
                )
            self.assertIn("partial", raised.exception.stdout)
            self.assertTrue(scope.close())
        finally:
            scope.close()

    def test_spawn_and_close_cannot_leave_a_late_child_unowned(self):
        scope = ProviderProbeScope()
        spawned, release = threading.Event(), threading.Event()
        children = []
        original = subprocess.Popen

        def delayed_registration(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            spawned.set()
            if not release.wait(3):
                child.kill()
                raise AssertionError("test did not release owned spawn")
            return child

        closed = []
        try:
            with mock.patch.object(process_scope.subprocess, "Popen", delayed_registration):
                thread, outcome = self._start(scope, "import time;time.sleep(60)")
                self.assertTrue(spawned.wait(3))
                closer = threading.Thread(target=lambda: closed.append(scope.close()), daemon=True)
                closer.start()
                release.set()
                closer.join(timeout=3)
                thread.join(timeout=3)
                self.assertFalse(closer.is_alive())
                self.assertFalse(thread.is_alive())
                self.assertEqual(closed, [True])
                self.assertIsInstance(outcome.get("error"), ProviderProbeCancelled)
                self.assertIsNotNone(children[0].poll())
        finally:
            release.set()
            scope.close()


if __name__ == "__main__":
    unittest.main()

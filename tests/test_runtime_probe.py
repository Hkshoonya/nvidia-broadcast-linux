import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from nvbroadcast.runtime import __main__ as runtime_main
from nvbroadcast.runtime import probe
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
        with mock.patch.object(probe.subprocess, "run", return_value=completed) as run:
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
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
        self.assertNotIn("PYTHONHOME", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["timeout"], 30.0)

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


if __name__ == "__main__":
    unittest.main()

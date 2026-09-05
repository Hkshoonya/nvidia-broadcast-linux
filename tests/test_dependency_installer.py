import unittest
import sys
import types
from unittest import mock


try:
    import gi  # noqa: F401
except Exception:
    gi = types.ModuleType("gi")
    repository = types.ModuleType("gi.repository")

    class _DummyGObjectModule:
        class Object:
            pass

        class SignalFlags:
            RUN_FIRST = 0

    class _DummyGLibModule:
        @staticmethod
        def idle_add(func, *args, **kwargs):
            return func(*args, **kwargs)

    def _require_version(*_args, **_kwargs):
        return None

    gi.require_version = _require_version
    repository.GObject = _DummyGObjectModule
    repository.GLib = _DummyGLibModule
    gi.repository = repository
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repository

from nvbroadcast.core import dependency_installer
from nvbroadcast.runtime.probe import ProbeProvider, RuntimeProbeResult


class DependencyInstallerTests(unittest.TestCase):
    def test_cuda_mode_runtime_requires_cupy_and_cuda_provider(self):
        with mock.patch.object(dependency_installer, "_has_cupy", return_value=True), \
             mock.patch.object(dependency_installer, "has_cuda_inference_runtime", return_value=False):
            self.assertFalse(dependency_installer._has_cuda_mode_runtime())

        with mock.patch.object(dependency_installer, "_has_cupy", return_value=True), \
             mock.patch.object(dependency_installer, "has_cuda_inference_runtime", return_value=True):
            self.assertTrue(dependency_installer._has_cuda_mode_runtime())

    def test_cuda_support_install_spec_does_not_change_runtime_owner(self):
        spec = dependency_installer.PACKAGE_SPECS["cupy"]
        install_args = spec["install_args"]
        self.assertIn("--upgrade", install_args)
        self.assertIn("cupy-cuda12x>=14.1.1,<15", install_args)
        self.assertNotIn("onnxruntime-gpu==1.24.4", install_args)
        self.assertNotIn("onnxruntime", install_args)
        self.assertIn("nvidia-cudnn-cu12", install_args)
        self.assertIn("nvidia-cuda-nvrtc-cu12", install_args)
        self.assertIn("already owned by the CUDA ONNX Runtime variant", spec["summary"])
        self.assertNotIn("onnxruntime", spec["help"])

    def test_cuda_support_requires_cuda_runtime_owner(self):
        with mock.patch.object(dependency_installer, "supports_linux_gpu_stack", return_value=True), \
             mock.patch.object(dependency_installer, "detect_runtime_variant", return_value=dependency_installer.RuntimeVariant.CPU):
            self.assertFalse(dependency_installer._supports_cuda_runtime())

        with mock.patch.object(dependency_installer, "supports_linux_gpu_stack", return_value=True), \
             mock.patch.object(dependency_installer, "detect_runtime_variant", return_value=dependency_installer.RuntimeVariant.CUDA):
            self.assertTrue(dependency_installer._supports_cuda_runtime())

    def test_cpu_source_runtime_directs_user_to_source_installer(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(dependency_installer, "supports_linux_gpu_stack", return_value=True), \
             mock.patch.object(dependency_installer, "detect_runtime_variant", return_value=dependency_installer.RuntimeVariant.CPU), \
             mock.patch.object(dependency_installer.sys, "prefix", "/home/user/project/.venv"):
            reason = installer.install_block_reason("cupy")

        self.assertIsNotNone(reason)
        self.assertIn("./install.sh --runtime cuda", reason)
        self.assertIn("user-owned source environment", reason)
        self.assertNotIn("system package manager", reason)

    def test_cpu_native_runtime_directs_user_to_package_manager(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(dependency_installer, "supports_linux_gpu_stack", return_value=True), \
             mock.patch.object(dependency_installer, "detect_runtime_variant", return_value=dependency_installer.RuntimeVariant.CPU), \
             mock.patch.object(dependency_installer.sys, "prefix", "/opt/nvbroadcast/.venv"):
            reason = installer.install_block_reason("cupy")

        self.assertIsNotNone(reason)
        self.assertIn("system package manager", reason)
        self.assertIn("nvidia-smi", reason)
        self.assertNotIn("./install.sh", reason)

    def test_cuda_runtime_reports_unsupported_platform_before_switch_guidance(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(dependency_installer, "supports_linux_gpu_stack", return_value=False), \
             mock.patch.object(dependency_installer, "detect_runtime_variant", return_value=dependency_installer.RuntimeVariant.CPU):
            reason = installer.install_block_reason("cupy")

        self.assertEqual(
            reason,
            "CUDA modes are currently available only on Linux x86_64.",
        )

    def test_cupy_verification_preloads_component_wheel_runtime(self):
        fake_array = mock.MagicMock()
        fake_array.__mul__.return_value.astype.return_value = object()
        fake_cupy = types.SimpleNamespace(
            asarray=mock.Mock(return_value=fake_array),
            float32=object(),
        )

        with mock.patch.dict(sys.modules, {"cupy": fake_cupy}), \
             mock.patch.object(
                 dependency_installer, "preload_nvidia_runtime_libs"
             ) as preload:
            self.assertTrue(dependency_installer._verify_cupy())

        preload.assert_called_once_with()

    def test_cuda_verification_retains_provider_probe_error(self):
        failed = RuntimeProbeResult.failure(
            ProbeProvider.CUDA,
            "CUDA session creation failed",
            diagnostics="libcudnn.so.9 could not be loaded",
        )
        with mock.patch.object(
            dependency_installer, "_verify_cupy_result", return_value=(True, "")
        ), mock.patch.object(
            dependency_installer,
            "cuda_inference_probe_result",
            return_value=failed,
        ):
            success, detail = dependency_installer._verify_cuda_mode_runtime_result()

        self.assertFalse(success)
        self.assertIn("CUDA session creation failed", detail)
        self.assertIn("libcudnn.so.9", detail)

    def test_tensorrt_verification_retains_native_loader_error(self):
        failed = RuntimeProbeResult.failure(
            ProbeProvider.TENSORRT,
            "TensorRT provider did not execute",
            diagnostics="libnvinfer.so.10 could not be loaded",
        )
        with mock.patch.object(
            dependency_installer,
            "tensorrt_inference_probe_result",
            return_value=failed,
        ):
            success, detail = dependency_installer._verify_tensorrt_runtime_result()

        self.assertFalse(success)
        self.assertIn("TensorRT provider did not execute", detail)
        self.assertIn("libnvinfer.so.10", detail)

    def test_gpu_runtime_install_requires_fresh_app_process(self):
        installer = dependency_installer.DependencyInstaller()
        installer._mark_restart_pending("tensorrt")
        with mock.patch.object(installer, "_emit_completed"):
            installer._finish_job("tensorrt", True, "installed")

        self.assertTrue(installer.restart_pending("tensorrt"))
        self.assertIn(
            "Restart NVBroadcast",
            installer.install_block_reason("tensorrt"),
        )
        self.assertIn(
            "Restart NVBroadcast",
            installer.unsupported_reason_for_mode("zeus"),
        )
        self.assertFalse(installer.restart_pending("whisper"))

    def test_noop_bundle_does_not_require_restart(self):
        installer = dependency_installer.DependencyInstaller()
        completed = []
        with mock.patch.object(
            installer, "is_available", return_value=True
        ), mock.patch.object(
            installer, "_emit_completed", side_effect=lambda *args: completed.append(args)
        ), mock.patch.object(
            dependency_installer.GLib,
            "idle_add",
            side_effect=lambda callback, *args: callback(*args),
        ):
            installer._run_install("premium_gpu_stack")

        self.assertFalse(installer.restart_pending("premium_gpu_stack"))
        self.assertIn("installed and ready", completed[0][2])

    def test_failed_verification_requires_restart_after_runtime_mutation(self):
        installer = dependency_installer.DependencyInstaller()
        proc = mock.Mock(stdout=[], wait=mock.Mock(return_value=0))
        completed = []
        with mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(installer, "is_supported", return_value=True), \
             mock.patch.object(dependency_installer, "_runtime_install_block_reason", return_value=None), \
             mock.patch.object(installer, "_emit_progress", return_value=False), \
             mock.patch.dict(
                 dependency_installer.PACKAGE_SPECS["tensorrt"],
                 {"verify": lambda: (False, "provider load failed")},
             ), \
             mock.patch.object(dependency_installer.subprocess, "Popen", return_value=proc), \
             mock.patch.object(
                 installer,
                 "_emit_completed",
                 side_effect=lambda *args: completed.append(args),
             ), \
             mock.patch.object(
                 dependency_installer.GLib,
                 "idle_add",
                 side_effect=lambda callback, *args: callback(*args),
             ):
            installer._run_install("tensorrt")

        self.assertTrue(installer.restart_pending("tensorrt"))
        self.assertFalse(completed[0][1])
        self.assertIn("Restart NVBroadcast", completed[0][2])

    def test_failed_gpu_pip_attempt_requires_restart(self):
        installer = dependency_installer.DependencyInstaller()
        proc = mock.Mock(
            stdout=["ERROR: install failed\n"],
            wait=mock.Mock(return_value=1),
        )
        with mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(installer, "is_supported", return_value=True), \
             mock.patch.object(dependency_installer, "_runtime_install_block_reason", return_value=None), \
             mock.patch.object(installer, "_emit_progress", return_value=False), \
             mock.patch.object(dependency_installer.subprocess, "Popen", return_value=proc):
            success, _message = installer._install_single(
                "tensorrt", "tensorrt"
            )

        self.assertFalse(success)
        self.assertTrue(installer.restart_pending("tensorrt"))

    def test_failed_probe_detail_reaches_install_completion_message(self):
        installer = dependency_installer.DependencyInstaller()
        proc = mock.Mock(stdout=[], wait=mock.Mock(return_value=0))
        detail = "Failed to load TensorRT provider: libnvinfer.so.10"
        with mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(installer, "is_supported", return_value=True), \
             mock.patch.object(dependency_installer, "_runtime_install_block_reason", return_value=None), \
             mock.patch.object(installer, "_emit_progress", return_value=False), \
             mock.patch.dict(
                 dependency_installer.PACKAGE_SPECS["tensorrt"],
                 {"verify": lambda: (False, detail)},
             ), \
             mock.patch.object(dependency_installer.subprocess, "Popen", return_value=proc), \
             mock.patch.object(dependency_installer, "clear_runtime_probe_cache") as clear_cache:
            success, message = installer._install_single("tensorrt", "tensorrt")

        self.assertFalse(success)
        self.assertIn("Probe details", message)
        self.assertIn("libnvinfer.so.10", message)
        clear_cache.assert_called_once_with()

    def test_snap_cuda_modes_report_package_limitation_when_runtime_missing(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.dict(dependency_installer.os.environ, {"SNAP": "/snap/nvbroadcast/current"}, clear=False), \
             mock.patch.object(dependency_installer, "_has_cuda_mode_runtime", return_value=False), \
             mock.patch.object(dependency_installer, "IS_LINUX", True), \
             mock.patch.object(dependency_installer, "IS_ARM64", False):
            reason = installer.unsupported_reason_for_mode("doczeus")

        self.assertIsNotNone(reason)
        self.assertIn("Snap build", reason)
        self.assertIn("latest Snap", reason)
        self.assertIn(".deb", reason)

    def test_immutable_runtime_error_includes_provider_probe_diagnostic(self):
        installer = dependency_installer.DependencyInstaller()
        failed = RuntimeProbeResult.failure(
            ProbeProvider.CUDA,
            "CUDA provider session creation failed",
            diagnostics="libcudnn.so.9 could not be loaded",
        )
        with mock.patch.dict(
            dependency_installer.os.environ,
            {"SNAP": "/snap/nvbroadcast/current"},
            clear=False,
        ), mock.patch.object(
            dependency_installer, "_has_cuda_mode_runtime", return_value=False
        ), mock.patch.object(
            dependency_installer,
            "cuda_inference_probe_result",
            return_value=failed,
        ), mock.patch.object(
            dependency_installer, "IS_LINUX", True
        ), mock.patch.object(
            dependency_installer, "IS_ARM64", False
        ):
            reason = installer.unsupported_reason_for_mode("doczeus")

        self.assertIn("Provider probe details", reason)
        self.assertIn("CUDA provider session creation failed", reason)
        self.assertIn("libcudnn.so.9", reason)

    def test_snap_tensorrt_mode_does_not_offer_runtime_install(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.dict(dependency_installer.os.environ, {"SNAP": "/snap/nvbroadcast/current"}, clear=False), \
             mock.patch.object(dependency_installer, "_has_cuda_mode_runtime", return_value=True), \
             mock.patch.object(dependency_installer, "has_tensorrt_runtime", return_value=False), \
             mock.patch.object(dependency_installer, "supports_tensorrt_python", return_value=True), \
             mock.patch.object(dependency_installer, "IS_LINUX", True), \
             mock.patch.object(dependency_installer, "IS_ARM64", False):
            reason = installer.unsupported_reason_for_mode("zeus")
            install_key = installer.install_key_for_mode("zeus")

        self.assertIsNotNone(reason)
        self.assertIn("unavailable in this Snap", reason)
        self.assertIsNone(install_key)

    def test_snap_installer_rejects_direct_runtime_mutation(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.dict(dependency_installer.os.environ, {"SNAP": "/snap/nvbroadcast/current"}, clear=False), \
             mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(installer, "is_supported", return_value=True), \
             mock.patch.object(dependency_installer.subprocess, "Popen") as popen:
            self.assertFalse(installer.start_install("tensorrt"))
            success, message = installer._install_single("tensorrt", "tensorrt")

        self.assertFalse(success)
        self.assertIn("immutable Snap", message)
        popen.assert_not_called()

    def test_flatpak_installer_rejects_direct_runtime_mutation(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.object(
            dependency_installer, "running_in_flatpak", return_value=True
        ), mock.patch.object(
            installer, "is_available", return_value=False
        ), mock.patch.object(
            installer, "is_supported", return_value=True
        ), mock.patch.object(
            dependency_installer.subprocess, "Popen"
        ) as popen:
            self.assertFalse(installer.start_install("tensorrt"))
            success, message = installer._install_single("tensorrt", "tensorrt")

        self.assertFalse(success)
        self.assertIn("immutable Flatpak", message)
        popen.assert_not_called()

    def test_flatpak_cpu_variant_does_not_offer_source_installer(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.object(
            installer, "is_available", return_value=False
        ), mock.patch.object(
            dependency_installer, "supports_linux_gpu_stack", return_value=True
        ), mock.patch.object(
            dependency_installer,
            "detect_runtime_variant",
            return_value=dependency_installer.RuntimeVariant.CPU,
        ), mock.patch.object(
            dependency_installer, "running_in_flatpak", return_value=True
        ):
            reason = installer.install_block_reason("cupy")

        self.assertIn("Flatpak was built with the CPU runtime variant", reason)
        self.assertNotIn("./install.sh", reason)

    def test_bundled_runtime_is_available_even_when_install_is_unsupported(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.dict(dependency_installer.PACKAGE_SPECS["tensorrt"], {"check": lambda: True}), \
             mock.patch.object(installer, "is_supported", return_value=False):
            self.assertTrue(installer.is_available("tensorrt"))
            self.assertIsNone(installer.install_block_reason("tensorrt"))

    def test_system_runtime_cannot_be_mutated_from_gui(self):
        with mock.patch.dict(dependency_installer.os.environ, {}, clear=True), \
             mock.patch.object(dependency_installer.sys, "prefix", "/usr"), \
             mock.patch.object(dependency_installer.sys, "base_prefix", "/usr"):
            reason = dependency_installer._runtime_install_block_reason()

        self.assertIsNotNone(reason)
        self.assertIn("managed by its installer", reason)

    def test_has_whisper_requires_visible_backend_spec(self):
        def fake_find_spec(name):
            return None

        with mock.patch.object(dependency_installer.importlib.util, "find_spec", side_effect=fake_find_spec):
            self.assertFalse(dependency_installer._has_whisper())

    def test_has_whisper_accepts_faster_whisper_without_importing(self):
        def fake_find_spec(name):
            if name == "faster_whisper":
                return object()
            if name == "whisper":
                return None
            raise AssertionError(f"Unexpected spec lookup: {name}")

        with mock.patch.object(dependency_installer.importlib.util, "find_spec", side_effect=fake_find_spec):
            self.assertTrue(dependency_installer._has_whisper())

    def test_has_whisper_disables_openai_whisper_probe_on_python_314(self):
        def fake_find_spec(name):
            if name == "faster_whisper":
                return None
            if name == "whisper":
                return object()
            raise AssertionError(f"Unexpected spec lookup: {name}")

        with mock.patch.object(dependency_installer.importlib.util, "find_spec", side_effect=fake_find_spec), \
             mock.patch.object(dependency_installer, "supports_openai_whisper_python", return_value=False):
            self.assertFalse(dependency_installer._has_whisper())

    def test_zeus_mode_allowed_when_tensorrt_runtime_already_present(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.dict(dependency_installer.PACKAGE_SPECS["cupy"], {"check": lambda: True}), \
             mock.patch.object(dependency_installer, "IS_LINUX", True), \
             mock.patch.object(dependency_installer, "IS_ARM64", False), \
             mock.patch.object(dependency_installer, "has_tensorrt_runtime", return_value=True), \
             mock.patch.object(dependency_installer, "supports_tensorrt_python", return_value=False):
            self.assertIsNone(installer.unsupported_reason_for_mode("zeus"))

    def test_zeus_mode_stays_blocked_on_linux_arm64_even_with_runtime_present(self):
        installer = dependency_installer.DependencyInstaller()
        with mock.patch.object(dependency_installer, "IS_LINUX", True), \
             mock.patch.object(dependency_installer, "IS_ARM64", True), \
             mock.patch.object(dependency_installer, "has_tensorrt_runtime", return_value=True), \
             mock.patch.object(dependency_installer, "supports_tensorrt_python", return_value=False):
            self.assertEqual(
                installer.unsupported_reason_for_mode("zeus"),
                "GPU CUDA and TensorRT modes are not available on Linux arm64 yet. Use CPU modes for now.",
            )

    def test_whisper_package_spec_installs_httpx(self):
        install_steps = dependency_installer.PACKAGE_SPECS["whisper"]["install_steps"]
        self.assertEqual(
            install_steps[0],
            [
                "install",
                "--no-deps",
                dependency_installer.FASTER_WHISPER_REQUIREMENT,
            ],
        )
        self.assertIn("httpx", install_steps[1])
        self.assertIn("av", install_steps[1])
        self.assertIn("tqdm", install_steps[1])
        self.assertIn(
            dependency_installer.FASTER_WHISPER_REQUIREMENT,
            dependency_installer.PACKAGE_SPECS["whisper"]["help"],
        )

    def test_whisper_install_runs_two_pip_steps(self):
        installer = dependency_installer.DependencyInstaller()
        procs = [
            mock.Mock(stdout=[], wait=mock.Mock(return_value=0)),
            mock.Mock(stdout=[], wait=mock.Mock(return_value=0)),
        ]

        with mock.patch.object(installer, "is_available", return_value=False), \
             mock.patch.object(installer, "is_supported", return_value=True), \
             mock.patch.object(dependency_installer, "_runtime_install_block_reason", return_value=None), \
             mock.patch.object(installer, "_emit_progress", return_value=False), \
             mock.patch.dict(dependency_installer.PACKAGE_SPECS["whisper"], {"verify": lambda: True}), \
             mock.patch.object(dependency_installer.subprocess, "Popen", side_effect=procs) as popen:
            success, _message = installer._install_single("whisper", "whisper")

        self.assertTrue(success)
        first_cmd = popen.call_args_list[0].args[0]
        second_cmd = popen.call_args_list[1].args[0]
        self.assertEqual(first_cmd[:3], [dependency_installer.sys.executable, "-m", "pip"])
        self.assertEqual(second_cmd[:3], [dependency_installer.sys.executable, "-m", "pip"])
        self.assertIn("--no-deps", first_cmd)
        self.assertIn(dependency_installer.FASTER_WHISPER_REQUIREMENT, first_cmd)
        self.assertNotIn("faster-whisper", first_cmd)
        self.assertNotIn("ctranslate2", first_cmd)
        self.assertNotIn("--no-deps", second_cmd)
        self.assertIn("ctranslate2", second_cmd)
        self.assertIn("httpx", second_cmd)
        self.assertIn("av", second_cmd)


if __name__ == "__main__":
    unittest.main()

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


class DependencyInstallerTests(unittest.TestCase):
    def test_cuda_mode_runtime_requires_cupy_and_cuda_provider(self):
        with mock.patch.object(dependency_installer, "_has_cupy", return_value=True), \
             mock.patch.object(dependency_installer, "has_cuda_inference_runtime", return_value=False):
            self.assertFalse(dependency_installer._has_cuda_mode_runtime())

        with mock.patch.object(dependency_installer, "_has_cupy", return_value=True), \
             mock.patch.object(dependency_installer, "has_cuda_inference_runtime", return_value=True):
            self.assertTrue(dependency_installer._has_cuda_mode_runtime())

    def test_cuda_runtime_install_spec_includes_gpu_inference_provider(self):
        spec = dependency_installer.PACKAGE_SPECS["cupy"]
        install_args = spec["install_args"]
        self.assertIn("--upgrade", install_args)
        self.assertIn("cupy-cuda12x>=14.1.1,<15", install_args)
        self.assertIn("onnxruntime-gpu==1.24.4", install_args)
        self.assertIn("nvidia-cudnn-cu12", install_args)
        self.assertIn("nvidia-cuda-nvrtc-cu12", install_args)
        self.assertIn("ONNX Runtime GPU", spec["summary"])
        self.assertIn("onnxruntime-gpu==1.24.4", spec["help"])
        self.assertNotIn("onnxruntime-gpu>=", spec["help"])

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
        self.assertIn("not included in this Snap", reason)
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
        with mock.patch.object(dependency_installer, "IS_LINUX", True), \
             mock.patch.object(dependency_installer, "IS_ARM64", False), \
             mock.patch.object(dependency_installer, "_has_cuda_mode_runtime", return_value=True), \
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
        self.assertEqual(install_steps[0], ["install", "--no-deps", "faster-whisper"])
        self.assertIn("httpx", install_steps[1])
        self.assertIn("av", install_steps[1])
        self.assertIn("tqdm", install_steps[1])

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
        self.assertIn("faster-whisper", first_cmd)
        self.assertNotIn("ctranslate2", first_cmd)
        self.assertNotIn("--no-deps", second_cmd)
        self.assertIn("ctranslate2", second_cmd)
        self.assertIn("httpx", second_cmd)
        self.assertIn("av", second_cmd)


if __name__ == "__main__":
    unittest.main()

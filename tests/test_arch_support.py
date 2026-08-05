import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.core.config import detect_compositing_backends, detect_system_capabilities
from nvbroadcast.core.dependency_installer import DependencyInstaller
from nvbroadcast.core.platform import (
    has_cuda_inference_runtime,
    get_onnx_providers,
    get_trt_cache_dir,
    get_tensorrt_lib_dirs,
    has_tensorrt_runtime,
    legacy_tray_enabled,
    linux_multiarch_triplet,
    preload_nvidia_runtime_libs,
    python_runtime_advisory,
    supports_openai_whisper_python,
    supports_tensorrt_python,
    tensorrt_python_unsupported_reason,
)


class ArchSupportTests(unittest.TestCase):
    def test_linux_multiarch_triplet_arm64(self):
        import nvbroadcast.core.platform as platform_mod

        with mock.patch.object(platform_mod, "IS_ARM64", True):
            self.assertEqual(linux_multiarch_triplet(), "aarch64-linux-gnu")

    def test_arm64_capabilities_fall_back_to_cpu(self):
        import nvbroadcast.core.platform as platform_mod

        with mock.patch.object(platform_mod, "IS_MACOS", False), \
             mock.patch.object(platform_mod, "IS_LINUX", True), \
             mock.patch.object(platform_mod, "IS_ARM64", True), \
             mock.patch("os.cpu_count", return_value=8), \
             mock.patch.object(platform_mod, "supports_linux_gpu_stack", return_value=False):
            caps = detect_system_capabilities()

        self.assertTrue(caps["has_linux_arm64"])
        self.assertFalse(caps["has_nvidia"])
        self.assertEqual(caps["recommended_mode"], "auto")
        self.assertEqual(caps["recommended_resolved_mode"], "cpu_quality")

    def test_arm64_compositing_backends_hide_cupy(self):
        import nvbroadcast.core.platform as platform_mod

        with mock.patch.object(platform_mod, "supports_linux_gpu_stack", return_value=False):
            backends = detect_compositing_backends()

        self.assertTrue(backends["cpu"])
        self.assertFalse(backends["cupy"])

    def test_arm64_gpu_modes_report_unsupported(self):
        installer = DependencyInstaller()
        with mock.patch("nvbroadcast.core.dependency_installer.IS_LINUX", True), \
             mock.patch("nvbroadcast.core.dependency_installer.IS_ARM64", True):
            reason = installer.unsupported_reason_for_mode("doczeus")
        self.assertIsNotNone(reason)
        self.assertIn("Linux arm64", reason)

    def test_tensorrt_python_support_range(self):
        self.assertTrue(supports_tensorrt_python((3, 13)))
        self.assertFalse(supports_tensorrt_python((3, 14)))
        self.assertTrue(supports_openai_whisper_python((3, 13)))
        with mock.patch("importlib.metadata.version", side_effect=Exception("missing")):
            self.assertFalse(supports_openai_whisper_python((3, 14)))

    def test_openai_whisper_python_support_recovers_with_compatible_native_stack(self):
        with mock.patch("importlib.metadata.version", side_effect=lambda name: {
            "numba": "0.63.0b1",
            "llvmlite": "0.46.0b1",
        }[name]):
            self.assertTrue(supports_openai_whisper_python((3, 14)))

    def test_tensorrt_modes_report_python_version_unsupported(self):
        installer = DependencyInstaller()
        with mock.patch("nvbroadcast.core.dependency_installer.has_tensorrt_runtime", return_value=False), \
             mock.patch("nvbroadcast.core.dependency_installer.supports_tensorrt_python", return_value=False), \
             mock.patch(
                 "nvbroadcast.core.dependency_installer.tensorrt_python_unsupported_reason",
                 return_value=tensorrt_python_unsupported_reason((3, 14)),
             ):
            reason = installer.unsupported_reason_for_mode("zeus")
        self.assertIsNotNone(reason)
        self.assertIn("Python 3.14", reason)
        self.assertIn("DocZeus", reason)

    def test_get_tensorrt_lib_dirs_accepts_current_cu12_package_name(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib_dir = root / "lib"
            lib_dir.mkdir()

            def fake_find_spec(name: str):
                if name == "tensorrt_cu12_libs":
                    return SimpleNamespace(submodule_search_locations=[str(root)])
                return None

            with mock.patch("importlib.util.find_spec", side_effect=fake_find_spec):
                dirs = get_tensorrt_lib_dirs()

        self.assertEqual(dirs, [root, lib_dir])

    def test_has_tensorrt_runtime_accepts_current_cu12_lib_package(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "libnvinfer.so.10").touch()

            fake_ort = SimpleNamespace(
                get_available_providers=lambda: ["TensorrtExecutionProvider"]
            )

            def fake_find_spec(name: str):
                if name == "tensorrt_cu12_libs":
                    return SimpleNamespace(submodule_search_locations=[str(root)])
                return None

            with mock.patch.dict("sys.modules", {"onnxruntime": fake_ort}), \
                 mock.patch("nvbroadcast.core.platform.supports_linux_gpu_stack", return_value=True), \
                 mock.patch("nvbroadcast.core.platform.ctypes.util.find_library", return_value=None), \
                 mock.patch("nvbroadcast.core.platform.ctypes.CDLL", return_value=object()), \
                 mock.patch("importlib.util.find_spec", side_effect=fake_find_spec):
                self.assertTrue(has_tensorrt_runtime())

    def test_has_cuda_inference_runtime_requires_cuda_provider(self):
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        )
        with mock.patch.dict("sys.modules", {"onnxruntime": fake_ort}), \
             mock.patch("nvbroadcast.core.platform.supports_linux_gpu_stack", return_value=True):
            self.assertFalse(has_cuda_inference_runtime())

        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        with mock.patch.dict("sys.modules", {"onnxruntime": fake_ort}), \
             mock.patch("nvbroadcast.core.platform.supports_linux_gpu_stack", return_value=True):
            self.assertTrue(has_cuda_inference_runtime())

    def test_preload_exposes_component_wheel_headers_to_cupy(self):
        import nvbroadcast.core.platform as platform_mod

        with TemporaryDirectory() as tmp:
            runtime_root = Path(tmp)
            include_dir = runtime_root / "include"
            lib_dir = runtime_root / "lib"
            include_dir.mkdir()
            lib_dir.mkdir()
            (include_dir / "cuda_runtime.h").touch()
            (lib_dir / "libcudart.so.12").touch()

            def fake_find_spec(name: str):
                if name == "nvidia.cuda_runtime":
                    return SimpleNamespace(submodule_search_locations=[str(runtime_root)])
                return None

            with mock.patch.object(platform_mod, "IS_LINUX", True), \
                 mock.patch.dict(platform_mod.os.environ, {}, clear=True), \
                 mock.patch("importlib.util.find_spec", side_effect=fake_find_spec), \
                 mock.patch.object(platform_mod.ctypes, "CDLL") as cdll:
                preload_nvidia_runtime_libs()
                self.assertEqual(
                    platform_mod.os.environ.get("CUDA_PATH"), str(runtime_root)
                )
                cdll.assert_called_once_with(
                    str(lib_dir / "libcudart.so.12"),
                    mode=platform_mod.ctypes.RTLD_GLOBAL,
                )

    def test_preload_preserves_explicit_cuda_path(self):
        import nvbroadcast.core.platform as platform_mod

        with mock.patch.object(platform_mod, "IS_LINUX", True), \
             mock.patch.dict(platform_mod.os.environ, {"CUDA_PATH": "/opt/cuda"}, clear=True), \
             mock.patch("importlib.util.find_spec", return_value=None):
            preload_nvidia_runtime_libs()
            self.assertEqual(platform_mod.os.environ.get("CUDA_PATH"), "/opt/cuda")

    def test_cupy_backend_requires_cuda_inference_provider(self):
        fake_cupy = SimpleNamespace()
        with mock.patch.dict("sys.modules", {"cupy": fake_cupy}), \
             mock.patch("nvbroadcast.core.platform.supports_linux_gpu_stack", return_value=True), \
             mock.patch("nvbroadcast.core.platform.has_cuda_inference_runtime", return_value=False):
            backends = detect_compositing_backends()

        self.assertFalse(backends["cupy"])

    def test_get_trt_cache_dir_is_per_gpu_under_config(self):
        config_dir = Path("/tmp/nvbroadcast-test-config")
        with mock.patch("nvbroadcast.core.constants.CONFIG_DIR", config_dir):
            self.assertEqual(get_trt_cache_dir(1), config_dir / "trt_cache" / "gpu1")

    def test_legacy_tray_enabled_by_default_outside_kde(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertTrue(legacy_tray_enabled())

    def test_legacy_tray_can_be_explicitly_enabled(self):
        with mock.patch.dict("os.environ", {"NVBROADCAST_ENABLE_LEGACY_TRAY": "1"}, clear=False):
            self.assertTrue(legacy_tray_enabled())

    def test_legacy_tray_can_be_explicitly_disabled(self):
        with mock.patch.dict("os.environ", {"NVBROADCAST_ENABLE_LEGACY_TRAY": "0"}, clear=False):
            self.assertFalse(legacy_tray_enabled())

    def test_legacy_tray_disabled_on_kde_by_default(self):
        with mock.patch.dict("os.environ", {"XDG_CURRENT_DESKTOP": "KDE"}, clear=False):
            self.assertFalse(legacy_tray_enabled())

    def test_python_runtime_advisory_describes_reduced_paths(self):
        with mock.patch("importlib.metadata.version", side_effect=Exception("missing")):
            notice = python_runtime_advisory((3, 14), has_trt_runtime=False)
        self.assertIsNotNone(notice)
        key, title, body = notice
        self.assertEqual(key, "python-runtime-3.14")
        self.assertIn("Python 3.14", title)
        self.assertIn("Zeus and Killer", body)
        self.assertIn("faster-whisper", body)

    def test_python_runtime_advisory_mentions_installed_tensorrt_runtime(self):
        with mock.patch("importlib.metadata.version", side_effect=Exception("missing")):
            notice = python_runtime_advisory((3, 14), has_trt_runtime=True)
        self.assertIsNotNone(notice)
        self.assertIn("already installed", notice[2])

    def test_cuda_provider_uses_unified_stream_for_threaded_live_pipelines(self):
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        with mock.patch.dict("sys.modules", {"onnxruntime": fake_ort}), \
             mock.patch("nvbroadcast.core.platform.supports_linux_gpu_stack", return_value=True):
            providers = get_onnx_providers(gpu_index=1)

        cuda = providers[0]
        self.assertEqual(cuda[0], "CUDAExecutionProvider")
        self.assertTrue(cuda[1]["do_copy_in_default_stream"])
        self.assertTrue(cuda[1]["use_ep_level_unified_stream"])


if __name__ == "__main__":
    unittest.main()

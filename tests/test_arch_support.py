import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.core.config import detect_compositing_backends, detect_system_capabilities
from nvbroadcast.core.dependency_installer import DependencyInstaller
from nvbroadcast.core.platform import (
    get_tensorrt_lib_dirs,
    has_tensorrt_runtime,
    linux_multiarch_triplet,
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

    def test_tensorrt_modes_report_python_version_unsupported(self):
        installer = DependencyInstaller()
        with mock.patch("nvbroadcast.core.dependency_installer.supports_tensorrt_python", return_value=False), \
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


if __name__ == "__main__":
    unittest.main()

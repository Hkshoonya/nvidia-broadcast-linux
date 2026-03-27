import unittest
from unittest import mock

from nvbroadcast.core.config import detect_compositing_backends, detect_system_capabilities
from nvbroadcast.core.dependency_installer import DependencyInstaller
from nvbroadcast.core.platform import linux_multiarch_triplet


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
        self.assertEqual(caps["recommended_mode"], "cpu_quality")

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


if __name__ == "__main__":
    unittest.main()

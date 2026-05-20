import unittest
import sys
import types
from unittest import mock


if "gi" not in sys.modules:
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
        steps = dependency_installer.PACKAGE_SPECS["whisper"]["install_steps"]
        flat = [token for step in steps for token in step]
        self.assertIn("httpx", flat)

    def test_whisper_package_spec_resolves_http_stack_with_deps(self):
        """faster-whisper goes in with --no-deps to skip CPU onnxruntime, but
        the http/hub stack must be installed with full dependency resolution so
        httpcore/anyio/h11/certifi/idna actually land in the venv."""
        steps = dependency_installer.PACKAGE_SPECS["whisper"]["install_steps"]
        self.assertEqual(len(steps), 2)
        deps_step, faster_whisper_step = steps
        self.assertNotIn("--no-deps", deps_step)
        self.assertIn("httpx", deps_step)
        self.assertIn("huggingface-hub", deps_step)
        self.assertIn("--no-deps", faster_whisper_step)
        self.assertIn("faster-whisper", faster_whisper_step)


if __name__ == "__main__":
    unittest.main()

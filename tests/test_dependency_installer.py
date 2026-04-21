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
    def test_has_whisper_requires_successful_import(self):
        def fake_import(name):
            if name == "faster_whisper":
                raise ModuleNotFoundError("No module named 'httpx'")
            if name == "whisper":
                raise ModuleNotFoundError("No module named 'whisper'")
            raise AssertionError(f"Unexpected import: {name}")

        with mock.patch.object(dependency_installer.importlib, "import_module", side_effect=fake_import):
            self.assertFalse(dependency_installer._has_whisper())

    def test_has_whisper_accepts_fallback_backend(self):
        def fake_import(name):
            if name == "faster_whisper":
                raise ModuleNotFoundError("No module named 'httpx'")
            if name == "whisper":
                return object()
            raise AssertionError(f"Unexpected import: {name}")

        with mock.patch.object(dependency_installer.importlib, "import_module", side_effect=fake_import):
            self.assertTrue(dependency_installer._has_whisper())

    def test_whisper_package_spec_installs_httpx(self):
        install_args = dependency_installer.PACKAGE_SPECS["whisper"]["install_args"]
        self.assertIn("httpx", install_args)


if __name__ == "__main__":
    unittest.main()

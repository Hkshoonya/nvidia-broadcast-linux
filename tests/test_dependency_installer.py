import unittest
from unittest import mock

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

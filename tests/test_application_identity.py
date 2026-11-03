"""Package identity must agree with the sandbox's exported desktop entry."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class ApplicationIdentityTests(unittest.TestCase):
    def _read_identity(self, flatpak):
        # Fresh imports matter: GTK and resource lookup both capture APP_ID.
        code = """
from unittest import mock
with mock.patch('nvbroadcast.core.platform.running_in_flatpak', return_value=FLATPAK):
    from nvbroadcast.core.constants import APP_ID
    from nvbroadcast.core.resources import APP_ICON, APP_ICON_PNG
    print(APP_ID, APP_ICON, APP_ICON_PNG, sep='\\n')
""".replace("FLATPAK", repr(flatpak))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        return subprocess.check_output(
            [sys.executable, "-c", code], env=env, text=True, timeout=10
        ).splitlines()

    def test_native_packages_keep_the_existing_desktop_and_icon_identity(self):
        self.assertEqual(self._read_identity(False), [
            "com.doczeus.NVBroadcast",
            "com.doczeus.NVBroadcast.svg",
            "com.doczeus.NVBroadcast.png",
        ])

    def test_flatpak_uses_the_verified_domain_identity_for_app_and_icons(self):
        self.assertEqual(self._read_identity(True), [
            "com.nvbroadcast.NVBroadcast",
            "com.nvbroadcast.NVBroadcast.svg",
            "com.nvbroadcast.NVBroadcast.png",
        ])


if __name__ == "__main__":
    unittest.main()

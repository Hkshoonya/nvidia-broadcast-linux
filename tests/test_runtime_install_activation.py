import unittest
from types import SimpleNamespace
from unittest import mock

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from nvbroadcast.ui.setup_wizard import SetupWizard
from nvbroadcast.ui.window import NVBroadcastWindow


class RuntimeInstallActivationTests(unittest.TestCase):
    def _window(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._stop_install_pulse = mock.Mock()
        window._install_progress = SimpleNamespace(set_fraction=mock.Mock())
        window._install_detail = SimpleNamespace(set_text=mock.Mock())
        window._install_close_btn = SimpleNamespace(set_sensitive=mock.Mock())
        window.set_status = mock.Mock()
        window.rebuild_mode_selector = mock.Mock()
        window._pending_mode_key = "killer"
        window._pending_meeting_start = False
        window._mode_devices = [{"device": "killer"}]
        window._profile_selector = SimpleNamespace(set_selected_index=mock.Mock())
        window._on_mode_changed_selector = mock.Mock()
        window._app = SimpleNamespace(
            config=SimpleNamespace(
                compositing="cupy", performance_profile="performance"
            ),
            start_meeting=mock.Mock(),
        )
        return window

    def test_window_does_not_activate_gpu_mode_before_restart(self):
        window = self._window()
        installer = SimpleNamespace(restart_pending=lambda _key: True)

        window._on_install_job_completed(
            installer,
            "premium_gpu_stack",
            True,
            "Restart NVBroadcast to activate it.",
        )

        self.assertEqual(window._pending_mode_key, "")
        window._profile_selector.set_selected_index.assert_not_called()
        window._on_mode_changed_selector.assert_not_called()

    def test_window_preserves_non_runtime_install_continuation(self):
        window = self._window()
        installer = SimpleNamespace(restart_pending=lambda _key: False)

        window._on_install_job_completed(
            installer,
            "whisper",
            True,
            "Meeting Transcription Runtime installed successfully.",
        )

        window._profile_selector.set_selected_index.assert_called_once_with(0)
        window._on_mode_changed_selector.assert_called_once_with(
            window._profile_selector, "killer"
        )

    def test_setup_wizard_keeps_gpu_mode_inactive_until_restart(self):
        wizard = SetupWizard.__new__(SetupWizard)
        wizard._install_key = "cupy"
        wizard._start_btn = SimpleNamespace(set_sensitive=mock.Mock())
        wizard._skip_btn = SimpleNamespace(set_sensitive=mock.Mock())
        wizard._status_label = SimpleNamespace(set_text=mock.Mock())
        wizard._caps = {"has_cupy": False}
        wizard._selected_mode_key = "gpu_cuda_best"
        wizard._finish = mock.Mock()
        installer = SimpleNamespace(restart_pending=lambda _key: True)

        wizard._on_install_completed(
            installer,
            "cupy",
            True,
            "Restart NVBroadcast to activate it.",
        )

        self.assertFalse(wizard._caps["has_cupy"])
        wizard._finish.assert_not_called()


if __name__ == "__main__":
    unittest.main()

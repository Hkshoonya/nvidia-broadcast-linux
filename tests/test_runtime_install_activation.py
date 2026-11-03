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

    def test_gpu_switch_checking_snapshot_does_not_run_provider_probes(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._mode_availability_ready = False
        installer = SimpleNamespace(
            is_available=mock.Mock(),
            unsupported_reason_for_mode=mock.Mock(),
            missing_for_mode=mock.Mock(),
        )
        window._app = SimpleNamespace(dependency_installer=installer)

        devices = window._build_mode_devices()

        keys = [device["device"] for device in devices]
        labels = {device["device"]: device["name"] for device in devices}
        self.assertEqual(keys[:4], ["auto", "cpu_quality", "cpu_light", "cpu_low"])
        self.assertIn("checking selected GPU", labels["doczeus"])
        installer.is_available.assert_not_called()
        installer.unsupported_reason_for_mode.assert_not_called()
        installer.missing_for_mode.assert_not_called()

    def test_gpu_mode_click_waits_for_existing_selected_device_check(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._mode_availability_ready = False
        window._sync_mode_selector = mock.Mock()
        window.set_status = mock.Mock()
        installer = SimpleNamespace(
            unsupported_reason_for_mode=mock.Mock(),
            temporary_unavailable_reason_for_mode=mock.Mock(),
        )
        window._app = SimpleNamespace(
            _restoring=False,
            dependency_installer=installer,
        )
        window._retry_temporarily_unavailable_mode = mock.Mock()

        window._on_mode_changed_selector(None, "doczeus")

        window._sync_mode_selector.assert_called_once_with()
        window.set_status.assert_called_once_with(
            "Selected GPU runtime check is already in progress"
        )
        window._retry_temporarily_unavailable_mode.assert_not_called()
        installer.unsupported_reason_for_mode.assert_not_called()

    def test_concrete_cpu_mode_cancels_deferred_auto_or_focus_intent(self):
        for deferred in ("auto", "focus"):
            with self.subTest(deferred=deferred):
                window = NVBroadcastWindow.__new__(NVBroadcastWindow)
                window._mode_availability_ready = False
                window._mode_retry_in_flight = True
                window._mode_retry_generation = 5
                window._mode_availability_snapshot = None
                window._pending_mode_key = ""
                window._sync_mode_selector = mock.Mock()
                window.set_status = mock.Mock()
                installer = SimpleNamespace(
                    unsupported_reason_for_mode=mock.Mock(return_value=None),
                    temporary_unavailable_reason_for_mode=mock.Mock(return_value=None),
                    missing_for_mode=mock.Mock(return_value=[]),
                )
                window._app = SimpleNamespace(
                    _restoring=False,
                    config=SimpleNamespace(compute_gpu=0),
                    dependency_installer=installer,
                    _pending_auto_mode_gpu=0 if deferred == "auto" else None,
                    _pending_compute_focus=("gpu", 0) if deferred == "focus" else None,
                    _mode_compute_focus=mock.Mock(return_value="cpu"),
                    set_compute_focus=mock.Mock(),
                    set_auto_mode_enabled=mock.Mock(),
                    apply_mode_key=mock.Mock(),
                )

                window._on_mode_changed_selector(None, "cpu_quality")

                self.assertIsNone(window._app._pending_auto_mode_gpu)
                self.assertIsNone(window._app._pending_compute_focus)
                window._app.apply_mode_key.assert_called_once_with("cpu_quality")
                self.assertFalse(window._mode_retry_in_flight)
                self.assertEqual(window._mode_retry_generation, 6)

    def test_new_auto_or_focus_selection_cancels_previous_retry(self):
        for choice in ("auto", "focus"):
            with self.subTest(choice=choice):
                window = NVBroadcastWindow.__new__(NVBroadcastWindow)
                window._mode_availability_ready = True
                window._mode_retry_in_flight = True
                window._mode_retry_generation = 5
                window.set_status = mock.Mock()
                window._app = SimpleNamespace(
                    _restoring=False,
                    set_auto_mode_enabled=mock.Mock(),
                    set_compute_focus=mock.Mock(),
                    config=SimpleNamespace(compute_gpu=0),
                )
                if choice == "auto":
                    window._on_mode_changed_selector(None, "auto")
                else:
                    window._on_compute_focus_changed(None, "cpu")
                self.assertFalse(window._mode_retry_in_flight)
                self.assertEqual(window._mode_retry_generation, 6)
                window._finish_mode_runtime_retry("doczeus", "", 5, 0)
                self.assertFalse(window._mode_retry_in_flight)

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

        window.rebuild_mode_selector.assert_called_once_with(
            window._app.config.compositing,
            window._app.config.performance_profile,
        )
        window._profile_selector.set_selected_index.assert_not_called()
        window._on_mode_changed_selector.assert_not_called()

    def test_setup_wizard_keeps_gpu_mode_inactive_until_restart(self):
        wizard = SetupWizard.__new__(SetupWizard)
        wizard._install_key = "cupy"
        wizard._start_btn = SimpleNamespace(set_sensitive=mock.Mock())
        wizard._skip_btn = SimpleNamespace(set_sensitive=mock.Mock())
        wizard._status_label = SimpleNamespace(set_text=mock.Mock())
        wizard._caps = {"has_cupy": False}
        wizard._selected_mode_key = "gpu_cuda_best"
        wizard._capability_refresh_closed = False
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

    def test_mode_retry_completion_ignores_stale_gpu_generation(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._mode_retry_generation = 4
        window._mode_retry_in_flight = True
        window._app = SimpleNamespace(
            config=SimpleNamespace(
                compute_gpu=1,
                compositing="cupy",
                performance_profile="balanced",
            )
        )
        window.rebuild_mode_selector = mock.Mock()
        window._sync_mode_selector = mock.Mock()
        window.set_status = mock.Mock()

        result = window._finish_mode_runtime_retry(
            "doczeus",
            "",
            3,
            0,
        )

        self.assertFalse(result)
        self.assertTrue(window._mode_retry_in_flight)
        window.rebuild_mode_selector.assert_not_called()
        window._sync_mode_selector.assert_not_called()
        window.set_status.assert_not_called()

    def test_gpu_change_invalidates_in_flight_mode_retry(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._mode_retry_generation = 7
        window._mode_retry_in_flight = True
        window._app = SimpleNamespace(
            config=SimpleNamespace(compute_gpu=0),
            set_compute_gpu=mock.Mock(),
        )

        window._on_gpu_changed(None, "1")

        self.assertEqual(window._mode_retry_generation, 8)
        self.assertFalse(window._mode_retry_in_flight)
        window._app.set_compute_gpu.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest import mock

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from nvbroadcast.ui.setup_wizard import SetupWizard
from nvbroadcast.ui.window import NVBroadcastWindow
from nvbroadcast.core.config import AppConfig
from nvbroadcast.runtime.process_scope import ProviderProbeScope


class RuntimeInstallActivationTests(unittest.TestCase):
    def test_repeated_ui_selections_coalesce_physical_retry_and_keep_latest_mode(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._mode_retry_generation = 0
        window._mode_retry_in_flight = False
        window._mode_retry_workers = {}
        window._mode_retry_request = None
        window._mode_availability_ready = True
        window._pending_mode_key = ""
        window._mode_devices = [{"device": key} for key in ("doczeus", "cuda_balanced")]
        window._profile_selector = SimpleNamespace(set_selected_index=mock.Mock())
        window._sync_mode_selector = mock.Mock()
        window.rebuild_mode_selector = mock.Mock()
        window.set_status = mock.Mock()
        installer = mock.Mock()
        installer.busy = False
        installer.restart_pending.return_value = False
        installer.retry_mode_runtime.return_value = None
        fresh = {"gpu_index": 0, "has_cuda": True, "has_tensorrt": True,
                 "modes": {key: {"unsupported": "", "temporary": "", "missing": ()}
                           for key in ("doczeus", "cuda_balanced")}}
        installer.mode_availability_snapshot.return_value = fresh
        scope = ProviderProbeScope()
        window._app = SimpleNamespace(
            _window=window, _restoring=False, _mode_availability_closed=False,
            _mode_probe_scope=scope, dependency_installer=installer,
            _mode_availability_workers={}, _resume_mode_availability=mock.Mock(),
            config=AppConfig(), _pending_auto_mode_gpu=None, _pending_compute_focus=None,
            _mode_compute_focus=mock.Mock(return_value="gpu"),
            set_compute_focus=mock.Mock(), set_auto_mode_enabled=mock.Mock(),
            apply_mode_key=mock.Mock(),
        )
        window._mode_availability_snapshot = {
            "gpu_index": 0,
            "modes": {key: {"unsupported": "GPU busy", "temporary": "GPU busy", "missing": ()}
                      for key in ("doczeus", "cuda_balanced")},
        }
        workers, callbacks = [], []

        class QueuedThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                workers.append(self.target)

        with (
            mock.patch("nvbroadcast.ui.window.threading.Thread", QueuedThread),
            mock.patch("nvbroadcast.ui.window.GLib.idle_add", side_effect=lambda cb, *args: (
                callbacks.append((cb, args)) or len(callbacks)
            )),
            mock.patch("nvbroadcast.core.config.detect_system_capabilities", return_value={
                "cpu_cores": 8, "has_nvidia": True, "gpu_vram_mb": 8192,
            }),
        ):
            window._app._mode_availability_workers[0] = 7
            for _ in range(20):
                window._on_mode_changed_selector(None, "doczeus")
            self.assertEqual(len(workers), 0)
            window._app._mode_availability_workers.clear()
            window._resume_mode_runtime_retry()
            self.assertEqual(len(workers), 1)
            window._on_mode_changed_selector(None, "cuda_balanced")
            self.assertEqual(len(workers), 1)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            window._app.apply_mode_key.assert_not_called()
            self.assertEqual(len(workers), 1)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
        self.assertEqual(installer.retry_mode_runtime.call_count, 2)
        window._app.apply_mode_key.assert_called_once_with("cuda_balanced")
        self.assertFalse(window._mode_retry_in_flight)
        scope.close()

    def _window(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._stop_install_pulse = mock.Mock()
        window._install_progress = SimpleNamespace(set_fraction=mock.Mock())
        window._install_detail = SimpleNamespace(set_text=mock.Mock())
        window._install_close_btn = SimpleNamespace(set_sensitive=mock.Mock())
        window.set_status = mock.Mock()
        window.rebuild_mode_selector = mock.Mock()
        window._pending_mode_key = "killer"
        window._mode_retry_workers = {}
        window._mode_retry_request = None
        window._pending_meeting_start = False
        window._mode_devices = [{"device": "killer"}]
        window._profile_selector = SimpleNamespace(set_selected_index=mock.Mock())
        window._on_mode_changed_selector = mock.Mock()
        window._app = SimpleNamespace(
            _mode_availability_closed=False,
            _resume_mode_availability=mock.Mock(),
            config=SimpleNamespace(
                compositing="cupy", performance_profile="performance"
            ),
            start_meeting=mock.Mock(),
        )
        window._app._window = window
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
        window._mode_retry_in_flight = False
        window._mode_availability_ready = False
        window._mode_retry_request = None
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
                window._mode_retry_request = None
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
                window._mode_retry_workers = {}
                window._mode_retry_request = None
                window.set_status = mock.Mock()
                window._app = SimpleNamespace(
                    _mode_availability_closed=False,
                    _resume_mode_availability=mock.Mock(),
                    _restoring=False,
                    set_auto_mode_enabled=mock.Mock(),
                    set_compute_focus=mock.Mock(),
                    config=SimpleNamespace(compute_gpu=0),
                )
                window._app._window = window
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
        parent = object()
        wizard._app = SimpleNamespace(_mode_availability_closed=False, _window=parent)
        wizard._parent_window = parent
        wizard._capability_refresh_in_flight = False
        wizard._caps_gpu = 0
        wizard._selected_gpu = 0
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
        window._mode_retry_workers = {1: 4}
        window._mode_retry_request = ("doczeus", 4, 1)
        window._app = SimpleNamespace(
            _mode_availability_closed=False,
            _resume_mode_availability=mock.Mock(),
            config=SimpleNamespace(
                compute_gpu=1,
                compositing="cupy",
                performance_profile="balanced",
            )
        )
        window._app._window = window
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

    def test_mode_retry_cannot_publish_to_a_replaced_window_on_same_gpu(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._mode_retry_generation = 4
        window._mode_retry_in_flight = True
        window._mode_retry_workers = {0: 4}
        window._mode_retry_request = ("doczeus", 4, 0)
        window._app = SimpleNamespace(
            _mode_availability_closed=False,
            _window=object(),
            config=SimpleNamespace(compute_gpu=0),
        )
        window.rebuild_mode_selector = mock.Mock()
        window._sync_mode_selector = mock.Mock()
        window.set_status = mock.Mock()
        window._finish_mode_runtime_retry(
            "doczeus", "", 4, 0,
            {"gpu_index": 0, "modes": {}}, {},
        )
        window.rebuild_mode_selector.assert_not_called()
        window._sync_mode_selector.assert_not_called()
        window.set_status.assert_not_called()
        self.assertFalse(hasattr(window._app, "_mode_availability_snapshot"))

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

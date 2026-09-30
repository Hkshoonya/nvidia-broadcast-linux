import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.core.dependency_installer import DependencyInstaller
from nvbroadcast.ui.setup_wizard import Gtk, SETUP_MODES, SetupWizard


def _caps(
    *,
    gpu_name="Test GPU",
    vram=8192,
    has_nvidia=True,
    has_gl=True,
    has_cupy=True,
):
    return {
        "cpu_cores": 12,
        "gpu_name": gpu_name,
        "gpu_vram_mb": vram,
        "has_nvidia": has_nvidia,
        "has_apple_silicon": False,
        "has_linux_arm64": False,
        "has_gl_compositor": has_gl,
        "has_cupy": has_cupy,
        "recommended_mode": "auto",
        "recommended_resolved_mode": (
            "gpu_cuda_best" if has_cupy else "cpu_quality"
        ),
    }


class _Control:
    def __init__(self, *, active=False, sensitive=True, label=""):
        self.active = active
        self.sensitive = sensitive
        self.label = label
        self.visible = True

    def get_active(self):
        return self.active

    def set_active(self, active):
        self.active = active

    def set_sensitive(self, sensitive):
        self.sensitive = sensitive

    def set_label(self, label):
        self.label = label

    def set_text(self, text):
        self.label = text

    def set_visible(self, visible):
        self.visible = visible


class SetupWizardGpuRefreshTests(unittest.TestCase):
    def _wizard(self, caps=None):
        caps = caps or _caps()
        wizard = SetupWizard.__new__(SetupWizard)
        app_installer = mock.Mock()
        wizard._app = SimpleNamespace(
            config=SimpleNamespace(compute_gpu=0),
            dependency_installer=app_installer,
        )
        wizard._installer_signal_ids = ()
        wizard._selected_gpu = 0
        wizard._caps_gpu = 0
        wizard._caps = caps
        wizard._selected_mode_key = "auto"
        wizard._install_in_flight = False
        wizard._capability_refresh_generation = 0
        wizard._capability_refresh_in_flight = False
        wizard._capability_refresh_closed = False
        wizard._mode_buttons = {
            mode["key"]: _Control(
                active=mode["key"] == "auto",
                sensitive=SetupWizard._mode_choice_state(mode, caps)[0],
                label=SetupWizard._mode_choice_state(mode, caps)[1],
            )
            for mode in SETUP_MODES
        }
        wizard._system_cpu_label = _Control()
        wizard._system_gpu_label = _Control()
        wizard._system_features_label = _Control()
        wizard._start_btn = _Control()
        wizard._skip_btn = _Control()
        wizard._status_label = _Control()
        return wizard, app_installer

    def test_available_mode_becomes_unavailable_and_falls_back_to_auto(self):
        wizard, _app_installer = self._wizard(_caps(has_gl=True))
        wizard._selected_gpu = 1
        wizard._selected_mode_key = "gpu_quality"
        wizard._mode_buttons["gpu_quality"].set_active(True)
        wizard._capability_refresh_generation = 4
        wizard._capability_refresh_in_flight = True

        wizard._finish_capability_refresh(
            4,
            1,
            _caps(gpu_name="GPU 1", has_gl=False),
        )

        self.assertFalse(wizard._mode_buttons["gpu_quality"].sensitive)
        self.assertIn(
            "needs GStreamer GL plugins",
            wizard._mode_buttons["gpu_quality"].label,
        )
        self.assertEqual(wizard._selected_mode_key, "auto")
        self.assertTrue(wizard._mode_buttons["auto"].active)
        self.assertTrue(wizard._start_btn.sensitive)
        self.assertTrue(wizard._skip_btn.sensitive)
        self.assertEqual(wizard._caps_gpu, 1)

    def test_unavailable_mode_becomes_available_after_gpu_refresh(self):
        wizard, _app_installer = self._wizard(_caps(has_gl=False))
        wizard._selected_gpu = 1
        wizard._capability_refresh_generation = 2
        wizard._capability_refresh_in_flight = True
        self.assertFalse(wizard._mode_buttons["gpu_balanced"].sensitive)

        wizard._finish_capability_refresh(
            2,
            1,
            _caps(gpu_name="GPU 1", has_gl=True),
        )

        self.assertTrue(wizard._mode_buttons["gpu_balanced"].sensitive)
        self.assertNotIn(
            "needs GStreamer GL plugins",
            wizard._mode_buttons["gpu_balanced"].label,
        )
        self.assertEqual(wizard._system_gpu_label.label, "GPU: GPU 1 (8192 MB)")
        self.assertEqual(wizard._caps_gpu, 1)

    def test_stale_capability_result_is_ignored(self):
        original_caps = _caps(gpu_name="GPU 0")
        wizard, _app_installer = self._wizard(original_caps)
        wizard._selected_gpu = 2
        wizard._capability_refresh_generation = 8
        wizard._capability_refresh_in_flight = True
        wizard._start_btn.sensitive = False
        wizard._skip_btn.sensitive = False

        wizard._finish_capability_refresh(
            7,
            1,
            _caps(gpu_name="Stale GPU", has_gl=False, has_cupy=False),
        )

        self.assertIs(wizard._caps, original_caps)
        self.assertEqual(wizard._caps_gpu, 0)
        self.assertTrue(wizard._capability_refresh_in_flight)
        self.assertFalse(wizard._start_btn.sensitive)
        self.assertFalse(wizard._skip_btn.sensitive)

    def test_toggle_then_close_does_not_change_app_gpu_target(self):
        wizard, app_installer = self._wizard()
        selected_button = _Control(active=True)
        worker = mock.Mock()

        with mock.patch(
            "nvbroadcast.ui.setup_wizard.threading.Thread",
            return_value=worker,
        ) as thread_cls:
            wizard._on_gpu_toggled(selected_button, 1)

        self.assertEqual(wizard._selected_gpu, 1)
        self.assertEqual(wizard._app.config.compute_gpu, 0)
        app_installer.set_compute_gpu.assert_not_called()
        thread_cls.assert_called_once()
        self.assertTrue(callable(thread_cls.call_args.kwargs["target"]))
        worker.start.assert_called_once_with()
        self.assertFalse(wizard._start_btn.sensitive)
        self.assertFalse(wizard._skip_btn.sensitive)

        wizard._on_close_requested()
        wizard._finish_capability_refresh(1, 1, _caps(gpu_name="GPU 1"))

        self.assertEqual(wizard._app.config.compute_gpu, 0)
        app_installer.set_compute_gpu.assert_not_called()
        self.assertNotEqual(wizard._caps["gpu_name"], "GPU 1")

    def test_install_attempt_uses_shared_installer_and_keeps_restart_block(self):
        wizard, _unused_installer = self._wizard()
        installer = DependencyInstaller(gpu_index=0)
        wizard._app.dependency_installer = installer
        wizard._install_key = "cupy"
        wizard._selected_gpu = 2
        dialog = SimpleNamespace(destroy=mock.Mock())

        with mock.patch.object(installer, "start_install", return_value=True) as start:
            wizard._on_install_prompt_response(dialog, Gtk.ResponseType.OK)

        start.assert_called_once_with("cupy", gpu_index=2)
        self.assertTrue(wizard._install_in_flight)
        installer._mark_restart_pending("cupy")
        wizard._on_install_completed(installer, "cupy", False, "install failed")

        self.assertFalse(wizard._install_in_flight)
        self.assertTrue(installer.restart_pending("cupy"))
        self.assertIn("Restart NVBroadcast", installer.unsupported_reason_for_mode("doczeus"))
        self.assertFalse(wizard._on_close_requested())
        self.assertTrue(installer.restart_pending("cupy"))

    def test_title_bar_close_is_refused_during_install(self):
        wizard, app_installer = self._wizard()
        wizard._install_in_flight = True

        self.assertTrue(wizard._on_close_requested())

        self.assertFalse(wizard._capability_refresh_closed)
        self.assertIn("Wait for", wizard._status_label.label)
        app_installer.disconnect.assert_not_called()

    def test_late_install_completion_after_close_is_ignored(self):
        wizard, app_installer = self._wizard()
        wizard._install_key = "cupy"
        wizard._finish = mock.Mock()
        original_caps = wizard._caps.copy()
        original_status = wizard._status_label.label

        self.assertFalse(wizard._on_close_requested())
        wizard._on_install_started(app_installer, "cupy", "started")
        wizard._on_install_progress(app_installer, "cupy", "progress", 0.5)
        wizard._on_install_completed(app_installer, "cupy", True, "complete")

        self.assertEqual(wizard._caps, original_caps)
        self.assertEqual(wizard._status_label.label, original_status)
        wizard._finish.assert_not_called()

    def test_apply_does_not_commit_during_capability_refresh(self):
        wizard, _app_installer = self._wizard()
        wizard._capability_refresh_in_flight = True
        wizard._finish = mock.Mock()

        wizard._on_start(None)

        wizard._finish.assert_not_called()
        self.assertIn("Wait for", wizard._status_label.label)


if __name__ == "__main__":
    unittest.main()

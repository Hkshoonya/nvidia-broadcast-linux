import unittest
from types import SimpleNamespace
from unittest import mock

from gi.repository import Gio, GLib

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig
from nvbroadcast.video.pipeline import VideoPipeline


class StartMinimizedTests(unittest.TestCase):
    @staticmethod
    def _app(*, requested=True, first_run=False, available=False, bus_ready=True):
        window = mock.Mock()
        window.get_visible.return_value = False
        app = SimpleNamespace(
            config=SimpleNamespace(first_run=first_run),
            _window=window,
            _tray=SimpleNamespace(available=available, bus_ready=bus_ready),
            _start_minimized=requested,
            _startup_tray_source_id=0,
            _startup_tray_seen=False,
            _startup_tray_deadline=0.0,
            _maybe_show_python_runtime_notice=mock.Mock(),
        )
        for name in (
            "_present_startup_window", "_present_window",
            "_cancel_startup_tray_wait", "_check_startup_tray",
        ):
            setattr(app, name, getattr(NVBroadcastApp, name).__get__(app))
        return app

    @mock.patch("nvbroadcast.app.GLib.timeout_add_seconds", return_value=42)
    @mock.patch("nvbroadcast.app.time.monotonic", return_value=100.0)
    def test_registered_tray_starts_hidden_without_presenting(self, _clock, timer):
        app = self._app(available=True)

        app._present_startup_window(initial_activation=True)

        app._window.present.assert_not_called()
        app._window.set_visible.assert_not_called()
        self.assertTrue(app._startup_tray_seen)
        self.assertEqual(app._startup_tray_source_id, 42)
        timer.assert_called_once_with(1, app._check_startup_tray)

    @mock.patch("nvbroadcast.app.GLib.timeout_add_seconds", return_value=42)
    @mock.patch("nvbroadcast.app.time.monotonic", return_value=100.0)
    def test_late_watcher_can_register_before_deadline(self, clock, _timer):
        app = self._app()
        app._present_startup_window(initial_activation=True)
        clock.return_value = 129.0
        self.assertTrue(app._check_startup_tray())
        app._tray.available = True
        clock.return_value = 131.0

        self.assertTrue(app._check_startup_tray())

        self.assertTrue(app._startup_tray_seen)
        app._window.present.assert_not_called()

    @mock.patch("nvbroadcast.app.GLib.source_remove")
    @mock.patch("nvbroadcast.app.GLib.timeout_add_seconds", return_value=42)
    @mock.patch("nvbroadcast.app.time.monotonic", return_value=100.0)
    def test_missing_watcher_shows_window_after_thirty_seconds(self, clock, _timer, remove):
        app = self._app()
        app._present_startup_window(initial_activation=True)
        clock.return_value = 130.0

        self.assertFalse(app._check_startup_tray())

        app._window.set_visible.assert_called_once_with(True)
        app._window.present.assert_called_once_with()
        self.assertEqual(app._startup_tray_source_id, 0)
        remove.assert_not_called()  # The currently executing timer removes itself.

    @mock.patch("nvbroadcast.app.GLib.timeout_add_seconds", return_value=42)
    @mock.patch("nvbroadcast.app.time.monotonic", return_value=100.0)
    def test_watcher_loss_restores_startup_hidden_window(self, _clock, _timer):
        app = self._app(available=True)
        app._present_startup_window(initial_activation=True)
        app._tray.available = False

        self.assertFalse(app._check_startup_tray())

        app._window.present.assert_called_once_with()
        self.assertEqual(app._startup_tray_source_id, 0)

    @mock.patch("nvbroadcast.app.GLib.timeout_add_seconds")
    def test_unexported_or_absent_tray_shows_immediately(self, timer):
        for tray in (None, SimpleNamespace(available=False, bus_ready=False)):
            with self.subTest(tray=tray):
                app = self._app()
                app._tray = tray

                app._present_startup_window(initial_activation=True)

                app._window.present.assert_called_once_with()
        timer.assert_not_called()

    @mock.patch("nvbroadcast.app.GLib.timeout_add_seconds")
    def test_first_run_and_normal_launch_are_always_visible(self, timer):
        for settings in ({"first_run": True}, {"requested": False}):
            with self.subTest(settings=settings):
                app = self._app(available=True, **settings)

                app._present_startup_window(initial_activation=True)

                app._window.present.assert_called_once_with()
        timer.assert_not_called()

    @mock.patch("nvbroadcast.app.GLib.source_remove")
    def test_second_activation_shows_existing_window_and_cancels_wait(self, remove):
        app = self._app(available=True)
        app._startup_tray_source_id = 42

        app._present_startup_window(initial_activation=False)

        remove.assert_called_once_with(42)
        app._window.present.assert_called_once_with()
        self.assertEqual(app._startup_tray_source_id, 0)

    @mock.patch("nvbroadcast.app.GLib.source_remove")
    def test_manual_show_stops_monitor_without_rehiding_window(self, remove):
        app = self._app(available=True)
        app._startup_tray_source_id = 42
        app._window.get_visible.return_value = True

        self.assertFalse(app._check_startup_tray())

        self.assertEqual(app._startup_tray_source_id, 0)
        app._window.set_visible.assert_not_called()
        app._window.present.assert_not_called()
        remove.assert_not_called()

    @mock.patch("nvbroadcast.app.GLib.source_remove")
    def test_cancel_removes_source_and_stale_callback_does_not_show_window(self, remove):
        app = self._app()
        app._startup_tray_source_id = 42

        app._cancel_startup_tray_wait()

        remove.assert_called_once_with(42)
        self.assertFalse(app._check_startup_tray())
        app._window.present.assert_not_called()

    def test_local_option_parser_continues_normal_application_activation(self):
        for requested in (False, True):
            with self.subTest(requested=requested):
                app = SimpleNamespace(_start_minimized=False)
                options = GLib.VariantDict.new(None)
                if requested:
                    options.insert_value("minimized", GLib.Variant("b", True))

                result = NVBroadcastApp.do_handle_local_options(app, options)

                self.assertEqual(result, -1)
                self.assertEqual(app._start_minimized, requested)


class MinimizedCommandLineTests(unittest.TestCase):
    def test_gapplication_parses_flag_and_activates_without_persisting_it(self):
        class Application(Gio.Application):
            def __init__(self):
                super().__init__(flags=Gio.ApplicationFlags.NON_UNIQUE)
                self.add_main_option(
                    "minimized", 0, GLib.OptionFlags.NONE, GLib.OptionArg.NONE,
                    "Start hidden in the system tray", None,
                )
                self._start_minimized = False
                self.activations = []

            do_handle_local_options = NVBroadcastApp.do_handle_local_options

            def do_activate(self):
                self.activations.append(self._start_minimized)

        app = Application()

        self.assertEqual(app.run(["nvbroadcast", "--minimized"]), 0)
        self.assertEqual(app.activations, [True])


class HiddenStartupPipelineTests(unittest.TestCase):
    @mock.patch("nvbroadcast.app.save_config")
    @mock.patch("nvbroadcast.video.virtual_camera.resolve_camera_device", return_value="/dev/test")
    @mock.patch("nvbroadcast.video.virtual_camera.select_camera_mode")
    @mock.patch("nvbroadcast.video.pipeline.GLib.timeout_add", return_value=99)
    def test_hidden_start_streams_without_scheduling_preview_textures(
        self, timer, select_mode, _resolve, _save,
    ):
        for visible in (False, True):
            with self.subTest(visible=visible):
                config = AppConfig()
                select_mode.return_value = {
                    "width": config.video.width,
                    "height": config.video.height,
                    "fps": config.video.fps,
                }
                app = SimpleNamespace(
                    config=config,
                    _eye_contact=mock.Mock(),
                    _clear_finished_teardown=mock.Mock(),
                    _video_pipeline=None,
                    _pipeline_teardown=None,
                    _get_valid_fps=mock.Mock(return_value=config.video.fps),
                    _active_vcam_device=mock.Mock(return_value="/dev/test-output"),
                    _use_nvdec=False,
                    _process_frame=mock.Mock(),
                    _update_alpha=mock.Mock(),
                    _on_recording_error=mock.Mock(),
                    _on_capture_error=mock.Mock(),
                    _inline_inference=True,
                    _sync_gpu_frame_path=mock.Mock(),
                    _video_effects=mock.Mock(_backend=None),
                    _beautifier=SimpleNamespace(),
                    _any_video_effects_active=mock.Mock(return_value=False),
                    _vcam_available=False,
                    _window=mock.Mock(),
                    _sync_tray_status=mock.Mock(),
                )
                app._window.get_visible.return_value = visible
                pipeline = VideoPipeline()

                def build_without_camera(_vcam_enabled):
                    pipeline._pipeline = mock.Mock()

                timer.reset_mock()
                with (
                    mock.patch("nvbroadcast.app.VideoPipeline", return_value=pipeline),
                    mock.patch.object(pipeline, "_build_passthrough_pipeline", build_without_camera),
                    mock.patch.object(pipeline, "start") as start,
                    mock.patch.object(pipeline, "prepare_recording_codecs"),
                ):
                    self.assertTrue(NVBroadcastApp._do_start_pipeline(app, "/dev/test"))

                start.assert_called_once_with()
                self.assertEqual(pipeline._preview_enabled, visible)
                self.assertEqual(pipeline._preview_timer_id, 99 if visible else 0)
                self.assertEqual(timer.call_count, int(visible))


if __name__ == "__main__":
    unittest.main()

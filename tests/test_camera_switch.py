"""Regression coverage for switching camera sources."""

from types import SimpleNamespace
import unittest
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig
from nvbroadcast.ui.window import NVBroadcastWindow


class CameraSwitchTests(unittest.TestCase):
    @mock.patch("nvbroadcast.app.save_config")
    @mock.patch(
        "nvbroadcast.video.virtual_camera.select_camera_mode",
        return_value={
            "format": "mjpeg",
            "width": 1280,
            "height": 720,
            "fps": 60,
        },
    )
    def test_active_camera_switch_restarts_with_supported_mode(
        self, select_camera_mode, save_config
    ):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.camera_device = "/dev/video0"
        app.config.video.width = 1920
        app.config.video.height = 1080
        app.config.video.fps = 30
        app.config.video.output_format = "YUY2"
        app._streaming = True
        app._window = mock.Mock()
        app.start_pipeline = mock.Mock()

        app.switch_camera("/dev/video2")

        select_camera_mode.assert_called_once_with(
            "/dev/video2", 1920, 1080, 30
        )
        self.assertEqual(app.config.video.camera_device, "/dev/video2")
        self.assertEqual(app.config.video.width, 1280)
        self.assertEqual(app.config.video.height, 720)
        self.assertEqual(app.config.video.fps, 60)
        save_config.assert_called_once_with(app.config)
        app._window.sync_video_input_controls.assert_called_once_with(app.config)
        app.start_pipeline.assert_called_once_with("/dev/video2", "YUY2")

    def test_camera_selector_switches_application_camera(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._updating_ui = False
        window._app = SimpleNamespace(
            _restoring=False,
            switch_camera=mock.Mock(),
        )

        window._on_camera_changed(None, "/dev/video2")

        window._app.switch_camera.assert_called_once_with("/dev/video2")

    def test_camera_selector_ignores_programmatic_updates(self):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._updating_ui = True
        window._app = SimpleNamespace(
            _restoring=False,
            switch_camera=mock.Mock(),
        )

        window._on_camera_changed(None, "/dev/video2")

        window._app.switch_camera.assert_not_called()

    @mock.patch(
        "nvbroadcast.video.virtual_camera.resolve_camera_device",
        return_value="",
    )
    def test_missing_camera_stops_before_pipeline_construction(
        self, resolve_camera_device
    ):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app._video_pipeline = None
        app._pipeline_teardown = None
        app._streaming = False
        app._window = mock.Mock()
        app._clear_finished_teardown = mock.Mock()

        app._do_start_pipeline("/dev/video0", "YUY2")

        resolve_camera_device.assert_called_once_with("/dev/video0")
        self.assertIsNone(app._video_pipeline)
        self.assertFalse(app._streaming)
        app._window.set_status.assert_called_once_with(
            "No usable camera found. Connect a camera and try again."
        )


if __name__ == "__main__":
    unittest.main()

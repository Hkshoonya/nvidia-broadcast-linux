import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.video import virtual_camera


class VirtualCameraConfigTests(unittest.TestCase):
    def test_macos_uses_the_coremedia_device_name(self):
        with mock.patch.object(virtual_camera, "IS_MACOS", True):
            device = virtual_camera.ensure_virtual_camera("/dev/video11")

        self.assertEqual(device, virtual_camera.VIRTUAL_CAM_LABEL)

    def test_preferred_virtual_camera_device_is_used_when_it_exists(self):
        with mock.patch.object(virtual_camera, "IS_MACOS", False), mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            side_effect=lambda path: path == "/dev/video11",
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.is_v4l2loopback_device",
            return_value=True,
        ):
            device = virtual_camera.ensure_virtual_camera("/dev/video11")

        self.assertEqual(device, "/dev/video11")

    def test_existing_physical_camera_is_rejected_as_virtual_output(self):
        with mock.patch.object(virtual_camera, "IS_MACOS", False), mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            return_value=True,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.is_v4l2loopback_device",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "not a v4l2loopback"):
                virtual_camera.ensure_virtual_camera("/dev/video0")

    def test_missing_preferred_device_reports_matching_video_number(self):
        with mock.patch.object(virtual_camera, "IS_MACOS", False), mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            return_value=False,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.is_v4l2loopback_loaded",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError) as raised:
                virtual_camera.ensure_virtual_camera("/dev/video11")

        message = str(raised.exception)
        self.assertIn("/dev/video11", message)
        self.assertIn("video_nr=11", message)

    def test_invalid_preferred_device_is_rejected_before_probing(self):
        with mock.patch.object(virtual_camera, "IS_MACOS", False), mock.patch(
            "nvbroadcast.video.virtual_camera.get_virtual_camera_device",
        ) as get_device:
            with self.assertRaisesRegex(RuntimeError, "/dev/video10"):
                virtual_camera.ensure_virtual_camera("not-a-device")

        get_device.assert_not_called()

    def test_default_virtual_camera_still_scans_existing_loopback_devices(self):
        listing = "NVbroadcast (platform:v4l2loopback-012):\n\t/dev/video12\n"
        completed = mock.Mock(stdout=listing)
        with mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            return_value=False,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.is_v4l2loopback_device",
            return_value=True,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.subprocess.run",
            return_value=completed,
        ):
            device = virtual_camera.get_virtual_camera_device()

        self.assertEqual(device, "/dev/video12")

    def test_reset_virtual_camera_uses_selected_video_number(self):
        with mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            side_effect=lambda path: path == "/dev/video12",
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.is_v4l2loopback_device",
            return_value=True,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.subprocess.run",
        ) as run_mock:
            ok = virtual_camera.reset_virtual_camera("/dev/video12")

        self.assertTrue(ok)
        modprobe_args = run_mock.call_args_list[1].args[0]
        self.assertIn("video_nr=12", modprobe_args)

    def test_reset_does_not_touch_modules_for_a_physical_camera(self):
        with mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            return_value=True,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.is_v4l2loopback_device",
            return_value=False,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.subprocess.run",
        ) as run_mock:
            ok = virtual_camera.reset_virtual_camera("/dev/video0")

        self.assertFalse(ok)
        run_mock.assert_not_called()

    def test_reset_rejects_malformed_device_without_touching_modules(self):
        with mock.patch(
            "nvbroadcast.video.virtual_camera.subprocess.run",
        ) as run_mock:
            ok = virtual_camera.reset_virtual_camera("not-a-device")

        self.assertFalse(ok)
        run_mock.assert_not_called()

    def test_video_zero_is_not_replaced_by_the_default_device_number(self):
        command = virtual_camera.v4l2loopback_modprobe_command("/dev/video0")

        self.assertIn("video_nr=0", command)

    def test_existing_loopback_driver_is_detected_for_video_zero(self):
        info = """Driver Info:
\tDriver name      : v4l2 loopback
\tCard type        : Custom output
"""
        with mock.patch(
            "nvbroadcast.video.virtual_camera._get_v4l2_device_info",
            return_value=info,
        ):
            self.assertTrue(virtual_camera.is_v4l2loopback_device("/dev/video0"))

    def test_app_does_not_persist_a_physical_camera_as_output(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = SimpleNamespace(
            video=SimpleNamespace(vcam_device="/dev/video10")
        )
        app._window = mock.Mock()

        with mock.patch("nvbroadcast.app.IS_LINUX", True), mock.patch(
            "nvbroadcast.app.os.path.exists",
            return_value=True,
        ), mock.patch(
            "nvbroadcast.app.is_v4l2loopback_device",
            return_value=False,
        ), mock.patch("nvbroadcast.app.save_config") as save_config:
            changed = NVBroadcastApp.set_vcam_device(app, "/dev/video0")

        self.assertFalse(changed)
        self.assertEqual(app.config.video.vcam_device, "/dev/video10")
        save_config.assert_not_called()
        app._window.set_status.assert_called_once_with(
            "/dev/video0 is not a v4l2loopback virtual camera."
        )


if __name__ == "__main__":
    unittest.main()

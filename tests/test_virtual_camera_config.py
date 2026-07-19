import unittest
from unittest import mock

from nvbroadcast.video import virtual_camera


class VirtualCameraConfigTests(unittest.TestCase):
    def test_preferred_virtual_camera_device_is_used_when_it_exists(self):
        with mock.patch.object(virtual_camera, "IS_MACOS", False), mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            side_effect=lambda path: path == "/dev/video11",
        ):
            device = virtual_camera.ensure_virtual_camera("/dev/video11")

        self.assertEqual(device, "/dev/video11")

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

    def test_default_virtual_camera_still_scans_existing_loopback_devices(self):
        listing = "NVbroadcast (platform:v4l2loopback-012):\n\t/dev/video12\n"
        completed = mock.Mock(stdout=listing)
        with mock.patch(
            "nvbroadcast.video.virtual_camera.os.path.exists",
            return_value=False,
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
            "nvbroadcast.video.virtual_camera.subprocess.run",
        ) as run_mock:
            ok = virtual_camera.reset_virtual_camera("/dev/video12")

        self.assertTrue(ok)
        modprobe_args = run_mock.call_args_list[1].args[0]
        self.assertIn("video_nr=12", modprobe_args)


if __name__ == "__main__":
    unittest.main()

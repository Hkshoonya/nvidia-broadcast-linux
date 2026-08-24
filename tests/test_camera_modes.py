import subprocess
import unittest
from unittest import mock

from nvbroadcast.video.virtual_camera import (
    camera_capture_candidates,
    camera_mode_candidates,
    list_camera_devices,
    list_camera_format_modes,
    list_camera_modes,
    resolve_camera_device,
    select_camera_mode,
    select_camera_capture_format,
)


class CameraModesTests(unittest.TestCase):
    def setUp(self):
        list_camera_format_modes.cache_clear()
        list_camera_modes.cache_clear()
        import nvbroadcast.video.virtual_camera as virtual_camera
        virtual_camera._get_v4l2_device_info.cache_clear()

    def tearDown(self):
        list_camera_format_modes.cache_clear()
        list_camera_modes.cache_clear()
        import nvbroadcast.video.virtual_camera as virtual_camera
        virtual_camera._get_v4l2_device_info.cache_clear()

    def test_list_camera_modes_returns_empty_on_timeout(self):
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", side_effect=subprocess.TimeoutExpired("v4l2-ctl", 3)):
            self.assertEqual(list_camera_modes("/dev/video99"), [])

    def test_resolve_camera_device_returns_empty_when_none_are_usable(self):
        with mock.patch(
            "nvbroadcast.video.virtual_camera.IS_MACOS", False
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.is_usable_camera_device",
            return_value=False,
        ), mock.patch(
            "nvbroadcast.video.virtual_camera.list_camera_devices",
            return_value=[],
        ):
            self.assertIsNone(resolve_camera_device("/dev/video0"))

    def test_list_camera_modes_is_cached(self):
        output = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 1280x720
                        Interval: Discrete 0.033s (30.000 fps)
"""
        run_result = mock.Mock(returncode=0, stdout=output)
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", return_value=run_result) as run:
            first = list_camera_modes("/dev/video0")
            second = list_camera_modes("/dev/video0")

        self.assertEqual(first, second)
        self.assertEqual(run.call_count, 1)

    def test_list_camera_modes_includes_raw_only_modes(self):
        output = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 640x480
                        Interval: Discrete 0.033s (30.000 fps)
        [1]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 1280x720
                        Interval: Discrete 0.033s (30.000 fps)
"""
        run_result = mock.Mock(returncode=0, stdout=output)
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", return_value=run_result):
            self.assertEqual(
                list_camera_format_modes("/dev/video0"),
                [
                    {"format": "YUYV", "width": 640, "height": 480, "fps": [30]},
                    {"format": "MJPG", "width": 1280, "height": 720, "fps": [30]},
                ],
            )
            self.assertEqual(
                list_camera_modes("/dev/video0"),
                [
                    {"width": 640, "height": 480, "fps": [30]},
                    {"width": 1280, "height": 720, "fps": [30]},
                ],
            )

    def test_select_camera_capture_format_falls_back_to_raw(self):
        output = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 640x480
                        Interval: Discrete 0.033s (30.000 fps)
"""
        run_result = mock.Mock(returncode=0, stdout=output)
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", return_value=run_result):
            self.assertEqual(
                select_camera_capture_format("/dev/video0", 640, 480, 30),
                "raw",
            )

    def test_select_camera_capture_format_prefers_mjpeg_when_available(self):
        output = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 1280x720
                        Interval: Discrete 0.033s (30.000 fps)
        [1]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 1280x720
                        Interval: Discrete 0.033s (30.000 fps)
"""
        run_result = mock.Mock(returncode=0, stdout=output)
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", return_value=run_result):
            self.assertEqual(
                select_camera_capture_format("/dev/video0", 1280, 720, 30),
                "mjpeg",
            )

    def test_camera_capture_candidates_keep_exact_geometry(self):
        output = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 640x360
                        Interval: Discrete 0.033s (30.000 fps)
        [1]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 640x360
                        Interval: Discrete 0.033s (30.000 fps)
        [2]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 1280x720
                        Interval: Discrete 0.033s (30.000 fps)
"""
        run_result = mock.Mock(returncode=0, stdout=output)
        with mock.patch(
            "nvbroadcast.video.virtual_camera.subprocess.run",
            return_value=run_result,
        ):
            candidates = camera_capture_candidates(
                "/dev/video0", 640, 360, 30
            )

        self.assertEqual(candidates, [
            {"format": "mjpeg", "width": 640, "height": 360, "fps": 30},
            {"format": "raw", "width": 640, "height": 360, "fps": 30},
        ])

    def test_camera_mode_candidates_step_down_from_high_fps_phone_mode(self):
        output = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 1280x720
                        Interval: Discrete 0.033s (30.000 fps)
        [1]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 640x480
                        Interval: Discrete 0.033s (30.000 fps)
"""
        run_result = mock.Mock(returncode=0, stdout=output)
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", return_value=run_result):
            candidates = camera_mode_candidates("/dev/video0", 1280, 720, 60)

        self.assertEqual(
            candidates[:2],
            [
                {"format": "mjpeg", "width": 1280, "height": 720, "fps": 30},
                {"format": "raw", "width": 640, "height": 480, "fps": 30},
            ],
        )
        self.assertEqual(
            select_camera_mode("/dev/video0", 1280, 720, 60),
            {"format": "mjpeg", "width": 1280, "height": 720, "fps": 30},
        )

    def test_camera_mode_candidates_parse_stepwise_phone_webcam_modes(self):
        output = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'MJPG' (Motion-JPEG, compressed)
                Size: Stepwise 320x240 - 1920x1080 with step 2/2
                        Interval: Stepwise 0.016s - 0.067s with step 0.001s
"""
        run_result = mock.Mock(returncode=0, stdout=output)
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", return_value=run_result):
            modes = list_camera_format_modes("/dev/video0")
            candidates = camera_mode_candidates("/dev/video0", 1280, 720, 60)

        self.assertIn(
            {"format": "MJPG", "width": 1280, "height": 720, "fps": [15, 24, 25, 30, 60, 62]},
            modes,
        )
        self.assertEqual(
            candidates[0],
            {"format": "mjpeg", "width": 1280, "height": 720, "fps": 60},
        )

    def test_list_camera_devices_skips_metadata_and_loopback_nodes(self):
        list_output = """
Dual Webcam:
        /dev/video0
        /dev/video1

NVIDIA Broadcast (platform:v4l2loopback-010):
        /dev/video10
"""
        info_by_device = {
            "/dev/video0": """
Driver Info:
        Card type        : Dual Webcam Metadata
Device Caps     : 0x04a00000
        Metadata Capture
        Streaming
""",
            "/dev/video1": """
Driver Info:
        Card type        : Dual Webcam
Device Caps     : 0x04200001
        Video Capture
        Streaming
""",
        }
        formats_by_device = {
            "/dev/video1": """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture
        [0]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 640x480
                        Interval: Discrete 0.033s (30.000 fps)
"""
        }

        def fake_run(args, **_kwargs):
            if args == ["v4l2-ctl", "--list-devices"]:
                return mock.Mock(returncode=0, stdout=list_output)
            if args[:3] == ["v4l2-ctl", "-D", "-d"]:
                return mock.Mock(returncode=0, stdout=info_by_device.get(args[3], ""))
            if len(args) == 4 and args[0:2] == ["v4l2-ctl", "-d"]:
                return mock.Mock(returncode=0, stdout=formats_by_device.get(args[2], ""))
            raise AssertionError(f"Unexpected command: {args}")

        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", side_effect=fake_run):
            self.assertEqual(
                list_camera_devices(),
                [{"name": "Dual Webcam", "device": "/dev/video1"}],
            )

    def test_list_camera_devices_keeps_capture_node_when_formats_probe_fails(self):
        list_output = """
USB Camera:
        /dev/video2
"""
        device_info = """
Driver Info:
        Card type        : USB Camera
Device Caps     : 0x04200001
        Video Capture
        Streaming
"""

        def fake_run(args, **_kwargs):
            if args == ["v4l2-ctl", "--list-devices"]:
                return mock.Mock(returncode=0, stdout=list_output)
            if args[:3] == ["v4l2-ctl", "-D", "-d"]:
                return mock.Mock(returncode=0, stdout=device_info)
            if len(args) == 4 and args[0:2] == ["v4l2-ctl", "-d"]:
                return mock.Mock(returncode=1, stdout="")
            raise AssertionError(f"Unexpected command: {args}")

        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", side_effect=fake_run):
            self.assertEqual(
                list_camera_devices(),
                [{"name": "USB Camera", "device": "/dev/video2"}],
            )


if __name__ == "__main__":
    unittest.main()

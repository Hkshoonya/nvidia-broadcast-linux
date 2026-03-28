import subprocess
import unittest
from unittest import mock

from nvbroadcast.video.virtual_camera import list_camera_modes


class CameraModesTests(unittest.TestCase):
    def setUp(self):
        list_camera_modes.cache_clear()

    def tearDown(self):
        list_camera_modes.cache_clear()

    def test_list_camera_modes_returns_empty_on_timeout(self):
        with mock.patch("nvbroadcast.video.virtual_camera.subprocess.run", side_effect=subprocess.TimeoutExpired("v4l2-ctl", 3)):
            self.assertEqual(list_camera_modes("/dev/video99"), [])

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


if __name__ == "__main__":
    unittest.main()

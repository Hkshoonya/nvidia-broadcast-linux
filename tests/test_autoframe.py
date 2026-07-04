import unittest
from unittest import mock

import numpy as np

from nvbroadcast.video.autoframe import AutoFrame


def _make_autoframe() -> AutoFrame:
    af = AutoFrame()
    af._initialized = True
    af._enabled = True
    return af


class AutoFrameTests(unittest.TestCase):
    def test_lateral_motion_below_old_dead_zone_still_smooths(self):
        af = _make_autoframe()
        frame = np.zeros((120, 160, 4), dtype=np.uint8)

        with mock.patch.object(af, "_detect_face", return_value=(0.56, 0.5)):
            af.process_frame_array(frame, 160, 120)

        self.assertGreater(af._smooth_cx, 0.5)
        self.assertLess(af._smooth_cx, 0.56)

    def test_detector_jitter_inside_dead_zone_is_ignored(self):
        af = _make_autoframe()
        frame = np.zeros((120, 160, 4), dtype=np.uint8)

        with mock.patch.object(af, "_detect_face", return_value=(0.505, 0.5)):
            af.process_frame_array(frame, 160, 120)

        self.assertEqual(af._smooth_cx, 0.5)

    def test_stable_mode_keeps_crop_center_fixed(self):
        af = _make_autoframe()
        af.mode = "stable"
        frame = np.zeros((120, 160, 4), dtype=np.uint8)

        with mock.patch.object(af, "_detect_face", return_value=(0.66, 0.5)):
            af.process_frame_array(frame, 160, 120)

        self.assertEqual(af._smooth_cx, 0.5)
        self.assertGreater(af._smooth_zoom, 1.0)

    def test_switching_to_stable_recenters_crop_immediately(self):
        af = _make_autoframe()
        af._smooth_cx = 0.62
        af._smooth_cy = 0.44

        af.mode = "stable"

        self.assertEqual(af._smooth_cx, 0.5)
        self.assertEqual(af._smooth_cy, 0.5)

    def test_switching_back_to_center_snaps_to_next_face(self):
        af = _make_autoframe()
        af.mode = "stable"
        af.mode = "center"
        frame = np.zeros((120, 160, 4), dtype=np.uint8)

        with mock.patch.object(af, "_detect_face", return_value=(0.66, 0.42)):
            af.process_frame_array(frame, 160, 120)

        self.assertEqual(af._smooth_cx, 0.66)
        self.assertEqual(af._smooth_cy, 0.42)

    def test_center_mode_framing_still_has_crop_margin_at_zero_zoom(self):
        af = _make_autoframe()
        af.zoom_level = 0.0
        af.smoothing = 0.0
        frame = np.zeros((120, 160, 4), dtype=np.uint8)

        with mock.patch.object(af, "_detect_face", return_value=(0.66, 0.5)):
            af.process_frame_array(frame, 160, 120)

        self.assertGreater(af._smooth_zoom, 1.0)

    def test_invalid_mode_falls_back_to_center(self):
        af = _make_autoframe()

        af.mode = "bogus"

        self.assertEqual(af.mode, "center")


if __name__ == "__main__":
    unittest.main()

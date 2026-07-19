import unittest

import cv2
import numpy as np

from nvbroadcast.video.eye_contact import EyeContactCorrector


class _Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class EyeContactTests(unittest.TestCase):
    @staticmethod
    def _moderate_right_gaze():
        frame = np.full((80, 80, 4), 100, dtype=np.uint8)
        frame[:, :, 3] = 77
        cv2.ellipse(frame, (40, 40), (20, 9), 0, 0, 360, (235, 235, 235, 77), -1)
        cv2.circle(frame, (50, 40), 5, (15, 15, 15, 77), -1)

        points = [
            (20, 40),
            (28, 32),
            (52, 32),
            (60, 40),
            (52, 48),
            (28, 48),
            (50, 40),
            (45, 40),
            (50, 35),
            (55, 40),
            (50, 45),
        ]
        landmarks = [_Point(x / 80, y / 80) for x, y in points]
        return frame, landmarks

    def test_default_intensity_moves_moderate_gaze_toward_eye_center(self):
        frame, landmarks = self._moderate_right_gaze()
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.intensity = 0.35

        output = corrector._correct_eye(
            frame.copy(),
            landmarks,
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            80,
            80,
        )

        before_dark = np.argwhere(frame[:, :, 0] < 50)
        after_dark = np.argwhere(output[:, :, 0] < 50)
        self.assertGreater(len(after_dark), 0)
        self.assertLess(after_dark[:, 1].mean(), before_dark[:, 1].mean() - 3.0)
        changed_channels = np.count_nonzero(
            output[:, :, :3] != frame[:, :, :3]
        )
        self.assertGreater(changed_channels, 100)
        self.assertTrue(np.array_equal(output[:, :, 3], frame[:, :, 3]))

    def test_zero_intensity_is_a_no_op(self):
        frame, landmarks = self._moderate_right_gaze()
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.intensity = 0.0

        output = corrector._correct_eye(
            frame.copy(),
            landmarks,
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            80,
            80,
        )

        self.assertTrue(np.array_equal(output, frame))

    def test_skips_unstable_large_gaze_offsets(self):
        frame = np.zeros((64, 64, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        frame[20:34, 16:36, :3] = 180

        landmarks = [
            _Point(16 / 64, 24 / 64),
            _Point(20 / 64, 20 / 64),
            _Point(32 / 64, 20 / 64),
            _Point(36 / 64, 24 / 64),
            _Point(32 / 64, 24 / 64),
            _Point(34 / 64, 24 / 64),
            _Point(35 / 64, 24 / 64),
            _Point(36 / 64, 24 / 64),
            _Point(37 / 64, 24 / 64),
        ]

        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.intensity = 1.0

        output = corrector._correct_eye(
            frame.copy(),
            landmarks,
            [0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            64,
            64,
        )

        self.assertTrue(np.array_equal(output, frame))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from nvbroadcast.video.eye_contact import EyeContactCorrector


class _Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class EyeContactTests(unittest.TestCase):
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

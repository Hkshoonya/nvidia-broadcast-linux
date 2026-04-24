from __future__ import annotations

import unittest

import numpy as np

from nvbroadcast.video.beautify import FaceBeautifier


class BeautifyTests(unittest.TestCase):
    def _make_beautifier(self) -> FaceBeautifier:
        beautifier = FaceBeautifier()
        beautifier._initialized = True
        beautifier._enabled = True
        beautifier.denoise = 0.5
        beautifier._face_bbox = (2, 2, 4, 4)
        beautifier._face_mask = np.zeros((8, 8), dtype=np.uint8)
        beautifier._face_mask[2:6, 2:6] = 255
        return beautifier

    def test_denoise_preserves_raw_history(self):
        beautifier = self._make_beautifier()
        previous = np.full((8, 8, 3), 200, dtype=np.uint8)
        beautifier._prev_frame = previous.copy()

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, :3] = 40
        frame[:, :, 3] = 255
        raw_bgr = frame[:, :, :3].copy()

        beautifier._apply_denoise(frame)

        self.assertTrue(
            np.array_equal(beautifier._prev_frame, raw_bgr),
            "temporal denoise must keep raw frame history instead of recursively storing blurred output",
        )

    def test_denoise_does_not_blur_full_frame_without_face_mask(self):
        beautifier = FaceBeautifier()
        beautifier._initialized = True
        beautifier._enabled = True
        beautifier.denoise = 0.5
        beautifier._prev_frame = np.full((8, 8, 3), 220, dtype=np.uint8)

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, :3] = 30
        frame[:, :, 3] = 255
        original = frame.copy()

        result = beautifier._apply_denoise(frame)

        self.assertTrue(
            np.array_equal(result[:, :, :3], original[:, :, :3]),
            "without a face mask, denoise should not smear the whole composited frame",
        )
        self.assertTrue(
            np.array_equal(beautifier._prev_frame, original[:, :, :3]),
            "raw history should still update when face landmarks are unavailable",
        )

    def test_denoise_motion_gate_keeps_fast_motion_close_to_current_frame(self):
        beautifier = self._make_beautifier()
        beautifier._prev_frame = np.full((8, 8, 3), 255, dtype=np.uint8)

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        before = frame[3:5, 3:5, :3].copy()

        result = beautifier._apply_denoise(frame)
        after = result[3:5, 3:5, :3].astype(np.int16)

        self.assertLess(
            int(after.mean()),
            60,
            "fast motion should heavily reduce temporal blending so the face does not ghost",
        )
        self.assertTrue(np.array_equal(before, np.zeros_like(before)))


if __name__ == "__main__":
    unittest.main()

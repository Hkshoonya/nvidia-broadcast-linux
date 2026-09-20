"""Keep bounded color donors intact when processing sparse fringe windows."""

import unittest

import numpy as np

from nvbroadcast.video.effects import VideoEffects


class ReplaceColorReferenceTests(unittest.TestCase):
    def test_sparse_windows_keep_ten_pixel_donors_across_both_tile_axes(self):
        for seam in (127, 128, 129, 511, 512, 513):
            for transpose in (False, True):
                with self.subTest(seam=seam, transpose=transpose):
                    frame = np.full((530, 550, 4), 110, dtype=np.uint8)
                    frame[:, :, 3] = 255
                    alpha = np.zeros(frame.shape[:2], dtype=np.float32)
                    alpha[130, seam - 1:seam + 1] = 0.5
                    alpha[130, seam + 10] = 1.0
                    frame[130, seam + 10, :3] = (24, 42, 60)
                    expected = frame.copy()
                    expected[130, seam, :3] = (24, 42, 60)
                    if transpose:
                        frame = frame.transpose(1, 0, 2)
                        alpha = alpha.T
                        expected = expected.transpose(1, 0, 2)
                    original_frame, original_alpha = frame.copy(), alpha.copy()

                    reference = VideoEffects._nearest_foreground_color(frame, alpha, 0.94)

                    np.testing.assert_array_equal(reference, expected)
                    np.testing.assert_array_equal(frame, original_frame)
                    np.testing.assert_array_equal(alpha, original_alpha)

    def test_fringe_and_solid_thresholds_remain_strict(self):
        frame = np.full((530, 550, 4), 110, dtype=np.uint8)
        frame[:, :, 3] = 255
        alpha = np.zeros(frame.shape[:2], dtype=np.float32)
        values = (
            np.float32(0.025),
            np.nextafter(np.float32(0.025), np.float32(1.0)),
            np.float32(0.94),
            np.nextafter(np.float32(0.94), np.float32(1.0)),
        )
        expected = frame.copy()
        for idx, value in enumerate(values):
            row = 127 + idx * 20
            alpha[row, 127] = value
            alpha[row, 129] = 1.0
            frame[row, 129, :3] = expected[row, 129, :3] = (24, 42, 60)
            if idx in (1, 2):
                expected[row, 127, :3] = (24, 42, 60)

        reference = VideoEffects._nearest_foreground_color(frame, alpha, 0.94)

        np.testing.assert_array_equal(reference, expected)

    def test_strided_cleanup_preserves_dark_border_and_opaque_white_clothing(self):
        backing = np.full((1060, 1100, 4), 255, dtype=np.uint8)
        frame = backing[::2, ::2]
        alpha = np.zeros((1060, 1100), dtype=np.float32)[::2, ::2]
        alpha[120:150, 128] = 0.75
        alpha[120:150, 129:180] = 1.0
        frame[120:150, 128, :3] = 81
        frame[120:150, 129, :3] = 24
        frame[120:150, 130:180, :3] = 240
        alpha[400:420, 400:420] = 0.5
        frame[400:420, 400:420, :3] = 110
        original_frame, original_alpha = frame.copy(), alpha.copy()
        effects = VideoEffects(compositing="cpu")
        effects._bg_mode = "replace"

        cleaned = effects._despill_fringe(frame, alpha)

        self.assertTrue(np.all(cleaned[120:150, 128, :3] <= 60))
        self.assertTrue(np.all(cleaned[120:150, 129, :3] == 24))
        self.assertTrue(np.all(cleaned[120:150, 130:180, :3] == 240))
        self.assertTrue(np.all(cleaned[400:420, 400:420, :3] == 110))
        np.testing.assert_array_equal(frame, original_frame)
        np.testing.assert_array_equal(alpha, original_alpha)


if __name__ == "__main__":
    unittest.main()

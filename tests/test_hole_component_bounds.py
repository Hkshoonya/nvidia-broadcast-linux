"""Keep per-component hole repair confined to the selected component."""

import unittest

import numpy as np

from nvbroadcast.video.effects import VideoEffects


class HoleComponentBoundsTests(unittest.TestCase):
    @staticmethod
    def _mask():
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[4:60, 4:92] = 255
        return mask

    def test_fill_changes_only_small_hole_pixels_below_cutoff(self):
        mask = self._mask()
        mask[20:28, 20:22] = 10
        mask[22:24, 20] = 80
        mask[35:45, 55:65] = 0
        original = mask.copy()
        expected = mask.copy()
        expected[20:28, 20:22] = np.where(
            original[20:28, 20:22] < 50, 180, original[20:28, 20:22],
        )

        result = VideoEffects._fill_small_internal_holes(
            mask, binary_threshold=100, fill_cutoff=50, fill_value=180,
            max_area_ratio=0.004, max_span_ratio=0.2,
        )

        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(mask, original)

    def test_preserve_selects_elongated_hole_without_its_box_neighbors(self):
        mask = self._mask()
        mask[20:32, 20:22] = 10
        mask[40:43, 40:43] = 0
        mask[35:45, 60:70] = 0
        expected = np.zeros_like(mask, dtype=bool)
        expected[20:32, 20:22] = True
        original = mask.copy()

        result = VideoEffects._preserve_large_internal_holes(
            mask, binary_threshold=90, min_area_ratio=0.001,
            min_span_ratio=0.02, min_aspect_ratio=2.0,
            max_area_ratio=0.005,
        )

        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(mask, original)

    def test_disjoint_holes_in_strided_masks(self):
        for stride in (2, 3):
            with self.subTest(stride=stride):
                backing = np.zeros((64 * stride, 96 * stride), dtype=np.uint8)
                mask = backing[::stride, ::stride]
                mask[4:60, 4:92] = 255
                expected = mask.copy()
                for y, x in ((10, 10), (10, 30), (30, 10), (45, 70)):
                    mask[y:y + 3, x:x + 2] = 0
                    expected[y:y + 3, x:x + 2] = 160
                original = mask.copy()

                result = VideoEffects._fill_small_internal_holes(
                    mask, binary_threshold=72, fill_cutoff=46, fill_value=160,
                    max_area_ratio=0.00014, max_span_ratio=0.08,
                )

                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(mask, original)


if __name__ == '__main__':
    unittest.main()

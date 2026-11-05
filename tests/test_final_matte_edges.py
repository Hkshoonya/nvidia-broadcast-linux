"""Preserve owned replacement mattes and previous-edge history."""

import threading
import unittest
from unittest import mock

import numpy as np

from nvbroadcast.video.effects import VideoEffects


class FinalMatteEdgeTests(unittest.TestCase):
    def test_hd_refinement_returns_an_owned_writable_float32_matte(self):
        effects = VideoEffects.__new__(VideoEffects)
        effects._quality = "quality"
        frame = np.zeros((540, 960, 4), dtype=np.uint8)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                matte = np.zeros(frame.shape[:2], dtype=dtype)
                matte[100:110, 200:220] = 0.5
                original = matte.copy()
                with mock.patch.object(
                    effects, "_edge_aware_replace_matte_region",
                    side_effect=lambda _frame, region, *_args, **_kwargs:
                        np.full_like(region, 0.75),
                ) as refine:
                    result = effects._edge_aware_replace_matte(frame, matte)
                self.assertEqual(refine.call_count, 1)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(result.flags.writeable)
                self.assertFalse(np.shares_memory(result, matte))
                self.assertEqual(float(result[105, 210]), 0.75)
                np.testing.assert_array_equal(matte, original)

    def test_edge_stabilization_leaves_solid_pixels_and_previous_history_unchanged(self):
        effects = VideoEffects.__new__(VideoEffects)
        effects._state_lock = threading.RLock()
        previous = np.zeros((48, 64), dtype=np.uint8)
        previous[8:16, 12:20] = 128
        previous[20:28, 30:38] = 200
        previous[1:4, 1:4] = 255
        effects._latest_final_matte_u8 = previous
        effects._latest_final_matte_size = (64, 48)
        matte = np.zeros(previous.shape, dtype=np.float32)
        matte[8:16, 12:20] = 0.4
        matte[20:28, 30:38] = 0.6
        matte[1:4, 1:4] = 1.0
        original, history = matte.copy(), previous.copy()

        result = effects._stabilize_replace_matte_edges(matte)

        self.assertIs(result, matte)
        self.assertTrue(np.all(result[8:16, 12:20] > original[8:16, 12:20]))
        self.assertTrue(np.all(result[8:16, 12:20] < 128 / 255))
        self.assertTrue(np.all(result[20:28, 30:38] > original[20:28, 30:38]))
        self.assertTrue(np.all(result[20:28, 30:38] < 200 / 255))
        solid = (original == 0) | (original == 1)
        np.testing.assert_array_equal(result[solid], original[solid])
        np.testing.assert_array_equal(previous, history)


if __name__ == "__main__":
    unittest.main()

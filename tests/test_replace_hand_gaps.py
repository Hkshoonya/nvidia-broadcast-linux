"""Keep actual camera-background openings between moving fingers transparent."""

import unittest

import numpy as np

from nvbroadcast.video.effects import VideoEffects


class _HandAlphaBackend:
    _lowres_refined = False

    def __init__(self, alpha):
        self.alpha = alpha

    def infer(self, _frame, _width, _height):
        return self.alpha.copy()


class ReplaceHandGapTests(unittest.TestCase):
    @staticmethod
    def _hand_scene(height, width, gap, length, shift=0):
        y, x = height // 3, width // 3 + shift
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[y:y + length, x - 12:x] = 1.0
        alpha[y:y + length, x + gap:x + gap + 12] = 1.0
        # Both fingers join the same opaque palm below the exterior opening.
        alpha[y + length:y + length + 12, x - 12:x + gap + 12] = 1.0
        frame = np.full((height, width, 4), 255, dtype=np.uint8)
        frame[alpha > 0, :3] = 64
        return frame, alpha, y, x

    @staticmethod
    def _effects(alpha):
        effects = VideoEffects(compositing="cpu")
        effects._quality = "ultra"
        effects._bg_mode = "replace"
        effects._use_fused_kernel = True
        effects._bg_removal_enabled = True
        effects._initialized = True
        effects._backend = _HandAlphaBackend(alpha)
        effects._learned_refiners = {}
        effects._bg_image = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        effects._bg_image[:, :, 3] = 255
        effects._refresh_temporal_strength()
        effects.update_edge_params(
            dilate_size=3, blur_size=5, sigmoid_strength=14, sigmoid_midpoint=0.45,
        )
        return effects

    def test_short_exterior_finger_gaps_survive_one_pixel_motion_at_both_resolutions(self):
        for height, width in ((360, 640), (720, 1280)):
            for gap in (1, 2):
                effects = None
                for shift in (0, 1, 2, 3, 2, 1):
                    with self.subTest(size=(width, height), gap=gap, shift=shift):
                        frame, alpha, y, x = self._hand_scene(
                            height, width, gap, length=20, shift=shift,
                        )
                        if effects is None:
                            effects = self._effects(alpha)
                        else:
                            effects._backend.alpha = alpha

                        output = effects.process_frame_array(frame, width, height)
                        matte = effects.latest_final_matte_u8(width, height)

                        self.assertTrue(np.all(alpha[y + 5:y + 15, x:x + gap] == 0))
                        self.assertTrue(
                            np.all(matte[y + 5:y + 15, x:x + gap] == 0),
                            "A true exterior gap must not become opaque after refinement.",
                        )
                        self.assertTrue(np.all(output[y + 5:y + 15, x:x + gap, :3] == 0))
                        self.assertGreaterEqual(int(matte[y + 26, x]), 250)
                        self.assertTrue(np.all(output[y + 26, x, :3] == 64))
                        self.assertGreaterEqual(int(matte[y + 10, x - 6]), 250)
                        self.assertTrue(np.all(output[y + 10, x - 6, :3] == 64))

    def test_hd_replacement_preserves_long_thin_gap_through_downsampling(self):
        height, width = 720, 1280
        effects = None
        for shift in (0, 1, 2, 3, 2, 1):
            with self.subTest(shift=shift):
                frame, alpha, y, x = self._hand_scene(
                    height, width, gap=1, length=60, shift=shift,
                )
                if effects is None:
                    effects = self._effects(alpha)
                else:
                    effects._backend.alpha = alpha

                output = effects.process_frame_array(frame, width, height)
                matte = effects.latest_final_matte_u8(width, height)

                self.assertLess(float(effects._cached_alpha[y + 30, x]), 0.03)
                self.assertEqual(int(matte[y + 30, x]), 0)
                self.assertTrue(np.all(output[y + 30, x, :3] == 0))
                self.assertGreaterEqual(int(matte[y + 66, x]), 250)

    def test_enclosed_segmentation_hole_still_repairs_opaque_white_foreground(self):
        height, width = 720, 1280
        frame, alpha, y, x = self._hand_scene(height, width, gap=2, length=20)
        # This low-confidence area is inside the palm, with no connection to
        # the camera background. Its real white foreground still needs repair.
        hole = (slice(y + 24, y + 27), slice(x - 6, x - 3))
        alpha[hole] = 0.0
        frame[hole][:, :, :3] = 240
        effects = self._effects(alpha)

        output = effects.process_frame_array(frame, width, height)
        matte = effects.latest_final_matte_u8(width, height)

        self.assertGreaterEqual(int(matte[y + 25, x - 5]), 250)
        self.assertTrue(np.all(output[y + 25, x - 5, :3] == 240))
        self.assertTrue(np.all(output[y + 10, x:x + 2, :3] == 0))


if __name__ == "__main__":
    unittest.main()

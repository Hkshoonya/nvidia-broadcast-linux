"""Synthetic moving-hand coverage for Replace temporal stabilization."""

import unittest

import numpy as np

from nvbroadcast.video.effects import VideoEffects


class _HandAlphaBackend:
    _lowres_refined = False

    def __init__(self, alpha):
        self.alpha = alpha

    def infer(self, _frame, _width, _height):
        return self.alpha.copy()


class ReplaceHandMotionTests(unittest.TestCase):
    @staticmethod
    def _scene(finger_width, translation=0):
        height, width = 360, 640
        hand = np.zeros((height, width), dtype=np.float32)
        hand[95:180, 350:350 + finger_width] = 1.0
        hand[175:215, 336:386] = 1.0
        alpha = np.roll(hand, translation, axis=1)
        alpha[100:340, 140:280] = 1.0

        subject = np.full((height, width, 3), 80, dtype=np.float32)
        subject[100:340, 140:280] = 24
        subject[235:325, 180:250] = 240
        frame = np.empty((height, width, 4), dtype=np.uint8)
        frame[:, :, :3] = (
            subject * alpha[:, :, np.newaxis]
            + 255 * (1.0 - alpha[:, :, np.newaxis])
        ).astype(np.uint8)
        frame[:, :, 3] = 255
        return frame, alpha

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
            dilate_size=3, blur_size=5,
            sigmoid_strength=14, sigmoid_midpoint=0.45,
        )
        return effects

    def test_moving_fingers_release_old_pixels_and_preserve_current_subject(self):
        for finger_width in (1, 2, 4, 8):
            for translation in (1, 4, 12):
                with self.subTest(width=finger_width, translation=translation):
                    before_frame, before_alpha = self._scene(finger_width)
                    effects = self._effects(before_alpha)
                    height, width = before_alpha.shape
                    effects.process_frame_array(before_frame, width, height)
                    frame, alpha = self._scene(finger_width, translation)
                    effects._backend.alpha = alpha

                    output = effects.process_frame_array(frame, width, height)
                    matte = effects.latest_final_matte_u8(width, height)

                    finger_region = np.zeros_like(alpha, dtype=bool)
                    finger_region[110:155, 300:450] = True
                    vacated = finger_region & (before_alpha == 1.0) & (alpha == 0.0)
                    self.assertTrue(vacated.any())
                    self.assertTrue(
                        np.all(output[vacated, :3] == 0),
                        "A finger leaving white camera background must not "
                        "leave a bright trail on the black replacement.",
                    )
                    self.assertTrue(np.all(matte[vacated] == 0))

                    x0 = 350 + translation
                    current_finger = matte[110:155, x0:x0 + finger_width]
                    if finger_width >= 2:
                        self.assertGreaterEqual(int(current_finger.min()), 250)
                        self.assertTrue(
                            np.all(output[110:155, x0:x0 + finger_width, :3] == 80),
                            "Removing a trail must preserve the newly positioned finger.",
                        )
                    else:
                        # A one-pixel finger is spatially feathered; it must
                        # remain visible when moving instead of disappearing.
                        self.assertGreaterEqual(int(current_finger.min()), 75)
                    self.assertTrue(
                        np.all(output[250:310, 190:240, :3] == 240),
                        "Opaque white clothing must preserve its source color.",
                    )

    def test_stationary_edge_jitter_still_receives_temporal_smoothing(self):
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[12:52, 12:52] = 0.5
        effects = self._effects(alpha)
        effects._temporal_smooth(alpha)
        inputs, outputs = [], []

        for value in (0.515, 0.485) * 8:
            current = alpha.copy()
            current[12:52, 12:52] = value
            smoothed = effects._temporal_smooth(current)
            inputs.append(value)
            outputs.append(float(smoothed[30, 30]))

        self.assertGreater(outputs[0], 0.5)
        self.assertLess(outputs[0], inputs[0])
        self.assertGreaterEqual(min(outputs), min(inputs))
        self.assertLessEqual(max(outputs), max(inputs))
        self.assertLess(
            np.ptp(outputs), np.ptp(inputs) * 0.75,
            "Fast local-motion tracking must retain damping of small "
            "stationary edge fluctuations.",
        )


if __name__ == "__main__":
    unittest.main()

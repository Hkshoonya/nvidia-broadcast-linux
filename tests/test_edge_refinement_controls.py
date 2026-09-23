"""Live edge controls affect every background mode without changing defaults."""

import unittest
from unittest import mock

import cv2
import numpy as np

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig, EdgeConfig
from nvbroadcast.ui.window import NVBroadcastWindow
from nvbroadcast.video.effects import VideoEffects


class _AlphaBackend:
    _lowres_refined = False

    def __init__(self, alpha):
        self.alpha = alpha

    def infer(self, _frame, _width, _height):
        return self.alpha.copy()


class EdgeRefinementControlTests(unittest.TestCase):
    @staticmethod
    def _alpha():
        alpha = np.zeros((360, 640), dtype=np.float32)
        alpha[80:310, 160:480] = 1.0
        return alpha

    @staticmethod
    def _effects(mode, quality="ultra"):
        effects = VideoEffects(compositing="cpu", edge_config=EdgeConfig())
        effects._bg_mode = mode
        effects._quality = quality
        return effects

    def test_no_config_uses_the_shipped_spatial_defaults(self):
        effects = VideoEffects(compositing="cpu")
        defaults = EdgeConfig()
        self.assertEqual(effects._dilate_size, defaults.dilate_size)
        self.assertEqual(effects._blur_size, defaults.blur_size)

    def test_default_spatial_operations_preserve_each_existing_ladder(self):
        alpha = self._alpha()
        for mode, quality, expected_gaussians, expected_dilation in (
            ("replace", "ultra", [3], []),
            ("replace", "balanced", [5, 3, 3], []),
            ("remove", "ultra", [17, 11, 7, 11], [(3, 1)]),
            ("blur", "ultra", [5, 3], [(3, 1)]),
        ):
            with self.subTest(mode=mode, quality=quality):
                effects = self._effects(mode, quality)
                with mock.patch.object(cv2, "GaussianBlur", wraps=cv2.GaussianBlur) as blur, \
                        mock.patch.object(cv2, "dilate", wraps=cv2.dilate) as dilate:
                    effects._refine_alpha(alpha)
                self.assertEqual(
                    [call.args[1][0] for call in blur.call_args_list],
                    expected_gaussians,
                )
                self.assertEqual(
                    [(call.args[1].shape[0], call.kwargs.get("iterations", 1))
                     for call in dilate.call_args_list],
                    expected_dilation,
                )

    def test_remove_large_edge_settings_keep_existing_broad_ladder(self):
        effects = self._effects("remove", "ultra")
        effects.update_edge_params(dilate_size=15, blur_size=25)
        with mock.patch.object(cv2, "morphologyEx", wraps=cv2.morphologyEx) as morphology, \
                mock.patch.object(cv2, "dilate", wraps=cv2.dilate) as dilate:
            effects._refine_alpha(self._alpha())

        self.assertIn((25, 25), [call.args[2].shape for call in morphology.call_args_list])
        self.assertIn((31, 2), [
            (call.args[1].shape[0], call.kwargs.get("iterations", 1))
            for call in dilate.call_args_list
        ])

    def test_dilate_moves_the_outer_boundary_in_each_mode(self):
        alpha = self._alpha()
        for mode in ("replace", "remove", "blur"):
            for quality in ("ultra", "balanced"):
                with self.subTest(mode=mode, quality=quality):
                    effects = self._effects(mode, quality)
                    totals = []
                    for value in (0, 3, 15):
                        effects.update_edge_params(dilate_size=value)
                        totals.append(float(effects._refine_alpha(alpha).sum()))
                    self.assertLess(totals[0], totals[1])
                    self.assertLess(totals[1], totals[2])

    def test_softness_widens_the_transition_in_each_mode(self):
        alpha = self._alpha()
        for mode in ("replace", "remove", "blur"):
            for quality in ("ultra", "balanced"):
                with self.subTest(mode=mode, quality=quality):
                    effects = self._effects(mode, quality)
                    transitions = []
                    for value in (1, 5, 25):
                        effects.update_edge_params(blur_size=value, sigmoid_strength=0)
                        result = effects._refine_alpha(alpha)
                        transitions.append(np.count_nonzero((result > 0.05) & (result < 0.95)))
                    self.assertLess(transitions[0], transitions[1])
                    self.assertLess(transitions[1], transitions[2])

    def test_returning_to_defaults_restores_output_without_mutating_source(self):
        alpha = self._alpha()[:, ::-1]
        original = alpha.copy()
        alpha.setflags(write=False)
        for mode in ("replace", "remove", "blur"):
            for quality in ("ultra", "balanced"):
                with self.subTest(mode=mode, quality=quality):
                    effects = self._effects(mode, quality)
                    baseline = effects._refine_alpha(alpha)
                    effects.update_edge_params(dilate_size=15, blur_size=25)
                    adjusted = effects._refine_alpha(alpha)
                    effects.update_edge_params(dilate_size=3, blur_size=5)
                    restored = effects._refine_alpha(alpha)
                    self.assertFalse(np.array_equal(baseline, adjusted))
                    np.testing.assert_array_equal(restored, baseline)
                    np.testing.assert_array_equal(alpha, original)
                    self.assertFalse(np.shares_memory(restored, alpha))

    @mock.patch("nvbroadcast.app.save_config")
    def test_slider_to_saved_config_to_composited_frame(self, save_config):
        alpha = self._alpha()
        frame = np.full((*alpha.shape, 4), 255, dtype=np.uint8)
        frame[alpha == 1, :3] = 64
        for mode in ("replace", "remove"):
            for handler, param, values in (
                (NVBroadcastWindow._on_edge_dilate, "dilate_size", (0, 15)),
                (NVBroadcastWindow._on_edge_blur, "blur_size", (1, 25)),
            ):
                with self.subTest(mode=mode, param=param):
                    outputs = []
                    for value in values:
                        effects = self._effects(mode)
                        effects._bg_removal_enabled = True
                        effects._initialized = True
                        effects._backend = _AlphaBackend(alpha)
                        effects._learned_refiners = {}
                        effects._bg_image = np.zeros_like(frame)
                        effects._bg_image[:, :, 3] = 255
                        app = NVBroadcastApp.__new__(NVBroadcastApp)
                        app.config = AppConfig()
                        app._video_effects = effects
                        window = mock.Mock(_app=app)
                        handler(window, None, value)
                        self.assertEqual(getattr(app.config.video.edge, param), value)
                        outputs.append(effects.process_frame_array(frame, 640, 360))
                    self.assertFalse(np.array_equal(*outputs))
        self.assertEqual(save_config.call_count, 8)


if __name__ == "__main__":
    unittest.main()

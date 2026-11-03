"""Preserve the CPU refinement tone curve and source-gap quantization."""

import unittest
from unittest import mock

import cv2
import numpy as np

from nvbroadcast.video.effects import VideoEffects


def _legacy_tone(a8, is_replace, preserve_detail, strength, midpoint):
    """Reference the original float32 mapping before its final quantization."""
    result = a8.astype(np.float32) * (1.0 / 255.0)
    scale = (0.42 if preserve_detail else 0.60) if is_replace else 0.45
    sig = strength * scale
    if sig > 0:
        result = 1.0 / (1.0 + np.exp(-sig * (result - midpoint)))
    threshold = (0.82 if preserve_detail else 0.78) if is_replace else 0.75
    exponent = (1.8 if preserve_detail else 2.2) if is_replace else 2.0
    noise = (0.025 if preserve_detail else 0.03) if is_replace else 0.02
    core = result > threshold
    result[core] = 1.0 - (1.0 - result[core]) ** exponent
    result[result < noise] = 0.0
    return result


class RefinementToneLutTests(unittest.TestCase):
    def setUp(self):
        self.effects = VideoEffects(compositing="cpu")

    def test_every_input_byte_matches_float32_curve_for_all_mode_branches(self):
        a8 = np.arange(256, dtype=np.uint8)
        for mode in ("replace", "blur", "remove"):
            for quality in ("ultra", "quality", "balanced", "performance"):
                is_replace = mode == "replace"
                detail = is_replace and quality in ("ultra", "quality")
                for strength in (-8.0, 0.0, 0.00001, 1.0, 12.0, 14.0, 24.0, 100.0):
                    for midpoint in (-0.2, 0.1, 0.45, 0.5, 0.9, 1.2):
                        with self.subTest(mode=mode, quality=quality,
                                          strength=strength, midpoint=midpoint):
                            self.effects.update_edge_params(
                                sigmoid_strength=strength,
                                sigmoid_midpoint=midpoint,
                            )
                            expected = np.clip(
                                _legacy_tone(a8, is_replace, detail, strength, midpoint) * 255,
                                0, 255,
                            ).astype(np.uint8)
                            actual = self.effects._refinement_tone_lut(is_replace, detail)
                            np.testing.assert_array_equal(actual, expected)

    def test_live_quality_mode_and_curve_changes_invalidate_the_cached_mapping(self):
        alpha = np.tile(np.arange(256, dtype=np.float32) / 255, (32, 1))
        previous = None
        for mode, quality, strength, midpoint in (
            ("replace", "ultra", 14.0, 0.45),
            ("replace", "balanced", 14.0, 0.45),
            ("replace", "ultra", 14.0, 0.45),
            ("replace", "ultra", 0.0, 0.45),
            ("replace", "ultra", 0.0, 0.9),
            ("blur", "ultra", 0.0, 0.9),
            ("remove", "balanced", 12.0, 0.5),
        ):
            with self.subTest(mode=mode, quality=quality, strength=strength,
                              midpoint=midpoint):
                self.effects._bg_mode = mode
                self.effects._quality = quality
                self.effects.update_edge_params(
                    sigmoid_strength=strength, sigmoid_midpoint=midpoint,
                )
                self.effects._refine_alpha_full(alpha)
                cached = self.effects._refinement_tone_cache
                self.assertIsNot(cached, previous)
                detail = mode == "replace" and quality in ("ultra", "quality")
                self.assertEqual(cached[0], (mode == "replace", detail, strength, midpoint))
                expected = np.clip(
                    _legacy_tone(
                        np.arange(256, dtype=np.uint8), mode == "replace",
                        detail, strength, midpoint,
                    ) * 255, 0, 255,
                ).astype(np.uint8)
                np.testing.assert_array_equal(cached[1], expected)
                self.effects._refine_alpha_full(alpha)
                self.assertIs(self.effects._refinement_tone_cache, cached)
                previous = cached

    def test_noncontiguous_uint8_spatial_result_uses_the_same_blur_curve(self):
        spatial = np.arange(64 * 512, dtype=np.uint8).reshape(64, 512)[:, ::2]
        self.assertFalse(spatial.flags.c_contiguous)
        self.effects._bg_mode = "blur"
        self.effects.update_edge_params(
            dilate_size=0, blur_size=0, sigmoid_strength=14, sigmoid_midpoint=0.45,
        )
        expected = np.clip(
            _legacy_tone(spatial, False, False, 14, 0.45) * 255, 0, 255,
        ).astype(np.uint8)
        expected = cv2.GaussianBlur(expected, (3, 3), 0).astype(np.float32) * (1.0 / 255.0)
        with mock.patch.object(self.effects, "_fill_small_internal_holes", return_value=spatial):
            actual = self.effects._refine_alpha_full(spatial.astype(np.float32) / 255)
        np.testing.assert_array_equal(actual, expected)

    def test_source_holes_and_slits_survive_before_and_after_final_feathering(self):
        shape = (48, 64)
        holes = np.zeros(shape, dtype=bool)
        slits = np.zeros(shape, dtype=bool)
        holes[20:34, 18:20] = True
        slits[:18, 34:36] = True
        spatial = np.full(shape, 230, dtype=np.uint8)
        self.effects._bg_mode = "replace"
        self.effects.update_edge_params(sigmoid_strength=14, sigmoid_midpoint=0.45)
        for quality in ("ultra", "balanced"):
            for dtype in (np.float32, np.float64):
                with self.subTest(quality=quality, dtype=dtype):
                    self.effects._quality = quality
                    detail = quality == "ultra"
                    alpha = np.ones(shape, dtype=dtype)
                    # Float64 immediately below a byte boundary rounds upward
                    # when assigned into the legacy float32 result array.
                    alpha[holes] = np.nextafter(dtype(17 / 255), dtype(0))
                    alpha[slits] = dtype(0.0711)
                    expected = _legacy_tone(spatial, True, detail, 14, 0.45)
                    expected[holes] = np.minimum(expected[holes], alpha[holes])
                    expected[slits] = np.minimum(expected[slits], alpha[slits])
                    expected = np.clip(expected * 255, 0, 255).astype(np.uint8)
                    if not detail:
                        expected = cv2.GaussianBlur(expected, (3, 3), 0)
                    expected = expected.astype(np.float32) * (1.0 / 255.0)
                    expected[holes] = np.minimum(expected[holes], alpha[holes])
                    expected[slits] = np.minimum(expected[slits], alpha[slits])

                    def preserve_slits(original, *_args, **_kwargs):
                        return slits.copy() if original.shape == shape else np.zeros_like(original, dtype=bool)

                    with (
                        mock.patch.object(self.effects, "_preserve_large_internal_holes", return_value=holes),
                        mock.patch.object(self.effects, "_preserve_narrow_exterior_gaps", side_effect=preserve_slits),
                        mock.patch.object(self.effects, "_fill_small_internal_holes", return_value=spatial),
                    ):
                        actual = self.effects._refine_alpha_full(alpha)
                    np.testing.assert_array_equal(actual, expected)

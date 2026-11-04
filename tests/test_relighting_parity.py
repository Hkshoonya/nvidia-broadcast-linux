"""Pixel/state differentials against the literal pre-optimization implementation."""
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from nvbroadcast.video.relighting import FaceRelighter
from tests.fixtures.relighting_reference import FaceRelighter as ReferenceRelighter
from tests.test_relighting import _Point


def landmarks(left=0.30, top=0.25, right=0.60, bottom=0.70):
    return [_Point(left, top), _Point(right, top), _Point(right, bottom), _Point(left, bottom)]


class RelightingParityTests(unittest.TestCase):
    def test_literal_reference_source_is_frozen(self):
        path = Path(__file__).parent / "fixtures/relighting_reference.py"
        literal_source = b"\n".join(path.read_bytes().split(b"\n")[2:])
        self.assertEqual(hashlib.sha256(literal_source).hexdigest(),
                         "040b3097ef75b85da4a764ef1b493799e20e290c922a224e6d845eda4d8402dc")

    def assert_same(self, actual, expected):
        self.assertEqual(actual._analyze_count, expected._analyze_count)
        self.assertEqual(actual._bg_luminance.hex(), expected._bg_luminance.hex())
        self.assertEqual(actual._bg_warmth.hex(), expected._bg_warmth.hex())

    def test_random720p_weighted_background_statistics_are_exact(self):
        rng = np.random.default_rng(371)
        frame = rng.integers(0, 256, (720, 1280, 4), dtype=np.uint8)
        random_alpha = rng.integers(0, 256, frame.shape[:2], dtype=np.uint8)
        for variant, source in (("bgra", frame), ("bgr", frame[:, :, :3]),
                                ("reverse", frame[::-1, ::-1]), ("fortran", np.asfortranarray(frame))):
            for mask_name, alpha in (("none", None), ("full", np.zeros(frame.shape[:2], np.uint8)),
                                     ("foreground", np.full(frame.shape[:2], 255, np.uint8)),
                                     ("random", random_alpha), ("reverse", random_alpha[::-1, ::-1])):
                actual, expected = FaceRelighter(), ReferenceRelighter()
                with self.subTest(frame=variant, alpha=mask_name):
                    actual._analyze_background(source, alpha)
                    expected._analyze_background(source, alpha)
                    self.assert_same(actual, expected)

    def test_alpha_count_boundary_and_nonbyte_fallback_are_exact(self):
        rng = np.random.default_rng(949)
        frame = rng.integers(0, 256, (49, 73, 4), dtype=np.uint8)
        for dtype in (np.uint8, np.uint16, np.float32):
            for count in (0, 100, 101, 102, frame.shape[0] * frame.shape[1]):
                alpha = np.full(frame.shape[:2], 128, dtype=np.uint8)
                alpha.flat[:count] = 127
                actual, expected = FaceRelighter(), ReferenceRelighter()
                with self.subTest(dtype=dtype, background_pixels=count):
                    actual._analyze_background(frame.astype(dtype), alpha)
                    expected._analyze_background(frame.astype(dtype), alpha)
                    self.assert_same(actual, expected)

    def test_moving_clipped_hulls_strength_warmth_and_cadence_match_pixels(self):
        rng = np.random.default_rng(602)
        frame = rng.integers(0, 180, (720, 1280, 4), dtype=np.uint8)
        alpha = rng.integers(0, 256, frame.shape[:2], dtype=np.uint8)
        frame_before, alpha_before = frame.copy(), alpha.copy()
        frame.setflags(write=False)
        for intensity in (0.0, 0.25, 0.5, 1.0):
            actual, expected = FaceRelighter(), ReferenceRelighter()
            for effect in (actual, expected):
                effect.enabled = True
                effect.intensity = intensity
            for index in range(21):
                # Includes clipped left/right/top/bottom hulls and a tiny ROI.
                bounds = ((0.30, 0.25, 0.60, 0.70), (-0.10, 0.1, 0.25, 0.5),
                          (0.8, 0.7, 1.1, 1.1), (0.3, -0.1, 0.55, 0.3),
                          (0.49, 0.49, 0.491, 0.491))[index % 5]
                if index in (2, 4, 12):
                    for effect in (actual, expected):
                        effect._bg_warmth = (-0.15, 0.0, 0.17)[index % 3]
                points = landmarks(*bounds)
                with self.subTest(intensity=intensity, frame=index, bounds=bounds):
                    output = actual.process_frame(frame, alpha, points)
                    reference = expected.process_frame(frame, alpha, points)
                    self.assertTrue(np.array_equal(output, reference), "processed pixel bytes changed")
                    self.assert_same(actual, expected)
                    self.assertTrue(np.array_equal(output[:, :, 3], frame[:, :, 3]))
                    if index == 0 and intensity > 0:
                        self.assertFalse(np.array_equal(output, frame), "fixture must apply a real correction")
        self.assertTrue(np.array_equal(frame, frame_before), "input frame mutated")
        self.assertTrue(np.array_equal(alpha, alpha_before), "alpha mutated")

    def test_rgb_and_strided_frame_output_matches_without_inline_detection(self):
        rng = np.random.default_rng(190)
        original = rng.integers(0, 256, (148, 214, 4), dtype=np.uint8)
        alpha = rng.integers(0, 256, original.shape[:2], dtype=np.uint8)
        for frame, mask in ((original[:, :, :3], alpha), (original[::-1, ::-1], alpha[::-1, ::-1]),
                            (original[::2, ::2], alpha[::2, ::2]),
                            (original.astype(np.float32), alpha), (original.astype(np.uint16), alpha)):
            actual, expected = FaceRelighter(), ReferenceRelighter()
            actual.enabled = expected.enabled = True
            with self.subTest(shape=frame.shape, strides=frame.strides, dtype=frame.dtype), \
                 patch("nvbroadcast.video.relighting.get_shared_landmarker", side_effect=AssertionError("inline detector")), \
                 patch("tests.fixtures.relighting_reference.get_shared_landmarker", side_effect=AssertionError("reference detector")):
                self.assertTrue(np.array_equal(actual.process_frame(frame, mask, landmarks()),
                                               expected.process_frame(frame, mask, landmarks())))
                self.assert_same(actual, expected)


if __name__ == "__main__":
    unittest.main()

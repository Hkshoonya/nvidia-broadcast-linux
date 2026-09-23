"""Model-free checks for Remove's foreground color donor handling."""

import unittest
import sys
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nvbroadcast.core.config import EdgeConfig
from nvbroadcast.video.effects import VideoEffects


try:
    import cupy as cp
except (ImportError, OSError):
    cp = None


class _FixedBackend:
    _lowres_refined = False

    def __init__(self, alpha):
        self.alpha = alpha

    def infer(self, *_args):
        return self.alpha.copy()


def _make_effects(alpha, fused=False, dilate=3, softness=5):
    effects = VideoEffects(compositing="cpu", edge_config=EdgeConfig())
    effects._bg_mode = "remove"
    effects._quality = "ultra"
    effects._use_fused_kernel = True
    effects._refresh_temporal_strength()
    effects._bg_removal_enabled = True
    effects._initialized = True
    effects._backend = _FixedBackend(alpha)
    effects._learned_refiners = {}
    effects.update_edge_params(dilate_size=dilate, blur_size=softness)
    if fused:
        effects._compositing = "cupy"
        effects._cupy = cp
    return effects


class RemoveDonorSupportTests(unittest.TestCase):
    def test_both_cpu_cleanup_passes_preserve_unsupported_dark_strand(self):
        h, w = 360, 640
        alpha = np.zeros((h, w), np.float32)
        alpha[80:180, 150] = 0.5
        frame = np.full((h, w, 4), 255, np.uint8)
        frame[80:180, 150, :3] = 64
        effects = _make_effects(alpha)

        with mock.patch.object(effects, "_clean_color_reference",
                               wraps=effects._clean_color_reference) as reference:
            cleaned = effects._prepare_greenscreen_foreground(frame, alpha)

        self.assertEqual(reference.call_count, 2)
        self.assertTrue(np.array_equal(cleaned[80:180, 150], frame[80:180, 150]))

    def test_opaque_donor_still_repairs_supported_fringe(self):
        alpha = np.zeros((48, 48), np.float32)
        alpha[8:40, 12:36] = 1.0
        alpha[8:40, 11] = 0.3
        frame = np.full((48, 48, 4), 240, np.uint8)
        frame[:, :, 3] = 255
        frame[8:40, 12:36, :3] = 36
        effects = _make_effects(alpha)

        cleaned = effects._prepare_greenscreen_foreground(frame, alpha)

        self.assertLess(int(cleaned[24, 11, 0]), int(frame[24, 11, 0]) - 40)
        self.assertTrue(np.array_equal(cleaned[24, 20], frame[24, 20]))

    def test_large_unsupported_fringe_retains_source_across_sampling_chunks(self):
        alpha = np.zeros((300, 300), np.float32)
        alpha[20:280, 20:280] = 0.5
        frame = np.full((300, 300, 4), 255, np.uint8)
        frame[20:280, 20:280, :3] = 64
        effects = _make_effects(alpha)

        cleaned = effects._prepare_greenscreen_foreground(frame, alpha)

        self.assertTrue(np.array_equal(cleaned, frame))

    def _gpu(self):
        if cp is None:
            self.skipTest("CuPy unavailable")
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                self.skipTest("CUDA device unavailable")
        except Exception as exc:
            self.skipTest(f"CUDA device unavailable: {exc}")

    def test_fused_remove_keeps_unsupported_source_color_without_fallback(self):
        self._gpu()
        h, w = 360, 640
        alpha = np.zeros((h, w), np.float32)
        alpha[80:180, 150] = 0.5
        frame = np.full((h, w, 4), 255, np.uint8)
        frame[80:180, 150, :3] = 64
        effects = _make_effects(alpha, fused=True)

        output = effects._composite_fused_gpu(cp.asarray(frame), alpha, w, h)

        self.assertIsInstance(output, cp.ndarray)
        self.assertTrue(np.array_equal(cp.asnumpy(output)[120, 150, :3],
                                       np.array([32, 159, 32], dtype=np.uint8)))

    def test_fused_textured_720p_matches_cpu_with_same_final_matte(self):
        self._gpu()
        h, w = 720, 1280
        alpha = np.zeros((h, w), np.float32)
        alpha[h // 5:h - 20, w // 3:w * 2 // 3] = 1
        frame = np.full((h, w, 4), 64, np.uint8)
        frame[:, :, 3] = 255
        frame[alpha == 0, :3] = 255
        for scene in ("textured", "white_clothing"):
            for dilate, softness in ((3, 5), (15, 25)):
                with self.subTest(scene=scene, dilate=dilate, softness=softness):
                    subject = frame.copy()
                    subject[h // 2:h - 40, w // 3 + 30:w * 2 // 3 - 30, :3] = 240
                    if scene == "textured":
                        subject[h // 4:h // 2, w * 2 // 3 - 2:w * 2 // 3, :3] = 24
                    else:
                        subject[h // 2:h - 20, w // 3:w // 3 + 2, :3] = 24
                    cpu = _make_effects(alpha, dilate=dilate, softness=softness)
                    gpu = _make_effects(alpha, fused=True, dilate=dilate,
                                        softness=softness)

                    expected = cpu.process_frame_array(subject.copy(), w, h)
                    with mock.patch.object(gpu, "_apply_green_screen",
                                           side_effect=AssertionError("CPU fallback")):
                        actual = gpu.process_frame_array(subject.copy(), w, h)

                    self.assertTrue(np.array_equal(cpu.latest_final_matte_u8(w, h),
                                                   gpu.latest_final_matte_u8(w, h)))
                    error = np.max(np.abs(expected[:, :, :3].astype(np.int16) -
                                          actual[:, :, :3].astype(np.int16)))
                    self.assertLessEqual(int(error), 6)


if __name__ == "__main__":
    unittest.main()

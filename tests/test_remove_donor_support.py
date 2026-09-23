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

    def test_dark_strand_touching_white_clothing_is_not_repainted(self):
        h, w = 360, 640
        alpha = np.zeros((h, w), np.float32)
        alpha[80:180, 150] = 0.5
        alpha[80:180, 151:200] = 1.0
        frame = np.full((h, w, 4), 255, np.uint8)
        frame[80:180, 150, :3] = 64
        frame[80:180, 151:200, :3] = 240
        effects = _make_effects(alpha)

        cleaned = effects._prepare_greenscreen_foreground(frame, alpha)

        np.testing.assert_array_equal(cleaned[80:180, 150, :3],
                                      frame[80:180, 150, :3])
        np.testing.assert_array_equal(cleaned[80:180, 151:200],
                                      frame[80:180, 151:200])

    def test_bright_camera_spill_on_dark_edge_is_still_repaired(self):
        h, w = 360, 640
        alpha = np.zeros((h, w), np.float32)
        alpha[80:180, 150] = 0.5
        alpha[80:180, 151:200] = 1.0
        frame = np.full((h, w, 4), 255, np.uint8)
        frame[80:180, 150, :3] = 180
        frame[80:180, 151:200, :3] = 36
        effects = _make_effects(alpha)

        cleaned = effects._prepare_greenscreen_foreground(frame, alpha)

        self.assertLess(int(cleaned[120, 150, 0]), 100)
        np.testing.assert_array_equal(cleaned[80:180, 151:200],
                                      frame[80:180, 151:200])

    def test_large_unsupported_fringe_retains_source_across_sampling_chunks(self):
        alpha = np.zeros((300, 300), np.float32)
        alpha[20:280, 20:280] = 0.5
        frame = np.full((300, 300, 4), 255, np.uint8)
        frame[20:280, 20:280, :3] = 64
        effects = _make_effects(alpha)

        cleaned = effects._prepare_greenscreen_foreground(frame, alpha)

        self.assertTrue(np.array_equal(cleaned, frame))

    def test_moving_remove_finger_gaps_stay_open_without_reopening_palm_hole(self):
        for h, w in ((360, 640), (720, 1280)):
            for gap in (1, 2, 4, 8):
                effects = None
                for shift in (0, 1, 3, 1):
                    with self.subTest(size=(w, h), gap=gap, shift=shift):
                        y, x = h // 3, w // 3 + shift
                        alpha = np.zeros((h, w), np.float32)
                        alpha[y:y + 20, x - 12:x] = 1.0
                        alpha[y:y + 20, x + gap:x + gap + 12] = 1.0
                        alpha[y + 20:y + 32, x - 12:x + gap + 12] = 1.0
                        alpha[y + 24:y + 27, x - 6:x - 3] = 0.0
                        frame = np.full((h, w, 4), 255, np.uint8)
                        frame[alpha > 0, :3] = 64
                        frame[y + 24:y + 27, x - 6:x - 3, :3] = 240
                        if effects is None:
                            effects = _make_effects(alpha)
                        else:
                            effects._backend.alpha = alpha

                        output = effects.process_frame_array(frame, w, h)
                        matte = effects.latest_final_matte_u8(w, h)

                        self.assertTrue(np.all(matte[y + 5:y + 15, x:x + gap] == 0))
                        self.assertTrue(np.all(output[y + 5:y + 15, x:x + gap, :3]
                                               == (0, 255, 0)))
                        self.assertGreaterEqual(int(matte[y + 25, x - 5]), 250)
                        self.assertTrue(np.all(output[y + 25, x - 5, :3] == 240))

    def test_ambiguous_weak_alpha_channel_stays_partial(self):
        # The model cannot distinguish a background-colored strand from a
        # weak-alpha camera-background channel. Keep partial model support,
        # but do not turn every such gap into opaque white foreground.
        h, w = 720, 1280
        y, x = 220, 550
        alpha = np.zeros((h, w), np.float32)
        alpha[y:y + 130, x - 20:x] = 1.0
        alpha[y:y + 130, x + 8:x + 28] = 1.0
        alpha[y + 130:y + 180, x - 20:x + 28] = 1.0
        for weak_alpha in (0.03, 0.10):
            alpha[y:y + 130, x:x + 8] = weak_alpha
            for background, foreground in ((20, 240), (255, 50), (255, 245)):
                with self.subTest(alpha=weak_alpha, background=background,
                                  foreground=foreground):
                    frame = np.full((h, w, 4), background, np.uint8)
                    frame[:, :, 3] = 255
                    frame[alpha >= 0.95, :3] = foreground
                    effects = _make_effects(alpha)

                    effects.process_frame_array(frame, w, h)
                    matte = effects.latest_final_matte_u8(w, h)

                    self.assertGreater(int(matte[y + 50, x + 4]), 120)
                    self.assertLess(int(matte[y + 50, x + 4]), 230)

    def test_dark_hair_outside_finger_channels_remains_visible(self):
        h, w = 720, 1280
        y, x = 200, 500
        alpha = np.zeros((h, w), np.float32)
        alpha[y:y + 130, x + 1:x + 60] = 1.0
        alpha[y:y + 130, x] = 0.5
        frame = np.full((h, w, 4), 255, np.uint8)
        frame[y:y + 130, x + 1:x + 60, :3] = 240
        frame[y:y + 130, x, :3] = 64
        effects = _make_effects(alpha)

        output = effects.process_frame_array(frame, w, h)
        matte = effects.latest_final_matte_u8(w, h)

        self.assertGreaterEqual(int(matte[y + 50, x]), 200)
        self.assertLess(int(output[y + 50, x, 0]), 150)

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

    def test_fused_remove_background_gate_matches_cpu_on_both_color_directions(self):
        self._gpu()
        h, w = 360, 640
        alpha = np.zeros((h, w), np.float32)
        alpha[80:180, 150] = 0.5
        alpha[80:180, 151:200] = 1.0
        for scene, background, fringe, donor in (
            ("dark_strand_white_background", 255, 64, 240),
            ("bright_spill_white_background", 255, 180, 36),
            ("dark_spill_dark_background", 24, 64, 240),
        ):
            with self.subTest(scene=scene):
                frame = np.full((h, w, 4), background, np.uint8)
                frame[:, :, 3] = 255
                frame[80:180, 150, :3] = fringe
                frame[80:180, 151:200, :3] = donor
                cpu = _make_effects(alpha)
                gpu = _make_effects(alpha, fused=True)
                cleaned = cpu._despill_fringe(frame, alpha)
                green = np.zeros_like(frame)
                green[:, :, 1] = 255
                green[:, :, 3] = 255
                expected = cpu._blend_cpu(cleaned, green, alpha)
                actual = cp.asnumpy(gpu._composite_fused_gpu(
                    cp.asarray(frame), alpha, w, h))

                error = np.abs(expected[:, :, :3].astype(np.int16)
                               - actual[:, :, :3].astype(np.int16))
                self.assertLessEqual(int(error.max()), 2)
                if scene.startswith("dark_strand"):
                    self.assertEqual(int(cleaned[120, 150, 0]), 64)
                else:
                    self.assertNotEqual(int(cleaned[120, 150, 0]), fringe)

    def test_fused_moving_remove_gap_uses_restored_matte(self):
        self._gpu()
        h, w, gap = 720, 1280, 2
        effects = None
        for shift in (0, 1, 3, 1):
            with self.subTest(shift=shift):
                y, x = h // 3, w // 3 + shift
                alpha = np.zeros((h, w), np.float32)
                alpha[y:y + 20, x - 12:x] = 1.0
                alpha[y:y + 20, x + gap:x + gap + 12] = 1.0
                alpha[y + 20:y + 32, x - 12:x + gap + 12] = 1.0
                frame = np.full((h, w, 4), 255, np.uint8)
                frame[alpha > 0, :3] = 64
                if effects is None:
                    effects = _make_effects(alpha, fused=True)
                else:
                    effects._backend.alpha = alpha

                with mock.patch.object(effects, "_apply_green_screen",
                                       side_effect=AssertionError("CPU fallback")):
                    output = effects.process_frame_array(frame, w, h)
                matte = effects.latest_final_matte_u8(w, h)
                self.assertTrue(np.all(matte[y + 5:y + 15, x:x + gap] == 0))
                self.assertTrue(np.all(output[y + 5:y + 15, x:x + gap, :3]
                                       == (0, 255, 0)))

    def test_fused_ambiguous_weak_alpha_matches_partial_cpu_matte(self):
        self._gpu()
        h, w = 720, 1280
        y, x = 220, 550
        alpha = np.zeros((h, w), np.float32)
        alpha[y:y + 130, x - 20:x] = 1.0
        alpha[y:y + 130, x + 8:x + 28] = 1.0
        alpha[y + 130:y + 180, x - 20:x + 28] = 1.0
        alpha[y:y + 130, x:x + 8] = 0.10
        frame = np.full((h, w, 4), 20, np.uint8)
        frame[:, :, 3] = 255
        frame[alpha >= 0.95, :3] = 240
        cpu = _make_effects(alpha)
        gpu = _make_effects(alpha, fused=True)

        expected = cpu.process_frame_array(frame.copy(), w, h)
        with mock.patch.object(gpu, "_apply_green_screen",
                               side_effect=AssertionError("CPU fallback")):
            actual = gpu.process_frame_array(frame.copy(), w, h)

        np.testing.assert_array_equal(
            cpu.latest_final_matte_u8(w, h), gpu.latest_final_matte_u8(w, h))
        self.assertLess(int(cpu.latest_final_matte_u8(w, h)[y + 50, x + 4]), 230)
        error = np.abs(expected[:, :, :3].astype(np.int16)
                       - actual[:, :, :3].astype(np.int16))
        self.assertLessEqual(int(error[y + 5:y + 120, x:x + 8].max()), 2)

    def test_fused_textured_720p_matches_cpu_with_same_final_matte(self):
        self._gpu()
        h, w = 720, 1280
        alpha = np.zeros((h, w), np.float32)
        alpha[h // 5:h - 20, w // 3:w * 2 // 3] = 1
        frame = np.full((h, w, 4), 64, np.uint8)
        frame[:, :, 3] = 255
        frame[alpha == 0, :3] = 255
        for scene in ("textured", "white_clothing"):
            for dilate, softness in ((3, 5), (3, 11), (15, 25)):
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

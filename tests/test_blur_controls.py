"""Regression coverage for adjustable background blur controls."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig
from nvbroadcast.ui.window import NVBroadcastWindow
from nvbroadcast.video.effects import VideoEffects


class BlurControlTests(unittest.TestCase):
    def setUp(self):
        self.effects = VideoEffects(compositing="cpu")

    def test_config_defaults_preserve_existing_blur_behavior(self):
        video = AppConfig().video

        self.assertEqual(video.blur_intensity, 0.7)
        self.assertEqual(video.blur_dim, 0.0)
        self.assertEqual(video.blur_desaturate, 0.0)

    def test_effect_setters_clamp_to_supported_range(self):
        self.effects.intensity = -2.0
        self.effects.blur_dim = -1.0
        self.effects.blur_desaturate = -0.5

        self.assertEqual(self.effects.intensity, 0.0)
        self.assertEqual(self.effects._blur_sigma, 1.5)
        self.assertEqual(self.effects.blur_dim, 0.0)
        self.assertEqual(self.effects.blur_desaturate, 0.0)

        self.effects.intensity = 2.0
        self.effects.blur_dim = 4.0
        self.effects.blur_desaturate = 3.0

        self.assertEqual(self.effects.intensity, 1.0)
        self.assertEqual(self.effects._blur_sigma, 60.0)
        self.assertEqual(self.effects.blur_dim, 1.0)
        self.assertEqual(self.effects.blur_desaturate, 1.0)

    def test_intensity_maps_monotonically_to_bounded_sigma(self):
        sigmas = []
        for intensity in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.effects.intensity = intensity
            sigmas.append(self.effects._blur_sigma)

        self.assertEqual(sigmas[0], 1.5)
        self.assertEqual(sigmas[-1], 60.0)
        self.assertTrue(
            all(left < right for left, right in zip(sigmas, sigmas[1:]))
        )

    @mock.patch("nvbroadcast.app.save_config")
    def test_app_persists_clamped_blur_values(self, save_config):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app._video_effects = self.effects

        NVBroadcastApp.set_blur_intensity(app, 2.0)
        NVBroadcastApp.set_blur_dim(app, -1.0)
        NVBroadcastApp.set_blur_desaturate(app, 3.0)

        self.assertEqual(app.config.video.blur_intensity, 1.0)
        self.assertEqual(app.config.video.blur_dim, 0.0)
        self.assertEqual(app.config.video.blur_desaturate, 1.0)
        self.assertEqual(save_config.call_count, 3)

    def test_cpu_blur_reduces_detail_and_applies_portrait_controls(self):
        rng = np.random.default_rng(29)
        frame = rng.integers(0, 256, (64, 80, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        original = frame.copy()

        self.effects.intensity = 1.0
        neutral = self.effects._blur_background_cpu(frame)

        self.effects.blur_dim = 0.5
        self.effects.blur_desaturate = 1.0
        portrait = self.effects._blur_background_cpu(frame)

        self.assertTrue(np.array_equal(frame, original))
        self.assertLess(
            float(neutral[:, :, :3].std()),
            float(frame[:, :, :3].std()),
        )
        self.assertLess(
            float(portrait[:, :, :3].mean()),
            float(neutral[:, :, :3].mean()),
        )
        channel_spread = np.ptp(
            portrait[:, :, :3].astype(np.int16),
            axis=2,
        )
        self.assertLessEqual(int(channel_spread.max()), 1)
        self.assertTrue(np.all(portrait[:, :, 3] == 255))

    def test_large_blur_tones_the_quarter_resolution_working_image(self):
        frame = np.full((64, 80, 4), 128, dtype=np.uint8)
        self.effects.intensity = 1.0
        self.effects.blur_dim = 0.5
        original = self.effects._tone_blurred_background_cpu

        with mock.patch.object(
            self.effects,
            "_tone_blurred_background_cpu",
            wraps=original,
        ) as tone:
            self.effects._blur_background_cpu(frame)

        self.assertEqual(tone.call_args.args[0].shape, (16, 20, 4))

    def test_blur_dilate_control_expands_the_subject_matte(self):
        self.effects._bg_mode = "blur"
        alpha = np.zeros((96, 96), dtype=np.float32)
        alpha[32:64, 32:64] = 1.0

        self.effects.update_edge_params(
            dilate_size=0,
            blur_size=1,
            sigmoid_strength=0,
        )
        tight = self.effects._refine_alpha_full(alpha)

        self.effects.update_edge_params(dilate_size=15)
        expanded = self.effects._refine_alpha_full(alpha)

        self.assertGreater(float(expanded.sum()), float(tight.sum()))

    def test_blur_softness_control_widens_the_transition(self):
        self.effects._bg_mode = "blur"
        alpha = np.zeros((96, 96), dtype=np.float32)
        alpha[32:64, 32:64] = 1.0

        self.effects.update_edge_params(
            dilate_size=0,
            blur_size=1,
            sigmoid_strength=0,
        )
        crisp = self.effects._refine_alpha_full(alpha)

        self.effects.update_edge_params(blur_size=25)
        soft = self.effects._refine_alpha_full(alpha)

        crisp_transition = np.count_nonzero((crisp > 0.05) & (crisp < 0.95))
        soft_transition = np.count_nonzero((soft > 0.05) & (soft < 0.95))
        self.assertGreater(soft_transition, crisp_transition)

    def test_blur_preserves_exterior_gap_between_raised_hands(self):
        self.effects._bg_mode = "blur"
        self.effects.update_edge_params(
            dilate_size=2,
            blur_size=6,
            sigmoid_strength=14,
            sigmoid_midpoint=0.45,
        )
        alpha = np.zeros((96, 96), dtype=np.float32)
        alpha[62:88, 20:76] = 1.0
        alpha[16:70, 20:38] = 1.0
        alpha[16:70, 58:76] = 1.0

        refined = self.effects._refine_alpha_full(alpha)

        self.assertLess(
            float(refined[40, 48]),
            0.15,
            "blur refinement must not bridge the open gap between raised hands",
        )

    def test_cpu_and_cuda_blur_match_when_cuda_is_available(self):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("CUDA device not available")
        except Exception as exc:
            self.skipTest(f"CuPy/CUDA unavailable: {exc}")

        rng = np.random.default_rng(30)
        frame = rng.integers(0, 256, (64, 80, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        self.effects.intensity = 1.0
        self.effects.blur_dim = 0.55
        self.effects.blur_desaturate = 0.75

        cpu = self.effects._blur_background_cpu(frame)
        self.effects._cupy = cp
        gpu = cp.asnumpy(
            self.effects._gpu_blur_bgra(
                cp.asarray(frame),
                self.effects._blur_sigma,
            )
        )

        difference = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
        self.assertLessEqual(int(difference.max()), 1)

    def test_cpu_and_cuda_blur_edge_controls_match_when_cuda_is_available(self):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("CUDA device not available")
        except Exception as exc:
            self.skipTest(f"CuPy/CUDA unavailable: {exc}")

        self.effects._bg_mode = "blur"
        self.effects._cupy = cp
        alpha = np.zeros((96, 96), dtype=np.float32)
        alpha[32:64, 32:64] = 1.0

        for dilate_size, blur_size in ((0, 1), (2, 6), (15, 25)):
            with self.subTest(dilate=dilate_size, softness=blur_size):
                self.effects.update_edge_params(
                    dilate_size=dilate_size,
                    blur_size=blur_size,
                    sigmoid_strength=14,
                )
                cpu = self.effects._refine_alpha_full(alpha)
                gpu = cp.asnumpy(
                    self.effects._refine_alpha_full_gpu(cp.asarray(alpha))
                )

                difference = np.abs(cpu - gpu)
                self.assertLessEqual(float(difference.max()), 0.02)

        hand_gap = np.zeros((96, 96), dtype=np.float32)
        hand_gap[62:88, 20:76] = 1.0
        hand_gap[16:70, 20:38] = 1.0
        hand_gap[16:70, 58:76] = 1.0
        self.effects.update_edge_params(
            dilate_size=2,
            blur_size=6,
            sigmoid_strength=14,
            sigmoid_midpoint=0.45,
        )
        cpu = self.effects._refine_alpha_full(hand_gap)
        gpu = cp.asnumpy(
            self.effects._refine_alpha_full_gpu(cp.asarray(hand_gap))
        )

        self.assertLess(float(gpu[40, 48]), 0.15)
        self.assertLessEqual(float(np.abs(cpu - gpu).max()), 0.02)

    def test_cuda_small_hole_fill_matches_cpu_when_cuda_is_available(self):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("CUDA device not available")
        except Exception as exc:
            self.skipTest(f"CuPy/CUDA unavailable: {exc}")

        self.effects._cupy = cp
        matte = np.zeros((96, 96), dtype=np.uint8)
        matte[16:80, 16:80] = 255
        matte[47:49, 47:49] = 0
        params = {
            "binary_threshold": 30,
            "fill_cutoff": 100,
            "fill_value": 220,
            "max_area_ratio": 0.0007,
            "max_span_ratio": 0.07,
        }

        cpu = self.effects._fill_small_internal_holes(matte, **params)
        gpu = cp.asnumpy(
            self.effects._fill_small_internal_holes_gpu(
                cp.asarray(matte),
                **params,
            )
        )

        np.testing.assert_array_equal(gpu, cpu)
        self.assertEqual(int(gpu[47, 47]), 220)


class BackgroundControlSensitivityTests(unittest.TestCase):
    @staticmethod
    def _window():
        control_names = (
            "_bg_mode",
            "_blur_slider",
            "_blur_dim_slider",
            "_blur_desat_slider",
            "_bg_image_picker",
            "_quality_selector",
            "_model_selector",
            "_edge_dilate",
            "_edge_blur",
            "_edge_strength",
            "_edge_midpoint",
            "_skip_interval",
            "_ema_weight",
        )
        window = SimpleNamespace(
            _bg_toggle=SimpleNamespace(active=False),
        )
        for name in control_names:
            setattr(
                window,
                name,
                SimpleNamespace(set_sensitive=mock.Mock()),
            )
        window._bg_mode.mode = "blur"
        return window

    def test_sensitivity_follows_enabled_state_and_background_mode(self):
        cases = (
            (False, "blur", False, False),
            (True, "blur", True, False),
            (True, "replace", False, True),
            (True, "remove", False, False),
        )

        for enabled, mode, blur_enabled, image_enabled in cases:
            with self.subTest(enabled=enabled, mode=mode):
                window = self._window()
                NVBroadcastWindow._sync_background_controls(
                    window,
                    enabled=enabled,
                    mode=mode,
                )

                window._bg_mode.set_sensitive.assert_called_once_with(enabled)
                window._blur_slider.set_sensitive.assert_called_once_with(
                    blur_enabled
                )
                window._blur_dim_slider.set_sensitive.assert_called_once_with(
                    blur_enabled
                )
                window._blur_desat_slider.set_sensitive.assert_called_once_with(
                    blur_enabled
                )
                window._bg_image_picker.set_sensitive.assert_called_once_with(
                    image_enabled
                )
                for control in (
                    window._quality_selector,
                    window._model_selector,
                    window._edge_dilate,
                    window._edge_blur,
                    window._edge_strength,
                    window._edge_midpoint,
                    window._skip_interval,
                    window._ema_weight,
                ):
                    control.set_sensitive.assert_called_once_with(enabled)

    def test_mode_change_syncs_controls_during_settings_restore(self):
        window = self._window()
        window._app = SimpleNamespace(
            _restoring=True,
            set_bg_mode=mock.Mock(),
        )
        window._sync_background_controls = mock.Mock()

        NVBroadcastWindow._on_bg_mode_changed(window, None, "replace")

        window._sync_background_controls.assert_called_once_with(mode="replace")
        window._app.set_bg_mode.assert_not_called()


if __name__ == "__main__":
    unittest.main()

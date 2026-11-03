"""Host temporal-kernel routing, ownership, and CPU recovery without CUDA."""

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from nvbroadcast.video.effects import VideoEffects


class _FakeCupy:
    class ndarray:
        pass

    def __init__(self, shape, fail=None):
        self.cuda = SimpleNamespace(Device=self._device)
        self.active_device = 9
        self.events = []
        self.launches = []
        self.output = np.full(shape, 0.375, dtype=np.float32)
        self.fail = fail
        self.on_launch = None

    @contextmanager
    def _device(self, index):
        previous = self.active_device
        self.active_device = index
        self.events.append(("enter", index))
        try:
            yield
        finally:
            self.events.append(("exit", index))
            self.active_device = previous

    def ElementwiseKernel(self, *_args, **_kwargs):
        self.events.append(("compile", self.active_device))
        if self.fail == "compile":
            raise RuntimeError("synthetic kernel compilation failure")
        return self._kernel

    def _kernel(self, *args):
        self.events.append(("launch", self.active_device))
        self.launches.append(args)
        if self.fail == "launch":
            raise RuntimeError("synthetic kernel launch failure")
        if self.on_launch is not None:
            self.on_launch()
        return self.output

    def asarray(self, value):
        self.events.append(("upload", self.active_device))
        if self.fail == "upload":
            raise RuntimeError("synthetic upload failure")
        return np.array(value, copy=True)

    def asnumpy(self, value):
        self.events.append(("download", self.active_device))
        if self.fail == "download":
            raise RuntimeError("synthetic download failure")
        return np.array(value, copy=True)


class TemporalGpuDispatchTests(unittest.TestCase):
    shape = (360, 640)
    replace_params = (0.16, 2.15, 3.15, 0.66, 0.03, 0.26, 0.2)

    def setUp(self):
        self.alpha = np.linspace(0, 1, np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        self.prev = np.roll(self.alpha, 7, axis=1)

    def _effects(self, fail=None):
        effects = VideoEffects(compositing="cpu", gpu_index=3)
        effects._bg_mode = "replace"
        effects._compositing = "cupy"
        effects._cupy = _FakeCupy(self.shape, fail=fail)
        return effects

    def _try_gpu(self, effects, alpha=None, prev=None, motion=0.02, weights=None):
        return effects._try_temporal_blend_gpu(
            self.alpha if alpha is None else alpha,
            self.prev if prev is None else prev,
            motion,
            (0.04, 0.5, 0.65) if weights is None else weights,
            0.03, 0.97, 0.03, 0.26, 0.66, 0.05, 0.2,
            local_release=True, cap_first=True,
        )

    def test_selected_device_and_output_ownership_with_strided_inputs(self):
        effects = self._effects()
        cp = effects._cupy
        alpha, prev = self.alpha[:, ::-1], self.prev[:, ::-1]
        alpha_before, prev_before = alpha.copy(), prev.copy()

        output = self._try_gpu(effects, alpha, prev)
        second = self._try_gpu(effects, alpha, prev)

        np.testing.assert_array_equal(output, cp.output)
        np.testing.assert_array_equal(alpha, alpha_before)
        np.testing.assert_array_equal(prev, prev_before)
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(output.shape, self.shape)
        self.assertFalse(np.shares_memory(output, alpha))
        self.assertFalse(np.shares_memory(output, prev))
        self.assertFalse(np.shares_memory(output, cp.output))
        self.assertFalse(np.shares_memory(output, second))
        self.assertTrue(all(device == 3 for _, device in cp.events))
        self.assertEqual(cp.active_device, 9)
        self.assertEqual(cp.events.count(("compile", 3)), 1)
        self.assertEqual(cp.events.count(("download", 3)), 2)

    def test_both_host_temporal_stages_dispatch_without_changing_history_ownership(self):
        effects = self._effects()
        cp = effects._cupy
        effects._prev_alpha = self.prev.copy()
        old_history = effects._prev_alpha
        gpu_history = object()
        effects._prev_alpha_gpu = gpu_history

        result = effects._temporal_smooth(self.alpha, effects._matte_version)

        np.testing.assert_array_equal(result, cp.output)
        np.testing.assert_array_equal(effects._prev_alpha, cp.output)
        np.testing.assert_array_equal(old_history, self.prev)
        self.assertFalse(np.shares_memory(result, effects._prev_alpha))
        self.assertIs(effects._prev_alpha_gpu, gpu_history)
        self.assertEqual(cp.launches[-1][-2:], (True, True))
        self.assertTrue(all(isinstance(value, np.float32) for value in cp.launches[-1][2:-2]))
        self.assertIsNone(effects._cached_source_alpha)

        effects._stable_alpha = self.prev
        stable = effects._stabilized_replacement_alpha(self.alpha, self.prev, True)
        np.testing.assert_array_equal(stable, cp.output)
        self.assertIs(effects._stable_alpha, self.prev)
        self.assertEqual(cp.launches[-1][-2:], (False, False))

    def test_ineligible_inputs_never_touch_cupy(self):
        cases = (
            "cpu", "gstreamer_gl", "blur", "remove", "missing_cupy",
            "small", "alpha_float64", "prev_float64", "shape", "ndim",
            "alpha_nan", "prev_inf", "motion_nan", "weight_inf",
        )
        for case in cases:
            with self.subTest(case=case):
                effects = self._effects()
                cp = effects._cupy
                alpha, prev = self.alpha, self.prev
                motion, weights = 0.02, None
                if case in ("cpu", "gstreamer_gl"):
                    effects._compositing = case
                elif case in ("blur", "remove"):
                    effects._bg_mode = case
                elif case == "missing_cupy":
                    effects._cupy = None
                elif case == "small":
                    alpha, prev = alpha[:-1], prev[:-1]
                elif case == "alpha_float64":
                    alpha = alpha.astype(np.float64)
                elif case == "prev_float64":
                    prev = prev.astype(np.float64)
                elif case == "shape":
                    prev = prev[:-1]
                elif case == "ndim":
                    alpha, prev = alpha.ravel(), prev.ravel()
                elif case == "alpha_nan":
                    alpha = alpha.copy()
                    alpha[0, 0] = np.nan
                elif case == "prev_inf":
                    prev = prev.copy()
                    prev[0, 0] = np.inf
                elif case == "motion_nan":
                    motion = np.nan
                elif case == "weight_inf":
                    weights = (0.04, np.inf, 0.65)

                self.assertIsNone(self._try_gpu(effects, alpha, prev, motion, weights))
                self.assertEqual(cp.events, [])

    def test_exceptions_fall_back_to_cpu_and_preserve_next_frame_history(self):
        for stage in ("compile", "upload", "launch", "download"):
            with self.subTest(stage=stage):
                effects = self._effects(fail=stage)
                cp = effects._cupy
                cpu = VideoEffects(compositing="cpu")
                cpu._bg_mode = "replace"
                effects._prev_alpha = self.prev.copy()
                cpu._prev_alpha = self.prev.copy()
                with mock.patch("builtins.print"):
                    first = effects._temporal_smooth(self.alpha, effects._matte_version)
                expected = cpu._temporal_smooth(self.alpha, cpu._matte_version)
                np.testing.assert_array_equal(first, expected)
                attempted = cp.events.copy()
                next_alpha = np.roll(self.alpha, 3, axis=1)
                second = effects._temporal_smooth(next_alpha, effects._matte_version)
                expected = cpu._temporal_smooth(next_alpha, cpu._matte_version)
                np.testing.assert_array_equal(second, expected)
                np.testing.assert_array_equal(effects._prev_alpha, cpu._prev_alpha)
                self.assertEqual(cp.events, attempted)
                self.assertEqual(cp.active_device, 9)

    def test_stable_stage_failure_uses_unchanged_cpu_result(self):
        effects = self._effects(fail="launch")
        cpu = VideoEffects(compositing="cpu")
        cpu._bg_mode = "replace"
        with mock.patch("builtins.print"):
            actual = effects._stabilized_replacement_alpha(self.alpha, self.prev, False)
        expected = cpu._stabilized_replacement_alpha(self.alpha, self.prev, False)
        np.testing.assert_array_equal(actual, expected)

    def test_device_failure_survives_matte_reset_and_retries_after_reselection(self):
        effects = self._effects(fail="launch")
        cp = effects._cupy
        with mock.patch("builtins.print"):
            self.assertIsNone(self._try_gpu(effects))
        attempts = cp.events.copy()
        effects.reset_cached_mattes()
        self.assertIsNone(self._try_gpu(effects))
        self.assertEqual(cp.events, attempts)

        cp.fail = None
        effects._gpu_index = 4
        self.assertIsNotNone(self._try_gpu(effects))
        self.assertIn(("launch", 4), cp.events)
        effects._gpu_index = 3
        attempts = cp.events.copy()
        self.assertIsNone(self._try_gpu(effects))
        self.assertEqual(cp.events, attempts)

        effects.set_compositing("cupy")
        self.assertIsNotNone(self._try_gpu(effects))
        self.assertEqual(cp.events.count(("compile", 3)), 2)

    def test_reset_during_gpu_work_cannot_restore_stale_cpu_history(self):
        for stage in ("temporal", "stable"):
            with self.subTest(stage=stage):
                effects = self._effects()
                effects._prev_alpha = self.prev
                effects._stable_alpha = self.prev
                effects._cached_source_alpha = self.alpha.copy()
                version = effects._matte_version
                effects._cupy.on_launch = effects.reset_cached_mattes
                if stage == "temporal":
                    actual = effects._temporal_smooth(self.alpha, version)
                else:
                    with mock.patch.object(
                        effects, "_replacement_matte_from_stable",
                        side_effect=lambda _alpha, stable, *_args, **_kwargs: stable.copy(),
                    ):
                        actual = effects._replacement_matte(self.alpha, version)
                self.assertIs(actual, self.alpha)
                self.assertIsNone(effects._prev_alpha)
                self.assertIsNone(effects._stable_alpha)
                self.assertIsNone(effects._cached_source_alpha)
                self.assertIsNone(effects._pending_source_alpha)


if __name__ == "__main__":
    unittest.main()

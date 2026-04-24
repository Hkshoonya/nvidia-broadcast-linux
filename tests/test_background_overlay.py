"""Focused tests for background replacement compositing.

These tests avoid model initialization and only exercise the replacement matte
and compositing logic with synthetic frames.
"""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest


try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    np = None

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    cv2 = None


def _install_fake_onnxruntime() -> None:
    if "onnxruntime" in sys.modules:
        return

    class _FakeSessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.log_severity_level = None

    class _FakeGraphOptimizationLevel:
        ORT_ENABLE_ALL = 0

    class _FakeInferenceSession:
        def __init__(self, *args, **kwargs):
            pass

    fake = types.SimpleNamespace(
        InferenceSession=_FakeInferenceSession,
        SessionOptions=_FakeSessionOptions,
        GraphOptimizationLevel=_FakeGraphOptimizationLevel,
    )
    sys.modules["onnxruntime"] = fake


@unittest.skipIf(np is None or cv2 is None, "numpy/cv2 not installed")
class BackgroundOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_onnxruntime()
        sys.path.insert(0, "src")
        from nvbroadcast.video.effects import VideoEffects

        cls.VideoEffects = VideoEffects

    def _make_effects(self):
        effects = self.VideoEffects()
        effects._bg_mode = "replace"
        effects._bg_image = np.zeros((4, 4, 4), dtype=np.uint8)
        effects._bg_image[:, :, 0] = 240
        effects._bg_image[:, :, 3] = 255
        return effects

    def test_replace_composite_uses_background_and_foreground(self):
        effects = self._make_effects()

        fg = np.zeros((4, 4, 4), dtype=np.uint8)
        fg[:, :, 2] = 200
        fg[:, :, 3] = 255

        alpha = np.zeros((4, 4), dtype=np.float32)
        alpha[1:3, 1:3] = 1.0

        out = np.frombuffer(effects._composite(fg, alpha, 4, 4), dtype=np.uint8).reshape(4, 4, 4)

        self.assertGreater(out[0, 0, 0], 200, "background pixels should come from replacement image")
        self.assertGreater(out[1, 1, 2], 150, "foreground pixels should preserve subject color")

    def test_replace_matte_suppresses_small_edge_jitter(self):
        effects = self._make_effects()

        alpha1 = np.array(
            [
                [0.0, 0.04, 0.12, 0.0],
                [0.02, 0.70, 0.95, 0.05],
                [0.01, 0.60, 0.92, 0.04],
                [0.0, 0.03, 0.10, 0.0],
            ],
            dtype=np.float32,
        )
        alpha2 = np.array(
            [
                [0.0, 0.06, 0.15, 0.0],
                [0.03, 0.66, 0.93, 0.07],
                [0.02, 0.57, 0.90, 0.05],
                [0.0, 0.05, 0.12, 0.0],
            ],
            dtype=np.float32,
        )

        matte1 = effects._replacement_matte(alpha1)
        matte2 = effects._replacement_matte(alpha2)

        raw_delta = np.abs(alpha2 - alpha1).mean()
        matte_delta = np.abs(matte2 - matte1).mean()

        self.assertLess(matte_delta, raw_delta, "replacement matte should be more stable than raw alpha")
        self.assertEqual(float(matte2[0, 0]), 0.0, "near-zero fringe should be clipped away")

    def test_mode_switch_resets_cached_mattes(self):
        effects = self._make_effects()
        effects._cached_alpha = np.ones((4, 4), dtype=np.float32)
        effects._prev_alpha = np.ones((4, 4), dtype=np.float32)
        effects._stable_alpha = np.ones((4, 4), dtype=np.float32)
        start_version = effects._matte_version

        effects.mode = "blur"

        self.assertIsNone(effects._cached_alpha)
        self.assertIsNone(effects._prev_alpha)
        self.assertIsNone(effects._stable_alpha)
        self.assertEqual(effects._matte_version, start_version + 1)

    def test_engine_mode_switch_waits_for_inflight_inference(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._bg_removal_enabled = True
        effects._refine_alpha = lambda alpha: alpha
        effects._temporal_smooth = lambda alpha, _version=None: alpha

        infer_started = threading.Event()
        release_infer = threading.Event()
        switch_done = threading.Event()

        class _FakeBackend:
            def __init__(self):
                self._MAX_INFER_HEIGHT = 720
                self._trt_requested = True
                self.tensorrt_requested = []
                self.reset_calls = 0

            def infer(self, frame, width, height):
                infer_started.set()
                release_infer.wait(1.0)
                return np.ones((height, width), dtype=np.float32)

            def set_tensorrt_requested(self, enabled):
                self.tensorrt_requested.append(enabled)

            def reset_state(self):
                self.reset_calls += 1

        backend = _FakeBackend()
        effects._backend = backend
        frame = np.zeros((4, 4, 4), dtype=np.uint8)

        infer_result: list[np.ndarray] = []

        def _run_infer():
            infer_result.append(effects._run_inference(frame, 4, 4, effects._matte_version))

        infer_thread = threading.Thread(target=_run_infer)
        infer_thread.start()
        self.assertTrue(infer_started.wait(0.5))

        def _switch_mode():
            effects.set_engine_mode(True, False)
            switch_done.set()

        switch_thread = threading.Thread(target=_switch_mode)
        switch_thread.start()

        time.sleep(0.05)
        self.assertFalse(switch_done.is_set(), "mode switch should wait for in-flight inference")

        release_infer.set()
        infer_thread.join(1.0)
        switch_thread.join(1.0)

        self.assertTrue(switch_done.is_set())
        self.assertEqual(len(infer_result), 1)
        self.assertEqual(backend.tensorrt_requested, [True])
        self.assertEqual(backend._MAX_INFER_HEIGHT, 480)
        self.assertEqual(backend.reset_calls, 1)

    def test_engine_mode_schedules_reload_when_tensorrt_boundary_changes(self):
        effects = self._make_effects()
        effects._initialized = True

        class _FakeBackend:
            def __init__(self, trt_requested: bool):
                self._MAX_INFER_HEIGHT = 720
                self._trt_requested = trt_requested

        original_backend = _FakeBackend(False)
        effects._backend = original_backend

        scheduled = []

        effects._schedule_engine_reload = lambda use_tensorrt, infer_h: scheduled.append(
            (use_tensorrt, infer_h)
        )

        effects.set_engine_mode(True, False)

        self.assertEqual(scheduled, [(True, 480)])
        self.assertIs(effects._backend, original_backend)

    def test_run_inference_skips_while_engine_reload_is_in_progress(self):
        effects = self._make_effects()
        effects._initialized = True

        class _FakeBackend:
            def __init__(self):
                self.calls = 0

            def infer(self, frame, width, height):
                self.calls += 1
                return np.ones((height, width), dtype=np.float32)

        backend = _FakeBackend()
        effects._backend = backend
        effects._engine_reload_in_progress = True
        frame = np.zeros((4, 4, 4), dtype=np.uint8)

        alpha = effects._run_inference(frame, 4, 4, effects._matte_version)

        self.assertIsNone(alpha)
        self.assertEqual(backend.calls, 0)

    def test_engine_reload_warms_backend_before_swap(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._last_frame_size = (4, 4)

        warmed = threading.Event()

        class _FakeBackend:
            def __init__(self, name):
                self.name = name
                self._MAX_INFER_HEIGHT = 720
                self.cleanup_calls = 0
                self.infer_calls = []
                self.reset_calls = 0

            def infer(self, frame, width, height):
                self.infer_calls.append((width, height, frame.shape))
                warmed.set()
                return np.ones((height, width), dtype=np.float32)

            def cleanup(self):
                self.cleanup_calls += 1

            def reset_state(self):
                self.reset_calls += 1

        original = _FakeBackend("original")
        replacement = _FakeBackend("replacement")
        effects._backend = original
        effects._build_backend = lambda: (replacement, "replacement ready")

        effects._schedule_engine_reload(True, 360)

        self.assertTrue(warmed.wait(1.0), "replacement backend should warm before swap")
        limit = time.time() + 1.0
        while time.time() < limit:
            if effects._backend is replacement and not effects._engine_reload_in_progress:
                break
            time.sleep(0.01)

        self.assertIs(effects._backend, replacement)
        self.assertFalse(effects._engine_reload_in_progress)
        self.assertEqual(replacement.reset_calls, 1)
        self.assertEqual(replacement.infer_calls, [(4, 4, (4, 4, 4))])
        self.assertEqual(original.cleanup_calls, 1)

    def test_replace_matte_fills_small_internal_holes(self):
        effects = self._make_effects()

        alpha = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.95, 0.92, 0.94, 0.0],
                [0.0, 0.91, 0.10, 0.90, 0.0],
                [0.0, 0.93, 0.89, 0.94, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        matte = effects._replacement_matte(alpha)

        self.assertGreater(float(matte[2, 2]), 0.7, "small internal holes should be filled in replace mode")

    def test_preserve_large_internal_holes_prefers_narrow_slits_over_blob_holes(self):
        effects = self._make_effects()

        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:35, 5:35] = 255
        mask[12:22, 12:22] = 0
        mask[8:28, 26:28] = 0

        preserved = effects._preserve_large_internal_holes(
            mask,
            binary_threshold=127,
            min_area_ratio=0.0001,
            min_span_ratio=0.01,
            min_aspect_ratio=2.0,
            max_area_ratio=0.05,
        )

        self.assertFalse(
            bool(preserved[16, 16]),
            "broad internal blob holes should not be preserved in replace mode",
        )
        self.assertTrue(
            bool(preserved[16, 26]),
            "narrow finger-like slits should stay eligible for preservation",
        )

    def test_refine_alpha_closes_broad_internal_gap_in_replace_mode(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"

        alpha = np.zeros((11, 11), dtype=np.float32)
        alpha[1:10, 1:10] = 0.95
        alpha[4:7, 4:7] = 0.01

        refined = effects._refine_alpha(alpha)

        self.assertGreater(
            float(refined[5, 5]),
            0.80,
            "replace refinement should close broad artifact holes inside the silhouette",
        )

    def test_replace_matte_preserves_narrow_hairline_gap(self):
        effects = self._make_effects()

        alpha = np.zeros((15, 15), dtype=np.float32)
        alpha[2:14, 3:12] = 0.95
        alpha[2:8, 7:8] = 0.02

        matte = effects._replacement_matte(alpha)

        self.assertLess(
            float(matte[4, 7]),
            0.45,
            "thin but meaningful gaps near the head should stay visible",
        )

    def test_replace_matte_reopens_gap_quickly_across_frames(self):
        effects = self._make_effects()

        alpha_closed = np.zeros((15, 15), dtype=np.float32)
        alpha_closed[2:14, 3:12] = 0.95

        alpha_open = alpha_closed.copy()
        alpha_open[2:8, 7:8] = 0.02

        effects._replacement_matte(alpha_closed)
        matte = effects._replacement_matte(alpha_open)

        self.assertLess(
            float(matte[4, 7]),
            0.55,
            "replacement temporal smoothing should not keep narrow gaps shut after they open",
        )

    def test_despill_skips_near_solid_subject_pixels(self):
        effects = self._make_effects()

        fg = np.zeros((4, 4, 4), dtype=np.uint8)
        fg[:, :, 2] = 180
        fg[:, :, 3] = 255

        alpha = np.full((4, 4), 0.9, dtype=np.float32)
        alpha[1:3, 1:3] = 0.95

        out = effects._despill_fringe(fg, alpha)

        self.assertTrue(np.array_equal(out, fg), "despill should not recolor solid foreground regions")

    def test_despill_repaints_dark_replace_fringe(self):
        effects = self._make_effects()

        fg = np.zeros((8, 8, 4), dtype=np.uint8)
        fg[:, :, 2] = 175
        fg[:, :, 3] = 255
        fg[2:6, 2:6, 2] = 220
        fg[:, 1, :3] = 22
        fg[:, 6, :3] = 22

        alpha = np.zeros((8, 8), dtype=np.float32)
        alpha[:, 1] = 0.22
        alpha[:, 2:6] = 0.98
        alpha[:, 6] = 0.22

        cleaned = effects._despill_fringe(fg, alpha)

        self.assertGreater(
            int(cleaned[4, 1, 2]),
            int(fg[4, 1, 2]),
            "replace-mode despill should pull dark shoulder fringe toward subject color",
        )
        self.assertGreater(
            int(cleaned[4, 6, 2]),
            int(fg[4, 6, 2]),
            "replace-mode despill should clean both sides of the fringe",
        )

    def test_edge_aware_replace_matte_hardens_transition_on_real_edges(self):
        effects = self._make_effects()

        frame = np.zeros((12, 12, 4), dtype=np.uint8)
        frame[:, :6, :3] = 25
        frame[:, 6:, :3] = 230
        frame[:, :, 3] = 255

        matte = np.zeros((12, 12), dtype=np.float32)
        matte[:, 4] = 0.18
        matte[:, 5] = 0.34
        matte[:, 6] = 0.66
        matte[:, 7] = 0.82
        matte[:, 8:] = 0.98

        refined = effects._edge_aware_replace_matte(frame, matte.copy())

        self.assertLess(float(refined[6, 5]), float(matte[6, 5]), "foreground entry edge should tighten")
        self.assertGreater(float(refined[6, 6]), float(matte[6, 6]), "foreground exit edge should harden")
        self.assertEqual(float(refined[6, 0]), 0.0, "weak-edge background should stay clipped")

    def test_edge_aware_replace_matte_preserves_supported_fine_fringe(self):
        effects = self._make_effects()

        frame = np.zeros((12, 12, 4), dtype=np.uint8)
        frame[:, :6, :3] = 35
        frame[:, 6:, :3] = 210
        frame[:, :, 3] = 255

        matte = np.zeros((12, 12), dtype=np.float32)
        matte[:, 4] = 0.07
        matte[:, 5] = 0.11
        matte[:, 6] = 0.42
        matte[:, 7] = 0.76
        matte[:, 8:] = 0.97

        refined = effects._edge_aware_replace_matte(frame, matte.copy())

        self.assertGreater(
            float(refined[6, 5]),
            0.05,
            "fine supported fringe near a real image edge should not be clipped away",
        )

    def test_greenscreen_matte_is_tighter_than_replace_matte(self):
        effects = self._make_effects()
        effects._bg_mode = "remove"

        frame = np.zeros((12, 12, 4), dtype=np.uint8)
        frame[:, :6, :3] = 20
        frame[:, 6:, :3] = 230
        frame[:, :, 3] = 255

        alpha = np.zeros((12, 12), dtype=np.float32)
        alpha[:, 4] = 0.14
        alpha[:, 5] = 0.30
        alpha[:, 6] = 0.72
        alpha[:, 7] = 0.90
        alpha[:, 8:] = 0.98

        replace_matte = effects._edge_aware_replace_matte(frame, effects._replacement_matte(alpha))
        green_matte = effects._greenscreen_matte(frame, alpha)

        self.assertLess(float(green_matte[6, 5]), float(replace_matte[6, 5]), "greenscreen should clip weak fringe harder")
        self.assertGreaterEqual(float(green_matte[6, 7]), 0.9, "solid foreground should stay solid")
        self.assertEqual(float(green_matte[6, 0]), 0.0, "background must remain clipped")

    def test_greenscreen_foreground_cleanup_repaints_dark_fringe(self):
        effects = self._make_effects()

        fg = np.zeros((8, 8, 4), dtype=np.uint8)
        fg[:, :, 2] = 180
        fg[:, :, 3] = 255
        fg[2:6, 2:6, 2] = 220
        fg[:, 1, :3] = 20
        fg[:, 6, :3] = 20

        alpha = np.zeros((8, 8), dtype=np.float32)
        alpha[:, 1] = 0.24
        alpha[:, 2:6] = 0.98
        alpha[:, 6] = 0.24

        cleaned = effects._prepare_greenscreen_foreground(fg, alpha)

        self.assertGreater(int(cleaned[4, 1, 2]), int(fg[4, 1, 2]), "dark fringe should be pulled toward subject color")
        self.assertGreater(int(cleaned[4, 6, 2]), int(fg[4, 6, 2]), "cleanup should work on both sides")

    def test_doczeus_uses_tighter_temporal_strength_than_killer(self):
        effects = self._make_effects()
        effects.set_engine_mode(False, True)
        doczeus_strength = effects._temporal_strength
        effects.set_engine_mode(True, True)
        killer_strength = effects._temporal_strength
        self.assertLess(doczeus_strength, killer_strength, "fused-only quality mode should smooth less than killer")

    def test_remove_mode_lowers_temporal_strength(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"
        effects._refresh_temporal_strength()
        replace_strength = effects._temporal_strength
        effects.mode = "remove"
        remove_strength = effects._temporal_strength
        self.assertLess(remove_strength, replace_strength, "green-screen mode should smooth less than replace mode")

    def test_final_matte_can_use_learned_replace_refiner(self):
        effects = self._make_effects()

        class _StubRefiner:
            available = True

            def refine(self, frame, matte):
                boosted = matte.copy()
                boosted[boosted > 0.2] = np.clip(boosted[boosted > 0.2] + 0.1, 0.0, 1.0)
                return boosted

        effects._learned_refiners["replace"] = _StubRefiner()
        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        alpha = np.zeros((8, 8), dtype=np.float32)
        alpha[:, 3:6] = 0.6

        heuristic = effects._edge_aware_replace_matte(frame, effects._replacement_matte(alpha))
        final = effects._final_matte(frame, alpha)

        self.assertGreater(float(final[4, 4]), float(heuristic[4, 4]), "learned refiner should be able to modify final replace matte")


if __name__ == "__main__":
    unittest.main()

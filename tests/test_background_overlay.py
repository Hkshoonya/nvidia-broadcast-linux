"""Focused tests for background replacement compositing.

These tests avoid model initialization and only exercise the replacement matte
and compositing logic with synthetic frames.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
import types
import unittest
from unittest import mock


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
    def test_constructor_falls_back_from_saved_cupy_setting_without_cupy(self):
        from nvbroadcast.video.effects import VideoEffects

        with mock.patch.dict(sys.modules, {"cupy": None}):
            effects = VideoEffects(compositing="cupy")
        self.assertEqual(effects._compositing, "cpu")
        self.assertTrue(effects._cpu_inference)
        self.assertFalse(effects._engine_reload_in_progress)
        self.assertIsNone(effects._backend)
        self.assertTrue(all(refiner._cpu_only for refiner in effects._learned_refiners.values()))

    @classmethod
    def setUpClass(cls):
        _install_fake_onnxruntime()
        sys.path.insert(0, "src")
        from nvbroadcast.video.effects import VideoEffects

        cls.VideoEffects = VideoEffects
        cls.effects_module = sys.modules["nvbroadcast.video.effects"]

    def _make_effects(self):
        effects = self.VideoEffects()
        effects._bg_mode = "replace"
        effects._bg_image = np.zeros((4, 4, 4), dtype=np.uint8)
        effects._bg_image[:, :, 0] = 240
        effects._bg_image[:, :, 3] = 255
        return effects

    def test_cpu_compositing_fallback_does_not_retry_fused_cuda_each_frame(self):
        effects = self._make_effects()
        effects._compositing = "cpu"
        effects._use_fused_kernel = True
        effects._cupy = types.SimpleNamespace(ndarray=type("DeviceArray", (), {}))
        frame = np.zeros((4, 4, 4), dtype=np.uint8)
        alpha = np.ones((4, 4), dtype=np.float32)
        with mock.patch.object(effects, "_final_matte", return_value=alpha), \
             mock.patch.object(effects, "_apply_replace", return_value=frame), \
             mock.patch.object(effects, "_composite_fused", return_value=None) as fused:
            effects._composite_array(frame, alpha, 4, 4)
            effects._composite_array(frame, alpha, 4, 4)
        fused.assert_not_called()

    class _FakeCupy:
        float32 = np.float32 if np is not None else float
        int32 = np.int32 if np is not None else int
        uint8 = np.uint8 if np is not None else "uint8"
        newaxis = np.newaxis if np is not None else None

        class cuda:
            class Stream:
                class null:
                    @staticmethod
                    def synchronize():
                        return None

        @staticmethod
        def asarray(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def asnumpy(value):
            return np.asarray(value)

        @staticmethod
        def empty_like(value):
            return np.empty_like(value)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def ones(shape, dtype=None):
            return np.ones(shape, dtype=dtype)

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

    def test_cpu_compositing_stays_on_cpu_after_cupy_was_loaded(self):
        effects = self._make_effects()
        effects._cupy = self._FakeCupy
        effects._compositing = "cpu"
        effects._blend_cpu = mock.Mock(return_value="cpu")
        effects._blend_cupy = mock.Mock(return_value="gpu")

        result = effects._blend(
            np.zeros((1, 1, 4), dtype=np.uint8),
            np.zeros((1, 1, 4), dtype=np.uint8),
            np.ones((1, 1), dtype=np.float32),
        )

        self.assertEqual(result, "cpu")
        effects._blend_cpu.assert_called_once()
        effects._blend_cupy.assert_not_called()

    def test_cpu_processing_builds_rvm_with_cpu_inference(self):
        effects = self.VideoEffects(compositing="cpu")
        backend = mock.Mock()
        backend.load.return_value = "RVM loaded on CPU"

        with mock.patch.object(
            self.effects_module, "_RVMBackend", return_value=backend
        ):
            built, message = effects._build_backend()

        self.assertIs(built, backend)
        self.assertEqual(message, "RVM loaded on CPU")
        backend.load.assert_called_once_with(
            "quality", use_tensorrt=False, cpu_only=True
        )

    def test_cpu_processing_builds_single_frame_model_with_cpu_inference(self):
        effects = self.VideoEffects(compositing="cpu")
        effects._model_type = "isnet"
        backend = mock.Mock()
        backend.load.return_value = "IS-Net loaded on CPU"

        with mock.patch.object(
            self.effects_module, "_SingleFrameBackend", return_value=backend
        ) as backend_type:
            built, message = effects._build_backend()

        self.assertIs(built, backend)
        self.assertEqual(message, "IS-Net loaded on CPU")
        backend_type.assert_called_once_with(0, "isnet", cpu_only=True)

    def test_live_cpu_mode_change_reloads_gpu_inference_backend(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._cpu_inference = False
        effects._compositing = "cupy"
        backend = mock.Mock()
        backend._MAX_INFER_HEIGHT = 720
        backend._trt_requested = False
        backend._quality = effects._quality
        backend._cpu_only = False
        effects._backend = backend
        effects._schedule_engine_reload = mock.Mock()

        effects.set_compositing("cpu")
        effects.set_engine_mode(False, False)

        self.assertTrue(effects._cpu_inference)
        effects._schedule_engine_reload.assert_called_once_with(False, 720)

    def test_live_cpu_mode_change_reloads_single_frame_inference_backend(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._model_type = "isnet"
        effects._cpu_inference = False
        effects._compositing = "cupy"
        backend = mock.Mock(spec=[])
        backend._cpu_only = False
        effects._backend = backend
        effects._schedule_engine_reload = mock.Mock()

        effects.set_compositing("cpu")
        effects.set_engine_mode(False, False)

        self.assertTrue(effects._cpu_inference)
        effects._schedule_engine_reload.assert_called_once_with(False, 720)

    def test_edge_refiner_cleanup_waits_for_inflight_inference(self):
        refiner = self.effects_module.EdgeRefiner(0)
        infer_started = threading.Event()
        release_infer = threading.Event()
        backend = mock.Mock()
        backend.infer.side_effect = lambda *_args: (
            infer_started.set(),
            release_infer.wait(1.0),
            np.ones((2, 2), dtype=np.float32),
        )[-1]
        refiner._backend = backend
        refiner._initialized = True

        refine_thread = threading.Thread(
            target=refiner.refine,
            args=(
                np.zeros((2, 2, 4), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.float32),
                2,
                2,
            ),
        )
        cleanup_thread = threading.Thread(target=refiner.cleanup)
        refine_thread.start()
        self.assertTrue(infer_started.wait(1.0))
        cleanup_thread.start()
        time.sleep(0.03)
        backend.cleanup.assert_not_called()

        release_infer.set()
        refine_thread.join(1.0)
        cleanup_thread.join(1.0)

        backend.cleanup.assert_called_once_with()
        self.assertFalse(refiner._initialized)

    def test_gpu_index_change_retargets_main_and_optional_refiners(self):
        effects = self._make_effects()
        effects._initialized = False
        effects._edge_refiner.set_gpu_index = mock.Mock(return_value=True)
        for refiner in effects._learned_refiners.values():
            refiner.set_gpu_index = mock.Mock(return_value=True)

        self.assertTrue(effects.set_gpu_index(1))

        self.assertEqual(effects._gpu_index, 1)
        effects._edge_refiner.set_gpu_index.assert_called_once_with(1)
        for refiner in effects._learned_refiners.values():
            refiner.set_gpu_index.assert_called_once_with(1)

    def test_gpu_index_change_invalidates_old_device_state_before_reload(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._bg_removal_enabled = True
        effects._cpu_inference = False
        effects._compositing = "cupy"
        effects._cupy = object()
        effects._use_fused_kernel = True

        class _Backend:
            def __init__(self, gpu_index):
                self._gpu_index = gpu_index
                self._cpu_only = False
                self._MAX_INFER_HEIGHT = 720
                self._trt_requested = False
                self._quality = effects._quality
                self.session = mock.Mock()
                self.session.get_providers.return_value = ["CUDAExecutionProvider"]
                self.infer_calls = 0
                self.cleaned = False

            def infer(self, *_args):
                self.infer_calls += 1
                return None

            def reset_state(self):
                return None

            def cleanup(self):
                self.cleaned = True

        active = _Backend(0)
        candidate = _Backend(1)
        effects._backend = active
        effects._cached_alpha = object()
        effects._cached_source_alpha = object()
        effects._prev_alpha_gpu = object()
        effects._latest_final_matte_u8_gpu = object()
        effects._pending_frame_gpu = (object(), object())
        effects._fused_bg_gpu = object()
        effects._fused_green_bg_gpu = object()
        effects._fused_face_mask_gpu = object()
        effects._fused_vignette_gpu = object()
        effects._gpu_morph_footprints = {3: object()}
        build_started = threading.Event()
        release_build = threading.Event()

        def build_backend():
            build_started.set()
            self.assertTrue(release_build.wait(1.0))
            return candidate, "ready"

        effects._build_backend = build_backend
        try:
            self.assertTrue(effects.set_gpu_index(1))
            self.assertTrue(build_started.wait(1.0))

            self.assertTrue(effects._engine_reload_in_progress)
            self.assertFalse(effects.gpu_output_eligible())
            self.assertIsNone(effects._cached_alpha)
            self.assertIsNone(effects._cached_source_alpha)
            self.assertIsNone(effects._prev_alpha_gpu)
            self.assertIsNone(effects._latest_final_matte_u8_gpu)
            self.assertIsNone(effects._pending_frame_gpu)
            self.assertIsNone(effects._fused_bg_gpu)
            self.assertIsNone(effects._fused_green_bg_gpu)
            self.assertIsNone(effects._fused_face_mask_gpu)
            self.assertIsNone(effects._fused_vignette_gpu)
            self.assertEqual(effects._gpu_morph_footprints, {})

            frame = np.zeros((4, 4, 4), dtype=np.uint8)
            self.assertIsNone(
                effects._run_inference(frame, 4, 4, effects._matte_version)
            )
            self.assertEqual(active.infer_calls, 0)
        finally:
            release_build.set()

        deadline = time.monotonic() + 1.0
        while effects._engine_reload_in_progress and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIs(effects._backend, candidate)
        self.assertTrue(active.cleaned)

    def test_provider_reload_failure_never_restores_incompatible_gpu_backend(self):
        effects = self._make_effects()
        effects._initialized = True
        active = mock.Mock()
        active._cpu_only = False
        active._gpu_index = 0
        active._MAX_INFER_HEIGHT = 720
        active._trt_requested = False
        active._quality = effects._quality
        effects._backend = active
        effects._build_backend = mock.Mock(side_effect=RuntimeError("CPU load failed"))

        effects.set_compositing("cpu")
        effects.set_engine_mode(False, False)
        deadline = time.monotonic() + 1.0
        while effects._engine_reload_in_progress and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(effects._engine_reload_in_progress)
        self.assertIsNone(effects._backend)
        self.assertFalse(effects._initialized)
        active.cleanup.assert_called_once_with()

    def test_mode_reselection_retries_after_incompatible_reload_failure(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._bg_removal_enabled = True
        effects._cpu_inference = False
        effects._compositing = "cupy"
        effects._cupy = object()

        class _Backend:
            def __init__(self, cpu_only):
                self._cpu_only = cpu_only
                self._gpu_index = 0
                self._MAX_INFER_HEIGHT = 720
                self._trt_requested = False
                self._quality = effects._quality
                self.cleaned = False

            def reset_state(self):
                return None

            def cleanup(self):
                self.cleaned = True

        active = _Backend(False)
        recovered = _Backend(True)
        effects._backend = active
        effects._build_backend = mock.Mock(
            side_effect=[RuntimeError("CPU load failed"), (recovered, "ready")]
        )

        effects.set_compositing("cpu")
        effects.set_engine_mode(False, False)
        deadline = time.monotonic() + 1.0
        while effects._engine_reload_in_progress and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(effects._backend)
        self.assertFalse(effects._initialized)

        effects.set_engine_mode(False, False)
        deadline = time.monotonic() + 1.0
        while effects._engine_reload_in_progress and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertIs(effects._backend, recovered)
        self.assertTrue(effects._initialized)
        self.assertEqual(effects._build_backend.call_count, 2)

    def test_gpu_mode_reselection_retries_actual_cpu_fallback(self):
        effects = self._make_effects()
        effects._cpu_inference = False
        effects._backend = types.SimpleNamespace(
            _cpu_only=False,
            _MAX_INFER_HEIGHT=720,
            _trt_requested=False,
            _quality=effects._quality,
            session=mock.Mock(),
        )
        effects._backend.session.get_providers.return_value = ["CPUExecutionProvider"]
        with mock.patch.object(effects, "_schedule_engine_reload") as reload_backend:
            effects.set_engine_mode(False, True)
        reload_backend.assert_called_once()

    def test_explicit_cpu_mode_does_not_retry_cuda(self):
        effects = self._make_effects()
        effects._cpu_inference = True
        effects._backend = types.SimpleNamespace(
            _cpu_only=True,
            _MAX_INFER_HEIGHT=effects._resolve_max_infer_height(),
            _trt_requested=False,
            _quality=effects._quality,
            session=mock.Mock(),
        )
        effects._backend.session.get_providers.return_value = ["CPUExecutionProvider"]
        with mock.patch.object(effects, "_schedule_engine_reload") as reload_backend:
            effects.set_engine_mode(False, False)
        reload_backend.assert_not_called()

    def test_device_reselection_retries_after_incompatible_reload_failure(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._bg_removal_enabled = True
        effects._cpu_inference = False
        effects._compositing = "cupy"
        effects._cupy = object()

        class _Backend:
            def __init__(self, gpu_index):
                self._cpu_only = False
                self._gpu_index = gpu_index
                self._MAX_INFER_HEIGHT = 720
                self._trt_requested = False
                self._quality = effects._quality

            def reset_state(self):
                return None

            def cleanup(self):
                return None

        effects._backend = _Backend(0)
        recovered = _Backend(1)
        effects._build_backend = mock.Mock(
            side_effect=[RuntimeError("GPU load failed"), (recovered, "ready")]
        )

        effects.set_gpu_index(1)
        deadline = time.monotonic() + 1.0
        while effects._engine_reload_in_progress and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(effects._backend)
        self.assertFalse(effects._initialized)

        self.assertTrue(effects.set_gpu_index(1))
        deadline = time.monotonic() + 1.0
        while effects._engine_reload_in_progress and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertIs(effects._backend, recovered)
        self.assertTrue(effects._initialized)
        self.assertEqual(effects._build_backend.call_count, 2)

    def test_compositing_provider_change_blocks_inference_until_reload_barrier(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._bg_removal_enabled = True
        effects._cpu_inference = True
        effects._compositing = "cpu"
        effects._cupy = object()
        effects._use_fused_kernel = True
        effects._use_tensorrt = False
        infer_started = threading.Event()
        release_infer = threading.Event()
        compositing_set = threading.Event()
        allow_engine_mode = threading.Event()
        switch_done = threading.Event()
        build_started = threading.Event()
        release_build = threading.Event()
        built_policies = []

        class _Backend:
            def __init__(self, cpu_only):
                self._cpu_only = cpu_only
                self._gpu_index = 0
                self._MAX_INFER_HEIGHT = 720
                self._trt_requested = False
                self._quality = effects._quality
                self.infer_calls = 0
                self.infer_gpu_calls = 0

            def infer(self, *_args):
                self.infer_calls += 1
                infer_started.set()
                release_infer.wait(1.0)
                return None

            def infer_gpu(self, *_args):
                self.infer_gpu_calls += 1
                return None

            def reset_state(self):
                return None

            def cleanup(self):
                return None

        active = _Backend(True)
        candidate = _Backend(False)
        effects._backend = active

        def build_backend():
            built_policies.append(
                (effects._use_tensorrt, effects._use_fused_kernel, effects._quality)
            )
            build_started.set()
            self.assertTrue(release_build.wait(1.0))
            return candidate, "ready"

        def switch_compositing_and_engine():
            effects.set_compositing("cupy")
            compositing_set.set()
            self.assertTrue(allow_engine_mode.wait(1.0))
            effects.set_engine_mode(True, True, quality="performance")
            switch_done.set()

        effects._build_backend = build_backend
        frame = np.zeros((4, 4, 4), dtype=np.uint8)
        infer_thread = threading.Thread(
            target=effects._run_inference,
            args=(frame, 4, 4, effects._matte_version),
        )
        switch_thread = threading.Thread(
            target=switch_compositing_and_engine
        )

        infer_thread.start()
        self.assertTrue(infer_started.wait(1.0))
        switch_thread.start()
        time.sleep(0.03)
        self.assertFalse(switch_done.is_set())

        release_infer.set()
        infer_thread.join(1.0)
        self.assertTrue(compositing_set.wait(1.0))
        self.assertTrue(effects._engine_reload_in_progress)
        self.assertIsNone(effects._engine_reload_target)
        self.assertFalse(build_started.wait(0.05))
        self.assertIsNone(
            effects._run_inference(frame, 4, 4, effects._matte_version)
        )
        self.assertEqual(active.infer_calls, 1)
        self.assertEqual(active.infer_gpu_calls, 0)

        allow_engine_mode.set()
        switch_thread.join(1.0)
        self.assertTrue(switch_done.is_set())
        self.assertTrue(build_started.wait(1.0))
        self.assertEqual(built_policies, [(True, True, "performance")])

        release_build.set()
        deadline = time.monotonic() + 1.0
        while effects._engine_reload_in_progress and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIs(effects._backend, candidate)

    def test_rapid_provider_reversal_cancels_stale_reload(self):
        for start_cpu in (False, True):
            with self.subTest(start_cpu=start_cpu):
                effects = self._make_effects()
                effects._initialized = True
                effects._cupy = object()

                class _Backend:
                    def __init__(self, cpu_only):
                        self._cpu_only = cpu_only
                        self._gpu_index = 0
                        self._MAX_INFER_HEIGHT = 720
                        self._trt_requested = False
                        self._quality = effects._quality
                        self.cleaned = False

                    def cleanup(self):
                        self.cleaned = True

                    def reset_state(self):
                        return None

                active = _Backend(start_cpu)
                effects._backend = active
                effects._cpu_inference = start_cpu
                effects._compositing = "cpu" if start_cpu else "cupy"
                first_build_started = threading.Event()
                release_first_build = threading.Event()
                built = []

                def build_backend():
                    candidate = _Backend(effects._cpu_inference)
                    built.append(candidate)
                    if len(built) == 1:
                        first_build_started.set()
                        self.assertTrue(release_first_build.wait(1.0))
                    return candidate, "ready"

                effects._build_backend = build_backend
                first_target = "cupy" if start_cpu else "cpu"
                final_target = "cpu" if start_cpu else "cupy"

                effects.set_compositing(first_target)
                effects.set_engine_mode(False, False)
                self.assertTrue(first_build_started.wait(1.0))
                effects.set_compositing(final_target)
                effects.set_engine_mode(False, False)
                release_first_build.set()

                deadline = time.monotonic() + 1.0
                while effects._engine_reload_in_progress and time.monotonic() < deadline:
                    time.sleep(0.01)

                self.assertFalse(effects._engine_reload_in_progress)
                self.assertEqual(effects._backend._cpu_only, start_cpu)
                self.assertGreaterEqual(len(built), 2)
                self.assertTrue(built[0].cleaned)

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
        self.assertFalse(
            effects._use_tensorrt,
            "target engine flags must not change while old inference is in flight",
        )

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

    def test_quality_change_reloads_instead_of_mutating_live_rvm_ratio(self):
        effects = self._make_effects()
        effects._initialized = True
        backend = self.effects_module._RVMBackend(0)
        backend._quality = "quality"
        backend._downsample_ratio = np.array([0.375], dtype=np.float32)
        effects._backend = backend
        effects._quality = "quality"
        effects._schedule_engine_reload = mock.Mock()

        effects.quality = "ultra"

        self.assertEqual(effects.quality, "ultra")
        self.assertEqual(float(backend._downsample_ratio[0]), 0.375)
        effects._schedule_engine_reload.assert_called_once_with(False, 720)

    def test_combined_quality_and_engine_change_schedules_one_reload(self):
        effects = self._make_effects()
        effects._initialized = True
        backend = self.effects_module._RVMBackend(0)
        backend._quality = "quality"
        backend._trt_requested = False
        backend._MAX_INFER_HEIGHT = 720
        backend._downsample_ratio = np.array([0.375], dtype=np.float32)
        effects._backend = backend
        effects._quality = "quality"
        effects._schedule_engine_reload = mock.Mock()

        effects.set_engine_mode(
            True,
            True,
            quality="performance",
            profile_infer_height=288,
        )

        self.assertEqual(effects.quality, "performance")
        self.assertEqual(float(backend._downsample_ratio[0]), 0.375)
        self.assertEqual(backend._MAX_INFER_HEIGHT, 720)
        effects._schedule_engine_reload.assert_called_once_with(True, 360)

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

    def test_gpu_inference_cuda_failure_rebuilds_damaged_session(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._gpu_matte_eligible = mock.Mock(return_value=True)
        effects._cupy = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                Device=mock.Mock(return_value=contextlib.nullcontext()),
            ),
        )
        backend = mock.Mock()
        failure = RuntimeError(
            "BFCArena::AllocateRawInternal: Available memory of 10999040 "
            "is smaller than requested bytes of 17855232"
        )
        backend.infer_gpu.side_effect = failure
        backend._is_cuda_runtime_error.return_value = True
        backend._recover_cuda_session.return_value = True
        effects._backend = backend

        alpha = effects._run_inference(
            None,
            4,
            4,
            effects._matte_version,
            frame_gpu=object(),
        )

        self.assertIsNone(alpha)
        backend._invalidate_iobinding.assert_called_once_with()
        backend._recover_cuda_session.assert_called_once_with(failure)

    def test_cpu_fallback_disables_gpu_pure_frame_routing(self):
        effects = self._make_effects()
        effects._initialized = True
        effects._bg_removal_enabled = True
        effects._compositing = "cupy"
        effects._gpu_matte_eligible = mock.Mock(return_value=True)
        backend = mock.Mock()
        backend.session.get_providers.return_value = ["CPUExecutionProvider"]
        effects._backend = backend

        with mock.patch.object(
            self.effects_module,
            "_get_fused_kernel",
            return_value=object(),
        ):
            self.assertFalse(effects.gpu_output_eligible())

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
        effects._quality = "performance"

        alpha = np.zeros((11, 11), dtype=np.float32)
        alpha[1:10, 1:10] = 0.95
        alpha[4:7, 4:7] = 0.01

        refined = effects._refine_alpha(alpha)

        self.assertGreater(
            float(refined[5, 5]),
            0.80,
            "replace refinement should close broad artifact holes inside the silhouette",
        )

    def test_refine_alpha_quality_preserves_narrow_hairline_gap(self):
        for quality in ("quality", "ultra"):
            with self.subTest(quality=quality):
                effects = self._make_effects()
                effects._bg_mode = "replace"
                effects._quality = quality

                alpha = np.zeros((15, 15), dtype=np.float32)
                alpha[2:14, 3:12] = 0.95
                alpha[2:8, 7:8] = 0.02

                refined = effects._refine_alpha(alpha)

                self.assertLess(
                    float(refined[4, 7]),
                    0.25,
                    f"{quality} replace refinement should preserve narrow hairline openings",
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

    def test_quality_and_performance_replace_mattes_reopen_fine_gaps(self):
        quality = self._make_effects()
        quality._bg_mode = "replace"
        quality._quality = "quality"

        performance = self._make_effects()
        performance._bg_mode = "replace"
        performance._quality = "performance"

        alpha_closed = np.zeros((15, 15), dtype=np.float32)
        alpha_closed[2:14, 3:12] = 0.95

        alpha_open = alpha_closed.copy()
        alpha_open[2:8, 7:8] = 0.02

        quality._replacement_matte(alpha_closed)
        performance._replacement_matte(alpha_closed)
        quality_gap = quality._replacement_matte(alpha_open)
        performance_gap = performance._replacement_matte(alpha_open)

        for preset, matte in (("quality", quality_gap), ("performance", performance_gap)):
            with self.subTest(preset=preset):
                self.assertLess(
                    float(matte[4, 7]), 0.12,
                    "both presets should reopen a detected background gap",
                )
                self.assertGreater(float(matte[10, 7]), 0.90)

    def test_refine_alpha_preserves_narrow_exterior_finger_gap(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"

        alpha = np.zeros((21, 21), dtype=np.float32)
        alpha[3:18, 6:9] = 0.95
        alpha[3:18, 10:13] = 0.95

        refined = effects._refine_alpha(alpha)

        self.assertLess(
            float(refined[10, 9]),
            0.55,
            "replace refinement should keep narrow exterior finger gaps open",
        )

    def test_refine_alpha_preserves_arm_torso_exterior_gap(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"

        alpha = np.zeros((25, 25), dtype=np.float32)
        alpha[4:22, 5:11] = 0.95
        alpha[8:22, 13:20] = 0.95
        alpha[18:22, 10:14] = 0.95

        refined = effects._refine_alpha(alpha)

        self.assertLess(
            float(refined[12, 12]),
            0.60,
            "replace refinement should keep the raised-arm torso opening visible",
        )
        self.assertGreater(
            float(refined[19, 12]),
            0.80,
            "real hand-body contact should stay connected where the silhouette is actually closed",
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

    def test_despill_removes_saturated_backlight_color_from_soft_edge(self):
        effects = self._make_effects()

        fg = np.zeros((16, 16, 4), dtype=np.uint8)
        fg[:, :, :3] = (65, 60, 70)
        fg[:, :, 3] = 255
        fg[:, 4, :3] = (90, 55, 245)
        fg[:, 11, :3] = (90, 55, 245)

        alpha = np.zeros((16, 16), dtype=np.float32)
        alpha[:, 4] = 0.90
        alpha[:, 5:11] = 0.99
        alpha[:, 11] = 0.72

        cleaned = effects._despill_fringe(fg, alpha)

        for x in (4, 11):
            before = int(fg[8, x, 2]) - int(fg[8, x, 1])
            after = int(cleaned[8, x, 2]) - int(cleaned[8, x, 1])
            self.assertLess(
                after,
                before - 80,
                "saturated physical-background color should not survive as an edge outline",
            )
        self.assertTrue(
            np.array_equal(cleaned[8, 8], fg[8, 8]),
            "solid foreground color should remain unchanged",
        )

    def test_despill_removes_neutral_bright_halo_from_dark_hair_edge(self):
        effects = self._make_effects()

        fg = np.zeros((16, 16, 4), dtype=np.uint8)
        fg[:, :, :3] = (34, 35, 36)
        fg[:, :, 3] = 255
        fg[:, 4, :3] = (188, 190, 192)
        fg[:, 11, :3] = (188, 190, 192)

        alpha = np.zeros((16, 16), dtype=np.float32)
        alpha[:, 4] = 0.90
        alpha[:, 5:11] = 0.99
        alpha[:, 11] = 0.78

        cleaned = effects._despill_fringe(fg, alpha)

        for x in (4, 11):
            self.assertLess(
                int(cleaned[8, x, :3].mean()),
                int(fg[8, x, :3].mean()) - 55,
                "neutral wall color should not survive as a light hair outline",
            )
        self.assertTrue(
            np.array_equal(cleaned[8, 8], fg[8, 8]),
            "solid dark hair color should remain unchanged",
        )

    def test_despill_crops_to_active_subject_region(self):
        effects = self._make_effects()

        fg = np.zeros((480, 640, 4), dtype=np.uint8)
        fg[:, :, :3] = 24
        fg[:, :, 3] = 255
        fg[170:310, 260:380, 2] = 220
        fg[160:320, 245:255, :3] = 22

        alpha = np.zeros((480, 640), dtype=np.float32)
        alpha[170:310, 260:380] = 0.98
        alpha[160:320, 245:255] = 0.20

        calls = []

        def clean_reference(fg_arg, alpha_arg, solid_threshold=0.88):
            calls.append(fg_arg.shape)
            clean = fg_arg.copy()
            clean[:, :, 2] = 220
            return clean

        effects._clean_color_reference = clean_reference
        cleaned = effects._despill_fringe(fg, alpha)

        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0][0] * calls[0][1], fg.shape[0] * fg.shape[1])
        self.assertGreater(int(cleaned[200, 250, 2]), int(fg[200, 250, 2]))
        self.assertTrue(np.array_equal(cleaned[20, 20], fg[20, 20]))

    def test_apply_replace_uses_shared_foreground_cleanup(self):
        effects = self._make_effects()

        fg = np.zeros((4, 4, 4), dtype=np.uint8)
        fg[:, :, 2] = 40
        fg[:, :, 3] = 255
        alpha = np.ones((4, 4), dtype=np.float32)

        calls = []

        def prepare_replace(fg_arg, alpha_arg):
            calls.append((fg_arg.shape, alpha_arg.shape))
            cleaned = fg_arg.copy()
            cleaned[:, :, 2] = 220
            return cleaned

        effects._prepare_replace_foreground = prepare_replace
        out = effects._apply_replace(fg, alpha, 4, 4)

        self.assertEqual(calls, [((4, 4, 4), (4, 4))])
        self.assertEqual(int(out[2, 2, 2]), 220)

    def test_fused_replace_uses_shared_foreground_cleanup(self):
        effects = self._make_effects()
        effects._cupy = self._FakeCupy()

        fg = np.zeros((4, 4, 4), dtype=np.uint8)
        fg[:, :, 2] = 40
        fg[:, :, 3] = 255
        alpha = np.ones((4, 4), dtype=np.float32)

        calls = []

        def prepare_replace(fg_arg, alpha_arg):
            calls.append((fg_arg.shape, alpha_arg.shape))
            cleaned = fg_arg.copy()
            cleaned[:, :, 2] = 220
            return cleaned
        effects._prepare_replace_foreground = prepare_replace

        kernel_calls = []

        def fake_kernel(_grid, _block, args):
            fg_gpu = args[0]
            output_gpu = args[6]
            output_gpu[:] = fg_gpu
            kernel_calls.append(int(args[12]))

        original_kernel = self.effects_module._get_fused_kernel
        self.effects_module._get_fused_kernel = lambda: fake_kernel
        try:
            out = effects._composite_fused(fg, alpha, 4, 4)
        finally:
            self.effects_module._get_fused_kernel = original_kernel

        self.assertEqual(calls, [((4, 4, 4), (4, 4))])
        self.assertEqual(kernel_calls, [0])
        self.assertIsNotNone(out)
        self.assertEqual(int(out[2, 2, 2]), 220)

    def test_fused_remove_enables_gpu_fringe_cleanup(self):
        effects = self._make_effects()
        effects._bg_mode = "remove"
        effects._cupy = self._FakeCupy()

        fg = np.zeros((4, 4, 4), dtype=np.uint8)
        fg[:, :, 2] = 40
        fg[:, :, 3] = 255
        alpha = np.ones((4, 4), dtype=np.float32)

        clean_color = np.zeros((2, 2, 4), dtype=np.uint8)
        clean_color[:, :, 2] = 220
        clean_color[:, :, 3] = 255
        reference_calls = []
        kernel_calls = []

        def gpu_clean_reference(fg_arg, alpha_arg):
            reference_calls.append((fg_arg.shape, alpha_arg.shape))
            return clean_color, 2

        effects._gpu_clean_color_reference = gpu_clean_reference

        def fake_kernel(_grid, _block, args):
            fg_gpu = args[0]
            output_gpu = args[6]
            output_gpu[:] = fg_gpu
            kernel_calls.append(
                {
                    "clean_color": args[3],
                    "frame_width": int(args[8]),
                    "clean_width": int(args[9]),
                    "clean_height": int(args[10]),
                    "clean_scale": int(args[11]),
                    "despill": int(args[12]),
                }
            )
            if int(args[12]):
                output_gpu[:, :, 2] = args[3][0, 0, 2]

        original_kernel = self.effects_module._get_fused_kernel
        self.effects_module._get_fused_kernel = lambda: fake_kernel
        try:
            out = effects._composite_fused(fg, alpha, 4, 4)
        finally:
            self.effects_module._get_fused_kernel = original_kernel

        self.assertEqual(reference_calls, [((4, 4, 4), (4, 4))])
        self.assertEqual(len(kernel_calls), 1)
        self.assertIs(kernel_calls[0]["clean_color"], clean_color)
        argument_names = (
            "frame_width",
            "clean_width",
            "clean_height",
            "clean_scale",
            "despill",
        )
        self.assertEqual(
            {key: kernel_calls[0][key] for key in argument_names},
            {
                "frame_width": 4,
                "clean_width": 2,
                "clean_height": 2,
                "clean_scale": 2,
                "despill": 1,
            },
        )
        self.assertIsNotNone(out)
        self.assertEqual(int(out[2, 2, 2]), 220)

    def test_fused_blur_preserves_soft_edge_without_gpu_fringe_cleanup(self):
        effects = self._make_effects()
        effects._bg_mode = "blur"
        effects._cupy = self._FakeCupy()
        effects._gpu_blur_bgra = lambda fg_arg, _sigma: np.zeros_like(fg_arg)

        fg = np.zeros((4, 4, 4), dtype=np.uint8)
        fg[:, :, 3] = 255
        alpha = np.ones((4, 4), dtype=np.float32)
        clean_color = np.zeros((2, 2, 4), dtype=np.uint8)
        reference_calls = []
        kernel_flags = []

        def gpu_clean_reference(fg_arg, alpha_arg):
            reference_calls.append((fg_arg.shape, alpha_arg.shape))
            return clean_color, 2

        effects._gpu_clean_color_reference = gpu_clean_reference

        def fake_kernel(_grid, _block, args):
            args[6][:] = args[0]
            kernel_flags.append(int(args[12]))

        original_kernel = self.effects_module._get_fused_kernel
        self.effects_module._get_fused_kernel = lambda: fake_kernel
        try:
            out = effects._composite_fused(fg, alpha, 4, 4)
        finally:
            self.effects_module._get_fused_kernel = original_kernel

        self.assertEqual(reference_calls, [])
        self.assertEqual(kernel_flags, [0])
        self.assertIsNotNone(out)

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

    def test_edge_aware_replace_matte_hd_uses_transition_roi(self):
        effects = self._make_effects()
        effects._quality = "ultra"

        frame = np.zeros((576, 1024, 4), dtype=np.uint8)
        frame[:, :512, :3] = 25
        frame[:, 512:, :3] = 230
        frame[:, :, 3] = 255

        matte = np.zeros((576, 1024), dtype=np.float32)
        matte[220:360, 508] = 0.18
        matte[220:360, 510] = 0.34
        matte[220:360, 512] = 0.66
        matte[220:360, 514] = 0.82
        matte[220:360, 516:540] = 0.98
        matte[10:20, 10:20] = 0.98

        calls = []
        original_region = effects._edge_aware_replace_matte_region

        def counted_region(frame_arg, matte_arg, transition_arg, preserve_detail, downsample):
            calls.append(frame_arg.shape[:2])
            return original_region(frame_arg, matte_arg, transition_arg, preserve_detail, downsample)

        effects._edge_aware_replace_matte_region = counted_region
        refined = effects._edge_aware_replace_matte(frame, matte.copy())

        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0][0], frame.shape[0], "HD edge pass should crop vertically to the transition band")
        self.assertLess(calls[0][1], frame.shape[1], "HD edge pass should crop horizontally to the transition band")
        self.assertLess(float(refined[280, 510]), float(matte[280, 510]), "ROI edge should still tighten entry fringe")
        self.assertGreater(float(refined[280, 512]), float(matte[280, 512]), "ROI edge should still harden exit fringe")
        self.assertEqual(float(refined[12, 12]), float(matte[12, 12]), "solid pixels outside the ROI must be untouched")

    def test_edge_refinement_is_local_with_sparse_or_dense_partial_opacity(self):
        effects = self._make_effects()
        frame = np.zeros((64, 96, 4), dtype=np.uint8)
        frame[:, 48:, :3] = 230
        frame[:, :, 3] = 255
        sparse = np.zeros((64, 96), dtype=np.float32)
        sparse[:, 48:] = 1.0
        sparse[:, 44:52] = np.array(
            [0.02, 0.07, 0.11, 0.34, 0.66, 0.82, 0.97, 0.99], dtype=np.float32,
        )
        dense = sparse.copy()
        dense[:, :44] = 0.25
        dense[:, 52:] = 0.75
        # Unrelated partial-opacity pixels must not change the shared edge.
        # Read-only inputs also catch accidental in-place cleanup.
        frame.setflags(write=False)
        sparse.setflags(write=False)
        dense.setflags(write=False)
        for preserve_detail in (False, True):
            for downsample in (False, True):
                with self.subTest(detail=preserve_detail, downsample=downsample):
                    results = []
                    for matte in (sparse, dense):
                        transition = (matte > 0.05) & (matte < 0.95)
                        results.append(effects._edge_aware_replace_matte_region(
                            frame, matte, transition, preserve_detail, downsample,
                        ))
                    np.testing.assert_allclose(
                        results[0][:, 44:52], results[1][:, 44:52],
                        rtol=0, atol=1e-7,
                    )
                    self.assertTrue(np.all(results[0][:, :44] == 0))
                    self.assertTrue(np.all(results[0][:, 52:] == 1))
                    self.assertTrue(np.all(sparse[:, :44] == 0))
                    self.assertTrue(np.all(dense[:, :44] == 0.25))

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

    def test_active_alpha_roi_bounds_includes_480p_meeting_frames(self):
        effects = self._make_effects()

        alpha = np.zeros((480, 854), dtype=np.float32)
        alpha[120:360, 280:560] = 0.95

        bounds = effects._active_alpha_roi_bounds(alpha, threshold=0.03, pad=32)

        self.assertIsNotNone(bounds)
        x0, y0, x1, y1 = bounds
        self.assertLess((x1 - x0) * (y1 - y0), alpha.size)

        small_alpha = np.zeros((320, 480), dtype=np.float32)
        small_alpha[90:230, 160:320] = 0.95
        self.assertIsNone(
            effects._active_alpha_roi_bounds(small_alpha, threshold=0.03, pad=32),
            "small preview frames should keep the simple full-frame path",
        )

    def test_temporal_smooth_480p_uses_full_frame_for_edge_stability(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"

        prev = np.zeros((480, 854), dtype=np.float32)
        prev[120:360, 280:560] = 0.92
        alpha = np.zeros_like(prev)
        alpha[122:362, 282:562] = 0.95
        effects._prev_alpha = prev

        calls = []
        original_region = effects._temporal_smooth_region

        def counted_region(alpha_arg, prev_arg, *args):
            calls.append(alpha_arg.shape)
            return original_region(alpha_arg, prev_arg, *args)

        effects._temporal_smooth_region = counted_region
        result = effects._temporal_smooth(alpha)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], alpha.shape, "480p temporal smoothing should stay full-frame to avoid edge shimmer")
        self.assertEqual(float(result[0, 0]), 0.0, "inactive background outside the ROI should remain clipped")

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

    def test_final_replace_matte_stabilizer_damps_small_edge_changes(self):
        effects = self._make_effects()

        prev = np.zeros((8, 8), dtype=np.float32)
        prev[:, 3] = 0.24
        prev[:, 4] = 0.76
        matte = np.zeros((8, 8), dtype=np.float32)
        matte[:, 3] = 0.20
        matte[:, 4] = 0.80

        effects._latest_final_matte_u8 = np.clip(prev * 255.0, 0, 255).astype(np.uint8)
        effects._latest_final_matte_size = (8, 8)

        original_entry = float(matte[4, 3])
        original_exit = float(matte[4, 4])
        stabilized = effects._stabilize_replace_matte_edges(matte)

        self.assertGreater(float(stabilized[4, 3]), original_entry)
        self.assertLess(float(stabilized[4, 4]), original_exit)
        self.assertEqual(float(stabilized[0, 0]), 0.0)

    def test_final_matte_quality_preserves_narrow_finger_gap(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"
        effects._quality = "quality"

        frame = np.zeros((21, 21, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        alpha = np.zeros((21, 21), dtype=np.float32)
        alpha[3:18, 6:9] = 0.95
        alpha[3:18, 10:13] = 0.95

        refined = effects._refine_alpha(alpha)
        final = effects._final_matte(frame, refined)

        self.assertLess(
            float(final[10, 9]),
            0.20,
            "quality final matte should keep narrow finger gaps from being blurred shut",
        )

    def test_final_matte_quality_preserves_hairline_gap(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"
        effects._quality = "quality"

        frame = np.zeros((15, 15, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        alpha = np.zeros((15, 15), dtype=np.float32)
        alpha[2:14, 3:12] = 0.95
        alpha[2:8, 7:8] = 0.02

        refined = effects._refine_alpha(alpha)
        final = effects._final_matte(frame, refined)

        self.assertLess(
            float(final[4, 7]),
            0.20,
            "quality final matte should keep narrow hairline openings from being blurred shut",
        )

    def test_final_matte_quality_still_closes_broad_internal_hole(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"
        effects._quality = "quality"

        frame = np.zeros((11, 11, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        alpha = np.zeros((11, 11), dtype=np.float32)
        alpha[1:10, 1:10] = 0.95
        alpha[4:7, 4:7] = 0.01

        refined = effects._refine_alpha(alpha)
        final = effects._final_matte(frame, refined)

        self.assertGreater(
            float(final[5, 5]),
            0.65,
            "quality final matte should still close broad interior artifact holes",
        )

    def test_composite_caches_latest_final_matte_for_followup_effects(self):
        effects = self._make_effects()
        effects._bg_mode = "replace"

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, 4:, :3] = 220
        frame[:, :, 3] = 255
        alpha = np.zeros((8, 8), dtype=np.float32)
        alpha[:, 3:6] = 0.7

        effects._commit_alpha(alpha.copy(), effects._matte_version)
        effects._composite(frame.copy(), alpha.copy(), 8, 8, effects._matte_version)
        latest = effects.latest_final_matte_u8(8, 8)

        self.assertIsNotNone(latest, "composite should cache the final matte for same-frame followup effects")
        self.assertEqual(latest.shape, (8, 8))
        self.assertGreater(int(latest[4, 4]), 0)

    def test_replace_matte_cache_tracks_committed_alpha_generation(self):
        effects = self._make_effects()
        calls = []
        original = effects._replacement_matte

        def _counted(alpha, matte_version=None):
            calls.append(alpha.copy())
            return original(alpha, matte_version)

        effects._replacement_matte = _counted

        alpha1 = np.zeros((8, 8), dtype=np.float32)
        alpha1[:, 3:6] = 0.7
        effects._commit_alpha(alpha1, effects._matte_version)
        matte1_first = effects._replacement_matte_cached(alpha1, effects._matte_version)
        matte1_second = effects._replacement_matte_cached(alpha1, effects._matte_version)

        self.assertEqual(len(calls), 1, "same committed alpha should reuse the cached replacement matte")
        self.assertTrue(np.array_equal(matte1_first, matte1_second))

        alpha2 = alpha1.copy()
        alpha2[:, 2:6] = 0.7
        effects._commit_alpha(alpha2, effects._matte_version)
        matte2 = effects._replacement_matte_cached(alpha2, effects._matte_version)

        self.assertEqual(len(calls), 2, "a newly committed alpha must rebuild the replacement matte")
        self.assertFalse(np.array_equal(matte1_first, matte2))


if __name__ == "__main__":
    unittest.main()

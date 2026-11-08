"""Generated dispatch tests; no application/native providers or devices loaded."""
import ast
from contextlib import nullcontext
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _methods(relative, class_name, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods, decorator_list=[]),
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    return namespace[class_name]


class _Cupy:
    """NumPy-backed transfers plus injectable post-kernel download failure."""
    ndarray = type("DeviceArray", (), {})
    float32, int32, uint8 = np.float32, np.int32, np.uint8

    def __init__(self):
        self.fail_download = False
        self.fail_sync = False
        self.cuda = SimpleNamespace(Device=lambda _index: nullcontext(),
                                    Stream=SimpleNamespace(null=SimpleNamespace(synchronize=self.synchronize)))

    asarray = staticmethod(np.asarray)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    empty_like = staticmethod(np.empty_like)
    ascontiguousarray = staticmethod(np.ascontiguousarray)

    def asnumpy(self, value):
        if self.fail_download:
            raise RuntimeError("injected generated download failure")
        return value.copy()

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError("injected generated synchronization failure")


class FusedOverlayFallbackTests(unittest.TestCase):
    def setUp(self):
        self.cp = _Cupy()
        self.fail_kernel = False
        self.kernel_available = True

        def kernel(_grid, _blocks, args):
            if self.fail_kernel:
                raise RuntimeError("injected generated kernel failure")
            # Distinguishable generated operations expose both omissions and
            # duplicate application. The test exercises real dispatch, not CUDA.
            output = args[6]
            output[:] = args[0]
            if args[13] > 0 and args[4].any():
                output[:, :, :3] += 10
            if args[14] > 0 and not np.all(args[5] == 1):
                output[:, :, :3] += 20

        names = {
            "_composite_array", "_composite_array_on_device", "_composite_fused",
            "_composite_fused_on_device", "_composite_fused_gpu",
            "set_fused_face_overlay", "_set_fused_face_overlay_locked",
            "process_frame_array", "composite_only_array",
            "process_frame_gpu", "composite_only_gpu",
        }
        cls = _methods("src/nvbroadcast/video/effects.py", "VideoEffects", names,
                       {"np": np, "cv2": cv2,
                        "_get_fused_kernel": lambda: kernel if self.kernel_available else None})
        effects = cls()
        effects._gpu_cache_lock = threading.RLock()
        effects._state_lock = threading.RLock()
        effects._gpu_index = 1
        effects._cupy = self.cp
        effects._compositing = "cupy"
        effects._use_fused_kernel = True
        effects._bg_removal_enabled = effects._initialized = effects.enabled = True
        effects._bg_mode = "replace"
        effects._bg_image = np.zeros((8, 12, 4), np.uint8)
        effects._matte_version = effects._frame_counter = 0
        effects._skip_interval = 1
        effects._pending_frame_gpu = None
        for name in ("_fused_bg_gpu", "_fused_bg_source_id", "_fused_bg_size",
                     "_fused_face_mask_gpu", "_fused_face_mask_source_id",
                     "_fused_vignette_gpu", "_fused_vignette_source_id"):
            setattr(effects, name, None)
        self.alpha = np.ones((8, 12), np.float32)
        self.has_alpha = True
        effects._remember_frame_size = lambda *_args: None
        effects._matte_color_snapshot = lambda: (
            self.alpha if self.has_alpha else None, None, 0)
        effects._matte_snapshot = lambda: (None, 0)
        effects._run_inference = lambda *_args, **_kwargs: self.alpha if self.has_alpha else None
        effects._commit_alpha = lambda *_args: True
        effects._final_matte = lambda _frame, alpha, *_args, **_kwargs: alpha
        effects._get_bg_image = lambda *_args: effects._bg_image
        effects._prepare_replace_foreground = lambda frame, _alpha: frame
        effects._apply_replace = lambda frame, *_args, **_kwargs: frame
        effects.latest_final_matte_u8 = lambda *_args: None
        effects.gpu_output_eligible = lambda: False
        self.effects = effects

        self.overlay = (np.ones((8, 12), np.uint8), np.full((8, 12), .5, np.float32),
                        .2, .15, 3., .04, 1.6)
        self.flags = []

        def beautify(frame, *_args, **kwargs):
            self.flags.append(kwargs)
            if not kwargs["skip_enhance"]:
                frame[:, :, :3] += 10
            if not kwargs["skip_edge_darken"]:
                frame[:, :, :3] += 20
            return frame

        tracker = SimpleNamespace(ready=False)
        app_cls = _methods("src/nvbroadcast/app.py", "NVBroadcastApp", {"_process_frame"},
                           {"get_shared_landmarker": lambda: tracker})
        app = app_cls()
        app._beautifier = SimpleNamespace(enabled=True, prime_face_cache=lambda *_a, **_k: None,
            fused_overlay_inputs=lambda *_args: self.overlay, process_frame_array=beautify)
        app._eye_contact = SimpleNamespace(enabled=False)
        app._relighter = app._autoframe = SimpleNamespace(enabled=False)
        app._video_effects = effects
        app._perf_monitor = SimpleNamespace(tick=lambda: None)
        app._mirror = False
        app._inline_inference = True
        self.app = app

    def frame(self):
        raw = np.full((8, 12, 4), 40, np.uint8)
        result = self.app._process_frame(raw.tobytes(), 12, 8)
        return np.frombuffer(result, np.uint8).reshape(raw.shape)

    def test_fused_failure_applies_both_beauty_operations_once(self):
        self.fail_kernel = True
        result = self.frame()
        np.testing.assert_array_equal(result[:, :, :3], 70)
        self.assertFalse(self.flags[-1]["skip_enhance"])
        self.assertFalse(self.flags[-1]["skip_edge_darken"])

    def test_non_fused_compositor_does_not_skip_offered_beauty_overlay(self):
        self.frame()
        for compositing, cupy in (("cpu", self.cp), ("cupy", None)):
            with self.subTest(compositing=compositing, cupy_present=cupy is not None):
                self.effects._compositing, self.effects._cupy = compositing, cupy
                result = self.frame()
                np.testing.assert_array_equal(result[:, :, :3], 70)
                self.assertFalse(self.flags[-1]["skip_enhance"])
                self.assertFalse(self.flags[-1]["skip_edge_darken"])
                self.assertTrue(self.flags[-1]["cache_prepared"])

    def test_success_download_failure_and_recovery_do_not_omit_or_double_apply(self):
        for failed in (False, True, False):
            with self.subTest(download_failed=failed):
                self.cp.fail_download = failed
                result = self.frame()
                np.testing.assert_array_equal(result[:, :, :3], 70)
                self.assertEqual(self.flags[-1]["skip_enhance"], not failed)
                self.assertEqual(self.flags[-1]["skip_edge_darken"], not failed)
                self.assertTrue(self.flags[-1]["cache_prepared"])

    def test_sync_failure_after_kernel_completion_applies_beauty_once(self):
        self.frame()
        self.cp.fail_sync = True
        result = self.frame()
        np.testing.assert_array_equal(result[:, :, :3], 70)
        self.assertFalse(self.flags[-1]["skip_enhance"])
        self.assertFalse(self.flags[-1]["skip_edge_darken"])

    def test_missing_overlay_keeps_primed_geometry_without_skipping_beauty(self):
        self.frame()
        self.overlay = None
        result = self.frame()
        np.testing.assert_array_equal(result[:, :, :3], 70)
        self.assertFalse(self.flags[-1]["skip_enhance"])
        self.assertFalse(self.flags[-1]["skip_edge_darken"])
        self.assertTrue(self.flags[-1]["cache_prepared"])

    def test_disabled_background_does_not_reuse_previous_overlay_success(self):
        self.frame()
        self.effects.enabled = False
        result = self.frame()
        np.testing.assert_array_equal(result[:, :, :3], 70)
        self.assertFalse(self.flags[-1]["skip_enhance"])
        self.assertFalse(self.flags[-1]["skip_edge_darken"])

    def test_missing_alpha_does_not_reuse_previous_overlay_success(self):
        self.frame()
        self.has_alpha = False
        result = self.frame()
        np.testing.assert_array_equal(result[:, :, :3], 70)
        self.assertFalse(self.flags[-1]["skip_enhance"])
        self.assertFalse(self.flags[-1]["skip_edge_darken"])

    def test_disabled_and_ineligible_entries_clear_previous_output_marker(self):
        for method in ("process_frame_array", "composite_only_array",
                       "process_frame_gpu", "composite_only_gpu"):
            with self.subTest(method=method):
                self.effects.last_output_face_overlay = (True, True)
                self.effects._bg_removal_enabled = False
                getattr(self.effects, method)(np.zeros((8, 12, 4), np.uint8), 12, 8)
                self.assertEqual(self.effects.last_output_face_overlay, (False, False))

    def test_partial_overlay_only_skips_the_operation_that_was_applied(self):
        for invalid_part in (0, 1):
            with self.subTest(invalid_part=invalid_part):
                overlay = list(self.overlay)
                overlay[invalid_part] = np.ones((4, 6), overlay[invalid_part].dtype)
                self.overlay = tuple(overlay)
                result = self.frame()
                np.testing.assert_array_equal(result[:, :, :3], 70)
                self.assertEqual(self.flags[-1]["skip_enhance"], invalid_part != 0)
                self.assertEqual(self.flags[-1]["skip_edge_darken"], invalid_part != 1)
                self.overlay = (np.ones((8, 12), np.uint8), np.full((8, 12), .5, np.float32),
                                .2, .15, 3., .04, 1.6)

    def test_stale_gpu_matte_early_return_clears_previous_marker(self):
        self.cp.ndarray = np.ndarray
        self.effects.last_output_face_overlay = (True, True)
        frame = np.zeros((8, 12, 4), np.uint8)
        self.assertIs(self.effects._composite_array(frame, self.alpha, 12, 8, -1), frame)
        self.assertEqual(self.effects.last_output_face_overlay, (False, False))

    def test_gpu_missing_alpha_and_core_no_background_clear_previous_marker(self):
        self.has_alpha = False
        self.effects.gpu_output_eligible = lambda: True
        frame = np.zeros((8, 12, 4), np.uint8)
        for method in ("composite_only_gpu", "process_frame_gpu"):
            with self.subTest(method=method):
                self.effects.last_output_face_overlay = (True, True)
                self.assertIsNone(getattr(self.effects, method)(frame, 12, 8))
                self.assertEqual(self.effects.last_output_face_overlay, (False, False))
        self.effects.last_output_face_overlay = (True, True)
        self.assertIsNone(self.effects._composite_fused_gpu(frame, self.alpha, 12, 8))
        self.assertEqual(self.effects.last_output_face_overlay, (False, False))

    def test_successful_fused_mirroring_preserves_both_output_markers(self):
        self.effects.set_fused_face_overlay(*self.overlay)
        frame = np.full((8, 12, 4), 40, np.uint8)
        output = self.effects.process_frame_array(frame, 12, 8, mirror=True)
        np.testing.assert_array_equal(output[:, :, :3], 70)
        self.assertTrue(self.effects.last_output_mirrored)
        self.assertEqual(self.effects.last_output_face_overlay, (True, True))

    def test_gpu_output_success_then_absence_and_failure_clear_marker(self):
        self.effects._bg_mode = "blur"
        self.effects.gpu_output_eligible = lambda: True
        self.effects._matte_snapshot = lambda: (self.alpha, 0)
        self.effects._store_final_matte_gpu = lambda *_args: None
        self.effects._gpu_blur_bgra = lambda *_args: self.effects._bg_image
        self.effects._blur_sigma = 2
        self.effects.set_fused_face_overlay(*self.overlay)
        frame = np.full((8, 12, 4), 40, np.uint8)
        for method in ("composite_only_gpu", "process_frame_gpu"):
            for outcome in ("success", "no_kernel", "success", "kernel_error"):
                with self.subTest(method=method, outcome=outcome):
                    self.kernel_available = outcome != "no_kernel"
                    self.fail_kernel = outcome == "kernel_error"
                    result = getattr(self.effects, method)(frame, 12, 8)
                    if outcome == "success":
                        self.assertIsNotNone(result)
                        self.assertEqual(self.effects.last_output_face_overlay, (True, True))
                    else:
                        self.assertIsNone(result)
                        self.assertEqual(self.effects.last_output_face_overlay, (False, False))


if __name__ == "__main__":
    unittest.main()

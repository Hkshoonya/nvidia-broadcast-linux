"""Selected-device routing on fresh threads without CUDA or model execution."""

from contextlib import contextmanager
from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np

from nvbroadcast.video.effects import VideoEffects


class _DeviceArray(np.ndarray):
    def __array_finalize__(self, original):
        self.device_id = getattr(original, "device_id", None)
        self.owner = getattr(original, "owner", None)

    def min(self, *args, **kwargs):
        self.owner.event("reduce")
        if self.owner.current != self.device_id:
            raise AssertionError("reduction used the wrong current device")
        return super().min(*args, **kwargs)

    def max(self, *args, **kwargs):
        self.owner.event("reduce")
        if self.owner.current != self.device_id:
            raise AssertionError("reduction used the wrong current device")
        return super().max(*args, **kwargs)


class _CupyDevices:
    ndarray = _DeviceArray
    float32, int32, uint8, newaxis = np.float32, np.int32, np.uint8, np.newaxis

    def __init__(self):
        self.local = threading.local()
        self.operations = []
        self.fail_upload = False
        self.cuda = SimpleNamespace(
            Device=self.device,
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: self.event("sync"))))

    @property
    def current(self):
        return getattr(self.local, "device", 0)

    @current.setter
    def current(self, value):
        self.local.device = value

    @contextmanager
    def device(self, selected):
        original = self.current
        self.current = selected
        try:
            yield
        finally:
            self.current = original

    def event(self, name):
        self.operations.append((name, self.current))

    def owned(self, value):
        result = np.asarray(value).view(_DeviceArray)
        result.device_id = self.current
        result.owner = self
        return result

    def asarray(self, value, dtype=None):
        self.event("upload")
        if self.fail_upload:
            raise RuntimeError("synthetic allocation failure")
        if isinstance(value, _DeviceArray) and value.device_id != self.current:
            raise AssertionError("cross-device array reached the compositor")
        return self.owned(np.asarray(value, dtype=dtype))

    def asnumpy(self, value):
        self.event("download")
        if value.device_id != self.current:
            raise AssertionError("download used the wrong current device")
        return np.asarray(value).copy()

    def empty_like(self, value):
        self.event("allocate")
        return self.owned(np.empty_like(value))

    def zeros(self, shape, dtype=None):
        self.event("allocate")
        return self.owned(np.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=None):
        self.event("allocate")
        return self.owned(np.ones(shape, dtype=dtype))

    def clip(self, value, low, high):
        self.event("clip")
        if isinstance(value, _DeviceArray) and value.device_id != self.current:
            raise AssertionError("matte clipping used the wrong device")
        return self.owned(np.clip(np.asarray(value), low, high))


class CompositorDeviceTests(unittest.TestCase):
    def make_effects(self, selected=1):
        effects = VideoEffects(gpu_index=selected, compositing="cpu")
        cp = _CupyDevices()
        effects._cupy = cp
        effects._compositing = "cupy"
        effects._bg_mode = "replace"
        effects._bg_image = np.full((4, 4, 4), 200, np.uint8)
        effects._prepare_replace_foreground = lambda frame, _alpha: frame
        frame = np.full((4, 4, 4), 60, np.uint8)
        alpha = np.full((4, 4), 0.5, np.float32)
        return effects, cp, frame, alpha

    def kernel(self, cp, fail=False):
        def apply(_grid, _blocks, args):
            cp.event("kernel")
            for value in args:
                if isinstance(value, _DeviceArray):
                    self.assertEqual(value.device_id, cp.current)
            if fail:
                raise RuntimeError("synthetic kernel failure")
            args[6][:] = args[0]
        return apply

    def test_both_cpu_entrypoints_select_gpu_on_fresh_threads_and_restore_caller(self):
        for route in ("fused", "blend"):
            with self.subTest(route=route):
                effects, cp, frame, alpha = self.make_effects()
                cp.current = 3
                observed, errors = [], []

                def worker():
                    try:
                        observed.append(cp.current)
                        if route == "fused":
                            output = effects._composite_fused(frame, alpha, 4, 4)
                            self.assertTrue(np.array_equal(output, frame))
                        else:
                            output = effects._blend_cupy(frame, effects._bg_image, alpha)
                            self.assertTrue(np.array_equal(output, np.full_like(frame, 130)))
                        observed.append(cp.current)
                    except BaseException as error:
                        errors.append(error)

                with mock.patch("nvbroadcast.video.effects._get_fused_kernel", return_value=self.kernel(cp)):
                    thread = threading.Thread(target=worker)
                    thread.start()
                    thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors, errors)
                self.assertEqual(observed, [0, 0])
                self.assertEqual(cp.current, 3)
                self.assertTrue(cp.operations)
                self.assertEqual({device for _name, device in cp.operations}, {1})

    def test_fused_caches_stay_selected_and_public_device_change_invalidates_them(self):
        effects, cp, frame, alpha = self.make_effects()
        with mock.patch("nvbroadcast.video.effects._get_fused_kernel", return_value=self.kernel(cp)):
            effects._composite_fused(frame, alpha, 4, 4)
            cached = effects._fused_bg_gpu
            self.assertEqual(cached.device_id, 1)
            effects._composite_fused(frame, alpha, 4, 4)
            self.assertIs(effects._fused_bg_gpu, cached)
            with cp.device(1):
                effects._pending_frame_gpu = (frame, cp.asarray(frame))
            self.assertTrue(effects.set_gpu_index(0))
            self.assertIsNone(effects._fused_bg_gpu)
            self.assertIsNone(effects._pending_frame_gpu)
            output = effects._composite_fused(frame, alpha, 4, 4)
        self.assertTrue(np.array_equal(output, frame))
        self.assertEqual(effects._fused_bg_gpu.device_id, 0)
        self.assertIsNot(effects._fused_bg_gpu, cached)
        self.assertEqual(cp.current, 0)

    def test_gpu_switch_during_upload_waits_then_clears_old_device_caches(self):
        effects, cp, frame, alpha = self.make_effects()
        uploading, release, setter_started, switched = (
            threading.Event() for _ in range(4))
        original_upload = cp.asarray
        errors, outputs = [], []

        def upload(value, dtype=None):
            if not uploading.is_set():
                uploading.set()
                if not release.wait(2):
                    raise AssertionError("test did not release the upload")
            return original_upload(value, dtype)

        def composite():
            try:
                outputs.append(effects._composite_fused(frame, alpha, 4, 4))
            except BaseException as error:
                errors.append(error)

        def switch():
            setter_started.set()
            try:
                effects.set_gpu_index(0)
            except BaseException as error:
                errors.append(error)
            finally:
                switched.set()

        cp.asarray = upload
        with mock.patch("nvbroadcast.video.effects._get_fused_kernel", return_value=self.kernel(cp)):
            worker, setter = (threading.Thread(target=composite, daemon=True),
                              threading.Thread(target=switch, daemon=True))
            worker.start()
            try:
                self.assertTrue(uploading.wait(1))
                acquired = effects._lock.acquire(blocking=False)
                if acquired:
                    effects._lock.release()
                self.assertTrue(acquired, "composition must not serialize with asynchronous inference")
                acquired = effects._gpu_cache_lock.acquire(blocking=False)
                if acquired:
                    effects._gpu_cache_lock.release()
                self.assertFalse(acquired, "upload/cache publication must serialize with GPU selection")
                setter.start()
                self.assertTrue(setter_started.wait(1))
                self.assertFalse(switched.wait(0.05))
            finally:
                release.set()
                worker.join(2)
                if setter.ident is not None:
                    setter.join(2)
            self.assertFalse(worker.is_alive())
            self.assertFalse(setter.is_alive())
            self.assertFalse(errors, errors)
            self.assertTrue(np.array_equal(outputs[0], frame))
            self.assertTrue(switched.is_set())
            self.assertEqual(effects._gpu_index, 0)
            self.assertIsNone(effects._fused_bg_gpu)
            output = effects._composite_fused(frame, alpha, 4, 4)
        self.assertTrue(np.array_equal(output, frame))
        self.assertEqual(effects._fused_bg_gpu.device_id, 0)

    def test_compositor_proceeds_while_other_thread_holds_inference_lock(self):
        effects, cp, frame, alpha = self.make_effects()
        complete, errors, outputs = threading.Event(), [], []

        def composite():
            try:
                outputs.append(effects._composite_fused(frame, alpha, 4, 4))
            except BaseException as error:
                errors.append(error)
            finally:
                complete.set()

        with mock.patch("nvbroadcast.video.effects._get_fused_kernel", return_value=self.kernel(cp)):
            with effects._lock:
                worker = threading.Thread(target=composite, daemon=True)
                worker.start()
                self.assertTrue(complete.wait(1), "render waited on the inference lock")
            worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        self.assertTrue(np.array_equal(outputs[0], frame))

    def test_old_gpu_matte_is_not_reduced_or_published_after_device_change(self):
        effects, cp, frame, alpha = self.make_effects()
        version = effects._matte_version
        with cp.device(1):
            gpu_alpha = cp.asarray(alpha)
        effects.set_gpu_index(0)
        cp.operations.clear()
        self.assertIs(effects._composite_array(frame, gpu_alpha, 4, 4, version), frame)
        self.assertFalse(effects._commit_alpha(gpu_alpha, version))
        self.assertEqual(cp.operations, [])

    def test_failures_restore_previous_device_and_preserve_existing_fallbacks(self):
        for route in ("fused", "blend"):
            with self.subTest(route=route):
                effects, cp, frame, alpha = self.make_effects(selected=0)
                cp.current = 3
                if route == "fused":
                    with mock.patch("nvbroadcast.video.effects._get_fused_kernel", return_value=self.kernel(cp, fail=True)):
                        self.assertIsNone(effects._composite_fused(frame, alpha, 4, 4))
                    self.assertEqual(effects._compositing, "cupy")
                else:
                    cp.fail_upload = True
                    output = effects._blend_cupy(frame, effects._bg_image, alpha)
                    expected = effects._blend_cpu(frame, effects._bg_image, alpha)
                    self.assertTrue(np.array_equal(output, expected))
                    self.assertEqual(effects._compositing, "cpu")
                self.assertEqual(cp.current, 3)
                self.assertEqual({device for _name, device in cp.operations}, {0})

    def test_gpu_matte_bookkeeping_and_lazy_download_keep_selected_device(self):
        effects, cp, frame, alpha = self.make_effects()
        alpha[0, 0] = 0.7
        effects._bg_mode = "blur"
        effects._use_fused_kernel = True
        effects._final_matte = lambda _frame, matte, *_args, **_kwargs: matte
        effects._gpu_blur_bgra = lambda fg, _sigma: cp.zeros(fg.shape, cp.uint8)
        with cp.device(1):
            gpu_alpha = cp.asarray(alpha)
        cp.operations.clear()
        with mock.patch("nvbroadcast.video.effects._get_fused_kernel", return_value=self.kernel(cp)):
            output = effects._composite_array(frame, gpu_alpha, 4, 4)
        self.assertTrue(np.array_equal(output, frame))
        self.assertEqual(effects._latest_final_matte_u8_gpu.device_id, 1)
        matte = effects.latest_final_matte_u8(4, 4)
        self.assertTrue(np.array_equal(matte, (alpha * 255).astype(np.uint8)))
        self.assertFalse(effects._alpha_is_degenerate(gpu_alpha))
        self.assertEqual(cp.current, 0)
        self.assertEqual({device for _name, device in cp.operations}, {1})

    def test_numpy_matte_reductions_never_select_a_gpu(self):
        effects, cp, _frame, alpha = self.make_effects()
        alpha[0, 0] = 0.7
        cp.cuda.Device = mock.Mock(side_effect=AssertionError("CPU reduction entered CUDA"))
        self.assertFalse(effects._alpha_is_degenerate(alpha))
        cp.cuda.Device.assert_not_called()

    def test_device_selection_failure_keeps_cpu_fallback_available(self):
        effects, cp, frame, alpha = self.make_effects()
        cp.cuda.Device = mock.Mock(side_effect=RuntimeError("synthetic unavailable device"))
        self.assertIsNone(effects._composite_fused(frame, alpha, 4, 4))
        output = effects._blend_cupy(frame, effects._bg_image, alpha)
        self.assertTrue(np.array_equal(output, effects._blend_cpu(frame, effects._bg_image, alpha)))
        self.assertEqual(effects._compositing, "cpu")
        self.assertEqual(cp.current, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nvbroadcast.video.beautify import FaceBeautifier


class _DeviceCupy:
    """Execute the unchanged array math on CPU while observing CUDA ownership."""

    def __init__(self, *, fail=None):
        self.current = 0
        self.fail = fail
        self.uploads = []
        self.downloads = []
        self.cuda = SimpleNamespace(Device=self.device)

    @contextmanager
    def device(self, index):
        if self.fail == "enter":
            raise RuntimeError("Generated device-entry failure")
        previous = self.current
        self.current = index
        try:
            yield
        finally:
            self.current = previous

    def __getattr__(self, name):
        return getattr(np, name)

    def asarray(self, value, **kwargs):
        self.uploads.append(self.current)
        if self.fail == "upload":
            raise RuntimeError("Generated upload failure")
        return np.asarray(value, **kwargs)

    def asnumpy(self, value):
        self.downloads.append(self.current)
        if self.fail == "download":
            raise RuntimeError("Generated download failure")
        return np.array(value, copy=True)


class BeautifyTests(unittest.TestCase):
    def _make_beautifier(self) -> FaceBeautifier:
        beautifier = FaceBeautifier()
        beautifier._initialized = True
        beautifier._enabled = True
        beautifier.denoise = 0.5
        beautifier._face_bbox = (2, 2, 4, 4)
        beautifier._face_mask = np.zeros((8, 8), dtype=np.uint8)
        beautifier._face_mask[2:6, 2:6] = 255
        return beautifier

    def test_denoise_preserves_raw_history(self):
        beautifier = self._make_beautifier()
        previous = np.full((8, 8, 3), 200, dtype=np.uint8)
        beautifier._prev_frame = previous.copy()

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, :3] = 40
        frame[:, :, 3] = 255
        raw_bgr = frame[:, :, :3].copy()

        beautifier._apply_denoise(frame)

        self.assertTrue(
            np.array_equal(beautifier._prev_frame, raw_bgr),
            "temporal denoise must keep raw frame history instead of recursively storing blurred output",
        )

    def test_denoise_does_not_blur_full_frame_without_face_mask(self):
        beautifier = FaceBeautifier()
        beautifier._initialized = True
        beautifier._enabled = True
        beautifier.denoise = 0.5
        beautifier._prev_frame = np.full((8, 8, 3), 220, dtype=np.uint8)

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, :3] = 30
        frame[:, :, 3] = 255
        original = frame.copy()

        result = beautifier._apply_denoise(frame)

        self.assertTrue(
            np.array_equal(result[:, :, :3], original[:, :, :3]),
            "without a face mask, denoise should not smear the whole composited frame",
        )
        self.assertTrue(
            np.array_equal(beautifier._prev_frame, original[:, :, :3]),
            "raw history should still update when face landmarks are unavailable",
        )

    def test_denoise_keeps_isolated_raw_history_from_strided_frames(self):
        rng = np.random.default_rng(392)
        for step in (2, -2):
            with self.subTest(step=step):
                storage = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
                frame = storage[::step, ::step]
                self.assertFalse(frame.flags.c_contiguous)
                original = frame.copy()
                beautifier = self._make_beautifier()
                reference = self._make_beautifier()
                beautifier._prev_frame = np.full((8, 8, 3), 200, dtype=np.uint8)
                reference._prev_frame = beautifier._prev_frame.copy()

                result = beautifier._apply_denoise(frame)
                expected = reference._apply_denoise(original.copy())

                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(result[:, :, 3], original[:, :, 3])
                np.testing.assert_array_equal(beautifier._prev_frame, original[:, :, :3])
                storage.fill(0)
                result.fill(17)
                np.testing.assert_array_equal(
                    beautifier._prev_frame,
                    original[:, :, :3],
                    err_msg="Later frame writes must not alter raw denoise history.",
                )

    def test_denoise_motion_gate_keeps_fast_motion_close_to_current_frame(self):
        beautifier = self._make_beautifier()
        beautifier._prev_frame = np.full((8, 8, 3), 255, dtype=np.uint8)

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        before = frame[3:5, 3:5, :3].copy()

        result = beautifier._apply_denoise(frame)
        after = result[3:5, 3:5, :3].astype(np.int16)

        self.assertLess(
            int(after.mean()),
            60,
            "fast motion should heavily reduce temporal blending so the face does not ghost",
        )
        self.assertTrue(np.array_equal(before, np.zeros_like(before)))

    def test_skin_smooth_skips_on_very_fast_face_motion(self):
        beautifier = self._make_beautifier()
        beautifier.skin_smooth = 0.3
        beautifier._face_motion_px = 24.0

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, 3] = 255

        with mock.patch("cv2.bilateralFilter") as bilateral:
            result = beautifier._apply_skin_smooth(frame.copy())

        self.assertTrue(np.array_equal(result, frame))
        bilateral.assert_not_called()

    def test_denoise_skips_temporal_work_on_fast_face_motion(self):
        beautifier = self._make_beautifier()
        beautifier.denoise = 0.4
        beautifier._face_motion_px = 24.0
        beautifier._prev_frame = np.full((8, 8, 3), 200, dtype=np.uint8)

        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        frame[:, :, 3] = 255

        with mock.patch("cv2.absdiff") as absdiff:
            result = beautifier._apply_denoise(frame.copy())

        self.assertTrue(np.array_equal(result[:, :, :3], frame[:, :, :3]))
        absdiff.assert_not_called()

    def test_hairline_taper_reduces_upper_face_mask_strength(self):
        mask = np.full((20, 20), 255, dtype=np.uint8)

        tapered = FaceBeautifier._apply_hairline_taper(mask, top=2, face_height=12)

        self.assertLess(int(tapered[2, 10]), int(tapered[10, 10]))
        self.assertGreater(int(tapered[10, 10]), 200)

    def test_tone_mask_stays_lower_and_inward_vs_detail_mask(self):
        mask = np.full((24, 24), 255, dtype=np.uint8)

        tone_mask = FaceBeautifier._build_tone_mask(mask, top=4, face_height=14)

        self.assertLess(int(tone_mask[4, 12]), int(tone_mask[12, 12]))
        self.assertLess(int(tone_mask[6, 1]), int(tone_mask[6, 12]))
        self.assertLess(int(tone_mask[4, 12]), 180)

    def test_fused_overlay_inputs_use_tone_mask_for_enhance(self):
        beautifier = self._make_beautifier()
        beautifier.enhance = 0.5
        beautifier._face_center = (4, 4)
        beautifier._tone_mask = np.zeros((8, 8), dtype=np.uint8)
        beautifier._tone_mask[3:5, 3:5] = 255

        overlay = beautifier.fused_overlay_inputs(8, 8)

        self.assertIsNotNone(overlay)
        self.assertTrue(np.array_equal(overlay[0], beautifier._tone_mask))

    def test_cpu_compositing_never_uses_cached_cupy_beautify_path(self):
        beautifier = self._make_beautifier()
        beautifier.denoise = 0.0
        beautifier.enhance = 0.5
        beautifier.sharpen = 0.5
        beautifier._cupy = object()
        beautifier.set_compositing("cpu")
        beautifier._apply_gpu_batch = mock.Mock()
        beautifier._apply_enhance = mock.Mock(side_effect=lambda frame: frame)
        beautifier._apply_sharpen = mock.Mock(side_effect=lambda frame: frame)
        frame = np.zeros((8, 8, 4), dtype=np.uint8)

        beautifier.process_frame_array(
            frame,
            8,
            8,
            cache_prepared=True,
        )

        beautifier._apply_gpu_batch.assert_not_called()
        beautifier._apply_enhance.assert_called_once_with(frame)
        beautifier._apply_sharpen.assert_called_once_with(frame)


class BeautifyGpuSelectionTests(unittest.TestCase):
    @staticmethod
    def _face(cp):
        face = FaceBeautifier()
        face._cupy = cp
        face._compositing = "cupy"
        face._gpu_index = 1
        face._face_bbox = (2, 2, 4, 4)
        face._face_mask = np.full((8, 8), 255, np.uint8)
        face._tone_mask = face._face_mask.copy()
        face.enhance = 0.3
        face.sharpen = 0.2
        frame = np.arange(8 * 8 * 4, dtype=np.uint8).reshape(8, 8, 4)
        return face, frame

    def test_selected_device_owns_uploads_and_downloads_and_restores_caller(self):
        cp = _DeviceCupy()
        face, frame = self._face(cp)

        result = face._apply_gpu_batch(frame.copy(), 8, 8)

        self.assertIsNotNone(result)
        self.assertEqual(cp.uploads, [1, 1, 1])
        self.assertEqual(cp.downloads, [1])
        self.assertEqual(cp.current, 0)

    def test_constructor_accepts_selected_device_without_loading_cuda(self):
        face = FaceBeautifier(gpu_index=1)
        self.assertEqual(face._gpu_index, 1)
        self.assertIsNone(face._cupy)

    def test_switch_routes_next_batch_without_resetting_cpu_face_history(self):
        cp = _DeviceCupy()
        face, frame = self._face(cp)
        face._prev_frame = np.full((8, 8, 3), 17, np.uint8)
        face._vignette_cache = np.ones((8, 8), np.float32)
        caches = (face._face_mask, face._tone_mask, face._prev_frame, face._vignette_cache)
        first = face._apply_gpu_batch(frame.copy(), 8, 8)
        cp.uploads.clear()
        cp.downloads.clear()

        face.set_gpu_index(0)
        second = face._apply_gpu_batch(frame.copy(), 8, 8)

        self.assertEqual(cp.uploads, [0, 0, 0])
        self.assertEqual(cp.downloads, [0])
        self.assertEqual(cp.current, 0)
        np.testing.assert_array_equal(first, second)
        for previous, current in zip(caches, (
            face._face_mask, face._tone_mask, face._prev_frame, face._vignette_cache,
        )):
            self.assertIs(previous, current)

    def test_gpu_failure_restores_caller_before_unchanged_cpu_fallback(self):
        for failure in ("enter", "upload", "download"):
            with self.subTest(failure=failure):
                cp = _DeviceCupy(fail=failure)
                face, frame = self._face(cp)
                reference, _frame = self._face(_DeviceCupy())
                expected = reference._apply_enhance(frame.copy())
                expected = reference._apply_sharpen(expected)
                fallback_devices = []
                original = face._apply_enhance

                def cpu_enhance(value):
                    fallback_devices.append(cp.current)
                    return original(value)

                with mock.patch.object(face, "_apply_enhance", side_effect=cpu_enhance):
                    result = face._apply_gpu_batch(frame.copy(), 8, 8)

                self.assertEqual(cp.current, 0)
                self.assertEqual(fallback_devices, [0])
                self.assertEqual(face._compositing, "cpu")
                np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

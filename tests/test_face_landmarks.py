import threading
import time
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nvbroadcast.video.face_landmarks import SharedFaceLandmarker


def _make_landmarker():
    lm = SharedFaceLandmarker.__new__(SharedFaceLandmarker)
    lm._landmarker = object()
    lm._initialized = True
    lm._last_frame_ref = None
    lm._last_result = None
    lm._frames_since_infer = 0
    lm._lock = threading.Lock()
    lm._detect_lock = threading.Lock()
    lm._async_busy = False
    lm._pending_job = None
    lm._pending_frame_ref = None
    lm._worker_event = mock.Mock()
    lm._worker_stop = False
    lm._worker_thread = None
    lm._mp = None
    return lm


class FaceLandmarkerTests(unittest.TestCase):
    def test_scale_detection_frame_caps_large_inputs(self):
        frame = np.zeros((576, 1024, 4), dtype=np.uint8)

        scaled = SharedFaceLandmarker._scale_detection_frame(frame)

        self.assertEqual(scaled.shape[:2], (288, 512))

    def test_request_async_replaces_pending_job_with_newest_frame(self):
        lm = _make_landmarker()
        lm._last_result = [SimpleNamespace(x=0.5, y=0.5)]
        frame_a = np.zeros((100, 120, 4), dtype=np.uint8)
        frame_b = np.zeros((100, 120, 4), dtype=np.uint8)

        lm.request_async(frame_a, reuse_frames=1)
        lm.request_async(frame_b, reuse_frames=1)

        self.assertTrue(lm._async_busy)
        self.assertIs(lm._pending_frame_ref(), frame_b)
        self.assertEqual(lm._pending_job.shape, frame_b.shape)
        self.assertEqual(lm._worker_event.set.call_count, 2)

    def test_detect_returns_cached_result_for_reused_frame_budget(self):
        lm = _make_landmarker()
        cached = [SimpleNamespace(x=0.5, y=0.5)]
        lm._last_result = cached
        frame = np.zeros((100, 120, 4), dtype=np.uint8)

        result = lm.detect(frame, reuse_frames=3)

        self.assertEqual(result, cached)
        self.assertEqual(lm._frames_since_infer, 1)
        self.assertIs(lm._last_frame_ref(), frame)

    def test_detect_runs_detection_when_reuse_budget_exhausted(self):
        lm = _make_landmarker()
        lm._last_result = [SimpleNamespace(x=0.5, y=0.5)]
        lm._frames_since_infer = 2
        frame = np.zeros((100, 120, 4), dtype=np.uint8)
        expected = [SimpleNamespace(x=0.4, y=0.4)]

        with mock.patch.object(lm, "_run_detection", return_value=expected) as run:
            result = lm.detect(frame, reuse_frames=3)

        self.assertEqual(result, expected)
        run.assert_called_once_with(frame)

    def test_detect_deduplicates_only_the_same_live_frame(self):
        lm = _make_landmarker()
        first = np.zeros((16, 16, 4), dtype=np.uint8)
        second = np.ones_like(first)
        expected_first = [SimpleNamespace(x=0.1)]
        expected_second = [SimpleNamespace(x=0.2)]

        # Model numeric ID recycling deterministically, without relying on the
        # allocator used by a particular Python/NumPy build. Distinct live
        # objects must still be recognized as different camera frames.
        with mock.patch("nvbroadcast.video.face_landmarks.id", return_value=42, create=True):
            with mock.patch.object(
                lm, "_run_detection", side_effect=[expected_first, expected_second]
            ) as run:
                self.assertIs(lm.detect(first), expected_first)
                self.assertIs(lm.detect(first), expected_first)
                self.assertIs(lm.detect(second), expected_second)
                self.assertEqual(run.call_count, 2)

    def test_detect_cache_does_not_retain_source_frame(self):
        lm = _make_landmarker()
        frame = np.zeros((16, 16, 4), dtype=np.uint8)
        source_ref = weakref.ref(frame)
        # A lambda avoids Mock's call history retaining the input array.
        lm._run_detection = lambda _frame: [SimpleNamespace(x=0.1)]
        lm.detect(frame)
        del frame

        self.assertIsNone(source_ref())
        self.assertIsNone(lm._last_frame_ref())

    def test_request_async_deduplicates_pending_live_frame(self):
        lm = _make_landmarker()
        frame = np.zeros((16, 16, 4), dtype=np.uint8)
        lm.request_async(frame)
        pending = lm._pending_job
        lm.request_async(frame)

        self.assertIs(lm._pending_job, pending)
        self.assertEqual(lm._worker_event.set.call_count, 1)

    def test_request_async_new_frame_replaces_released_pending_source(self):
        lm = _make_landmarker()
        with mock.patch("nvbroadcast.video.face_landmarks.id", return_value=42, create=True):
            first = np.zeros((16, 16, 4), dtype=np.uint8)
            source_ref = weakref.ref(first)
            lm.request_async(first)
            del first
            self.assertIsNone(source_ref())

            second = np.ones((16, 16, 4), dtype=np.uint8)
            lm.request_async(second)

        self.assertIs(lm._pending_frame_ref(), second)
        np.testing.assert_array_equal(lm._pending_job, second)
        self.assertEqual(lm._worker_event.set.call_count, 2)

    def test_async_worker_refreshes_after_source_frame_is_released(self):
        lm = _make_landmarker()
        lm._worker_event = threading.Event()
        expected = [[SimpleNamespace(x=0.1)], [SimpleNamespace(x=0.2)]]
        detections = []

        def detect(frame):
            value = int(frame[0, 0, 0])
            detections.append(value)
            return expected[value]

        lm._run_detection = detect
        worker = threading.Thread(target=lm._async_loop, daemon=True)
        worker.start()

        def wait_for_result(result):
            deadline = time.monotonic() + 2.0
            while lm.latest() is not result and time.monotonic() < deadline:
                threading.Event().wait(0.001)
            self.assertIs(lm.latest(), result, "landmark worker did not update")

        try:
            with mock.patch("nvbroadcast.video.face_landmarks.id", return_value=42, create=True):
                first = np.zeros((16, 16, 4), dtype=np.uint8)
                source_ref = weakref.ref(first)
                lm.request_async(first)
                wait_for_result(expected[0])
                self.assertIs(lm.request_async(first), expected[0])
                self.assertIsNone(lm._pending_job)
                del first
                self.assertIsNone(source_ref())

                second = np.ones((16, 16, 4), dtype=np.uint8)
                self.assertIs(lm.request_async(second), expected[0])
                wait_for_result(expected[1])
                self.assertIs(lm.request_async(second), expected[1])
                self.assertIsNone(lm._pending_job)
                self.assertEqual(detections, [0, 1])
        finally:
            lm._worker_stop = True
            lm._worker_event.set()
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())

    def test_async_reuse_budget_counts_new_frames_once(self):
        lm = _make_landmarker()
        cached = [SimpleNamespace(x=0.1)]
        lm._last_result = cached
        frames = [np.full((16, 16, 4), value, dtype=np.uint8) for value in range(3)]

        self.assertIs(lm.request_async(frames[0], reuse_frames=3), cached)
        self.assertIs(lm.request_async(frames[0], reuse_frames=3), cached)
        self.assertEqual(lm._frames_since_infer, 1)
        self.assertIs(lm.request_async(frames[1], reuse_frames=3), cached)
        self.assertEqual(lm._frames_since_infer, 2)
        self.assertIsNone(lm._pending_job)
        self.assertIs(lm.request_async(frames[2], reuse_frames=3), cached)
        np.testing.assert_array_equal(lm._pending_job, frames[2])
        self.assertEqual(lm._worker_event.set.call_count, 1)


if __name__ == "__main__":
    unittest.main()

"""Output FIFO checks with generated samples and fake clocks; no audio devices."""

import queue
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from nvbroadcast.audio.pipeline import AudioPipeline, _OutputChunk


SECOND = 1_000_000_000
MSECOND = 1_000_000


class AudioOutputFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = AudioPipeline(use_helper_process=False)

    def test_queue_overflow_retains_newest_without_waiting_for_capacity(self):
        self.pipeline._output_buffer_queue = queue.Queue(maxsize=2)
        audio = np.arange(8, dtype=np.float32)
        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", side_effect=(100, 200, 300)), \
             mock.patch.object(self.pipeline._output_buffer_queue, "put", wraps=self.pipeline._output_buffer_queue.put) as put:
            for _ in range(3):
                self.pipeline._enqueue_output_audio(audio)

        self.assertTrue(all(call.kwargs == {"block": False} for call in put.call_args_list))
        self.assertEqual(self.pipeline._output_buffer_queue.get_nowait().queued_ns, 200)
        self.assertEqual(self.pipeline._output_buffer_queue.get_nowait().queued_ns, 300)

    def test_pass_through_view_is_owned_before_gstreamer_unmaps(self):
        backing = np.arange(16, dtype=np.float32)
        self.pipeline._enqueue_output_audio(backing[::2])
        backing[:] = -1

        chunk = self.pipeline._output_buffer_queue.get_nowait()
        np.testing.assert_array_equal(chunk.audio, np.arange(0, 16, 2, dtype=np.float32))

    def test_queue_clock_starts_at_admission_after_upstream_work(self):
        # Any preceding capture or DSP work is outside this FIFO's budget.
        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", return_value=9 * SECOND):
            self.pipeline._enqueue_output_audio(np.zeros(960, dtype=np.float32))
        chunk = self.pipeline._output_buffer_queue.get_nowait()
        self.assertEqual(chunk.queued_ns, 9 * SECOND)
        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", return_value=9 * SECOND):
            self.assertIs(self.pipeline._fresh_output_audio(chunk), chunk.audio)

    def test_expired_block_drops_whole_without_inferred_sample_timestamps(self):
        audio = np.arange(960, dtype=np.float32)
        chunk = _OutputChunk(audio, queued_ns=SECOND)
        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", return_value=1_207_000_000):
            fresh = self.pipeline._fresh_output_audio(chunk)

        # All samples entered at once, including the final 13 ms of this block.
        self.assertEqual(len(fresh), 0)
        self.assertEqual(chunk.queued_ns, SECOND)

    def test_expiry_is_queue_age_not_sample_rate(self):
        self.pipeline.configure(sample_rate=44100)
        audio = np.arange(4, dtype=np.float32)
        chunk = _OutputChunk(audio, queued_ns=SECOND)
        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", return_value=1_200_000_001):
            fresh = self.pipeline._fresh_output_audio(chunk)

        self.assertEqual(len(fresh), 0)

    def test_fresh_boundary_and_fully_expired_chunk(self):
        audio = np.zeros(960, dtype=np.float32)
        chunk = _OutputChunk(audio, queued_ns=SECOND)
        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", return_value=1_200_000_000):
            self.assertIs(self.pipeline._fresh_output_audio(chunk), audio)
        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", return_value=1_220_000_000):
            self.assertEqual(len(self.pipeline._fresh_output_audio(chunk)), 0)

    def test_healthy_stream_preserves_every_sample_in_fifo_order(self):
        clock = SimpleNamespace(now=SECOND)
        expected = [np.arange(960, dtype=np.float32) + i * 960 for i in range(40)]
        written = []

        def sink(audio):
            written.append(audio.copy())
            clock.now += 20 * MSECOND
            if len(written) < len(expected):
                self.pipeline._enqueue_output_audio(expected[len(written)])
            else:
                self.pipeline._output_buffer_queue.put_nowait(None)
            return "fake-sink"

        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", side_effect=lambda: clock.now), \
             mock.patch.object(self.pipeline, "_push_buffer_direct", side_effect=sink):
            self.pipeline._enqueue_output_audio(expected[0])
            self.pipeline._output_worker_main()

        self.assertEqual(len(written), len(expected))
        for actual, original in zip(written, expected):
            np.testing.assert_array_equal(actual, original)

    def test_worker_recovers_queue_residence_after_550ms_sink_stall(self):
        clock = SimpleNamespace(now=SECOND)
        written = []

        def sink(audio):
            tag = int(audio[0])
            written.append((tag, len(audio), clock.now))
            if tag == 0:
                # New output arrives while an already-started write stalls.
                for i in range(1, 28):
                    clock.now = SECOND + i * 20 * MSECOND
                    self.pipeline._enqueue_output_audio(np.full(960, i, dtype=np.float32))
                clock.now = SECOND + 550 * MSECOND
                self.pipeline._output_buffer_queue.put_nowait(None)
            else:
                clock.now += len(audio) * SECOND // 48000
            return "fake-sink"

        with mock.patch("nvbroadcast.audio.pipeline.time.monotonic_ns", side_effect=lambda: clock.now), \
             mock.patch.object(self.pipeline, "_push_buffer_direct", side_effect=sink):
            self.pipeline._enqueue_output_audio(np.zeros(960, dtype=np.float32))
            self.pipeline._output_worker_main()

        self.assertEqual([tag for tag, _, _ in written], [0, *range(18, 28)])
        for tag, frames, now in written[1:]:
            self.assertEqual(frames, 960)
            queued_ns = SECOND + tag * 20 * MSECOND
            self.assertLessEqual(now - queued_ns, self.pipeline._output_queue_max_age_ns)

    def test_stopped_producer_cannot_replace_stop_sentinel(self):
        self.pipeline._output_buffer_queue = queue.Queue(maxsize=1)
        self.pipeline._stop_output_worker.set()
        self.pipeline._output_buffer_queue.put_nowait(None)

        self.assertEqual(self.pipeline._enqueue_output_audio(np.zeros(960, dtype=np.float32)), "stopped")
        self.assertIsNone(self.pipeline._output_buffer_queue.get_nowait())

    def test_stop_wins_against_a_producer_waiting_for_full_queue_lock(self):
        self.pipeline._output_buffer_queue = queue.Queue(maxsize=1)
        self.pipeline._enqueue_output_audio(np.zeros(960, dtype=np.float32))
        started = threading.Event()
        statuses, errors = [], []
        self.pipeline._output_worker = mock.Mock()

        def producer():
            started.set()
            try:
                statuses.append(self.pipeline._enqueue_output_audio(np.ones(960, dtype=np.float32)))
            except BaseException as error:
                errors.append(error)

        def join_worker(*, timeout):
            self.assertIsNone(self.pipeline._output_buffer_queue.get_nowait())

        def stopper():
            try:
                self.pipeline._stop_output_worker_sync()
            except BaseException as error:
                errors.append(error)

        self.pipeline._output_worker.join.side_effect = join_worker
        with self.pipeline._output_queue_lock:
            producer_thread = threading.Thread(target=producer)
            producer_thread.start()
            self.assertTrue(started.wait(1))
            stop_thread = threading.Thread(target=stopper)
            stop_thread.start()
            self.assertTrue(self.pipeline._stop_output_worker.wait(1))

        producer_thread.join(1)
        stop_thread.join(1)
        self.assertFalse(producer_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(statuses, ["stopped"])
        self.assertTrue(self.pipeline._output_buffer_queue.empty())
        self.assertIsNone(self.pipeline._output_worker)

    def test_stop_discards_pending_audio_and_joins_waiting_worker(self):
        self.pipeline._start_output_worker()
        worker = self.pipeline._output_worker
        self.pipeline._stop_output_worker_sync()

        self.assertFalse(worker.is_alive())
        self.assertTrue(self.pipeline._output_buffer_queue.empty())
        self.assertIsNone(self.pipeline._output_worker)

    def test_stop_replaces_full_queue_with_sentinel_before_join(self):
        self.pipeline._output_buffer_queue = queue.Queue(maxsize=1)
        self.pipeline._enqueue_output_audio(np.zeros(960, dtype=np.float32))
        worker = mock.Mock()
        self.pipeline._output_worker = worker

        def join(*, timeout):
            self.assertTrue(self.pipeline._stop_output_worker.is_set())
            self.assertIsNone(self.pipeline._output_buffer_queue.get_nowait())

        worker.join.side_effect = join
        self.pipeline._stop_output_worker_sync()

        worker.join.assert_called_once_with(timeout=2.0)
        self.assertTrue(self.pipeline._output_buffer_queue.empty())
        self.assertIsNone(self.pipeline._output_worker)


if __name__ == "__main__":
    unittest.main()

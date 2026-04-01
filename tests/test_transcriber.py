import unittest
from unittest import mock

import numpy as np

from nvbroadcast.ai.transcriber import MeetingTranscriber


class MeetingTranscriberTests(unittest.TestCase):
    def test_start_returns_false_when_initialize_fails(self):
        transcriber = MeetingTranscriber("base")
        with mock.patch.object(transcriber, "initialize", return_value=False):
            self.assertFalse(transcriber.start())
        self.assertFalse(transcriber.recording)

    def test_start_returns_true_when_initialized(self):
        transcriber = MeetingTranscriber("base")
        transcriber._initialized = True
        self.assertTrue(transcriber.start())
        self.assertTrue(transcriber.recording)

    def test_feed_audio_sanitizes_and_clips(self):
        transcriber = MeetingTranscriber("base")
        transcriber._recording = True

        transcriber.feed_audio(
            np.array([np.nan, 2.0, -2.0, 0.5], dtype=np.float32),
            sample_rate=16000,
        )

        self.assertEqual(len(transcriber._audio_buffer), 1)
        buffered = transcriber._audio_buffer[0]
        self.assertTrue(np.isfinite(buffered).all())
        self.assertLessEqual(float(buffered.max()), 1.0)
        self.assertGreaterEqual(float(buffered.min()), -1.0)

    def test_prepare_audio_removes_dc_and_normalizes_rms(self):
        transcriber = MeetingTranscriber("base")
        audio = np.full(1600, 0.02, dtype=np.float32)
        prepared = transcriber._prepare_audio(audio, sample_rate=16000)
        self.assertLess(abs(float(prepared.mean())), 1e-3)
        self.assertTrue(np.isfinite(prepared).all())

    def test_store_segment_replaces_shorter_overlap(self):
        transcriber = MeetingTranscriber("base")
        first = {
            "text": "hello",
            "start_time": 0.0,
            "end_time": 1.0,
            "confidence": -0.2,
        }
        second = {
            "text": "hello world",
            "start_time": 0.4,
            "end_time": 1.4,
            "confidence": -0.1,
        }

        self.assertTrue(transcriber._store_segment(type("Seg", (), first)()))
        self.assertFalse(transcriber._store_segment(type("Seg", (), second)()))
        self.assertEqual(len(transcriber.segments), 1)
        self.assertEqual(transcriber.segments[0].text, "hello world")

    def test_on_future_done_appends_segments_and_emits_callback(self):
        transcriber = MeetingTranscriber("base")
        callback = mock.Mock()
        transcriber.set_segment_callback(callback)
        future = mock.Mock()
        future.result.return_value = [
            {
                "text": "hello world",
                "start_time": 1.0,
                "end_time": 2.0,
                "confidence": -0.1,
            }
        ]

        transcriber._on_future_done(future)

        self.assertEqual(len(transcriber.segments), 1)
        self.assertEqual(transcriber.segments[0].text, "hello world")
        callback.assert_called_once()

    def test_on_future_done_skips_duplicate_overlap(self):
        transcriber = MeetingTranscriber("base")
        callback = mock.Mock()
        transcriber.set_segment_callback(callback)
        first = mock.Mock()
        first.result.return_value = [
            {
                "text": "hello world",
                "start_time": 0.0,
                "end_time": 1.0,
                "confidence": -0.1,
            }
        ]
        second = mock.Mock()
        second.result.return_value = [
            {
                "text": "hello world",
                "start_time": 0.3,
                "end_time": 1.2,
                "confidence": -0.1,
            }
        ]

        transcriber._on_future_done(first)
        transcriber._on_future_done(second)

        self.assertEqual(len(transcriber.segments), 1)
        callback.assert_called_once()

    def test_select_final_model_prefers_medium_for_short_meetings(self):
        transcriber = MeetingTranscriber("base", final_model_size="small")
        with mock.patch.object(transcriber, "_estimate_audio_duration", return_value=300.0):
            self.assertEqual(transcriber._select_final_model("/tmp/fake.wav"), "medium")

    def test_select_final_model_keeps_default_for_long_meetings(self):
        transcriber = MeetingTranscriber("base", final_model_size="small")
        with mock.patch.object(transcriber, "_estimate_audio_duration", return_value=4000.0):
            self.assertEqual(transcriber._select_final_model("/tmp/fake.wav"), "small")


if __name__ == "__main__":
    unittest.main()

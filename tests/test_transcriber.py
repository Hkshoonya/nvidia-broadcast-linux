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


if __name__ == "__main__":
    unittest.main()

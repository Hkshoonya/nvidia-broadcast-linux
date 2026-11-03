import unittest
import wave
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np

from nvbroadcast.ai.transcriber import (
    MeetingTranscriber,
    _backend_candidates,
    _has_supported_backend,
    _missing_backend_help,
    _normalize_backend_preference,
)
from nvbroadcast.core.platform import supports_openai_whisper_python


class MeetingTranscriberTests(unittest.TestCase):
    def test_backend_preference_normalization(self):
        self.assertEqual(_normalize_backend_preference("whisper"), "whisper")
        self.assertEqual(_normalize_backend_preference("faster-whisper"), "faster-whisper")
        self.assertEqual(_normalize_backend_preference("bogus"), "auto")

    def test_backend_candidates_respect_preference(self):
        self.assertEqual(_backend_candidates("whisper"), ["whisper"])
        self.assertEqual(_backend_candidates("faster-whisper"), ["faster-whisper"])
        with mock.patch("nvbroadcast.ai.transcriber.supports_openai_whisper_python", return_value=True):
            self.assertEqual(_backend_candidates("auto"), ["faster-whisper", "whisper"])

    def test_openai_whisper_backend_disabled_on_python_314(self):
        with mock.patch("importlib.metadata.version", side_effect=Exception("missing")):
            self.assertFalse(supports_openai_whisper_python((3, 14)))

    def test_has_supported_backend_ignores_whisper_on_python_314(self):
        def fake_find_spec(name):
            if name == "faster_whisper":
                return None
            if name == "whisper":
                return object()
            raise AssertionError(f"Unexpected spec lookup: {name}")

        with mock.patch("nvbroadcast.ai.transcriber.importlib.util.find_spec", side_effect=fake_find_spec), \
             mock.patch("nvbroadcast.ai.transcriber.supports_openai_whisper_python", return_value=False):
            self.assertFalse(_has_supported_backend())

    def test_has_supported_backend_allows_whisper_preference_on_supported_python(self):
        def fake_find_spec(name):
            if name == "faster_whisper":
                return None
            if name == "whisper":
                return object()
            raise AssertionError(f"Unexpected spec lookup: {name}")

        with mock.patch("nvbroadcast.ai.transcriber.importlib.util.find_spec", side_effect=fake_find_spec):
            self.assertTrue(_has_supported_backend("whisper"))

    def test_has_supported_backend_allows_explicit_whisper_even_when_auto_fallback_disabled(self):
        def fake_find_spec(name):
            if name == "faster_whisper":
                return None
            if name == "whisper":
                return object()
            raise AssertionError(f"Unexpected spec lookup: {name}")

        with mock.patch("nvbroadcast.ai.transcriber.importlib.util.find_spec", side_effect=fake_find_spec), \
             mock.patch("nvbroadcast.ai.transcriber.supports_openai_whisper_python", return_value=False):
            self.assertTrue(_has_supported_backend("whisper"))

    def test_initialize_returns_false_without_supported_backend(self):
        transcriber = MeetingTranscriber("base")
        with mock.patch("nvbroadcast.ai.transcriber._has_supported_backend", return_value=False):
            self.assertFalse(transcriber.initialize())

    def test_initialize_stops_worker_after_import_error(self):
        transcriber = MeetingTranscriber("base")
        executor = mock.Mock()
        executor.submit.return_value.result.side_effect = ImportError("missing runtime")
        with mock.patch("nvbroadcast.ai.transcriber._has_supported_backend", return_value=True), \
             mock.patch("nvbroadcast.ai.transcriber.ProcessPoolExecutor", return_value=executor):
            self.assertFalse(transcriber.initialize())
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        self.assertIsNone(transcriber._executor)

    def test_worker_exit_poll_handles_early_join_return(self):
        from nvbroadcast.ai.transcriber import _wait_for_worker_exit

        worker = mock.Mock()
        states = iter([True, True, True])
        worker.is_alive.side_effect = lambda: next(states, False)
        with mock.patch("nvbroadcast.ai.transcriber.time.sleep"):
            self.assertTrue(_wait_for_worker_exit(worker, 1))
        self.assertEqual(worker.join.call_count, 2)

    def test_missing_backend_help_mentions_optional_whisper(self):
        self.assertIn("openai-whisper", _missing_backend_help("whisper"))

    def test_auto_missing_backend_help_uses_complete_faster_whisper_recipe(self):
        help_text = _missing_backend_help("auto")
        self.assertIn("pip install --no-deps faster-whisper", help_text)
        self.assertIn("pip install ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", help_text)

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

    def test_load_audio_source_reads_wav_without_external_decoder(self):
        transcriber = MeetingTranscriber("base")
        samples = np.array([0, 16384, -16384, 8192], dtype=np.int16)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "meeting.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(samples.tobytes())

            source = transcriber._load_audio_source(str(wav_path))

        self.assertIsInstance(source, np.ndarray)
        self.assertEqual(source.dtype, np.float32)
        self.assertEqual(source.shape[0], samples.shape[0])
        self.assertTrue(np.isfinite(source).all())


if __name__ == "__main__":
    unittest.main()

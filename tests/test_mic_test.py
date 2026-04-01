import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nvbroadcast.audio.mic_test import MicTest


class _DummyBus:
    def timed_pop_filtered(self, *_args, **_kwargs):
        return None

    def add_signal_watch(self):
        return None

    def connect(self, *_args, **_kwargs):
        return None


class _DummyPipeline:
    def __init__(self):
        self.state_calls = []
        self.events = []
        self.bus = _DummyBus()

    def set_state(self, state):
        self.state_calls.append(state)

    def send_event(self, event):
        self.events.append(event)

    def get_bus(self):
        return self.bus


class MicTestTests(unittest.TestCase):
    @mock.patch("nvbroadcast.audio.mic_test.Gst.parse_launch")
    @mock.patch("nvbroadcast.audio.mic_test.Gst.ElementFactory.find")
    def test_play_recording_uses_selected_speaker_target(self, element_find, parse_launch):
        element_find.return_value = object()
        pipeline = _DummyPipeline()
        parse_launch.return_value = pipeline

        mic_test = MicTest()
        with tempfile.TemporaryDirectory() as tmp:
            mic_test._test_file = str(Path(tmp) / "mic.wav")
            Path(mic_test._test_file).write_bytes(b"RIFFfake")
            mic_test.play_recording(speaker_device="speaker0")

        self.assertTrue(parse_launch.called)
        pipeline_desc = parse_launch.call_args.args[0]
        self.assertIn("pulsesink device=speaker0", pipeline_desc)
        self.assertTrue(mic_test.is_playing)

    @mock.patch("nvbroadcast.audio.mic_test.Gst.parse_launch")
    def test_start_recording_uses_requested_duration(self, parse_launch):
        pipeline = _DummyPipeline()
        parse_launch.return_value = pipeline

        mic_test = MicTest()
        with mock.patch("threading.Thread") as thread_cls:
            thread_cls.return_value.start.return_value = None
            mic_test.start_recording("mic0", duration=45)

        self.assertEqual(mic_test._duration, 45)
        self.assertTrue(mic_test.is_recording)
        self.assertIn("pipewiresrc", parse_launch.call_args.args[0])


if __name__ == "__main__":
    unittest.main()

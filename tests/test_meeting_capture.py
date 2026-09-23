"""Meeting WAV source selection and failed-native-source cleanup."""

import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from nvbroadcast.audio.meeting_capture import (
    MeetingAudioCapture,
    has_recorded_meeting_audio,
)


class MeetingAudioCaptureTests(unittest.TestCase):
    def test_only_a_wav_with_pcm_frames_counts_as_recorded_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.wav")
            empty = str(Path(directory) / "empty.wav")
            header_only = str(Path(directory) / "header.wav")
            valid = str(Path(directory) / "valid.wav")
            Path(empty).touch()
            for path, frames in ((header_only, b""), (valid, b"\0" * 320)):
                with wave.open(path, "wb") as recording:
                    recording.setnchannels(1)
                    recording.setsampwidth(2)
                    recording.setframerate(16000)
                    recording.writeframes(frames)
            self.assertFalse(has_recorded_meeting_audio(missing))
            self.assertFalse(has_recorded_meeting_audio(empty))
            self.assertFalse(has_recorded_meeting_audio(header_only))
            self.assertTrue(has_recorded_meeting_audio(valid))

    def test_build_uses_pulse_names_for_mic_and_speaker_monitor(self):
        capture = MeetingAudioCapture()
        if Gst.ElementFactory.find("pulsesrc") is None:
            self.skipTest("pulsesrc unavailable")
        with mock.patch(
            "nvbroadcast.audio.meeting_capture.probe_audio_source",
            return_value=("pulsesrc", ""),
        ), mock.patch(
            "nvbroadcast.audio.meeting_capture.resolve_pulse_source_name",
            return_value="alsa_input.demo",
        ), mock.patch(
            "nvbroadcast.audio.meeting_capture.resolve_speaker_monitor_name",
            return_value="alsa_output.demo.monitor",
        ):
            capture.build("59", "alsa_output.demo", "/tmp/meeting audio.wav")
        try:
            self.assertEqual(
                capture._pipeline.get_by_name("mic-src").get_property("device"),
                "alsa_input.demo",
            )
            self.assertEqual(
                capture._pipeline.get_by_name("speaker-src").get_property("device"),
                "alsa_output.demo.monitor",
            )
            self.assertEqual(
                capture._pipeline.get_by_name("meeting-file").get_property("location"),
                "/tmp/meeting audio.wav",
            )
        finally:
            capture.stop()

    def test_unavailable_source_fails_before_creating_a_pipeline(self):
        capture = MeetingAudioCapture()
        with mock.patch(
            "nvbroadcast.audio.meeting_capture.probe_audio_source",
            return_value=(None, "PipeWire socket is unavailable"),
        ), self.assertRaisesRegex(RuntimeError, "No usable meeting audio source"):
            capture.build("", "", "/tmp/meeting.wav")
        self.assertIsNone(capture._pipeline)
        self.assertFalse(capture.running)

    def test_stale_numeric_microphone_uses_probed_pulse_default(self):
        capture = MeetingAudioCapture()
        if Gst.ElementFactory.find("pulsesrc") is None:
            self.skipTest("pulsesrc unavailable")
        with mock.patch(
            "nvbroadcast.audio.meeting_capture.probe_audio_source",
            return_value=("pulsesrc", ""),
        ), mock.patch(
            "nvbroadcast.audio.meeting_capture.resolve_pulse_source_name",
            return_value="",
        ), mock.patch(
            "nvbroadcast.audio.meeting_capture.resolve_speaker_monitor_name",
            return_value="",
        ):
            capture.build("123", "", "/tmp/meeting.wav")
        try:
            self.assertFalse(
                capture._pipeline.get_by_name("mic-src").get_property("device")
            )
            self.assertIn("saved microphone unavailable", capture.route_warning)
        finally:
            capture.stop()

    def test_stale_numeric_speaker_omits_monitor_but_keeps_microphone(self):
        capture = MeetingAudioCapture()
        if Gst.ElementFactory.find("pulsesrc") is None:
            self.skipTest("pulsesrc unavailable")
        with mock.patch(
            "nvbroadcast.audio.meeting_capture.probe_audio_source",
            return_value=("pulsesrc", ""),
        ), mock.patch(
            "nvbroadcast.audio.meeting_capture.resolve_pulse_source_name",
            return_value="alsa_input.demo",
        ), mock.patch(
            "nvbroadcast.audio.meeting_capture.resolve_speaker_monitor_name",
            return_value="",
        ):
            capture.build("59", "123", "/tmp/meeting.wav")
        try:
            self.assertIsNotNone(capture._pipeline.get_by_name("mic-src"))
            self.assertIsNone(capture._pipeline.get_by_name("speaker-src"))
            self.assertIn("WAV captures microphone only", capture.route_warning)
        finally:
            capture.stop()

    def test_error_clears_running_and_stop_skips_eos(self):
        capture = MeetingAudioCapture()
        pipeline = mock.Mock()
        capture._pipeline = pipeline
        capture._pipeline.get_state.return_value = (
            Gst.StateChangeReturn.SUCCESS, Gst.State.NULL, Gst.State.VOID_PENDING
        )
        capture._bus = mock.Mock()
        capture._running = True
        on_error = mock.Mock()
        capture.set_error_callback(on_error)
        msg = mock.Mock()
        msg.parse_error.return_value = (
            SimpleNamespace(message="can't connect"), "missing socket"
        )

        capture._on_error(capture._bus, msg)
        self.assertFalse(capture.running)
        self.assertEqual(capture.last_error, "can't connect")
        capture.stop()

        pipeline.send_event.assert_not_called()
        on_error.assert_called_once_with("can't connect")
        self.assertIsNone(capture._bus)
        self.assertIsNone(capture._pipeline)

    def test_stop_disposes_zero_byte_wav_after_failed_start(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "meeting audio.wav"
            output.touch()
            capture = MeetingAudioCapture()
            capture._output_path = str(output)
            capture._pipeline = mock.Mock()
            capture._pipeline.get_state.return_value = (
                Gst.StateChangeReturn.FAILURE, Gst.State.NULL,
                Gst.State.VOID_PENDING,
            )
            capture.stop()
            self.assertFalse(output.exists())

    def test_queued_error_prevents_eos_even_if_state_is_playing(self):
        capture = MeetingAudioCapture()
        pipeline = mock.Mock()
        pipeline.get_state.return_value = (
            Gst.StateChangeReturn.SUCCESS, Gst.State.PLAYING,
            Gst.State.VOID_PENDING,
        )
        bus = mock.Mock()
        error = mock.Mock()
        error.parse_error.return_value = (
            SimpleNamespace(message="source disconnected"), "debug"
        )
        bus.timed_pop_filtered.return_value = error
        capture._pipeline = pipeline
        capture._bus = bus
        capture._running = True

        capture.stop()

        pipeline.send_event.assert_not_called()
        self.assertEqual(capture.last_error, "source disconnected")
        self.assertFalse(capture.running)

    def test_real_failed_playing_state_does_not_send_eos_in_bounded_child(self):
        script = """
from unittest import mock
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from nvbroadcast.audio.meeting_capture import MeetingAudioCapture
Gst.init(None)
pipeline = Gst.parse_launch(
    'audiotestsrc is-live=true ! identity error-after=3 ! fakesink')
bus = pipeline.get_bus()
bus.add_signal_watch()
pipeline.set_state(Gst.State.PLAYING)
error = bus.timed_pop_filtered(3 * Gst.SECOND, Gst.MessageType.ERROR)
if error is None:
    sys.exit(77)
state_return, state, _pending = pipeline.get_state(0)
if state_return != Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
    sys.exit(77)
capture = MeetingAudioCapture()
capture._pipeline = mock.Mock(wraps=pipeline)
capture._pipeline.send_event.side_effect = AssertionError('EOS after ERROR')
capture._bus = bus
capture._running = True
capture.stop()
print('failed native state stopped without EOS', flush=True)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            env=env, timeout=10, check=False,
        )
        if result.returncode == 77:
            self.skipTest("GStreamer did not reach the PLAYING/FAILURE race")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("without EOS", result.stdout)

    def test_failed_pipewire_source_stop_is_bounded_child_process(self):
        MeetingAudioCapture()
        if Gst.ElementFactory.find("pipewiresrc") is None:
            self.skipTest("pipewiresrc unavailable")
        script = """
import os, sys, tempfile
from unittest import mock
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from nvbroadcast.audio.meeting_capture import MeetingAudioCapture
with tempfile.TemporaryDirectory() as directory:
    capture = MeetingAudioCapture()
    with mock.patch('nvbroadcast.audio.meeting_capture.probe_audio_source',
                    return_value=('pipewiresrc', '')), \\
         mock.patch('nvbroadcast.audio.meeting_capture.resolve_speaker_monitor',
                    return_value=''):
        capture.build('', '', os.path.join(directory, 'meeting.wav'))
    try:
        capture.start()
        started = True
    except RuntimeError:
        started = False
    msg = capture._bus.timed_pop_filtered(3 * Gst.SECOND, Gst.MessageType.ERROR)
    if started and msg is None:
        sys.exit(77)
    capture.stop()
    print('failed source stopped safely', flush=True)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        env["PIPEWIRE_REMOTE"] = "nvb-issue112-nonexistent-remote"
        with tempfile.TemporaryDirectory() as runtime_dir:
            env["PIPEWIRE_RUNTIME_DIR"] = runtime_dir
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True,
                env=env, timeout=10, check=False,
            )
        if result.returncode == 77:
            self.skipTest("PipeWire plugin did not report its missing remote")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("failed source stopped safely", result.stdout)

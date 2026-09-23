"""Meeting WAV source selection and failed-native-source cleanup."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from nvbroadcast.audio.meeting_capture import MeetingAudioCapture


class MeetingAudioCaptureTests(unittest.TestCase):
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

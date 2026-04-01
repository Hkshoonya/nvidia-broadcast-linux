# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Microphone test — record and playback for testing audio setup.

Records a short clip from the selected mic, applies voice effects,
and plays it back so the user can hear their processed voice.
"""

import threading
import tempfile
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from nvbroadcast.audio.devices import resolve_pipewire_target

Gst.init(None)


class MicTest:
    """Record and playback mic audio for testing."""

    def __init__(self):
        self._recording = False
        self._playing = False
        self._rec_pipeline = None
        self._play_pipeline = None
        self._test_file = str(Path(tempfile.gettempdir()) / "nvbroadcast_mic_test.wav")
        self._duration = 30  # seconds
        self._on_complete = None
        self._record_token = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_playing(self) -> bool:
        return self._playing

    def start_recording(self, mic_device: str = "", duration: int = 30,
                       on_complete=None):
        """Record from mic for `duration` seconds.

        Args:
            mic_device: PipeWire device ID or empty for default
            duration: seconds to record
            on_complete: callback when recording finishes
        """
        if self._recording or self._playing:
            return

        self._duration = duration
        self._on_complete = on_complete
        self._recording = True
        self._record_token += 1
        token = self._record_token

        # Prefer Pulse on desktop Linux for reliable timed recording.
        src = "pulsesrc"
        if mic_device and Gst.ElementFactory.find("pulsesrc") is not None:
            src = f"pulsesrc device={mic_device}"
        elif mic_device:
            target = resolve_pipewire_target(mic_device)
            src = f"pipewiresrc do-timestamp=true target-object={target}"

        try:
            self._rec_pipeline = Gst.parse_launch(
                f"{src} ! audioconvert ! audioresample ! "
                f"audio/x-raw,format=S16LE,rate=48000,channels=1 ! "
                f"wavenc ! filesink location={self._test_file}"
            )
            self._rec_pipeline.set_state(Gst.State.PLAYING)
            print(f"[Mic Test] Recording for {duration}s...")

            # Stop after duration
            def _stop():
                import time
                time.sleep(duration)
                if self._recording and token == self._record_token:
                    self._stop_recording()

            threading.Thread(target=_stop, daemon=True).start()

        except Exception as e:
            print(f"[Mic Test] Recording failed: {e}")
            self._recording = False

    def stop_recording(self):
        """Stop an active recording early and finalize the WAV file."""
        self._stop_recording()

    def _stop_recording(self):
        if self._rec_pipeline:
            bus = self._rec_pipeline.get_bus()
            try:
                self._rec_pipeline.send_event(Gst.Event.new_eos())
                bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            finally:
                self._rec_pipeline.set_state(Gst.State.NULL)
                self._rec_pipeline = None
        self._recording = False
        print("[Mic Test] Recording complete")
        if self._on_complete:
            from gi.repository import GLib
            GLib.idle_add(self._on_complete)

    def play_recording(self, speaker_device: str = "", on_complete=None):
        """Play back the test recording."""
        if self._recording or self._playing:
            return
        if not Path(self._test_file).exists():
            print("[Mic Test] No recording to play")
            return

        self._playing = True
        self._on_complete = on_complete

        try:
            sink = "autoaudiosink sync=false"
            if speaker_device:
                pulse_sink = Gst.ElementFactory.find("pulsesink") is not None
                if pulse_sink:
                    sink = f"pulsesink device={speaker_device} sync=false"
                else:
                    target = resolve_pipewire_target(speaker_device)
                    sink = f"pipewiresink target-object={target} sync=false"
            self._play_pipeline = Gst.parse_launch(
                f"filesrc location={self._test_file} ! "
                f"wavparse ! audioconvert ! audioresample ! "
                f"audio/x-raw,rate=48000,channels=1 ! "
                f"{sink}"
            )
            bus = self._play_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message::eos", self._on_playback_eos)
            bus.connect("message::error", self._on_playback_error)
            self._play_pipeline.set_state(Gst.State.PLAYING)
            print("[Mic Test] Playing back...")
        except Exception as e:
            print(f"[Mic Test] Playback failed: {e}")
            self._playing = False

    def _on_playback_eos(self, bus, msg):
        self._play_pipeline.set_state(Gst.State.NULL)
        self._play_pipeline = None
        self._playing = False
        print("[Mic Test] Playback complete")
        if self._on_complete:
            from gi.repository import GLib
            GLib.idle_add(self._on_complete)

    def _on_playback_error(self, bus, msg):
        err, _ = msg.parse_error()
        print(f"[Mic Test] Playback error: {err.message}")
        if self._play_pipeline:
            self._play_pipeline.set_state(Gst.State.NULL)
            self._play_pipeline = None
        self._playing = False
        if self._on_complete:
            from gi.repository import GLib
            GLib.idle_add(self._on_complete)

    def stop(self):
        """Stop any recording or playback."""
        if self._rec_pipeline:
            self._rec_pipeline.set_state(Gst.State.NULL)
            self._rec_pipeline = None
        if self._play_pipeline:
            self._play_pipeline.set_state(Gst.State.NULL)
            self._play_pipeline = None
        self._recording = False
        self._playing = False

    def cleanup(self):
        """Remove temp file."""
        self.stop()
        try:
            Path(self._test_file).unlink(missing_ok=True)
        except Exception:
            pass

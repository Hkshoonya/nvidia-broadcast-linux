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
import time
import wave
import tempfile
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)


class MicTest:
    """Record and playback mic audio for testing."""

    def __init__(self):
        self._recording = False
        self._playing = False
        self._rec_pipeline = None
        self._play_pipeline = None
        self._test_file = str(Path(tempfile.gettempdir()) / "nvbroadcast_mic_test.wav")
        self._duration = 5  # seconds
        self._on_complete = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_playing(self) -> bool:
        return self._playing

    def start_recording(self, mic_device: str = "", duration: int = 5,
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

        # Build recording pipeline
        src = "pipewiresrc"
        if mic_device:
            src = f"pipewiresrc target-object={mic_device}"

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
                time.sleep(duration)
                self._stop_recording()

            threading.Thread(target=_stop, daemon=True).start()

        except Exception as e:
            print(f"[Mic Test] Recording failed: {e}")
            self._recording = False

    def _stop_recording(self):
        if self._rec_pipeline:
            self._rec_pipeline.send_event(Gst.Event.new_eos())
            time.sleep(0.5)
            self._rec_pipeline.set_state(Gst.State.NULL)
            self._rec_pipeline = None
        self._recording = False
        print("[Mic Test] Recording complete")
        if self._on_complete:
            from gi.repository import GLib
            GLib.idle_add(self._on_complete)

    def play_recording(self, on_complete=None):
        """Play back the test recording."""
        if self._recording or self._playing:
            return
        if not Path(self._test_file).exists():
            print("[Mic Test] No recording to play")
            return

        self._playing = True
        self._on_complete = on_complete

        try:
            self._play_pipeline = Gst.parse_launch(
                f"filesrc location={self._test_file} ! "
                f"wavparse ! audioconvert ! audioresample ! "
                f"pipewiresink"
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
        self._playing = False

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

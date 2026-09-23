# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""Meeting audio capture for mixed mic + speaker notes/transcription."""

from __future__ import annotations

from pathlib import Path
import wave

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np

from nvbroadcast.audio.devices import (
    resolve_pipewire_target,
    resolve_pulse_source_name,
    resolve_speaker_monitor,
    resolve_speaker_monitor_name,
)
from nvbroadcast.audio.source_probe import probe_audio_source


def has_recorded_meeting_audio(path: str) -> bool:
    """Require a readable WAV with at least one recorded PCM frame."""
    if not path:
        return False
    try:
        with wave.open(path, "rb") as recorded:
            return recorded.getnframes() > 0 and bool(recorded.readframes(1))
    except (OSError, EOFError, wave.Error):
        return False


class MeetingAudioCapture:
    """Capture meeting audio, mix both directions, and expose live PCM."""

    def __init__(self):
        Gst.init(None)
        self._pipeline: Gst.Pipeline | None = None
        self._sample_rate = 16000
        self._channels = 1
        self._sample_callback = None
        self._running = False
        self._bus = None
        self._source_backend = ""
        self._last_error = ""
        self._error_callback = None
        self._output_path = ""
        self._route_warning = ""

    def set_sample_callback(self, callback):
        self._sample_callback = callback

    def set_error_callback(self, callback):
        self._error_callback = callback

    def build(self, mic_device: str, speaker_device: str, output_path: str):
        source_backend, source_error = probe_audio_source()
        if source_backend is None:
            raise RuntimeError(f"No usable meeting audio source: {source_error}")
        self._source_backend = source_backend
        self._last_error = ""
        self._output_path = output_path
        self._route_warning = ""
        self._pipeline = Gst.Pipeline.new("nvbroadcast-meeting-capture")
        if source_backend == "pulsesrc":
            mic_target = resolve_pulse_source_name(mic_device)
            speaker_target = resolve_speaker_monitor_name(speaker_device)
            warnings = []
            if mic_device.isdigit() and not mic_target:
                warnings.append("saved microphone unavailable; using default")
            if speaker_device.isdigit() and not speaker_target:
                warnings.append("saved speaker unavailable; WAV captures microphone only")
            self._route_warning = "; ".join(warnings)
            if self._route_warning:
                print(f"[NV Broadcast Meeting] {self._route_warning}")
        else:
            mic_target = resolve_pipewire_target(mic_device)
            speaker_target = resolve_speaker_monitor(speaker_device)

        mixer = Gst.ElementFactory.make("audiomixer", "meeting-mixer")
        tee = Gst.ElementFactory.make("tee", "meeting-tee")
        file_queue = Gst.ElementFactory.make("queue", "file-queue")
        file_convert = Gst.ElementFactory.make("audioconvert", "file-convert")
        file_resample = Gst.ElementFactory.make("audioresample", "file-resample")
        file_caps = Gst.ElementFactory.make("capsfilter", "file-caps")
        file_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=S16LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )
        live_queue = Gst.ElementFactory.make("queue", "live-queue")
        wavenc = Gst.ElementFactory.make("wavenc", "meeting-wav")
        filesink = Gst.ElementFactory.make("filesink", "meeting-file")
        filesink.set_property("location", output_path)

        live_convert = Gst.ElementFactory.make("audioconvert", "live-convert")
        live_resample = Gst.ElementFactory.make("audioresample", "live-resample")
        live_caps = Gst.ElementFactory.make("capsfilter", "live-caps")
        live_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )
        appsink = Gst.ElementFactory.make("appsink", "meeting-live")
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 8)
        appsink.set_property("drop", True)
        appsink.connect("new-sample", self._on_new_sample)

        elements = [
            mixer, tee, file_queue, file_convert, file_resample, file_caps,
            live_queue, wavenc, filesink, live_convert, live_resample,
            live_caps, appsink,
        ]
        for element in elements:
            self._pipeline.add(element)

        self._add_source_branch("mic", mic_target, mixer)
        if speaker_target:
            self._add_source_branch("speaker", speaker_target, mixer)

        mixer.link(tee)
        tee.link(file_queue)
        file_queue.link(file_convert)
        file_convert.link(file_resample)
        file_resample.link(file_caps)
        file_caps.link(wavenc)
        wavenc.link(filesink)
        tee.link(live_queue)
        live_queue.link(live_convert)
        live_convert.link(live_resample)
        live_resample.link(live_caps)
        live_caps.link(appsink)

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_error)

    def _add_source_branch(self, name: str, target: str, mixer):
        source = Gst.ElementFactory.make(self._source_backend, f"{name}-src")
        if source is None:
            raise RuntimeError(f"Meeting audio source missing: {self._source_backend}")
        if target:
            source.set_property(
                "device" if self._source_backend == "pulsesrc" else "target-object",
                target,
            )

        convert = Gst.ElementFactory.make("audioconvert", f"{name}-convert")
        resample = Gst.ElementFactory.make("audioresample", f"{name}-resample")
        caps = Gst.ElementFactory.make("capsfilter", f"{name}-caps")
        caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )

        for element in [source, convert, resample, caps]:
            self._pipeline.add(element)
        source.link(convert)
        convert.link(resample)
        resample.link(caps)
        caps.link(mixer)

    def _on_new_sample(self, appsink):
        if self._sample_callback is None:
            return Gst.FlowReturn.OK
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK
        try:
            audio = np.frombuffer(map_info.data, dtype=np.float32).copy()
        finally:
            buf.unmap(map_info)
        try:
            self._sample_callback(audio, self._sample_rate)
        except Exception:
            pass
        return Gst.FlowReturn.OK

    def start(self):
        if self._pipeline:
            if self._pipeline.set_state(Gst.State.PLAYING) == \
                    Gst.StateChangeReturn.FAILURE:
                self._running = False
                raise RuntimeError("Meeting audio source could not start")
            self._running = True

    def stop(self):
        if self._pipeline:
            try:
                # Sending EOS to a source that already failed can crash its
                # native teardown. An error or NULL state has no valid WAV to
                # finalize, so release it without injecting another event.
                if self._running and self._bus:
                    pending_error = self._bus.timed_pop_filtered(
                        0, Gst.MessageType.ERROR
                    )
                    if pending_error:
                        self._on_error(self._bus, pending_error)
                state_return, state, _pending = self._pipeline.get_state(0)
                if (self._running and not self._last_error
                        and state_return != Gst.StateChangeReturn.FAILURE
                        and state != Gst.State.NULL):
                    self._pipeline.send_event(Gst.Event.new_eos())
                    if self._bus:
                        msg = self._bus.timed_pop_filtered(
                            2 * Gst.SECOND,
                            Gst.MessageType.EOS | Gst.MessageType.ERROR,
                        )
                        if msg and msg.type == Gst.MessageType.ERROR:
                            self._on_error(self._bus, msg)
            finally:
                self._pipeline.set_state(Gst.State.NULL)
                if self._bus:
                    self._bus.remove_signal_watch()
                self._bus = None
                self._pipeline = None
        if self._output_path:
            try:
                output = Path(self._output_path)
                if output.is_file() and output.stat().st_size == 0:
                    output.unlink()
            except OSError:
                pass
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def _on_error(self, _bus, msg):
        err, debug = msg.parse_error()
        self._last_error = err.message
        self._running = False
        print(f"[NV Broadcast Meeting] Error: {err.message}")
        if debug:
            print(f"[NV Broadcast Meeting] Debug: {debug}")
        if self._error_callback:
            self._error_callback(err.message)

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def route_warning(self) -> str:
        return self._route_warning

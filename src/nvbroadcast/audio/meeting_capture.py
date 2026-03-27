# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""Meeting audio capture for mixed mic + speaker notes/transcription."""

from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np

from nvbroadcast.audio.devices import resolve_pipewire_target, resolve_speaker_monitor


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

    def set_sample_callback(self, callback):
        self._sample_callback = callback

    def build(self, mic_device: str, speaker_device: str, output_path: str):
        self._pipeline = Gst.Pipeline.new("nvbroadcast-meeting-capture")
        mic_target = resolve_pipewire_target(mic_device)
        speaker_target = resolve_speaker_monitor(speaker_device)

        mixer = Gst.ElementFactory.make("audiomixer", "meeting-mixer")
        tee = Gst.ElementFactory.make("tee", "meeting-tee")
        file_queue = Gst.ElementFactory.make("queue", "file-queue")
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

        elements = [mixer, tee, file_queue, live_queue, wavenc, filesink, live_convert, live_resample, live_caps, appsink]
        for element in elements:
            self._pipeline.add(element)

        self._add_source_branch("mic", mic_target, mixer)
        if speaker_target:
            self._add_source_branch("speaker", speaker_target, mixer)

        mixer.link(tee)
        tee.link(file_queue)
        file_queue.link(wavenc)
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
        source = Gst.ElementFactory.make("pipewiresrc", f"{name}-src")
        if target:
            source.set_property("target-object", target)

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
            self._pipeline.set_state(Gst.State.PLAYING)
            self._running = True

    def stop(self):
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def _on_error(self, _bus, msg):
        err, debug = msg.parse_error()
        print(f"[NV Broadcast Meeting] Error: {err.message}")
        if debug:
            print(f"[NV Broadcast Meeting] Debug: {debug}")

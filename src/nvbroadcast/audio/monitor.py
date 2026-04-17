# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Speaker output monitoring and denoising.

Captures application audio output, applies noise removal,
and routes the clean audio to the configured speakers.
Prefers PulseAudio/PipeWire explicit routing and falls back safely.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np

from nvbroadcast.audio.devices import (
    resolve_speaker_monitor,
    resolve_speaker_monitor_name,
    resolve_speaker_sink,
)
from nvbroadcast.audio.effects import AudioEffects


class SpeakerMonitor:
    """Capture and denoise speaker output audio.

    Architecture (loopback approach):
        Application audio -> PipeWire default sink (virtual)
            -> capture from monitor -> denoise -> real speakers
    """

    def __init__(self):
        Gst.init(None)
        self._pipeline: Gst.Pipeline | None = None
        self._bus = None
        self._appsrc = None
        self._effects = AudioEffects()
        self._sample_rate = 48000
        self._running = False
        self._speaker_device = ""

    @property
    def effects(self) -> AudioEffects:
        return self._effects

    def configure(self, speaker_device: str = "", sample_rate: int = 48000):
        self._speaker_device = speaker_device
        self._sample_rate = sample_rate

    def _select_capture_backend(self) -> tuple[str, str]:
        if Gst.ElementFactory.find("pulsesrc") is not None:
            return "pulsesrc", resolve_speaker_monitor_name(self._speaker_device)
        if Gst.ElementFactory.find("pipewiresrc") is not None:
            return "pipewiresrc", resolve_speaker_monitor(self._speaker_device)
        return "autoaudiosrc", ""

    def _select_output_backend(self) -> tuple[str, str]:
        target = resolve_speaker_sink(self._speaker_device)
        if Gst.ElementFactory.find("pulsesink") is not None:
            return "pulsesink", target
        if Gst.ElementFactory.find("pipewiresink") is not None:
            return "pipewiresink", target
        return "autoaudiosink", ""

    def _make_source(self, backend: str, target: str):
        source = Gst.ElementFactory.make(backend, "speaker-monitor")
        if source is None:
            raise RuntimeError(f"Failed to create audio source backend: {backend}")

        if backend == "pulsesrc":
            if target:
                source.set_property("device", target)
            source.set_property("do-timestamp", True)
        elif backend == "pipewiresrc":
            if target:
                source.set_property("target-object", target)
            source.set_property(
                "stream-properties",
                Gst.Structure.new_from_string(
                    "properties,media.class=Audio/Sink,"
                    "node.name=nvbroadcast_speaker_monitor,"
                    'node.description="NVIDIA Broadcast Speaker Monitor"'
                ),
            )
            source.set_property("do-timestamp", True)
            source.set_property("min-buffers", 2)
            source.set_property("max-buffers", 4)
        return source

    def _make_sink(self, backend: str, target: str):
        sink = Gst.ElementFactory.make(backend, "speakers")
        if sink is None:
            raise RuntimeError(f"Failed to create audio sink backend: {backend}")

        if backend == "pulsesink":
            if target:
                sink.set_property("device", target)
        elif backend == "pipewiresink":
            if target:
                sink.set_property("target-object", target)
        sink.set_property("sync", False)
        sink.set_property("async", False)
        return sink

    def _teardown_pipeline(self):
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._bus:
            self._bus.remove_signal_watch()
        self._pipeline = None
        self._bus = None
        self._appsrc = None
        self._running = False

    def build(self) -> None:
        """Build the speaker denoising pipeline.

        Uses the best available routed backend to capture application audio,
        then processes and outputs to the configured real audio device.
        """
        self._teardown_pipeline()
        self._pipeline = Gst.Pipeline.new("nvbroadcast-speaker")
        capture_backend, capture_target = self._select_capture_backend()
        output_backend, output_target = self._select_output_backend()

        source = self._make_source(capture_backend, capture_target)

        convert_in = Gst.ElementFactory.make("audioconvert", "convert-in")
        resample = Gst.ElementFactory.make("audioresample", "resample")

        caps = Gst.ElementFactory.make("capsfilter", "mono-caps")
        caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                "channels=1,layout=interleaved"
            ),
        )

        appsink = Gst.ElementFactory.make("appsink", "speaker-sink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 2)
        appsink.set_property("drop", True)
        appsink.connect("new-sample", self._on_new_sample)

        self._appsrc = Gst.ElementFactory.make("appsrc", "speaker-src")
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("max-buffers", 2)
        self._appsrc.set_property("max-time", 40 * Gst.MSECOND)
        self._appsrc.set_property("leaky-type", 1)
        self._appsrc.set_property("block", False)
        self._appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                "channels=1,layout=interleaved"
            ),
        )

        convert_out = Gst.ElementFactory.make("audioconvert", "convert-out")

        # Output to configured speakers using the selected backend.
        sink = self._make_sink(output_backend, output_target)

        for el in [source, convert_in, resample, caps, appsink,
                   self._appsrc, convert_out, sink]:
            self._pipeline.add(el)

        source.link(convert_in)
        convert_in.link(resample)
        resample.link(caps)
        caps.link(appsink)

        self._appsrc.link(convert_out)
        convert_out.link(sink)

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_error)

    def _on_new_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        audio = np.frombuffer(map_info.data, dtype=np.float32).copy()
        buf.unmap(map_info)

        processed = self._effects.process_chunk(audio, self._sample_rate)

        new_buf = Gst.Buffer.new_allocate(None, len(processed.tobytes()), None)
        new_buf.fill(0, processed.tobytes())
        new_buf.pts = buf.pts
        new_buf.dts = buf.dts
        new_buf.duration = buf.duration

        self._appsrc.emit("push-buffer", new_buf)
        return Gst.FlowReturn.OK

    def start(self):
        if self._pipeline:
            self._effects.initialize()
            self._pipeline.set_state(Gst.State.PLAYING)
            self._running = True

    def stop(self):
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._running = False

    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"[NVIDIA Broadcast Speaker] Error: {err.message}")
        if debug:
            print(f"[NVIDIA Broadcast Speaker] Debug: {debug}")

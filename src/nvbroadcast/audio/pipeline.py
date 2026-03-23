# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Audio pipeline: mic capture -> denoise -> virtual mic output via GStreamer/PipeWire."""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

from nvbroadcast.audio.effects import AudioEffects


class AudioPipeline:
    """GStreamer audio pipeline for real-time noise removal.

    Pipeline:
        pipewiresrc (mic) -> audioconvert -> appsink
        [Python: RNNoise denoising]
        appsrc -> audioconvert -> pipewiresink (virtual mic)
    """

    def __init__(self):
        Gst.init(None)
        self._pipeline: Gst.Pipeline | None = None
        self._appsrc = None
        self._effects = AudioEffects()
        self._sample_rate = 48000
        self._channels = 1
        self._running = False
        self._mic_device = ""

    @property
    def effects(self) -> AudioEffects:
        return self._effects

    def configure(self, mic_device: str = "", sample_rate: int = 48000):
        self._mic_device = mic_device
        self._sample_rate = sample_rate

    def build(self) -> None:
        """Build the audio pipeline."""
        self._pipeline = Gst.Pipeline.new("nvbroadcast-audio")

        # Source: microphone via PipeWire
        source = Gst.ElementFactory.make("pipewiresrc", "mic-source")
        if self._mic_device:
            source.set_property("target-object", self._mic_device)

        # Convert to float32 mono
        convert_in = Gst.ElementFactory.make("audioconvert", "convert-in")
        resample_in = Gst.ElementFactory.make("audioresample", "resample-in")

        in_caps = Gst.ElementFactory.make("capsfilter", "in-caps")
        in_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )

        # Appsink: extract audio for processing
        appsink = Gst.ElementFactory.make("appsink", "audio-sink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 10)
        appsink.set_property("drop", False)
        appsink.connect("new-sample", self._on_new_sample)

        # Appsrc: inject processed audio
        self._appsrc = Gst.ElementFactory.make("appsrc", "audio-src")
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )

        # Output: virtual mic via PipeWire
        convert_out = Gst.ElementFactory.make("audioconvert", "convert-out")
        sink = Gst.ElementFactory.make("pipewiresink", "mic-output")
        sink.set_property("stream-properties",
                          Gst.Structure.new_from_string(
                              "properties,media.class=Audio/Source,"
                              "node.name=nvbroadcast_mic,"
                              'node.description="NVIDIA Broadcast Microphone"'
                          ))

        # Add all elements
        for el in [source, convert_in, resample_in, in_caps, appsink,
                   self._appsrc, convert_out, sink]:
            self._pipeline.add(el)

        # Link input chain
        source.link(convert_in)
        convert_in.link(resample_in)
        resample_in.link(in_caps)
        in_caps.link(appsink)

        # Link output chain
        self._appsrc.link(convert_out)
        convert_out.link(sink)

        # Error handling
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)

    def _on_new_sample(self, appsink):
        """Process audio through denoiser."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        # Convert to numpy float32 array
        audio = np.frombuffer(map_info.data, dtype=np.float32).copy()
        buf.unmap(map_info)

        # Apply denoising
        processed = self._effects.process_chunk(audio, self._sample_rate)

        # Push processed audio
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
        print(f"[NVIDIA Broadcast Audio] Error: {err.message}")
        if debug:
            print(f"[NVIDIA Broadcast Audio] Debug: {debug}")

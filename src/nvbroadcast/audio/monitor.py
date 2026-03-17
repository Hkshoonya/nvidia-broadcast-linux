# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Speaker output monitoring and denoising.

Captures application audio output, applies noise removal,
and routes the clean audio to the physical speakers.
Uses PipeWire loopback approach.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

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
        self._appsrc = None
        self._effects = AudioEffects()
        self._sample_rate = 48000
        self._running = False

    @property
    def effects(self) -> AudioEffects:
        return self._effects

    def build(self) -> None:
        """Build the speaker denoising pipeline.

        Uses pipewiresrc with a monitor target to capture application audio,
        then processes and outputs to the real audio device.
        """
        self._pipeline = Gst.Pipeline.new("nvbroadcast-speaker")

        # Source: capture from default sink's monitor
        source = Gst.ElementFactory.make("pipewiresrc", "speaker-monitor")
        source.set_property(
            "stream-properties",
            Gst.Structure.new_from_string(
                "properties,media.class=Audio/Sink,"
                "node.name=nvbroadcast_speaker_monitor,"
                'node.description="NVIDIA Broadcast Speaker Monitor"'
            ),
        )

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
        appsink.set_property("max-buffers", 10)
        appsink.connect("new-sample", self._on_new_sample)

        self._appsrc = Gst.ElementFactory.make("appsrc", "speaker-src")
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                "channels=1,layout=interleaved"
            ),
        )

        convert_out = Gst.ElementFactory.make("audioconvert", "convert-out")

        # Output to real speakers via autoaudiosink
        sink = Gst.ElementFactory.make("autoaudiosink", "speakers")

        for el in [source, convert_in, resample, caps, appsink,
                   self._appsrc, convert_out, sink]:
            self._pipeline.add(el)

        source.link(convert_in)
        convert_in.link(resample)
        resample.link(caps)
        caps.link(appsink)

        self._appsrc.link(convert_out)
        convert_out.link(sink)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)

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

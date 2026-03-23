# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""GStreamer video pipeline with two modes:

- Passthrough: Direct GStreamer pipeline, zero Python overhead, minimal CPU
- Effects: appsink/appsrc with Python processing

Switches between modes when effects are toggled.
"""

import threading

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gst, GstVideo, GLib, Gdk

from nvbroadcast.core.constants import DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS


class VideoPipeline:
    def __init__(self):
        Gst.init(None)
        self._pipeline: Gst.Pipeline | None = None
        self._vcam_pipeline: Gst.Pipeline | None = None
        self._source_device: str = "/dev/video0"
        self._vcam_device: str = "/dev/video10"
        self._width: int = DEFAULT_WIDTH
        self._height: int = DEFAULT_HEIGHT
        self._fps: int = DEFAULT_FPS
        self._output_format: str = "YUY2"
        self._effects_fps: int = 30  # Can be reduced by performance profile
        self._effect_callback = None
        self._preview_callback = None
        self._vcam_appsrc = None
        self._vcam_enabled = True
        self._effects_active = False
        self._frame_count = 0
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._teardown_done = True

    def configure(self, source_device, vcam_device,
                  width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                  fps=DEFAULT_FPS, output_format="YUY2",
                  effects_fps=30):
        self._source_device = source_device
        self._vcam_device = vcam_device
        self._width = width
        self._height = height
        self._fps = fps
        self._output_format = output_format
        self._effects_fps = min(effects_fps, fps)

    def set_effect_callback(self, callback):
        self._effect_callback = callback

    def set_preview_callback(self, callback):
        self._preview_callback = callback

    def set_effects_active(self, active: bool):
        """Switch between passthrough (fast) and effects (processing) mode."""
        if active == self._effects_active:
            return
        if not self._running:
            self._effects_active = active
            return

        # Rebuild pipeline in the new mode (deferred to avoid blocking GTK main loop)
        self._effects_active = active
        GLib.idle_add(self._rebuild_pipeline)

    def build(self, vcam_enabled: bool = True) -> None:
        self._vcam_enabled = vcam_enabled

        if self._effects_active:
            self._build_effects_pipeline(vcam_enabled)
        else:
            self._build_passthrough_pipeline(vcam_enabled)

        if self._preview_callback:
            GLib.timeout_add(33, self._tick_preview)  # ~30fps preview

    def _build_passthrough_pipeline(self, vcam_enabled: bool):
        """Direct GStreamer pipeline - ZERO Python processing.

        webcam -> jpegdec -> tee -> videoconvert -> v4l2sink (virtual cam)
                               |-> appsink (preview only, low priority)

        CPU usage: near zero (all in GStreamer C code).
        """
        tee_branch = ""
        if vcam_enabled:
            tee_branch = (
                f"tee name=t "
                f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! "
                f"video/x-raw,format={self._output_format},width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"v4l2sink device={self._vcam_device} sync=false "
                f"t. ! queue max-size-buffers=1 leaky=downstream ! "
                f"videoconvert ! video/x-raw,format=BGRA ! "
                f"appsink name=preview emit-signals=true max-buffers=1 drop=true sync=false"
            )
        else:
            tee_branch = (
                f"videoconvert ! video/x-raw,format=BGRA ! "
                f"appsink name=preview emit-signals=true max-buffers=1 drop=true sync=false"
            )

        # Use GPU JPEG decode in passthrough mode too
        if self._has_gst_element("nvjpegdec"):
            decoder = "nvjpegdec ! cudadownload"
        else:
            decoder = "jpegdec"

        self._pipeline = Gst.parse_launch(
            f"v4l2src device={self._source_device} ! "
            f"image/jpeg,width={self._width},height={self._height},"
            f"framerate={self._fps}/1 ! "
            f"{decoder} ! "
            f"{tee_branch}"
        )

        preview_sink = self._pipeline.get_by_name("preview")
        if preview_sink:
            preview_sink.connect("new-sample", self._on_preview_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)

    def _build_effects_pipeline(self, vcam_enabled: bool):
        """appsink/appsrc pipeline for Python effect processing."""
        # Use NVIDIA GPU for JPEG decode — saves ~60% CPU vs software jpegdec
        if self._has_gst_element("nvjpegdec"):
            decoder = "nvjpegdec ! cudadownload ! videoconvert"
        else:
            decoder = "jpegdec ! videoconvert"

        # Throttle effects FPS via videorate — fewer frames = less CPU
        # (camera still captures at full fps, but we drop frames before Python)
        efps = self._effects_fps
        if efps < self._fps:
            throttle = f"videorate ! video/x-raw,framerate={efps}/1 ! "
        else:
            throttle = ""

        self._pipeline = Gst.parse_launch(
            f"v4l2src device={self._source_device} ! "
            f"image/jpeg,width={self._width},height={self._height},"
            f"framerate={self._fps}/1 ! "
            f"{decoder} ! "
            f"video/x-raw,format=BGRA,width={self._width},height={self._height} ! "
            f"{throttle}"
            f"appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
        )
        sink = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_effects_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)

        if vcam_enabled:
            self._vcam_pipeline = Gst.parse_launch(
                f"appsrc name=src is-live=true format=time "
                f"caps=video/x-raw,format=BGRA,width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! "
                f"video/x-raw,format={self._output_format},width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"v4l2sink device={self._vcam_device} sync=false"
            )
            self._vcam_appsrc = self._vcam_pipeline.get_by_name("src")

            vbus = self._vcam_pipeline.get_bus()
            vbus.add_signal_watch()
            vbus.connect("message::error", self._on_vcam_error)

    def _on_preview_sample(self, appsink):
        """Lightweight preview-only callback (passthrough mode)."""
        if not self._running:
            return Gst.FlowReturn.EOS
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        with self._lock:
            self._latest_frame = bytes(info.data)
        buf.unmap(info)

        self._frame_count += 1
        return Gst.FlowReturn.OK

    def _on_effects_sample(self, appsink):
        """Full processing callback (effects mode)."""
        if not self._running:
            return Gst.FlowReturn.EOS
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        frame_data = bytes(info.data)
        buf.unmap(info)

        expected = self._width * self._height * 4
        if len(frame_data) != expected:
            return Gst.FlowReturn.OK

        # Apply effects
        if self._effect_callback:
            output = self._effect_callback(frame_data, self._width, self._height)
            if output is None:
                output = frame_data
        else:
            output = frame_data

        # Store for preview
        with self._lock:
            self._latest_frame = output

        # Push to vcam
        if self._vcam_enabled and self._vcam_appsrc:
            vcam_buf = Gst.Buffer.new_allocate(None, len(output), None)
            vcam_buf.fill(0, output)
            vcam_buf.pts = buf.pts
            vcam_buf.duration = buf.duration
            self._vcam_appsrc.emit("push-buffer", vcam_buf)

        self._frame_count += 1
        return Gst.FlowReturn.OK

    def _tick_preview(self) -> bool:
        if self._pipeline is None:
            return False

        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None

        if frame is None or self._preview_callback is None:
            return True

        expected = self._width * self._height * 4
        if len(frame) != expected:
            return True

        try:
            gbytes = GLib.Bytes.new(frame)
            texture = Gdk.MemoryTexture.new(
                self._width, self._height,
                Gdk.MemoryFormat.B8G8R8A8,
                gbytes, self._width * 4,
            )
            self._preview_callback(texture)
        except Exception as e:
            if self._frame_count < 5:
                print(f"[NV Broadcast] Preview error: {e}")

        return True

    def _rebuild_pipeline(self):
        """Rebuild pipeline after mode change. Runs on GTK main loop via idle_add."""
        self.stop()
        GLib.timeout_add(300, self._finish_rebuild)
        return False

    def _finish_rebuild(self):
        if not self._teardown_done:
            return True  # Keep polling until teardown finishes
        self.build(vcam_enabled=self._vcam_enabled)
        self.start()
        return False

    def start(self):
        if self._vcam_pipeline:
            self._vcam_pipeline.set_state(Gst.State.PLAYING)
        if self._pipeline:
            self._pipeline.set_state(Gst.State.PLAYING)
        self._running = True

    def stop(self):
        self._running = False
        cap = self._pipeline
        vcam = self._vcam_pipeline
        self._pipeline = None
        self._vcam_pipeline = None
        self._vcam_appsrc = None

        # Disable signals to prevent callback deadlock
        if cap:
            for name in ("sink", "preview"):
                elem = cap.get_by_name(name)
                if elem:
                    elem.set_property("emit-signals", False)

        # Teardown in background thread — never block GTK main loop
        self._teardown_done = False
        if cap or vcam:
            def _teardown():
                if cap:
                    cap.set_state(Gst.State.NULL)
                    cap.get_state(5 * Gst.SECOND)
                if vcam:
                    vcam.set_state(Gst.State.NULL)
                    vcam.get_state(5 * Gst.SECOND)
                self._teardown_done = True
            threading.Thread(target=_teardown, daemon=True).start()
        else:
            self._teardown_done = True

    @staticmethod
    def _has_gst_element(name: str) -> bool:
        """Check if a GStreamer element is available."""
        factory = Gst.ElementFactory.find(name)
        return factory is not None

    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"[NV Broadcast] Capture error: {err.message}")
        if debug:
            print(f"[NV Broadcast] Debug: {debug}")

    def _on_vcam_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"[NV Broadcast] VCam error: {err.message}")
        if debug:
            print(f"[NV Broadcast] Debug: {debug}")

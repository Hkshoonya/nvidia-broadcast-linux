# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
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
        self._throttle_acc = 0.0
        self._last_effect_output = None
        self._pending_frame = None       # Latest raw frame for effects thread
        self._effects_busy = False       # True while effects thread is processing

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
            preview_ms = max(16, 1000 // self._fps)  # Match camera fps (16ms = 60fps)
            GLib.timeout_add(preview_ms, self._tick_preview)

    def _build_passthrough_pipeline(self, vcam_enabled: bool):
        """Direct GStreamer pipeline - ZERO Python processing.

        webcam -> jpegdec -> tee -> videoconvert -> v4l2sink (virtual cam)
                               |-> appsink (preview only, low priority)

        CPU usage: near zero (all in GStreamer C code).
        """
        from nvbroadcast.core.platform import IS_MACOS, get_gst_camera_caps

        tee_branch = ""
        if vcam_enabled and not IS_MACOS:
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

        # Camera source — platform-aware
        camera_src = get_gst_camera_caps(
            self._source_device, self._width, self._height, self._fps
        )

        if IS_MACOS:
            # macOS: avfvideosrc outputs raw video, no JPEG decode needed
            decoder = "videoconvert"
        elif self._has_gst_element("nvjpegdec"):
            decoder = "nvjpegdec ! cudadownload"
        else:
            decoder = "jpegdec"

        self._pipeline = Gst.parse_launch(
            f"{camera_src} ! "
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
        from nvbroadcast.core.platform import IS_MACOS, get_gst_camera_caps

        camera_src = get_gst_camera_caps(
            self._source_device, self._width, self._height, self._fps
        )

        if IS_MACOS:
            decoder = "videoconvert"
        elif self._has_gst_element("nvjpegdec"):
            decoder = "nvjpegdec ! cudadownload ! videoconvert"
        else:
            decoder = "jpegdec ! videoconvert"

        # No videorate — frame throttling is done in Python (_on_effects_sample)
        # so mode/profile changes never require a pipeline restart.
        self._pipeline = Gst.parse_launch(
            f"{camera_src} ! "
            f"{decoder} ! "
            f"video/x-raw,format=BGRA,width={self._width},height={self._height} ! "
            f"appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
        )
        sink = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_effects_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)

        if vcam_enabled:
            if IS_MACOS:
                # macOS: CoreMediaIO frame bridge (proprietary) → pyvirtualcam fallback
                self._vcam_pipeline = None
                self._vcam_appsrc = None
                self._frame_bridge = None
                self._pyvirtualcam = None
                try:
                    from macos.NVBroadcastHelper.frame_bridge import FrameBridge
                    self._frame_bridge = FrameBridge(
                        width=self._width, height=self._height
                    )
                    print("[NV Broadcast] macOS virtual camera: CoreMediaIO extension")
                except Exception:
                    try:
                        import pyvirtualcam
                        self._pyvirtualcam = pyvirtualcam.Camera(
                            width=self._width, height=self._height,
                            fps=self._fps, backend="obs",
                        )
                        print(f"[NV Broadcast] macOS virtual camera: {self._pyvirtualcam.device}")
                    except Exception as e:
                        print(f"[NV Broadcast] macOS virtual camera not available: {e}")
            else:
                self._pyvirtualcam = None
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

    def set_effects_fps(self, efps: int):
        """Change effects throttle at runtime — no pipeline restart needed."""
        self._effects_fps = min(efps, self._fps)

    def _on_effects_sample(self, appsink):
        """Capture callback — NEVER blocks. Effects run in a background thread.

        This keeps preview/vcam at full camera fps with minimal latency.
        Effects results are applied as soon as they're ready.
        """
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

        self._frame_count += 1

        # Kick off effects processing in background (non-blocking)
        if self._effect_callback and not self._effects_busy:
            run_effects = True
            if self._effects_fps < self._fps:
                self._throttle_acc += 1.0 - (self._effects_fps / self._fps)
                if self._throttle_acc >= 1.0:
                    self._throttle_acc -= 1.0
                    run_effects = False
            if run_effects:
                self._pending_frame = frame_data
                self._effects_busy = True
                threading.Thread(
                    target=self._process_effects_bg,
                    args=(frame_data, self._width, self._height, expected),
                    daemon=True,
                ).start()

        # Output: latest processed frame, or raw if none ready yet
        cached = self._last_effect_output
        output = cached if (cached and len(cached) == expected) else frame_data

        # Store for preview (always current — no lag)
        with self._lock:
            self._latest_frame = output

        # Push to vcam at full fps
        if self._vcam_enabled:
            if self._vcam_appsrc:
                # Linux: GStreamer v4l2sink
                vcam_buf = Gst.Buffer.new_allocate(None, len(output), None)
                vcam_buf.fill(0, output)
                vcam_buf.pts = buf.pts
                vcam_buf.duration = buf.duration
                self._vcam_appsrc.emit("push-buffer", vcam_buf)
            elif getattr(self, '_frame_bridge', None):
                # macOS: CoreMediaIO shared memory bridge
                self._frame_bridge.write_frame(output)
            elif getattr(self, '_pyvirtualcam', None):
                # macOS fallback: pyvirtualcam (OBS)
                import numpy as np
                frame = np.frombuffer(output, dtype=np.uint8).reshape(
                    self._height, self._width, 4
                )
                self._pyvirtualcam.send(frame[:, :, :3])  # BGRA→BGR

        return Gst.FlowReturn.OK

    def _process_effects_bg(self, frame_data, width, height, expected):
        """Run effects in background thread — never blocks the capture."""
        try:
            output = self._effect_callback(frame_data, width, height)
            if output is not None and len(output) == expected:
                self._last_effect_output = output
        except Exception as e:
            if self._frame_count <= 5:
                print(f"[NV Broadcast] Effects error: {e}")
        finally:
            self._effects_busy = False

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

        # Clean up macOS vcam backends
        if getattr(self, '_frame_bridge', None):
            self._frame_bridge.close()
            self._frame_bridge = None
        if getattr(self, '_pyvirtualcam', None):
            self._pyvirtualcam.close()
            self._pyvirtualcam = None

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
                # Send EOS first to help v4l2 release the device cleanly
                if cap:
                    cap.send_event(Gst.Event.new_eos())
                if vcam:
                    vcam.send_event(Gst.Event.new_eos())
                # Short timeout — don't block forever waiting for NULL
                if cap:
                    cap.set_state(Gst.State.NULL)
                    cap.get_state(2 * Gst.SECOND)
                if vcam:
                    vcam.set_state(Gst.State.NULL)
                    vcam.get_state(2 * Gst.SECOND)
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

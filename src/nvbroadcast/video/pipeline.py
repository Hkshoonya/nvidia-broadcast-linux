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
import time
import subprocess
import os

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gst, GstVideo, GLib, Gdk

from nvbroadcast.core.constants import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FPS,
    VIRTUAL_CAM_LABEL,
)


# Formats cudaconvert handles reliably for the up/convert/download segment.
_CUDA_SAFE_FORMATS = {"YUY2", "UYVY", "NV12", "I420", "BGRA", "RGBA"}


class VideoPipeline:
    # One-time probe result shared across instances (None = not yet probed)
    _cuda_convert_probe_result: bool | None = None

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
        self._capture_format: str = "mjpeg"
        self._prefer_hw_decode = False
        self._effects_fps: int = 30  # Can be reduced by performance profile
        self._effect_callback = None
        self._alpha_callback = None
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
        self._effects_busy = False       # True while alpha worker is processing
        self._alpha_worker_enabled = True
        self._alpha_thread = None
        self._alpha_pending = None
        self._alpha_condition = threading.Condition()
        self._alpha_shutdown = False
        self._vcam_failed = False
        self._recording = False
        self._recording_pipeline = None
        self._rec_appsrc = None
        self._paused = False
        self._frozen_frame = None
        self._teardown_lock = threading.Lock()
        self._teardown_source_id = 0
        self._teardown_capture = None
        self._teardown_vcam = None
        self._rebuild_source_id = 0
        self._rebuild_pending = False
        self._callbacks_in_flight = 0
        self._callback_lock = threading.Lock()
        self._cuda_convert_demoted = False
        self._vcam_used_cuda = False
        self._capture_idle = False
        self._preview_enabled = True
        self._preview_timer_id = 0
        # Device-resident frame path (GpuFramePath). The processor and the
        # per-frame plan callable come from the app; demotion is one-shot,
        # mirroring _cuda_convert_demoted.
        self._frame_processor = None
        self._frame_plan = None
        self._gpu_path_demoted = False
        self._gpu_path_errors = 0
        self._gpu_capture_active = False
        self._gpu_jpeg_active = False
        self._gpu_jpeg_demoted = False
        self._gpu_frame_size = (0, 0)
        self._frozen_yuy2 = None

    def _v4l2sink_segment(self) -> str:
        """Build a loopback sink path that avoids buggy allocation queries.

        On some newer kernels, v4l2loopback + GstV4l2Sink can trip over the
        default MMAP allocation path and fail immediately with
        "buffer 0 was not queued". Forcing allocation-drop before the sink
        keeps the loopback path stable while preserving the faster sink mode.
        """
        return (
            "identity drop-allocation=true ! "
            f"v4l2sink device={self._vcam_device} io-mode=2 sync=false async=false"
        )

    def _vcam_reports_output(self) -> bool:
        """Return whether the loopback device currently reports output caps."""
        if not self._vcam_device.startswith("/dev/video"):
            return True
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-D", "-d", self._vcam_device],
                capture_output=True,
                text=True,
                timeout=1,
            )
        except Exception:
            return True
        return "Video Output" in result.stdout

    def _describe_vcam_state(self) -> str:
        """Summarize current loopback caps and openers for diagnostics."""
        caps = "unknown"
        holders = "unknown"

        if self._vcam_device.startswith("/dev/video"):
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "-D", "-d", self._vcam_device],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if "Video Output" in result.stdout:
                    caps = "output"
                elif "Video Capture" in result.stdout:
                    caps = "capture"
                else:
                    caps = "unreported"
            except Exception:
                pass

            try:
                result = subprocess.run(
                    ["fuser", "-v", self._vcam_device],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                merged = (result.stdout + " " + result.stderr).strip()
                holders = merged.replace("\n", " | ") if merged else "none"
            except Exception:
                pass

        return f"caps={caps}, holders={holders}"

    def _wait_for_vcam_output(self, timeout_seconds: float = 0.8) -> bool:
        """Wait briefly for v4l2loopback to return to output mode."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._vcam_reports_output():
                return True
            time.sleep(0.05)
        return self._vcam_reports_output()

    def _start_vcam_with_retry(self, attempts: int = 3) -> bool:
        """Start the loopback sink with a short retry window.

        `exclusive_caps=1` can briefly leave the loopback node reporting capture
        caps after a previous writer or consumer disconnects. Retry a few times
        instead of disabling virtual camera output immediately.
        """
        if not self._vcam_pipeline:
            return True

        last_state = self._describe_vcam_state()
        for attempt in range(attempts):
            if attempt:
                try:
                    self._vcam_pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
                self._wait_for_vcam_output()
                time.sleep(0.1 * attempt)

            ret = self._vcam_pipeline.set_state(Gst.State.PLAYING)
            if ret != Gst.StateChangeReturn.FAILURE:
                return True

            last_state = self._describe_vcam_state()

        try:
            self._vcam_pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        self._vcam_pipeline = None
        self._vcam_appsrc = None

        # Before giving up, retry once on the CPU convert path in case the
        # CUDA segment is what failed to start.
        if self._vcam_used_cuda and not self._cuda_convert_demoted:
            self._cuda_convert_demoted = True
            print("[NV Broadcast] VCam start failed on CUDA path — retrying "
                  "with CPU videoconvert", flush=True)
            try:
                self._build_vcam_pipeline()
                return self._start_vcam_with_retry(attempts=attempts)
            except Exception as e:
                print(f"[NV Broadcast] VCam CPU rebuild failed: {e}", flush=True)

        print(
            "[NV Broadcast] VCam pipeline failed to start — disabling vcam "
            f"({last_state})",
            flush=True,
        )
        self._vcam_enabled = False
        self._vcam_failed = True
        return False

    def configure(self, source_device, vcam_device,
                  width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                  fps=DEFAULT_FPS, output_format="YUY2",
                  effects_fps=30, prefer_hw_decode: bool = False):
        self._source_device = source_device
        self._vcam_device = vcam_device
        self._width = width
        self._height = height
        self._fps = fps
        self._output_format = output_format
        self._effects_fps = min(effects_fps, fps)
        self._prefer_hw_decode = prefer_hw_decode
        from nvbroadcast.core.platform import IS_MACOS
        if IS_MACOS:
            self._capture_format = "raw"
        else:
            from nvbroadcast.video.virtual_camera import select_camera_capture_format
            self._capture_format = select_camera_capture_format(
                source_device, width, height, fps
            )

    def _select_decoder(self, effects_mode: bool) -> str:
        """Choose camera decode path for the active mode.

        Killer mode is intended to be the one explicit hardware-decode path.
        Other modes should remain stable and comparable, so they stay on the
        software JPEG path unless hardware decode was specifically requested.
        """
        from nvbroadcast.core.platform import IS_MACOS

        if IS_MACOS:
            return "videoconvert"

        # jpegdec has no GPU replacement in stock GStreamer, but the
        # post-decode conversion to BGRA can run on CUDA.
        convert = self._convert_segment("BGRA")

        if self._capture_format == "raw":
            return convert if effects_mode else "identity"

        if self._prefer_hw_decode and self._has_gst_element("nvjpegdec"):
            if effects_mode:
                return "nvjpegdec ! cudadownload ! videoconvert n-threads=2 qos=false"
            return "nvjpegdec ! cudadownload"

        if effects_mode:
            return f"jpegdec ! {convert}"
        return "jpegdec"

    def set_effect_callback(self, callback):
        self._effect_callback = callback

    def set_frame_processor(self, processor, plan_callback):
        """Register the device-resident frame path.

        ``processor`` is a GpuFramePath; ``plan_callback()`` is consulted
        per frame and returns (gpu_pure, mirror, inline_inference). When
        gpu_pure is False the frame round-trips through the legacy bytes
        effect callback (CPU face effects) but still uses the GPU for
        colorspace conversion and the convert-free vcam leg.
        """
        self._frame_processor = processor
        self._frame_plan = plan_callback

    def _gpu_path_enabled(self) -> bool:
        return (
            self._frame_processor is not None
            and not self._gpu_path_demoted
            and self._output_format == "YUY2"
            and os.getenv("NVBROADCAST_NO_GPU_FRAME_PATH") != "1"
        )

    def _inference_due(self) -> bool:
        """Effects-fps throttle: False on frames whose inference is skipped."""
        if self._effects_fps < self._fps:
            self._throttle_acc += 1.0 - (self._effects_fps / self._fps)
            if self._throttle_acc >= 1.0:
                self._throttle_acc -= 1.0
                return False
        return True

    def set_alpha_callback(self, callback):
        """Set callback for background alpha inference (heavy work)."""
        self._alpha_callback = callback

    def set_alpha_worker_enabled(self, enabled: bool):
        """Enable/disable background alpha inference worker.

        When inline inference is active, alpha updates must not run in parallel
        against the same cached matte or replacement edges will fight themselves.
        """
        self._alpha_worker_enabled = enabled
        if not enabled:
            with self._alpha_condition:
                self._alpha_pending = None
                self._alpha_condition.notify_all()

    def _ensure_alpha_worker(self) -> None:
        """Start a dedicated alpha worker thread once for this pipeline.

        ONNX Runtime CUDA sessions are stable when one thread owns inference,
        but they can fail with invalid resource handles if inference hops across
        many short-lived Python threads. Keep one long-lived worker instead of
        spawning a fresh thread for each frame.
        """
        with self._alpha_condition:
            if self._alpha_shutdown:
                return
            if self._alpha_thread and self._alpha_thread.is_alive():
                return
            self._alpha_thread = threading.Thread(
                target=self._alpha_worker_loop,
                name="nvbroadcast-alpha",
                daemon=True,
            )
            self._alpha_thread.start()

    def _submit_alpha_frame(self, frame_data: bytes, width: int, height: int) -> None:
        """Queue the newest frame for background alpha inference."""
        self._ensure_alpha_worker()
        with self._alpha_condition:
            self._alpha_pending = (frame_data, width, height)
            self._alpha_condition.notify()

    def _alpha_worker_loop(self) -> None:
        """Own alpha inference on one persistent worker thread."""
        while True:
            with self._alpha_condition:
                while self._alpha_pending is None and not self._alpha_shutdown:
                    self._alpha_condition.wait()
                if self._alpha_shutdown:
                    return
                frame_data, width, height = self._alpha_pending
                self._alpha_pending = None
                self._effects_busy = True
            try:
                if self._alpha_callback:
                    self._alpha_callback(frame_data, width, height)
            except Exception as e:
                if self._frame_count <= 10 or self._frame_count % 300 == 0:
                    print(f"[NV Broadcast] Alpha/effects error: {e}", flush=True)
            finally:
                self._effects_busy = False

    def _stop_alpha_worker(self, timeout_seconds: float = 0.5) -> None:
        """Stop the dedicated alpha worker during app shutdown."""
        thread = None
        with self._alpha_condition:
            self._alpha_shutdown = True
            self._alpha_pending = None
            thread = self._alpha_thread
            self._alpha_condition.notify_all()
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout_seconds)
        with self._alpha_condition:
            self._alpha_thread = None
            self._alpha_shutdown = False

    def set_preview_callback(self, callback):
        self._preview_callback = callback

    def set_preview_enabled(self, enabled: bool):
        """Gate the preview timer on window visibility.

        Building a Gdk texture is a full-frame copy plus a GPU upload per
        tick; while the window is hidden the timer itself is removed so the
        main loop stops waking at frame rate entirely.
        """
        self._preview_enabled = bool(enabled)
        if self._preview_enabled:
            self._start_preview_timer()
        else:
            self._stop_preview_timer()

    def _start_preview_timer(self):
        if (self._preview_timer_id or not self._preview_enabled
                or not self._preview_callback or self._pipeline is None):
            return
        preview_ms = max(16, 1000 // self._fps)  # Match camera fps (16ms = 60fps)
        self._preview_timer_id = GLib.timeout_add(preview_ms, self._tick_preview)

    def _stop_preview_timer(self):
        if self._preview_timer_id:
            GLib.source_remove(self._preview_timer_id)
            self._preview_timer_id = 0

    def set_capture_idle(self, idle: bool) -> bool:
        """Idle the capture pipeline while keeping the vcam device open.

        PAUSED stops the camera stream at the source (webcam LED off, no
        JPEG decode, no conversion, no callbacks) but keeps every element
        alive. In effects mode the separate vcam appsrc pipeline stays
        PLAYING, so v4l2loopback keeps its writer and meeting apps still
        see the device; resume is one state change back to PLAYING.
        """
        if self._pipeline is None or self._capture_idle == idle:
            return self._capture_idle
        try:
            target = Gst.State.PAUSED if idle else Gst.State.PLAYING
            if self._pipeline.set_state(target) == Gst.StateChangeReturn.FAILURE:
                print("[NV Broadcast] Capture idle state change failed", flush=True)
                return self._capture_idle
            self._capture_idle = idle
            print(f"[NV Broadcast] Capture {'idled (power save)' if idle else 'resumed'}",
                  flush=True)
        except Exception as e:
            print(f"[NV Broadcast] Capture idle toggle failed: {e}", flush=True)
        return self._capture_idle

    @property
    def capture_idle(self) -> bool:
        return self._capture_idle

    def set_paused(self, paused: bool):
        """Freeze/unfreeze video output. Camera stays on but output freezes."""
        self._paused = paused
        if paused:
            with self._lock:
                self._frozen_frame = self._latest_frame
        else:
            self._frozen_frame = None
            self._frozen_yuy2 = None

    def set_effects_active(self, active: bool):
        """Switch between passthrough (fast) and effects (processing) mode."""
        if active == self._effects_active:
            return
        if not self._running:
            self._effects_active = active
            return

        # Rebuild pipeline in the new mode only after teardown completes.
        self._effects_active = active
        self._queue_rebuild()

    def _queue_rebuild(self):
        """Schedule one teardown-safe rebuild on the GTK main loop."""
        self._rebuild_pending = True
        if self._rebuild_source_id:
            return
        self._rebuild_source_id = GLib.timeout_add(
            10, self._rebuild_pipeline, priority=GLib.PRIORITY_HIGH
        )

    def _cancel_rebuild(self):
        """Drop any pending internal rebuild request."""
        if self._rebuild_source_id:
            GLib.source_remove(self._rebuild_source_id)
            self._rebuild_source_id = 0
        self._rebuild_pending = False

    def build(self, vcam_enabled: bool = True) -> None:
        self._vcam_enabled = vcam_enabled

        if self._effects_active:
            self._build_effects_pipeline(vcam_enabled)
        else:
            self._build_passthrough_pipeline(vcam_enabled)

        self._start_preview_timer()

    def _build_passthrough_pipeline(self, vcam_enabled: bool):
        """Direct GStreamer pipeline - ZERO Python processing.

        webcam -> decode/convert -> tee -> videoconvert -> v4l2sink (virtual cam)
                                      |-> appsink (preview only, low priority)

        CPU usage: near zero (all in GStreamer C code).
        """
        from nvbroadcast.core.platform import IS_MACOS, get_gst_camera_caps

        tee_branch = ""
        if vcam_enabled and not IS_MACOS:
            tee_branch = (
                f"tee name=t "
                f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                f"{self._convert_segment(self._output_format)} ! "
                f"video/x-raw,format={self._output_format},width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"{self._v4l2sink_segment()} "
                f"t. ! queue max-size-buffers=1 leaky=downstream ! "
                f"videoconvert n-threads=2 qos=false ! video/x-raw,format=BGRA ! "
                f"appsink name=preview emit-signals=true max-buffers=1 drop=true sync=false"
            )
        else:
            tee_branch = (
                f"videoconvert n-threads=2 qos=false ! video/x-raw,format=BGRA ! "
                f"appsink name=preview emit-signals=true max-buffers=1 drop=true sync=false"
            )

        # Camera source — platform-aware
        camera_src = get_gst_camera_caps(
            self._source_device, self._width, self._height, self._fps,
            capture_format=self._capture_format,
        )

        decoder = self._select_decoder(effects_mode=False)

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
            self._source_device, self._width, self._height, self._fps,
            capture_format=self._capture_format,
        )

        fresh_queue = (
            "queue max-size-buffers=1 max-size-bytes=0 "
            "max-size-time=0 leaky=downstream"
        )

        self._gpu_capture_active = self._gpu_path_enabled() and not IS_MACOS
        self._gpu_jpeg_active = (
            self._gpu_capture_active
            and self._capture_format != "raw"
            and not self._gpu_jpeg_demoted
            and getattr(self._frame_processor, "supports_jpeg", False)
        )
        if self._gpu_jpeg_active:
            # nvImageCodec decodes the MJPEG frames on the GPU: only the
            # compressed bytes (~100-300KB) ever cross into Python.
            self._pipeline = Gst.parse_launch(
                f"{camera_src} ! "
                f"{fresh_queue} ! "
                f"appsink name=sink"
            )
            sink = self._pipeline.get_by_name("sink")
            sink.set_property("caps", Gst.Caps.from_string(
                f"image/jpeg,width={self._width},height={self._height}"
            ))
        elif self._gpu_capture_active:
            # Device-resident path: hand the appsink the decoder's NATIVE
            # format (jpegdec emits I420 — 1.4MB vs 3.7MB BGRA) and do all
            # colorspace conversion on the GPU. No convert element at all.
            if self._capture_format == "raw":
                decoder = "identity"
            else:
                decoder = "jpegdec"
            self._pipeline = Gst.parse_launch(
                f"{camera_src} ! "
                f"{fresh_queue} ! "
                f"{decoder} ! "
                f"{fresh_queue} ! "
                f"appsink name=sink"
            )
            sink = self._pipeline.get_by_name("sink")
            sink.set_property("caps", Gst.Caps.from_string(
                "video/x-raw,format=(string){ I420, NV12, YUY2, BGRA },"
                f"width={self._width},height={self._height}"
            ))
        else:
            decoder = self._select_decoder(effects_mode=True)
            # No videorate — frame throttling is done in Python
            # (_on_effects_sample) so mode/profile changes never require a
            # pipeline restart. The leaky queues keep the capture path pinned
            # to the newest frame if Python or face effects fall behind.
            self._pipeline = Gst.parse_launch(
                f"{camera_src} ! "
                f"{fresh_queue} ! "
                f"{decoder} ! "
                f"video/x-raw,format=BGRA,width={self._width},height={self._height} ! "
                f"{fresh_queue} ! "
                f"appsink name=sink"
            )
            sink = self._pipeline.get_by_name("sink")
        sink.set_property("emit-signals", True)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.set_property("sync", False)
        sink.set_property("qos", True)
        sink.set_property("processing-deadline", 0)
        sink.set_property("max-lateness", 0)
        sink.set_property("enable-last-sample", False)
        if self._gpu_capture_active:
            sink.connect("new-sample", self._on_effects_sample_gpu)
        else:
            sink.connect("new-sample", self._on_effects_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)

        if vcam_enabled:
            if IS_MACOS:
                # macOS: CoreMediaIO frame bridge publishes the branded device.
                # OBS fallback is opt-in only because it exposes a different
                # camera name in meeting apps.
                self._vcam_pipeline = None
                self._vcam_appsrc = None
                self._frame_bridge = None
                self._pyvirtualcam = None
                try:
                    from macos.NVBroadcastHelper.frame_bridge import FrameBridge
                    self._frame_bridge = FrameBridge(
                        width=self._width, height=self._height
                    )
                    print(f"[NV Broadcast] macOS virtual camera: {VIRTUAL_CAM_LABEL}")
                except Exception:
                    if os.getenv("NVBROADCAST_ALLOW_OBS_VCAM_FALLBACK") == "1":
                        try:
                            import pyvirtualcam
                            self._pyvirtualcam = pyvirtualcam.Camera(
                                width=self._width, height=self._height,
                                fps=self._fps, backend="obs",
                            )
                            print(f"[NV Broadcast] macOS fallback virtual camera: {self._pyvirtualcam.device}")
                        except Exception as e:
                            print(f"[NV Broadcast] macOS virtual camera not available: {e}")
                    else:
                        print(
                            "[NV Broadcast] macOS NVbroadcast virtual camera extension "
                            "not available; OBS fallback disabled"
                        )
            else:
                self._pyvirtualcam = None
                self._build_vcam_pipeline()

    def _build_vcam_pipeline(self) -> None:
        """Build the appsrc → v4l2sink virtual-camera pipeline."""
        if self._gpu_capture_active:
            # The frame processor already delivers output-format YUY2 —
            # feed the sink directly, no convert element, half the bytes.
            # PAR and colorimetry must be explicit: v4l2sink accept-caps is a
            # SUBSET check against device caps that pin both fields (a convert
            # element used to fixate them; without one, omitting a field means
            # "any" and negotiation fails). 2:4:7:1 is the BT.601 colorimetry
            # the conversion kernel produces and the legacy leg negotiated.
            self._vcam_used_cuda = False
            self._vcam_pipeline = Gst.parse_launch(
                f"appsrc name=src is-live=true format=time "
                f"caps=video/x-raw,format=YUY2,width={self._width},"
                f"height={self._height},framerate={self._fps}/1,"
                f"pixel-aspect-ratio=1/1,colorimetry=2:4:7:1 ! "
                f"queue max-size-buffers=1 max-size-bytes=0 "
                f"max-size-time=0 leaky=downstream ! "
                f"{self._v4l2sink_segment()}"
            )
            appsrc_bytes = self._width * self._height * 2
        else:
            convert = self._convert_segment(self._output_format)
            self._vcam_used_cuda = "cudaconvert" in convert
            self._vcam_pipeline = Gst.parse_launch(
                f"appsrc name=src is-live=true format=time "
                f"caps=video/x-raw,format=BGRA,width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"queue max-size-buffers=1 max-size-bytes=0 "
                f"max-size-time=0 leaky=downstream ! "
                f"{convert} ! "
                f"video/x-raw,format={self._output_format},width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"{self._v4l2sink_segment()}"
            )
            appsrc_bytes = self._width * self._height * 4
        self._vcam_appsrc = self._vcam_pipeline.get_by_name("src")
        if self._vcam_appsrc:
            self._vcam_appsrc.set_property("max-buffers", 1)
            self._vcam_appsrc.set_property("max-bytes", appsrc_bytes)
            self._vcam_appsrc.set_property("leaky-type", 2)
            self._vcam_appsrc.set_property("block", False)

        vbus = self._vcam_pipeline.get_bus()
        vbus.add_signal_watch()
        vbus.connect("message::error", self._on_vcam_error)

    def _on_preview_sample(self, appsink):
        """Lightweight preview-only callback (passthrough mode)."""
        with self._callback_lock:
            self._callbacks_in_flight += 1
        if not self._running:
            with self._callback_lock:
                self._callbacks_in_flight -= 1
            return Gst.FlowReturn.EOS
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK

            # Paused: keep draining the appsink but don't update the frame
            if self._paused:
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
        finally:
            with self._callback_lock:
                self._callbacks_in_flight -= 1

    def set_effects_fps(self, efps: int):
        """Change effects throttle at runtime — no pipeline restart needed."""
        self._effects_fps = min(efps, self._fps)

    def _on_effects_sample(self, appsink):
        """Capture callback — composites EVERY frame inline for zero edge lag.

        Architecture:
        - Background thread: runs heavy AI inference to update the alpha mask
        - Inline (this callback): composites the CURRENT frame with latest alpha
          + applies face effects, beautify, mirror (all lightweight)

        Result: edges always match the current frame position. The alpha mask
        may be 1-2 frames old but compositing is always fresh.
        """
        with self._callback_lock:
            self._callbacks_in_flight += 1
        if not self._running:
            with self._callback_lock:
                self._callbacks_in_flight -= 1
            return Gst.FlowReturn.EOS
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK

            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.OK

            frame_data = bytes(info.data)
            # Copy timing info before unmapping — buf may be recycled after unmap
            buf_pts = buf.pts
            buf_duration = buf.duration
            buf.unmap(info)

            expected = self._width * self._height * 4
            if len(frame_data) != expected:
                return Gst.FlowReturn.OK

            self._frame_count += 1

            # Paused: push frozen frame, skip all processing
            if self._paused and self._frozen_frame:
                output = self._frozen_frame
                appsrc = self._vcam_appsrc
                if self._vcam_enabled and self._running and appsrc:
                    vcam_buf = Gst.Buffer.new_allocate(None, len(output), None)
                    vcam_buf.fill(0, output)
                    vcam_buf.pts = buf_pts
                    vcam_buf.duration = buf_duration
                    appsrc.emit("push-buffer", vcam_buf)
                return Gst.FlowReturn.OK

            # Background: kick off alpha inference (heavy, non-blocking)
            if self._alpha_worker_enabled and self._alpha_callback:
                if self._inference_due():
                    self._submit_alpha_frame(frame_data, self._width, self._height)

            # Inline: composite current frame with latest alpha + all light effects
            if self._effect_callback:
                try:
                    output = self._effect_callback(frame_data, self._width, self._height)
                    if output is None or len(output) != expected:
                        output = frame_data
                except Exception as e:
                    if self._frame_count <= 5:
                        print(f"[NV Broadcast] Effects error: {e}")
                    output = frame_data
            else:
                output = frame_data

            # Store for preview
            with self._lock:
                self._latest_frame = output

            # Push to vcam at full fps
            appsrc = self._vcam_appsrc
            frame_bridge = getattr(self, '_frame_bridge', None)
            pyvirtualcam = getattr(self, '_pyvirtualcam', None)
            if self._vcam_enabled and self._running:
                if appsrc:
                    vcam_buf = Gst.Buffer.new_allocate(None, len(output), None)
                    vcam_buf.fill(0, output)
                    vcam_buf.pts = buf_pts
                    vcam_buf.duration = buf_duration
                    appsrc.emit("push-buffer", vcam_buf)
                elif frame_bridge:
                    frame_bridge.write_frame(output)
                elif pyvirtualcam:
                    import numpy as np
                    frame = np.frombuffer(output, dtype=np.uint8).reshape(
                        self._height, self._width, 4
                    )
                    pyvirtualcam.send(frame[:, :, :3])

            # Push to recording if active
            rec_appsrc = self._rec_appsrc
            if self._recording and rec_appsrc:
                rec_buf = Gst.Buffer.new_allocate(None, len(output), None)
                rec_buf.fill(0, output)
                rec_buf.pts = buf_pts
                rec_appsrc.emit("push-buffer", rec_buf)

            return Gst.FlowReturn.OK
        finally:
            with self._callback_lock:
                self._callbacks_in_flight -= 1

    def _on_effects_sample_gpu(self, appsink):
        """Device-resident capture callback (GpuFramePath).

        Per frame: one small native-format ingest copy, one H2D, effects on
        the GPU, one YUY2 D2H directly into the outgoing Gst buffer. BGRA
        only materializes on frames where preview/recording/CPU stages ask.
        """
        with self._callback_lock:
            self._callbacks_in_flight += 1
        if not self._running:
            with self._callback_lock:
                self._callbacks_in_flight -= 1
            return Gst.FlowReturn.EOS
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            buf = sample.get_buffer()
            buf_pts = buf.pts
            buf_duration = buf.duration

            # Paused: replay the frozen output without touching the GPU.
            if self._paused and self._frozen_yuy2 is not None:
                self._push_yuy2_bytes(self._frozen_yuy2, buf_pts, buf_duration)
                return Gst.FlowReturn.OK

            try:
                proc = self._frame_processor
                s = sample.get_caps().get_structure(0)
                is_jpeg = s.get_name() == "image/jpeg"
                fmt = "JPEG" if is_jpeg else s.get_string("format")
                width = s.get_value("width")
                height = s.get_value("height")
                if not proc.configure(width, height, fmt):
                    raise RuntimeError(
                        f"appsink negotiated unsupported {fmt} {width}x{height}")
                self._gpu_frame_size = (width, height)

                ok, info = buf.map(Gst.MapFlags.READ)
                if not ok:
                    return Gst.FlowReturn.OK
                try:
                    if is_jpeg:
                        proc.ingest_jpeg(info.data)
                    else:
                        proc.ingest(info.data)
                finally:
                    buf.unmap(info)

                self._frame_count += 1
                if self._frame_plan is not None:
                    gpu_pure, mirror, inline = self._frame_plan()
                else:
                    gpu_pure, mirror, inline = True, False, True
                inference_due = self._inference_due()

                if gpu_pure:
                    proc.composite(mirror=mirror, inline_inference=inline)
                    if (not inline and self._alpha_worker_enabled
                            and self._alpha_callback and inference_due):
                        src = proc.download_bgra(source=True)
                        self._submit_alpha_frame(src.tobytes(), width, height)
                else:
                    # Mixed mode: CPU stages run through the legacy bytes
                    # callback on the GPU-converted source frame.
                    frame_bytes = proc.download_bgra(source=True).tobytes()
                    if (self._alpha_worker_enabled and self._alpha_callback
                            and inference_due):
                        self._submit_alpha_frame(frame_bytes, width, height)
                    output = None
                    if self._effect_callback:
                        try:
                            output = self._effect_callback(
                                frame_bytes, width, height)
                            if output is not None and \
                                    len(output) != width * height * 4:
                                output = None
                        except Exception as e:
                            if self._frame_count <= 5:
                                print(f"[NV Broadcast] Effects error: {e}")
                            output = None
                    if output is not None and output is not frame_bytes:
                        proc.set_output_bgra(output)

                # BGRA copy only for consumers that actually need one.
                if (self._preview_enabled and self._preview_callback) \
                        or (self._recording and self._rec_appsrc):
                    out_bgra = proc.download_bgra()
                    with self._lock:
                        self._latest_frame = out_bgra
                    if self._recording and self._rec_appsrc:
                        rec_bytes = out_bgra.tobytes()
                        rec_buf = Gst.Buffer.new_allocate(
                            None, len(rec_bytes), None)
                        rec_buf.fill(0, rec_bytes)
                        rec_buf.pts = buf_pts
                        self._rec_appsrc.emit("push-buffer", rec_buf)

                appsrc = self._vcam_appsrc
                if self._vcam_enabled and self._running and appsrc:
                    # PyGObject maps buffers as a bytes COPY, so the pinned
                    # download + C-side fill() is the only correct way to
                    # get the frame into the outgoing buffer.
                    payload = proc.download_yuy2().tobytes()
                    if self._paused and self._frozen_yuy2 is None:
                        self._frozen_yuy2 = payload
                    vcam_buf = Gst.Buffer.new_allocate(None, len(payload), None)
                    vcam_buf.fill(0, payload)
                    vcam_buf.pts = buf_pts
                    vcam_buf.duration = buf_duration
                    appsrc.emit("push-buffer", vcam_buf)

                self._gpu_path_errors = 0
                return Gst.FlowReturn.OK
            except Exception as e:
                self._gpu_path_errors += 1
                print(f"[NV Broadcast] GPU frame path error "
                      f"({self._gpu_path_errors}/3): {e}", flush=True)
                if self._gpu_path_errors >= 3:
                    self._gpu_path_errors = 0
                    if self._gpu_jpeg_active and not self._gpu_jpeg_demoted:
                        # Stage 1: give up only on GPU JPEG decode and
                        # rebuild with jpegdec feeding the same GPU path.
                        self._gpu_jpeg_demoted = True
                        print("[NV Broadcast] Demoting GPU JPEG decode to "
                              "jpegdec", flush=True)
                        GLib.idle_add(self._queue_rebuild)
                    elif not self._gpu_path_demoted:
                        # Stage 2: one-shot demotion, mirroring
                        # _cuda_convert_demoted — legacy BGRA path for good.
                        self._gpu_path_demoted = True
                        print("[NV Broadcast] Demoting to CPU frame path",
                              flush=True)
                        GLib.idle_add(self._queue_rebuild)
                return Gst.FlowReturn.OK
        finally:
            with self._callback_lock:
                self._callbacks_in_flight -= 1

    def _push_yuy2_bytes(self, payload: bytes, pts, duration) -> None:
        appsrc = self._vcam_appsrc
        if not (self._vcam_enabled and self._running and appsrc):
            return
        vcam_buf = Gst.Buffer.new_allocate(None, len(payload), None)
        vcam_buf.fill(0, payload)
        vcam_buf.pts = pts
        vcam_buf.duration = duration
        appsrc.emit("push-buffer", vcam_buf)

    def _update_alpha_bg(self, frame_data, width, height):
        """Run alpha inference in background — never blocks the capture."""
        if self._alpha_callback:
            self._alpha_callback(frame_data, width, height)

    def start_recording(self, filepath: str):
        """Start recording the processed output to an MP4 file."""
        if self._recording:
            return

        # Use NVENC if available, else x264
        from nvbroadcast.core.platform import IS_MACOS
        if not IS_MACOS and self._has_gst_element("nvh264enc"):
            if self._cuda_convert_available() and not self._cuda_convert_demoted:
                # Convert on-GPU and hand NVENC CUDA memory directly —
                # no download on the recording leg at all.
                convert = ("cudaupload ! cudaconvert ! "
                           "video/x-raw(memory:CUDAMemory),format=NV12")
            else:
                convert = "videoconvert n-threads=2 qos=false"
            encoder = f"{convert} ! nvh264enc preset=low-latency-hq bitrate=8000"
        else:
            encoder = ("videoconvert n-threads=2 qos=false ! "
                       "x264enc tune=zerolatency speed-preset=ultrafast bitrate=8000")

        try:
            # Video + Audio recording pipeline
            self._recording_pipeline = Gst.parse_launch(
                f"mp4mux name=mux fragment-duration=1000 ! filesink location={filepath} "
                f"appsrc name=recsrc is-live=true format=time "
                f"caps=video/x-raw,format=BGRA,width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"queue max-size-buffers=3 leaky=downstream ! "
                f"{encoder} ! "
                f"h264parse ! mux.video_0 "
                f"pipewiresrc ! audioconvert ! audioresample ! "
                f"audio/x-raw,format=S16LE,rate=48000,channels=1 ! "
                f"queue max-size-buffers=10 ! "
                f"avenc_aac bitrate=128000 ! aacparse ! mux.audio_0"
            )
        except Exception:
            # Fallback: video only
            self._recording_pipeline = Gst.parse_launch(
                f"appsrc name=recsrc is-live=true format=time "
                f"caps=video/x-raw,format=BGRA,width={self._width},"
                f"height={self._height},framerate={self._fps}/1 ! "
                f"queue max-size-buffers=3 leaky=downstream ! "
                f"{encoder} ! "
                f"h264parse ! "
                f"mp4mux fragment-duration=1000 ! "
                f"filesink location={filepath}"
            )
            print("[NV Broadcast] Recording without audio (audio unavailable)")
        self._rec_appsrc = self._recording_pipeline.get_by_name("recsrc")
        self._recording_pipeline.set_state(Gst.State.PLAYING)
        self._recording = True
        print(f"[NV Broadcast] Recording started: {filepath}")

    def stop_recording(self):
        """Stop recording and finalize the MP4 file."""
        if not self._recording:
            return
        self._recording = False
        if self._rec_appsrc:
            self._rec_appsrc.emit("end-of-stream")
        # Wait for EOS to propagate
        if self._recording_pipeline:
            self._recording_pipeline.get_bus().timed_pop_filtered(
                2 * Gst.SECOND, Gst.MessageType.EOS
            )
            self._recording_pipeline.set_state(Gst.State.NULL)
        self._recording_pipeline = None
        self._rec_appsrc = None
        print("[NV Broadcast] Recording stopped")

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _tick_preview(self) -> bool:
        if self._pipeline is None:
            self._preview_timer_id = 0
            return False

        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None

        if frame is None or self._preview_callback is None:
            return True
        if not self._preview_enabled:
            # Window hidden: skip the full-frame copy + texture upload.
            return True

        # The GPU frame path stores a numpy BGRA array; the legacy path bytes.
        if not isinstance(frame, (bytes, bytearray)):
            try:
                frame = frame.tobytes()
            except AttributeError:
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
        """Rebuild pipeline after mode change once devices are actually free."""
        if not self._rebuild_pending:
            self._rebuild_source_id = 0
            return False

        if self._pipeline is not None:
            self.stop(clear_rebuild_request=False)
            return True

        if not self._teardown_done:
            return True

        self._rebuild_source_id = 0
        try:
            self.build(vcam_enabled=self._vcam_enabled)
            self.start()
        finally:
            self._rebuild_pending = False
        return False

    def start(self):
        # Start vcam first — if it fails, disable it but keep streaming
        self._vcam_failed = False
        if self._vcam_pipeline:
            self._start_vcam_with_retry()
        if self._pipeline:
            self._pipeline.set_state(Gst.State.PLAYING)
        self._running = True

    def stop(self, clear_rebuild_request: bool = True):
        if clear_rebuild_request:
            self._cancel_rebuild()
        self._stop_preview_timer()
        self._capture_idle = False
        with self._teardown_lock:
            if not self._teardown_done:
                return
            self._running = False
            with self._alpha_condition:
                self._alpha_pending = None
                self._alpha_condition.notify_all()

            cap = self._pipeline
            vcam = self._vcam_pipeline
            self._pipeline = None
            self._vcam_pipeline = None
            self._vcam_appsrc = None
            self._latest_frame = None
            self._pending_frame = None
            self._frozen_frame = None
            self._teardown_capture = cap
            self._teardown_vcam = vcam

            self._teardown_done = False
            if self._teardown_source_id:
                GLib.source_remove(self._teardown_source_id)
            self._teardown_source_id = GLib.timeout_add(
                10, self._poll_teardown, priority=GLib.PRIORITY_HIGH
            )

    def shutdown_sync(self, timeout_seconds: float = 3.0):
        """Synchronous teardown for app shutdown.

        The normal `stop()` path is deferred to keep the UI responsive. During
        application shutdown we want the opposite: drain callbacks deterministically
        and release native resources before Python/GTK starts finalizing.
        """
        self._cancel_rebuild()
        self._stop_preview_timer()
        with self._teardown_lock:
            self._running = False
            if self._teardown_source_id:
                GLib.source_remove(self._teardown_source_id)
                self._teardown_source_id = 0

            cap = self._pipeline if self._pipeline is not None else self._teardown_capture
            vcam = self._vcam_pipeline if self._vcam_pipeline is not None else self._teardown_vcam

            self._pipeline = None
            self._vcam_pipeline = None
            self._vcam_appsrc = None
            self._latest_frame = None
            self._pending_frame = None
            self._frozen_frame = None
            self._teardown_capture = None
            self._teardown_vcam = None
            self._teardown_done = False

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self._callback_lock:
                callbacks_busy = self._callbacks_in_flight
            if callbacks_busy <= 0 and not self._effects_busy:
                break
            time.sleep(0.01)

        try:
            if self._recording:
                self.stop_recording()

            if getattr(self, '_frame_bridge', None):
                self._frame_bridge.close()
                self._frame_bridge = None
            if getattr(self, '_pyvirtualcam', None):
                self._pyvirtualcam.close()
                self._pyvirtualcam = None

            if vcam and not self._vcam_failed:
                vcam.set_state(Gst.State.NULL)
            if cap:
                cap.set_state(Gst.State.NULL)
        finally:
            self._stop_alpha_worker(timeout_seconds=max(0.0, deadline - time.monotonic()))
            self._teardown_done = True

    def _poll_teardown(self):
        with self._callback_lock:
            callbacks_busy = self._callbacks_in_flight
        if callbacks_busy > 0 or self._effects_busy:
            return True

        cap = self._teardown_capture
        vcam = self._teardown_vcam
        self._teardown_capture = None
        self._teardown_vcam = None
        self._teardown_source_id = 0

        try:
            if self._recording:
                self.stop_recording()

            # Clean up macOS vcam backends first so they stop holding output
            # resources before the GStreamer side is released.
            if getattr(self, '_frame_bridge', None):
                self._frame_bridge.close()
                self._frame_bridge = None
            if getattr(self, '_pyvirtualcam', None):
                self._pyvirtualcam.close()
                self._pyvirtualcam = None

            if cap:
                cap.set_state(Gst.State.NULL)

            if vcam and not self._vcam_failed:
                vcam.set_state(Gst.State.NULL)
        finally:
            self._teardown_done = True
        return False

    @staticmethod
    def _has_gst_element(name: str) -> bool:
        """Check if a GStreamer element is available."""
        factory = Gst.ElementFactory.find(name)
        return factory is not None

    @classmethod
    def _cuda_convert_available(cls) -> bool:
        """One-time probe: does cudaupload!cudaconvert!cudadownload work?

        Element registration alone is not enough — CUDA context creation or
        caps negotiation can still fail at runtime, so push real buffers once.
        """
        if cls._cuda_convert_probe_result is not None:
            return cls._cuda_convert_probe_result
        if os.getenv("NVBROADCAST_NO_CUDA_CONVERT") == "1":
            cls._cuda_convert_probe_result = False
            return False
        if not all(cls._has_gst_element(e)
                   for e in ("cudaupload", "cudaconvert", "cudadownload")):
            cls._cuda_convert_probe_result = False
            return False
        ok = False
        pipe = None
        try:
            pipe = Gst.parse_launch(
                "videotestsrc num-buffers=2 ! "
                "video/x-raw,format=BGRA,width=320,height=240 ! "
                "cudaupload ! cudaconvert ! cudadownload ! "
                "video/x-raw,format=YUY2 ! fakesink"
            )
            if pipe.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE:
                msg = pipe.get_bus().timed_pop_filtered(
                    2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                ok = bool(msg) and msg.type == Gst.MessageType.EOS
        except Exception:
            ok = False
        finally:
            if pipe is not None:
                try:
                    pipe.set_state(Gst.State.NULL)
                except Exception:
                    pass
        cls._cuda_convert_probe_result = ok
        print(
            "[NV Broadcast] CUDA convert offload: "
            f"{'enabled' if ok else 'unavailable, using CPU videoconvert'}",
            flush=True,
        )
        return ok

    def _convert_segment(self, out_format: str) -> str:
        """Colorspace-conversion segment — GPU when the CUDA probe passed."""
        if (not self._cuda_convert_demoted
                and out_format in _CUDA_SAFE_FORMATS
                and self._cuda_convert_available()):
            return "cudaupload ! cudaconvert ! cudadownload"
        return "videoconvert n-threads=2 qos=false"

    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"[NV Broadcast] Capture error: {err.message}")
        if debug:
            print(f"[NV Broadcast] Debug: {debug}")
        # GPU-path capture failures (e.g. caps negotiation) fall back to the
        # legacy BGRA pipeline instead of leaving the camera dead.
        if self._gpu_capture_active and not self._gpu_path_demoted:
            self._gpu_path_demoted = True
            print("[NV Broadcast] Demoting to CPU frame path after "
                  "capture pipeline error", flush=True)
            GLib.idle_add(self._queue_rebuild)

    def _on_vcam_error(self, bus, msg):
        err, debug = msg.parse_error()
        self._vcam_appsrc = None
        if self._vcam_pipeline is not None:
            try:
                self._vcam_pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._vcam_pipeline = None
        print(f"[NV Broadcast] VCam error: {err.message}")
        print(f"[NV Broadcast] VCam state: {self._describe_vcam_state()}")
        if debug:
            print(f"[NV Broadcast] Debug: {debug}")
        # A GPU-path vcam failure demotes the whole device-resident path —
        # the YUY2 appsrc leg only exists when that path feeds it.
        if self._gpu_capture_active and not self._gpu_path_demoted:
            self._gpu_path_demoted = True
            print("[NV Broadcast] Demoting to CPU frame path after "
                  "vcam pipeline error", flush=True)
            GLib.idle_add(self._queue_rebuild)
            return
        # A CUDA hiccup must not cost the virtual camera: demote to the CPU
        # convert path once and rebuild instead of declaring failure.
        if self._vcam_used_cuda and not self._cuda_convert_demoted:
            self._cuda_convert_demoted = True
            print("[NV Broadcast] Demoting vcam to CPU videoconvert after "
                  "CUDA pipeline error", flush=True)
            GLib.idle_add(self._rebuild_vcam_after_demotion)
            return
        self._vcam_failed = True

    def _rebuild_vcam_after_demotion(self) -> bool:
        """Rebuild the vcam pipeline on the CPU path after a CUDA demotion."""
        try:
            self._build_vcam_pipeline()
            if not self._start_vcam_with_retry():
                self._vcam_failed = True
        except Exception as e:
            print(f"[NV Broadcast] VCam CPU rebuild failed: {e}", flush=True)
            self._vcam_failed = True
        return False

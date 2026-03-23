# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""NVIDIA Broadcast - setup once and forget.

Auto-starts broadcast on launch, restores all saved settings,
minimizes to background on close. Browser picks up virtual camera automatically.
"""

import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gst", "1.0")
from gi.repository import Gtk, Adw, Gst, Gio, Gdk, GLib

from nvbroadcast.core.constants import APP_ID, COMPUTE_GPU_INDEX
from nvbroadcast.core.config import load_config, save_config
from nvbroadcast.video.pipeline import VideoPipeline
from nvbroadcast.video.effects import VideoEffects
from nvbroadcast.video.autoframe import AutoFrame
from nvbroadcast.video.beautify import FaceBeautifier
from nvbroadcast.video.virtual_camera import ensure_virtual_camera
from nvbroadcast.audio.pipeline import AudioPipeline
from nvbroadcast.audio.monitor import SpeakerMonitor
from nvbroadcast.ui.window import NVBroadcastWindow


class NVBroadcastApp(Adw.Application):
    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        self.config = load_config()
        self._window = None
        self._video_pipeline = None
        self._audio_pipeline = None
        self._speaker_monitor = None
        self._video_effects = VideoEffects(
            gpu_index=self.config.compute_gpu,
            edge_config=self.config.video.edge,
            compositing=self.config.compositing,
        )
        self._autoframe = AutoFrame(gpu_index=self.config.compute_gpu)
        self._beautifier = FaceBeautifier(compositing=self.config.compositing)
        self._vcam_device = None
        self._vcam_available = False
        self._mirror = True  # Default: mirror (like looking in a mirror)
        self._tray = None
        self._vcam_consumers = 0  # Track virtual camera consumers
        self._camera_power_save = True  # Stop camera when no consumers
        self._streaming = False

    def do_startup(self):
        Adw.Application.do_startup(self)
        Gst.init(None)

        # Load CSS
        css_provider = Gtk.CssProvider()
        css_path = Path(__file__).parent / "ui" / "style.css"
        if css_path.exists():
            css_provider.load_from_path(str(css_path))
            display = Gdk.Display.get_default()
            if display:
                Gtk.StyleContext.add_provider_for_display(
                    display, css_provider,
                    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
                )

        try:
            self._vcam_device = ensure_virtual_camera()
            self._vcam_available = True
        except RuntimeError as e:
            print(f"[NV Broadcast] Virtual camera unavailable: {e}")

    def do_activate(self):
        if self._window is None:
            self._window = NVBroadcastWindow(self)

            # System tray icon
            try:
                from nvbroadcast.ui.tray import TrayIcon
                self._tray = TrayIcon(self)
                if self._tray.available:
                    print("[NV Broadcast] System tray icon active")
            except Exception as e:
                print(f"[NV Broadcast] Tray icon not available: {e}")

            # Camera power save: poll for vcam consumers
            GLib.timeout_add(5000, self._check_vcam_consumers)

            # Intercept window close -> minimize to background instead of quit
            self._window.connect("close-request", self._on_close_request)

            # Restore saved settings to UI
            self._restore_settings()

            # First-run setup wizard
            if self.config.first_run:
                from nvbroadcast.ui.setup_wizard import SetupWizard
                wizard = SetupWizard(self._window)
                wizard.connect("setup-complete", self._on_setup_complete)
                wizard.present()
            elif self.config.auto_start and self._vcam_available:
                GLib.idle_add(self._auto_start)

        self._window.present()

    def _on_setup_complete(self, wizard, profile_name, gpu_index, compositing):
        """Called when first-run wizard finishes."""
        from nvbroadcast.core.config import apply_performance_profile, PERFORMANCE_PROFILES
        # Apply profile
        apply_performance_profile(self.config, profile_name)
        self.config.compute_gpu = gpu_index
        self.config.compositing = compositing
        self.config.first_run = False

        # Apply to effects engine
        self._video_effects._gpu_index = gpu_index
        self._video_effects._apply_edge_config(self.config.video.edge)
        self._video_effects.set_compositing(compositing)
        self._beautifier.set_compositing(compositing)
        profile = PERFORMANCE_PROFILES[profile_name]
        self._video_effects._skip_interval = profile["skip_interval"]

        save_config(self.config)
        print(f"[NV Broadcast] Profile: {profile['label']} | GPU: {gpu_index} | Compositing: {compositing}")

        # Rebuild mode dropdown with updated backends (e.g. CuPy just installed)
        self._window.rebuild_mode_selector(compositing, profile_name)
        if hasattr(self._window, '_gpu_selector') and self._window._gpu_selector:
            self._window._gpu_selector.set_selected_index(gpu_index)

        # Update edge tuning sliders
        self._window._edge_dilate._scale.set_value(self.config.video.edge.dilate_size)
        self._window._edge_blur._scale.set_value(self.config.video.edge.blur_size)
        self._window._edge_strength._scale.set_value(self.config.video.edge.sigmoid_strength)
        self._window._edge_midpoint._scale.set_value(self.config.video.edge.sigmoid_midpoint)

        self._window.set_status(f"Setup complete: {profile['label']} | {compositing} compositing")

        # Now auto-start
        if self.config.auto_start and self._vcam_available:
            GLib.idle_add(self._auto_start)

    def _on_close_request(self, window):
        """Minimize to tray instead of quitting."""
        if self.config.minimize_on_close:
            window.set_visible(False)
            status = "streaming" if self._streaming else "idle"
            if self._tray and self._tray.available:
                self._tray.update_status(self._streaming, status)
                print(f"[NV Broadcast] Minimized to tray ({status})")
            else:
                print(f"[NV Broadcast] Minimized to background ({status})")
            return True  # Prevent destruction
        return False  # Allow normal close

    def _check_vcam_consumers(self):
        """Poll virtual camera device for active consumers.

        When no app is reading from /dev/video10, pause the camera to save
        power/CPU/GPU. Resume automatically when a consumer connects.
        """
        if not self._camera_power_save or not self._vcam_available:
            return True  # Keep polling

        import subprocess
        try:
            # Count processes reading from the vcam device (exclude our own)
            result = subprocess.run(
                ["fuser", self._vcam_device or "/dev/video10"],
                capture_output=True, text=True, timeout=2,
            )
            pids = result.stdout.strip().split()
            import os
            own_pid = str(os.getpid())
            consumers = [p for p in pids if p.strip() and p.strip() != own_pid]
            new_count = len(consumers)
        except Exception:
            new_count = self._vcam_consumers  # Keep current state on error

        if new_count != self._vcam_consumers:
            old = self._vcam_consumers
            self._vcam_consumers = new_count

            if new_count > 0 and old == 0:
                # Consumer connected — resume camera if not streaming
                if not self._streaming and self.config.auto_start:
                    print(f"[NV Broadcast] Consumer detected — starting camera")
                    cam = self.config.video.camera_device
                    fmt = self.config.video.output_format
                    self._do_start_pipeline(cam, fmt)
                    if self._window:
                        self._window._streaming = True
                        self._window._stream_btn.set_label("Stop Broadcast")
                        self._window._stream_btn.remove_css_class("suggested-action")
                        self._window._stream_btn.add_css_class("destructive-action")

            elif new_count == 0 and old > 0 and self._streaming:
                # All consumers gone — pause camera to save power
                print("[NV Broadcast] No consumers — pausing camera (power save)")
                self.stop_pipeline()
                if self._window:
                    self._window._streaming = False
                    self._window._stream_btn.set_label("Start Broadcast")
                    self._window._stream_btn.remove_css_class("destructive-action")
                    self._window._stream_btn.add_css_class("suggested-action")
                    self._window.set_status("Power save — waiting for consumer")

            if self._tray and self._tray.available:
                status = f"streaming ({new_count} consumer{'s' if new_count != 1 else ''})" if self._streaming else "idle"
                self._tray.update_status(self._streaming, status)

        return True  # Keep polling

    def _preload_effects(self):
        """Pre-initialize AI models in background to eliminate first-use delay."""
        import threading
        def _init():
            try:
                if self.config.video.background_removal:
                    self._video_effects.initialize()
            except Exception as e:
                print(f"[NV Broadcast] Background model preload failed: {e}")
        threading.Thread(target=_init, daemon=True).start()

    def _auto_start(self):
        """Auto-start broadcast with saved settings."""
        if not self._streaming:
            camera = self.config.video.camera_device
            fmt = self.config.video.output_format
            self.start_pipeline(camera, fmt)
            self._window._streaming = True
            self._window._stream_btn.set_label("Stop Broadcast")
            self._window._stream_btn.remove_css_class("suggested-action")
            self._window._stream_btn.add_css_class("destructive-action")
        return False  # Don't repeat

    def _restore_settings(self):
        """Restore all saved settings to the UI and effects."""
        c = self.config

        # Restore model and quality preset
        self._video_effects._model_type = c.video.model
        self._video_effects._quality = c.video.quality_preset

        # Restore background settings
        if c.video.background_image:
            self._video_effects.set_background_image(c.video.background_image)

        self._video_effects.mode = c.video.background_mode
        self._video_effects.intensity = c.video.blur_intensity

        # Restore autoframe
        self._autoframe.zoom_level = c.video.auto_frame_zoom

        # Tell window to restore UI controls
        self._window.restore_settings(c)

        if self._vcam_available:
            self._window.set_status(f"Ready - Virtual camera at {self._vcam_device}")
        else:
            self._window.set_status(
                "Virtual camera not available. Run: "
                'sudo modprobe v4l2loopback devices=1 video_nr=10 '
                'card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4')

    # --- Video Pipeline ---

    def start_pipeline(self, camera_device: str, output_format: str = "YUY2"):
        if self._streaming:
            # Stop first, then restart after delay (same as user clicking Stop then Start)
            self.stop_pipeline()
            if self._window:
                self._window._streaming = False
                self._window._stream_btn.set_label("Start Broadcast")
                self._window._stream_btn.remove_css_class("destructive-action")
                self._window._stream_btn.add_css_class("suggested-action")
                self._window.set_status("Restarting...")
            self._pending_start = (camera_device, output_format)
            GLib.timeout_add(2000, self._restart_after_stop)
        else:
            self._do_start_pipeline(camera_device, output_format)

    def _restart_after_stop(self):
        """Restart pipeline 2 seconds after stop — simple, no race conditions."""
        if hasattr(self, '_pending_start'):
            cam, fmt = self._pending_start
            del self._pending_start
            self._do_start_pipeline(cam, fmt)
            if self._streaming and self._window:
                self._window._streaming = True
                self._window._stream_btn.set_label("Stop Broadcast")
                self._window._stream_btn.remove_css_class("suggested-action")
                self._window._stream_btn.add_css_class("destructive-action")
        return False

    def _do_start_pipeline(self, camera_device: str, output_format: str = "YUY2"):
        from nvbroadcast.core.config import PERFORMANCE_PROFILES
        profile = PERFORMANCE_PROFILES.get(self.config.performance_profile, {})
        # Validate fps before building pipeline
        camera_fps = self._get_valid_fps(
            self.config.video.width, self.config.video.height, self.config.video.fps
        )
        if camera_fps != self.config.video.fps:
            self.config.video.fps = camera_fps
            save_config(self.config)
        effects_fps = max(5, int(profile.get("effects_ratio", 1.0) * camera_fps))

        self._video_pipeline = VideoPipeline()
        self._video_pipeline.configure(
            source_device=camera_device,
            vcam_device=self._vcam_device or "/dev/video10",
            width=self.config.video.width,
            height=self.config.video.height,
            fps=self.config.video.fps,
            output_format=output_format,
            effects_fps=effects_fps,
        )

        self._video_pipeline.set_effect_callback(self._process_frame)
        self._video_pipeline.set_preview_callback(
            lambda texture: self._window.update_preview(texture)
        )

        # Reset all resolution-dependent state BEFORE new pipeline processes frames
        self._video_effects._cached_alpha = None
        if self._video_effects._backend:
            self._video_effects._backend.reset_state()
        self._beautifier._face_mask = None
        self._beautifier._vignette_cache = None
        self._beautifier._face_bbox = None
        self._beautifier._face_center = None
        self._beautifier._prev_frame = None

        # Start in effects mode if effects were previously enabled
        if self._any_video_effects_active():
            self._video_pipeline._effects_active = True

        try:
            self._video_pipeline.build(vcam_enabled=self._vcam_available)
            self._video_pipeline.start()
            self._streaming = True

            w, h = self.config.video.width, self.config.video.height
            status = f"Streaming: {camera_device} {w}x{h}@{self.config.video.fps}fps"
            if self._vcam_available:
                status += f" -> {self._vcam_device}"
            self._window.set_status(status)
            self.config.video.camera_device = camera_device
            self.config.video.output_format = output_format
            save_config(self.config)

            if self._tray and self._tray.available:
                self._tray.update_status(True, status)

        except Exception as e:
            if self._video_pipeline:
                self._video_pipeline.stop()
                self._video_pipeline = None
            self._window.set_status(f"Pipeline error: {e}")
            print(f"[NV Broadcast] Pipeline failed: {e}")

        return False  # Don't repeat (for GLib.timeout_add)

    def stop_pipeline(self):
        if self._video_pipeline:
            self._video_pipeline.stop()
            self._video_pipeline = None
        self._streaming = False

    def _process_frame(self, frame_data: bytes, width: int, height: int) -> bytes:
        import cv2
        result = frame_data
        if self._video_effects.enabled:
            result = self._video_effects.process_frame(result, width, height)
        if self._beautifier.enabled:
            result = self._beautifier.process_frame(result, width, height)
        if self._autoframe.enabled:
            result = self._autoframe.process_frame(result, width, height)
        # Mirror flip (horizontal) — applied last so all effects process unflipped
        if self._mirror:
            import numpy as np
            frame = np.frombuffer(result, dtype=np.uint8).reshape(height, width, 4)
            result = cv2.flip(frame, 1).tobytes()
        return result

    def _any_video_effects_active(self) -> bool:
        return (self._video_effects.enabled or self._autoframe.enabled or
                self._beautifier.enabled)

    def _update_pipeline_mode(self):
        if self._video_pipeline:
            self._video_pipeline.set_effects_active(self._any_video_effects_active())

    # --- Effect Controls (save on every change) ---

    def set_bg_removal(self, enabled: bool):
        self._video_effects.enabled = enabled
        self.config.video.background_removal = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_bg_mode(self, mode: str):
        self._video_effects.mode = mode
        self.config.video.background_mode = mode
        save_config(self.config)

    def set_bg_image(self, path: str):
        if self._video_effects.set_background_image(path):
            self.config.video.background_image = path
            save_config(self.config)
            self._window.set_status(f"Background: {Path(path).name}")
        else:
            self._window.set_status("Failed to load background image")

    def set_blur_intensity(self, value: float):
        self._video_effects.intensity = value
        self.config.video.blur_intensity = value
        save_config(self.config)

    def set_performance_profile(self, profile_name: str, compositing: str | None = None,
                                use_tensorrt: bool = False, use_fused_kernel: bool = False,
                                use_nvdec: bool = False):
        """Switch performance profile. All changes apply live — no pipeline restart."""
        from nvbroadcast.core.config import apply_performance_profile, PERFORMANCE_PROFILES
        if profile_name not in PERFORMANCE_PROFILES:
            return

        # Apply compositing change
        if compositing and compositing != self.config.compositing:
            self.config.compositing = compositing
            self._video_effects.set_compositing(compositing)
            self._beautifier.set_compositing(compositing)

        # Apply engine mode (TensorRT / Fused CUDA kernel)
        self._video_effects.set_engine_mode(use_tensorrt, use_fused_kernel)

        # NVDEC: enable GPU JPEG decode in pipeline (Killer mode)
        self._use_nvdec = use_nvdec

        apply_performance_profile(self.config, profile_name)
        profile = PERFORMANCE_PROFILES[profile_name]

        # All settings apply immediately — no pipeline restart needed
        self._video_effects._skip_interval = profile["skip_interval"]
        self._video_effects._apply_edge_config(self.config.video.edge)

        # Compute effects_fps from ratio * camera fps
        effects_fps = max(5, int(profile["effects_ratio"] * self.config.video.fps))
        if self._video_pipeline:
            self._video_pipeline.set_effects_fps(effects_fps)

        save_config(self.config)

        b = self._video_effects._backend
        infer_h = b._MAX_INFER_HEIGHT if b else "?"
        print(f"[NV Broadcast] Mode: {profile_name} | infer={infer_h} skip={profile['skip_interval']} "
              f"fused={use_fused_kernel} comp={self.config.compositing} "
              f"efps={effects_fps}")

        if self._window:
            self._window.set_status(f"Mode: {profile['label']} | {infer_h}p")

    def set_compute_gpu(self, gpu_index: int):
        """Switch the GPU used for AI compute."""
        if gpu_index == self.config.compute_gpu:
            return
        self.config.compute_gpu = gpu_index
        self._video_effects._gpu_index = gpu_index
        # Reload the model on the new GPU
        if self._video_effects.available:
            self._video_effects._cleanup_backend()
            self._video_effects._cached_alpha = None
            self._video_effects.initialize()
        save_config(self.config)
        from nvbroadcast.core.gpu import detect_gpus
        gpus = detect_gpus()
        name = gpus[gpu_index].name if gpu_index < len(gpus) else f"GPU {gpu_index}"
        if self._window:
            self._window.set_status(f"Compute GPU: {name}")

    def set_model(self, model: str):
        """Switch segmentation model."""
        self.config.video.model = model
        self._video_effects.set_model(model)
        save_config(self.config)
        if self._window:
            self._window.set_status(f"Model: {model}")

    def set_quality(self, quality: str):
        self._video_effects.quality = quality
        self.config.video.quality_preset = quality
        save_config(self.config)

    def set_skip_interval(self, value: int):
        """Set how many frames to skip between inferences."""
        self._video_effects._skip_interval = max(1, value)

    def set_ema_weight(self, value: float):
        """Set temporal smoothing weight for single-frame models."""
        backend = self._video_effects._backend
        if backend and hasattr(backend, '_ema_weight'):
            backend._ema_weight = max(0.0, min(0.5, value))

    def set_mirror(self, enabled: bool):
        """Toggle mirror (horizontal flip) on preview and vcam output."""
        self._mirror = enabled

    def set_edge_refine(self, enabled: bool):
        """Toggle neural edge refinement for Zeus/Killer modes."""
        self._video_effects._edge_refine_enabled = enabled

    def set_edge_param(self, param: str, value: float):
        """Update a single edge refinement parameter."""
        setattr(self.config.video.edge, param, value)
        self._video_effects.update_edge_params(**{param: value})
        save_config(self.config)

    def _get_valid_fps(self, width: int, height: int, desired_fps: int) -> int:
        """Return the closest supported FPS for the given resolution."""
        from nvbroadcast.video.virtual_camera import list_camera_modes
        modes = list_camera_modes(self.config.video.camera_device)
        for mode in modes:
            if mode["width"] == width and mode["height"] == height:
                supported = mode["fps"]
                if desired_fps in supported:
                    return desired_fps
                # Pick the closest supported fps
                return min(supported, key=lambda f: abs(f - desired_fps))
        return desired_fps  # Unknown resolution — try anyway

    def set_resolution(self, width: int, height: int):
        """Change capture resolution — validates FPS and restarts pipeline."""
        if width == self.config.video.width and height == self.config.video.height:
            return
        self.config.video.width = width
        self.config.video.height = height

        # Clamp FPS to what the camera supports at the new resolution
        valid_fps = self._get_valid_fps(width, height, self.config.video.fps)
        if valid_fps != self.config.video.fps:
            self.config.video.fps = valid_fps
            print(f"[NV Broadcast] FPS clamped to {valid_fps} for {width}x{height}")

        save_config(self.config)

        # Caches are reset in _do_start_pipeline (after old pipeline stops)

        if self._streaming:
            camera = self.config.video.camera_device
            fmt = self.config.video.output_format
            self.start_pipeline(camera, fmt)

        if self._window:
            self._window.set_status(f"Resolution: {width}x{height} @ {self.config.video.fps}fps")

    def set_fps(self, fps: int):
        """Change camera FPS — validates against camera capabilities."""
        if fps == self.config.video.fps:
            return
        # Validate against camera capabilities
        valid_fps = self._get_valid_fps(
            self.config.video.width, self.config.video.height, fps
        )
        self.config.video.fps = valid_fps
        save_config(self.config)

        if self._streaming:
            camera = self.config.video.camera_device
            fmt = self.config.video.output_format
            self.start_pipeline(camera, fmt)

        if self._window:
            self._window.set_status(f"FPS: {valid_fps}")

    def set_autoframe(self, enabled: bool):
        self._autoframe.enabled = enabled
        self.config.video.auto_frame = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_autoframe_zoom(self, value: float):
        self._autoframe.zoom_level = value
        self.config.video.auto_frame_zoom = value
        save_config(self.config)

    # --- Beautification ---

    def set_beautify(self, enabled: bool):
        self._beautifier.enabled = enabled
        self._update_pipeline_mode()

    def set_beautify_param(self, param: str, value: float):
        """Set a beautification parameter (skin_smooth, denoise, edge_darken, enhance, sharpen)."""
        setattr(self._beautifier, param, value)

    # --- Audio ---

    def set_noise_removal(self, enabled: bool):
        self.config.audio.noise_removal = enabled
        if enabled:
            if self._audio_pipeline is None:
                self._audio_pipeline = AudioPipeline()
                self._audio_pipeline.configure(sample_rate=48000)
                self._audio_pipeline.build()
            self._audio_pipeline.effects.enabled = True
            self._audio_pipeline.start()
        else:
            if self._audio_pipeline:
                self._audio_pipeline.stop()
        save_config(self.config)

    def set_noise_intensity(self, value: float):
        self.config.audio.noise_intensity = value
        if self._audio_pipeline:
            self._audio_pipeline.effects.intensity = value
        save_config(self.config)

    def set_speaker_denoise(self, enabled: bool):
        self.config.audio.speaker_denoise = enabled
        if enabled:
            if self._speaker_monitor is None:
                self._speaker_monitor = SpeakerMonitor()
                self._speaker_monitor.build()
            self._speaker_monitor.effects.enabled = True
            self._speaker_monitor.start()
        else:
            if self._speaker_monitor:
                self._speaker_monitor.stop()
        save_config(self.config)

    # --- Lifecycle ---

    def do_shutdown(self):
        save_config(self.config)
        self.stop_pipeline()
        if self._audio_pipeline:
            self._audio_pipeline.stop()
        if self._speaker_monitor:
            self._speaker_monitor.stop()
        self._video_effects.cleanup()
        self._autoframe.cleanup()
        self._beautifier.cleanup()
        Adw.Application.do_shutdown(self)

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
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
        )
        self._autoframe = AutoFrame(gpu_index=self.config.compute_gpu)
        self._vcam_device = None
        self._vcam_available = False
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

            # Intercept window close -> minimize to background instead of quit
            self._window.connect("close-request", self._on_close_request)

            # Restore saved settings to UI
            self._restore_settings()

            # Auto-start broadcast (initializes model on first frame)
            if self.config.auto_start and self._vcam_available:
                GLib.idle_add(self._auto_start)

        self._window.present()

    def _on_close_request(self, window):
        """Minimize to background instead of quitting."""
        if self.config.minimize_on_close:
            window.set_visible(False)
            if self._streaming:
                print("[NV Broadcast] Minimized to background - virtual camera still active")
            return True  # Prevent destruction
        return False  # Allow normal close

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
        self.stop_pipeline()

        self._video_pipeline = VideoPipeline()
        self._video_pipeline.configure(
            source_device=camera_device,
            vcam_device=self._vcam_device or "/dev/video10",
            width=self.config.video.width,
            height=self.config.video.height,
            fps=self.config.video.fps,
            output_format=output_format,
        )

        self._video_pipeline.set_effect_callback(self._process_frame)
        self._video_pipeline.set_preview_callback(
            lambda texture: self._window.update_preview(texture)
        )

        # Start in effects mode if effects were previously enabled
        if self._any_video_effects_active():
            self._video_pipeline._effects_active = True

        try:
            self._video_pipeline.build(vcam_enabled=self._vcam_available)
            self._video_pipeline.start()
            self._streaming = True

            status = f"Streaming: {camera_device}"
            if self._vcam_available:
                status += f" -> {self._vcam_device} ({output_format})"
            self._window.set_status(status)
            self.config.video.camera_device = camera_device
            self.config.video.output_format = output_format
            save_config(self.config)

        except Exception as e:
            self._window.set_status(f"Pipeline error: {e}")
            print(f"[NV Broadcast] Pipeline failed: {e}")

    def stop_pipeline(self):
        if self._video_pipeline:
            self._video_pipeline.stop()
            self._video_pipeline = None
        self._streaming = False

    def _process_frame(self, frame_data: bytes, width: int, height: int) -> bytes:
        result = frame_data
        if self._video_effects.enabled:
            result = self._video_effects.process_frame(result, width, height)
        if self._autoframe.enabled:
            result = self._autoframe.process_frame(result, width, height)
        return result

    def _any_video_effects_active(self) -> bool:
        return self._video_effects.enabled or self._autoframe.enabled

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

    def set_edge_param(self, param: str, value: float):
        """Update a single edge refinement parameter."""
        setattr(self.config.video.edge, param, value)
        self._video_effects.update_edge_params(**{param: value})
        save_config(self.config)

    def set_autoframe(self, enabled: bool):
        self._autoframe.enabled = enabled
        self.config.video.auto_frame = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_autoframe_zoom(self, value: float):
        self._autoframe.zoom_level = value
        self.config.video.auto_frame_zoom = value
        save_config(self.config)

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
        Adw.Application.do_shutdown(self)

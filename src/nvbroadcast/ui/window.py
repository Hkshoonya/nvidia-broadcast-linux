# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Main window - NVIDIA Broadcast layout."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio

from nvbroadcast.core.constants import APP_NAME, APP_SUBTITLE
from nvbroadcast.core.gpu import detect_gpus, select_compute_gpu
from nvbroadcast.ui.video_preview import VideoPreview
from nvbroadcast.ui.controls import (
    EffectToggle, EffectSlider, BackgroundModeSelector, BackgroundImagePicker
)
from nvbroadcast.ui.device_selector import DeviceSelector
from nvbroadcast.video.virtual_camera import list_camera_devices, list_camera_modes


class NVBroadcastWindow(Adw.ApplicationWindow):
    """Layout: large preview on top, Camera | Audio sections below."""

    def __init__(self, app):
        super().__init__(application=app, title=APP_NAME)
        self.set_default_size(1280, 900)
        self._app = app
        self._streaming = False
        self._build_ui()
        self._populate_devices()

    def _build_ui(self):
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Header
        header = Adw.HeaderBar()
        header.add_css_class("flat")
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.CENTER)
        title_lbl = Gtk.Label(label=APP_NAME)
        title_lbl.add_css_class("app-title")
        title_box.append(title_lbl)
        sub_lbl = Gtk.Label(label=APP_SUBTITLE)
        sub_lbl.add_css_class("app-subtitle")
        title_box.append(sub_lbl)
        header.set_title_widget(title_box)

        self._stream_btn = Gtk.Button(label="Start Broadcast")
        self._stream_btn.add_css_class("suggested-action")
        self._stream_btn.connect("clicked", self._on_stream_toggle)
        header.pack_end(self._stream_btn)

        # Quit button (actually exits, vs close which minimizes)
        quit_btn = Gtk.Button(icon_name="application-exit-symbolic",
                              tooltip_text="Quit NVIDIA Broadcast")
        quit_btn.connect("clicked", lambda _: self._app.quit())
        header.pack_end(quit_btn)

        # About button
        about_btn = Gtk.Button(icon_name="help-about-symbolic",
                               tooltip_text="About")
        about_btn.connect("clicked", self._show_about)
        header.pack_end(about_btn)

        gpu_btn = Gtk.MenuButton(icon_name="applications-graphics-symbolic",
                                 tooltip_text="GPU Information")
        self._gpu_popover = Gtk.Popover()
        self._gpu_label = Gtk.Label()
        self._gpu_label.add_css_class("gpu-label")
        self._gpu_label.set_margin_top(8)
        self._gpu_label.set_margin_bottom(8)
        self._gpu_label.set_margin_start(12)
        self._gpu_label.set_margin_end(12)
        self._gpu_popover.set_child(self._gpu_label)
        gpu_btn.set_popover(self._gpu_popover)
        header.pack_start(gpu_btn)
        self._update_gpu_info()
        main_box.append(header)

        # Resizable split: Preview (top) | Controls (bottom)
        # Using Gtk.Paned for draggable divider between preview and controls
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        paned.set_vexpand(True)
        paned.set_shrink_start_child(True)
        paned.set_shrink_end_child(False)
        paned.set_resize_start_child(True)
        paned.set_resize_end_child(True)

        # Top: Preview + controls bar
        preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        preview_frame = Gtk.Frame()
        preview_frame.set_margin_start(16)
        preview_frame.set_margin_end(16)
        preview_frame.set_margin_top(8)
        self._preview = VideoPreview()
        self._preview.set_vexpand(True)
        self._preview.set_size_request(-1, 200)  # Minimum height
        preview_frame.set_child(self._preview)
        self._preview_frame = preview_frame
        preview_box.append(preview_frame)

        # Preview controls bar
        preview_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        preview_bar.set_margin_end(16)
        preview_bar.set_margin_top(2)
        preview_bar.append(Gtk.Box(hexpand=True))

        self._freeze_btn = Gtk.ToggleButton(label="Pause View")
        self._freeze_btn.set_tooltip_text("Freeze/resume camera preview")
        self._freeze_btn.add_css_class("flat")
        self._freeze_btn.connect("toggled", self._on_freeze_toggled)
        self._preview_frozen = False
        preview_bar.append(self._freeze_btn)

        self._hide_btn = Gtk.ToggleButton(label="Hide Preview")
        self._hide_btn.set_tooltip_text("Show/hide camera preview")
        self._hide_btn.add_css_class("flat")
        self._hide_btn.connect("toggled", self._on_hide_toggled)
        preview_bar.append(self._hide_btn)

        preview_box.append(preview_bar)
        paned.set_start_child(preview_box)

        # Bottom: Scrollable controls
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_vexpand(True)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        controls.set_margin_start(16)
        controls.set_margin_end(16)
        controls.set_margin_top(12)
        controls.set_margin_bottom(8)

        cam = self._build_camera_section()
        cam.set_hexpand(True)
        controls.append(cam)
        controls.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        aud = self._build_audio_section()
        aud.set_hexpand(True)
        controls.append(aud)

        scroll.set_child(controls)
        paned.set_end_child(scroll)

        # Set initial divider position (60% preview, 40% controls)
        paned.set_position(450)
        main_box.append(paned)

        # Status bar with attribution
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self._status_bar = Gtk.Label(label="Ready - Click 'Start Broadcast' to begin")
        self._status_bar.add_css_class("status-bar")
        self._status_bar.set_xalign(0)
        self._status_bar.set_hexpand(True)
        self._status_bar.set_ellipsize(3)
        self._status_bar.set_margin_start(4)
        status_box.append(self._status_bar)

        credit = Gtk.Label(label="by doczeus")
        credit.add_css_class("app-subtitle")
        credit.set_margin_end(8)
        status_box.append(credit)

        main_box.append(status_box)

        self.set_content(main_box)

    def _build_camera_section(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        hdr = Gtk.Label(label="CAMERA")
        hdr.add_css_class("section-header")
        hdr.set_xalign(0)
        box.append(hdr)

        # Input settings in a compact card
        input_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        input_card.add_css_class("effect-card")

        self._camera_selector = DeviceSelector("Source")
        input_card.append(self._camera_selector)

        # Query camera capabilities
        cam_device = self._app.config.video.camera_device
        self._camera_modes = list_camera_modes(cam_device)
        self._updating_ui = False  # Guard against signal cascades

        # Resolution selector (from camera capabilities)
        self._res_selector = DeviceSelector("Resolution")
        res_devices = []
        _RES_LABELS = {
            (640, 360): "360p", (640, 480): "480p", (800, 600): "600p",
            (1024, 576): "576p", (960, 720): "720p 4:3",
            (1280, 720): "720p", (1280, 960): "960p",
            (1920, 1080): "1080p", (2560, 1440): "1440p",
            (3840, 2160): "4K",
        }
        for mode in self._camera_modes:
            w, h = mode["width"], mode["height"]
            label = _RES_LABELS.get((w, h), f"{w}x{h}")
            max_fps = max(mode["fps"]) if mode["fps"] else 30
            res_devices.append({
                "name": f"{label} ({w}x{h}) {max_fps}fps",
                "device": f"{w}x{h}",
            })
        if not res_devices:
            res_devices = [{"name": "1280x720", "device": "1280x720"}]
        self._res_selector.set_devices(res_devices)
        # Select current resolution
        current_res = f"{self._app.config.video.width}x{self._app.config.video.height}"
        for i, d in enumerate(res_devices):
            if d["device"] == current_res:
                self._res_selector.set_selected_index(i)
                break
        self._res_selector.connect("device-changed", self._on_resolution_changed)
        input_card.append(self._res_selector)

        self._format_selector = DeviceSelector("Format")
        self._format_selector.set_devices([
            {"name": "YUY2 (Chrome / Zoom)", "device": "YUY2"},
            {"name": "I420 (Firefox / WebRTC)", "device": "I420"},
            {"name": "NV12 (General)", "device": "NV12"},
        ])
        input_card.append(self._format_selector)

        # FPS selector — show what the current resolution supports
        self._fps_selector = DeviceSelector("FPS")
        self._refresh_fps_options()
        self._fps_selector.connect("device-changed", self._on_fps_changed)
        input_card.append(self._fps_selector)

        box.append(input_card)

        # Processing card (mode, GPU, mirror, edge refine)
        proc_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        proc_card.add_css_class("effect-card")

        from nvbroadcast.core.config import detect_compositing_backends
        backends = detect_compositing_backends()
        has_cuda = backends.get("cupy", False)

        self._mode_devices = []
        if has_cuda:
            self._mode_devices.append({"name": "Killer - NVDEC + TensorRT + Fused CUDA", "device": "killer"})
            self._mode_devices.append({"name": "Zeus - TensorRT Inference", "device": "zeus"})
            self._mode_devices.append({"name": "DocZeus - Fused CUDA Kernel", "device": "doczeus"})
            self._mode_devices.append({"name": "CUDA - Max Quality", "device": "cuda_max"})
            self._mode_devices.append({"name": "CUDA - Balanced", "device": "cuda_balanced"})
            self._mode_devices.append({"name": "CUDA - Performance", "device": "cuda_perf"})
        self._mode_devices.append({"name": "CPU - Quality", "device": "cpu_quality"})
        self._mode_devices.append({"name": "CPU - Light", "device": "cpu_light"})
        self._mode_devices.append({"name": "CPU - Low End", "device": "cpu_low"})

        self._profile_selector = DeviceSelector("Mode")
        self._profile_selector.set_devices(self._mode_devices)

        # Map current config to the right index
        comp = self._app.config.compositing
        prof = self._app.config.performance_profile
        current_key = self._profile_and_comp_to_mode(prof, comp)
        for i, d in enumerate(self._mode_devices):
            if d["device"] == current_key:
                self._profile_selector.set_selected_index(i)
                break

        self._profile_selector.connect("device-changed", self._on_mode_changed_selector)
        proc_card.append(self._profile_selector)

        # GPU selector
        from nvbroadcast.core.gpu import detect_gpus
        gpus = detect_gpus()
        if len(gpus) > 1:
            self._gpu_selector = DeviceSelector("GPU")
            gpu_devices = [
                {"name": f"GPU {g.index}: {g.name} ({g.memory_total_mb}MB)", "device": str(g.index)}
                for g in gpus
            ]
            self._gpu_selector.set_devices(gpu_devices)
            self._gpu_selector.set_selected_index(self._app.config.compute_gpu)
            self._gpu_selector.connect("device-changed", self._on_gpu_changed)
            proc_card.append(self._gpu_selector)
        else:
            self._gpu_selector = None

        # Mirror toggle
        self._mirror_toggle = EffectToggle(
            "Mirror", "Flip video horizontally"
        )
        self._mirror_toggle.active = True
        self._mirror_toggle.connect("toggled", self._on_mirror_toggled)
        proc_card.append(self._mirror_toggle)

        # Edge Refine toggle (visible only for Zeus/Killer)
        self._edge_refine_toggle = EffectToggle(
            "Edge Refine", "Neural edge refinement (Zeus/Killer)"
        )
        self._edge_refine_toggle.set_sensitive(False)
        self._edge_refine_toggle.set_visible(False)
        self._edge_refine_toggle.connect("toggled", self._on_edge_refine_toggled)
        proc_card.append(self._edge_refine_toggle)

        box.append(proc_card)

        # Background effect card
        bg_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bg_card.add_css_class("effect-card")
        self._bg_toggle = EffectToggle("Background", "Remove, blur, or replace background")
        self._bg_toggle.connect("toggled", self._on_bg_toggled)
        bg_card.append(self._bg_toggle)

        # Model selector
        self._model_selector = DeviceSelector("Model")
        self._model_selector.set_devices([
            {"name": "RVM - Person Matting (fastest)", "device": "rvm"},
            {"name": "IS-Net - General Objects", "device": "isnet"},
            {"name": "BiRefNet - Best Quality (heavy)", "device": "birefnet"},
        ])
        self._model_selector.set_sensitive(False)
        self._model_selector.set_selected_index(0)
        self._model_selector.connect("device-changed", self._on_model_changed)
        bg_card.append(self._model_selector)

        # Quality selector
        self._quality_selector = DeviceSelector("Quality")
        self._quality_selector.set_devices([
            {"name": "Performance (fastest)", "device": "performance"},
            {"name": "Balanced (fast, better edges)", "device": "balanced"},
            {"name": "Quality (detailed edges)", "device": "quality"},
            {"name": "Ultra (best, sharpest edges)", "device": "ultra"},
        ])
        self._quality_selector.set_sensitive(False)
        self._quality_selector.set_selected_index(2)  # Default: Quality
        self._quality_selector.connect("device-changed", self._on_quality_changed)
        bg_card.append(self._quality_selector)

        self._bg_mode = BackgroundModeSelector()
        self._bg_mode.set_sensitive(False)
        self._bg_mode.connect("mode-changed", self._on_bg_mode_changed)
        bg_card.append(self._bg_mode)
        self._bg_image_picker = BackgroundImagePicker()
        self._bg_image_picker.set_sensitive(False)
        self._bg_image_picker.connect("image-selected", self._on_bg_image_selected)
        bg_card.append(self._bg_image_picker)
        self._blur_slider = EffectSlider("Strength", 0.7)
        self._blur_slider.set_sensitive(False)
        self._blur_slider.connect("value-changed", self._on_blur_changed)
        bg_card.append(self._blur_slider)

        # Advanced Edge Tuning (collapsible)
        adv_expander = Gtk.Expander(label="Advanced Edge Tuning")
        adv_expander.set_margin_start(8)
        adv_expander.set_margin_end(8)
        adv_expander.set_margin_top(4)
        adv_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        self._edge_dilate = EffectSlider("Dilate", 5.0, 0.0, 15.0)
        self._edge_dilate.set_sensitive(False)
        self._edge_dilate.connect("value-changed", self._on_edge_dilate)
        adv_box.append(self._edge_dilate)

        self._edge_blur = EffectSlider("Softness", 9.0, 1.0, 25.0)
        self._edge_blur.set_sensitive(False)
        self._edge_blur.connect("value-changed", self._on_edge_blur)
        adv_box.append(self._edge_blur)

        self._edge_strength = EffectSlider("Sharpness", 12.0, 1.0, 30.0)
        self._edge_strength.set_sensitive(False)
        self._edge_strength.connect("value-changed", self._on_edge_strength)
        adv_box.append(self._edge_strength)

        self._edge_midpoint = EffectSlider("Midpoint", 0.5, 0.1, 0.9)
        self._edge_midpoint.set_sensitive(False)
        self._edge_midpoint.connect("value-changed", self._on_edge_midpoint)
        adv_box.append(self._edge_midpoint)

        # Separator
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep.set_margin_top(6)
        sep.set_margin_bottom(2)
        adv_box.append(sep)

        perf_lbl = Gtk.Label(label="Performance")
        perf_lbl.set_xalign(0)
        perf_lbl.set_margin_start(16)
        perf_lbl.add_css_class("device-label")
        adv_box.append(perf_lbl)

        self._skip_interval = EffectSlider("Frame Skip", 1.0, 1.0, 5.0)
        self._skip_interval.set_sensitive(False)
        self._skip_interval.connect("value-changed", self._on_skip_interval)
        adv_box.append(self._skip_interval)

        self._ema_weight = EffectSlider("Smoothing", 0.15, 0.0, 0.5)
        self._ema_weight.set_sensitive(False)
        self._ema_weight.connect("value-changed", self._on_ema_weight)
        adv_box.append(self._ema_weight)

        adv_expander.set_child(adv_box)
        bg_card.append(adv_expander)
        self._adv_expander = adv_expander

        box.append(bg_card)

        # Auto Frame card
        af_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        af_card.add_css_class("effect-card")
        self._autoframe_toggle = EffectToggle("Auto Frame", "Track face and auto-zoom")
        self._autoframe_toggle.connect("toggled", self._on_autoframe_toggled)
        af_card.append(self._autoframe_toggle)
        self._zoom_slider = EffectSlider("Zoom", 1.5, 1.0, 3.0)
        self._zoom_slider.set_sensitive(False)
        self._zoom_slider.connect("value-changed", self._on_zoom_changed)
        af_card.append(self._zoom_slider)
        box.append(af_card)

        # Video Enhancement card
        beauty_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        beauty_card.add_css_class("effect-card")
        self._beauty_toggle = EffectToggle(
            "Video Enhancement", "Skin smooth, denoise, enhance, sharpen, vignette"
        )
        self._beauty_toggle.connect("toggled", self._on_beauty_toggled)
        beauty_card.append(self._beauty_toggle)

        # Preset dropdown
        self._beauty_preset = DeviceSelector("Preset")
        self._beauty_preset.set_devices([
            {"name": "Natural (subtle)", "device": "natural"},
            {"name": "Broadcast (professional)", "device": "broadcast"},
            {"name": "Glamour (strong)", "device": "glamour"},
            {"name": "Custom", "device": "custom"},
        ])
        self._beauty_preset.set_sensitive(False)
        self._beauty_preset.set_selected_index(0)
        self._beauty_preset.connect("device-changed", self._on_beauty_preset)
        beauty_card.append(self._beauty_preset)

        # Individual effect rows: each has a toggle + slider
        self._beauty_controls = {}
        beauty_effects = [
            ("skin_smooth", "Skin Smooth", "Bilateral filter — smooths skin, preserves edges", 0.5),
            ("denoise", "Denoise", "Temporal + spatial noise reduction", 0.3),
            ("enhance", "Enhance", "Brightness, contrast, warmth on face", 0.4),
            ("sharpen", "Sharpen", "Unsharp mask — crisper eyes and lips", 0.3),
            ("edge_darken", "Edge Darken", "Vignette centered on face — studio look", 0.3),
        ]

        for key, title, subtitle, default_val in beauty_effects:
            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

            toggle = EffectToggle(title, subtitle)
            toggle.set_sensitive(False)
            toggle.connect("toggled", self._on_beauty_effect_toggled, key)
            row_box.append(toggle)

            slider = EffectSlider("Intensity", default_val)
            slider.set_sensitive(False)
            slider.connect("value-changed", self._on_beauty_effect_value, key)
            row_box.append(slider)

            beauty_card.append(row_box)
            self._beauty_controls[key] = {"toggle": toggle, "slider": slider, "default": default_val}

        box.append(beauty_card)

        return box

    def _build_audio_section(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        hdr = Gtk.Label(label="AUDIO")
        hdr.add_css_class("section-header")
        hdr.set_xalign(0)
        box.append(hdr)

        # Mic card
        mic_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        mic_card.add_css_class("effect-card")
        mic_lbl = Gtk.Label(label="Microphone")
        mic_lbl.set_xalign(0)
        mic_lbl.add_css_class("device-label")
        mic_card.append(mic_lbl)
        self._noise_toggle = EffectToggle("Noise Removal", "Remove background noise from mic")
        self._noise_toggle.connect("toggled", self._on_noise_toggled)
        mic_card.append(self._noise_toggle)
        self._noise_slider = EffectSlider("Strength", 1.0)
        self._noise_slider.set_sensitive(False)
        self._noise_slider.connect("value-changed", self._on_noise_intensity_changed)
        mic_card.append(self._noise_slider)
        box.append(mic_card)

        # Speaker card
        spk_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        spk_card.add_css_class("effect-card")
        spk_lbl = Gtk.Label(label="Speakers")
        spk_lbl.set_xalign(0)
        spk_lbl.add_css_class("device-label")
        spk_card.append(spk_lbl)
        self._speaker_toggle = EffectToggle("Noise Removal", "Remove noise from incoming audio")
        self._speaker_toggle.connect("toggled", self._on_speaker_toggled)
        spk_card.append(self._speaker_toggle)
        box.append(spk_card)

        box.append(Gtk.Box(vexpand=True))  # spacer
        return box

    # --- Signals ---
    def _on_stream_toggle(self, btn):
        if self._streaming:
            self._app.stop_pipeline()
            self._streaming = False
            btn.set_label("Start Broadcast")
            btn.remove_css_class("destructive-action")
            btn.add_css_class("suggested-action")
            self.set_status("Stopped")
        else:
            fmt = self._format_selector.get_selected_device() or "YUY2"
            cam = self._camera_selector.get_selected_device() or "/dev/video0"
            self._app.start_pipeline(cam, fmt)
            self._streaming = True
            btn.set_label("Stop Broadcast")
            btn.remove_css_class("suggested-action")
            btn.add_css_class("destructive-action")

    def _on_bg_toggled(self, t, active):
        self._app.set_bg_removal(active)
        self._bg_mode.set_sensitive(active)
        self._blur_slider.set_sensitive(active)
        self._quality_selector.set_sensitive(active)
        self._model_selector.set_sensitive(active)
        self._edge_dilate.set_sensitive(active)
        self._edge_blur.set_sensitive(active)
        self._edge_strength.set_sensitive(active)
        self._edge_midpoint.set_sensitive(active)
        self._skip_interval.set_sensitive(active)
        self._ema_weight.set_sensitive(active)
        mode = self._bg_mode.mode
        self._bg_image_picker.set_sensitive(active and mode == "replace")

    @staticmethod
    def _profile_and_comp_to_mode(profile, compositing):
        """Map profile+compositing config to unified mode key."""
        if compositing in ("cupy", "gstreamer_gl"):
            if profile == "max_quality":
                return "cuda_max"
            if profile == "balanced":
                return "cuda_balanced"
            return "cuda_perf"
        if profile == "max_quality":
            return "cpu_quality"
        if profile in ("performance", "balanced"):
            return "cpu_light"
        if profile == "potato":
            return "cpu_low"
        return "cpu_quality"

    # (profile, compositing, use_tensorrt, use_fused_kernel, use_nvdec)
    _MODE_MAP = {
        "killer":       ("max_quality", "cupy", True,  True,  True),
        "zeus":         ("max_quality", "cupy", True,  False, False),
        "doczeus":      ("max_quality", "cupy", False, True,  False),
        "cuda_max":     ("max_quality", "cupy", False, False, False),
        "cuda_balanced": ("balanced",   "cupy", False, False, False),
        "cuda_perf":    ("performance", "cupy", False, False, False),
        "cpu_quality":  ("max_quality", "cpu",  False, False, False),
        "cpu_light":    ("performance", "cpu",  False, False, False),
        "cpu_low":      ("potato",      "cpu",  False, False, False),
    }

    def _on_mode_changed_selector(self, selector, mode_key):
        if mode_key in self._MODE_MAP:
            profile, comp, trt, fused, nvdec = self._MODE_MAP[mode_key]
            self._app.set_performance_profile(
                profile, compositing=comp,
                use_tensorrt=trt, use_fused_kernel=fused, use_nvdec=nvdec,
            )
            # Show edge refine toggle only for premium speed modes
            is_premium = mode_key in ("killer", "zeus")
            self._edge_refine_toggle.set_visible(is_premium)
            self._edge_refine_toggle.set_sensitive(is_premium)
            if is_premium and not self._edge_refine_toggle.active:
                self._edge_refine_toggle.active = True  # Enable by default

    def _on_gpu_changed(self, selector, gpu_str):
        self._app.set_compute_gpu(int(gpu_str))

    def _refresh_fps_options(self):
        """Rebuild FPS dropdown for the current resolution. Blocks signal cascade."""
        w, h = self._app.config.video.width, self._app.config.video.height
        available_fps = [30]
        for mode in self._camera_modes:
            if mode["width"] == w and mode["height"] == h:
                available_fps = mode["fps"]
                break
        self._updating_ui = True  # Block signal cascade
        devices = [{"name": f"{fps} fps", "device": str(fps)} for fps in available_fps]
        self._fps_selector.set_devices(devices)
        # Select current fps or highest available
        current = self._app.config.video.fps
        best_idx = len(devices) - 1  # Default to highest
        for i, d in enumerate(devices):
            if int(d["device"]) == current:
                best_idx = i
                break
        self._fps_selector.set_selected_index(best_idx)
        self._updating_ui = False

    def _on_resolution_changed(self, selector, res_str):
        if self._updating_ui:
            return
        w, h = res_str.split("x")
        self._app.set_resolution(int(w), int(h))
        # Update FPS options for new resolution (guarded — won't cascade)
        self._refresh_fps_options()

    def _on_fps_changed(self, selector, fps_str):
        if self._updating_ui:
            return
        self._app.set_fps(int(fps_str))

    def _on_model_changed(self, selector, model):
        self._app.set_model(model)

    def _on_quality_changed(self, selector, quality):
        self._app.set_quality(quality)

    def _on_bg_mode_changed(self, s, mode):
        self._app.set_bg_mode(mode)
        self._bg_image_picker.set_sensitive(mode == "replace")

    def _on_bg_image_selected(self, p, path):
        self._app.set_bg_image(path)

    def _on_blur_changed(self, s, v):
        self._app.set_blur_intensity(v)

    def _on_edge_dilate(self, s, v):
        self._app.set_edge_param("dilate_size", int(v))

    def _on_edge_blur(self, s, v):
        self._app.set_edge_param("blur_size", int(v))

    def _on_edge_strength(self, s, v):
        self._app.set_edge_param("sigmoid_strength", v)

    def _on_edge_midpoint(self, s, v):
        self._app.set_edge_param("sigmoid_midpoint", v)

    def _on_skip_interval(self, s, v):
        self._app.set_skip_interval(int(v))

    def _on_ema_weight(self, s, v):
        self._app.set_ema_weight(v)

    def _on_mirror_toggled(self, t, active):
        self._app.set_mirror(active)

    def _on_edge_refine_toggled(self, t, active):
        self._app.set_edge_refine(active)

    # Beauty presets: (skin_smooth, denoise, enhance, sharpen, edge_darken)
    _BEAUTY_PRESETS = {
        "natural":   {"skin_smooth": 0.3, "denoise": 0.2, "enhance": 0.2, "sharpen": 0.2, "edge_darken": 0.15},
        "broadcast": {"skin_smooth": 0.5, "denoise": 0.3, "enhance": 0.4, "sharpen": 0.3, "edge_darken": 0.3},
        "glamour":   {"skin_smooth": 0.8, "denoise": 0.4, "enhance": 0.6, "sharpen": 0.4, "edge_darken": 0.5},
        "custom":    None,  # Don't change sliders
    }

    def _on_beauty_toggled(self, t, active):
        self._app.set_beautify(active)
        self._beauty_preset.set_sensitive(active)
        for ctrl in self._beauty_controls.values():
            ctrl["toggle"].set_sensitive(active)
            # Slider enabled only if master + individual toggle both on
            ctrl["slider"].set_sensitive(active and ctrl["toggle"].active)
        if active:
            # Apply current preset
            self._on_beauty_preset(None, self._beauty_preset.get_selected_device() or "natural")

    def _on_beauty_preset(self, selector, preset_key):
        values = self._BEAUTY_PRESETS.get(preset_key)
        if values is None:
            return  # Custom — don't touch sliders
        for key, val in values.items():
            ctrl = self._beauty_controls.get(key)
            if ctrl:
                ctrl["toggle"].active = val > 0
                ctrl["slider"]._scale.set_value(val)
                ctrl["slider"].set_sensitive(val > 0 and self._beauty_toggle.active)
                self._app.set_beautify_param(key, val)

    def _on_beauty_effect_toggled(self, toggle, active, key):
        ctrl = self._beauty_controls[key]
        ctrl["slider"].set_sensitive(active and self._beauty_toggle.active)
        if active:
            self._app.set_beautify_param(key, ctrl["slider"].value)
        else:
            self._app.set_beautify_param(key, 0.0)
        # Switch preset to Custom when user manually toggles
        self._beauty_preset.set_selected_index(3)  # Custom

    def _on_beauty_effect_value(self, slider, value, key):
        self._app.set_beautify_param(key, value)
        # Switch preset to Custom when user manually adjusts
        self._beauty_preset.set_selected_index(3)  # Custom

    def _on_autoframe_toggled(self, t, active):
        self._app.set_autoframe(active)
        self._zoom_slider.set_sensitive(active)

    def _on_zoom_changed(self, s, v):
        self._app.set_autoframe_zoom(v)

    def _on_noise_toggled(self, t, active):
        self._app.set_noise_removal(active)
        self._noise_slider.set_sensitive(active)

    def _on_noise_intensity_changed(self, s, v):
        self._app.set_noise_intensity(v)

    def _on_speaker_toggled(self, t, active):
        self._app.set_speaker_denoise(active)

    # --- Public ---
    def _populate_devices(self):
        cameras = list_camera_devices()
        if cameras:
            self._camera_selector.set_devices(cameras)

    def _update_gpu_info(self):
        gpus = detect_gpus()
        if gpus:
            compute = select_compute_gpu(gpus)
            lines = [
                f"GPU {g.index}: {g.name}"
                f"{' [Compute]' if compute and g.index == compute.index else ' [Display]'}"
                for g in gpus
            ]
            self._gpu_label.set_text("\n".join(lines))
        else:
            self._gpu_label.set_text("No NVIDIA GPUs detected")

    def restore_settings(self, config):
        """Restore saved settings to all UI controls."""
        v = config.video
        a = config.audio

        # Model (0=rvm, 1=isnet, 2=birefnet)
        model_map = {"rvm": 0, "isnet": 1, "birefnet": 2}
        if v.model in model_map:
            self._model_selector.set_selected_index(model_map[v.model])

        # Quality preset (0=perf, 1=balanced, 2=quality, 3=ultra)
        quality_map = {"performance": 0, "balanced": 1, "quality": 2, "ultra": 3}
        if v.quality_preset in quality_map:
            self._quality_selector.set_selected_index(quality_map[v.quality_preset])

        # Format (0=YUY2, 1=I420, 2=NV12)
        fmt_map = {"YUY2": 0, "I420": 1, "NV12": 2}
        if v.output_format in fmt_map:
            self._format_selector.set_selected_index(fmt_map[v.output_format])

        # Background mode (0=blur, 1=replace, 2=remove)
        mode_map = {"blur": 0, "replace": 1, "remove": 2}
        if v.background_mode in mode_map:
            self._bg_mode._dropdown.set_selected(mode_map[v.background_mode])

        # Background image path
        if v.background_image:
            import os
            self._bg_image_picker._selected_path = v.background_image
            self._bg_image_picker._path_label.set_text(os.path.basename(v.background_image))
            self._bg_image_picker._path_label.set_opacity(1.0)

        # Sliders
        self._blur_slider._scale.set_value(v.blur_intensity)
        self._zoom_slider._scale.set_value(v.auto_frame_zoom)

        # Advanced edge tuning
        self._edge_dilate._scale.set_value(v.edge.dilate_size)
        self._edge_blur._scale.set_value(v.edge.blur_size)
        self._edge_strength._scale.set_value(v.edge.sigmoid_strength)
        self._edge_midpoint._scale.set_value(v.edge.sigmoid_midpoint)

        # Toggles - set without triggering signals first
        if v.background_removal:
            self._bg_toggle.active = True
            self._bg_mode.set_sensitive(True)
            self._blur_slider.set_sensitive(True)
            self._quality_selector.set_sensitive(True)
            self._model_selector.set_sensitive(True)
            self._edge_dilate.set_sensitive(True)
            self._edge_blur.set_sensitive(True)
            self._edge_strength.set_sensitive(True)
            self._edge_midpoint.set_sensitive(True)
            self._skip_interval.set_sensitive(True)
            self._ema_weight.set_sensitive(True)
            self._bg_image_picker.set_sensitive(v.background_mode == "replace")

        if v.auto_frame:
            self._autoframe_toggle.active = True
            self._zoom_slider.set_sensitive(True)

        if a.noise_removal:
            self._noise_toggle.active = True
            self._noise_slider.set_sensitive(True)

        if a.speaker_denoise:
            self._speaker_toggle.active = True

    def _show_about(self, button):
        about = Adw.AboutWindow(
            transient_for=self,
            application_name=APP_NAME,
            application_icon="com.doczeus.NVBroadcast",
            version=__import__("nvbroadcast").__version__,
            developer_name="doczeus",
            website="https://github.com/Hkshoonya/nvidia-broadcast-linux",
            issue_url="https://github.com/Hkshoonya/nvidia-broadcast-linux/issues",
            license_type=Gtk.License.GPL_3_0,
            copyright="Copyright (c) 2026 doczeus",
            developers=["doczeus https://github.com/Hkshoonya"],
            comments=(
                "AI-powered virtual camera for Linux.\n\n"
                "Background removal, blur, replacement, auto-framing, "
                "video enhancement, and noise cancellation.\n"
                "9 processing modes including Killer, Zeus, and DocZeus "
                "with fused CUDA kernels and edge refinement.\n\n"
                "Created by doczeus | AI Powered"
            ),
        )
        about.present()

    def rebuild_mode_selector(self, compositing: str, profile: str):
        """Rebuild the Mode dropdown with currently available backends."""
        from nvbroadcast.core.config import detect_compositing_backends
        backends = detect_compositing_backends()
        has_cuda = backends.get("cupy", False) or compositing == "cupy"

        self._mode_devices = []
        if has_cuda:
            self._mode_devices.append({"name": "Killer - NVDEC + TensorRT + Fused CUDA", "device": "killer"})
            self._mode_devices.append({"name": "Zeus - TensorRT Inference", "device": "zeus"})
            self._mode_devices.append({"name": "DocZeus - Fused CUDA Kernel", "device": "doczeus"})
            self._mode_devices.append({"name": "CUDA - Max Quality", "device": "cuda_max"})
            self._mode_devices.append({"name": "CUDA - Balanced", "device": "cuda_balanced"})
            self._mode_devices.append({"name": "CUDA - Performance", "device": "cuda_perf"})
        self._mode_devices.append({"name": "CPU - Quality", "device": "cpu_quality"})
        self._mode_devices.append({"name": "CPU - Light", "device": "cpu_light"})
        self._mode_devices.append({"name": "CPU - Low End", "device": "cpu_low"})

        self._profile_selector.set_devices(self._mode_devices)
        # Signal already connected in _build_camera_section — don't duplicate

        mode_key = self._profile_and_comp_to_mode(profile, compositing)
        for i, d in enumerate(self._mode_devices):
            if d["device"] == mode_key:
                self._profile_selector.set_selected_index(i)
                break

    def _on_freeze_toggled(self, btn):
        self._preview_frozen = btn.get_active()
        btn.set_label("Resume View" if self._preview_frozen else "Pause View")

    def _on_hide_toggled(self, btn):
        hidden = btn.get_active()
        self._preview_frame.set_visible(not hidden)
        btn.set_label("Show Preview" if hidden else "Hide Preview")
        self._freeze_btn.set_sensitive(not hidden)

    def update_preview(self, texture):
        if not self._preview_frozen:
            self._preview.update_texture(texture)

    def set_status(self, text: str):
        self._status_bar.set_text(text)

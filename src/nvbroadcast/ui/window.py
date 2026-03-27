# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Main window - NVIDIA Broadcast layout."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, GLib

from nvbroadcast.core.constants import APP_NAME, APP_SUBTITLE
from nvbroadcast.core.config import save_config
from nvbroadcast.core.gpu import detect_gpus, select_compute_gpu
from nvbroadcast.ui.video_preview import VideoPreview
from nvbroadcast.ui.controls import (
    EffectToggle, EffectSlider, BackgroundModeSelector, BackgroundImagePicker
)
from nvbroadcast.ui.device_selector import DeviceSelector
from nvbroadcast.video.virtual_camera import (
    list_camera_devices, list_camera_modes,
    get_firefox_profiles, is_firefox_pipewire_disabled, set_firefox_pipewire,
)
from nvbroadcast.core.platform import has_tensorrt_runtime
from nvbroadcast.core.resources import find_app_icon


def _collapsible_card(title: str, content: Gtk.Widget, expanded: bool = True) -> Gtk.Box:
    """Wrap a card in a collapsible container with a clickable header."""
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
    outer.add_css_class("effect-card")

    # Header row: title + chevron
    header_btn = Gtk.Button()
    header_btn.add_css_class("card-header-btn")
    header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    header_box.set_margin_start(2)
    header_box.set_margin_end(2)
    lbl = Gtk.Label(label=title, xalign=0, hexpand=True)
    lbl.add_css_class("card-title")
    header_box.append(lbl)
    chevron = Gtk.Image.new_from_icon_name(
        "pan-down-symbolic" if expanded else "pan-end-symbolic"
    )
    chevron.add_css_class("card-chevron")
    header_box.append(chevron)
    header_btn.set_child(header_box)
    outer.append(header_btn)

    # Content revealer
    revealer = Gtk.Revealer()
    revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
    revealer.set_transition_duration(200)
    revealer.set_reveal_child(expanded)
    revealer.set_child(content)
    outer.append(revealer)

    def _toggle(btn):
        visible = not revealer.get_reveal_child()
        revealer.set_reveal_child(visible)
        chevron.set_from_icon_name(
            "pan-down-symbolic" if visible else "pan-end-symbolic"
        )

    header_btn.connect("clicked", _toggle)
    return outer


class NVBroadcastWindow(Adw.ApplicationWindow):
    """Layout: large preview on top, Camera | Audio sections below."""

    def __init__(self, app):
        super().__init__(application=app, title=APP_NAME)
        self.set_default_size(1280, 900)
        self._app = app
        self._streaming = False
        self._installer = None
        self._install_pulse_id = 0
        self._pending_mode_key = ""
        self._pending_meeting_start = False
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

        # Meeting button (recording + transcription)
        self._meeting_btn = Gtk.Button(label="Start Meeting")
        self._meeting_btn.add_css_class("recording-btn")
        self._meeting_btn.add_css_class("idle")
        self._meeting_btn.set_tooltip_text("Record + transcribe meeting")
        self._meeting_btn.connect("clicked", self._on_meeting_toggle)
        header.pack_end(self._meeting_btn)

        self._notes_sidebar_btn = Gtk.ToggleButton(label="Meeting Notes")
        self._notes_sidebar_btn.add_css_class("flat")
        self._notes_sidebar_btn.set_tooltip_text("Show or hide live transcript and meeting history")
        self._notes_sidebar_btn.connect("toggled", self._on_meeting_sidebar_toggled)
        header.pack_end(self._notes_sidebar_btn)

        # Record button (video only)
        self._record_btn = Gtk.Button(label="Rec")
        self._record_btn.add_css_class("recording-btn")
        self._record_btn.add_css_class("idle")
        self._record_btn.set_tooltip_text("Record to MP4")
        self._record_btn.connect("clicked", self._on_record_toggle)
        header.pack_end(self._record_btn)

        # Profile selector
        self._profile_btn = Gtk.MenuButton(label="Profile")
        self._profile_btn.set_tooltip_text("Switch profile")
        profile_popover = Gtk.Popover()
        profile_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        profile_box.set_margin_top(8)
        profile_box.set_margin_bottom(8)
        profile_box.set_margin_start(8)
        profile_box.set_margin_end(8)

        # Built-in profiles
        from nvbroadcast.core.config import get_builtin_profiles, list_profiles
        for name, info in get_builtin_profiles().items():
            btn = Gtk.Button(label=f"{name}")
            btn.set_tooltip_text(info["description"])
            btn.add_css_class("flat")
            btn.connect("clicked", self._on_profile_selected, name, profile_popover)
            profile_box.append(btn)

        # Separator
        profile_box.append(Gtk.Separator())

        # User profiles
        for name in list_profiles():
            btn = Gtk.Button(label=f"{name}")
            btn.add_css_class("flat")
            btn.connect("clicked", self._on_user_profile_selected, name, profile_popover)
            profile_box.append(btn)

        # Save current as profile
        save_btn = Gtk.Button(label="Save Current as Profile...")
        save_btn.add_css_class("flat")
        save_btn.connect("clicked", self._on_save_profile, profile_popover)
        profile_box.append(Gtk.Separator())
        profile_box.append(save_btn)

        profile_popover.set_child(profile_box)
        self._profile_btn.set_popover(profile_popover)
        self._profile_popover_box = profile_box
        header.pack_start(self._profile_btn)

        # Quit button
        quit_btn = Gtk.Button(icon_name="application-exit-symbolic",
                              tooltip_text="Quit NVIDIA Broadcast")
        quit_btn.connect("clicked", lambda _: self._app.quit())
        header.pack_end(quit_btn)

        # About button
        about_btn = Gtk.Button(icon_name="help-about-symbolic",
                               tooltip_text="About")
        about_btn.connect("clicked", self._show_about)
        header.pack_end(about_btn)

        self._update_url = ""
        self._update_btn = Gtk.Button(label="Update Available")
        self._update_btn.add_css_class("suggested-action")
        self._update_btn.set_visible(False)
        self._update_btn.set_tooltip_text("Open the latest release download page")
        self._update_btn.connect("clicked", self._open_update_release)
        header.pack_end(self._update_btn)

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

        body_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)

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

        paned.set_position(450)
        body_box.append(paned)

        self._meeting_sidebar = self._build_meeting_sidebar()
        body_box.append(self._meeting_sidebar)
        main_box.append(body_box)

        footer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        self._install_revealer = Gtk.Revealer()
        self._install_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_UP)
        self._install_revealer.set_transition_duration(180)
        self._install_revealer.set_reveal_child(False)

        install_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        install_box.add_css_class("status-bar")
        install_text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        install_text.set_hexpand(True)

        self._install_title = Gtk.Label(label="")
        self._install_title.set_xalign(0)
        self._install_title.add_css_class("device-label")
        install_text.append(self._install_title)

        self._install_detail = Gtk.Label(label="")
        self._install_detail.set_xalign(0)
        self._install_detail.set_wrap(True)
        self._install_detail.set_ellipsize(3)
        install_text.append(self._install_detail)
        install_box.append(install_text)

        self._install_progress = Gtk.ProgressBar()
        self._install_progress.set_hexpand(True)
        install_box.append(self._install_progress)

        self._install_close_btn = Gtk.Button(label="Dismiss")
        self._install_close_btn.add_css_class("flat")
        self._install_close_btn.set_sensitive(False)
        self._install_close_btn.connect("clicked", self._dismiss_install_banner)
        install_box.append(self._install_close_btn)

        self._install_revealer.set_child(install_box)
        footer_box.append(self._install_revealer)

        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        status_box.add_css_class("status-bar")

        self._status_bar = Gtk.Label(label="Ready")
        self._status_bar.set_xalign(0)
        self._status_bar.set_hexpand(True)
        self._status_bar.set_ellipsize(3)
        status_box.append(self._status_bar)

        self._perf_label = Gtk.Label(label="")
        self._perf_label.add_css_class("status-gpu")
        status_box.append(self._perf_label)

        credit = Gtk.Label(label="by doczeus")
        credit.add_css_class("app-subtitle")
        status_box.append(credit)

        footer_box.append(status_box)
        main_box.append(footer_box)

        def _update_perf():
            pm = self._app.perf_monitor
            self._perf_label.set_text(pm.format_status())
            return True
        GLib.timeout_add(1000, _update_perf)

        self.set_content(main_box)

    def _build_camera_section(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        hdr = Gtk.Label(label="CAMERA")
        hdr.add_css_class("section-header")
        hdr.set_xalign(0)
        box.append(hdr)

        # Input settings in a compact card
        input_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

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
            {"name": "YUY2 — Chrome, Edge, Zoom, Discord, Meet", "device": "YUY2"},
            {"name": "I420 — Firefox, Teams, WebRTC (most compatible)", "device": "I420"},
            {"name": "NV12 — OBS, VLC, GStreamer apps", "device": "NV12"},
        ])
        input_card.append(self._format_selector)

        # Firefox compatibility toggle (only shown if Firefox is installed)
        if get_firefox_profiles():
            self._firefox_toggle = EffectToggle(
                "Firefox Mode", "Auto-configure Firefox for virtual camera"
            )
            pw_disabled = is_firefox_pipewire_disabled()
            self._firefox_toggle.active = pw_disabled if pw_disabled else False
            self._firefox_toggle.connect("toggled", self._on_firefox_toggled)
            input_card.append(self._firefox_toggle)
        else:
            self._firefox_toggle = None

        # FPS selector — show what the current resolution supports
        self._fps_selector = DeviceSelector("FPS")
        self._refresh_fps_options()
        self._fps_selector.connect("device-changed", self._on_fps_changed)
        input_card.append(self._fps_selector)

        box.append(_collapsible_card("Input", input_card, expanded=True))

        # Processing card (mode, GPU, mirror, edge refine)
        proc_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

        self._mode_devices = self._build_mode_devices()

        self._profile_selector = DeviceSelector("Mode")
        self._profile_selector.set_devices(self._mode_devices)

        # Map current config to the right index
        comp = self._app.config.compositing
        prof = self._app.config.performance_profile
        current_key = self._app.config.mode_key or self._profile_and_comp_to_mode(prof, comp)
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

        box.append(_collapsible_card("Processing", proc_card, expanded=True))

        # Background effect card
        bg_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._bg_toggle = EffectToggle("Background", "Remove, blur, or replace background")
        self._bg_toggle.connect("toggled", self._on_bg_toggled)
        bg_card.append(self._bg_toggle)

        # Model selector
        self._model_selector = DeviceSelector("Model")
        self._model_selector.set_devices([
            {"name": "RVM - Person Matting (fastest)", "device": "rvm"},
            {"name": "IS-Net - General Objects", "device": "isnet"},
            {"name": "BiRefNet - Best Quality (heavy, CPU fallback)", "device": "birefnet"},
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

        box.append(_collapsible_card("Background", bg_card, expanded=False))

        # Auto Frame card
        af_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._autoframe_toggle = EffectToggle("Auto Frame", "Track face and auto-zoom")
        self._autoframe_toggle.connect("toggled", self._on_autoframe_toggled)
        af_card.append(self._autoframe_toggle)
        self._zoom_slider = EffectSlider("Zoom", 1.5, 1.0, 3.0)
        self._zoom_slider.set_sensitive(False)
        self._zoom_slider.connect("value-changed", self._on_zoom_changed)
        af_card.append(self._zoom_slider)
        box.append(_collapsible_card("Auto Frame", af_card, expanded=False))

        # Eye Contact card
        ec_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._eye_contact_toggle = EffectToggle(
            "Eye Contact", "Redirect gaze to look at camera"
        )
        self._eye_contact_toggle.connect("toggled", self._on_eye_contact_toggled)
        ec_card.append(self._eye_contact_toggle)
        self._eye_contact_slider = EffectSlider("Intensity", 0.7, 0.0, 1.0)
        self._eye_contact_slider.set_sensitive(False)
        self._eye_contact_slider.connect("value-changed", self._on_eye_contact_intensity)
        ec_card.append(self._eye_contact_slider)
        box.append(_collapsible_card("Eye Contact", ec_card, expanded=False))

        # Face Relighting card
        rl_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._relighting_toggle = EffectToggle(
            "Face Relighting", "Match face lighting to background"
        )
        self._relighting_toggle.connect("toggled", self._on_relighting_toggled)
        rl_card.append(self._relighting_toggle)
        self._relighting_slider = EffectSlider("Intensity", 0.5, 0.0, 1.0)
        self._relighting_slider.set_sensitive(False)
        self._relighting_slider.connect("value-changed", self._on_relighting_intensity)
        rl_card.append(self._relighting_slider)
        box.append(_collapsible_card("Face Relighting", rl_card, expanded=False))

        # Video Enhancement card
        beauty_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
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

        box.append(_collapsible_card("Video Enhancement", beauty_card, expanded=False))

        return box

    def _build_audio_section(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        hdr = Gtk.Label(label="AUDIO")
        hdr.add_css_class("section-header")
        hdr.set_xalign(0)
        box.append(hdr)

        # Mic card — now with device selector + VU meter
        mic_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        # Mic device selector
        self._mic_selector = DeviceSelector("Microphone")
        self._mic_selector.connect("device-changed", self._on_mic_changed)
        mic_card.append(self._mic_selector)

        # VU meter (progress bar showing mic level)
        vu_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        vu_box.set_margin_start(16)
        vu_box.set_margin_end(16)
        vu_lbl = Gtk.Label(label="Level")
        vu_lbl.set_xalign(0)
        vu_lbl.set_size_request(80, -1)
        vu_lbl.add_css_class("device-label")
        vu_box.append(vu_lbl)
        self._vu_meter = Gtk.LevelBar()
        self._vu_meter.set_min_value(0.0)
        self._vu_meter.set_max_value(1.0)
        self._vu_meter.set_value(0.0)
        self._vu_meter.set_hexpand(True)
        self._vu_meter.add_css_class("vu-meter")
        vu_box.append(self._vu_meter)
        self._vu_db_label = Gtk.Label(label="-60 dB")
        self._vu_db_label.set_size_request(50, -1)
        self._vu_db_label.add_css_class("device-label")
        vu_box.append(self._vu_db_label)
        mic_card.append(vu_box)

        # Noise removal toggle + strength
        self._noise_toggle = EffectToggle("Noise Removal", "Remove background noise from mic")
        self._noise_toggle.connect("toggled", self._on_noise_toggled)
        mic_card.append(self._noise_toggle)
        self._noise_slider = EffectSlider("Strength", 1.0)
        self._noise_slider.set_sensitive(False)
        self._noise_slider.connect("value-changed", self._on_noise_intensity_changed)
        mic_card.append(self._noise_slider)
        box.append(_collapsible_card("Microphone", mic_card, expanded=True))

        # Voice Effects card
        vfx_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        self._vfx_toggle = EffectToggle("Voice Effects", "Bass, treble, warmth, compression")
        self._vfx_toggle.connect("toggled", self._on_vfx_toggled)
        vfx_card.append(self._vfx_toggle)

        # Preset selector
        self._vfx_preset = DeviceSelector("Preset")
        self._vfx_preset.set_devices([
            {"name": "Natural", "device": "Natural"},
            {"name": "Radio (warm + compressed)", "device": "Radio"},
            {"name": "Podcast (balanced)", "device": "Podcast"},
            {"name": "Deep Voice (bass heavy)", "device": "Deep Voice"},
            {"name": "Bright (treble boost)", "device": "Bright"},
            {"name": "Studio (professional)", "device": "Studio"},
        ])
        self._vfx_preset.set_sensitive(False)
        self._vfx_preset.connect("device-changed", self._on_vfx_preset)
        vfx_card.append(self._vfx_preset)

        # GPU/CPU mode
        self._vfx_gpu_toggle = EffectToggle("GPU Acceleration", "Use CUDA for audio processing")
        self._vfx_gpu_toggle.set_sensitive(False)
        self._vfx_gpu_toggle.connect("toggled", self._on_vfx_gpu_toggled)
        vfx_card.append(self._vfx_gpu_toggle)

        # Individual sliders
        self._vfx_sliders = {}
        for key, label, default in [
            ("bass_boost", "Bass", 0.0),
            ("treble", "Treble", 0.0),
            ("warmth", "Warmth", 0.0),
            ("compression", "Compression", 0.0),
            ("gate_threshold", "Noise Gate", 0.0),
            ("gain", "Output Gain", 0.0),
        ]:
            slider = EffectSlider(label, default, -1.0 if key in ("bass_boost", "treble", "gain") else 0.0, 1.0)
            slider.set_sensitive(False)
            slider.connect("value-changed", self._on_vfx_slider, key)
            vfx_card.append(slider)
            self._vfx_sliders[key] = slider

        box.append(_collapsible_card("Voice Effects", vfx_card, expanded=False))

        # Mic Test card
        test_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        test_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        test_row.set_margin_start(16)
        test_row.set_margin_end(16)

        self._test_rec_btn = Gtk.Button(label="Record 5s")
        self._test_rec_btn.add_css_class("flat")
        self._test_rec_btn.connect("clicked", self._on_test_record)
        test_row.append(self._test_rec_btn)

        self._test_play_btn = Gtk.Button(label="Play Back")
        self._test_play_btn.add_css_class("flat")
        self._test_play_btn.set_sensitive(False)
        self._test_play_btn.connect("clicked", self._on_test_play)
        test_row.append(self._test_play_btn)

        self._test_status = Gtk.Label(label="Ready")
        self._test_status.add_css_class("device-label")
        self._test_status.set_hexpand(True)
        self._test_status.set_xalign(0)
        test_row.append(self._test_status)

        test_card.append(test_row)
        box.append(_collapsible_card("Mic Test", test_card, expanded=False))

        # Speaker card
        spk_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        # Speaker device selector
        self._speaker_selector = DeviceSelector("Speaker")
        self._speaker_selector.connect("device-changed", self._on_speaker_changed)
        spk_card.append(self._speaker_selector)

        self._speaker_toggle = EffectToggle("Noise Removal", "Remove noise from incoming audio")
        self._speaker_toggle.connect("toggled", self._on_speaker_toggled)
        spk_card.append(self._speaker_toggle)
        box.append(_collapsible_card("Speakers", spk_card, expanded=True))

        box.append(Gtk.Box(vexpand=True))  # spacer

        # Populate audio devices
        self._populate_mics()
        self._populate_speakers()

        return box

    def _build_meeting_sidebar(self) -> Gtk.Widget:
        self._meeting_sidebar_revealer = Gtk.Revealer()
        self._meeting_sidebar_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_LEFT)
        self._meeting_sidebar_revealer.set_transition_duration(180)
        self._meeting_sidebar_revealer.set_reveal_child(False)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        outer.set_margin_top(8)
        outer.set_margin_bottom(8)
        outer.set_margin_end(12)
        outer.set_margin_start(8)
        outer.set_size_request(360, -1)
        outer.add_css_class("effect-card")

        title = Gtk.Label(label="Meeting Assistant")
        title.add_css_class("card-title")
        title.set_xalign(0)
        outer.append(title)

        subtitle = Gtk.Label(
            label="Live on-device transcript, notes preview, and meeting history. Sessions are kept for 7 days."
        )
        subtitle.set_wrap(True)
        subtitle.set_xalign(0)
        subtitle.add_css_class("dim-label")
        outer.append(subtitle)

        self._meeting_summary_label = Gtk.Label(label="No live meeting yet")
        self._meeting_summary_label.set_xalign(0)
        self._meeting_summary_label.set_wrap(True)
        outer.append(self._meeting_summary_label)

        self._meeting_live_view = Gtk.TextView()
        self._meeting_live_view.set_editable(False)
        self._meeting_live_view.set_cursor_visible(False)
        self._meeting_live_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._meeting_live_view.set_vexpand(True)
        live_scroll = Gtk.ScrolledWindow()
        live_scroll.set_vexpand(True)
        live_scroll.set_child(self._meeting_live_view)
        outer.append(live_scroll)

        history_title = Gtk.Label(label="Recent Meetings")
        history_title.add_css_class("device-label")
        history_title.set_xalign(0)
        outer.append(history_title)

        self._meeting_history = Gtk.ListBox()
        self._meeting_history.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._meeting_history.connect("row-selected", self._on_meeting_history_selected)
        history_scroll = Gtk.ScrolledWindow()
        history_scroll.set_min_content_height(220)
        history_scroll.set_child(self._meeting_history)
        outer.append(history_scroll)

        self._meeting_sidebar_revealer.set_child(outer)
        return self._meeting_sidebar_revealer

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
                return "doczeus"
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
        "doczeus":      ("max_quality", "cupy", False, True,  False),
        "cuda_max":     ("max_quality", "cupy", False, False, False),
        "cuda_balanced": ("balanced",   "cupy", False, False, False),
        "zeus":         ("balanced",    "cupy", True,  False, False),
        "killer":       ("performance", "cupy", True,  True,  True),
        "cuda_perf":    ("performance", "cupy", False, False, False),
        "cpu_quality":  ("max_quality", "cpu",  False, False, False),
        "cpu_light":    ("performance", "cpu",  False, False, False),
        "cpu_low":      ("potato",      "cpu",  False, False, False),
    }

    @staticmethod
    def _mode_status_message(mode_key: str) -> str:
        messages = {
            "doczeus": "DocZeus: best GPU quality for background replacement",
            "cuda_max": "CUDA High Quality: strong quality without fused compositing",
            "cuda_balanced": "CUDA Balanced: good quality with lighter GPU load",
            "zeus": "Zeus: fast GPU mode with TensorRT and edge refine",
            "killer": "Killer: fastest GPU mode, softer edges under motion",
            "cuda_perf": "CUDA Fast: lower GPU cost, reduced edge quality",
            "cpu_quality": "CPU High Quality: most compatible, highest CPU cost",
            "cpu_light": "CPU Fast: reduced CPU cost with lower quality",
            "cpu_low": "CPU Low End: fallback mode for weaker systems",
        }
        return messages.get(mode_key, "")

    def _build_mode_devices(self) -> list[dict[str, str]]:
        from nvbroadcast.core.config import detect_compositing_backends

        backends = detect_compositing_backends()
        has_cuda = backends.get("cupy", False)
        has_trt = has_tensorrt_runtime()
        devices: list[dict[str, str]] = []
        for mode_key, label in [
            ("doczeus", "DocZeus - Best Quality GPU"),
            ("cuda_max", "CUDA - High Quality"),
            ("cuda_balanced", "CUDA - Balanced"),
            ("zeus", "Zeus - Fast GPU Mode"),
            ("killer", "Killer - Fastest GPU Mode"),
            ("cuda_perf", "CUDA - Fast"),
            ("cpu_quality", "CPU - High Quality"),
            ("cpu_light", "CPU - Fast"),
            ("cpu_low", "CPU - Low End"),
        ]:
            unsupported = self._app.dependency_installer.unsupported_reason_for_mode(mode_key)
            missing = self._app.dependency_installer.missing_for_mode(mode_key)
            if unsupported:
                label += " (not available on this architecture)"
                devices.append({"name": label, "device": mode_key})
                continue
            if not has_cuda and mode_key.startswith(("doczeus", "cuda_", "zeus", "killer")):
                missing = sorted(set(missing + ["cupy"]))
            if mode_key in ("zeus", "killer") and not has_trt:
                missing = sorted(set(missing + ["tensorrt"]))
            if missing:
                readable = ", ".join(
                    self._app.dependency_installer.describe(dep)["title"] for dep in missing
                )
                label += f" (installs {readable})"
            devices.append({"name": label, "device": mode_key})
        return devices

    def _sync_mode_selector(self):
        mode_key = self._app.config.mode_key or self._profile_and_comp_to_mode(
            self._app.config.performance_profile,
            self._app.config.compositing,
        )
        for i, d in enumerate(self._mode_devices):
            if d["device"] == mode_key:
                self._profile_selector.set_selected_index(i)
                break

    def _on_mode_changed_selector(self, selector, mode_key):
        if getattr(self._app, '_restoring', False):
            return
        unsupported = self._app.dependency_installer.unsupported_reason_for_mode(mode_key)
        if unsupported:
            self._sync_mode_selector()
            self.set_status(unsupported)
            return
        install_key = self._app.dependency_installer.install_key_for_mode(mode_key)
        if install_key:
            self._pending_mode_key = mode_key
            self._sync_mode_selector()
            meta = self._app.dependency_installer.describe(install_key)
            self._prompt_dependency_install(
                install_key,
                title="Install required runtime?",
                reason=(
                    f"{self._mode_status_message(mode_key)}\n\n"
                    f"This mode needs {meta['title']} ({meta['size']}). "
                    "The download runs in the background and the rest of the app stays usable."
                ),
            )
            return
        if mode_key in self._MODE_MAP:
            profile, comp, trt, fused, nvdec = self._MODE_MAP[mode_key]
            self._app.set_performance_profile(
                profile, compositing=comp,
                use_tensorrt=trt, use_fused_kernel=fused, use_nvdec=nvdec,
                mode_key=mode_key,
            )
            msg = self._mode_status_message(mode_key)
            if msg:
                self.set_status(msg)
            # Show edge refine toggle only for premium speed modes
            is_premium = mode_key in ("killer", "zeus")
            self._edge_refine_toggle.set_visible(is_premium)
            self._edge_refine_toggle.set_sensitive(is_premium)
            desired = is_premium and self._app.config.premium_edge_refine
            if self._edge_refine_toggle.active != desired:
                self._edge_refine_toggle.active = desired

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

    def _on_firefox_toggled(self, t, active):
        ok, msg = set_firefox_pipewire(disabled=active)
        if ok:
            self.set_status(msg)
            # Auto-switch format to I420 for Firefox when enabled
            if active:
                for i, d in enumerate(self._format_selector._devices):
                    if d["device"] == "I420":
                        self._format_selector.set_selected_index(i)
                        break
        else:
            self.set_status(f"Firefox config failed: {msg}")

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
        self._app.config.video.beauty.preset = preset_key
        save_config(self._app.config)
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

    # --- Meeting ---
    def _on_meeting_toggle(self, btn):
        if self._app.meeting_active:
            notes_path = self._app.stop_meeting()
            self._meeting_btn.set_label("Start Meeting")
            self._meeting_btn.add_css_class("idle")
            self._meeting_btn.remove_css_class("recording-btn")
            if notes_path:
                self.set_status(f"Meeting saved: {notes_path}")
            else:
                self.set_status("Meeting ended")
        else:
            if not self._app.dependency_installer.is_available("whisper"):
                self._pending_meeting_start = True
                self._prompt_dependency_install(
                    "whisper",
                    title="Install meeting transcription files?",
                    reason=(
                        "Meeting transcription is optional and runs locally. "
                        "Install Whisper now to enable Start Meeting with transcription."
                    ),
                )
                return
            filepath = self._app.start_meeting()
            self._meeting_btn.set_label("End Meeting")
            self._meeting_btn.remove_css_class("idle")
            self._meeting_btn.add_css_class("recording-btn")
            self.set_status(f"Meeting recording: {filepath}")

    # --- Mic Selection ---
    def _populate_mics(self):
        try:
            mics = self._app.list_microphones()
            self._mic_selector.set_devices(mics)
            # Select saved mic
            saved = self._app.config.audio.mic_device
            if saved:
                for i, m in enumerate(mics):
                    if m["device"] == saved:
                        self._mic_selector.set_selected_index(i)
                        break
        except Exception as e:
            print(f"[NV Broadcast] Mic enumeration failed: {e}")

    def _on_mic_changed(self, selector, device):
        self._app.set_microphone(device)
        self.set_status(f"Microphone: {device}")

    def _populate_speakers(self):
        try:
            from nvbroadcast.audio.devices import list_speakers
            spk = list_speakers()
            self._speaker_selector.set_devices(spk)
        except Exception as e:
            print(f"[NV Broadcast] Speaker enumeration failed: {e}")

    def _on_speaker_changed(self, selector, device):
        self.set_status(f"Speaker: {device}")

    def _on_meeting_sidebar_toggled(self, btn):
        self._meeting_sidebar_revealer.set_reveal_child(btn.get_active())

    def reset_live_meeting_view(self):
        self._notes_sidebar_btn.set_active(True)
        self._meeting_summary_label.set_text("Meeting in progress. Transcript updates below.")
        buf = self._meeting_live_view.get_buffer()
        buf.set_text("")

    def update_live_meeting_summary(self, summary: str, transcript: str):
        self._notes_sidebar_btn.set_active(True)
        self._meeting_summary_label.set_text(summary or "Listening...")
        buf = self._meeting_live_view.get_buffer()
        buf.set_text(transcript)

    def load_meeting_sessions(self, sessions):
        while True:
            row = self._meeting_history.get_row_at_index(0)
            if row is None:
                break
            self._meeting_history.remove(row)

        for session in sessions:
            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            row_box.set_margin_top(6)
            row_box.set_margin_bottom(6)
            row_box.set_margin_start(6)
            row_box.set_margin_end(6)

            title = Gtk.Label(label=session.title[:80] or session.session_id)
            title.set_xalign(0)
            title.add_css_class("device-label")
            row_box.append(title)

            summary = Gtk.Label(label=session.summary[:140] or "No summary")
            summary.set_xalign(0)
            summary.set_wrap(True)
            summary.add_css_class("dim-label")
            row_box.append(summary)

            row = Gtk.ListBoxRow()
            row.set_child(row_box)
            row._meeting_session = session
            self._meeting_history.append(row)

    def show_meeting_session(self, session):
        self._notes_sidebar_btn.set_active(True)
        self._meeting_summary_label.set_text(session.summary or session.title)
        contents = self._app.load_meeting_file(session.notes_path) or self._app.load_meeting_file(session.transcript_path)
        buf = self._meeting_live_view.get_buffer()
        buf.set_text(contents)

    def _on_meeting_history_selected(self, _listbox, row):
        if row is None or not hasattr(row, "_meeting_session"):
            return
        self.show_meeting_session(row._meeting_session)

    # --- Voice Effects ---
    def _on_vfx_toggled(self, t, active):
        if self._app._audio_pipeline:
            self._app._audio_pipeline.voice_fx.enabled = active
        self._vfx_preset.set_sensitive(active)
        self._vfx_gpu_toggle.set_sensitive(active)
        for slider in self._vfx_sliders.values():
            slider.set_sensitive(active)

    def _on_vfx_preset(self, selector, preset_name):
        from nvbroadcast.audio.voice_fx import VOICE_PRESETS
        if preset_name in VOICE_PRESETS:
            preset = VOICE_PRESETS[preset_name]
            if self._app._audio_pipeline:
                self._app._audio_pipeline.voice_fx.settings = preset
            # Update sliders
            for key, slider in self._vfx_sliders.items():
                slider._scale.set_value(getattr(preset, key, 0.0))

    def _on_vfx_gpu_toggled(self, t, active):
        if self._app._audio_pipeline:
            self._app._audio_pipeline.voice_fx.use_gpu = active

    def _on_vfx_slider(self, slider, value, key):
        if self._app._audio_pipeline:
            setattr(self._app._audio_pipeline.voice_fx.settings, key, value)

    # --- Mic Test ---
    def _on_test_record(self, btn):
        from nvbroadcast.audio.mic_test import MicTest
        if not hasattr(self, '_mic_test'):
            self._mic_test = MicTest()

        if self._mic_test.is_recording:
            return

        mic = self._mic_selector.get_selected_device() or ""
        self._test_rec_btn.set_sensitive(False)
        self._test_play_btn.set_sensitive(False)
        self._test_status.set_text("Recording...")

        def on_done():
            self._test_rec_btn.set_sensitive(True)
            self._test_play_btn.set_sensitive(True)
            self._test_status.set_text("Recorded. Click Play.")

        self._mic_test.start_recording(mic, duration=5, on_complete=on_done)

    def _on_test_play(self, btn):
        if not hasattr(self, '_mic_test'):
            return

        self._test_play_btn.set_sensitive(False)
        self._test_status.set_text("Playing...")

        def on_done():
            self._test_play_btn.set_sensitive(True)
            self._test_status.set_text("Ready")

        self._mic_test.play_recording(on_complete=on_done)

    # --- VU Meter ---
    def _start_vu_meter(self):
        """Poll audio level and update VU meter every 100ms."""
        from gi.repository import GLib
        def _update():
            if self._app._audio_pipeline and hasattr(self._app._audio_pipeline, '_level_monitor'):
                mon = self._app._audio_pipeline._level_monitor
                if mon:
                    self._vu_meter.set_value(mon.level_normalized)
                    self._vu_db_label.set_text(f"{mon.level_db:.0f} dB")
            return True  # Keep polling
        GLib.timeout_add(100, _update)

    # --- Eye Contact ---
    def _on_eye_contact_toggled(self, t, active):
        self._app.set_eye_contact(active)
        self._eye_contact_slider.set_sensitive(active)
        self._app.config.video.eye_contact = active
        save_config(self._app.config)

    def _on_eye_contact_intensity(self, s, v):
        self._app.set_eye_contact_intensity(v)
        self._app.config.video.eye_contact_intensity = v
        save_config(self._app.config)

    # --- Face Relighting ---
    def _on_relighting_toggled(self, t, active):
        self._app.set_relighting(active)
        self._relighting_slider.set_sensitive(active)
        self._app.config.video.relighting = active
        save_config(self._app.config)

    def _on_relighting_intensity(self, s, v):
        self._app.set_relighting_intensity(v)
        self._app.config.video.relighting_intensity = v
        save_config(self._app.config)

    # --- Recording ---
    def _on_record_toggle(self, btn):
        if self._app.is_recording:
            self._app.stop_recording()
            self._record_btn.set_label("Rec")
            self._record_btn.add_css_class("idle")
            self._record_btn.remove_css_class("recording-btn")
            self.set_status("Recording saved")
        else:
            filepath = self._app.start_recording()
            self._record_btn.set_label("Stop Rec")
            self._record_btn.remove_css_class("idle")
            self._record_btn.add_css_class("recording-btn")
            self.set_status(f"Recording to {filepath}")

    # --- Profiles ---
    def _on_profile_selected(self, btn, name, popover):
        from nvbroadcast.core.config import apply_builtin_profile, save_config
        apply_builtin_profile(self._app.config, name)
        save_config(self._app.config)
        self._apply_config_to_ui(self._app.config)
        self._profile_btn.set_label(f"Profile: {name}")
        popover.popdown()
        self.set_status(f"Switched to {name} profile")

    def _on_user_profile_selected(self, btn, name, popover):
        from nvbroadcast.core.config import load_profile, save_config
        loaded = load_profile(name)
        if loaded:
            self._app.config = loaded
            save_config(loaded)
            self._apply_config_to_ui(loaded)
            self._profile_btn.set_label(f"Profile: {name}")
            popover.popdown()
            self.set_status(f"Switched to {name} profile")

    def _on_save_profile(self, btn, popover):
        popover.popdown()
        # Show a simple dialog to get profile name
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text="Save Profile",
            secondary_text="Enter a name for this profile:",
        )
        entry = Gtk.Entry()
        entry.set_placeholder_text("e.g. My Meeting Setup")
        dialog.get_content_area().append(entry)
        dialog.connect("response", self._on_save_profile_response, entry)
        dialog.present()

    def _on_save_profile_response(self, dialog, response, entry):
        if response == Gtk.ResponseType.OK:
            name = entry.get_text().strip()
            if name:
                from nvbroadcast.core.config import save_profile
                save_profile(name, self._app.config)
                self.set_status(f"Profile saved: {name}")
        dialog.destroy()

    def _apply_config_to_ui(self, config):
        """Apply a config to all UI controls (after profile switch)."""
        v = config.video
        # Eye contact
        self._eye_contact_toggle.active = v.eye_contact
        self._eye_contact_slider.set_sensitive(v.eye_contact)
        self._eye_contact_slider._scale.set_value(v.eye_contact_intensity)
        # Relighting
        self._relighting_toggle.active = v.relighting
        self._relighting_slider.set_sensitive(v.relighting)
        self._relighting_slider._scale.set_value(v.relighting_intensity)
        # Background
        self._bg_toggle.active = v.background_removal
        # Mirror
        self._mirror_toggle.active = v.mirror
        # Auto frame
        self._autoframe_toggle.active = v.auto_frame
        self._zoom_slider._scale.set_value(v.auto_frame_zoom)
        # Beauty
        self._beauty_toggle.active = v.beauty.enabled
        self._beauty_preset.set_sensitive(v.beauty.enabled)
        preset_map = {"natural": 0, "broadcast": 1, "glamour": 2, "custom": 3}
        if v.beauty.preset in preset_map:
            self._beauty_preset.set_selected_index(preset_map[v.beauty.preset])
        for key, ctrl in self._beauty_controls.items():
            value = float(getattr(v.beauty, key))
            ctrl["toggle"].active = value > 0.0
            ctrl["toggle"].set_sensitive(v.beauty.enabled)
            ctrl["slider"]._scale.set_value(value)
            ctrl["slider"].set_sensitive(v.beauty.enabled and value > 0.0)
        # Apply effects to backend
        self._app._eye_contact.enabled = v.eye_contact
        self._app._eye_contact.intensity = v.eye_contact_intensity
        self._app._relighter.enabled = v.relighting
        self._app._relighter.intensity = v.relighting_intensity
        self._app._mirror = v.mirror
        self._app._video_effects.enabled = v.background_removal
        self._app._beautifier.enabled = v.beauty.enabled
        self._app._beautifier.skin_smooth = v.beauty.skin_smooth
        self._app._beautifier.denoise = v.beauty.denoise
        self._app._beautifier.enhance = v.beauty.enhance
        self._app._beautifier.sharpen = v.beauty.sharpen
        self._app._beautifier.edge_darken = v.beauty.edge_darken
        self._app._autoframe.enabled = v.auto_frame
        self._app._update_pipeline_mode()
        print(f"[NV Broadcast] Profile applied: bg={v.background_removal}, eye={v.eye_contact}, relight={v.relighting}, beauty={v.beauty.enabled}")

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
        mode_key = config.mode_key or self._profile_and_comp_to_mode(
            config.performance_profile, config.compositing
        )

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

        for i, d in enumerate(self._mode_devices):
            if d["device"] == mode_key:
                self._profile_selector.set_selected_index(i)
                break

        is_premium = mode_key in ("killer", "zeus")
        self._edge_refine_toggle.set_visible(is_premium)
        self._edge_refine_toggle.set_sensitive(is_premium)
        desired = is_premium and config.premium_edge_refine
        if self._edge_refine_toggle.active != desired:
            self._edge_refine_toggle.active = desired

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

        # Eye contact
        if v.eye_contact:
            self._eye_contact_toggle.active = True
            self._eye_contact_slider.set_sensitive(True)
        self._eye_contact_slider._scale.set_value(v.eye_contact_intensity)

        # Relighting
        if v.relighting:
            self._relighting_toggle.active = True
            self._relighting_slider.set_sensitive(True)
        self._relighting_slider._scale.set_value(v.relighting_intensity)

        # Beauty
        self._beauty_toggle.active = v.beauty.enabled
        self._beauty_preset.set_sensitive(v.beauty.enabled)
        preset_map = {"natural": 0, "broadcast": 1, "glamour": 2, "custom": 3}
        if v.beauty.preset in preset_map:
            self._beauty_preset.set_selected_index(preset_map[v.beauty.preset])
        for key, ctrl in self._beauty_controls.items():
            value = float(getattr(v.beauty, key))
            ctrl["toggle"].active = value > 0.0
            ctrl["toggle"].set_sensitive(v.beauty.enabled)
            ctrl["slider"]._scale.set_value(value)
            ctrl["slider"].set_sensitive(v.beauty.enabled and value > 0.0)

        # Mirror
        self._mirror_toggle.active = v.mirror

    def _show_about(self, button):
        # Load app icon from installed assets or source tree.
        icon_path = find_app_icon()
        icon_name = "com.doczeus.NVBroadcast"
        if icon_path is not None and icon_path.exists():
            # Register the icon with GTK's icon theme so AboutWindow can find it
            display = self.get_display()
            if display:
                icon_theme = Gtk.IconTheme.get_for_display(display)
                icon_theme.add_search_path(str(icon_path.parent))

        about = Adw.AboutWindow(
            transient_for=self,
            application_name=APP_NAME,
            application_icon=icon_name,
            version=__import__("nvbroadcast").__version__,
            developer_name="doczeus",
            website="https://github.com/Hkshoonya/nvidia-broadcast-linux",
            issue_url="https://github.com/Hkshoonya/nvidia-broadcast-linux/issues",
            license_type=Gtk.License.GPL_3_0,
            copyright="Copyright (c) 2026 doczeus",
            developers=["doczeus https://github.com/Hkshoonya"],
            comments=(
                "Unofficial NVIDIA Broadcast for Linux and other OS.\n\n"
                "AI-powered virtual camera with background removal, blur, "
                "replacement, auto-framing, video enhancement, and noise "
                "cancellation using GPU-accelerated deep learning.\n\n"
                "9 processing modes including Killer, Zeus, and DocZeus "
                "with fused CUDA kernels and edge refinement.\n\n"
                "Created by doczeus | AI Powered"
            ),
        )
        about.present()

    def _open_update_release(self, button):
        if not self._update_url:
            return
        Gio.AppInfo.launch_default_for_uri(self._update_url, None)

    def set_update_available(self, version: str, url: str):
        self._update_url = url
        self._update_btn.set_label(f"Update v{version}")
        self._update_btn.set_tooltip_text(f"Open release v{version}")
        self._update_btn.set_visible(True)

    def rebuild_mode_selector(self, compositing: str, profile: str):
        """Rebuild the Mode dropdown with currently available backends."""
        _ = compositing, profile
        self._mode_devices = self._build_mode_devices()
        self._profile_selector.set_devices(self._mode_devices)
        self._sync_mode_selector()

    def _on_freeze_toggled(self, btn):
        self._preview_frozen = btn.get_active()
        btn.set_label("Resume View" if self._preview_frozen else "Pause View")
        # Pause the actual pipeline (freeze vcam output + skip processing)
        if self._app._video_pipeline:
            self._app._video_pipeline.set_paused(self._preview_frozen)

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

    def bind_dependency_installer(self, installer):
        self._installer = installer
        installer.connect("job-started", self._on_install_job_started)
        installer.connect("job-progress", self._on_install_job_progress)
        installer.connect("job-completed", self._on_install_job_completed)

    def _prompt_dependency_install(self, install_key: str, title: str, reason: str):
        meta = self._app.dependency_installer.describe(install_key)
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.NONE,
            text=title,
            secondary_text=(
                f"{reason}\n\n"
                f"Package: {meta['title']}\n"
                f"Download size: {meta['size']}\n"
                f"{meta['summary']}\n\n"
                "Install runs in the background. Skip keeps the app on the current working setup."
            ),
        )
        dialog.add_button("Skip", Gtk.ResponseType.CANCEL)
        dialog.add_button("Install", Gtk.ResponseType.OK)
        dialog.connect("response", self._on_dependency_install_response, install_key)
        dialog.present()

    def _on_dependency_install_response(self, dialog, response, install_key: str):
        dialog.destroy()
        if response != Gtk.ResponseType.OK:
            self._pending_meeting_start = False
            self._pending_mode_key = ""
            self.set_status("Optional runtime install skipped")
            return
        if not self._app.dependency_installer.start_install(install_key):
            self.set_status("Another optional runtime install is already running")

    def _dismiss_install_banner(self, _button):
        self._stop_install_pulse()
        self._install_revealer.set_reveal_child(False)

    def _start_install_pulse(self):
        if self._install_pulse_id:
            return

        def _pulse():
            self._install_progress.pulse()
            return self._install_revealer.get_reveal_child()

        self._install_pulse_id = GLib.timeout_add(120, _pulse)

    def _stop_install_pulse(self):
        if self._install_pulse_id:
            GLib.source_remove(self._install_pulse_id)
            self._install_pulse_id = 0

    def _on_install_job_started(self, _installer, key: str, text: str):
        meta = self._app.dependency_installer.describe(key)
        self._install_title.set_text(meta["title"])
        self._install_detail.set_text(text)
        self._install_progress.set_fraction(0.0)
        self._install_close_btn.set_sensitive(False)
        self._install_revealer.set_reveal_child(True)
        self._start_install_pulse()
        self.set_status(text)

    def _on_install_job_progress(self, _installer, _key: str, text: str, fraction: float):
        self._install_detail.set_text(text)
        if fraction >= 0.0:
            self._stop_install_pulse()
            self._install_progress.set_fraction(fraction)
        else:
            self._start_install_pulse()
        self.set_status(text)

    def _on_install_job_completed(self, _installer, _key: str, success: bool, text: str):
        self._stop_install_pulse()
        self._install_progress.set_fraction(1.0 if success else 0.0)
        self._install_detail.set_text(text)
        self._install_close_btn.set_sensitive(True)
        self.set_status(text)
        self.rebuild_mode_selector(self._app.config.compositing, self._app.config.performance_profile)

        if success and self._pending_mode_key:
            pending_mode = self._pending_mode_key
            self._pending_mode_key = ""
            for i, d in enumerate(self._mode_devices):
                if d["device"] == pending_mode:
                    self._profile_selector.set_selected_index(i)
                    break
            self._on_mode_changed_selector(self._profile_selector, pending_mode)
        else:
            self._pending_mode_key = ""

        if success and self._pending_meeting_start:
            self._pending_meeting_start = False
            filepath = self._app.start_meeting()
            self._meeting_btn.set_label("End Meeting")
            self._meeting_btn.remove_css_class("idle")
            self._meeting_btn.add_css_class("recording-btn")
            self.set_status(f"Meeting recording: {filepath}")
        else:
            self._pending_meeting_start = False

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""NVIDIA Broadcast - setup once and forget.

Auto-starts broadcast on launch, restores all saved settings,
minimizes to background on close. Browser picks up virtual camera automatically.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gst", "1.0")
from gi.repository import Gtk, Adw, Gst, Gio, Gdk, GLib

from nvbroadcast.core.constants import (
    APP_ID,
    COMPUTE_GPU_INDEX,
    VIRTUAL_CAM_DEVICE,
)
from nvbroadcast.core import startup_trace
from nvbroadcast.core.gpu import apply_cuda_blocking_sync
from nvbroadcast.core.config import load_config, save_config
from nvbroadcast.core.updates import (
    fetch_latest_release,
    is_newer_version,
    resolve_update_target,
    should_check_for_updates,
)
from nvbroadcast.video.pipeline import VideoPipeline
from nvbroadcast.video.effects import VideoEffects
from nvbroadcast.video.autoframe import AutoFrame
from nvbroadcast.video.beautify import FaceBeautifier
from nvbroadcast.video.virtual_camera import (
    ensure_virtual_camera,
    is_v4l2loopback_device,
    v4l2loopback_modprobe_command,
)
from nvbroadcast.video.eye_contact import EyeContactCorrector
from nvbroadcast.video.relighting import FaceRelighter
from nvbroadcast.video.face_landmarks import get_shared_landmarker
from nvbroadcast.video.perf_monitor import PerfMonitor
from nvbroadcast.video.vcam_monitor import VcamConsumerMonitor
from nvbroadcast.ai.transcriber import MeetingTranscriber, save_transcript
from nvbroadcast.ai.summarizer import MeetingSummarizer
from nvbroadcast.core.platform import (
    IS_MACOS,
    IS_LINUX,
    IS_ARM64,
    legacy_tray_enabled,
    python_runtime_advisory,
)
from nvbroadcast.core.resources import find_ui_css
from nvbroadcast.core.dependency_installer import DependencyInstaller
from nvbroadcast.core.meeting_store import (
    create_session, save_session, list_sessions, MeetingSession, cleanup_old_sessions,
)
from nvbroadcast.audio.pipeline import AudioPipeline
from nvbroadcast.audio.monitor import SpeakerMonitor
from nvbroadcast.audio.meeting_capture import MeetingAudioCapture
from nvbroadcast.audio.virtual_mic import has_virtual_mic_backend
from nvbroadcast.ui.window import NVBroadcastWindow
from nvbroadcast import __version__

_AUTO_MODE_TARGET_FPS = {
    "doczeus": 22.0,
    "cuda_balanced": 20.0,
    "cuda_perf": 16.0,
    "cpu_quality": 18.0,
    "cpu_light": 14.0,
    "cpu_low": 10.0,
}

_COMPUTE_FOCUS_VALUES = {"auto", "gpu", "cpu"}
_COMPUTE_FOCUS_LABELS = {
    "auto": "Auto",
    "gpu": "GPU Focused",
    "cpu": "CPU Focused",
}

_GPU_AUTO_MODES = ("doczeus", "cuda_balanced", "cuda_perf")
_CPU_AUTO_MODES = ("cpu_quality", "cpu_light", "cpu_low")
_AUTO_MODE_ORDER = (*_GPU_AUTO_MODES, *_CPU_AUTO_MODES)

_MODE_LABELS = {
    "doczeus": "DocZeus - Best Quality GPU",
    "cuda_max": "CUDA - High Quality",
    "cuda_balanced": "CUDA - Balanced",
    "zeus": "Zeus - Fast GPU Mode",
    "killer": "Killer - Fastest GPU Mode",
    "cuda_perf": "CUDA - Fast",
    "cpu_quality": "CPU - High Quality",
    "cpu_light": "CPU - Fast",
    "cpu_low": "CPU - Low End",
}

_MODE_QUALITY_PRESETS = {
    "doczeus": "ultra",
    "cuda_max": "quality",
    "cuda_balanced": "balanced",
    "zeus": "balanced",
    "killer": "performance",
    "cuda_perf": "performance",
    "cpu_quality": "quality",
    "cpu_light": "performance",
    "cpu_low": "performance",
}


class NVBroadcastApp(Adw.Application):
    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        # Must precede the first cupy/ORT CUDA call (primary-context creation)
        # or GPU waits keep busy-spinning a core.
        if apply_cuda_blocking_sync():
            print("[NV Broadcast] CUDA blocking-sync enabled "
                  "(NVBROADCAST_CUDA_SYNC=spin restores spin-wait)")
        self.config = load_config()
        if IS_LINUX and IS_ARM64 and self.config.mode_key in {
            "doczeus", "cuda_max", "cuda_balanced", "cuda_perf", "zeus", "killer",
        }:
            self.config.mode_key = "cpu_quality"
            self.config.compute_focus = "cpu"
            self.config.compositing = "cpu"
            self.config.performance_profile = "max_quality"
            self.config.use_tensorrt = False
            self.config.use_fused_kernel = False
            self.config.use_nvdec = False
        self._window = None
        self._video_pipeline = None
        self._audio_pipeline = None
        self._gpu_frame_path = None
        self._gpu_frame_path_failed = False
        self._speaker_monitor = None
        self._video_effects = VideoEffects(
            gpu_index=self.config.compute_gpu,
            edge_config=self.config.video.edge,
            compositing=self.config.compositing,
        )
        self._autoframe = AutoFrame(gpu_index=self.config.compute_gpu)
        self._beautifier = FaceBeautifier(compositing=self.config.compositing)
        self._eye_contact = EyeContactCorrector()
        self._relighter = FaceRelighter()
        self._perf_monitor = PerfMonitor(gpu_index=self.config.compute_gpu)
        live_transcriber_model = os.getenv(
            "NVBROADCAST_TRANSCRIBER_MODEL",
            "base" if IS_ARM64 else "small",
        ).strip() or "base"
        final_transcriber_model = os.getenv(
            "NVBROADCAST_TRANSCRIBER_FINAL_MODEL",
            "small" if IS_ARM64 else "small",
        ).strip() or live_transcriber_model
        self._transcriber = MeetingTranscriber(
            model_size=live_transcriber_model,
            final_model_size=final_transcriber_model,
        )
        self._summarizer = MeetingSummarizer()
        self._dependency_installer = DependencyInstaller()
        self._meeting_capture = None
        self._meeting_session_id = ""
        self._meeting_session_dir = None
        self._meeting_audio_path = ""
        self._meeting_video_path = ""
        self._meeting_active = False
        self._meeting_finalizing = False
        self._transcriber_preload_started = False
        self._vcam_device = None
        self._vcam_available = False
        self._vcam_monitor = None  # v4l2 client-usage watcher (Linux)
        self._mirror = True  # Default: mirror (like looking in a mirror)
        self._tray = None
        self._legacy_tray_enabled = legacy_tray_enabled()
        self._vcam_consumers = 0  # Track virtual camera consumers
        self._idle_active = False   # Camera power save engaged
        self._idle_strikes = 0      # Consecutive no-consumer polls
        self._streaming = False
        self._use_nvdec = self.config.use_nvdec
        self._inline_inference = self.config.performance_profile in ("max_quality", "balanced")
        self._update_release = None
        self._pending_start = None
        self._restart_source_id = 0
        self._pipeline_teardown = None
        self._auto_tune_low_streak = 0
        self._auto_tune_high_streak = 0
        self._last_auto_tune_change = 0.0
        self._manual_low_fps_streak = 0
        self._last_manual_warning = 0.0
        self._last_auto_capture_change = 0.0
        self._hotkey_manager = None
        self._hotkey_active = False
        self._hotkey_status = "Global hotkeys are unavailable"
        self._hotkey_display: dict[str, str] = {}
        self._transcriber.set_segment_callback(self._on_transcript_segment)

    def do_startup(self):
        startup_trace.mark("do_startup begin")
        Adw.Application.do_startup(self)
        self._register_global_hotkeys()
        Gst.init(None)
        startup_trace.mark("Gst.init done")
        cleanup_old_sessions()
        Adw.StyleManager.get_default().set_color_scheme(Adw.ColorScheme.DEFAULT)

        # Load CSS
        css_provider = Gtk.CssProvider()
        css_path = find_ui_css()
        if css_path is not None and css_path.exists():
            css_provider.load_from_path(str(css_path))
            display = Gdk.Display.get_default()
            if display:
                Gtk.StyleContext.add_provider_for_display(
                    display, css_provider,
                    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
                )

        self._stop_headless_vcam_service()
        try:
            self._vcam_device = ensure_virtual_camera(self._preferred_vcam_device())
            self._vcam_available = True
        except RuntimeError as e:
            print(f"[NV Broadcast] Virtual camera unavailable: {e}")
        if self._vcam_available and not IS_MACOS:
            # Started here — before any pipeline can open the device — so
            # the monitor's own-fd baseline cannot race our own opens, and
            # kept for the whole session so consumers stay visible across
            # pipeline restarts.
            monitor = VcamConsumerMonitor(
                self._vcam_device, wake_callback=self._on_vcam_consumer_wake)
            if monitor.start():
                self._vcam_monitor = monitor
                print("[NV Broadcast] Camera power save: v4l2 client-usage "
                      "monitor active", flush=True)
            else:
                print("[NV Broadcast] Camera power save: v4l2 events "
                      "unavailable, falling back to fuser", flush=True)
        startup_trace.mark("do_startup end (virtual camera ready)")

    def _preferred_vcam_device(self) -> str | None:
        if not IS_LINUX:
            return None
        configured = (self.config.video.vcam_device or "").strip()
        if not configured or configured == VIRTUAL_CAM_DEVICE:
            return None
        return configured

    def _active_vcam_device(self) -> str:
        return self._vcam_device or self.config.video.vcam_device or VIRTUAL_CAM_DEVICE

    def _stop_headless_vcam_service(self) -> bool:
        """Stop the optional headless passthrough service before GUI capture.

        The GUI owns the physical camera and v4l2loopback sink while it is
        running. If the headless service is still active from login, both
        processes race for `/dev/video*` and the GUI starts with a white
        preview or a busy virtual camera.
        """
        if not IS_LINUX:
            return False

        service = "nvbroadcast-vcam.service"
        try:
            active = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", service],
                check=False,
                timeout=1,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

        if active.returncode != 0:
            return False

        try:
            subprocess.run(
                ["systemctl", "--user", "stop", service],
                check=False,
                timeout=3,
            )
            time.sleep(0.2)
            print(
                "[NV Broadcast] Stopped headless virtual camera service; "
                "GUI will own the camera while open",
                flush=True,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def do_activate(self):
        startup_trace.mark("do_activate begin")
        if self._window is None:
            self._window = NVBroadcastWindow(self)
            startup_trace.mark("window constructed")
            self._window.bind_dependency_installer(self._dependency_installer)
            self._window.load_meeting_sessions(self.list_meeting_sessions())

            # Native SNI (StatusNotifierItem) tray — pure D-Bus, safe in a
            # GTK4 process, works on KDE/Hyprland/waybar/quickshell.
            try:
                from nvbroadcast.ui.sni_tray import SniTray
                self._tray = SniTray(self)
            except Exception as e:
                print(f"[NV Broadcast] SNI tray failed: {e}")
                self._tray = None

            # Legacy GTK3 AppIndicator tray as fallback. Mixing GTK3 tray
            # code into this GTK4 app can terminate startup natively on some
            # Linux desktops without a Python traceback, so it stays gated.
            if (self._tray is None or not getattr(self._tray, "bus_ready", False)) \
                    and self._legacy_tray_enabled:
                try:
                    from nvbroadcast.ui.tray import TrayIcon
                    self._tray = TrayIcon(self)
                    if self._tray.available:
                        print("[NV Broadcast] System tray icon active (legacy)")
                except Exception as e:
                    print(f"[NV Broadcast] Tray icon not available: {e}")

            # Preview textures are only worth building while the window is
            # visible; each tick otherwise costs a full-frame copy plus a
            # GTK GPU upload.
            def _on_window_mapped(*_a):
                if self._video_pipeline is not None:
                    self._video_pipeline.set_preview_enabled(True)

            def _on_window_unmapped(*_a):
                if self._video_pipeline is not None:
                    self._video_pipeline.set_preview_enabled(False)

            self._window.connect("map", _on_window_mapped)
            self._window.connect("unmap", _on_window_unmapped)

            # Showing the window must instantly wake camera power save.
            self._window.connect("map", lambda *a: (
                self._exit_idle("window shown") if self._idle_active else None))

            # Camera power save: poll for vcam consumers. Seconds-granularity
            # so GLib can coalesce the wakeup; the 1s _idle_wake_tick handles
            # fast wake-from-idle, this poll only latches idle entry.
            GLib.timeout_add_seconds(10, self._check_vcam_consumers)

            # Start performance monitor
            self._perf_monitor.start()
            GLib.timeout_add(2500, self._auto_tune_tick)

            # Intercept window close -> minimize to background instead of quit
            self._window.connect("close-request", self._on_close_request)

            # Restore saved settings to UI (guard prevents toggle callbacks
            # from resetting effect states during restore)
            self._restoring = True
            self._restore_settings()
            startup_trace.mark("settings restored")
            if self.config.auto_mode:
                self.set_auto_mode_enabled(True)
            else:
                GLib.idle_add(self._maybe_warn_weak_device)

            # First-run setup wizard
            if self.config.first_run:
                self._restoring = False
                from nvbroadcast.ui.setup_wizard import SetupWizard
                wizard = SetupWizard(self._window, self)
                wizard.connect("setup-complete", self._on_setup_complete)
                wizard.present()
                GLib.idle_add(self._sync_global_hotkeys)
            elif self.config.auto_start:
                GLib.idle_add(self._finish_restore_and_auto_start)
            else:
                GLib.idle_add(self._finish_restore)

            if self.config.auto_start:
                GLib.timeout_add_seconds(30, self._preload_transcriber_when_idle)
            else:
                self._preload_transcriber()
            self._maybe_check_for_updates()

        self._window.set_visible(True)
        self._window.present()
        startup_trace.mark("window presented")
        print(f"[NV Broadcast] Window up in {startup_trace.elapsed():.1f}s",
              flush=True)
        self._maybe_show_python_runtime_notice()

    def _on_setup_complete(self, wizard, profile_name, gpu_index, compositing):
        """Called when first-run wizard finishes."""
        from nvbroadcast.core.config import apply_performance_profile, PERFORMANCE_PROFILES
        if profile_name == "auto":
            self.config.compute_gpu = gpu_index
            self.config.first_run = False
            self.config.current_profile = "Auto"
            self._video_effects._gpu_index = gpu_index
            self.set_auto_mode_enabled(True)
            save_config(self.config)
            self._window.rebuild_mode_selector(
                self.config.compositing, self.config.performance_profile
            )
            if hasattr(self._window, '_gpu_selector') and self._window._gpu_selector:
                self._window._gpu_selector.set_selected_index(gpu_index)
            self._window.set_status("Auto mode enabled")
            return

        # Apply profile
        apply_performance_profile(self.config, profile_name)
        self.config.compute_gpu = gpu_index
        self.config.compositing = compositing
        self.config.first_run = False
        self.config.current_profile = profile_name

        # Apply to effects engine
        self._video_effects._gpu_index = gpu_index
        self._video_effects._apply_edge_config(self.config.video.edge)
        self._video_effects.set_compositing(compositing)
        self._beautifier.set_compositing(compositing)
        profile = PERFORMANCE_PROFILES[profile_name]
        self.config.mode_key = NVBroadcastWindow._profile_and_comp_to_mode(
            profile_name, compositing
        )
        expected_quality = self._mode_quality_preset(self.config.mode_key)
        if expected_quality:
            self.config.video.quality_preset = expected_quality
            self._video_effects._quality = expected_quality
        mapped = NVBroadcastWindow._MODE_MAP.get(self.config.mode_key)
        if mapped is not None:
            _, _, use_tensorrt, use_fused_kernel, use_nvdec = mapped
        else:
            use_tensorrt = use_fused_kernel = use_nvdec = False
        self._video_effects.set_profile_infer_height(
            self._profile_infer_height(
                profile_name,
                use_tensorrt=use_tensorrt,
                use_fused_kernel=use_fused_kernel,
            )
        )
        self._video_effects._skip_interval = profile["skip_interval"]
        self.config.use_tensorrt = use_tensorrt
        self.config.use_fused_kernel = use_fused_kernel
        self.config.use_nvdec = use_nvdec
        self._use_nvdec = use_nvdec
        self._video_effects.set_engine_mode(use_tensorrt, use_fused_kernel)

        save_config(self.config)
        print(f"[NV Broadcast] Profile: {profile['label']} | GPU: {gpu_index} | Compositing: {compositing}")

        # Rebuild mode dropdown with updated backends (e.g. CuPy just installed)
        self._window.rebuild_mode_selector(compositing, profile_name)
        if hasattr(self._window, "_sync_quality_selector"):
            self._window._sync_quality_selector()
        if hasattr(self._window, '_gpu_selector') and self._window._gpu_selector:
            self._window._gpu_selector.set_selected_index(gpu_index)

        # Update edge tuning sliders
        self._window._edge_dilate._scale.set_value(self.config.video.edge.dilate_size)
        self._window._edge_blur._scale.set_value(self.config.video.edge.blur_size)
        self._window._edge_strength._scale.set_value(self.config.video.edge.sigmoid_strength)
        self._window._edge_midpoint._scale.set_value(self.config.video.edge.sigmoid_midpoint)

        self._window.set_status(f"Setup complete: {profile['label']} | {compositing} compositing")

        # Now auto-start
        if self.config.auto_start:
            GLib.idle_add(self._auto_start)

    def _finish_restore(self):
        """Release the startup restore guard after initial UI events settle."""
        self._restoring = False
        self._sync_global_hotkeys()
        return False

    def _finish_restore_and_auto_start(self):
        """Auto-start while restore guards still suppress startup signal noise."""
        try:
            self._auto_start()
        finally:
            self._restoring = False
        self._sync_global_hotkeys()
        return False

    def _register_global_hotkeys(self) -> None:
        """Export effect actions and select a supported desktop backend."""
        from nvbroadcast.core.global_hotkeys import (
            HOTKEY_ACTIONS,
            GlobalHotkeyManager,
        )

        for hotkey in HOTKEY_ACTIONS:
            action = Gio.SimpleAction.new(hotkey.action_id, None)
            action.set_enabled(False)
            action.connect(
                "activate",
                self._on_global_hotkey_action_activated,
                hotkey.action_id,
            )
            self.add_action(action)
        try:
            self._hotkey_manager = GlobalHotkeyManager(
                self._queue_global_hotkey_action,
                self._on_global_hotkey_state,
            )
        except (
            ImportError,
            GLib.Error,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            print(
                f"[NV Broadcast] Global hotkeys unavailable: {error}",
                flush=True,
            )
            self._hotkey_manager = None

    def _set_global_hotkey_actions_enabled(self, enabled: bool) -> None:
        from nvbroadcast.core.global_hotkeys import HOTKEY_ACTIONS

        for hotkey in HOTKEY_ACTIONS:
            action = self.lookup_action(hotkey.action_id)
            if action is not None:
                action.set_enabled(bool(enabled))

    def _queue_global_hotkey_action(self, action_id: str) -> None:
        """Route portal activations through the exported application action."""
        GLib.idle_add(self._activate_global_hotkey_action, action_id)

    def _activate_global_hotkey_action(self, action_id: str) -> bool:
        action = self.lookup_action(action_id)
        if action is not None and action.get_enabled():
            action.activate(None)
        return False

    def _on_global_hotkey_action_activated(
        self,
        _action,
        _parameter,
        action_id: str,
    ) -> None:
        GLib.idle_add(self._toggle_effect_from_hotkey, action_id)

    def _toggle_effect_from_hotkey(self, action_id: str) -> bool:
        if getattr(self, "_restoring", False) or self._window is None:
            return False
        controls = {
            "toggle-background": ("_bg_toggle", "Background"),
            "toggle-auto-frame": ("_autoframe_toggle", "Auto Frame"),
            "toggle-eye-contact": ("_eye_contact_toggle", "Eye Contact"),
            "toggle-mirror": ("_mirror_toggle", "Mirror"),
            "toggle-mic-noise": ("_noise_toggle", "Mic Noise Removal"),
        }
        target = controls.get(action_id)
        if target is None:
            return False
        attribute, title = target
        toggle = getattr(self._window, attribute, None)
        if toggle is None or not toggle.get_sensitive():
            self._window.set_status(f"{title} is not available")
            return False
        toggle.active = not toggle.active
        self._window.set_status(
            f"{title}: {'On' if toggle.active else 'Off'}"
        )
        return False

    def _on_global_hotkey_state(
        self,
        active: bool,
        status: str,
        display: dict[str, str],
    ) -> None:
        self._hotkey_active = bool(active)
        self._hotkey_status = status
        if display:
            self._hotkey_display = dict(display)
        elif not active:
            self._hotkey_display = {}
        if (
            status == "Global shortcut setup was canceled"
            and self.config.hotkeys.enabled
        ):
            self.config.hotkeys.enabled = False
            save_config(self.config)
        self._set_global_hotkey_actions_enabled(
            active and self.config.hotkeys.enabled
        )
        if self._window is not None:
            self._window.sync_hotkey_settings()

    def _sync_global_hotkeys(self) -> bool:
        if self._hotkey_manager is None:
            self._set_global_hotkey_actions_enabled(False)
            if self._window is not None:
                self._window.sync_hotkey_settings()
            return False
        from nvbroadcast.core.global_hotkeys import (
            HotkeyValidationError,
            bindings_from_config,
            sanitize_bindings,
        )

        bindings = bindings_from_config(self.config.hotkeys)
        try:
            self._set_global_hotkey_actions_enabled(False)
            self._hotkey_manager.apply(
                self.config.hotkeys.enabled,
                bindings,
            )
        except HotkeyValidationError as error:
            self.config.hotkeys.enabled = False
            for key, value in sanitize_bindings(bindings).items():
                setattr(self.config.hotkeys, key, value)
            self._hotkey_manager.apply(
                False,
                bindings_from_config(self.config.hotkeys),
            )
            self._hotkey_active = False
            self._hotkey_status = f"{error} Invalid saved shortcuts were cleared."
            self._hotkey_display = {}
            self._set_global_hotkey_actions_enabled(False)
            save_config(self.config)
        if self._window is not None:
            self._window.sync_hotkey_settings()
        return False

    def set_hotkeys_enabled(self, enabled: bool) -> bool:
        if self._hotkey_manager is None:
            return False
        if enabled and not self._hotkey_manager.available:
            self._hotkey_manager.apply(True, {})
            return False
        from nvbroadcast.core.global_hotkeys import bindings_from_config

        if not self._hotkey_manager.apply(
            enabled,
            bindings_from_config(self.config.hotkeys),
        ):
            return False
        self.config.hotkeys.enabled = bool(enabled)
        self._set_global_hotkey_actions_enabled(
            enabled and self._hotkey_active
        )
        save_config(self.config)
        if self._window is not None:
            self._window.sync_hotkey_settings()
        return True

    def set_hotkey_binding(
        self,
        config_key: str,
        accelerator: str,
    ) -> tuple[bool, str]:
        from nvbroadcast.core.global_hotkeys import (
            HotkeyValidationError,
            bindings_from_config,
            normalize_bindings,
        )

        if (
            self._hotkey_manager is None
            or not self._hotkey_manager.inline_editable
        ):
            return False, "Shortcuts are managed by the desktop"
        current = bindings_from_config(self.config.hotkeys)
        if config_key not in current:
            return False, "Unknown shortcut action"
        proposed = dict(current)
        proposed[config_key] = accelerator
        try:
            normalized = normalize_bindings(proposed)
        except HotkeyValidationError as error:
            return False, str(error)
        if not self._hotkey_manager.apply(
            self.config.hotkeys.enabled,
            normalized,
        ):
            self._hotkey_manager.apply(
                self.config.hotkeys.enabled,
                current,
            )
            return False, self._hotkey_status
        for key, value in normalized.items():
            setattr(self.config.hotkeys, key, value)
        save_config(self.config)
        if self._window is not None:
            self._window.sync_hotkey_settings()
        return True, ""

    def configure_global_hotkeys(self) -> bool:
        if self._hotkey_manager is None:
            return False
        return self._hotkey_manager.configure()

    def _on_close_request(self, window):
        """Minimize to tray instead of quitting.

        Stop the live pipeline first so closing the window always releases the
        camera instead of keeping a hidden capture session running.
        """
        if self.config.minimize_on_close and self._tray and self._tray.available:
            if self._streaming:
                self.stop_pipeline()
                if self._window:
                    self._window._streaming = False
                    self._window._stream_btn.set_label("Start Broadcast")
                    self._window._stream_btn.remove_css_class("destructive-action")
                    self._window._stream_btn.add_css_class("suggested-action")
            window.set_visible(False)
            status = "idle"
            if self._tray and self._tray.available:
                self._tray.update_status(self._streaming, status)
                print("[NV Broadcast] Pipeline stopped and app minimized to tray")
            else:
                print("[NV Broadcast] Pipeline stopped and app minimized to background")
            return True  # Prevent destruction
        if self.config.minimize_on_close:
            print("[NV Broadcast] No tray available; closing window will quit the app")
        return False  # Allow normal close

    def _probe_vcam_consumers(self) -> int | None:
        """Count external processes holding the vcam device.

        Primary source is the v4l2loopback client-usage monitor: inside
        the bubblewrap user namespace, fuser cannot stat other processes'
        /proc/PID/fd links and silently reports zero consumers, which
        used to freeze live calls. fuser remains only as a fallback when
        the v4l2 event is unavailable, with a liveness guard against
        exactly that blindness.

        Returns None when the answer is not trustworthy — callers MUST
        treat None as "camera in use" so a detection failure can never
        freeze someone's camera.
        """
        if IS_MACOS:
            return None
        if self._vcam_monitor is not None and self._vcam_monitor.running:
            return self._vcam_monitor.consumers()
        import os
        import subprocess
        try:
            result = subprocess.run(
                ["fuser", self._active_vcam_device()],
                capture_output=True, text=True, timeout=2,
            )
        except Exception:
            return None
        # fuser: 0 = at least one accessor, 1 = none (or failure — but then
        # stdout is empty either way, which is the safe reading).
        if result.returncode not in (0, 1):
            return None
        own_pid = str(os.getpid())
        count = 0
        own_seen = False
        for token in result.stdout.split():
            pid = "".join(ch for ch in token if ch.isdigit())
            if not pid:
                continue
            if pid == own_pid:
                own_seen = True
            else:
                count += 1
        pipeline_holds_device = (
            self._video_pipeline is not None
            and self._vcam_available
            and not getattr(self._video_pipeline, "_vcam_failed", False)
        )
        if pipeline_holds_device and not own_seen:
            # fuser cannot even see our own fd on the device: it is blind
            # (user namespace), so its count of others is worthless.
            return None
        return count

    def _check_vcam_consumers(self):
        """Poll vcam consumers for status display and camera power save.

        The pipeline is never stopped for power save — stopping it while a
        consumer holds the device breaks v4l2sink with exclusive_caps=1.
        Idle only pauses the capture leg (camera off, vcam device stays
        open), and it errs hard toward "in use": unknown counts as in use,
        and three consecutive idle verdicts are required before pausing.
        """
        if not self._vcam_available or not self._streaming:
            self._idle_strikes = 0
            return True  # Keep polling

        consumers = self._probe_vcam_consumers()

        if consumers is not None and consumers != self._vcam_consumers:
            self._vcam_consumers = consumers
            if self._tray and self._tray.available:
                status = (f"streaming ({consumers} consumer"
                          f"{'s' if consumers != 1 else ''})"
                          if self._streaming else "idle")
                self._tray.update_status(self._streaming, status)

        window_hidden = not (self._window and self._window.get_visible())
        pipeline = self._video_pipeline
        can_idle = (
            getattr(self.config, "auto_idle", True)
            and pipeline is not None
            and not pipeline.is_recording
            and consumers == 0          # None (unknown) never idles
            and window_hidden
        )

        if self._idle_active:
            if not can_idle:
                self._exit_idle("activity detected")
            return True

        if can_idle:
            self._idle_strikes += 1
            if self._idle_strikes >= 3:
                self._enter_idle()
        else:
            self._idle_strikes = 0
        return True  # Keep polling

    def _enter_idle(self):
        pipeline = self._video_pipeline
        if pipeline is None:
            return
        if not pipeline.set_capture_idle(True):
            self._idle_strikes = 0
            return
        self._idle_active = True
        self._idle_strikes = 0
        if self._tray and self._tray.available:
            self._tray.update_status(self._streaming, "power save (camera unused)")
        # Fast wake poll: a consumer must never wait on the 10s status
        # poll. The inotify monitor also wakes us event-driven; this tick
        # is belt-and-braces (and covers window-shown).
        GLib.timeout_add(1000, self._idle_wake_tick)

    def _exit_idle(self, reason: str):
        self._idle_active = False
        self._idle_strikes = 0
        pipeline = self._video_pipeline
        if pipeline is not None:
            pipeline.set_capture_idle(False)
        print(f"[NV Broadcast] Power save resumed: {reason}", flush=True)
        if self._tray and self._tray.available:
            self._tray.update_status(self._streaming, "streaming")

    def _on_vcam_consumer_wake(self):
        """Called from the monitor thread on a sustained device open."""
        GLib.idle_add(self._wake_from_vcam_monitor)

    def _wake_from_vcam_monitor(self):
        if self._idle_active:
            self._exit_idle("consumer detected (v4l2 event)")
        return False  # One-shot idle source

    def _idle_wake_tick(self):
        """1s wake poll while idle. Any doubt resumes the camera."""
        if not self._idle_active:
            return False  # Stop this timer
        consumers = self._probe_vcam_consumers()
        window_visible = bool(self._window and self._window.get_visible())
        if consumers != 0 or window_visible:
            self._exit_idle("consumer detected" if not window_visible
                            else "window shown")
            return False
        return True

    def set_auto_idle(self, enabled: bool):
        """Toggle camera + mic power save from the UI."""
        self.config.auto_idle = bool(enabled)
        save_config(self.config)
        if not enabled and self._idle_active:
            self._exit_idle("power save disabled")
        self._idle_strikes = 0
        # The audio helper reads auto_idle from its spawn state — restart
        # it so the mic-side monitor follows the new setting.
        if self._audio_pipeline is not None:
            self._audio_pipeline.auto_idle = bool(enabled)
            self._restart_audio_pipeline_for_live_settings()

    def _preload_effects(self):
        """Pre-initialize AI models in background to eliminate first-use delay."""
        def _init():
            try:
                if self.config.video.background_removal:
                    self._video_effects.initialize()
            except Exception as e:
                print(f"[NV Broadcast] Background model preload failed: {e}")
        threading.Thread(target=_init, daemon=True).start()

    def _preload_transcriber(self):
        """Warm Whisper in the background so Start Meeting does not stall the UI."""
        if self._transcriber_preload_started or self._transcriber.initialized:
            return
        if not self._dependency_installer.is_available("whisper"):
            return

        self._transcriber_preload_started = True

        def _init():
            try:
                self._transcriber.initialize()
            except Exception as e:
                print(f"[NV Broadcast] Meeting transcription preload failed: {e}")
                self._transcriber_preload_started = False

        threading.Thread(target=_init, daemon=True).start()

    def _preload_transcriber_when_idle(self):
        """Avoid transcriber warmup while the live camera pipeline is already busy."""
        if self._meeting_active or self._meeting_finalizing:
            return False
        if self._streaming:
            return True
        self._preload_transcriber()
        return False

    def _maybe_show_python_runtime_notice(self):
        if self._window is None:
            return
        notice = python_runtime_advisory()
        if notice is None:
            return
        notice_key, title, reason = notice
        if self.config.last_python_runtime_notice == notice_key:
            return
        self.config.last_python_runtime_notice = notice_key
        save_config(self.config)
        self._window.show_advisory(notice_key, title, reason)

    def _maybe_check_for_updates(self):
        if self._window is None or not should_check_for_updates(self.config):
            return

        def _worker():
            release = fetch_latest_release(timeout=5)
            self.config.last_update_check = int(time.time())
            if release and is_newer_version(release.version, __version__):
                self._update_release = release
                target = resolve_update_target(release)
                if self.config.last_notified_version != release.version:
                    self.config.last_notified_version = release.version
                    GLib.idle_add(
                        self._show_update_available,
                        release.version,
                        target.button_label,
                        target.tooltip,
                        target.url,
                        True,
                    )
                else:
                    GLib.idle_add(
                        self._show_update_available,
                        release.version,
                        target.button_label,
                        target.tooltip,
                        target.url,
                        False,
                    )
            save_config(self.config)

        threading.Thread(target=_worker, daemon=True).start()

    def _show_update_available(self, version: str, label: str, tooltip: str, url: str,
                               announce: bool):
        if self._window is None:
            return False
        self._window.set_update_available(version, label, tooltip, url)
        if announce:
            self._window.set_status(f"Recommended stable update: v{version}")
        return False

    def _auto_start(self):
        """Auto-start broadcast with saved settings."""
        startup_trace.mark("auto-start begin")
        print(f"[NV Broadcast] Auto-start: streaming={self._streaming} vcam={self._vcam_available}", flush=True)
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
        from nvbroadcast.core.config import PERFORMANCE_PROFILES

        c = self.config
        normalized_quality = False
        expected_quality = self._mode_quality_preset(c.mode_key)
        if expected_quality and c.video.quality_preset != expected_quality:
            c.video.quality_preset = expected_quality
            normalized_quality = True

        # Restore model and quality preset
        self._video_effects._model_type = c.video.model
        self._video_effects._quality = c.video.quality_preset
        self._video_effects._gpu_index = c.compute_gpu
        self._perf_monitor.set_gpu_index(c.compute_gpu)
        self._video_effects.set_compositing(c.compositing)
        self._beautifier.set_compositing(c.compositing)
        mapped = NVBroadcastWindow._MODE_MAP.get(c.mode_key)
        if mapped is not None:
            _, _, use_tensorrt, use_fused_kernel, use_nvdec = mapped
        else:
            use_tensorrt = c.use_tensorrt
            use_fused_kernel = c.use_fused_kernel
            use_nvdec = c.use_nvdec
        self._video_effects.set_profile_infer_height(
            self._profile_infer_height(
                c.performance_profile,
                use_tensorrt=use_tensorrt,
                use_fused_kernel=use_fused_kernel,
            )
        )
        self._video_effects.set_engine_mode(use_tensorrt, use_fused_kernel)
        self._use_nvdec = use_nvdec
        profile = PERFORMANCE_PROFILES.get(c.performance_profile, {})
        self._video_effects._skip_interval = profile.get("skip_interval", 1)
        self._video_effects._apply_edge_config(c.video.edge)
        self._video_effects._edge_refine_enabled = c.premium_edge_refine and c.mode_key in ("killer", "zeus")

        # Restore background settings
        self._video_effects.enabled = c.video.background_removal
        if c.video.background_image:
            self._video_effects.set_background_image(c.video.background_image)
        self._video_effects.mode = c.video.background_mode
        self._video_effects.intensity = c.video.blur_intensity
        self._video_effects.blur_dim = getattr(c.video, "blur_dim", 0.0)
        self._video_effects.blur_desaturate = getattr(c.video, "blur_desaturate", 0.0)

        # Tell window to restore UI controls FIRST (may fire toggle callbacks)
        self._window.restore_settings(c)

        # Then force-set ALL effect states from config (overrides any
        # callbacks that toggled effects off or changed modes during UI restore)
        self._video_effects.enabled = c.video.background_removal
        self._video_effects.mode = c.video.background_mode
        self._video_effects.intensity = c.video.blur_intensity
        self._video_effects.blur_dim = getattr(c.video, "blur_dim", 0.0)
        self._video_effects.blur_desaturate = getattr(c.video, "blur_desaturate", 0.0)
        if c.video.background_image:
            self._video_effects.set_background_image(c.video.background_image)
        self._eye_contact.enabled = c.video.eye_contact
        self._eye_contact.intensity = c.video.eye_contact_intensity
        self._eye_contact.mode = c.video.eye_contact_mode
        self._relighter.enabled = c.video.relighting
        self._relighter.intensity = c.video.relighting_intensity
        self._beautifier.enabled = c.video.beauty.enabled
        self._beautifier.skin_smooth = c.video.beauty.skin_smooth
        self._beautifier.denoise = c.video.beauty.denoise
        self._beautifier.enhance = c.video.beauty.enhance
        self._beautifier.sharpen = c.video.beauty.sharpen
        self._beautifier.edge_darken = c.video.beauty.edge_darken
        self._mirror = c.video.mirror
        self._autoframe.enabled = c.video.auto_frame
        self._autoframe.zoom_level = c.video.auto_frame_zoom
        self._autoframe.mode = c.video.auto_frame_mode
        self._refresh_inference_policy()

        if self._audio_pipeline_should_publish() or c.audio.noise_removal or c.audio.voice_fx_enabled:
            audio_pipeline = self._ensure_audio_pipeline()
            audio_pipeline.auto_idle = getattr(c, "auto_idle", True)
            audio_pipeline.effects.engine = c.audio.noise_engine
            audio_pipeline.effects.enabled = c.audio.noise_removal
            audio_pipeline.effects.intensity = c.audio.noise_intensity
            audio_pipeline.voice_fx.enabled = c.audio.voice_fx_enabled
            audio_pipeline.voice_fx.use_gpu = c.audio.voice_fx_use_gpu
            self._apply_voice_fx_settings_from_config(audio_pipeline)
            self._refresh_audio_pipeline()

        if self._video_pipeline:
            effects_fps = max(5, int(profile.get("effects_ratio", 1.0) * c.video.fps))
            self._video_pipeline.set_effects_fps(effects_fps)
            self._video_pipeline.set_alpha_worker_enabled(not self._inline_inference)

        if self._vcam_available:
            self._window.set_status(
                f"Ready - Virtual camera at {self._active_vcam_device()}"
            )
        else:
            self._window.set_status(
                "Virtual camera not available. Run: "
                + v4l2loopback_modprobe_command(self.config.video.vcam_device)
            )

        if normalized_quality:
            save_config(c)

    def restore_current_config(self):
        """Replay the current config into UI and runtime under restore guards."""
        previous = getattr(self, "_restoring", False)
        self._restoring = True
        try:
            self._restore_settings()
        finally:
            self._restoring = previous

    @staticmethod
    def _capture_mode_rank(mode: tuple[int, int, int]) -> tuple[int, int]:
        return (mode[0] * mode[1], mode[2])

    def _current_capture_mode(self) -> tuple[int, int, int]:
        return (
            self.config.video.width,
            self.config.video.height,
            self.config.video.fps,
        )

    def _available_capture_modes(self) -> list[tuple[int, int, int]]:
        from nvbroadcast.video.virtual_camera import list_camera_modes

        capture_modes: list[tuple[int, int, int]] = []
        for mode in list_camera_modes(self.config.video.camera_device):
            width = mode["width"]
            height = mode["height"]
            for fps in sorted(set(mode["fps"]), reverse=True):
                capture_modes.append((width, height, fps))
        capture_modes.sort(key=self._capture_mode_rank, reverse=True)
        return capture_modes

    def _next_lower_capture_mode(self) -> tuple[int, int, int] | None:
        available = self._available_capture_modes()
        if not available:
            return None

        current = self._current_capture_mode()
        current_rank = self._capture_mode_rank(current)
        if current in available:
            idx = available.index(current)
            if idx < len(available) - 1:
                return available[idx + 1]
            return None

        for mode in available:
            if self._capture_mode_rank(mode) < current_rank:
                return mode
        return None

    def _apply_capture_mode_choice(
        self,
        width: int,
        height: int,
        fps: int,
        *,
        status_prefix: str,
        advisory_key: str | None = None,
        advisory_title: str | None = None,
        advisory_reason: str | None = None,
    ) -> bool:
        valid_fps = self._get_valid_fps(width, height, fps)
        if self._current_capture_mode() == (width, height, valid_fps):
            return False

        self.config.video.width = width
        self.config.video.height = height
        self.config.video.fps = valid_fps
        save_config(self.config)

        if self._window is not None and hasattr(self._window, "sync_video_input_controls"):
            self._window.sync_video_input_controls(self.config)

        mode_text = f"{width}x{height} @ {valid_fps} fps"
        if self._window is not None:
            if self._streaming:
                self._window.set_status(f"{status_prefix} {mode_text}. Restart the app to apply.")
                if advisory_key and advisory_title and advisory_reason:
                    self._window.show_advisory(advisory_key, advisory_title, advisory_reason)
            else:
                self._window.set_status(f"{status_prefix} {mode_text}")
        return True

    # --- Video Pipeline ---

    def _clear_finished_teardown(self):
        if self._pipeline_teardown and self._pipeline_teardown._teardown_done:
            self._pipeline_teardown = None

    def _queue_pipeline_restart(self):
        if self._restart_source_id:
            return
        self._restart_source_id = GLib.timeout_add(100, self._restart_after_stop)

    def start_pipeline(self, camera_device: str, output_format: str = "YUY2"):
        self._clear_finished_teardown()
        self._pending_start = (camera_device, output_format)

        if self._video_pipeline or self._pipeline_teardown:
            self.stop_pipeline(clear_pending_start=False)
            if self._window:
                self._window._streaming = False
                self._window._stream_btn.set_label("Start Broadcast")
                self._window._stream_btn.remove_css_class("destructive-action")
                self._window._stream_btn.add_css_class("suggested-action")
                self._window.set_status("Restarting...")
            self._queue_pipeline_restart()
            return

        self._restart_after_stop()

    def _restart_after_stop(self):
        """Restart after the previous pipeline has fully released devices."""
        self._clear_finished_teardown()
        if self._video_pipeline or self._pipeline_teardown:
            return True

        self._restart_source_id = 0
        if self._pending_start is None:
            return False

        cam, fmt = self._pending_start
        self._pending_start = None
        self._do_start_pipeline(cam, fmt)
        if self._streaming and self._window:
            self._window._streaming = True
            self._window._stream_btn.set_label("Stop Broadcast")
            self._window._stream_btn.remove_css_class("suggested-action")
            self._window._stream_btn.add_css_class("destructive-action")
        return False

    def _do_start_pipeline(self, camera_device: str, output_format: str = "YUY2"):
        self._clear_finished_teardown()
        if self._video_pipeline or self._pipeline_teardown:
            self._pending_start = (camera_device, output_format)
            self._queue_pipeline_restart()
            return False

        startup_trace.mark("start_pipeline begin")
        from nvbroadcast.core.config import PERFORMANCE_PROFILES
        from nvbroadcast.video.virtual_camera import resolve_camera_device, select_camera_mode

        resolved_camera = resolve_camera_device(
            camera_device or self.config.video.camera_device
        )
        if resolved_camera != camera_device:
            print(
                f"[NV Broadcast] Camera changed: {camera_device} -> {resolved_camera}",
                flush=True,
            )
            camera_device = resolved_camera

        selected_mode = select_camera_mode(
            camera_device,
            self.config.video.width,
            self.config.video.height,
            self.config.video.fps,
        )
        if (
            selected_mode["width"] != self.config.video.width
            or selected_mode["height"] != self.config.video.height
            or selected_mode["fps"] != self.config.video.fps
        ):
            print(
                "[NV Broadcast] Camera mode changed: "
                f"{self.config.video.width}x{self.config.video.height}@"
                f"{self.config.video.fps} -> "
                f"{selected_mode['width']}x{selected_mode['height']}@"
                f"{selected_mode['fps']}",
                flush=True,
            )
            self.config.video.width = selected_mode["width"]
            self.config.video.height = selected_mode["height"]
            self.config.video.fps = selected_mode["fps"]
            save_config(self.config)

        profile = PERFORMANCE_PROFILES.get(self.config.performance_profile, {})
        # Validate fps before building pipeline
        camera_fps = self._get_valid_fps(
            self.config.video.width, self.config.video.height, self.config.video.fps,
            camera_device=camera_device,
        )
        if camera_fps != self.config.video.fps:
            self.config.video.fps = camera_fps
            save_config(self.config)
        effects_fps = max(5, int(profile.get("effects_ratio", 1.0) * camera_fps))

        startup_trace.mark("camera modes probed")
        self._video_pipeline = VideoPipeline()
        self._video_pipeline.configure(
            source_device=camera_device,
            vcam_device=self._active_vcam_device(),
            width=self.config.video.width,
            height=self.config.video.height,
            fps=self.config.video.fps,
            output_format=output_format,
            effects_fps=effects_fps,
            prefer_hw_decode=self._use_nvdec,
        )

        self._video_pipeline.set_effect_callback(self._process_frame)
        self._video_pipeline.set_alpha_callback(self._update_alpha)
        self._video_pipeline.set_alpha_worker_enabled(not self._inline_inference)
        self._sync_gpu_frame_path(output_format)
        startup_trace.mark("gpu frame path ready")
        self._video_pipeline.set_preview_callback(
            lambda texture: self._window.update_preview(texture)
        )

        # Reset all resolution-dependent state BEFORE new pipeline processes frames
        self._video_effects.reset_cached_mattes()
        if self._video_effects._backend:
            self._video_effects._backend.reset_state()
        self._beautifier._face_mask = None
        self._beautifier._tone_mask = None
        self._beautifier._vignette_cache = None
        self._beautifier._face_bbox = None
        self._beautifier._face_center = None
        self._beautifier._prev_frame = None

        # Start in effects mode if effects were previously enabled
        if self._any_video_effects_active():
            self._video_pipeline._effects_active = True

        try:
            self._video_pipeline.build(vcam_enabled=self._vcam_available)
            startup_trace.mark("pipeline built")
            self._video_pipeline.start()
            startup_trace.mark("pipeline started")
            self._streaming = True

            w, h = self.config.video.width, self.config.video.height
            status = f"Streaming: {camera_device} {w}x{h}@{self.config.video.fps}fps"
            if (
                self._vcam_available
                and self._video_pipeline.virtual_camera_active
            ):
                status += f" -> {self._active_vcam_device()}"
            elif self._vcam_available:
                status += " - virtual camera unavailable"
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

    def stop_pipeline(self, clear_pending_start: bool = True):
        if clear_pending_start:
            self._pending_start = None
        self._idle_active = False
        self._idle_strikes = 0
        if self._restart_source_id:
            GLib.source_remove(self._restart_source_id)
            self._restart_source_id = 0
        if self._video_pipeline:
            pipeline = self._video_pipeline
            pipeline.stop()
            if pipeline._teardown_done:
                self._pipeline_teardown = None
            else:
                self._pipeline_teardown = pipeline
            self._video_pipeline = None
        self._streaming = False

    def _update_alpha(self, frame_data: bytes, width: int, height: int) -> None:
        """Background thread — only updates the alpha mask."""
        self._video_effects.update_alpha(frame_data, width, height)

    def _gpu_frame_path_allowed(self, output_format: str | None = None) -> bool:
        """Return whether the active resource policy permits CUDA transport."""
        target_format = output_format or self.config.video.output_format
        return (
            not IS_MACOS
            and self.config.compute_focus != "cpu"
            and self.config.compositing in ("cupy", "gstreamer_gl")
            and target_format == "YUY2"
            and os.getenv("NVBROADCAST_NO_GPU_FRAME_PATH") != "1"
        )

    def _sync_gpu_frame_path(self, output_format: str | None = None) -> None:
        """Create, detach, or rebind the frame path for the active policy."""
        allowed = self._gpu_frame_path_allowed(output_format)
        if allowed:
            if self._gpu_frame_path is None and not self._gpu_frame_path_failed:
                from nvbroadcast.video.gpu_frame_path import GpuFramePath

                self._gpu_frame_path = GpuFramePath.create(
                    self._video_effects, gpu_index=self.config.compute_gpu)
                if self._gpu_frame_path is None:
                    self._gpu_frame_path_failed = True
            processor = self._gpu_frame_path
        else:
            processor = None
            self._gpu_frame_path = None
            # A policy change is a valid reason to retry CUDA if the user
            # explicitly selects a GPU mode later.
            self._gpu_frame_path_failed = False

        if self._video_pipeline is not None:
            self._video_pipeline.set_frame_processor(
                processor,
                self._gpu_frame_plan if processor is not None else None,
                wait_for_inflight=processor is None,
            )

    def _gpu_frame_plan(self):
        """Per-frame routing for the device-resident path.

        Returns (gpu_pure, mirror, inline_inference). gpu_pure means every
        active stage runs on the GPU; any CPU face stage routes the frame
        through the legacy bytes callback instead (still convert-free).
        """
        face_fx = (
            self._beautifier.enabled
            or self._eye_contact.enabled
            or self._relighter.enabled
        )
        gpu_pure = (
            not face_fx
            and not self._autoframe.enabled
            and self._video_effects.enabled
            and self._video_effects.gpu_output_eligible()
        )
        return gpu_pure, self._mirror, self._inline_inference

    def _process_frame(self, frame_data: bytes, width: int, height: int) -> bytes:
        """Inline callback — processes EVERY frame with ALL effects.
        Runs composite + face effects + mirror on the current frame."""
        import cv2
        import numpy as np

        self._perf_monitor.tick()
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)

        face_effects_active = (
            self._beautifier.enabled
            or self._eye_contact.enabled
            or self._relighter.enabled
        )
        # Only pay the ~8MB writeable copy when a CPU stage might mutate
        # the raw frame; the GPU blur path reads it exactly once, and the
        # effects processor makes its own copy for remove/replace modes.
        if not frame.flags.writeable and (
            face_effects_active or self._autoframe.enabled
        ):
            frame = frame.copy()
        result_frame = frame
        landmarks = None
        fused_beautify_overlay = False
        if face_effects_active:
            landmarker = get_shared_landmarker()
            raw_frame = result_frame
            if landmarker.ready:
                landmarks = landmarker.request_async(
                    raw_frame,
                    reuse_frames=self._landmark_reuse_frames(),
                )
            if (
                self._beautifier.enabled
                and self._video_effects.enabled
                and self._video_effects._use_fused_kernel
            ):
                self._beautifier.prime_face_cache(
                    raw_frame,
                    width,
                    height,
                    landmarks=landmarks,
                    allow_inline_landmarks=False,
                )
                overlay = self._beautifier.fused_overlay_inputs(width, height)
                if overlay is not None:
                    self._video_effects.set_fused_face_overlay(*overlay)
                    fused_beautify_overlay = True
                else:
                    self._video_effects.set_fused_face_overlay(None, None)
            elif self._video_effects.enabled:
                self._video_effects.set_fused_face_overlay(None, None)
        elif self._video_effects.enabled:
            self._video_effects.set_fused_face_overlay(None, None)

        # Inline-inference profiles own the alpha path entirely. The pipeline
        # disables the background alpha worker in that mode to avoid cache races.
        # Mirror on-GPU only when no CPU stage runs after compositing —
        # later stages would otherwise operate on an already-flipped frame.
        gpu_mirror = (
            self._mirror
            and not face_effects_active
            and not self._autoframe.enabled
            and self._video_effects.enabled
        )
        if self._video_effects.enabled:
            if self._inline_inference:
                result_frame = self._video_effects.process_frame_array(
                    result_frame, width, height, mirror=gpu_mirror)
            else:
                result_frame = self._video_effects.composite_only_array(
                    result_frame, width, height, mirror=gpu_mirror)

        if face_effects_active:
            if self._beautifier.enabled:
                result_frame = self._beautifier.process_frame_array(
                    result_frame,
                    width,
                    height,
                    landmarks=landmarks,
                    allow_inline_landmarks=False,
                    skip_enhance=fused_beautify_overlay,
                    skip_edge_darken=fused_beautify_overlay,
                    cache_prepared=fused_beautify_overlay,
                )

            alpha_u8 = None
            if self._video_effects.enabled:
                alpha_u8 = self._video_effects.latest_final_matte_u8(width, height)

            if self._eye_contact.enabled and landmarks is not None:
                result_frame = self._eye_contact.process_frame(result_frame, landmarks=landmarks)
            if self._relighter.enabled and landmarks is not None:
                result_frame = self._relighter.process_frame(result_frame, alpha_u8, landmarks=landmarks)

        if self._autoframe.enabled:
            result_frame = self._autoframe.process_frame_array(result_frame, width, height)

        # Mirror flip (skipped when the fused GPU path already flipped)
        if self._mirror and not (
            gpu_mirror and self._video_effects.last_output_mirrored
        ):
            result_frame = cv2.flip(result_frame, 1)
        return result_frame.tobytes()

    def _any_video_effects_active(self) -> bool:
        return (self._video_effects.enabled or self._autoframe.enabled or
                self._beautifier.enabled or self._eye_contact.enabled or
                self._relighter.enabled)

    def _face_effect_load_score(self) -> int:
        """Estimate how expensive the live face stack is on the display thread."""
        score = 0
        if self._beautifier.enabled:
            score += 3
        if self._eye_contact.enabled:
            score += 1
        if self._relighter.enabled:
            score += 1
        if self._autoframe.enabled:
            score += 1
        return score

    def _landmark_reuse_frames(self) -> int:
        """Choose how aggressively to reuse shared face landmarks."""
        score = self._face_effect_load_score()
        if self._eye_contact.enabled and score == 1:
            return 1
        if score >= 5 and not self._autoframe.enabled:
            return 4
        return 2

    def _compute_inline_inference(self) -> bool:
        """Choose whether alpha inference should run inline on the live frame.

        Async alpha helped throughput on some heavy stacks, but it also makes
        replaced-background edges visibly trail motion because the current frame
        is composited against an older alpha. For `max_quality` and `balanced`,
        prioritize edge freshness and match the pre-1.1.6 live behavior.

        CUDA Fast keeps the lightweight async path for blur/remove, but replace
        mode is visually unforgiving around hair, glasses, hands, and fingers.
        On the fused CuPy path, inline replace inference is the better default:
        it spends GPU time where available instead of letting CPU-side stale
        mattes create a delayed edge.
        """
        if self.config.performance_profile in ("max_quality", "balanced"):
            return True
        return (
            self.config.performance_profile == "performance"
            and self.config.compositing == "cupy"
            and bool(self.config.use_fused_kernel)
            and not bool(self.config.use_tensorrt)
            and bool(getattr(self._video_effects, "enabled", False))
            and getattr(self._video_effects, "mode", "") == "replace"
        )

    def _refresh_inference_policy(self) -> None:
        inline = self._compute_inline_inference()
        self._inline_inference = inline
        if self._video_pipeline:
            self._video_pipeline.set_alpha_worker_enabled(not inline)

    def _update_pipeline_mode(self):
        if self._video_pipeline:
            self._video_pipeline.set_effects_active(self._any_video_effects_active())
        self._refresh_inference_policy()

    # --- Effect Controls (save on every change) ---

    def set_bg_removal(self, enabled: bool):
        if getattr(self, '_restoring', False):
            return
        self._video_effects.enabled = enabled
        self.config.video.background_removal = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_bg_mode(self, mode: str):
        if getattr(self, '_restoring', False):
            return
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
        self.config.video.blur_intensity = self._video_effects.intensity
        save_config(self.config)

    def set_blur_dim(self, value: float):
        self._video_effects.blur_dim = value
        self.config.video.blur_dim = self._video_effects.blur_dim
        save_config(self.config)

    def set_blur_desaturate(self, value: float):
        self._video_effects.blur_desaturate = value
        self.config.video.blur_desaturate = self._video_effects.blur_desaturate
        save_config(self.config)

    def set_performance_profile(self, profile_name: str, compositing: str | None = None,
                                use_tensorrt: bool = False, use_fused_kernel: bool = False,
                                use_nvdec: bool = False, mode_key: str | None = None,
                                quality_preset: str | None = None):
        """Switch performance profile and apply its live processing policy."""
        from nvbroadcast.core.config import apply_performance_profile, PERFORMANCE_PROFILES
        if profile_name not in PERFORMANCE_PROFILES:
            return

        # Apply compositing change
        if compositing and compositing != self.config.compositing:
            self.config.compositing = compositing
            self._video_effects.set_compositing(compositing)
            self._beautifier.set_compositing(compositing)

        profile_infer_height = self._profile_infer_height(
            profile_name,
            use_tensorrt=use_tensorrt,
            use_fused_kernel=use_fused_kernel,
        )

        # Apply model quality, inference dimensions, TensorRT, and the fused
        # path as one transition so no frame can observe mixed mode settings.
        self._video_effects.set_engine_mode(
            use_tensorrt,
            use_fused_kernel,
            quality=quality_preset,
            profile_infer_height=profile_infer_height,
        )
        self.config.use_tensorrt = use_tensorrt
        self.config.use_fused_kernel = use_fused_kernel

        # NVDEC: enable GPU JPEG decode in pipeline (Killer mode)
        self._use_nvdec = use_nvdec
        self.config.use_nvdec = use_nvdec

        self.config.mode_key = mode_key or NVBroadcastWindow._profile_and_comp_to_mode(
            profile_name, self.config.compositing
        )

        apply_performance_profile(self.config, profile_name)
        profile = PERFORMANCE_PROFILES[profile_name]

        # Model settings update immediately. A CPU/GPU transport change asks
        # VideoPipeline for its normal teardown-safe internal rebuild below.
        self._video_effects._skip_interval = profile["skip_interval"]
        self._video_effects._apply_edge_config(self.config.video.edge)

        # Compute effects_fps from ratio * camera fps
        effects_fps = max(5, int(profile["effects_ratio"] * self.config.video.fps))
        self._refresh_inference_policy()
        if self._video_pipeline:
            self._video_pipeline.set_effects_fps(effects_fps)
            self._video_pipeline.set_alpha_worker_enabled(not self._inline_inference)
        self._sync_gpu_frame_path()

        save_config(self.config)

        b = self._video_effects._backend
        infer_h = getattr(b, "_MAX_INFER_HEIGHT", "?")
        print(f"[NV Broadcast] Mode: {profile_name} | infer={infer_h} skip={profile['skip_interval']} "
              f"fused={use_fused_kernel} nvdec={use_nvdec} comp={self.config.compositing} "
              f"efps={effects_fps}")

        if self._window:
            self._window.set_status(f"Mode: {profile['label']} | {infer_h}p")

    def apply_mode_key(self, mode_key: str, status: str | None = None) -> bool:
        """Apply one of the stable named modes and sync related UI state."""
        mapped = NVBroadcastWindow._MODE_MAP.get(mode_key)
        if mapped is None:
            return False

        profile, comp, trt, fused, nvdec = mapped
        expected_quality = self._mode_quality_preset(mode_key)
        self.set_performance_profile(
            profile,
            compositing=comp,
            use_tensorrt=trt,
            use_fused_kernel=fused,
            use_nvdec=nvdec,
            mode_key=mode_key,
            quality_preset=expected_quality,
        )
        if expected_quality:
            self._video_effects.quality = expected_quality
            self.config.video.quality_preset = expected_quality
            save_config(self.config)

        if self._window is not None:
            is_premium = mode_key in ("killer", "zeus")
            toggle = getattr(self._window, "_edge_refine_toggle", None)
            if toggle is not None:
                toggle.set_visible(is_premium)
                toggle.set_sensitive(is_premium)
                desired = is_premium and self.config.premium_edge_refine
                if toggle.active != desired:
                    toggle.active = desired
            if hasattr(self._window, "_sync_mode_selector"):
                self._window._sync_mode_selector()
            if hasattr(self._window, "_sync_compute_focus_selector"):
                self._window._sync_compute_focus_selector()
            if hasattr(self._window, "_sync_quality_selector"):
                self._window._sync_quality_selector()
            if status:
                self._window.set_status(status)
            else:
                msg = NVBroadcastWindow._mode_status_message(mode_key)
                if msg:
                    self._window.set_status(msg)
        return True

    def _available_auto_modes(self) -> list[str]:
        """Return the stable modes that are usable on this machine right now."""
        modes: list[str] = []
        for mode_key in _AUTO_MODE_ORDER:
            if self._dependency_installer.unsupported_reason_for_mode(mode_key):
                continue
            if self._dependency_installer.missing_for_mode(mode_key):
                continue
            modes.append(mode_key)

        focus = self._compute_focus()
        if focus == "cpu":
            cpu_modes = [mode for mode in _CPU_AUTO_MODES if mode in modes]
            return cpu_modes or modes or ["cpu_low"]
        if focus == "gpu":
            gpu_modes = [mode for mode in _GPU_AUTO_MODES if mode in modes]
            cpu_fallback = [mode for mode in _CPU_AUTO_MODES if mode in modes]
            return gpu_modes + cpu_fallback if gpu_modes else cpu_fallback or modes or ["cpu_low"]
        return modes or ["cpu_low"]

    def _preferred_auto_mode(self) -> str:
        """Pick the best stable starting mode for the current hardware."""
        from nvbroadcast.core.config import detect_system_capabilities

        caps = detect_system_capabilities()
        available = self._available_auto_modes()
        focus = self._compute_focus()

        if focus == "cpu":
            if caps["cpu_cores"] >= 8:
                preferred = ["cpu_quality", "cpu_light", "cpu_low"]
            elif caps["cpu_cores"] >= 4:
                preferred = ["cpu_light", "cpu_quality", "cpu_low"]
            else:
                preferred = ["cpu_low", "cpu_light", "cpu_quality"]
        elif caps["has_nvidia"]:
            if caps["gpu_vram_mb"] >= 8192:
                preferred = ["doczeus", "cuda_balanced", "cuda_perf"]
            else:
                preferred = ["cuda_balanced", "cuda_perf", "doczeus"]
            preferred.extend(["cpu_quality", "cpu_light", "cpu_low"])
        else:
            if caps["cpu_cores"] >= 8:
                preferred = ["cpu_quality", "cpu_light", "cpu_low"]
            elif caps["cpu_cores"] >= 4:
                preferred = ["cpu_light", "cpu_quality", "cpu_low"]
            else:
                preferred = ["cpu_low", "cpu_light", "cpu_quality"]

        for mode_key in preferred:
            if mode_key in available:
                return mode_key
        return available[0]

    def _compute_focus(self) -> str:
        """Return the normalized compute-focus policy."""
        focus = getattr(self.config, "compute_focus", "auto")
        return focus if focus in _COMPUTE_FOCUS_VALUES else "auto"

    @staticmethod
    def _mode_compute_focus(mode_key: str) -> str:
        """Infer the resource focus represented by a concrete mode."""
        stable = NVBroadcastApp._stable_mode_key(mode_key)
        if stable in _GPU_AUTO_MODES:
            return "gpu"
        if stable in _CPU_AUTO_MODES:
            return "cpu"
        return "auto"

    def _resolved_mode_key(self) -> str:
        """Return the active concrete mode key."""
        return self.config.mode_key or NVBroadcastWindow._profile_and_comp_to_mode(
            self.config.performance_profile, self.config.compositing
        )

    @staticmethod
    def _stable_mode_key(mode_key: str) -> str:
        """Map premium or legacy modes onto the stable auto ladder."""
        return {
            "cuda_max": "doczeus",
            "zeus": "cuda_balanced",
            "killer": "cuda_perf",
        }.get(mode_key, mode_key)

    def _mode_label(self, mode_key: str) -> str:
        """Return a human-readable label for a mode."""
        return _MODE_LABELS.get(mode_key, mode_key)

    def _mode_quality_preset(self, mode_key: str) -> str | None:
        """Return the expected RVM quality preset for a stable named mode."""
        return _MODE_QUALITY_PRESETS.get(mode_key)

    def _profile_infer_height(
        self,
        profile_name: str,
        *,
        use_tensorrt: bool | None = None,
        use_fused_kernel: bool | None = None,
    ) -> int:
        """Return the target infer-height cap for the active profile/mode."""
        from nvbroadcast.core.config import PERFORMANCE_PROFILES

        profile = PERFORMANCE_PROFILES.get(profile_name, {})
        scale = float(profile.get("process_scale", 1.0))
        source_h = max(1, int(self.config.video.height))
        infer_h = int(round(source_h * scale)) & ~1
        infer_h = max(240, min(720, infer_h))

        if use_tensorrt is None:
            use_tensorrt = self.config.use_tensorrt
        if use_fused_kernel is None:
            use_fused_kernel = self.config.use_fused_kernel

        # Fused non-TRT fast mode stays quality-sensitive around hair and hand
        # gaps. Keep a source-capped 480p floor so CUDA Fast still has enough
        # matte detail to avoid jagged hair/finger edges without forcing 720p.
        if profile_name == "performance" and use_fused_kernel and not use_tensorrt:
            infer_h = min(source_h, max(480, infer_h))
        return infer_h

    @staticmethod
    def _is_very_weak_device(caps: dict) -> bool:
        """Return whether the detected hardware is likely latency-limited."""
        if caps.get("has_linux_arm64") and not caps.get("has_nvidia"):
            return True
        if not caps.get("has_nvidia") and not caps.get("has_apple_silicon"):
            return caps.get("cpu_cores", 4) <= 4
        if caps.get("has_nvidia") and caps.get("gpu_vram_mb", 0) <= 2048:
            return caps.get("cpu_cores", 4) <= 4
        return False

    def _recommended_capture_mode(self) -> tuple[int, int, int] | None:
        """Return a lighter capture mode recommendation when one exists."""
        from nvbroadcast.video.virtual_camera import list_camera_modes

        modes = list_camera_modes(self.config.video.camera_device)
        if not modes:
            return None

        preferred = [(640, 360, 30), (640, 480, 30), (800, 600, 30)]
        for width, height, fps in preferred:
            for mode in modes:
                if mode["width"] != width or mode["height"] != height:
                    continue
                supported = [f for f in mode["fps"] if f <= fps]
                if supported:
                    return width, height, max(supported)
                if mode["fps"]:
                    return width, height, min(mode["fps"], key=lambda value: abs(value - fps))

        smallest = min(modes, key=lambda mode: (mode["width"] * mode["height"], max(mode["fps"]) if mode["fps"] else 999))
        if not smallest["fps"]:
            return None
        return smallest["width"], smallest["height"], min(smallest["fps"], key=lambda value: abs(value - 30))

    def _recommendation_text(self, fallback_mode: str) -> str:
        """Build user-facing advice for lower-latency manual fallback."""
        focus_label = _COMPUTE_FOCUS_LABELS.get(self._compute_focus(), "Auto")
        lines = [
            f"Recommended Mode: {focus_label} or {self._mode_label(fallback_mode)}.",
        ]
        capture = self._recommended_capture_mode()
        if capture is not None:
            width, height, fps = capture
            if (
                self.config.video.width * self.config.video.height > width * height
                or self.config.video.fps > fps
            ):
                lines.append(
                    f"Recommended Camera Mode: {width}x{height} @ {fps} fps."
                )
                lines.append(
                    "Resolution/FPS changes are saved and apply on the next clean app start."
                )
        lines.append(
            "Your current saved settings stay unchanged until you explicitly change mode, profile, defaults, resolution, or FPS."
        )
        return "\n".join(lines)

    def _lower_recommendation_mode(self) -> str:
        """Return the next lower stable mode relative to the current one."""
        ladder = self._available_auto_modes()
        current = self._stable_mode_key(self._resolved_mode_key())
        if current in ladder:
            idx = ladder.index(current)
            if idx < len(ladder) - 1:
                return ladder[idx + 1]
            return current
        return self._preferred_auto_mode()

    def _maybe_warn_weak_device(self):
        """Warn once per launch when a very weak device uses a heavy manual mode."""
        if self._window is None or self.config.first_run or self.config.auto_mode:
            return False

        from nvbroadcast.core.config import detect_system_capabilities

        caps = detect_system_capabilities()
        if not self._is_very_weak_device(caps):
            return False

        resolved_mode = self._resolved_mode_key()
        fallback_mode = self._lower_recommendation_mode()
        capture = self._recommended_capture_mode()
        capture_heavy = (
            capture is not None
            and (
                self.config.video.width * self.config.video.height > capture[0] * capture[1]
                or self.config.video.fps > capture[2]
            )
        )
        if resolved_mode in ("cpu_light", "cpu_low") and not capture_heavy:
            return False

        title = "Weak device detected"
        reason = (
            "This hardware is likely to struggle with heavier live video modes.\n\n"
            f"{self._recommendation_text(fallback_mode)}"
        )
        self._window.set_status(
            f"Weak device detected. Consider Auto or {self._mode_label(fallback_mode)}."
        )
        self._window.show_advisory("weak-device", title, reason)
        return False

    def set_auto_mode_enabled(self, enabled: bool):
        """Enable or disable adaptive mode selection."""
        self.config.auto_mode = enabled
        if enabled:
            self.config.compute_focus = "auto"
        self._auto_tune_low_streak = 0
        self._auto_tune_high_streak = 0
        self._last_auto_tune_change = time.monotonic()
        self._manual_low_fps_streak = 0
        self._last_manual_warning = 0.0
        self._last_auto_capture_change = 0.0

        if enabled:
            resolved = self._preferred_auto_mode()
            detail = NVBroadcastWindow._mode_status_message(resolved)
            self.apply_mode_key(resolved, status=f"Auto: {detail}")
            capture = self._recommended_capture_mode()
            if capture is not None:
                current_rank = self._capture_mode_rank(self._current_capture_mode())
                target_rank = self._capture_mode_rank(capture)
                if target_rank < current_rank:
                    self._apply_capture_mode_choice(
                        *capture,
                        status_prefix="Auto capture:",
                        advisory_key="auto-capture-enable" if self._streaming else None,
                        advisory_title="Auto capture adjustment" if self._streaming else None,
                        advisory_reason=(
                            f"Auto mode saved a lighter camera mode ({capture[0]}x{capture[1]} @ {capture[2]} fps) "
                            "to improve stability on this hardware. The current session keeps running and the new "
                            "camera mode applies on the next clean app start."
                        ) if self._streaming else None,
                    )
        else:
            save_config(self.config)
            if self._window is not None and hasattr(self._window, "_sync_mode_selector"):
                self._window._sync_mode_selector()

    def set_compute_focus(self, focus: str, *, apply_mode: bool = True) -> bool:
        """Switch the high-level CPU/GPU resource policy."""
        if focus not in _COMPUTE_FOCUS_VALUES:
            return False

        self.config.compute_focus = focus
        self._auto_tune_low_streak = 0
        self._auto_tune_high_streak = 0
        self._last_auto_tune_change = time.monotonic()
        self._manual_low_fps_streak = 0
        self._last_manual_warning = 0.0

        if focus == "auto":
            self.set_auto_mode_enabled(True)
            return True

        self.config.auto_mode = False
        if apply_mode:
            mode_key = self._preferred_auto_mode()
            detail = NVBroadcastWindow._mode_status_message(mode_key)
            return self.apply_mode_key(
                mode_key,
                status=f"{_COMPUTE_FOCUS_LABELS[focus]}: {detail}",
            )

        self._sync_gpu_frame_path()
        save_config(self.config)
        if self._window is not None:
            if hasattr(self._window, "_sync_compute_focus_selector"):
                self._window._sync_compute_focus_selector()
            if hasattr(self._window, "_sync_mode_selector"):
                self._window._sync_mode_selector()
        return True

    def _auto_tune_tick(self):
        """Adapt between stable modes when live FPS stays too low."""
        if (
            not self._streaming
            or self._dependency_installer.busy
            or self._pending_start is not None
            or self._pipeline_teardown is not None
            or not self._any_video_effects_active()
        ):
            self._auto_tune_low_streak = 0
            self._auto_tune_high_streak = 0
            self._manual_low_fps_streak = 0
            return True

        fps = self._perf_monitor.fps
        if fps < 1.0:
            return True

        if not self.config.auto_mode:
            stable_mode = self._stable_mode_key(self._resolved_mode_key())
            target = _AUTO_MODE_TARGET_FPS.get(stable_mode, 15.0)
            if fps < max(8.0, target - 2.0):
                self._manual_low_fps_streak += 1
            else:
                self._manual_low_fps_streak = max(0, self._manual_low_fps_streak - 1)

            now = time.monotonic()
            if self._manual_low_fps_streak >= 3 and now - self._last_manual_warning >= 20.0:
                fallback_mode = self._lower_recommendation_mode()
                self._last_manual_warning = now
                if self._window is not None:
                    title = "Low live FPS detected"
                    reason = (
                        f"Processed video is currently rendering around {fps:.0f} fps in manual mode.\n\n"
                        f"{self._recommendation_text(fallback_mode)}"
                    )
                    self._window.set_status(
                        f"Low live FPS detected. Consider Auto or {self._mode_label(fallback_mode)}."
                    )
                    self._window.show_advisory("manual-low-fps", title, reason)
                self._manual_low_fps_streak = 0
            return True

        ladder = self._available_auto_modes()
        current = self.config.mode_key if self.config.mode_key in ladder else self._preferred_auto_mode()
        if current not in ladder:
            return True

        idx = ladder.index(current)
        target = _AUTO_MODE_TARGET_FPS.get(current, 15.0)
        now = time.monotonic()
        if now - self._last_auto_tune_change < 8.0:
            return True

        if fps < max(8.0, target - 2.0):
            self._auto_tune_low_streak += 1
            self._auto_tune_high_streak = 0
        else:
            self._auto_tune_low_streak = max(0, self._auto_tune_low_streak - 1)
            next_up = ladder[idx - 1] if idx > 0 else None
            next_up_target = _AUTO_MODE_TARGET_FPS.get(next_up, target) if next_up else target
            if next_up and fps > next_up_target + 2.0:
                self._auto_tune_high_streak += 1
            else:
                self._auto_tune_high_streak = max(0, self._auto_tune_high_streak - 1)

        if self._auto_tune_low_streak >= 3 and idx < len(ladder) - 1:
            next_mode = ladder[idx + 1]
            detail = NVBroadcastWindow._mode_status_message(next_mode)
            if self.apply_mode_key(
                next_mode,
                status=f"Auto: switched to {detail} to keep live FPS stable",
            ):
                self._last_auto_tune_change = now
                self._auto_tune_low_streak = 0
                self._auto_tune_high_streak = 0
        elif self._auto_tune_low_streak >= 3 and idx == len(ladder) - 1:
            next_capture = self._next_lower_capture_mode()
            if next_capture and now - self._last_auto_capture_change >= 20.0:
                if self._apply_capture_mode_choice(
                    *next_capture,
                    status_prefix="Auto capture: saved lighter camera mode",
                    advisory_key="auto-capture-low-fps",
                    advisory_title="Auto mode saved a lighter camera mode",
                    advisory_reason=(
                        "Auto mode is already on the lightest stable processing path, "
                        f"so it saved {next_capture[0]}x{next_capture[1]} @ {next_capture[2]} fps "
                        "for the next clean app start to reduce severe FPS collapse on this hardware."
                    ),
                ):
                    self._last_auto_capture_change = now
                    self._last_auto_tune_change = now
                    self._auto_tune_low_streak = 0
                    self._auto_tune_high_streak = 0
        elif self._auto_tune_high_streak >= 8 and idx > 0:
            next_mode = ladder[idx - 1]
            detail = NVBroadcastWindow._mode_status_message(next_mode)
            if self.apply_mode_key(
                next_mode,
                status=f"Auto: restored {detail}",
            ):
                self._last_auto_tune_change = now
                self._auto_tune_low_streak = 0
                self._auto_tune_high_streak = 0

        return True

    def set_compute_gpu(self, gpu_index: int):
        """Switch the GPU used for AI compute."""
        if gpu_index == self.config.compute_gpu:
            return

        # Detach first and wait for callbacks that captured the old processor.
        # This prevents old-device CuPy buffers from reaching a newly reloaded
        # VideoEffects backend on another GPU.
        if self._video_pipeline is not None:
            self._video_pipeline.set_frame_processor(
                None, None, wait_for_inflight=True)
        self._gpu_frame_path = None
        self._gpu_frame_path_failed = False

        self.config.compute_gpu = gpu_index
        self._video_effects._gpu_index = gpu_index
        self._perf_monitor.set_gpu_index(gpu_index)
        # Reload the model on the new GPU
        if self._video_effects.available:
            self._video_effects._cleanup_backend()
            self._video_effects.initialize()
        self._sync_gpu_frame_path()
        save_config(self.config)
        from nvbroadcast.core.gpu import detect_gpus
        gpus = detect_gpus()
        name = gpus[gpu_index].name if gpu_index < len(gpus) else f"GPU {gpu_index}"
        if self._window:
            self._window._update_gpu_info()
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

    def set_output_format(self, output_format: str):
        if output_format == self.config.video.output_format:
            return
        self.config.video.output_format = output_format
        save_config(self.config)
        if self._window:
            if self._streaming:
                self._window.set_status(
                    f"Format saved: {output_format}. Restart the app to apply."
                )
            else:
                self._window.set_status(f"Format: {output_format}")

    def set_vcam_device(self, device: str) -> bool:
        if not IS_LINUX:
            return False
        device = (device or "").strip() or VIRTUAL_CAM_DEVICE
        suffix = device.removeprefix("/dev/video")
        if not device.startswith("/dev/video") or not suffix.isdigit():
            if self._window:
                self._window.set_status(
                    "Virtual camera device must look like /dev/video10."
                )
            return False
        if os.path.exists(device) and not is_v4l2loopback_device(device):
            if self._window:
                self._window.set_status(
                    f"{device} is not a v4l2loopback virtual camera."
                )
            return False

        if device == self.config.video.vcam_device:
            return True

        self.config.video.vcam_device = device
        save_config(self.config)

        if self._streaming:
            if self._window:
                self._window.set_status(
                    f"Virtual camera saved: {device}. Restart broadcast to apply."
                )
            return True

        try:
            self._vcam_device = ensure_virtual_camera(self._preferred_vcam_device())
            self._vcam_available = True
            status = f"Virtual camera: {self._active_vcam_device()}"
        except RuntimeError as e:
            self._vcam_device = None
            self._vcam_available = False
            first_line = str(e).splitlines()[0]
            status = f"Virtual camera saved: {device}. {first_line}"

        if self._window:
            self._window.set_status(status)
        return True

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
        self.config.video.mirror = enabled
        save_config(self.config)

    def set_edge_refine(self, enabled: bool):
        """Toggle neural edge refinement for Zeus/Killer modes."""
        self._video_effects._edge_refine_enabled = enabled
        self.config.premium_edge_refine = enabled
        save_config(self.config)

    def set_edge_param(self, param: str, value: float):
        """Update a single edge refinement parameter."""
        setattr(self.config.video.edge, param, value)
        self._video_effects.update_edge_params(**{param: value})
        save_config(self.config)

    def _get_valid_fps(
        self,
        width: int,
        height: int,
        desired_fps: int,
        camera_device: str | None = None,
    ) -> int:
        """Return the closest supported FPS for the given resolution."""
        from nvbroadcast.video.virtual_camera import list_camera_modes
        modes = list_camera_modes(camera_device or self.config.video.camera_device)
        for mode in modes:
            if mode["width"] == width and mode["height"] == height:
                supported = mode["fps"]
                if desired_fps in supported:
                    return desired_fps
                if not supported:
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

        if self._streaming:
            # Live v4l2loopback reconfiguration is currently unstable on some
            # systems. Save the new mode immediately but defer applying it
            # until the next clean app start instead of hanging the session.
            if self._window:
                self._window.set_status(
                    f"Resolution saved: {width}x{height} @ {self.config.video.fps}fps. "
                    "Restart the app to apply."
                )
            return

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
            if self._window:
                self._window.set_status(
                    f"FPS saved: {valid_fps}. Restart the app to apply."
                )
            return

        if self._window:
            self._window.set_status(f"FPS: {valid_fps}")

    def set_autoframe(self, enabled: bool):
        if getattr(self, '_restoring', False):
            return
        self._autoframe.enabled = enabled
        self.config.video.auto_frame = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_autoframe_zoom(self, value: float):
        self._autoframe.zoom_level = value
        self.config.video.auto_frame_zoom = value
        save_config(self.config)

    def set_autoframe_mode(self, mode: str):
        if getattr(self, '_restoring', False):
            return
        if mode not in ("center", "stable"):
            mode = "center"
        self._autoframe.mode = mode
        self.config.video.auto_frame_mode = mode
        save_config(self.config)

    # --- Beautification ---

    def set_beautify(self, enabled: bool):
        if getattr(self, '_restoring', False):
            return
        self._beautifier.enabled = enabled
        self.config.video.beauty.enabled = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_beautify_param(self, param: str, value: float):
        """Set a beautification parameter (skin_smooth, denoise, edge_darken, enhance, sharpen)."""
        setattr(self._beautifier, param, value)
        if hasattr(self.config.video.beauty, param):
            setattr(self.config.video.beauty, param, value)
        save_config(self.config)

    # --- Eye Contact ---

    def set_eye_contact(self, enabled: bool):
        if getattr(self, '_restoring', False):
            return
        self._eye_contact.enabled = enabled
        self.config.video.eye_contact = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_eye_contact_intensity(self, value: float):
        self._eye_contact.intensity = value
        self.config.video.eye_contact_intensity = value
        save_config(self.config)

    def set_eye_contact_mode(self, mode: str):
        if getattr(self, '_restoring', False):
            return
        if mode not in ("natural", "gaze_lock"):
            mode = "natural"
        self._eye_contact.mode = mode
        self.config.video.eye_contact_mode = mode
        save_config(self.config)

    # --- Face Relighting ---

    def set_relighting(self, enabled: bool):
        if getattr(self, '_restoring', False):
            return
        self._relighter.enabled = enabled
        self.config.video.relighting = enabled
        self._update_pipeline_mode()
        save_config(self.config)

    def set_relighting_intensity(self, value: float):
        self._relighter.intensity = value
        self.config.video.relighting_intensity = value
        save_config(self.config)

    # --- Recording ---

    def start_recording(self):
        """Start recording to ~/Videos/NVBroadcast_<timestamp>.mp4."""
        import time
        from pathlib import Path
        videos_dir = Path.home() / "Videos"
        videos_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = str(videos_dir / f"NVBroadcast_{timestamp}.mp4")
        if self._idle_active:
            self._exit_idle("recording started")
        if self._video_pipeline:
            self._video_pipeline.start_recording(filepath)
        self._last_recording_path = filepath
        return filepath

    def stop_recording(self):
        if self._video_pipeline:
            self._video_pipeline.stop_recording()

    @property
    def is_recording(self) -> bool:
        return self._video_pipeline and self._video_pipeline.is_recording

    # --- Meeting (Recording + AI Transcription) ---

    def start_meeting(self) -> str:
        """Start meeting: records video+audio and transcribes speech."""
        from pathlib import Path

        self._meeting_session_id, self._meeting_session_dir = create_session()
        self._meeting_video_path = str(self._meeting_session_dir / "meeting.mp4")
        self._meeting_audio_path = str(self._meeting_session_dir / "meeting_audio.wav")

        filepath = self._meeting_video_path
        if self._video_pipeline:
            self._video_pipeline.start_recording(filepath)
        self._last_recording_path = filepath

        self._meeting_capture = MeetingAudioCapture()
        self._meeting_capture.set_sample_callback(self._transcriber.feed_audio)
        speaker_device = self.config.audio.speaker_device
        if self._window and getattr(self._window, "_speaker_selector", None):
            selected_speaker = self._window._speaker_selector.get_selected_device()
            if selected_speaker:
                speaker_device = selected_speaker
        try:
            self._meeting_capture.build(
                self.config.audio.mic_device,
                speaker_device,
                self._meeting_audio_path,
            )
            self._meeting_capture.start()
        except Exception as exc:
            print(f"[NV Broadcast] Meeting audio capture unavailable: {exc}")
            self._meeting_capture = None

        if not self._transcriber.start():
            if self._meeting_capture:
                self._meeting_capture.stop()
                self._meeting_capture = None
            self.stop_recording()
            self._meeting_session_id = ""
            self._meeting_session_dir = None
            self._meeting_audio_path = ""
            self._meeting_video_path = ""
            if self._window:
                self._window.set_status("Meeting transcription could not start")
            return ""
        self._meeting_active = True
        if self._window:
            self._window.reset_live_meeting_view()
        print(f"[NV Broadcast] Meeting started: {filepath}")
        return filepath

    def stop_meeting(self) -> str:
        """Stop meeting, save transcript + summary."""
        import time
        from pathlib import Path
        self._meeting_active = False
        self.stop_recording()
        if self._meeting_capture:
            self._meeting_capture.stop()
            self._meeting_capture = None
        segments = self._transcriber.stop()
        if self._meeting_audio_path and Path(self._meeting_audio_path).exists():
            try:
                if self._window:
                    self._window.set_status("Finalizing high-accuracy meeting transcript...")
                final_segments = self._transcriber.transcribe_file(self._meeting_audio_path)
                if final_segments:
                    segments = final_segments
                    self._transcriber.replace_segments(final_segments)
            except Exception as exc:
                print(f"[NV Broadcast] Final meeting transcription pass failed: {exc}")
        transcript_path = ""
        transcript_srt_path = ""
        notes_path = ""
        if segments:
            base_path = str(self._meeting_session_dir / "transcript")
            transcript_path = save_transcript(segments, base_path, format="txt")
            transcript_srt_path = save_transcript(segments, base_path, format="srt")

            transcript_text = self._transcriber.get_full_transcript()
            duration = segments[-1].end_time if segments else 0
            notes = self._summarizer.summarize(transcript_text, duration)
            notes_md = self._summarizer.format_notes(notes)
            notes_path = str(self._meeting_session_dir / "notes.md")
            Path(notes_path).write_text(notes_md)
            print(f"[NV Broadcast] Meeting notes saved: {notes_path}")

            session = MeetingSession(
                session_id=self._meeting_session_id,
                created_at=int(time.time()),
                title=notes.title,
                summary=notes.summary,
                transcript_preview="\n".join(seg.text for seg in segments[:6])[:600],
                duration_seconds=duration,
                notes_path=notes_path,
                transcript_path=transcript_path,
                transcript_srt_path=transcript_srt_path,
                audio_path=self._meeting_audio_path,
                video_path=self._meeting_video_path,
            )
            save_session(session)
            if self._window:
                self._window.load_meeting_sessions(self.list_meeting_sessions())
                self._window.show_meeting_session(session)

        print(f"[NV Broadcast] Meeting ended. Transcript: {transcript_path}")
        self._meeting_session_id = ""
        self._meeting_session_dir = None
        self._meeting_audio_path = ""
        self._meeting_video_path = ""
        return notes_path or transcript_path

    def stop_meeting_async(self, callback=None) -> bool:
        """Stop meeting quickly and finalize transcript/notes off the UI thread."""
        import threading

        if not self._meeting_active or self._meeting_finalizing:
            return False

        meeting_session_id = self._meeting_session_id
        meeting_session_dir = self._meeting_session_dir
        meeting_audio_path = self._meeting_audio_path
        meeting_video_path = self._meeting_video_path

        self._meeting_active = False
        self._meeting_finalizing = True
        self.stop_recording()
        if self._meeting_capture:
            self._meeting_capture.stop()
            self._meeting_capture = None
        segments = self._transcriber.stop()

        self._meeting_session_id = ""
        self._meeting_session_dir = None
        self._meeting_audio_path = ""
        self._meeting_video_path = ""

        def _worker():
            result_path = ""
            status = "Meeting ended"
            session = None
            try:
                result_path, session = self._finalize_meeting_outputs(
                    meeting_session_id,
                    meeting_session_dir,
                    meeting_audio_path,
                    meeting_video_path,
                    segments,
                )
                status = f"Meeting saved: {result_path}" if result_path else "Meeting ended"
            except Exception as exc:
                print(f"[NV Broadcast] Meeting finalization failed: {exc}")
                status = "Meeting ended, but transcript finalization failed"
            finally:
                def _finish():
                    self._meeting_finalizing = False
                    if session and self._window:
                        self._window.load_meeting_sessions(self.list_meeting_sessions())
                        self._window.show_meeting_session(session)
                    if callback:
                        callback(result_path, status)
                    return False
                GLib.idle_add(_finish)

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def _finalize_meeting_outputs(
        self,
        meeting_session_id: str,
        meeting_session_dir,
        meeting_audio_path: str,
        meeting_video_path: str,
        segments,
    ):
        import time
        from pathlib import Path

        if meeting_session_dir is None:
            return "", None

        if meeting_audio_path and Path(meeting_audio_path).exists():
            try:
                final_segments = self._transcriber.transcribe_file(meeting_audio_path)
                if final_segments:
                    segments = final_segments
                    self._transcriber.replace_segments(final_segments)
            except Exception as exc:
                print(f"[NV Broadcast] Final meeting transcription pass failed: {exc}")

        transcript_path = ""
        transcript_srt_path = ""
        notes_path = ""
        session = None

        if segments:
            base_path = str(meeting_session_dir / "transcript")
            transcript_path = save_transcript(segments, base_path, format="txt")
            transcript_srt_path = save_transcript(segments, base_path, format="srt")

            transcript_text = self._transcriber.get_full_transcript()
            duration = segments[-1].end_time if segments else 0
            notes = self._summarizer.summarize(transcript_text, duration)
            notes_md = self._summarizer.format_notes(notes)
            notes_path = str(meeting_session_dir / "notes.md")
            Path(notes_path).write_text(notes_md)
            print(f"[NV Broadcast] Meeting notes saved: {notes_path}")

            session = MeetingSession(
                session_id=meeting_session_id,
                created_at=int(time.time()),
                title=notes.title,
                summary=notes.summary,
                transcript_preview="\n".join(seg.text for seg in segments[:6])[:600],
                duration_seconds=duration,
                notes_path=notes_path,
                transcript_path=transcript_path,
                transcript_srt_path=transcript_srt_path,
                audio_path=meeting_audio_path,
                video_path=meeting_video_path,
            )
            save_session(session)

        print(f"[NV Broadcast] Meeting ended. Transcript: {transcript_path}")
        return notes_path or transcript_path, session

    @property
    def meeting_active(self) -> bool:
        return self._meeting_active

    @property
    def meeting_finalizing(self) -> bool:
        return self._meeting_finalizing

    @property
    def dependency_installer(self) -> DependencyInstaller:
        return self._dependency_installer

    def list_meeting_sessions(self) -> list[MeetingSession]:
        return list_sessions()

    def load_meeting_file(self, path: str) -> str:
        from nvbroadcast.core.meeting_store import read_file
        return read_file(path)

    def _on_transcript_segment(self, segment):
        if self._window is None:
            return

        def _update():
            transcript = self._transcriber.get_timestamped_transcript()
            notes = self._summarizer.summarize(
                self._transcriber.get_full_transcript(),
                segment.end_time,
            )
            self._window.update_live_meeting_summary(notes.summary, transcript)
            return False

        GLib.idle_add(_update)

    # --- Microphone Selection ---

    def list_microphones(self) -> list[dict]:
        from nvbroadcast.audio.devices import list_microphones
        return list_microphones()

    def set_microphone(self, device: str):
        self.config.audio.mic_device = device
        save_config(self.config)
        if self._audio_pipeline is not None:
            self._rebuild_audio_pipeline(restart=self._audio_pipeline._running)

    def set_speaker_device(self, device: str):
        self.config.audio.speaker_device = device
        save_config(self.config)
        if self.config.audio.speaker_denoise:
            self._refresh_speaker_monitor()

    # --- Multi-camera ---

    def switch_camera(self, device: str):
        """Hot-switch to a different camera device."""
        if self.config.video.camera_device == device:
            return

        from nvbroadcast.video.virtual_camera import select_camera_mode

        selected_mode = select_camera_mode(
            device,
            self.config.video.width,
            self.config.video.height,
            self.config.video.fps,
        )
        self.config.video.camera_device = device
        self.config.video.width = selected_mode["width"]
        self.config.video.height = selected_mode["height"]
        self.config.video.fps = selected_mode["fps"]
        save_config(self.config)

        if self._window:
            self._window.sync_video_input_controls(self.config)

        if self._streaming:
            self.start_pipeline(device, self.config.video.output_format)
        elif self._window:
            self._window.set_status(
                f"Camera: {device} {self.config.video.width}x"
                f"{self.config.video.height}@{self.config.video.fps}fps"
            )

    # --- Performance Monitor ---

    @property
    def perf_monitor(self) -> PerfMonitor:
        return self._perf_monitor

    # --- Audio ---

    def _resolved_audio_capture_device(self) -> str:
        from nvbroadcast.audio.devices import resolve_pipewire_target
        from nvbroadcast.audio.virtual_mic import virtual_mic_backend

        if IS_LINUX and virtual_mic_backend() == "pulse":
            return self.config.audio.mic_device
        return resolve_pipewire_target(self.config.audio.mic_device)

    def _ensure_audio_pipeline(self) -> AudioPipeline:
        if self._audio_pipeline is None:
            self._audio_pipeline = AudioPipeline()
            self._audio_pipeline.configure(
                mic_device=self._resolved_audio_capture_device(),
                sample_rate=48000,
            )
            self._audio_pipeline.build()
        return self._audio_pipeline

    def _audio_pipeline_should_publish(self) -> bool:
        """Keep the exported mic live while the app is running.

        Users select `nvbroadcast` in meeting apps and expect it to keep working
        even when processing toggles are off. In that idle state the pipeline is
        just passthrough, while noise removal / voice FX still remain optional.
        """
        return IS_LINUX and has_virtual_mic_backend()

    def _audio_pipeline_should_run(self) -> bool:
        if self._audio_pipeline_should_publish():
            return True
        if self.config.audio.noise_removal:
            return True
        return bool(self.config.audio.voice_fx_enabled)

    def _refresh_audio_pipeline(self):
        pipeline = self._audio_pipeline
        if pipeline is None:
            if self._audio_pipeline_should_run():
                pipeline = self._ensure_audio_pipeline()
            else:
                return

        if self._audio_pipeline_should_run():
            pipeline.start()
        else:
            pipeline.stop()

    def _rebuild_audio_pipeline(self, restart: bool | None = None):
        if self._audio_pipeline is None:
            return

        should_restart = self._audio_pipeline._running if restart is None else restart
        self._audio_pipeline.stop()
        self._audio_pipeline.configure(
            mic_device=self._resolved_audio_capture_device(),
            sample_rate=48000,
        )
        self._audio_pipeline.build()
        if should_restart and self._audio_pipeline_should_run():
            self._audio_pipeline.start()

    def _restart_audio_pipeline_for_live_settings(self):
        if self._audio_pipeline is None or not self._audio_pipeline._running:
            return
        if not self._audio_pipeline.uses_helper_process:
            return
        self._rebuild_audio_pipeline(restart=True)

    def _apply_voice_fx_settings_from_config(self, pipeline=None):
        from nvbroadcast.audio.voice_fx import VoiceFXSettings, normalize_voice_fx_preset_name

        if pipeline is None:
            pipeline = self._ensure_audio_pipeline()

        self.config.audio.voice_fx_preset = normalize_voice_fx_preset_name(
            self.config.audio.voice_fx_preset
        )
        pipeline.voice_fx.settings = VoiceFXSettings(
            bass_boost=self.config.audio.voice_fx_bass_boost,
            treble=self.config.audio.voice_fx_treble,
            warmth=self.config.audio.voice_fx_warmth,
            compression=self.config.audio.voice_fx_compression,
            gate_threshold=self.config.audio.voice_fx_gate_threshold,
            gain=self.config.audio.voice_fx_gain,
        )
        return pipeline

    def set_noise_removal(self, enabled: bool):
        self.config.audio.noise_removal = enabled
        pipeline = self._ensure_audio_pipeline()
        pipeline.effects.engine = self.config.audio.noise_engine
        pipeline.effects.enabled = enabled
        self._refresh_audio_pipeline()
        save_config(self.config)

    def set_noise_engine(self, engine: str):
        """Switch the denoiser engine ("auto" = DeepFilterNet, "rnnoise")."""
        engine = engine if engine in ("auto", "rnnoise") else "auto"
        self.config.audio.noise_engine = engine
        pipeline = self._ensure_audio_pipeline()
        pipeline.effects.engine = engine
        self._restart_audio_pipeline_for_live_settings()
        save_config(self.config)

    def set_noise_intensity(self, value: float):
        self.config.audio.noise_intensity = value
        if self._audio_pipeline:
            self._audio_pipeline.effects.intensity = value
        self._restart_audio_pipeline_for_live_settings()
        save_config(self.config)

    def set_voice_fx_enabled(self, enabled: bool):
        pipeline = self._apply_voice_fx_settings_from_config()
        pipeline.voice_fx.enabled = enabled
        self._refresh_audio_pipeline()
        self.config.audio.voice_fx_enabled = enabled
        save_config(self.config)

    def set_voice_fx_use_gpu(self, enabled: bool):
        pipeline = self._ensure_audio_pipeline()
        pipeline.voice_fx.use_gpu = enabled
        self.config.audio.voice_fx_use_gpu = pipeline.voice_fx.use_gpu
        self._restart_audio_pipeline_for_live_settings()
        save_config(self.config)

    def _sync_voice_fx_config(self, preset_name: str | None = None):
        if self._audio_pipeline is None or self._audio_pipeline._voice_fx is None:
            return
        settings = self._audio_pipeline.voice_fx.settings
        self.config.audio.voice_fx_enabled = self._audio_pipeline.voice_fx.enabled
        self.config.audio.voice_fx_use_gpu = self._audio_pipeline.voice_fx.use_gpu
        if preset_name is not None:
            self.config.audio.voice_fx_preset = preset_name
        self.config.audio.voice_fx_bass_boost = settings.bass_boost
        self.config.audio.voice_fx_treble = settings.treble
        self.config.audio.voice_fx_warmth = settings.warmth
        self.config.audio.voice_fx_compression = settings.compression
        self.config.audio.voice_fx_gate_threshold = settings.gate_threshold
        self.config.audio.voice_fx_gain = settings.gain

    def set_voice_fx_preset(self, preset_name: str):
        from nvbroadcast.audio.voice_fx import get_voice_fx_preset, normalize_voice_fx_preset_name

        preset = get_voice_fx_preset(preset_name)
        if preset is None:
            return

        pipeline = self._ensure_audio_pipeline()
        pipeline.voice_fx.settings = preset
        self._sync_voice_fx_config(preset_name=normalize_voice_fx_preset_name(preset_name))
        self._restart_audio_pipeline_for_live_settings()
        save_config(self.config)

    def set_voice_fx_param(self, param: str, value: float):
        pipeline = self._ensure_audio_pipeline()
        setattr(pipeline.voice_fx.settings, param, value)
        self._sync_voice_fx_config()
        self._restart_audio_pipeline_for_live_settings()
        save_config(self.config)

    def _ensure_speaker_monitor(self) -> SpeakerMonitor:
        if self._speaker_monitor is None:
            self._speaker_monitor = SpeakerMonitor()
        return self._speaker_monitor

    def _refresh_speaker_monitor(self):
        if not self.config.audio.speaker_denoise:
            if self._speaker_monitor:
                self._speaker_monitor.stop()
            return

        monitor = self._ensure_speaker_monitor()
        monitor.configure(
            speaker_device=self.config.audio.speaker_device,
            sample_rate=48000,
        )
        monitor.build()
        monitor.effects.enabled = True
        monitor.start()

    def set_speaker_denoise(self, enabled: bool):
        self.config.audio.speaker_denoise = enabled
        if enabled:
            self._refresh_speaker_monitor()
        else:
            if self._speaker_monitor:
                self._speaker_monitor.stop()
        save_config(self.config)

    # --- Lifecycle ---

    def do_shutdown(self):
        save_config(self.config)
        if self._hotkey_manager is not None:
            self._hotkey_manager.close()
            self._hotkey_manager = None
        if self._vcam_monitor:
            self._vcam_monitor.stop()
            self._vcam_monitor = None
        # Unregister the SNI item and its dbusmenu explicitly; otherwise the
        # tray host only notices when the bus connection dies and a stale
        # icon can linger. The legacy tray has no shutdown, hence the guard.
        if self._tray is not None and hasattr(self._tray, "shutdown"):
            try:
                self._tray.shutdown()
            except Exception:
                pass
            self._tray = None
        if self._meeting_capture:
            self._meeting_capture.stop()
            self._meeting_capture = None
        if self._video_pipeline:
            self._video_pipeline.shutdown_sync()
            self._video_pipeline = None
        elif self._pipeline_teardown:
            self._pipeline_teardown.shutdown_sync()
        self._pipeline_teardown = None
        if self._audio_pipeline:
            self._audio_pipeline.stop()
        if self._speaker_monitor:
            self._speaker_monitor.stop()
        self._transcriber.cleanup()
        self._video_effects.cleanup()
        self._autoframe.cleanup()
        self._beautifier.cleanup()
        self._perf_monitor.stop()
        Adw.Application.do_shutdown(self)

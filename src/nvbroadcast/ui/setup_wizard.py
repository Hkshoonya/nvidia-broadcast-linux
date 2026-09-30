# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""First-run setup wizard — auto-detects system and configures optimally."""

import threading
import weakref

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GObject, GLib

from nvbroadcast.core.config import detect_system_capabilities
from nvbroadcast.core.gpu import detect_gpus


# Unified modes: each combines compositing + performance
SETUP_MODES = [
    {
        "key": "auto",
        "label": "Auto - Adaptive",
        "description": "Automatically picks the best stable mode for this device and adjusts if live FPS stays low.",
        "compositing": "auto",
        "profile": "auto",
        "needs_cupy": False,
        "needs_gl": False,
        "min_vram": 0,
    },
    {
        "key": "gpu_cuda_best",
        "label": "CUDA GPU - Maximum Quality",
        "description": "CUDA mode runtime for GPU compositing and ONNX GPU inference.",
        "compositing": "cupy",
        "profile": "max_quality",
        "needs_cupy": True,
        "needs_gl": False,
        "min_vram": 4096,
    },
    {
        "key": "gpu_quality",
        "label": "GPU - Best Quality",
        "description": "GPU-assisted compositing. 30fps, every frame. Very low CPU.",
        "compositing": "gstreamer_gl",
        "profile": "max_quality",
        "needs_cupy": False,
        "needs_gl": True,
        "min_vram": 2048,
    },
    {
        "key": "gpu_balanced",
        "label": "GPU - Balanced",
        "description": "GPU-assisted compositing. 20fps effects. Best balance.",
        "compositing": "gstreamer_gl",
        "profile": "balanced",
        "needs_cupy": False,
        "needs_gl": True,
        "min_vram": 2048,
    },
    {
        "key": "cpu_quality",
        "label": "CPU - High Quality",
        "description": "CPU inference and compositing. Highest CPU quality; no video GPU workload.",
        "compositing": "cpu",
        "profile": "max_quality",
        "needs_cupy": False,
        "needs_gl": False,
        "min_vram": 0,
    },
    {
        "key": "cpu_light",
        "label": "CPU - Light",
        "description": "CPU inference and compositing with fewer effect frames.",
        "compositing": "cpu",
        "profile": "performance",
        "needs_cupy": False,
        "needs_gl": False,
        "min_vram": 0,
    },
    {
        "key": "low_end",
        "label": "Low-End System",
        "description": "Minimal resources. 10fps, half resolution.",
        "compositing": "cpu",
        "profile": "potato",
        "needs_cupy": False,
        "needs_gl": False,
        "min_vram": 0,
    },
]


class SetupWizard(Adw.Window):
    """Auto-detects system, recommends best config, installs what's needed."""

    __gsignals__ = {
        "setup-complete": (GObject.SignalFlags.RUN_FIRST, None, (str, int, str)),
    }

    def __init__(self, parent, app):
        super().__init__(
            transient_for=parent,
            modal=True,
            title="NVIDIA Broadcast - Setup",
            default_width=580,
            default_height=580,
        )

        self._app = app
        self._gpus = detect_gpus()
        configured_gpu = max(0, int(app.config.compute_gpu))
        detected_gpu_indexes = {gpu.index for gpu in self._gpus}
        self._selected_gpu = (
            configured_gpu
            if not self._gpus or configured_gpu in detected_gpu_indexes
            else self._gpus[0].index
        )
        self._caps = detect_system_capabilities(self._selected_gpu)
        self._caps_gpu = self._selected_gpu
        self._selected_mode_key = self._caps["recommended_mode"]
        self._install_key = ""
        self._install_in_flight = False
        self._capability_refresh_generation = 0
        self._capability_refresh_in_flight = False
        self._capability_refresh_closed = False
        self.connect("close-request", self._on_close_requested)

        main = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Header
        header = Adw.HeaderBar()
        header.add_css_class("flat")
        title = Gtk.Label(label="First-Time Setup")
        title.add_css_class("title-2")
        header.set_title_widget(title)
        main.append(header)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_start(24)
        content.set_margin_end(24)
        content.set_margin_top(8)
        content.set_margin_bottom(16)

        intro = Gtk.Label(
            label=(
                "This setup helps you pick the right mode for your machine.\n"
                "Auto mode is recommended for most users. GPU modes give the best quality, "
                "CPU modes are the safest fallback, and some premium paths download extra "
                "runtimes on demand."
            )
        )
        intro.set_wrap(True)
        intro.set_xalign(0)
        content.append(intro)

        # System Info
        sys_frame = Gtk.Frame(label="Your System")
        sys_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        sys_box.set_margin_start(8)
        sys_box.set_margin_end(8)
        sys_box.set_margin_top(6)
        sys_box.set_margin_bottom(6)

        c = self._caps
        self._system_cpu_label = Gtk.Label()
        self._system_gpu_label = Gtk.Label()
        self._system_features_label = Gtk.Label()
        for lbl in (
            self._system_cpu_label,
            self._system_gpu_label,
            self._system_features_label,
        ):
            lbl.set_xalign(0)
            sys_box.append(lbl)
        self._update_system_info(c)

        sys_frame.set_child(sys_box)
        content.append(sys_frame)

        # GPU Selection (only if multiple)
        if len(self._gpus) > 1:
            gpu_frame = Gtk.Frame(label="GPU for AI Effects")
            gpu_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            gpu_box.set_margin_start(8)
            gpu_box.set_margin_end(8)
            gpu_box.set_margin_top(6)
            gpu_box.set_margin_bottom(6)
            self._gpu_group = None
            selected_gpu_button = None
            self._gpu_buttons = {}
            for g in self._gpus:
                btn = Gtk.CheckButton(
                    label=f"GPU {g.index}: {g.name} ({g.memory_total_mb} MB)"
                )
                if self._gpu_group is None:
                    self._gpu_group = btn
                else:
                    btn.set_group(self._gpu_group)
                if g.index == self._selected_gpu:
                    selected_gpu_button = btn
                self._gpu_buttons[g.index] = btn
                gpu_box.append(btn)
            (selected_gpu_button or self._gpu_group).set_active(True)
            for gpu_index, btn in self._gpu_buttons.items():
                btn.connect("toggled", self._on_gpu_toggled, gpu_index)
            gpu_frame.set_child(gpu_box)
            content.append(gpu_frame)
        else:
            self._gpu_buttons = {}

        # Processing Mode
        mode_frame = Gtk.Frame(label="Processing Mode")
        mode_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        mode_box.set_margin_start(8)
        mode_box.set_margin_end(8)
        mode_box.set_margin_top(6)
        mode_box.set_margin_bottom(6)

        self._mode_group = None
        self._mode_buttons = {}
        for mode in SETUP_MODES:
            available, label = self._mode_choice_state(mode, c)
            btn = Gtk.CheckButton(label=label)
            btn.set_sensitive(available)

            desc = Gtk.Label(label=f"  {mode['description']}")
            desc.set_xalign(0)
            desc.add_css_class("dim-label")
            desc.set_margin_start(24)
            desc.set_wrap(True)

            if self._mode_group is None:
                self._mode_group = btn
            else:
                btn.set_group(self._mode_group)

            if mode["key"] == self._caps["recommended_mode"]:
                btn.set_active(True)

            btn.connect("toggled", self._on_mode_toggled, mode["key"])
            self._mode_buttons[mode["key"]] = btn
            mode_box.append(btn)
            mode_box.append(desc)

        mode_frame.set_child(mode_box)
        content.append(mode_frame)

        # Status label for install progress
        self._status_label = Gtk.Label(label="")
        self._status_label.set_xalign(0)
        self._status_label.set_wrap(True)
        content.append(self._status_label)

        # Note
        note = Gtk.Label(
            label="You can change this anytime from the Mode dropdown in the app."
        )
        note.set_xalign(0)
        note.set_wrap(True)
        note.add_css_class("dim-label")
        content.append(note)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._skip_btn = Gtk.Button(label="Skip for Now")
        self._skip_btn.add_css_class("flat")
        self._skip_btn.connect("clicked", self._on_skip)
        actions.append(self._skip_btn)

        self._start_btn = Gtk.Button(label="Apply Selection")
        self._start_btn.add_css_class("suggested-action")
        self._start_btn.set_margin_top(4)
        self._start_btn.connect("clicked", self._on_start)
        actions.append(self._start_btn)
        content.append(actions)

        scroll = Gtk.ScrolledWindow()
        scroll.set_child(content)
        scroll.set_vexpand(True)
        main.append(scroll)

        self.set_content(main)
        installer = self._app.dependency_installer
        self._installer_signal_ids = (
            installer.connect("job-started", self._on_install_started),
            installer.connect("job-progress", self._on_install_progress),
            installer.connect("job-completed", self._on_install_completed),
        )

    @staticmethod
    def _mode_choice_state(mode, caps):
        """Return whether a setup mode can be chosen and its current label."""
        available = True
        reason = ""
        needs_gpu = mode["needs_gl"] or mode["needs_cupy"]
        if needs_gpu and not caps["has_nvidia"]:
            available = False
            reason = " [needs NVIDIA GPU]"
        elif mode["min_vram"] > caps["gpu_vram_mb"]:
            available = False
            reason = f" [needs {mode['min_vram']}MB VRAM]"
        elif mode["needs_gl"] and not caps["has_gl_compositor"]:
            available = False
            reason = " [needs GStreamer GL plugins]"
        elif mode["needs_cupy"] and not caps["has_cupy"]:
            # The runtime can be installed after this mode is selected.
            reason = " [will install CUDA runtime ~2GB]"

        label = mode["label"]
        if mode["key"] == caps["recommended_mode"]:
            label += "  ★ recommended"
        return available, f"{label}{reason}"

    def _update_system_info(self, caps):
        """Update the capability summary for the tentatively selected GPU."""
        self._system_cpu_label.set_text(f"CPU: {caps['cpu_cores']} cores")
        gpu_text = (
            f"GPU: {caps['gpu_name']} ({caps['gpu_vram_mb']} MB)"
            if caps["has_nvidia"]
            else "GPU: None detected"
        )
        self._system_gpu_label.set_text(gpu_text)
        features = []
        if caps["has_gl_compositor"]:
            features.append("OpenGL compositor")
        if caps["has_cupy"]:
            features.append("CUDA mode runtime")
        self._system_features_label.set_text(
            f"Available: {', '.join(features)}" if features else ""
        )
        self._system_features_label.set_visible(bool(features))

    def _update_action_sensitivity(self):
        enabled = not (
            getattr(self, "_capability_refresh_in_flight", False)
            or getattr(self, "_install_in_flight", False)
        )
        self._start_btn.set_sensitive(enabled)
        self._skip_btn.set_sensitive(enabled)

    def _apply_capabilities(self, gpu_index, caps):
        """Apply one current worker result to the GTK controls."""
        self._caps = caps
        self._caps_gpu = gpu_index
        self._update_system_info(caps)

        selectable = {}
        for mode in SETUP_MODES:
            available, label = self._mode_choice_state(mode, caps)
            selectable[mode["key"]] = available
            button = self._mode_buttons[mode["key"]]
            button.set_label(label)
            button.set_sensitive(available)

        if not selectable.get(self._selected_mode_key, False):
            candidates = (
                caps.get("recommended_mode"),
                caps.get("recommended_resolved_mode"),
                "auto",
                "cpu_quality",
                "cpu_light",
                "low_end",
            )
            fallback = next(
                key for key in candidates if key and selectable.get(key, False)
            )
            self._selected_mode_key = fallback
            self._mode_buttons[fallback].set_active(True)

    def _on_gpu_toggled(self, btn, gpu_index):
        if not btn.get_active() or self._capability_refresh_closed:
            return
        gpu_index = max(0, int(gpu_index))
        if gpu_index == self._selected_gpu and gpu_index == self._caps_gpu:
            return
        self._selected_gpu = gpu_index
        self._request_capability_refresh(gpu_index)

    def _request_capability_refresh(self, gpu_index):
        """Probe a tentative GPU away from the GTK thread."""
        self._capability_refresh_generation += 1
        generation = self._capability_refresh_generation
        self._capability_refresh_in_flight = True
        self._update_action_sensitivity()
        self._status_label.set_text(f"Checking GPU {gpu_index} capabilities…")
        wizard_ref = weakref.ref(self)

        def probe():
            caps = None
            error = ""
            try:
                caps = detect_system_capabilities(gpu_index)
            except Exception as exc:
                error = f"Capability check failed: {exc}"
            wizard = wizard_ref()
            if wizard is not None:
                GLib.idle_add(
                    wizard._finish_capability_refresh,
                    generation,
                    gpu_index,
                    caps,
                    error,
                )

        threading.Thread(
            target=probe,
            name=f"nvbroadcast-setup-gpu-{gpu_index}",
            daemon=True,
        ).start()

    def _finish_capability_refresh(self, generation, gpu_index, caps, error=""):
        """Apply a capability result only while its GPU selection is current."""
        if (
            self._capability_refresh_closed
            or generation != self._capability_refresh_generation
            or gpu_index != self._selected_gpu
        ):
            return False

        self._capability_refresh_in_flight = False
        self._update_action_sensitivity()
        if error or caps is None:
            self._status_label.set_text(
                error or f"Could not check GPU {gpu_index} capabilities."
            )
            return False

        self._apply_capabilities(gpu_index, caps)
        self._status_label.set_text("")
        return False

    def _on_close_requested(self, *_args):
        """Invalidate workers without committing the tentative GPU choice."""
        if self._install_in_flight:
            self._status_label.set_text(
                "Wait for the optional runtime installation to finish."
            )
            return True
        self._capability_refresh_closed = True
        self._capability_refresh_generation += 1
        self._capability_refresh_in_flight = False
        installer = self._app.dependency_installer
        for handler_id in getattr(self, "_installer_signal_ids", ()):
            try:
                installer.disconnect(handler_id)
            except (TypeError, ValueError):
                pass
        self._installer_signal_ids = ()
        return False

    def _on_mode_toggled(self, btn, mode_key):
        if btn.get_active():
            self._selected_mode_key = mode_key

    def _on_start(self, btn):
        if self._capability_refresh_in_flight:
            self._status_label.set_text("Wait for the GPU capability check to finish.")
            return
        if self._caps_gpu != self._selected_gpu:
            self._request_capability_refresh(self._selected_gpu)
            return
        mode = next(m for m in SETUP_MODES if m["key"] == self._selected_mode_key)

        # If a CUDA mode is selected but the full runtime is missing, install it first.
        if mode["needs_cupy"] and not self._caps["has_cupy"]:
            self._install_key = "cupy"
            block_reason = self._app.dependency_installer.install_block_reason_for_gpu(
                self._install_key,
                self._selected_gpu,
            )
            if block_reason:
                self._status_label.set_text(block_reason)
                return
            self._prompt_install(
                "Install CUDA mode runtime?",
                "This mode needs CUDA compositing and ONNX GPU inference packages. The download runs in the background and you can keep using other parts of the app.",
            )
            return

        self._finish(mode)

    def _on_skip(self, _btn):
        if self._capability_refresh_in_flight:
            self._status_label.set_text("Wait for the GPU capability check to finish.")
            return
        if self._caps_gpu != self._selected_gpu:
            self._request_capability_refresh(self._selected_gpu)
            return
        fallback_key = self._caps["recommended_mode"]
        if fallback_key == "gpu_cuda_best" and not self._caps["has_cupy"]:
            fallback_key = "gpu_balanced" if self._caps["has_gl_compositor"] else "cpu_quality"
        mode = next(m for m in SETUP_MODES if m["key"] == fallback_key)
        self._status_label.set_text("Setup skipped. You can change modes later from the app.")
        self._finish(mode)

    def _prompt_install(self, title: str, reason: str):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.NONE,
            text=title,
            secondary_text=reason,
        )
        dialog.add_button("Skip", Gtk.ResponseType.CANCEL)
        dialog.add_button("Install", Gtk.ResponseType.OK)
        dialog.connect("response", self._on_install_prompt_response)
        dialog.present()

    def _on_install_prompt_response(self, dialog, response):
        dialog.destroy()
        if self._capability_refresh_closed:
            return
        if response != Gtk.ResponseType.OK:
            self._status_label.set_text("Optional runtime install skipped.")
            return
        self._install_in_flight = True
        self._update_action_sensitivity()
        if not self._app.dependency_installer.start_install(
            self._install_key,
            gpu_index=self._selected_gpu,
        ):
            self._install_in_flight = False
            self._update_action_sensitivity()
            reason = self._app.dependency_installer.install_block_reason_for_gpu(
                self._install_key,
                self._selected_gpu,
            )
            self._status_label.set_text(reason or "Another optional runtime install is already running.")

    def _on_install_started(self, _installer, key: str, text: str):
        if (
            getattr(self, "_capability_refresh_closed", False)
            or key != self._install_key
        ):
            return
        self._status_label.set_text(text)

    def _on_install_progress(self, _installer, key: str, text: str, _fraction: float):
        if (
            getattr(self, "_capability_refresh_closed", False)
            or key != self._install_key
        ):
            return
        self._status_label.set_text(text)

    def _on_install_completed(self, _installer, key: str, success: bool, text: str):
        if (
            getattr(self, "_capability_refresh_closed", False)
            or key != self._install_key
        ):
            return
        self._install_in_flight = False
        self._update_action_sensitivity()
        self._status_label.set_text(text)
        if success:
            if _installer.restart_pending(key):
                self._caps["has_cupy"] = False
                return
            self._caps["has_cupy"] = True
            mode = next(m for m in SETUP_MODES if m["key"] == self._selected_mode_key)
            self._finish(mode)

    def _finish(self, mode):
        self.emit(
            "setup-complete",
            mode["profile"],
            self._selected_gpu,
            mode["compositing"],
        )
        self.close()

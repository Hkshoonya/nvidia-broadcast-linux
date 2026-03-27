# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""First-run setup wizard — auto-detects system and configures optimally."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GObject, GLib

from nvbroadcast.core.config import detect_system_capabilities
from nvbroadcast.core.gpu import detect_gpus


# Unified modes: each combines compositing + performance
SETUP_MODES = [
    {
        "key": "gpu_cuda_best",
        "label": "CUDA GPU - Maximum Quality",
        "description": "CuPy CUDA compositing. All processing on GPU. Near-zero CPU.",
        "compositing": "cupy",
        "profile": "max_quality",
        "needs_cupy": True,
        "needs_gl": False,
        "min_vram": 4096,
    },
    {
        "key": "gpu_quality",
        "label": "GPU OpenGL - Best Quality",
        "description": "GStreamer GL compositing. 30fps, every frame. Very low CPU.",
        "compositing": "gstreamer_gl",
        "profile": "max_quality",
        "needs_cupy": False,
        "needs_gl": True,
        "min_vram": 2048,
    },
    {
        "key": "gpu_balanced",
        "label": "GPU OpenGL - Balanced",
        "description": "GStreamer GL compositing. 20fps effects. Best balance.",
        "compositing": "gstreamer_gl",
        "profile": "balanced",
        "needs_cupy": False,
        "needs_gl": True,
        "min_vram": 2048,
    },
    {
        "key": "cpu_quality",
        "label": "CPU - High Quality",
        "description": "CPU compositing. 30fps. Uses ~250% CPU.",
        "compositing": "cpu",
        "profile": "max_quality",
        "needs_cupy": False,
        "needs_gl": False,
        "min_vram": 0,
    },
    {
        "key": "cpu_light",
        "label": "CPU - Light",
        "description": "CPU compositing. 15fps. Uses ~60% CPU.",
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
        self._caps = detect_system_capabilities()
        self._selected_gpu = 0
        self._selected_mode_key = self._caps["recommended_mode"]
        self._install_key = ""

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
                "GPU modes give the best quality, CPU modes are the safest fallback, "
                "and some premium paths download extra runtimes on demand."
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
        sys_info = [
            f"CPU: {c['cpu_cores']} cores",
            f"GPU: {c['gpu_name']} ({c['gpu_vram_mb']} MB)" if c['has_nvidia'] else "GPU: None detected",
        ]
        features = []
        if c["has_gl_compositor"]:
            features.append("OpenGL compositor")
        if c["has_cupy"]:
            features.append("CuPy CUDA")
        if features:
            sys_info.append(f"Available: {', '.join(features)}")

        for line in sys_info:
            lbl = Gtk.Label(label=line)
            lbl.set_xalign(0)
            sys_box.append(lbl)

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
            for g in self._gpus:
                btn = Gtk.CheckButton(
                    label=f"GPU {g.index}: {g.name} ({g.memory_total_mb} MB)"
                )
                if self._gpu_group is None:
                    self._gpu_group = btn
                    btn.set_active(True)
                else:
                    btn.set_group(self._gpu_group)
                btn.connect("toggled", self._on_gpu_toggled, g.index)
                gpu_box.append(btn)
            gpu_frame.set_child(gpu_box)
            content.append(gpu_frame)

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
            # Determine availability
            available = True
            reason = ""
            if mode["needs_gl"] and not c["has_gl_compositor"]:
                available = False
                reason = " [needs GStreamer GL plugins]"
            if mode["needs_cupy"] and not c["has_cupy"]:
                reason = " [will install CuPy ~800MB]"
                # Still available — we'll install it
            if mode["min_vram"] > c["gpu_vram_mb"] and c["has_nvidia"]:
                available = False
                reason = f" [needs {mode['min_vram']}MB VRAM]"
            if not c["has_nvidia"] and (mode["needs_gl"] or mode["needs_cupy"]):
                available = False
                reason = " [needs NVIDIA GPU]"

            is_recommended = mode["key"] == self._caps["recommended_mode"]
            label = mode["label"]
            if is_recommended:
                label += "  ★ recommended"

            btn = Gtk.CheckButton(label=f"{label}{reason}")
            btn.set_sensitive(available or mode["needs_cupy"])  # CuPy modes always selectable

            desc = Gtk.Label(label=f"  {mode['description']}")
            desc.set_xalign(0)
            desc.add_css_class("dim-label")
            desc.set_margin_start(24)
            desc.set_wrap(True)

            if self._mode_group is None:
                self._mode_group = btn
            else:
                btn.set_group(self._mode_group)

            if is_recommended:
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
        self._app.dependency_installer.connect("job-started", self._on_install_started)
        self._app.dependency_installer.connect("job-progress", self._on_install_progress)
        self._app.dependency_installer.connect("job-completed", self._on_install_completed)

    def _on_gpu_toggled(self, btn, gpu_index):
        if btn.get_active():
            self._selected_gpu = gpu_index

    def _on_mode_toggled(self, btn, mode_key):
        if btn.get_active():
            self._selected_mode_key = mode_key

    def _on_start(self, btn):
        mode = next(m for m in SETUP_MODES if m["key"] == self._selected_mode_key)

        # If CuPy mode selected but not installed, install it first
        if mode["needs_cupy"] and not self._caps["has_cupy"]:
            self._install_key = "cupy"
            self._prompt_install(
                "Install GPU compositing files?",
                "This mode needs the CUDA compositing runtime. The download runs in the background and you can keep using other parts of the app.",
            )
            return

        self._finish(mode)

    def _on_skip(self, _btn):
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
        if response != Gtk.ResponseType.OK:
            self._status_label.set_text("Optional runtime install skipped.")
            return
        self._start_btn.set_sensitive(False)
        self._skip_btn.set_sensitive(False)
        self._app.dependency_installer.start_install(self._install_key)

    def _on_install_started(self, _installer, key: str, text: str):
        if key != self._install_key:
            return
        self._status_label.set_text(text)

    def _on_install_progress(self, _installer, key: str, text: str, _fraction: float):
        if key != self._install_key:
            return
        self._status_label.set_text(text)

    def _on_install_completed(self, _installer, key: str, success: bool, text: str):
        if key != self._install_key:
            return
        self._start_btn.set_sensitive(True)
        self._skip_btn.set_sensitive(True)
        self._status_label.set_text(text)
        if success:
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

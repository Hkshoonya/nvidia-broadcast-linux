# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Effect control widgets - toggles, sliders, mode selectors, and background picker."""

import os
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GObject, Gio


class EffectToggle(Adw.ActionRow):
    """A toggle switch for enabling/disabling an effect."""

    __gsignals__ = {
        "toggled": (GObject.SignalFlags.RUN_FIRST, None, (bool,)),
    }

    def __init__(self, title: str, subtitle: str = "", available: bool = True):
        super().__init__(title=title, subtitle=subtitle)

        self._switch = Gtk.Switch()
        self._switch.set_valign(Gtk.Align.CENTER)
        self._switch.set_sensitive(available)
        self._switch.connect("notify::active", self._on_toggled)
        self.add_suffix(self._switch)
        self.set_activatable_widget(self._switch)

        if not available:
            self.set_subtitle("Not available")

    @property
    def active(self) -> bool:
        return self._switch.get_active()

    @active.setter
    def active(self, value: bool):
        self._switch.set_active(value)

    def set_available(self, available: bool, subtitle: str = ""):
        """Update availability state."""
        self._switch.set_sensitive(available)
        if subtitle:
            self.set_subtitle(subtitle)

    def _on_toggled(self, switch, _pspec):
        self.emit("toggled", switch.get_active())


class EffectSlider(Gtk.Box):
    """A labeled slider for effect intensity."""

    __gsignals__ = {
        "value-changed": (GObject.SignalFlags.RUN_FIRST, None, (float,)),
    }

    def __init__(self, label: str, value: float = 0.7, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.set_margin_start(16)
        self.set_margin_end(16)

        lbl = Gtk.Label(label=label)
        lbl.set_xalign(0)
        lbl.set_size_request(80, -1)
        self.append(lbl)

        self._scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, min_val, max_val, 0.05
        )
        self._scale.set_value(value)
        self._scale.set_hexpand(True)
        self._scale.set_draw_value(True)
        self._scale.set_value_pos(Gtk.PositionType.RIGHT)
        self._scale.connect("value-changed", self._on_changed)
        self.append(self._scale)

    @property
    def value(self) -> float:
        return self._scale.get_value()

    def _on_changed(self, scale):
        self.emit("value-changed", scale.get_value())


class BackgroundModeSelector(Gtk.Box):
    """Selector for background mode: blur or replace."""

    __gsignals__ = {
        "mode-changed": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
    }

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.set_margin_start(16)
        self.set_margin_end(16)

        lbl = Gtk.Label(label="Mode")
        lbl.set_xalign(0)
        lbl.set_size_request(80, -1)
        self.append(lbl)

        self._dropdown = Gtk.DropDown.new_from_strings(["Blur", "Replace with Image", "Remove (Green Screen)"])
        self._dropdown.set_hexpand(True)
        self._dropdown.connect("notify::selected", self._on_changed)
        self.append(self._dropdown)

    @property
    def mode(self) -> str:
        idx = self._dropdown.get_selected()
        if idx == 0:
            return "blur"
        elif idx == 1:
            return "replace"
        else:
            return "remove"

    def _on_changed(self, dropdown, _pspec):
        self.emit("mode-changed", self.mode)


class BackgroundImagePicker(Gtk.Box):
    """File chooser for selecting a custom background image."""

    __gsignals__ = {
        "image-selected": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
    }

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.set_margin_start(16)
        self.set_margin_end(16)

        lbl = Gtk.Label(label="Image")
        lbl.set_xalign(0)
        lbl.set_size_request(80, -1)
        self.append(lbl)

        self._path_label = Gtk.Label(label="None selected")
        self._path_label.set_hexpand(True)
        self._path_label.set_xalign(0)
        self._path_label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        self._path_label.set_opacity(0.7)
        self.append(self._path_label)

        btn = Gtk.Button(label="Browse")
        btn.connect("clicked", self._on_browse)
        self.append(btn)

        self._selected_path = ""

    @property
    def selected_path(self) -> str:
        return self._selected_path

    def _on_browse(self, button):
        """Open native file chooser dialog for image selection."""
        window = self.get_root()

        dialog = Gtk.FileDialog()
        dialog.set_title("Select Background Image")

        # Image file filter
        img_filter = Gtk.FileFilter()
        img_filter.set_name("Images")
        img_filter.add_mime_type("image/png")
        img_filter.add_mime_type("image/jpeg")
        img_filter.add_mime_type("image/webp")
        img_filter.add_mime_type("image/bmp")
        img_filter.add_mime_type("image/tiff")
        img_filter.add_mime_type("image/svg+xml")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(img_filter)
        all_filter = Gtk.FileFilter()
        all_filter.set_name("All Files")
        all_filter.add_pattern("*")
        filters.append(all_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(img_filter)

        # Start in previous directory or Pictures
        if self._selected_path:
            start_dir = Gio.File.new_for_path(os.path.dirname(self._selected_path))
        else:
            pictures = os.path.expanduser("~/Pictures")
            start_dir = Gio.File.new_for_path(pictures if os.path.isdir(pictures) else os.path.expanduser("~"))
        dialog.set_initial_folder(start_dir)

        dialog.open(window, None, self._on_file_chosen)

    def _on_file_chosen(self, dialog, result):
        try:
            gfile = dialog.open_finish(result)
            if gfile:
                path = gfile.get_path()
                self._selected_path = path
                self._path_label.set_text(os.path.basename(path))
                self._path_label.set_opacity(1.0)
                self.emit("image-selected", path)
        except Exception:
            pass  # User cancelled

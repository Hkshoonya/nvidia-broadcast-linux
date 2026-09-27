# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Camera and audio device selection widgets."""

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GObject, Pango


class DeviceSelector(Gtk.Box):
    """Dropdown selector for camera/mic/speaker devices."""

    __gsignals__ = {
        "device-changed": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
    }

    def __init__(self, label: str, devices: list[dict[str, str]] | None = None):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        self._label = Gtk.Label(label=label)
        self._label.set_xalign(0)
        self._label.set_hexpand(False)
        self.append(self._label)

        self._dropdown = Gtk.DropDown()
        self._dropdown.set_hexpand(True)
        # A closed selector is allowed to shrink to the row width. The popup
        # still exposes the complete name, while the selected item and its
        # tooltip carry the readable/ellipsized form. Without an explicit
        # zero width request, newer GTK runtimes can use a long model item to
        # widen the entire compact window past its tested minimum.
        self._dropdown.set_size_request(0, -1)
        # The selected item must not give the whole window the width of a
        # verbose device or format name. Keep the full names in the popup and
        # tooltip, while allowing the closed selector to ellipsize.
        selected_factory = Gtk.SignalListItemFactory()
        selected_factory.connect("setup", self._setup_selected_item)
        selected_factory.connect("bind", self._bind_selected_item)
        self._dropdown.set_factory(selected_factory)
        list_factory = Gtk.SignalListItemFactory()
        list_factory.connect("setup", self._setup_list_item)
        list_factory.connect("bind", self._bind_list_item)
        self._dropdown.set_list_factory(list_factory)
        self.append(self._dropdown)

        self._devices: list[dict[str, str]] = []
        self._handler_id = None
        if devices:
            self.set_devices(devices)

    def set_devices(self, devices: list[dict[str, str]]) -> bool:
        """Set available devices, preserving selection when possible.

        Returns whether the backing model changed. Reusing an identical model
        avoids closing an open dropdown or needlessly resetting its selection
        during periodic device discovery.
        """
        if devices == self._devices:
            return False

        selected_device = self.get_selected_device()
        self._devices = devices
        names = [d["name"] for d in devices]
        # Block handler during model change to prevent spurious signals
        if self._handler_id:
            self._dropdown.handler_block(self._handler_id)
        string_list = Gtk.StringList.new(names)
        self._dropdown.set_model(string_list)
        if self._devices:
            selected_index = next(
                (
                    index
                    for index, device in enumerate(self._devices)
                    if device["device"] == selected_device
                ),
                0,
            )
            self._dropdown.set_selected(selected_index)
        self._update_tooltip()
        if self._handler_id:
            self._dropdown.handler_unblock(self._handler_id)
        elif self._devices:
            self._handler_id = self._dropdown.connect(
                "notify::selected", self._on_selection_changed
            )
        return True

    def get_selected_device(self) -> str:
        """Return the device path of the selected device."""
        idx = self._dropdown.get_selected()
        if 0 <= idx < len(self._devices):
            return self._devices[idx]["device"]
        return ""

    def set_selected_index(self, index: int):
        """Programmatically select a device by index without firing callbacks."""
        if 0 <= index < len(self._devices):
            if self._handler_id:
                self._dropdown.handler_block(self._handler_id)
            self._dropdown.set_selected(index)
            self._update_tooltip()
            if self._handler_id:
                self._dropdown.handler_unblock(self._handler_id)

    def set_selected_device(self, device: str) -> bool:
        """Select a device by its stable identifier without firing callbacks."""
        for index, candidate in enumerate(self._devices):
            if candidate["device"] == device:
                self.set_selected_index(index)
                return True
        return False

    def _on_selection_changed(self, dropdown, _pspec):
        self._update_tooltip()
        device = self.get_selected_device()
        if device:
            self.emit("device-changed", device)

    @staticmethod
    def _setup_selected_item(_factory, item):
        label = Gtk.Label(xalign=0)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        label.set_width_chars(6)
        item.set_child(label)

    @staticmethod
    def _bind_selected_item(_factory, item):
        item.get_child().set_text(item.get_item().get_string())

    @staticmethod
    def _setup_list_item(_factory, item):
        label = Gtk.Label(xalign=0)
        label.set_wrap(True)
        label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        label.set_max_width_chars(32)
        item.set_child(label)

    @staticmethod
    def _bind_list_item(_factory, item):
        name = item.get_item().get_string()
        item.get_child().set_text(name)
        item.get_child().set_tooltip_text(name)

    def _update_tooltip(self):
        index = self._dropdown.get_selected()
        self._dropdown.set_tooltip_text(
            self._devices[index]["name"] if 0 <= index < len(self._devices) else None
        )

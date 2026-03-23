# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Camera and audio device selection widgets."""

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GObject


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
        self.append(self._dropdown)

        self._devices: list[dict[str, str]] = []
        self._connected = False
        if devices:
            self.set_devices(devices)

    def set_devices(self, devices: list[dict[str, str]]):
        """Set available devices. Each dict has 'name' and 'device' keys."""
        self._devices = devices
        names = [d["name"] for d in devices]
        string_list = Gtk.StringList.new(names)
        self._dropdown.set_model(string_list)
        # Connect signal only ONCE (set_devices may be called multiple times)
        if not self._connected:
            self._dropdown.connect("notify::selected", self._on_selection_changed)
            self._connected = True

    def get_selected_device(self) -> str:
        """Return the device path of the selected device."""
        idx = self._dropdown.get_selected()
        if 0 <= idx < len(self._devices):
            return self._devices[idx]["device"]
        return ""

    def set_selected_index(self, index: int):
        """Programmatically select a device by index."""
        if 0 <= index < len(self._devices):
            self._dropdown.set_selected(index)

    def _on_selection_changed(self, dropdown, _pspec):
        device = self.get_selected_device()
        if device:
            self.emit("device-changed", device)

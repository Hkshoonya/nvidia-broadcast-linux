# NV Broadcast - Unofficial NVIDIA Broadcast for Linux and other OS
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""System tray icon for NV Broadcast.

Shows a persistent tray icon when the app is minimized.
Supports AyatanaAppIndicator3 (Ubuntu/GNOME) with fallback to
basic GtkStatusIcon where available.
"""

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

_ICON_PATH = str(Path(__file__).parent.parent.parent.parent / "data" / "icons" / "com.doczeus.NVBroadcast.svg")


class TrayIcon:
    """System tray icon with menu: Show/Hide, Broadcast On/Off, Quit."""

    def __init__(self, app):
        self._app = app
        self._indicator = None
        self._active = False
        self._setup_indicator()

    def _setup_indicator(self):
        """Try AyatanaAppIndicator3, then AppIndicator3."""
        try:
            gi.require_version('AyatanaAppIndicator3', '0.1')
            from gi.repository import AyatanaAppIndicator3 as AI
            self._indicator = AI.Indicator.new(
                "nvbroadcast",
                _ICON_PATH,
                AI.IndicatorCategory.APPLICATION_STATUS,
            )
            self._indicator.set_status(AI.IndicatorStatus.ACTIVE)
            self._indicator.set_menu(self._build_menu_gtk3())
            self._active = True
            return
        except Exception:
            pass

        try:
            gi.require_version('AppIndicator3', '0.1')
            from gi.repository import AppIndicator3 as AI
            self._indicator = AI.Indicator.new(
                "nvbroadcast",
                _ICON_PATH,
                AI.IndicatorCategory.APPLICATION_STATUS,
            )
            self._indicator.set_status(AI.IndicatorStatus.ACTIVE)
            self._indicator.set_menu(self._build_menu_gtk3())
            self._active = True
            return
        except Exception:
            pass

    def _build_menu_gtk3(self):
        """Build GTK3 menu for the tray indicator."""
        # AppIndicator uses GTK3 menus, not GTK4
        import gi as _gi
        _gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk as Gtk3

        menu = Gtk3.Menu()

        # Show/Hide
        show_item = Gtk3.MenuItem(label="Show NV Broadcast")
        show_item.connect("activate", self._on_show)
        menu.append(show_item)

        # Broadcast toggle
        self._broadcast_item = Gtk3.MenuItem(label="Start Broadcast")
        self._broadcast_item.connect("activate", self._on_broadcast_toggle)
        menu.append(self._broadcast_item)

        menu.append(Gtk3.SeparatorMenuItem())

        # Status
        self._status_item = Gtk3.MenuItem(label="Status: Idle")
        self._status_item.set_sensitive(False)
        menu.append(self._status_item)

        menu.append(Gtk3.SeparatorMenuItem())

        # Quit
        quit_item = Gtk3.MenuItem(label="Quit")
        quit_item.connect("activate", self._on_quit)
        menu.append(quit_item)

        menu.show_all()
        return menu

    def _on_show(self, item):
        win = self._app._window
        if win:
            win.set_visible(True)
            win.present()

    def _on_broadcast_toggle(self, item):
        if self._app._streaming:
            self._app.stop_pipeline()
            if self._app._window:
                self._app._window._streaming = False
                self._app._window._stream_btn.set_label("Start Broadcast")
        else:
            cam = self._app.config.video.camera_device
            fmt = self._app.config.video.output_format
            self._app.start_pipeline(cam, fmt)
            if self._app._window:
                self._app._window._streaming = True
                self._app._window._stream_btn.set_label("Stop Broadcast")

    def _on_quit(self, item):
        self._app.quit()

    def update_status(self, streaming: bool, status_text: str = ""):
        """Update tray menu to reflect current state."""
        if not self._active:
            return
        if hasattr(self, '_broadcast_item'):
            GLib.idle_add(
                self._broadcast_item.set_label,
                "Stop Broadcast" if streaming else "Start Broadcast"
            )
        if hasattr(self, '_status_item') and status_text:
            GLib.idle_add(self._status_item.set_label, f"Status: {status_text}")

    @property
    def available(self) -> bool:
        return self._active

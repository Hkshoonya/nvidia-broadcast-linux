# NV Broadcast - Unofficial NVIDIA Broadcast for Linux and other OS
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Native StatusNotifierItem tray icon (SNI over D-Bus).

Implements org.kde.StatusNotifierItem plus a minimal com.canonical.dbusmenu
directly with Gio — no GTK3, no AppIndicator libraries. This is the tray
protocol modern desktops speak (KDE, Hyprland/waybar/quickshell, GNOME with
extension), and unlike the legacy AppIndicator bridge it is safe inside a
GTK4 process.

Left click (Activate) toggles the main window; the context menu offers
Show/Hide, Start/Stop Broadcast, a status line, and Quit.
"""

import struct

import gi

gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gio, GLib

_SNI_XML = """
<node>
  <interface name="org.kde.StatusNotifierItem">
    <property name="Category" type="s" access="read"/>
    <property name="Id" type="s" access="read"/>
    <property name="Title" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="IconName" type="s" access="read"/>
    <property name="IconPixmap" type="a(iiay)" access="read"/>
    <property name="ToolTip" type="(sa(iiay)ss)" access="read"/>
    <property name="ItemIsMenu" type="b" access="read"/>
    <property name="Menu" type="o" access="read"/>
    <method name="Activate">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="SecondaryActivate">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="ContextMenu">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="Scroll">
      <arg name="delta" type="i" direction="in"/>
      <arg name="orientation" type="s" direction="in"/>
    </method>
    <signal name="NewIcon"/>
    <signal name="NewToolTip"/>
    <signal name="NewStatus">
      <arg name="status" type="s"/>
    </signal>
  </interface>
</node>
"""

_MENU_XML = """
<node>
  <interface name="com.canonical.dbusmenu">
    <property name="Version" type="u" access="read"/>
    <property name="TextDirection" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="IconThemePath" type="as" access="read"/>
    <method name="GetLayout">
      <arg name="parentId" type="i" direction="in"/>
      <arg name="recursionDepth" type="i" direction="in"/>
      <arg name="propertyNames" type="as" direction="in"/>
      <arg name="revision" type="u" direction="out"/>
      <arg name="layout" type="(ia{sv}av)" direction="out"/>
    </method>
    <method name="GetGroupProperties">
      <arg name="ids" type="ai" direction="in"/>
      <arg name="propertyNames" type="as" direction="in"/>
      <arg name="properties" type="a(ia{sv})" direction="out"/>
    </method>
    <method name="GetProperty">
      <arg name="id" type="i" direction="in"/>
      <arg name="name" type="s" direction="in"/>
      <arg name="value" type="v" direction="out"/>
    </method>
    <method name="Event">
      <arg name="id" type="i" direction="in"/>
      <arg name="eventId" type="s" direction="in"/>
      <arg name="data" type="v" direction="in"/>
      <arg name="timestamp" type="u" direction="in"/>
    </method>
    <method name="EventGroup">
      <arg name="events" type="a(isvu)" direction="in"/>
      <arg name="idErrors" type="ai" direction="out"/>
    </method>
    <method name="AboutToShow">
      <arg name="id" type="i" direction="in"/>
      <arg name="needUpdate" type="b" direction="out"/>
    </method>
    <method name="AboutToShowGroup">
      <arg name="ids" type="ai" direction="in"/>
      <arg name="updatesNeeded" type="ai" direction="out"/>
      <arg name="idErrors" type="ai" direction="out"/>
    </method>
    <signal name="LayoutUpdated">
      <arg name="revision" type="u"/>
      <arg name="parent" type="i"/>
    </signal>
    <signal name="ItemsPropertiesUpdated">
      <arg name="updatedProps" type="a(ia{sv})"/>
      <arg name="removedProps" type="a(ias)"/>
    </signal>
  </interface>
</node>
"""

_MENU_PATH = "/MenuBar"
_ITEM_PATH = "/StatusNotifierItem"

# Menu item ids
_ID_SHOW = 1
_ID_BROADCAST = 2
_ID_STATUS = 3
_ID_QUIT = 4


def _load_icon_pixmaps(icon_path, sizes=(22, 24, 32, 48)) -> list:
    """Rasterize the app icon into SNI ARGB32 (network byte order) pixmaps."""
    pixmaps = []
    try:
        from gi.repository import GdkPixbuf
        for size in sizes:
            pb = GdkPixbuf.Pixbuf.new_from_file_at_size(str(icon_path), size, size)
            if pb.get_colorspace() != GdkPixbuf.Colorspace.RGB or pb.get_bits_per_sample() != 8:
                continue
            has_alpha = pb.get_has_alpha()
            width, height = pb.get_width(), pb.get_height()
            rowstride = pb.get_rowstride()
            data = pb.get_pixels()
            argb = bytearray(width * height * 4)
            channels = 4 if has_alpha else 3
            for y in range(height):
                row = y * rowstride
                for x in range(width):
                    o = row + x * channels
                    r, g, b = data[o], data[o + 1], data[o + 2]
                    a = data[o + 3] if has_alpha else 255
                    struct.pack_into(">BBBB", argb, (y * width + x) * 4, a, r, g, b)
            pixmaps.append((width, height, bytes(argb)))
    except Exception:
        pass
    return pixmaps


class SniTray:
    """StatusNotifierItem tray icon with a native D-Bus menu."""

    def __init__(self, app):
        self._app = app
        self._active = False
        self._conn = None
        self._reg_ids = []
        self._watch_id = 0
        self._revision = 1
        self._streaming = False
        self._status_text = "Idle"
        self._pixmaps = []
        try:
            self._setup()
        except Exception as e:
            print(f"[NV Broadcast] SNI tray unavailable: {e}", flush=True)

    # ─── Setup ───────────────────────────────────────────────────────────

    def _setup(self):
        from nvbroadcast.core.resources import find_app_icon, find_app_icon_png
        # PNG first: GdkPixbuf's PNG loader is always built in, while the
        # SVG loader is often absent in sandboxed runtimes, which would
        # leave the pixmap list empty and the tray icon blank.
        for icon_path in (find_app_icon_png(), find_app_icon()):
            if icon_path is not None:
                self._pixmaps = _load_icon_pixmaps(icon_path)
                if self._pixmaps:
                    break

        self._conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        item_node = Gio.DBusNodeInfo.new_for_xml(_SNI_XML)
        menu_node = Gio.DBusNodeInfo.new_for_xml(_MENU_XML)

        self._reg_ids.append(self._conn.register_object(
            _ITEM_PATH, item_node.interfaces[0],
            self._on_item_call, self._on_item_get, None))
        self._reg_ids.append(self._conn.register_object(
            _MENU_PATH, menu_node.interfaces[0],
            self._on_menu_call, self._on_menu_get, None))

        # Register with the watcher immediately if one is running, and
        # re-register whenever a watcher (re)appears — tray hosts restart
        # with the shell.
        self._on_watcher_appeared(self._conn, "org.kde.StatusNotifierWatcher", "")
        self._watch_id = Gio.bus_watch_name_on_connection(
            self._conn, "org.kde.StatusNotifierWatcher",
            Gio.BusNameWatcherFlags.NONE,
            self._on_watcher_appeared, None)

    def _on_watcher_appeared(self, conn, name, owner):
        # Always (re-)register — tray hosts restart with the shell — but
        # only announce the first success.
        try:
            conn.call_sync(
                "org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher",
                "org.kde.StatusNotifierWatcher", "RegisterStatusNotifierItem",
                GLib.Variant("(s)", (conn.get_unique_name(),)),
                None, Gio.DBusCallFlags.NONE, 2000, None)
            if not self._active:
                print("[NV Broadcast] SNI tray icon registered", flush=True)
            self._active = True
        except Exception as e:
            if owner:  # A watcher exists but rejected us — worth logging
                print(f"[NV Broadcast] SNI watcher registration failed: {e}",
                      flush=True)

    @property
    def bus_ready(self) -> bool:
        """Object exported on the bus (a watcher may still appear later)."""
        return self._conn is not None and bool(self._reg_ids)

    # ─── StatusNotifierItem interface ───────────────────────────────────

    def _on_item_get(self, conn, sender, path, iface, prop):
        if prop == "Category":
            return GLib.Variant("s", "ApplicationStatus")
        if prop == "Id":
            return GLib.Variant("s", "nvbroadcast")
        if prop == "Title":
            return GLib.Variant("s", "NV Broadcast")
        if prop == "Status":
            return GLib.Variant("s", "Active")
        if prop == "IconName":
            return GLib.Variant("s", "" if self._pixmaps else "camera-video")
        if prop == "IconPixmap":
            return GLib.Variant("a(iiay)", self._pixmaps)
        if prop == "ToolTip":
            return GLib.Variant("(sa(iiay)ss)",
                                ("", [], "NV Broadcast", self._status_text))
        if prop == "ItemIsMenu":
            return GLib.Variant("b", False)
        if prop == "Menu":
            return GLib.Variant("o", _MENU_PATH)
        return None

    def _on_item_call(self, conn, sender, path, iface, method, params, invocation):
        if method == "Activate":
            self._toggle_window()
            invocation.return_value(None)
        elif method == "SecondaryActivate":
            # Middle click: quick broadcast toggle without opening the menu.
            GLib.idle_add(self._on_menu_clicked, _ID_BROADCAST)
            invocation.return_value(None)
        elif method in ("ContextMenu", "Scroll"):
            # Right click is rendered by the host from the Menu property.
            invocation.return_value(None)
        else:
            invocation.return_dbus_error(
                "org.freedesktop.DBus.Error.UnknownMethod", method)

    # ─── com.canonical.dbusmenu interface ───────────────────────────────

    def _menu_items(self):
        win = getattr(self._app, "_window", None)
        visible = bool(win and win.get_visible())
        return [
            (_ID_SHOW, {"label": GLib.Variant("s", "Hide NV Broadcast" if visible
                                              else "Show NV Broadcast")}),
            (_ID_BROADCAST, {"label": GLib.Variant(
                "s", "Stop Broadcast" if self._streaming else "Start Broadcast")}),
            (_ID_STATUS, {"label": GLib.Variant("s", f"Status: {self._status_text}"),
                          "enabled": GLib.Variant("b", False)}),
            (_ID_QUIT, {"label": GLib.Variant("s", "Quit")}),
        ]

    def _layout_reply(self):
        """Full (u(ia{sv}av)) GetLayout reply.

        Built in one GLib.Variant call from plain Python structures. The
        'av' children slot takes a list of GLib.Variant objects directly;
        wrapping them in an extra "v" variant makes construction throw,
        the handler dies, and the host's GetLayout call times out with no
        menu ever shown.
        """
        children = [
            GLib.Variant("(ia{sv}av)", (item_id, props, []))
            for item_id, props in self._menu_items()
        ]
        root = (0, {"children-display": GLib.Variant("s", "submenu")}, children)
        return GLib.Variant("(u(ia{sv}av))", (self._revision, root))

    def _on_menu_get(self, conn, sender, path, iface, prop):
        if prop == "Version":
            return GLib.Variant("u", 3)
        if prop == "TextDirection":
            return GLib.Variant("s", "ltr")
        if prop == "Status":
            return GLib.Variant("s", "normal")
        if prop == "IconThemePath":
            return GLib.Variant("as", [])
        return None

    def _on_menu_call(self, conn, sender, path, iface, method, params, invocation):
        if method == "GetLayout":
            invocation.return_value(self._layout_reply())
        elif method == "GetGroupProperties":
            ids = set(params.unpack()[0])
            props = [(i, p) for i, p in self._menu_items() if not ids or i in ids]
            invocation.return_value(GLib.Variant("(a(ia{sv}))", (props,)))
        elif method == "GetProperty":
            invocation.return_value(GLib.Variant("(v)", (GLib.Variant("s", ""),)))
        elif method == "Event":
            item_id, event_id, _data, _ts = params.unpack()
            if event_id == "clicked":
                GLib.idle_add(self._on_menu_clicked, item_id)
            invocation.return_value(None)
        elif method == "EventGroup":
            for item_id, event_id, _data, _ts in params.unpack()[0]:
                if event_id == "clicked":
                    GLib.idle_add(self._on_menu_clicked, item_id)
            invocation.return_value(GLib.Variant("(ai)", ([],)))
        elif method == "AboutToShow":
            self._bump_revision()
            invocation.return_value(GLib.Variant("(b)", (True,)))
        elif method == "AboutToShowGroup":
            self._bump_revision()
            invocation.return_value(GLib.Variant("(aiai)", ([], [])))
        else:
            invocation.return_dbus_error(
                "org.freedesktop.DBus.Error.UnknownMethod", method)

    def _bump_revision(self):
        self._revision += 1
        try:
            self._conn.emit_signal(
                None, _MENU_PATH, "com.canonical.dbusmenu", "LayoutUpdated",
                GLib.Variant("(ui)", (self._revision, 0)))
        except Exception:
            pass

    # ─── Actions ─────────────────────────────────────────────────────────

    def _toggle_window(self):
        win = getattr(self._app, "_window", None)
        if not win:
            return
        if win.get_visible():
            win.set_visible(False)
        else:
            win.set_visible(True)
            win.present()

    def _on_menu_clicked(self, item_id):
        if item_id == _ID_SHOW:
            self._toggle_window()
        elif item_id == _ID_BROADCAST:
            app = self._app
            if app._streaming:
                app.stop_pipeline()
                if app._window:
                    app._window._streaming = False
                    app._window._stream_btn.set_label("Start Broadcast")
            else:
                cam = app.config.video.camera_device
                fmt = app.config.video.output_format
                app.start_pipeline(cam, fmt)
                if app._window:
                    app._window._streaming = True
                    app._window._stream_btn.set_label("Stop Broadcast")
        elif item_id == _ID_QUIT:
            self._app.quit()
        return False

    # ─── Public API (mirrors legacy TrayIcon) ────────────────────────────

    def update_status(self, streaming: bool, status_text: str = ""):
        self._streaming = bool(streaming)
        if status_text:
            self._status_text = status_text
        if not self._active:
            return
        self._bump_revision()
        try:
            self._conn.emit_signal(
                None, _ITEM_PATH, "org.kde.StatusNotifierItem", "NewToolTip", None)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._active

    def shutdown(self):
        if self._watch_id:
            Gio.bus_unwatch_name(self._watch_id)
            self._watch_id = 0
        if self._conn:
            for rid in self._reg_ids:
                try:
                    self._conn.unregister_object(rid)
                except Exception:
                    pass
        self._reg_ids = []
        self._active = False

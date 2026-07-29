# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Safe global effect shortcuts for supported Linux desktops."""

from __future__ import annotations

import os
import platform
import re
import secrets
import shlex
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import gi

from nvbroadcast.core.constants import APP_ID

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk


@dataclass(frozen=True)
class HotkeyAction:
    action_id: str
    config_key: str
    title: str
    description: str


HOTKEY_ACTIONS = (
    HotkeyAction(
        "toggle-background",
        "toggle_background",
        "Background",
        "Toggle background processing",
    ),
    HotkeyAction(
        "toggle-auto-frame",
        "toggle_auto_frame",
        "Auto Frame",
        "Toggle automatic framing",
    ),
    HotkeyAction(
        "toggle-eye-contact",
        "toggle_eye_contact",
        "Eye Contact",
        "Toggle eye contact correction",
    ),
    HotkeyAction(
        "toggle-mirror",
        "toggle_mirror",
        "Mirror",
        "Toggle horizontal mirroring",
    ),
    HotkeyAction(
        "toggle-mic-noise",
        "toggle_mic_noise",
        "Mic Noise Removal",
        "Toggle microphone noise removal",
    ),
)

_ACTIONS_BY_ID = {action.action_id: action for action in HOTKEY_ACTIONS}

_STRONG_MODIFIERS = (
    Gdk.ModifierType.CONTROL_MASK
    | Gdk.ModifierType.ALT_MASK
    | Gdk.ModifierType.SUPER_MASK
    | Gdk.ModifierType.META_MASK
    | Gdk.ModifierType.HYPER_MASK
)
_FUNCTION_KEY_RE = re.compile(r"^F(?:[1-9]|[12][0-9]|3[0-5])$")
_XKB_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")
_XF86_KEY_MIN = 0x1008FF00
_XF86_KEY_MAX = 0x1008FFFF

_PORTAL_BUS_NAME = "org.freedesktop.portal.Desktop"
_PORTAL_OBJECT_PATH = "/org/freedesktop/portal/desktop"
_PORTAL_INTERFACE = "org.freedesktop.portal.GlobalShortcuts"
_REQUEST_INTERFACE = "org.freedesktop.portal.Request"
_SESSION_INTERFACE = "org.freedesktop.portal.Session"

_GNOME_ROOT_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
_GNOME_SHORTCUT_SCHEMA = (
    "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
)
_GNOME_SHORTCUT_ROOT = (
    "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/"
)

ActionCallback = Callable[[str], None]
StateCallback = Callable[[bool, str, dict[str, str]], None]


class HotkeyValidationError(ValueError):
    """A shortcut is unsafe, invalid, or duplicates another action."""


def _parse_accelerator(value: str) -> tuple[int, Gdk.ModifierType] | None:
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    if not all(char.isprintable() for char in value):
        return None
    parsed, keyval, modifiers = Gtk.accelerator_parse(value)
    if not parsed or not keyval:
        return None
    modifiers &= Gtk.accelerator_get_default_mod_mask()
    if not Gtk.accelerator_valid(keyval, modifiers):
        return None
    return keyval, modifiers


def normalize_accelerator(value: str) -> str:
    """Return a canonical safe accelerator, or an empty string if invalid."""
    parsed = _parse_accelerator(value)
    if parsed is None:
        return ""
    keyval, modifiers = parsed
    key_name = Gdk.keyval_name(keyval) or ""
    standalone_safe = bool(
        _FUNCTION_KEY_RE.fullmatch(key_name)
        or _XF86_KEY_MIN <= keyval <= _XF86_KEY_MAX
    )
    if not modifiers & _STRONG_MODIFIERS and not standalone_safe:
        return ""
    return Gtk.accelerator_name(keyval, modifiers) or ""


def accelerator_label(value: str) -> str:
    """Return a localized display label for a validated accelerator."""
    normalized = normalize_accelerator(value)
    parsed = _parse_accelerator(normalized)
    if parsed is None:
        return ""
    return Gtk.accelerator_get_label(*parsed) or normalized


def accelerator_to_portal_trigger(value: str) -> str:
    """Convert a GTK accelerator to the freedesktop shortcuts syntax."""
    normalized = normalize_accelerator(value)
    parsed = _parse_accelerator(normalized)
    if parsed is None:
        return ""
    keyval, modifiers = parsed
    parts: list[str] = []
    if modifiers & Gdk.ModifierType.CONTROL_MASK:
        parts.append("CTRL")
    if modifiers & Gdk.ModifierType.ALT_MASK:
        parts.append("ALT")
    if modifiers & Gdk.ModifierType.SHIFT_MASK:
        parts.append("SHIFT")
    if modifiers & (
        Gdk.ModifierType.SUPER_MASK
        | Gdk.ModifierType.META_MASK
        | Gdk.ModifierType.HYPER_MASK
    ):
        parts.append("LOGO")

    key_name = Gdk.keyval_name(keyval) or ""
    if _XF86_KEY_MIN <= keyval <= _XF86_KEY_MAX and not key_name.startswith("XF86"):
        key_name = f"XF86{key_name}"
    if not _XKB_KEY_RE.fullmatch(key_name):
        return ""
    parts.append(key_name)
    return "+".join(parts)


def normalize_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    """Validate all known bindings and reject duplicate accelerators."""
    normalized: dict[str, str] = {}
    owners: dict[str, HotkeyAction] = {}
    for action in HOTKEY_ACTIONS:
        raw = bindings.get(action.config_key, "")
        if not raw:
            normalized[action.config_key] = ""
            continue
        accelerator = normalize_accelerator(raw)
        if not accelerator:
            raise HotkeyValidationError(
                f"{action.title} needs Control, Alt, or Super, "
                "unless it uses a function or media key."
            )
        duplicate_key = accelerator.casefold()
        previous = owners.get(duplicate_key)
        if previous is not None:
            raise HotkeyValidationError(
                f"{action.title} and {previous.title} cannot use the same shortcut."
            )
        owners[duplicate_key] = action
        normalized[action.config_key] = accelerator
    return normalized


def sanitize_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    """Preserve valid bindings while dropping invalid or duplicate values."""
    sanitized: dict[str, str] = {}
    owners: set[str] = set()
    for action in HOTKEY_ACTIONS:
        accelerator = normalize_accelerator(
            bindings.get(action.config_key, "")
        )
        duplicate_key = accelerator.casefold()
        if not accelerator or duplicate_key in owners:
            sanitized[action.config_key] = ""
            continue
        owners.add(duplicate_key)
        sanitized[action.config_key] = accelerator
    return sanitized


def bindings_from_config(config) -> dict[str, str]:
    return {
        action.config_key: getattr(config, action.config_key, "")
        for action in HOTKEY_ACTIONS
    }


def _unpack(value):
    while isinstance(value, GLib.Variant):
        value = value.unpack()
    return value


def _portal_create_parameters(
    request_token: str,
    session_token: str,
) -> GLib.Variant:
    return GLib.Variant(
        "(a{sv})",
        ({
            "handle_token": GLib.Variant("s", request_token),
            "session_handle_token": GLib.Variant("s", session_token),
        },),
    )


def _portal_bind_parameters(
    session_handle: str,
    bindings: Mapping[str, str],
    request_token: str,
) -> GLib.Variant:
    shortcuts = []
    for action in HOTKEY_ACTIONS:
        properties = {
            "description": GLib.Variant("s", action.description),
        }
        preferred = accelerator_to_portal_trigger(
            bindings.get(action.config_key, "")
        )
        if preferred:
            properties["preferred_trigger"] = GLib.Variant("s", preferred)
        shortcuts.append((action.action_id, properties))
    return GLib.Variant(
        "(oa(sa{sv})sa{sv})",
        (
            session_handle,
            shortcuts,
            "",
            {"handle_token": GLib.Variant("s", request_token)},
        ),
    )


def _portal_request_path(connection, token: str) -> str:
    sender = (connection.get_unique_name() or "").lstrip(":").replace(".", "_")
    return f"{_PORTAL_OBJECT_PATH}/request/{sender}/{token}"


@dataclass
class _PortalRequest:
    path: str
    subscription_id: int
    callback: Callable[[int, dict], None]
    generation: int


class _PortalBackend:
    name = "portal"
    title = "Desktop Portal"
    inline_editable = False

    def __init__(
        self,
        connection,
        version: int,
        on_action: ActionCallback,
        on_state: StateCallback,
    ):
        self._connection = connection
        self.version = version
        self._on_action = on_action
        self._on_state = on_state
        self._session_handle = ""
        self._generation = 0
        self._requests: dict[str, _PortalRequest] = {}
        self._signal_ids = [
            connection.signal_subscribe(
                _PORTAL_BUS_NAME,
                _PORTAL_INTERFACE,
                "Activated",
                _PORTAL_OBJECT_PATH,
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_activated,
            ),
            connection.signal_subscribe(
                _PORTAL_BUS_NAME,
                _PORTAL_INTERFACE,
                "ShortcutsChanged",
                _PORTAL_OBJECT_PATH,
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_shortcuts_changed,
            ),
        ]

    @classmethod
    def try_create(
        cls,
        on_action: ActionCallback,
        on_state: StateCallback,
    ) -> _PortalBackend | None:
        try:
            connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            result = connection.call_sync(
                _PORTAL_BUS_NAME,
                _PORTAL_OBJECT_PATH,
                "org.freedesktop.DBus.Properties",
                "Get",
                GLib.Variant("(ss)", (_PORTAL_INTERFACE, "version")),
                GLib.VariantType.new("(v)"),
                Gio.DBusCallFlags.NONE,
                750,
                None,
            )
            version = int(_unpack(result)[0])
            if version < 1:
                return None
            return cls(connection, version, on_action, on_state)
        except (GLib.Error, TypeError, ValueError):
            return None

    def apply(self, enabled: bool, bindings: Mapping[str, str]) -> bool:
        self._generation += 1
        self._close_session()
        if not enabled:
            self._on_state(False, "Global hotkeys are off", {})
            return True

        generation = self._generation
        request_token = self._new_token("nvb_create")
        session_token = self._new_token("nvb_session")
        parameters = _portal_create_parameters(request_token, session_token)

        def _created(response: int, results: dict) -> None:
            if generation != self._generation:
                return
            if response != 0:
                status = (
                    "Global shortcut setup was canceled"
                    if response == 1
                    else "The desktop could not create a global shortcut session"
                )
                self._on_state(False, status, {})
                return
            session_handle = str(_unpack(results.get("session_handle", "")))
            if (
                not GLib.Variant.is_object_path(session_handle)
                or not session_handle.startswith(
                    f"{_PORTAL_OBJECT_PATH}/session/"
                )
            ):
                self._on_state(
                    False,
                    "The desktop returned an invalid global shortcut session",
                    {},
                )
                return
            self._session_handle = session_handle
            self._bind_shortcuts(bindings, generation)

        self._call_request(
            "CreateSession",
            parameters,
            request_token,
            _created,
            generation,
        )
        self._on_state(False, "Waiting for desktop shortcut setup", {})
        return True

    def configure(self) -> bool:
        if self.version < 2 or not self._session_handle:
            return False
        self._connection.call(
            _PORTAL_BUS_NAME,
            _PORTAL_OBJECT_PATH,
            _PORTAL_INTERFACE,
            "ConfigureShortcuts",
            GLib.Variant(
                "(osa{sv})",
                (self._session_handle, "", {}),
            ),
            GLib.VariantType.new("()"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._on_configure_returned,
            None,
        )
        return True

    def _bind_shortcuts(
        self,
        bindings: Mapping[str, str],
        generation: int,
    ) -> None:
        request_token = self._new_token("nvb_bind")
        parameters = _portal_bind_parameters(
            self._session_handle,
            bindings,
            request_token,
        )

        def _bound(response: int, results: dict) -> None:
            if generation != self._generation:
                return
            if response != 0:
                status = (
                    "Global shortcut setup was canceled"
                    if response == 1
                    else "The desktop could not bind global shortcuts"
                )
                self._on_state(False, status, {})
                return
            display = self._display_bindings(results.get("shortcuts", []))
            self._on_state(True, "Global hotkeys are active", display)

        self._call_request(
            "BindShortcuts",
            parameters,
            request_token,
            _bound,
            generation,
        )

    def _call_request(
        self,
        method: str,
        parameters: GLib.Variant,
        token: str,
        callback: Callable[[int, dict], None],
        generation: int,
    ) -> None:
        expected_path = _portal_request_path(self._connection, token)
        subscription_id = self._subscribe_request(expected_path)
        request = _PortalRequest(
            expected_path,
            subscription_id,
            callback,
            generation,
        )
        self._requests[expected_path] = request
        self._connection.call(
            _PORTAL_BUS_NAME,
            _PORTAL_OBJECT_PATH,
            _PORTAL_INTERFACE,
            method,
            parameters,
            GLib.VariantType.new("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._on_request_returned,
            (expected_path, method),
        )

    def _subscribe_request(self, path: str) -> int:
        return self._connection.signal_subscribe(
            _PORTAL_BUS_NAME,
            _REQUEST_INTERFACE,
            "Response",
            path,
            None,
            Gio.DBusSignalFlags.NONE,
            self._on_request_response,
        )

    def _on_request_returned(self, connection, result, user_data) -> None:
        expected_path, method = user_data
        request = self._requests.get(expected_path)
        try:
            returned_path = str(_unpack(connection.call_finish(result))[0])
        except GLib.Error as error:
            if request is not None:
                self._finish_request(expected_path)
                if request.generation == self._generation:
                    self._on_state(
                        False,
                        f"Global shortcut {method} failed: {error.message}",
                        {},
                    )
            return

        if request is None or returned_path == expected_path:
            return
        new_subscription = self._subscribe_request(returned_path)
        self._requests[returned_path] = _PortalRequest(
            returned_path,
            new_subscription,
            request.callback,
            request.generation,
        )
        self._finish_request(expected_path)

    def _on_request_response(
        self,
        _connection,
        _sender_name,
        object_path,
        _interface_name,
        _signal_name,
        parameters,
    ) -> None:
        request = self._requests.get(object_path)
        if request is None:
            return
        response, results = _unpack(parameters)
        self._finish_request(object_path)
        request.callback(int(response), dict(results))

    def _finish_request(self, path: str) -> None:
        request = self._requests.pop(path, None)
        if request is not None:
            self._connection.signal_unsubscribe(request.subscription_id)

    def _on_activated(
        self,
        _connection,
        _sender_name,
        _object_path,
        _interface_name,
        _signal_name,
        parameters,
    ) -> None:
        session_handle, action_id, _timestamp, _options = _unpack(parameters)
        if session_handle == self._session_handle and action_id in _ACTIONS_BY_ID:
            self._on_action(action_id)

    def _on_shortcuts_changed(
        self,
        _connection,
        _sender_name,
        _object_path,
        _interface_name,
        _signal_name,
        parameters,
    ) -> None:
        session_handle, shortcuts = _unpack(parameters)
        if session_handle == self._session_handle:
            self._on_state(
                True,
                "Global hotkeys are active",
                self._display_bindings(shortcuts),
            )

    def _on_configure_returned(self, connection, result, _user_data) -> None:
        try:
            connection.call_finish(result)
        except GLib.Error as error:
            self._on_state(
                bool(self._session_handle),
                f"Desktop shortcut settings could not open: {error.message}",
                {},
            )

    @staticmethod
    def _display_bindings(shortcuts) -> dict[str, str]:
        display: dict[str, str] = {}
        for action_id, raw_properties in _unpack(shortcuts):
            action = _ACTIONS_BY_ID.get(str(action_id))
            if action is None:
                continue
            properties = {
                str(key): _unpack(value)
                for key, value in dict(raw_properties).items()
            }
            trigger = str(properties.get("trigger_description", ""))
            display[action.config_key] = trigger
        return display

    @staticmethod
    def _new_token(prefix: str) -> str:
        return f"{prefix}_{secrets.token_hex(12)}"

    def _close_session(self) -> None:
        for path in list(self._requests):
            request = self._requests.get(path)
            if request is not None:
                self._connection.call(
                    _PORTAL_BUS_NAME,
                    path,
                    _REQUEST_INTERFACE,
                    "Close",
                    None,
                    GLib.VariantType.new("()"),
                    Gio.DBusCallFlags.NO_AUTO_START,
                    500,
                    None,
                    None,
                    None,
                )
            self._finish_request(path)
        if self._session_handle:
            self._connection.call(
                _PORTAL_BUS_NAME,
                self._session_handle,
                _SESSION_INTERFACE,
                "Close",
                None,
                GLib.VariantType.new("()"),
                Gio.DBusCallFlags.NO_AUTO_START,
                500,
                None,
                None,
                None,
            )
            self._session_handle = ""

    def close(self) -> None:
        self._generation += 1
        self._close_session()
        for subscription_id in self._signal_ids:
            self._connection.signal_unsubscribe(subscription_id)
        self._signal_ids.clear()


def _gnome_gapplication_path() -> str:
    if os.environ.get("SNAP"):
        # GNOME executes custom-keybinding commands on the host, where paths
        # mounted only inside the snap namespace do not exist.
        return "/usr/bin/gapplication"
    candidate = shutil.which("gapplication")
    return os.path.realpath(candidate) if candidate else ""


class _GnomeSettingsBackend:
    name = "gnome"
    title = "GNOME"
    inline_editable = True
    version = 1

    def __init__(
        self,
        root_settings,
        settings_factory,
        gapplication_path: str,
        on_action: ActionCallback,
        on_state: StateCallback,
    ):
        self._root_settings = root_settings
        self._settings_factory = settings_factory
        self._gapplication_path = gapplication_path
        self._on_action = on_action
        self._on_state = on_state

    @classmethod
    def try_create(
        cls,
        on_action: ActionCallback,
        on_state: StateCallback,
    ) -> _GnomeSettingsBackend | None:
        desktops = os.environ.get("XDG_CURRENT_DESKTOP", "").casefold()
        if "gnome" not in desktops and "ubuntu" not in desktops:
            return None
        schema_source = Gio.SettingsSchemaSource.get_default()
        if schema_source is None:
            return None
        root_schema = schema_source.lookup(_GNOME_ROOT_SCHEMA, True)
        shortcut_schema = schema_source.lookup(_GNOME_SHORTCUT_SCHEMA, True)
        gapplication_path = _gnome_gapplication_path()
        if (
            root_schema is None
            or shortcut_schema is None
            or not gapplication_path
            or not os.path.isabs(gapplication_path)
        ):
            return None
        root_settings = Gio.Settings.new_full(root_schema, None, None)
        if not root_settings.is_writable("custom-keybindings"):
            return None

        def _factory(path: str):
            return Gio.Settings.new_full(shortcut_schema, None, path)

        return cls(
            root_settings,
            _factory,
            os.path.realpath(gapplication_path),
            on_action,
            on_state,
        )

    @staticmethod
    def path_for(action: HotkeyAction) -> str:
        return f"{_GNOME_SHORTCUT_ROOT}nvbroadcast-{action.action_id}/"

    def command_for(self, action: HotkeyAction) -> str:
        return shlex.join([
            self._gapplication_path,
            "action",
            APP_ID,
            action.action_id,
        ])

    def apply(self, enabled: bool, bindings: Mapping[str, str]) -> bool:
        owned_paths = {self.path_for(action) for action in HOTKEY_ACTIONS}
        current_paths = list(self._root_settings.get_strv("custom-keybindings"))
        unrelated_paths = [
            path for path in current_paths if path not in owned_paths
        ]
        active_paths: list[str] = []

        try:
            for action in HOTKEY_ACTIONS:
                path = self.path_for(action)
                settings = self._settings_factory(path)
                binding = bindings.get(action.config_key, "") if enabled else ""
                if binding:
                    if not all(
                        settings.is_writable(key)
                        for key in ("name", "command", "binding")
                    ):
                        raise RuntimeError("GNOME shortcut settings are read-only")
                    values = (
                        ("name", f"NV Broadcast: {action.title}"),
                        ("command", self.command_for(action)),
                        ("binding", binding),
                    )
                    if not all(
                        settings.set_string(key, value)
                        for key, value in values
                    ):
                        raise RuntimeError("GNOME rejected a shortcut")
                    active_paths.append(path)
                else:
                    settings.reset("binding")
                    settings.reset("command")
                    settings.reset("name")
            if not self._root_settings.set_strv(
                "custom-keybindings",
                [*unrelated_paths, *active_paths],
            ):
                raise RuntimeError("GNOME rejected the shortcut list")
            Gio.Settings.sync()
        except (GLib.Error, RuntimeError) as error:
            try:
                self._root_settings.set_strv(
                    "custom-keybindings",
                    unrelated_paths,
                )
                Gio.Settings.sync()
            except GLib.Error:
                pass
            self._on_state(False, f"Could not update GNOME hotkeys: {error}", {})
            return False

        display = {
            key: accelerator_label(value)
            for key, value in bindings.items()
            if enabled and value
        }
        status = (
            "Global hotkeys are active"
            if active_paths
            else "Add a shortcut to activate global hotkeys"
            if enabled
            else "Global hotkeys are off"
        )
        self._on_state(bool(active_paths), status, display)
        return True

    def configure(self) -> bool:
        return False

    def close(self) -> None:
        # GNOME owns persistent custom shortcuts. Keep them registered so the
        # preference survives normal app restarts.
        pass


class _UnavailableBackend:
    name = "unavailable"
    title = "Unavailable"
    inline_editable = False
    version = 0

    def __init__(self, on_state: StateCallback):
        self._on_state = on_state

    def apply(self, enabled: bool, _bindings: Mapping[str, str]) -> bool:
        if enabled:
            self._on_state(
                False,
                "Global hotkeys are not supported by this desktop",
                {},
            )
            return False
        self._on_state(False, "Global hotkeys are unavailable", {})
        return True

    def configure(self) -> bool:
        return False

    def close(self) -> None:
        pass


class GlobalHotkeyManager:
    """Select a supported backend and keep global effect actions synchronized."""

    def __init__(
        self,
        on_action: ActionCallback,
        on_state: StateCallback | None = None,
        backend=None,
    ):
        self._on_action = on_action
        self._on_state = on_state or (lambda _active, _status, _display: None)
        if backend is not None:
            self._backend = backend
        elif platform.system() != "Linux":
            self._backend = _UnavailableBackend(self._on_state)
        else:
            self._backend = (
                _PortalBackend.try_create(self._dispatch_action, self._on_state)
                or _GnomeSettingsBackend.try_create(
                    self._dispatch_action,
                    self._on_state,
                )
                or _UnavailableBackend(self._on_state)
            )

    @property
    def available(self) -> bool:
        return self._backend.name != "unavailable"

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def backend_title(self) -> str:
        return self._backend.title

    @property
    def inline_editable(self) -> bool:
        return self._backend.inline_editable

    @property
    def can_configure(self) -> bool:
        return (
            self._backend.name == "portal"
            and self._backend.version >= 2
        )

    def apply(self, enabled: bool, bindings: Mapping[str, str]) -> bool:
        normalized = normalize_bindings(bindings)
        try:
            return self._backend.apply(bool(enabled), normalized)
        except (GLib.Error, RuntimeError, TypeError, ValueError) as error:
            self._on_state(
                False,
                f"Global hotkey backend failed: {error}",
                {},
            )
            return False

    def configure(self) -> bool:
        try:
            return self._backend.configure()
        except (GLib.Error, RuntimeError, TypeError, ValueError) as error:
            self._on_state(
                False,
                f"Desktop shortcut settings failed: {error}",
                {},
            )
            return False

    def close(self) -> None:
        try:
            self._backend.close()
        except (GLib.Error, RuntimeError, TypeError, ValueError):
            pass

    def _dispatch_action(self, action_id: str) -> None:
        if action_id in _ACTIONS_BY_ID:
            self._on_action(action_id)

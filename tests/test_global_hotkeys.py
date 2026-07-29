import unittest
from types import SimpleNamespace
from unittest import mock

from gi.repository import GLib

from nvbroadcast.core.constants import APP_ID
from nvbroadcast.core.global_hotkeys import (
    HOTKEY_ACTIONS,
    GlobalHotkeyManager,
    HotkeyValidationError,
    _gnome_gapplication_path,
    _GnomeSettingsBackend,
    _portal_bind_parameters,
    _portal_create_parameters,
    _PortalBackend,
    accelerator_label,
    accelerator_to_portal_trigger,
    normalize_accelerator,
    normalize_bindings,
)


class AcceleratorValidationTests(unittest.TestCase):
    def test_strong_modifier_shortcuts_are_normalized(self):
        self.assertEqual(
            normalize_accelerator("<Primary><Alt>b"),
            "<Control><Alt>b",
        )
        self.assertEqual(
            normalize_accelerator("<Super><Shift>F12"),
            "<Shift><Super>F12",
        )

    def test_plain_and_shift_only_keys_are_rejected(self):
        self.assertEqual(normalize_accelerator("b"), "")
        self.assertEqual(normalize_accelerator("<Shift>b"), "")
        self.assertEqual(normalize_accelerator("<Control>"), "")

    def test_function_and_media_keys_are_safe_without_modifiers(self):
        self.assertEqual(normalize_accelerator("F9"), "F9")
        self.assertTrue(normalize_accelerator("XF86AudioMute"))

    def test_control_characters_and_oversized_values_are_rejected(self):
        self.assertEqual(normalize_accelerator("<Control>a\n"), "")
        self.assertEqual(normalize_accelerator("a" * 129), "")

    def test_duplicate_bindings_are_rejected(self):
        with self.assertRaisesRegex(
            HotkeyValidationError,
            "cannot use the same shortcut",
        ):
            normalize_bindings({
                "toggle_background": "<Control><Alt>b",
                "toggle_auto_frame": "<Primary><Alt>b",
            })

    def test_unknown_binding_keys_are_not_returned(self):
        normalized = normalize_bindings({
            "toggle_background": "<Control><Alt>b",
            "start-recording": "<Control><Alt>r",
        })
        self.assertNotIn("start-recording", normalized)
        self.assertEqual(set(normalized), {
            action.config_key for action in HOTKEY_ACTIONS
        })

    def test_labels_and_portal_triggers_use_desktop_formats(self):
        self.assertTrue(accelerator_label("<Control><Alt>b"))
        self.assertEqual(
            accelerator_to_portal_trigger("<Control><Alt>b"),
            "CTRL+ALT+b",
        )
        self.assertEqual(
            accelerator_to_portal_trigger("<Shift><Super>F12"),
            "SHIFT+LOGO+F12",
        )


class PortalVariantTests(unittest.TestCase):
    def test_create_session_parameters_have_both_unpredictable_tokens(self):
        params = _portal_create_parameters("request_token", "session_token")
        self.assertEqual(params.get_type_string(), "(a{sv})")
        options = params.unpack()[0]
        self.assertEqual(options["handle_token"], "request_token")
        self.assertEqual(options["session_handle_token"], "session_token")

    def test_bind_parameters_only_include_valid_preferred_triggers(self):
        params = _portal_bind_parameters(
            "/org/freedesktop/portal/desktop/session/1_2/test",
            {
                "toggle_background": "<Control><Alt>b",
                "toggle_auto_frame": "",
            },
            "request_token",
        )
        self.assertEqual(params.get_type_string(), "(oa(sa{sv})sa{sv})")
        session, shortcuts, parent, options = params.unpack()
        self.assertEqual(
            session,
            "/org/freedesktop/portal/desktop/session/1_2/test",
        )
        self.assertEqual(parent, "")
        self.assertEqual(options["handle_token"], "request_token")
        shortcut_map = {action_id: props for action_id, props in shortcuts}
        self.assertEqual(
            shortcut_map["toggle-background"]["preferred_trigger"],
            "CTRL+ALT+b",
        )
        self.assertNotIn(
            "preferred_trigger",
            shortcut_map["toggle-auto-frame"],
        )

    @mock.patch("nvbroadcast.core.global_hotkeys.Gio.bus_get_sync")
    def test_portal_probe_reads_the_version_property(self, bus_get_sync):
        connection = mock.Mock()
        connection.call_sync.return_value = GLib.Variant(
            "(v)",
            (GLib.Variant("u", 2),),
        )
        connection.signal_subscribe.side_effect = [11, 12]
        bus_get_sync.return_value = connection

        backend = _PortalBackend.try_create(mock.Mock(), mock.Mock())

        self.assertIsNotNone(backend)
        self.assertEqual(backend.version, 2)
        args = connection.call_sync.call_args.args
        self.assertEqual(len(args), 9)
        self.assertEqual(args[2], "org.freedesktop.DBus.Properties")
        self.assertEqual(args[3], "Get")
        self.assertEqual(args[5].dup_string(), "(v)")
        backend.close()


class _FakePortalConnection:
    def __init__(self):
        self._next_subscription = 1
        self.subscriptions = {}
        self.calls = []

    def get_unique_name(self):
        return ":1.77"

    def signal_subscribe(
        self,
        sender,
        interface,
        member,
        path,
        _arg0,
        _flags,
        callback,
    ):
        subscription = self._next_subscription
        self._next_subscription += 1
        self.subscriptions[subscription] = {
            "sender": sender,
            "interface": interface,
            "member": member,
            "path": path,
            "callback": callback,
        }
        return subscription

    def signal_unsubscribe(self, subscription):
        self.subscriptions.pop(subscription, None)

    def call(
        self,
        bus_name,
        object_path,
        interface,
        method,
        parameters,
        _reply_type,
        flags,
        timeout,
        _cancellable,
        callback,
        user_data,
    ):
        self.calls.append({
            "bus_name": bus_name,
            "object_path": object_path,
            "interface": interface,
            "method": method,
            "parameters": parameters,
            "flags": flags,
            "timeout": timeout,
        })
        if callback is None:
            return
        if method == "CreateSession":
            token = parameters.unpack()[0]["handle_token"]
        elif method == "BindShortcuts":
            token = parameters.unpack()[3]["handle_token"]
        else:
            callback(
                self,
                SimpleNamespace(reply=GLib.Variant("()", ())),
                user_data,
            )
            return
        path = (
            "/org/freedesktop/portal/desktop/request/1_77/"
            f"{token}"
        )
        callback(
            self,
            SimpleNamespace(reply=GLib.Variant("(o)", (path,))),
            user_data,
        )

    @staticmethod
    def call_finish(result):
        return result.reply

    def emit(self, interface, member, path, parameters):
        matches = [
            entry["callback"]
            for entry in self.subscriptions.values()
            if entry["interface"] == interface
            and entry["member"] == member
            and entry["path"] == path
        ]
        if not matches:
            raise AssertionError(
                f"No subscription for {interface}.{member} at {path}"
            )
        for callback in matches:
            callback(
                self,
                "org.freedesktop.portal.Desktop",
                path,
                interface,
                member,
                parameters,
            )


class PortalBackendFlowTests(unittest.TestCase):
    def setUp(self):
        self.connection = _FakePortalConnection()
        self.actions = []
        self.states = []
        self.backend = _PortalBackend(
            self.connection,
            2,
            self.actions.append,
            lambda *state: self.states.append(state),
        )

    def _pending_request_path(self):
        paths = [
            entry["path"]
            for entry in self.connection.subscriptions.values()
            if entry["interface"] == "org.freedesktop.portal.Request"
        ]
        self.assertEqual(len(paths), 1)
        return paths[0]

    def test_create_bind_activate_configure_and_close_flow(self):
        bindings = normalize_bindings({
            "toggle_background": "<Control><Alt>b",
        })

        self.assertTrue(self.backend.apply(True, bindings))
        self.assertEqual(
            self.connection.calls[-1]["method"],
            "CreateSession",
        )
        create_path = self._pending_request_path()
        session = (
            "/org/freedesktop/portal/desktop/session/1_77/"
            "nvb_session_test"
        )
        self.connection.emit(
            "org.freedesktop.portal.Request",
            "Response",
            create_path,
            GLib.Variant(
                "(ua{sv})",
                (
                    0,
                    {"session_handle": GLib.Variant("s", session)},
                ),
            ),
        )

        self.assertEqual(
            self.connection.calls[-1]["method"],
            "BindShortcuts",
        )
        bind_path = self._pending_request_path()
        self.connection.emit(
            "org.freedesktop.portal.Request",
            "Response",
            bind_path,
            GLib.Variant(
                "(ua{sv})",
                (
                    0,
                    {
                        "shortcuts": GLib.Variant(
                            "a(sa{sv})",
                            [(
                                "toggle-background",
                                {
                                    "description": GLib.Variant(
                                        "s",
                                        "Toggle background processing",
                                    ),
                                    "trigger_description": GLib.Variant(
                                        "s",
                                        "Ctrl+Alt+B",
                                    ),
                                },
                            )],
                        ),
                    },
                ),
            ),
        )

        self.assertTrue(self.states[-1][0])
        self.assertEqual(
            self.states[-1][2]["toggle_background"],
            "Ctrl+Alt+B",
        )
        self.connection.emit(
            "org.freedesktop.portal.GlobalShortcuts",
            "Activated",
            "/org/freedesktop/portal/desktop",
            GLib.Variant(
                "(osta{sv})",
                (session, "toggle-background", 123, {}),
            ),
        )
        self.assertEqual(self.actions, ["toggle-background"])

        self.assertTrue(self.backend.configure())
        self.assertEqual(
            self.connection.calls[-1]["method"],
            "ConfigureShortcuts",
        )
        self.backend.close()
        self.assertEqual(self.connection.calls[-1]["method"], "Close")
        self.assertEqual(
            self.connection.calls[-1]["object_path"],
            session,
        )

    def test_canceled_session_never_binds_or_activates(self):
        self.backend.apply(True, normalize_bindings({}))
        create_path = self._pending_request_path()

        self.connection.emit(
            "org.freedesktop.portal.Request",
            "Response",
            create_path,
            GLib.Variant("(ua{sv})", (1, {})),
        )

        self.assertFalse(self.states[-1][0])
        self.assertIn("canceled", self.states[-1][1])
        self.assertEqual(
            [call["method"] for call in self.connection.calls],
            ["CreateSession"],
        )


class _FakeRootSettings:
    def __init__(self, paths):
        self.paths = list(paths)

    def get_strv(self, _key):
        return list(self.paths)

    def set_strv(self, _key, value):
        self.paths = list(value)
        return True


class _FakeShortcutSettings:
    def __init__(self, reject_key=None):
        self.values = {}
        self.reject_key = reject_key

    def is_writable(self, _key):
        return True

    def set_string(self, key, value):
        if key == self.reject_key:
            return False
        self.values[key] = value
        return True

    def reset(self, key):
        self.values.pop(key, None)


class GnomeSettingsBackendTests(unittest.TestCase):
    def setUp(self):
        self.unrelated = (
            "/org/gnome/settings-daemon/plugins/media-keys/"
            "custom-keybindings/my-existing-shortcut/"
        )
        self.root = _FakeRootSettings([self.unrelated])
        self.children = {}
        self.states = []
        self.backend = _GnomeSettingsBackend(
            self.root,
            lambda path: self.children.setdefault(
                path,
                _FakeShortcutSettings(),
            ),
            "/usr/bin/gapplication",
            mock.Mock(),
            lambda *state: self.states.append(state),
        )

    @mock.patch.object(GLib, "idle_add", wraps=GLib.idle_add)
    @mock.patch("nvbroadcast.core.global_hotkeys.Gio.Settings.sync")
    def test_registers_fixed_commands_without_overwriting_user_shortcuts(
        self,
        sync,
        _idle_add,
    ):
        bindings = {
            "toggle_background": "<Control><Alt>b",
            "toggle_mirror": "<Control><Alt>m",
        }

        self.assertTrue(self.backend.apply(True, normalize_bindings(bindings)))

        self.assertEqual(self.root.paths[0], self.unrelated)
        self.assertEqual(len(self.root.paths), 3)
        background = HOTKEY_ACTIONS[0]
        values = self.children[self.backend.path_for(background)].values
        self.assertEqual(values["name"], "NV Broadcast: Background")
        self.assertEqual(
            values["command"],
            f"/usr/bin/gapplication action {APP_ID} toggle-background",
        )
        self.assertEqual(values["binding"], "<Control><Alt>b")
        sync.assert_called_once_with()
        self.assertTrue(self.states[-1][0])

    @mock.patch("nvbroadcast.core.global_hotkeys.Gio.Settings.sync")
    def test_disabling_removes_only_owned_paths_and_values(self, _sync):
        normalized = normalize_bindings({
            "toggle_background": "<Control><Alt>b",
        })
        self.backend.apply(True, normalized)

        self.assertTrue(self.backend.apply(False, normalized))

        self.assertEqual(self.root.paths, [self.unrelated])
        for child in self.children.values():
            self.assertEqual(child.values, {})
        self.assertFalse(self.states[-1][0])

    @mock.patch("nvbroadcast.core.global_hotkeys.Gio.Settings.sync")
    def test_rebinding_replaces_owned_path_without_duplicates(self, _sync):
        self.backend.apply(
            True,
            normalize_bindings({
                "toggle_background": "<Control><Alt>b",
            }),
        )
        self.backend.apply(
            True,
            normalize_bindings({
                "toggle_background": "<Control><Alt>g",
            }),
        )

        owned = self.backend.path_for(HOTKEY_ACTIONS[0])
        self.assertEqual(self.root.paths.count(owned), 1)
        self.assertEqual(
            self.children[owned].values["binding"],
            "<Control><Alt>g",
        )

    @mock.patch("nvbroadcast.core.global_hotkeys.Gio.Settings.sync")
    def test_failed_child_write_deactivates_owned_shortcuts(self, _sync):
        background = HOTKEY_ACTIONS[0]
        owned = self.backend.path_for(background)
        self.root.paths.append(owned)
        self.children[owned] = _FakeShortcutSettings(
            reject_key="command"
        )

        self.assertFalse(self.backend.apply(
            True,
            normalize_bindings({
                "toggle_background": "<Control><Alt>b",
            }),
        ))

        self.assertEqual(self.root.paths, [self.unrelated])
        self.assertFalse(self.states[-1][0])

    @mock.patch.dict(
        "nvbroadcast.core.global_hotkeys.os.environ",
        {"SNAP": "/snap/nvbroadcast/current"},
        clear=False,
    )
    @mock.patch("nvbroadcast.core.global_hotkeys.shutil.which")
    def test_snap_command_uses_a_host_visible_executable(self, which):
        self.assertEqual(
            _gnome_gapplication_path(),
            "/usr/bin/gapplication",
        )
        which.assert_not_called()


class _FakeBackend:
    name = "fake"
    title = "Fake"
    inline_editable = True
    version = 1

    def __init__(self):
        self.applied = None
        self.closed = False

    def apply(self, enabled, bindings):
        self.applied = (enabled, bindings)
        return True

    def configure(self):
        return False

    def close(self):
        self.closed = True


class _FailingBackend(_FakeBackend):
    def apply(self, _enabled, _bindings):
        raise RuntimeError("backend offline")


class GlobalHotkeyManagerTests(unittest.TestCase):
    def test_manager_validates_before_backend_write(self):
        backend = _FakeBackend()
        manager = GlobalHotkeyManager(mock.Mock(), backend=backend)

        self.assertTrue(manager.apply(True, {
            "toggle_background": "<Primary><Alt>b",
        }))

        self.assertEqual(
            backend.applied[1]["toggle_background"],
            "<Control><Alt>b",
        )

    def test_manager_does_not_forward_unknown_action_ids(self):
        callback = mock.Mock()
        manager = GlobalHotkeyManager(callback, backend=_FakeBackend())

        manager._dispatch_action("start-recording")
        manager._dispatch_action("toggle-background")

        callback.assert_called_once_with("toggle-background")

    def test_backend_exception_is_reported_without_escaping_ui_callback(self):
        states = []
        manager = GlobalHotkeyManager(
            mock.Mock(),
            lambda *state: states.append(state),
            backend=_FailingBackend(),
        )

        self.assertFalse(manager.apply(True, {}))

        self.assertFalse(states[-1][0])
        self.assertIn("backend offline", states[-1][1])


if __name__ == "__main__":
    unittest.main()

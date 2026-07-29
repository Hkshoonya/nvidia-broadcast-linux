import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig
from nvbroadcast.core.global_hotkeys import HotkeyValidationError


class _FakeToggle:
    def __init__(self, active=False, sensitive=True):
        self.active = active
        self._sensitive = sensitive

    def get_sensitive(self):
        return self._sensitive


class AppHotkeyActionTests(unittest.TestCase):
    @staticmethod
    def _fake_app():
        window = SimpleNamespace(
            _bg_toggle=_FakeToggle(),
            _autoframe_toggle=_FakeToggle(),
            _eye_contact_toggle=_FakeToggle(),
            _mirror_toggle=_FakeToggle(active=True),
            _noise_toggle=_FakeToggle(),
            set_status=mock.Mock(),
        )
        return SimpleNamespace(
            _restoring=False,
            _window=window,
        )

    def test_each_exported_action_toggles_its_existing_ui_control(self):
        app = self._fake_app()
        expected = {
            "toggle-background": "_bg_toggle",
            "toggle-auto-frame": "_autoframe_toggle",
            "toggle-eye-contact": "_eye_contact_toggle",
            "toggle-mirror": "_mirror_toggle",
            "toggle-mic-noise": "_noise_toggle",
        }

        for action_id, attribute in expected.items():
            toggle = getattr(app._window, attribute)
            previous = toggle.active
            self.assertFalse(
                NVBroadcastApp._toggle_effect_from_hotkey(app, action_id)
            )
            self.assertIs(toggle.active, not previous)

        self.assertEqual(app._window.set_status.call_count, len(expected))

    def test_unavailable_effect_is_not_toggled(self):
        app = self._fake_app()
        app._window._eye_contact_toggle._sensitive = False

        NVBroadcastApp._toggle_effect_from_hotkey(
            app,
            "toggle-eye-contact",
        )

        self.assertFalse(app._window._eye_contact_toggle.active)
        app._window.set_status.assert_called_once_with(
            "Eye Contact is not available"
        )

    def test_actions_are_ignored_during_config_restore(self):
        app = self._fake_app()
        app._restoring = True

        NVBroadcastApp._toggle_effect_from_hotkey(
            app,
            "toggle-background",
        )

        self.assertFalse(app._window._bg_toggle.active)


class AppHotkeySettingsTests(unittest.TestCase):
    def test_restore_clears_only_invalid_saved_bindings(self):
        config = AppConfig()
        config.hotkeys.enabled = True
        config.hotkeys.toggle_background = "<Control><Alt>b"
        config.hotkeys.toggle_auto_frame = "<Primary><Alt>b"
        config.hotkeys.toggle_eye_contact = "e"
        config.hotkeys.toggle_mirror = "F12"
        manager = SimpleNamespace(
            apply=mock.Mock(
                side_effect=[
                    HotkeyValidationError("Invalid saved shortcuts"),
                    True,
                ]
            ),
        )
        app = SimpleNamespace(
            config=config,
            _hotkey_manager=manager,
            _hotkey_active=True,
            _hotkey_status="",
            _hotkey_display={"toggle_background": "Ctrl+Alt+B"},
            _window=None,
            _set_global_hotkey_actions_enabled=mock.Mock(),
        )

        with mock.patch("nvbroadcast.app.save_config") as save:
            self.assertFalse(NVBroadcastApp._sync_global_hotkeys(app))

        self.assertFalse(config.hotkeys.enabled)
        self.assertEqual(
            config.hotkeys.toggle_background,
            "<Control><Alt>b",
        )
        self.assertEqual(config.hotkeys.toggle_auto_frame, "")
        self.assertEqual(config.hotkeys.toggle_eye_contact, "")
        self.assertEqual(config.hotkeys.toggle_mirror, "F12")
        self.assertEqual(manager.apply.call_count, 2)
        self.assertIn("were cleared", app._hotkey_status)
        self.assertEqual(app._hotkey_display, {})
        save.assert_called_once_with(config)

    def test_portal_cancel_disables_exported_actions_and_preference(self):
        config = AppConfig()
        config.hotkeys.enabled = True
        app = SimpleNamespace(
            config=config,
            _hotkey_active=True,
            _hotkey_status="",
            _hotkey_display={"toggle_background": "Ctrl+Alt+B"},
            _window=None,
            _set_global_hotkey_actions_enabled=mock.Mock(),
        )

        with mock.patch("nvbroadcast.app.save_config") as save:
            NVBroadcastApp._on_global_hotkey_state(
                app,
                False,
                "Global shortcut setup was canceled",
                {},
            )

        self.assertFalse(config.hotkeys.enabled)
        app._set_global_hotkey_actions_enabled.assert_called_once_with(False)
        save.assert_called_once_with(config)

    def test_active_backend_enables_actions_only_when_preference_is_on(self):
        config = AppConfig()
        config.hotkeys.enabled = True
        app = SimpleNamespace(
            config=config,
            _hotkey_active=False,
            _hotkey_status="",
            _hotkey_display={},
            _window=None,
            _set_global_hotkey_actions_enabled=mock.Mock(),
        )

        NVBroadcastApp._on_global_hotkey_state(
            app,
            True,
            "Global hotkeys are active",
            {"toggle_background": "Ctrl+Alt+B"},
        )

        app._set_global_hotkey_actions_enabled.assert_called_once_with(True)

    def test_binding_is_saved_only_after_backend_accepts_it(self):
        config = AppConfig()
        manager = SimpleNamespace(
            inline_editable=True,
            apply=mock.Mock(return_value=True),
        )
        app = SimpleNamespace(
            config=config,
            _hotkey_manager=manager,
            _hotkey_status="",
            _window=None,
        )

        with mock.patch("nvbroadcast.app.save_config") as save:
            ok, message = NVBroadcastApp.set_hotkey_binding(
                app,
                "toggle_background",
                "<Primary><Alt>b",
            )

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertEqual(
            config.hotkeys.toggle_background,
            "<Control><Alt>b",
        )
        save.assert_called_once_with(config)

    def test_duplicate_binding_is_rejected_before_backend_write(self):
        config = AppConfig()
        config.hotkeys.toggle_background = "<Control><Alt>b"
        manager = SimpleNamespace(
            inline_editable=True,
            apply=mock.Mock(return_value=True),
        )
        app = SimpleNamespace(
            config=config,
            _hotkey_manager=manager,
            _hotkey_status="",
            _window=None,
        )

        ok, message = NVBroadcastApp.set_hotkey_binding(
            app,
            "toggle_auto_frame",
            "<Primary><Alt>b",
        )

        self.assertFalse(ok)
        self.assertIn("cannot use the same shortcut", message)
        manager.apply.assert_not_called()

    def test_failed_backend_write_does_not_persist_binding(self):
        config = AppConfig()
        manager = SimpleNamespace(
            inline_editable=True,
            apply=mock.Mock(side_effect=[False, True]),
        )
        app = SimpleNamespace(
            config=config,
            _hotkey_manager=manager,
            _hotkey_status="Desktop rejected shortcut",
            _window=None,
        )

        with mock.patch("nvbroadcast.app.save_config") as save:
            ok, message = NVBroadcastApp.set_hotkey_binding(
                app,
                "toggle_background",
                "<Control><Alt>b",
            )

        self.assertFalse(ok)
        self.assertEqual(message, "Desktop rejected shortcut")
        self.assertEqual(config.hotkeys.toggle_background, "")
        self.assertEqual(manager.apply.call_count, 2)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()

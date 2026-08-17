import unittest
from types import SimpleNamespace
from unittest import mock

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from nvbroadcast.core.config import AppConfig
from nvbroadcast.ui.sni_tray import SniTray, _ID_BROADCAST
from nvbroadcast.ui.tray import TrayIcon
from nvbroadcast.ui.window import NVBroadcastWindow


def _make_app(start_pipeline_return=True):
    return SimpleNamespace(
        config=AppConfig(),
        _streaming=False,
        start_pipeline=mock.Mock(return_value=start_pipeline_return),
        stop_pipeline=mock.Mock(),
        restore_current_config=mock.Mock(),
        set_auto_mode_enabled=mock.Mock(),
    )


def _make_window(app):
    """Build a window instance without GTK __init__ (no display needed)."""
    win = NVBroadcastWindow.__new__(NVBroadcastWindow)
    win._streaming = False
    win._app = app
    win._profile_btn = SimpleNamespace(set_label=mock.Mock())
    win._stream_btn = SimpleNamespace(
        set_label=mock.Mock(),
        remove_css_class=mock.Mock(),
        add_css_class=mock.Mock(),
    )
    win._format_selector = SimpleNamespace(get_selected_device=lambda: "YUY2")
    win._camera_selector = SimpleNamespace(get_selected_device=lambda: "/dev/video0")
    win._rebuild_profile_popover = mock.Mock()
    win.set_status = mock.Mock()
    return win


class ProfileSelectionAutoStartTests(unittest.TestCase):
    """Selecting a profile starts the broadcast only with an explicit opt-in."""

    def _select_profile(self, win, loaded):
        popover = SimpleNamespace(popdown=mock.Mock())
        with mock.patch(
            "nvbroadcast.core.config.load_profile", return_value=loaded
        ), mock.patch("nvbroadcast.core.config.save_config") as save_config:
            win._on_user_profile_selected(None, "P", popover)
        return popover, save_config

    def test_select_profile_with_optin_starts_broadcast(self):
        app = _make_app()
        win = _make_window(app)
        loaded = AppConfig()
        loaded.auto_start_on_select = True

        self._select_profile(win, loaded)

        app.start_pipeline.assert_called_once_with("/dev/video0", "YUY2")
        self.assertTrue(win._streaming)
        win._stream_btn.set_label.assert_called_with("Stop Broadcast")

    def test_select_profile_without_optin_does_not_start(self):
        app = _make_app()
        win = _make_window(app)
        loaded = AppConfig()  # auto_start_on_select defaults False

        self._select_profile(win, loaded)

        app.start_pipeline.assert_not_called()
        self.assertFalse(win._streaming)

    def test_select_profile_never_stops_when_optin_off_while_streaming(self):
        app = _make_app()
        app._streaming = True
        win = _make_window(app)
        win._streaming = True
        loaded = AppConfig()

        self._select_profile(win, loaded)

        app.stop_pipeline.assert_not_called()
        self.assertTrue(win._streaming)

    def test_select_profile_start_failure_keeps_state_aligned(self):
        app = _make_app(start_pipeline_return=False)
        win = _make_window(app)
        loaded = AppConfig()
        loaded.auto_start_on_select = True

        self._select_profile(win, loaded)

        app.start_pipeline.assert_called_once_with("/dev/video0", "YUY2")
        self.assertFalse(win._streaming)
        win._stream_btn.set_label.assert_not_called()
        self.assertFalse(app._streaming)

    def test_select_profile_optin_does_not_restart_when_window_flag_stale(self):
        app = _make_app()
        app._streaming = True          # authoritative: pipeline is active
        win = _make_window(app)
        win._streaming = False         # stale window flag disagrees
        loaded = AppConfig()
        loaded.auto_start_on_select = True

        self._select_profile(win, loaded)

        app.start_pipeline.assert_not_called()   # no restart of an active pipeline
        app.stop_pipeline.assert_not_called()    # never force-stops either
        self.assertTrue(win._streaming)
        win._stream_btn.set_label.assert_called_with("Stop Broadcast")
        win._stream_btn.remove_css_class.assert_called_with("suggested-action")
        win._stream_btn.add_css_class.assert_called_with("destructive-action")

        win._on_stream_toggle(win._stream_btn)

        app.start_pipeline.assert_not_called()
        app.stop_pipeline.assert_called_once_with()
        self.assertFalse(win._streaming)


class ProfileSaveOptInTests(unittest.TestCase):
    """Saving a profile records the explicit opt-in, never transient state."""

    def _save_profile(self, win, optin_active, streaming=True):
        win._streaming = streaming
        dialog = SimpleNamespace(destroy=mock.Mock())
        entry = SimpleNamespace(get_text=lambda: "My Stream")
        optin = SimpleNamespace(get_active=lambda: optin_active)
        saved = {}

        def _fake_save_profile(name, config):
            saved["config"] = config

        with mock.patch(
            "nvbroadcast.core.config.save_profile", side_effect=_fake_save_profile
        ), mock.patch("nvbroadcast.core.config.save_config") as save_config:
            win._on_save_profile_response(dialog, Gtk.ResponseType.OK, entry, optin)
        return saved, save_config

    def test_save_records_optin_not_transient_state(self):
        app = _make_app()
        win = _make_window(app)

        saved, _ = self._save_profile(win, optin_active=False, streaming=True)

        # The profile was saved while streaming, but the opt-in flag must come
        # from the dialog checkbox only; self._streaming is never consulted.
        self.assertIs(saved["config"].auto_start_on_select, False)
        win._rebuild_profile_popover.assert_called_once()

    def test_save_records_optin_true(self):
        app = _make_app()
        win = _make_window(app)

        saved, _ = self._save_profile(win, optin_active=True, streaming=False)

        self.assertIs(saved["config"].auto_start_on_select, True)
        win._rebuild_profile_popover.assert_called_once()

    def test_new_save_dialog_checkbox_defaults_off_when_current_config_flag_true(self):
        app = _make_app()
        app.config.auto_start_on_select = True   # current profile is opted-in
        win = _make_window(app)
        popover = SimpleNamespace(popdown=mock.Mock())
        dialog = SimpleNamespace(
            present=mock.Mock(),
            connect=mock.Mock(),
            get_content_area=lambda: SimpleNamespace(append=mock.Mock()),
        )
        optin = SimpleNamespace()
        optin.set_active = mock.Mock()

        with mock.patch("nvbroadcast.ui.window.Gtk.MessageDialog", return_value=dialog), \
             mock.patch("nvbroadcast.ui.window.Gtk.CheckButton", return_value=optin), \
             mock.patch(
                 "nvbroadcast.ui.window.Gtk.Entry",
                 return_value=SimpleNamespace(set_placeholder_text=mock.Mock()),
             ):
            win._on_save_profile(None, popover)

        optin.set_active.assert_called_once_with(False)


class ManualToggleFailureTests(unittest.TestCase):
    def test_manual_toggle_start_failure_keeps_state_aligned(self):
        app = _make_app(start_pipeline_return=False)
        win = _make_window(app)

        win._on_stream_toggle(win._stream_btn)

        app.start_pipeline.assert_called_once_with("/dev/video0", "YUY2")
        self.assertFalse(win._streaming)
        win._stream_btn.set_label.assert_not_called()


class TrayStartAlignmentTests(unittest.TestCase):
    """Tray start paths must also keep window state aligned on failure."""

    def _tray_app(self, win, start_pipeline_return):
        return SimpleNamespace(
            _streaming=False,
            _window=win,
            config=AppConfig(),
            start_pipeline=mock.Mock(return_value=start_pipeline_return),
            stop_pipeline=mock.Mock(),
        )

    def test_legacy_tray_toggle_start_success_marks_window_streaming(self):
        app = self._tray_app(_make_window(_make_app()), True)
        tray = TrayIcon.__new__(TrayIcon)
        tray._app = app

        tray._on_broadcast_toggle(None)

        self.assertTrue(app._window._streaming)
        app._window._stream_btn.set_label.assert_called_with("Stop Broadcast")

    def test_legacy_tray_toggle_start_failure_keeps_state_aligned(self):
        win = _make_window(_make_app())
        app = self._tray_app(win, False)
        tray = TrayIcon.__new__(TrayIcon)
        tray._app = app

        tray._on_broadcast_toggle(None)

        app.start_pipeline.assert_called_once()
        self.assertFalse(win._streaming)
        win._stream_btn.set_label.assert_not_called()

    def test_sni_tray_toggle_start_success_marks_window_streaming(self):
        app = self._tray_app(_make_window(_make_app()), True)
        tray = SniTray.__new__(SniTray)
        tray._app = app

        tray._on_menu_clicked(_ID_BROADCAST)

        self.assertTrue(app._window._streaming)
        app._window._stream_btn.set_label.assert_called_with("Stop Broadcast")

    def test_sni_tray_toggle_start_failure_keeps_state_aligned(self):
        win = _make_window(_make_app())
        app = self._tray_app(win, False)
        tray = SniTray.__new__(SniTray)
        tray._app = app

        tray._on_menu_clicked(_ID_BROADCAST)

        app.start_pipeline.assert_called_once()
        self.assertFalse(win._streaming)
        win._stream_btn.set_label.assert_not_called()


if __name__ == "__main__":
    unittest.main()

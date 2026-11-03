import unittest
from types import SimpleNamespace
from unittest import mock

from gi.repository import Gio

from nvbroadcast.ui import sni_tray


class SniTrayTests(unittest.TestCase):
    def test_dbus_interfaces_are_valid(self):
        item = Gio.DBusNodeInfo.new_for_xml(sni_tray._SNI_XML)
        menu = Gio.DBusNodeInfo.new_for_xml(sni_tray._MENU_XML)

        self.assertEqual(item.interfaces[0].name, "org.kde.StatusNotifierItem")
        self.assertEqual(menu.interfaces[0].name, "com.canonical.dbusmenu")

    def test_broadcast_label_uses_authoritative_application_state(self):
        tray = sni_tray.SniTray.__new__(sni_tray.SniTray)
        tray._app = SimpleNamespace(_streaming=False, _window=None)
        tray._streaming = True
        tray._status_text = "Streaming"

        items = dict(tray._menu_items())

        self.assertEqual(
            items[sni_tray._ID_BROADCAST]["label"].unpack(),
            "Start Broadcast",
        )

    def test_shutdown_releases_watcher_and_exported_objects(self):
        tray = sni_tray.SniTray.__new__(sni_tray.SniTray)
        tray._watch_id = 42
        tray._conn = mock.Mock()
        tray._reg_ids = [7, 8]
        tray._active = True

        with mock.patch.object(sni_tray.Gio, "bus_unwatch_name") as unwatch:
            tray.shutdown()

        unwatch.assert_called_once_with(42)
        self.assertEqual(
            tray._conn.unregister_object.call_args_list,
            [mock.call(7), mock.call(8)],
        )
        self.assertEqual(tray._watch_id, 0)
        self.assertEqual(tray._reg_ids, [])
        self.assertFalse(tray._active)

    def test_watcher_loss_marks_tray_unavailable_without_unexporting_item(self):
        tray = sni_tray.SniTray.__new__(sni_tray.SniTray)
        tray._conn = mock.Mock()
        tray._reg_ids = [7, 8]
        tray._active = True

        tray._on_watcher_vanished(tray._conn, "org.kde.StatusNotifierWatcher")

        self.assertFalse(tray.available)
        self.assertTrue(tray.bus_ready)
        tray._conn.unregister_object.assert_not_called()

    def test_replacement_watcher_must_accept_registration_before_available(self):
        tray = sni_tray.SniTray.__new__(sni_tray.SniTray)
        tray._active = True
        connection = mock.Mock()
        connection.get_unique_name.return_value = ":1.42"
        connection.call_sync.side_effect = RuntimeError("Registration rejected")

        tray._on_watcher_appeared(
            connection, "org.kde.StatusNotifierWatcher", ":1.43"
        )

        self.assertFalse(tray.available)
        connection.call_sync.side_effect = None
        tray._on_watcher_appeared(
            connection, "org.kde.StatusNotifierWatcher", ":1.44"
        )
        self.assertTrue(tray.available)


if __name__ == "__main__":
    unittest.main()

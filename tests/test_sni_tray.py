import unittest
from unittest import mock

from gi.repository import Gio

from nvbroadcast.ui import sni_tray


class SniTrayTests(unittest.TestCase):
    def test_dbus_interfaces_are_valid(self):
        item = Gio.DBusNodeInfo.new_for_xml(sni_tray._SNI_XML)
        menu = Gio.DBusNodeInfo.new_for_xml(sni_tray._MENU_XML)

        self.assertEqual(item.interfaces[0].name, "org.kde.StatusNotifierItem")
        self.assertEqual(menu.interfaces[0].name, "com.canonical.dbusmenu")

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


if __name__ == "__main__":
    unittest.main()

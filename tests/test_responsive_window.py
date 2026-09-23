"""Real GTK allocation checks; run with NVBROADCAST_TEST_LAYOUT=1 under Xvfb.

Opt-in keeps the ordinary unit suite from opening windows on a user's desktop.
Device discovery, backend probes, saved profiles, and periodic work are mocked.
"""

from contextlib import ExitStack
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from nvbroadcast.core.config import AppConfig
from nvbroadcast.ui.window import Adw, Gdk, Gio, GLib, Gtk, NVBroadcastWindow


@unittest.skipUnless(
    os.environ.get("NVBROADCAST_TEST_LAYOUT") == "1",
    "requires an explicitly enabled isolated GTK display",
)
class ResponsiveWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = Adw.Application(
            application_id="com.doczeus.NVBroadcast.LayoutTest",
            flags=Gio.ApplicationFlags.NON_UNIQUE,
        )
        cls.app.register(None)
        cls.css = Gtk.CssProvider()
        cls.css.load_from_path(str(
            Path(__file__).parents[1] / "src/nvbroadcast/ui/style.css"
        ))
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), cls.css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    @classmethod
    def tearDownClass(cls):
        Gtk.StyleContext.remove_provider_for_display(
            Gdk.Display.get_default(), cls.css,
        )

    def setUp(self):
        self.app.config = AppConfig()
        self.app.dependency_installer = mock.Mock()
        self.app.dependency_installer.unsupported_reason_for_mode.return_value = ""
        self.app.dependency_installer.missing_for_mode.return_value = []
        self.app.perf_monitor = SimpleNamespace(format_status=lambda: "")
        self.app.set_vcam_device = mock.Mock(return_value=True)
        self.patches = self.enterContext(ExitStack())
        for method in (
            "_populate_devices", "_populate_mics", "_populate_speakers",
            "_update_gpu_info", "_rebuild_profile_popover", "sync_hotkey_settings",
        ):
            self.patches.enter_context(mock.patch.object(NVBroadcastWindow, method))
        for name, result in (
            ("nvbroadcast.ui.window.list_camera_modes", []),
            ("nvbroadcast.ui.window.get_firefox_profiles", []),
            ("nvbroadcast.core.gpu.detect_gpus", []),
            ("nvbroadcast.core.config.detect_compositing_backends", {"cupy": True}),
            ("nvbroadcast.ui.window.has_tensorrt_runtime", True),
            ("nvbroadcast.ui.window.GLib.timeout_add_seconds", 0),
        ):
            self.patches.enter_context(mock.patch(name, return_value=result))
        self.window = NVBroadcastWindow(self.app)
        self.window._profile_btn.set_label("Profile: Default")
        for selector, name, device in (
            (self.window._camera_selector, "Test Camera", "/dev/video0"),
            (self.window._mic_selector, "Test Microphone", "test-mic"),
            (self.window._speaker_selector, "Test Speakers", "test-speaker"),
        ):
            selector.set_devices([{"name": name, "device": device}])
        self.addCleanup(self.window.destroy)
        header = self.window.get_content().get_first_child()
        self.paned = header.get_next_sibling().get_first_child()
        self.scroll = self.paned.get_end_child()
        self.controls = self.scroll.get_child().get_child()

    @staticmethod
    def _settle():
        deadline = time.monotonic() + 0.15
        context = GLib.MainContext.default()
        while time.monotonic() < deadline:
            while context.pending() and time.monotonic() < deadline:
                context.iteration(False)
            time.sleep(0.005)

    def _show(self, width, height=800):
        self.window.set_default_size(width, height)
        self.window.present()
        self._settle()

    def test_sections_wrap_and_fit_portrait_widths(self):
        for width, stacked in ((1280, False), (1080, True), (1024, True)):
            with self.subTest(width=width):
                self._show(width)
                self.assertLessEqual(
                    self.window.measure(Gtk.Orientation.HORIZONTAL, -1).minimum,
                    width,
                )
                self.assertLessEqual(self.window.get_width(), width)
                camera = self.controls.get_child_at_index(0)
                audio = self.controls.get_child_at_index(1)
                _, cam_bounds = camera.compute_bounds(self.controls)
                _, aud_bounds = audio.compute_bounds(self.controls)
                if stacked:
                    self.assertEqual(cam_bounds.get_x(), aud_bounds.get_x())
                    self.assertGreaterEqual(aud_bounds.get_y(), cam_bounds.get_height())
                else:
                    self.assertEqual(cam_bounds.get_y(), aud_bounds.get_y())
                    self.assertGreater(aud_bounds.get_x(), cam_bounds.get_x())
                for bounds in (cam_bounds, aud_bounds):
                    self.assertLessEqual(
                        bounds.get_x() + bounds.get_width(),
                        self.controls.get_width(),
                    )

    def test_stacked_audio_is_reachable_by_tab_and_scrolling(self):
        self._show(1080, 640)
        camera = self.controls.get_child_at_index(0)
        audio = self.controls.get_child_at_index(1)
        self.assertTrue(camera.child_focus(Gtk.DirectionType.TAB_FORWARD))
        reached_audio = False
        for _ in range(100):
            self.window.child_focus(Gtk.DirectionType.TAB_FORWARD)
            focus = self.window.get_focus()
            if focus is not None and focus.is_ancestor(audio):
                reached_audio = True
                break
        self.assertTrue(reached_audio, "Tab traversal must reach the stacked Audio section")
        self._settle()
        adjustment = self.scroll.get_vadjustment()
        self.assertGreater(adjustment.get_value(), 0)
        end = adjustment.get_upper() - adjustment.get_page_size()
        adjustment.set_value(end)
        self._settle()
        self.assertAlmostEqual(adjustment.get_value(), end)
        self.assertGreater(end, 0)

    def test_divider_can_give_stacked_controls_more_vertical_space(self):
        self._show(1080, 640)
        before = self.scroll.get_height()
        self.window._hide_btn.set_active(True)
        self.paned.set_position(0)
        self._settle()
        self.assertFalse(self.window._preview_frame.get_visible())
        self.assertGreater(self.scroll.get_height(), before)

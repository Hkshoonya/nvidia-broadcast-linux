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
        self.window._set_profile_name("Default")
        for selector, name, device in (
            (self.window._camera_selector, "Test Camera", "/dev/video0"),
            (self.window._mic_selector, "Test Microphone", "test-mic"),
            (self.window._speaker_selector, "Test Speakers", "test-speaker"),
        ):
            selector.set_devices([{"name": name, "device": device}])
        self.addCleanup(self.window.destroy)
        header = self.window.get_content().get_first_child()
        self.actions = header.get_next_sibling()
        self.body = self.actions.get_next_sibling()
        self.paned = self.body.get_content()
        self.controls_pane = self.paned.get_end_child()
        self.section_nav = self.controls_pane.get_first_child()
        self.scroll = self.section_nav.get_next_sibling()
        self.controls = self.scroll.get_child().get_child()

    @staticmethod
    def _settle(duration=0.15):
        deadline = time.monotonic() + duration
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
        for width, stacked in (
            (1280, False), (1080, None), (1024, None),
            (800, True), (600, "switch"), (480, "switch"), (420, "switch"),
        ):
            with self.subTest(width=width):
                self._show(width)
                self.assertLessEqual(
                    self.window.get_content().measure(
                        Gtk.Orientation.HORIZONTAL, -1
                    ).minimum,
                    self.window.get_width(),
                )
                self.assertLessEqual(self.window.get_width(), width)
                camera = self.controls.get_child_at_index(0)
                audio = self.controls.get_child_at_index(1)
                if stacked == "switch":
                    self.assertTrue(self.section_nav.get_visible())
                    self.assertTrue(camera.get_visible())
                    self.assertFalse(audio.get_visible())
                    self.window._audio_section_btn.set_active(True)
                    self._settle()
                    self.assertTrue(audio.get_visible())
                    self.assertFalse(camera.get_visible())
                    self.window._camera_section_btn.set_active(True)
                    self._settle()
                    continue
                self.assertFalse(self.section_nav.get_visible())
                _, cam_bounds = camera.compute_bounds(self.controls)
                _, aud_bounds = audio.compute_bounds(self.controls)
                if stacked is True:
                    self.assertEqual(cam_bounds.get_x(), aud_bounds.get_x())
                    self.assertGreaterEqual(aud_bounds.get_y(), cam_bounds.get_height())
                elif stacked is False:
                    self.assertEqual(cam_bounds.get_y(), aud_bounds.get_y())
                    self.assertGreater(aud_bounds.get_x(), cam_bounds.get_x())
                for bounds in (cam_bounds, aud_bounds):
                    self.assertLessEqual(
                        bounds.get_x() + bounds.get_width(),
                        self.controls.get_width(),
                    )

    def test_header_actions_and_meeting_notes_fit_narrow_window(self):
        self._show(1280)
        self.window._notes_sidebar_btn.set_active(True)
        self._settle(0.3)
        self.assertFalse(self.body.get_folded())
        self.assertTrue(self.body.get_reveal_flap())
        self.assertTrue(self.paned.is_sensitive())

        self._show(540)
        self._settle(0.3)
        self.assertTrue(self.body.get_folded())
        self.assertLessEqual(self.window.get_width(), 540)
        self.assertGreaterEqual(self.paned.get_width(), self.window.get_width() - 10)
        self.assertFalse(self.paned.is_sensitive())

        self._show(1280)
        self._settle(0.3)
        self.assertFalse(self.body.get_folded())
        self.assertTrue(self.body.get_reveal_flap())
        self.assertTrue(self.paned.is_sensitive())
        self._show(540)
        self._settle(0.3)
        self.assertTrue(self.body.get_folded())
        self.assertFalse(self.paned.is_sensitive())

        self.window.set_update_available(
            "1.5.3", "Update Available", "A new release", "https://example.com"
        )
        self._settle()
        self.assertLessEqual(self.window.get_width(), 540)
        content = self.window.get_content()
        _, actions_bounds = self.actions.compute_bounds(content)
        _, body_bounds = self.body.compute_bounds(content)
        self.assertLessEqual(
            actions_bounds.get_y() + actions_bounds.get_height(), body_bounds.get_y()
        )
        for button in (
            self.window._record_btn,
            self.window._notes_sidebar_btn,
            self.window._meeting_btn,
            self.window._update_btn,
        ):
            _, bounds = button.compute_bounds(self.actions)
            self.assertGreater(button.get_width(), 0)
            self.assertGreaterEqual(bounds.get_x(), 0)
            self.assertLessEqual(
                bounds.get_x() + bounds.get_width(), self.actions.get_width()
            )

        self.window._notes_sidebar_btn.grab_focus()
        self.assertIs(self.window.get_focus(), self.window._notes_sidebar_btn)
        self.window._notes_sidebar_btn.set_active(False)
        self._settle(0.3)
        self.assertFalse(self.body.get_reveal_flap())
        self.assertTrue(self.paned.is_sensitive())

    def test_long_profile_name_does_not_block_resize(self):
        name = "Wide Profile " * 8
        self.window._set_profile_name(name)
        self._show(480)
        self.assertLessEqual(self.window.get_width(), 480)
        self.assertEqual(self.window._profile_text.get_text(), f"Profile: {name}")
        self.assertIn(name, self.window._profile_btn.get_tooltip_text())
        self.assertLessEqual(self.window._profile_btn.get_width(), 200)
        self.window._profile_btn.popup()
        self._settle()
        self.assertTrue(self.window._profile_popover.get_visible())
        self.window._profile_btn.popdown()

    def test_compact_audio_is_reachable_by_switcher_and_tab(self):
        self._show(600, 640)
        camera = self.controls.get_child_at_index(0)
        audio = self.controls.get_child_at_index(1)
        self.assertTrue(camera.child_focus(Gtk.DirectionType.TAB_FORWARD))
        self.window._audio_section_btn.grab_focus()
        self.window._audio_section_btn.set_active(True)
        self._settle()
        self.assertFalse(camera.get_visible())
        self.assertTrue(audio.get_visible())
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
        self.assertAlmostEqual(adjustment.get_value(), 0)
        end = adjustment.get_upper() - adjustment.get_page_size()
        adjustment.set_value(end)
        self._settle()
        self.assertAlmostEqual(adjustment.get_value(), end)
        self.assertGreater(end, 0)

    def test_divider_can_give_stacked_controls_more_vertical_space(self):
        self._show(600, 640)
        before = self.scroll.get_height()
        self.window._hide_btn.set_active(True)
        self.paned.set_position(0)
        self._settle()
        self.assertFalse(self.window._preview_frame.get_visible())
        self.assertGreater(self.scroll.get_height(), before)

    def test_compact_header_and_preview_follow_resizing(self):
        self._show(1280, 900)
        self.assertFalse(self.window._stream_btn.is_ancestor(self.actions))
        self.assertFalse(self.section_nav.get_visible())
        self.assertTrue(self.controls.get_child_at_index(0).get_visible())
        self.assertTrue(self.controls.get_child_at_index(1).get_visible())
        self.assertGreaterEqual(self.window._preview.get_height(), 350)

        self._show(729, 720)
        self.assertTrue(self.window._stream_btn.is_ancestor(self.actions))
        self.assertTrue(self.section_nav.get_visible())
        self.assertLessEqual(self.window.get_width(), 729)
        self.assertLessEqual(self.window._preview.get_height(), 310)

        self._show(540, 640)
        self.assertTrue(self.window._stream_btn.is_ancestor(self.actions))
        self.assertLessEqual(self.window.get_width(), 540)
        self.assertGreaterEqual(self.window._preview.get_height(), 160)
        self.assertLessEqual(self.window._preview.get_height(), 220)
        self.assertGreaterEqual(self.scroll.get_height(), 170)

        self._show(540, 800)
        self.assertTrue(self.window._stream_btn.is_ancestor(self.actions))
        self.assertGreater(self.window._preview.get_height(), 220)
        self.assertLessEqual(self.window._preview.get_height(), 300)
        self._show(540, 640)
        self.assertTrue(self.window._stream_btn.is_ancestor(self.actions))
        self.assertLessEqual(self.window._preview.get_height(), 220)

        self._show(760, 720)
        self.assertTrue(self.window._stream_btn.is_ancestor(self.actions))
        self._show(780, 720)
        self.assertFalse(self.window._stream_btn.is_ancestor(self.actions))
        self._show(760, 720)
        self.assertTrue(self.window._stream_btn.is_ancestor(self.actions))

        self._show(800, 800)
        self.assertFalse(self.window._stream_btn.is_ancestor(self.actions))
        self.assertFalse(self.section_nav.get_visible())
        self.assertTrue(self.controls.get_child_at_index(1).get_visible())
        self.assertLessEqual(self.window._preview.get_height(), 320)
        self._show(1280, 900)
        self.assertFalse(self.window._stream_btn.is_ancestor(self.actions))
        self.assertGreaterEqual(self.window._preview.get_height(), 350)

    def test_audio_and_update_remain_reachable_at_420px(self):
        self.window.set_update_available(
            "1.5.3", "Update Available", "A new release", "https://example.com"
        )
        self._show(420, 540)
        self.assertLessEqual(self.window.get_width(), 420)
        self.assertLessEqual(self.actions.get_width(), self.window.get_width())
        self.assertGreater(self.window._update_btn.get_width(), 0)
        update_bounds = self.window._update_btn.compute_bounds(self.actions)[1]
        self.assertLessEqual(
            update_bounds.get_x() + update_bounds.get_width(),
            self.actions.get_width(),
        )

        format_dropdown = self.window._format_selector._dropdown
        self.assertLess(format_dropdown.measure(Gtk.Orientation.HORIZONTAL, -1).minimum, 200)
        self.assertIn("Chrome", format_dropdown.get_tooltip_text())

        self.window._audio_section_btn.set_active(True)
        self._settle()
        audio = self.controls.get_child_at_index(1)
        self.assertTrue(audio.get_visible())
        adjustment = self.scroll.get_vadjustment()
        adjustment.set_value(adjustment.get_upper() - adjustment.get_page_size())
        self._settle()
        _, bounds = audio.compute_bounds(self.scroll)
        self.assertLess(bounds.get_y(), self.scroll.get_height())
        self.assertGreater(bounds.get_y() + bounds.get_height(), 0)

        self.window._notes_sidebar_btn.set_active(True)
        self._settle(0.3)
        self.assertTrue(self.body.get_folded())
        self.assertLessEqual(self.window._meeting_sidebar.get_width(), self.window.get_width())
        self.window._notes_sidebar_btn.set_active(False)
        self._settle(0.3)
        self.assertTrue(self.paned.is_sensitive())

    def test_device_popup_keeps_full_names_at_compact_width(self):
        self._show(420, 540)
        dropdown = self.window._format_selector._dropdown
        toggle = dropdown.get_first_child()
        toggle.set_active(True)
        self._settle(0.3)

        def labels(widget):
            found = []
            child = widget.get_first_child()
            while child is not None:
                if isinstance(child, Gtk.Label):
                    found.append(child)
                found.extend(labels(child))
                child = child.get_next_sibling()
            return found

        popup_labels = labels(dropdown)
        for device in self.window._format_selector._devices:
            self.assertTrue(any(
                label.get_text() == device["name"]
                and label.get_wrap()
                and label.get_ellipsize().value_nick == "none"
                for label in popup_labels
            ), device["name"])
        toggle.set_active(False)

    def test_compact_about_closes_menu_before_opening_dialog(self):
        self._show(420, 540)
        self.window._compact_menu.popup()
        self._settle()
        self.assertTrue(self.window._compact_popover.get_visible())
        items = self.window._compact_popover.get_child()
        about = items.get_first_child().get_next_sibling().get_next_sibling()
        with mock.patch.object(self.window, "_show_about") as show_about:
            about.emit("clicked")
        self._settle()
        self.assertFalse(self.window._compact_popover.get_visible())
        show_about.assert_called_once_with(about)

    def test_resizing_to_compact_keeps_focused_audio_visible(self):
        self._show(1280, 900)
        audio = self.controls.get_child_at_index(1)
        self.window._mic_selector._dropdown.grab_focus()
        self.assertTrue(self.window.get_focus().is_ancestor(audio))

        self._show(540, 640)
        self.assertTrue(self.window._audio_section_btn.get_active())
        self.assertTrue(audio.get_visible())
        self.assertFalse(self.controls.get_child_at_index(0).get_visible())
        self.assertTrue(self.window.get_focus().is_ancestor(audio))

        self._show(540, 800)
        self.assertTrue(self.window._audio_section_btn.get_active())
        self.assertTrue(audio.get_visible())
        self._show(1280, 900)
        self.assertTrue(audio.get_visible())
        self.assertTrue(self.controls.get_child_at_index(0).get_visible())

    def test_large_text_and_meeting_states_fit_compact_width(self):
        settings = Gtk.Settings.get_default()
        previous_dpi = settings.get_property("gtk-xft-dpi")
        self.addCleanup(settings.set_property, "gtk-xft-dpi", previous_dpi)
        self.window._record_btn.set_label("Stop Rec")

        for scale in (1.5, 2.0):
            with self.subTest(scale=scale):
                settings.set_property("gtk-xft-dpi", int(96 * 1024 * scale))
                self._show(420, 540)
                self.window._set_meeting_button_state("active")
                self._settle()
                self.assertEqual(self.window._meeting_btn.get_label(), "End")
                self.assertLessEqual(
                    self.window.get_content().measure(
                        Gtk.Orientation.HORIZONTAL, -1
                    ).minimum,
                    self.window.get_width(),
                )

                self._show(1280, 900)
                self.assertEqual(self.window._meeting_btn.get_label(), "End Meeting")
                self._show(420, 540)
                self.assertEqual(self.window._meeting_btn.get_label(), "End")

                self.app.meeting_active = True
                self.app.meeting_finalizing = False
                self.app.stop_meeting_async = mock.Mock(return_value=True)
                self.window._on_meeting_toggle(self.window._meeting_btn)
                self._settle()
                self.assertEqual(self.window._meeting_btn.get_label(), "Saving…")
                self.assertFalse(self.window._meeting_btn.get_sensitive())
                self.assertLessEqual(
                    self.window.get_content().measure(
                        Gtk.Orientation.HORIZONTAL, -1
                    ).minimum,
                    self.window.get_width(),
                )
                self._show(1280, 900)
                self.assertEqual(self.window._meeting_btn.get_label(), "Finalizing...")
                self._show(420, 540)
                self.assertEqual(self.window._meeting_btn.get_label(), "Saving…")

                self.app.stop_meeting_async.call_args.args[0]("", "Meeting saved")
                self.assertEqual(self.window._meeting_btn.get_label(), "Meeting")
                self.assertTrue(self.window._meeting_btn.get_sensitive())
                self.app.meeting_active = False

    def test_without_breakpoint_api_uses_legacy_desktop_layout(self):
        with mock.patch(
            "nvbroadcast.ui.window._supports_responsive_breakpoints",
            return_value=False,
        ):
            legacy = NVBroadcastWindow(self.app)
        self.addCleanup(legacy.destroy)
        self.assertFalse(legacy._responsive_breakpoints)
        self.assertEqual(legacy.get_size_request()[0], 730)
        legacy.set_default_size(600, 800)
        legacy.present()
        self._settle()
        self.assertGreaterEqual(legacy.get_width(), 720)
        self.assertFalse(legacy._section_nav.get_visible())
        self.assertTrue(legacy._camera_flow_child.get_visible())
        self.assertTrue(legacy._audio_flow_child.get_visible())

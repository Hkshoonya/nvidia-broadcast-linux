# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0-or-later - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Slider scroll regressions; native input requires explicit opt-in under Xvfb."""

import ctypes
import ctypes.util
import os
import time
import unittest
from unittest import mock

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from nvbroadcast.ui.controls import EffectSlider


@unittest.skipUnless(Gtk.init_check(), "requires a GTK display")
class EffectSliderScrollTests(unittest.TestCase):
    """Controller checks create no windows on the user's desktop."""

    def test_scale_disables_only_its_scroll_controllers(self):
        slider = EffectSlider("Strength", 0.75)
        plain_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1, 0.05)

        scroll = []
        other = []
        for controller in list(slider._scale.observe_controllers()):
            if isinstance(controller, Gtk.EventControllerScroll):
                scroll.append(controller)
            else:
                other.append((type(controller), controller.get_propagation_phase()))
        expected_other = [
            (type(controller), controller.get_propagation_phase())
            for controller in list(plain_scale.observe_controllers())
            if not isinstance(controller, Gtk.EventControllerScroll)
        ]

        self.assertTrue(scroll, "GTK must expose the range's scroll controller")
        self.assertTrue(all(
            controller.get_propagation_phase() == Gtk.PropagationPhase.NONE
            for controller in scroll
        ))
        self.assertCountEqual(other, expected_other)

    def test_explicit_value_changes_still_emit(self):
        slider = EffectSlider("Strength", 0.75)
        changed = mock.Mock()
        slider.connect("value-changed", changed)

        slider._scale.set_value(0.25)

        changed.assert_called_once()
        self.assertEqual(slider.value, 0.25)


@unittest.skipUnless(
    os.environ.get("NVBROADCAST_TEST_NATIVE_INPUT") == "1",
    "run with NVBROADCAST_TEST_NATIVE_INPUT=1 and GDK_BACKEND=x11 under Xvfb",
)
class EffectSliderNativeInputTests(unittest.TestCase):
    """Send XTest input only when an isolated Xvfb run was explicitly requested."""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("GDK_BACKEND") != "x11" or not Gtk.init_check():
            raise unittest.SkipTest("requires the X11 backend under Xvfb")
        gi.require_version("GdkX11", "4.0")
        from gi.repository import GdkX11

        cls.GdkX11 = GdkX11
        x11 = ctypes.util.find_library("X11")
        xtst = ctypes.util.find_library("Xtst")
        if not x11 or not xtst:
            raise unittest.SkipTest("requires libX11 and libXtst")
        cls.x = ctypes.CDLL(x11)
        cls.xt = ctypes.CDLL(xtst)
        cls.x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        cls.x.XOpenDisplay.restype = ctypes.c_void_p
        cls.x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        cls.x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        cls.x.XDefaultRootWindow.restype = ctypes.c_ulong
        cls.x.XTranslateCoordinates.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
        ]
        cls.x.XSetInputFocus.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong,
        ]
        cls.x.XFlush.argtypes = [ctypes.c_void_p]
        cls.x.XStringToKeysym.argtypes = [ctypes.c_char_p]
        cls.x.XStringToKeysym.restype = ctypes.c_ulong
        cls.x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        cls.x.XKeysymToKeycode.restype = ctypes.c_uint
        cls.xt.XTestFakeMotionEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong,
        ]
        cls.xt.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong,
        ]
        cls.xt.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong,
        ]
        cls.display = cls.x.XOpenDisplay(os.environ["DISPLAY"].encode())
        if not cls.display:
            raise unittest.SkipTest("cannot open the isolated X11 display")

    @classmethod
    def tearDownClass(cls):
        cls.x.XCloseDisplay(cls.display)

    @staticmethod
    def _spin(seconds=0.08):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            while GLib.MainContext.default().pending():
                GLib.MainContext.default().iteration(False)
            time.sleep(0.002)

    def setUp(self):
        self.window = Gtk.Window(title="Isolated EffectSlider input regression")
        self.window.set_default_size(640, 420)
        self.addCleanup(self.window.destroy)
        self.scrolled = Gtk.ScrolledWindow()
        self.scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.box.set_size_request(1100, -1)
        spacer = Gtk.Box()
        spacer.set_size_request(-1, 300)
        self.box.append(spacer)
        self.slider = EffectSlider("Strength", 0.5)
        self.box.append(self.slider)
        end = Gtk.Box()
        end.set_size_request(-1, 1400)
        self.box.append(end)
        self.scrolled.set_child(self.box)
        self.window.set_child(self.scrolled)
        self.window.present()
        self._spin(0.3)
        self.va = self.scrolled.get_vadjustment()
        self.ha = self.scrolled.get_hadjustment()
        self.xid = self.GdkX11.X11Surface.get_xid(self.window.get_surface())
        self.x.XSetInputFocus(self.display, self.xid, 2, 0)
        self.x.XFlush(self.display)
        self.ox, self.oy = ctypes.c_int(), ctypes.c_int()
        child = ctypes.c_ulong()
        self.assertTrue(self.x.XTranslateCoordinates(
            self.display, self.xid, self.x.XDefaultRootWindow(self.display), 0, 0,
            ctypes.byref(self.ox), ctypes.byref(self.oy), ctypes.byref(child),
        ))
        self.changed = mock.Mock()
        self.slider.connect("value-changed", self.changed)

    def _motion(self, on_scale=True, fraction=0.4):
        if on_scale:
            ok, bounds = self.slider._scale.compute_bounds(self.window)
            self.assertTrue(ok)
            x = bounds.get_x() + (bounds.get_width() - 50) * fraction
            y = bounds.get_y() + bounds.get_height() / 2
        else:
            x, y = 100, 50
        self.assertTrue(self.xt.XTestFakeMotionEvent(
            self.display, -1, int(self.ox.value + x), int(self.oy.value + y), 0,
        ))
        self.x.XFlush(self.display)
        self._spin()

    def _button(self, number, pressed):
        self.assertTrue(self.xt.XTestFakeButtonEvent(
            self.display, number, int(pressed), 0,
        ))
        self.x.XFlush(self.display)
        self._spin()

    def _key(self, name, pressed):
        keycode = self.x.XKeysymToKeycode(
            self.display, self.x.XStringToKeysym(name.encode()),
        )
        self.assertTrue(keycode)
        self.assertTrue(self.xt.XTestFakeKeyEvent(
            self.display, keycode, int(pressed), 0,
        ))
        self.x.XFlush(self.display)
        self._spin()

    def _wheel(self, button, on_scale, shift=False):
        self.va.set_value(180)
        self.ha.set_value(100)
        self.slider._scale.set_value(0.5)
        self._spin()
        self.changed.reset_mock()
        self._motion(on_scale)
        if shift:
            self._key("Shift_L", True)
        try:
            self._button(button, True)
            self._button(button, False)
        finally:
            if shift:
                self._key("Shift_L", False)
        self.assertEqual(self.slider.value, 0.5)
        self.changed.assert_not_called()
        return self.ha.get_value() - 100, self.va.get_value() - 180

    def test_wheel_axes_and_shift_match_scrolling_over_the_page(self):
        for button, shift, axis, direction in (
            (4, False, 1, -1), (5, False, 1, 1),
            (6, False, 0, -1), (7, False, 0, 1),
            (4, True, 0, -1), (5, True, 0, 1),
            (6, True, 1, -1), (7, True, 1, 1),
        ):
            with self.subTest(button=button, shift=shift):
                expected = self._wheel(button, on_scale=False, shift=shift)
                actual = self._wheel(button, on_scale=True, shift=shift)
                self.assertGreater(expected[axis] * direction, 0)
                for got, want in zip(actual, expected):
                    self.assertAlmostEqual(got, want)

    def test_non_scrollable_inner_window_allows_outer_scrolling(self):
        self.box.remove(self.slider)
        inner = Gtk.ScrolledWindow()
        inner.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        inner.set_min_content_height(80)
        inner.set_max_content_height(80)
        inner.set_child(self.slider)
        self.box.insert_child_after(inner, self.box.get_first_child())
        self._spin()

        _dx, dy = self._wheel(5, on_scale=True)

        self.assertGreater(dy, 0)
        self.assertEqual(inner.get_vadjustment().get_value(), 0)

    def test_wheel_without_a_scroller_does_not_edit_the_value(self):
        self.box.remove(self.slider)
        self.window.set_child(self.slider)
        self._spin()

        self.assertEqual(self._wheel(5, on_scale=True), (0, 0))

    def test_keyboard_still_changes_the_value(self):
        self.assertTrue(self.slider._scale.grab_focus())
        self._key("Right", True)
        self._key("Right", False)

        self.assertAlmostEqual(self.slider.value, 0.55)
        self.changed.assert_called_once()

    def test_drag_still_changes_the_value(self):
        self.va.set_value(180)
        self.ha.set_value(100)
        self._spin()
        self._motion(fraction=0.5)
        self._button(1, True)
        try:
            self._motion(fraction=0.35)
        finally:
            self._button(1, False)

        self.assertNotEqual(self.slider.value, 0.5)
        self.assertTrue(self.changed.called)


if __name__ == "__main__":
    unittest.main()

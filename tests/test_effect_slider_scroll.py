# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0-or-later - see LICENSE file
# Original author: doczeus | AI Powered
#
"""The mouse wheel must scroll the page, never change a slider underneath it."""

import unittest
from unittest import mock

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, Gtk

from nvbroadcast.ui.controls import EffectSlider


@unittest.skipUnless(Gtk.init_check(), "requires a GTK display")
class EffectSliderScrollTests(unittest.TestCase):
    """No window is created, so nothing appears on the user's desktop."""

    def _slider_in_scrolled_window(self):
        slider = EffectSlider("Strength", 0.75)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.append(slider)
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_child(box)
        scrolled.get_vadjustment().configure(100, 0, 2000, 10, 300, 300)
        return slider, scrolled

    @staticmethod
    def _scroll(unit=Gdk.ScrollUnit.WHEEL):
        controller = mock.Mock()
        controller.get_unit.return_value = unit
        return controller

    def test_scale_intercepts_scroll_before_its_own_handler(self):
        slider = EffectSlider("Strength", 0.75)

        controllers = [
            controller
            for controller in list(slider._scale.observe_controllers())
            if isinstance(controller, Gtk.EventControllerScroll)
            and controller.get_propagation_phase() == Gtk.PropagationPhase.CAPTURE
        ]

        self.assertEqual(len(controllers), 1)

    def test_wheel_does_not_change_value_or_emit(self):
        slider, _scrolled = self._slider_in_scrolled_window()
        changed = mock.Mock()
        slider.connect("value-changed", changed)

        handled = slider._on_scroll(self._scroll(), 0.0, 1.0)

        self.assertTrue(handled)
        self.assertEqual(slider.value, 0.75)
        changed.assert_not_called()

    def test_wheel_scrolls_the_page_in_both_directions(self):
        slider, scrolled = self._slider_in_scrolled_window()
        adjustment = scrolled.get_vadjustment()

        slider._on_scroll(self._scroll(), 0.0, 1.0)
        after_down = adjustment.get_value()
        self.assertGreater(after_down, 100)

        slider._on_scroll(self._scroll(), 0.0, -2.0)
        self.assertLess(adjustment.get_value(), after_down)

    def test_touchpad_scrolls_by_exact_pixels(self):
        slider, scrolled = self._slider_in_scrolled_window()

        slider._on_scroll(self._scroll(Gdk.ScrollUnit.SURFACE), 0.0, 12.0)

        self.assertEqual(scrolled.get_vadjustment().get_value(), 112)

    def test_slider_outside_a_scrolled_window_still_ignores_the_wheel(self):
        slider = EffectSlider("Strength", 0.75)

        self.assertTrue(slider._on_scroll(self._scroll(), 0.0, 1.0))
        self.assertEqual(slider.value, 0.75)

    def test_dragging_still_changes_the_value(self):
        slider = EffectSlider("Strength", 0.75)
        changed = mock.Mock()
        slider.connect("value-changed", changed)

        slider._scale.set_value(0.25)

        changed.assert_called_once()
        self.assertEqual(slider.value, 0.25)


if __name__ == "__main__":
    unittest.main()

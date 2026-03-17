# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Video preview widget using Gdk.Texture rendering."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, Gdk


class VideoPreview(Gtk.Box):
    """Widget that displays video frames as Gdk.Textures."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.add_css_class("video-preview")

        self._picture = Gtk.Picture()
        self._picture.set_hexpand(True)
        self._picture.set_vexpand(True)
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.append(self._picture)

    def update_texture(self, texture: Gdk.Texture):
        """Update the displayed frame with a new texture."""
        self._picture.set_paintable(texture)

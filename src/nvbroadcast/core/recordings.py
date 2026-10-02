"""User-visible locations for ordinary camera recordings."""

import os
from pathlib import Path

from gi.repository import GLib


def recordings_directory() -> Path:
    """Honor the desktop Videos directory, including Snap's host mapping."""
    directory = GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_VIDEOS)
    if directory and Path(directory).is_absolute():
        return Path(directory)
    # Snap remaps HOME to a revision-specific private directory. Never use it
    # as the default destination for files the user expects in their home.
    real_home = os.environ.get("SNAP_REAL_HOME") if os.environ.get("SNAP") else None
    home = Path(real_home) if real_home and Path(real_home).is_absolute() else Path.home()
    return home / "Videos"


def legacy_recordings_directory() -> Path | None:
    """Keep recordings from older Snap versions reachable without moving them."""
    if os.environ.get("SNAP"):
        old_directory = Path.home() / "Videos"
        if old_directory != recordings_directory() and old_directory.is_dir():
            return old_directory
    return None

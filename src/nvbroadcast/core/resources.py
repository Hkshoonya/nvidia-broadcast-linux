"""Runtime resource lookup helpers for source and installed layouts."""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path


APP_ICON = "com.doczeus.NVBroadcast.svg"
DEFAULT_BACKGROUND = "studio_bg.png"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_ui_css() -> Path | None:
    try:
        css = resources.files("nvbroadcast.ui").joinpath("style.css")
        if css.is_file():
            return Path(css)
    except Exception:
        pass
    return _existing([
        PROJECT_ROOT / "src" / "nvbroadcast" / "ui" / "style.css",
    ])


def find_app_icon() -> Path | None:
    share_candidates = [
        Path(sys.prefix) / "share" / "icons" / "hicolor" / "scalable" / "apps" / APP_ICON,
        Path.home() / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps" / APP_ICON,
        Path("/usr/local/share/icons/hicolor/scalable/apps") / APP_ICON,
        Path("/usr/share/icons/hicolor/scalable/apps") / APP_ICON,
        PROJECT_ROOT / "data" / "icons" / APP_ICON,
    ]
    return _existing(share_candidates)


def find_backgrounds_dir() -> Path | None:
    share_candidates = [
        Path(sys.prefix) / "share" / "nvbroadcast" / "backgrounds",
        Path.home() / ".local" / "share" / "nvbroadcast" / "backgrounds",
        Path("/usr/local/share/nvbroadcast/backgrounds"),
        Path("/usr/share/nvbroadcast/backgrounds"),
        PROJECT_ROOT / "data" / "backgrounds",
    ]
    return _existing(share_candidates)


def find_bundled_backgrounds() -> list[Path]:
    bg_dir = find_backgrounds_dir()
    if bg_dir is None:
        return []

    backgrounds = sorted(bg_dir.glob("*.png"))
    if not backgrounds:
        return []

    defaults = [path for path in backgrounds if path.name == DEFAULT_BACKGROUND]
    others = [path for path in backgrounds if path.name != DEFAULT_BACKGROUND]
    return defaults + others

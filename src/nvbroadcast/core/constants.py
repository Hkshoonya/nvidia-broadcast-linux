# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Application constants."""

import os
from pathlib import Path

APP_ID = "com.doczeus.NVBroadcast"
APP_NAME = "NV Broadcast"
APP_SUBTITLE = "by doczeus | AI Powered"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

import platform as _pf
VIRTUAL_CAM_DEVICE = "/dev/video10" if _pf.system() != "Darwin" else "pyvirtualcam"
VIRTUAL_CAM_LABEL = "NVIDIA Broadcast"

if _pf.system() == "Darwin":
    CONFIG_DIR = Path.home() / "Library" / "Application Support" / "nvbroadcast"
else:
    CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "nvbroadcast"
CONFIG_FILE = CONFIG_DIR / "config.toml"

MAXINE_VFX_PATH = Path("/usr/local/VideoFX")
MAXINE_AFX_PATH = Path("/usr/local/AudioFX")
MAXINE_AR_PATH = Path("/usr/local/ARFX")

COMPUTE_GPU_INDEX = 0  # RTX 5060 (Blackwell)

# NVIDIA Brand Colors
NVIDIA_GREEN = "#76b900"
NVIDIA_DARK_BG = "#1a1a1a"
NVIDIA_CARD_BG = "#2a2a2a"
NVIDIA_TEXT = "#e0e0e0"

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""V4L2 loopback virtual camera management."""

import subprocess
import os
from pathlib import Path

from nvbroadcast.core.constants import VIRTUAL_CAM_DEVICE, VIRTUAL_CAM_LABEL


def is_v4l2loopback_loaded() -> bool:
    """Check if v4l2loopback kernel module is loaded."""
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, check=True
        )
        return "v4l2loopback" in result.stdout
    except subprocess.CalledProcessError:
        return False


def get_virtual_camera_device() -> str | None:
    """Find existing v4l2loopback device, or return None."""
    if os.path.exists(VIRTUAL_CAM_DEVICE):
        return VIRTUAL_CAM_DEVICE

    # Search for any v4l2loopback device
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.split("\n")
        for i, line in enumerate(lines):
            if "v4l2loopback" in line.lower() or "nvbroadcast" in line.lower():
                # Next line contains the device path
                if i + 1 < len(lines):
                    dev = lines[i + 1].strip()
                    if dev.startswith("/dev/video"):
                        return dev
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def ensure_virtual_camera() -> str:
    """Ensure a v4l2loopback device exists and return its path.

    If no device exists, attempts to create one (requires sudo).
    """
    device = get_virtual_camera_device()
    if device:
        return device

    if not is_v4l2loopback_loaded():
        raise RuntimeError(
            "v4l2loopback kernel module is not loaded.\n"
            "Install it with: sudo apt install v4l2loopback-dkms\n"
            "Load it with: sudo modprobe v4l2loopback "
            f'devices=1 video_nr=10 card_label="{VIRTUAL_CAM_LABEL}" '
            "exclusive_caps=1 max_buffers=4"
        )

    raise RuntimeError(
        f"v4l2loopback is loaded but no device found at {VIRTUAL_CAM_DEVICE}.\n"
        "Try: sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback "
        f'devices=1 video_nr=10 card_label="{VIRTUAL_CAM_LABEL}" exclusive_caps=1 max_buffers=4'
    )


def list_camera_devices() -> list[dict[str, str]]:
    """List available physical camera devices (one per physical camera)."""
    cameras = []
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.split("\n")
        current_name = ""
        first_dev_in_group = True
        for line in lines:
            if line and not line.startswith("\t") and not line.startswith(" "):
                current_name = line.rstrip(":")
                first_dev_in_group = True
            elif line.strip().startswith("/dev/video"):
                dev = line.strip()
                # Skip v4l2loopback devices and take only first device per camera
                if first_dev_in_group and "v4l2loopback" not in current_name.lower() and "nvbroadcast" not in current_name.lower():
                    cameras.append({"name": current_name, "device": dev})
                first_dev_in_group = False
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return cameras

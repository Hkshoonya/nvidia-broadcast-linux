# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Virtual camera management — v4l2loopback (Linux) / pyvirtualcam (macOS)."""

import subprocess
import os
from functools import lru_cache
from pathlib import Path

from nvbroadcast.core.constants import VIRTUAL_CAM_DEVICE, VIRTUAL_CAM_LABEL
from nvbroadcast.core.platform import IS_LINUX, IS_MACOS


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
    """Ensure a virtual camera device exists and return its path/identifier.

    Linux: v4l2loopback device at /dev/video10
    macOS: pyvirtualcam (OBS Virtual Camera) — returns "pyvirtualcam" sentinel
    """
    if IS_MACOS:
        try:
            import pyvirtualcam  # noqa: F401
            return "pyvirtualcam"
        except ImportError:
            raise RuntimeError(
                "Virtual camera requires pyvirtualcam + OBS on macOS.\n"
                "Install: pip install pyvirtualcam\n"
                "Also install OBS Studio: brew install --cask obs"
            )

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


def get_firefox_profiles() -> list[str]:
    """Find all Firefox profile directories (regular, snap, flatpak, macOS)."""
    from nvbroadcast.core.platform import get_firefox_profile_dirs
    profiles = []
    for base in get_firefox_profile_dirs():
        base = Path(base)
        if base.is_dir():
            for p in base.iterdir():
                if (p / "prefs.js").exists():
                    profiles.append(str(p))
    return profiles


def is_firefox_pipewire_disabled() -> bool | None:
    """Check if Firefox PipeWire camera is disabled. None = no Firefox found."""
    profiles = get_firefox_profiles()
    if not profiles:
        return None
    for prof in profiles:
        # Check user.js first (overrides prefs.js)
        user_js = os.path.join(prof, "user.js")
        if os.path.exists(user_js):
            with open(user_js) as f:
                content = f.read()
                if "allow-pipewire" in content and "false" in content:
                    return True
        # Check prefs.js
        prefs_js = os.path.join(prof, "prefs.js")
        if os.path.exists(prefs_js):
            with open(prefs_js) as f:
                content = f.read()
                if 'allow-pipewire", false' in content:
                    return True
    return False


def set_firefox_pipewire(disabled: bool) -> tuple[bool, str]:
    """Enable/disable PipeWire camera in Firefox for v4l2loopback compatibility.

    Writes to user.js in ALL Firefox profiles. Firefox must be restarted.
    Returns (success, message).
    """
    profiles = get_firefox_profiles()
    if not profiles:
        return False, "No Firefox profiles found"

    line = f'user_pref("media.webrtc.camera.allow-pipewire", {str(not disabled).lower()});\n'
    updated = 0

    for prof in profiles:
        user_js = os.path.join(prof, "user.js")
        try:
            # Read existing user.js
            existing = ""
            if os.path.exists(user_js):
                with open(user_js) as f:
                    existing = f.read()

            # Remove old pipewire lines
            lines = [l for l in existing.splitlines()
                     if "allow-pipewire" not in l]
            lines.append(line.strip())

            with open(user_js, "w") as f:
                f.write("\n".join(lines) + "\n")
            updated += 1
        except Exception:
            pass

    if updated == 0:
        return False, "Could not write to any Firefox profile"

    action = "disabled" if disabled else "enabled"
    return True, f"PipeWire camera {action} in {updated} profile(s). Restart Firefox to apply."


def reset_virtual_camera() -> bool:
    """Reset v4l2loopback device to accept new format/resolution.

    Needed when changing output format (YUY2/I420/NV12) or resolution,
    because v4l2loopback with exclusive_caps=1 locks the format after
    the first producer writes. Close all consumers (browsers) first.
    """
    try:
        subprocess.run(
            ["sudo", "modprobe", "-r", "v4l2loopback"],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["sudo", "modprobe", "v4l2loopback",
             "devices=1", "video_nr=10",
             f'card_label={VIRTUAL_CAM_LABEL}',
             "exclusive_caps=1", "max_buffers=4"],
            capture_output=True, timeout=5,
        )
        return os.path.exists(VIRTUAL_CAM_DEVICE)
    except Exception:
        return False


@lru_cache(maxsize=8)
def list_camera_modes(device: str = "/dev/video0") -> list[dict]:
    """Query camera supported resolutions and FPS in MJPEG mode.

    Returns list of {"width": int, "height": int, "fps": [int, ...]} sorted by resolution.
    """
    modes = {}
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True, text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return []
        in_mjpg = False
        current_res = None
        for line in result.stdout.split("\n"):
            stripped = line.strip()
            if "'MJPG'" in stripped:
                in_mjpg = True
                continue
            if in_mjpg and stripped.startswith("["):
                in_mjpg = False
                continue
            if in_mjpg and stripped.startswith("Size: Discrete"):
                res = stripped.split("Discrete")[1].strip()
                w, h = res.split("x")
                current_res = (int(w), int(h))
                if current_res not in modes:
                    modes[current_res] = []
            if in_mjpg and current_res and "fps" in stripped:
                # e.g. "Interval: Discrete 0.017s (60.000 fps)"
                fps_str = stripped.split("(")[1].split(" fps")[0]
                modes[current_res].append(int(float(fps_str)))
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    result = []
    for (w, h), fps_list in sorted(modes.items(), key=lambda x: x[0][0] * x[0][1]):
        result.append({"width": w, "height": h, "fps": sorted(set(fps_list))})
    return result


def list_camera_devices() -> list[dict[str, str]]:
    """List available physical camera devices (one per physical camera).

    Linux: uses v4l2-ctl
    macOS: uses system_profiler
    """
    if IS_MACOS:
        from nvbroadcast.core.platform import list_cameras_macos
        return list_cameras_macos()

    cameras = []
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.split("\n")
        current_name = ""
        is_loopback = False
        first_dev_in_group = True
        for line in lines:
            if line and not line.startswith("\t") and not line.startswith(" "):
                current_name = line.rstrip(":")
                is_loopback = False
                first_dev_in_group = True
            elif line.strip() and not line.strip().startswith("/dev/"):
                # Continuation line (e.g. "  Broadcast (platform:v4l2loopback-010):")
                cont = line.strip().rstrip(":")
                if "v4l2loopback" in cont.lower():
                    is_loopback = True
                current_name = f"{current_name} {cont}".strip()
            elif line.strip().startswith("/dev/video"):
                dev = line.strip()
                # Skip v4l2loopback devices and take only first device per camera
                if first_dev_in_group and not is_loopback and "nvidia broadcast" not in current_name.lower():
                    cameras.append({"name": current_name, "device": dev})
                first_dev_in_group = False
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return cameras

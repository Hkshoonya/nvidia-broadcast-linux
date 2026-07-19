# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Virtual camera management — v4l2loopback (Linux) / CoreMediaIO (macOS)."""

import subprocess
import os
import re
from functools import lru_cache
from pathlib import Path

from nvbroadcast.core.constants import VIRTUAL_CAM_DEVICE, VIRTUAL_CAM_LABEL
from nvbroadcast.core.platform import IS_LINUX, IS_MACOS

_MJPEG_FORMATS = {"MJPG", "JPEG"}
_RAW_FORMATS = {
    "YUYV", "YUY2", "UYVY", "YVYU", "VYUY",
    "NV12", "NV21", "YU12", "YV12", "I420",
    "RGB3", "BGR3", "RGB4", "BGR4",
}
_VIRTUAL_CAMERA_MARKERS = (
    "v4l2loopback",
    "nvidia broadcast",
    "nvbroadcast",
    "obs virtual camera",
)
_SAFE_FPS = (30, 25, 24, 15)
_STEPWISE_RESOLUTION_CANDIDATES = (
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (800, 600),
    (640, 480),
    (640, 360),
    (320, 240),
)


def _is_virtual_camera_name(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _VIRTUAL_CAMERA_MARKERS)


def _is_mjpeg_format(fmt: str) -> bool:
    return fmt.upper() in _MJPEG_FORMATS


def _is_raw_format(fmt: str) -> bool:
    return fmt.upper() in _RAW_FORMATS


def _is_supported_camera_format(fmt: str) -> bool:
    return _is_mjpeg_format(fmt) or _is_raw_format(fmt)


def _camera_format_name(fmt: str) -> str:
    return "mjpeg" if _is_mjpeg_format(fmt) else "raw"


def _camera_format_priority(fmt: str) -> int:
    """Prefer compressed capture first, then raw fallback."""
    return 0 if _is_mjpeg_format(fmt) else 1


def _fps_sort_key(candidate_fps: int, desired_fps: int) -> tuple[int, int, int]:
    """Sort by closest FPS, preferring lower stable rates on ties."""
    return (
        abs(candidate_fps - desired_fps),
        1 if candidate_fps > desired_fps else 0,
        -candidate_fps,
    )


def _parse_interval_fps_values(line: str) -> list[int]:
    """Parse FPS values from V4L2 interval lines.

    Most devices print `(30.000 fps)`. Some virtual/phone webcams print a
    stepwise seconds range instead; in that case, keep common stable rates that
    fit the reported range.
    """
    if "fps" in line and "(" in line:
        try:
            fps_str = line.split("(", 1)[1].split(" fps", 1)[0]
            return [int(float(fps_str))]
        except (IndexError, ValueError):
            return []

    interval_range = line.split("with step", 1)[0]
    seconds = [
        float(value)
        for value in re.findall(r"(\d+(?:\.\d+)?)s", interval_range)
    ]
    if not seconds:
        return []
    if len(seconds) == 1:
        if seconds[0] <= 0:
            return []
        return [max(1, int(round(1 / seconds[0])))]

    min_seconds = min(seconds)
    max_seconds = max(seconds)
    if min_seconds <= 0 or max_seconds <= 0:
        return []

    min_fps = int(round(1 / max_seconds))
    max_fps = int(round(1 / min_seconds))
    values = {
        fps for fps in (60, *_SAFE_FPS)
        if min_fps <= fps <= max_fps
    }
    values.update({min_fps, max_fps})
    return sorted(value for value in values if value > 0)


def _stepwise_resolutions(line: str) -> list[tuple[int, int]]:
    """Return conservative candidate resolutions from a V4L2 stepwise range."""
    pairs = [
        (int(width), int(height))
        for width, height in re.findall(r"(\d+)x(\d+)", line)
    ]
    if len(pairs) < 2:
        return pairs

    min_width = min(width for width, _height in pairs)
    max_width = max(width for width, _height in pairs)
    min_height = min(height for _width, height in pairs)
    max_height = max(height for _width, height in pairs)

    candidates = set(pairs)
    for width, height in _STEPWISE_RESOLUTION_CANDIDATES:
        if min_width <= width <= max_width and min_height <= height <= max_height:
            candidates.add((width, height))

    return sorted(candidates, key=lambda item: item[0] * item[1])


def is_v4l2loopback_loaded() -> bool:
    """Check if v4l2loopback kernel module is loaded."""
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, check=True
        )
        return "v4l2loopback" in result.stdout
    except subprocess.CalledProcessError:
        return False


def _video_nr_from_device(device: str | None) -> int | None:
    """Return the numeric v4l2 video index for `/dev/videoN` paths."""
    if not device:
        return None
    match = re.fullmatch(r"/dev/video(\d+)", device.strip())
    if match is None:
        return None
    return int(match.group(1))


def _video_nr_for_device(device: str | None) -> int:
    selected = _video_nr_from_device(device)
    if selected is not None:
        return selected
    default = _video_nr_from_device(VIRTUAL_CAM_DEVICE)
    return default if default is not None else 10


def _v4l2loopback_modprobe_command(device: str | None = None) -> str:
    video_nr = _video_nr_for_device(device)
    return (
        "sudo modprobe v4l2loopback "
        f'devices=1 video_nr={video_nr} card_label="{VIRTUAL_CAM_LABEL}" '
        "exclusive_caps=1 max_buffers=4"
    )


def v4l2loopback_modprobe_command(device: str | None = None) -> str:
    """Return the modprobe command for the selected virtual camera device."""
    return _v4l2loopback_modprobe_command(device)


def is_v4l2loopback_device(device: str) -> bool:
    """Return whether an existing V4L2 node is backed by v4l2loopback."""
    info = _get_v4l2_device_info(device)
    return bool(
        re.search(
            r"^\s*Driver name\s*:\s*v4l2[\s_-]*loopback\s*$",
            info,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )


def get_virtual_camera_device(preferred_device: str | None = None) -> str | None:
    """Find existing v4l2loopback device, or return None."""
    preferred = (preferred_device or "").strip()
    if preferred:
        if os.path.exists(preferred) and is_v4l2loopback_device(preferred):
            return preferred
        return None

    if (
        os.path.exists(VIRTUAL_CAM_DEVICE)
        and is_v4l2loopback_device(VIRTUAL_CAM_DEVICE)
    ):
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
            if _is_virtual_camera_name(line):
                # Next line contains the device path
                if i + 1 < len(lines):
                    dev = lines[i + 1].strip()
                    if (
                        dev.startswith("/dev/video")
                        and is_v4l2loopback_device(dev)
                    ):
                        return dev
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def ensure_virtual_camera(preferred_device: str | None = None) -> str:
    """Ensure a virtual camera device exists and return its path/identifier.

    Linux: v4l2loopback device, defaulting to /dev/video10
    macOS: proprietary CoreMediaIO extension — returns the branded device name
    """
    if IS_MACOS:
        return VIRTUAL_CAM_LABEL

    preferred = (preferred_device or "").strip()
    if preferred and _video_nr_from_device(preferred) is None:
        raise RuntimeError(
            "Virtual camera device must be a Linux video device path like /dev/video10."
        )
    if (
        preferred
        and os.path.exists(preferred)
        and not is_v4l2loopback_device(preferred)
    ):
        raise RuntimeError(
            f"{preferred} exists but is not a v4l2loopback virtual camera."
        )

    device = get_virtual_camera_device(preferred)
    if device:
        return device

    target_device = preferred or VIRTUAL_CAM_DEVICE
    modprobe_cmd = _v4l2loopback_modprobe_command(target_device)

    if not is_v4l2loopback_loaded():
        raise RuntimeError(
            "v4l2loopback kernel module is not loaded.\n"
            "Install it with: sudo apt install v4l2loopback-dkms\n"
            f"Load {target_device} with: {modprobe_cmd}"
        )

    raise RuntimeError(
        f"v4l2loopback is loaded but no device found at {target_device}.\n"
        f"Try: sudo modprobe -r v4l2loopback && {modprobe_cmd}"
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


def reset_virtual_camera(device: str | None = None) -> bool:
    """Reset v4l2loopback device to accept new format/resolution.

    Needed when changing output format (YUY2/I420/NV12) or resolution,
    because v4l2loopback with exclusive_caps=1 locks the format after
    the first producer writes. Close all consumers (browsers) first.
    """
    target_device = (device or VIRTUAL_CAM_DEVICE).strip()
    if _video_nr_from_device(target_device) is None:
        return False
    if (
        os.path.exists(target_device)
        and not is_v4l2loopback_device(target_device)
    ):
        return False
    video_nr = _video_nr_for_device(target_device)
    try:
        subprocess.run(
            ["sudo", "modprobe", "-r", "v4l2loopback"],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["sudo", "modprobe", "v4l2loopback",
             "devices=1", f"video_nr={video_nr}",
             f'card_label={VIRTUAL_CAM_LABEL}',
             "exclusive_caps=1", "max_buffers=4"],
            capture_output=True, timeout=5,
        )
        return os.path.exists(target_device)
    except Exception:
        return False


def _parse_v4l2_format_modes(output: str) -> list[dict]:
    """Parse `v4l2-ctl --list-formats-ext` into supported camera modes."""
    modes: dict[tuple[str, int, int], set[int]] = {}
    current_format = ""
    current_resolutions: list[tuple[int, int]] = []

    for line in output.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("[") and "]: '" in stripped:
            parts = stripped.split("'", 2)
            current_format = parts[1].upper() if len(parts) > 1 else ""
            current_resolutions = []
            continue

        if not current_format or not _is_supported_camera_format(current_format):
            continue

        if stripped.startswith("Size: Discrete"):
            try:
                res = stripped.split("Discrete", 1)[1].strip()
                width, height = res.split("x", 1)
                current_resolutions = [(int(width), int(height))]
                for current_res in current_resolutions:
                    modes.setdefault((current_format, *current_res), set())
            except (IndexError, ValueError):
                current_resolutions = []
            continue

        if stripped.startswith(("Size: Stepwise", "Size: Continuous")):
            current_resolutions = _stepwise_resolutions(stripped)
            for current_res in current_resolutions:
                modes.setdefault((current_format, *current_res), set())
            continue

        if current_resolutions and stripped.startswith("Interval:"):
            for fps in _parse_interval_fps_values(stripped):
                for current_res in current_resolutions:
                    modes[(current_format, *current_res)].add(fps)

    result = []
    for (fmt, width, height), fps_list in sorted(
        modes.items(), key=lambda item: item[0][1] * item[0][2]
    ):
        if fps_list:
            result.append({
                "format": fmt,
                "width": width,
                "height": height,
                "fps": sorted(fps_list),
            })
    return result


@lru_cache(maxsize=8)
def list_camera_format_modes(device: str = "/dev/video0") -> list[dict]:
    """Query camera supported resolutions, FPS, and input formats."""
    if IS_MACOS:
        return []

    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True, text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return []
        return _parse_v4l2_format_modes(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []


@lru_cache(maxsize=8)
def list_camera_modes(device: str = "/dev/video0") -> list[dict]:
    """Query supported camera resolutions and FPS.

    Returns list of {"width": int, "height": int, "fps": [int, ...]} sorted by resolution.
    """
    modes: dict[tuple[int, int], set[int]] = {}
    for mode in list_camera_format_modes(device):
        key = (mode["width"], mode["height"])
        fps_set = modes.setdefault(key, set())
        fps_set.update(mode["fps"])

    result = []
    for (w, h), fps_set in sorted(modes.items(), key=lambda x: x[0][0] * x[0][1]):
        result.append({"width": w, "height": h, "fps": sorted(fps_set)})
    return result


def camera_mode_candidates(
    device: str,
    width: int,
    height: int,
    fps: int,
) -> list[dict]:
    """Return ordered capture modes for a camera.

    The first candidate preserves the requested mode when it is supported. The
    rest are safer fallbacks for phone/virtual webcams that advertise unusual
    modes or fail at high FPS during GStreamer startup.
    """
    if IS_MACOS:
        return [{"format": "raw", "width": width, "height": height, "fps": fps}]

    format_modes = list_camera_format_modes(device)
    if not format_modes:
        # Preserve previous behavior when probing fails, but keep one raw retry
        # available for cameras that reject MJPEG at runtime.
        return [
            {"format": "mjpeg", "width": width, "height": height, "fps": fps},
            {"format": "raw", "width": width, "height": height, "fps": fps},
        ]

    expanded: list[dict] = []
    for mode in format_modes:
        for mode_fps in mode["fps"]:
            expanded.append({
                "format": _camera_format_name(mode["format"]),
                "format_code": mode["format"],
                "width": mode["width"],
                "height": mode["height"],
                "fps": mode_fps,
            })

    desired_area = max(1, width * height)

    def format_priority(mode: dict) -> int:
        return _camera_format_priority(mode.get("format_code", ""))

    groups: list[list[dict]] = []

    exact = [
        mode for mode in expanded
        if mode["width"] == width and mode["height"] == height and mode["fps"] == fps
    ]
    groups.append(sorted(exact, key=format_priority))

    same_resolution = [
        mode for mode in expanded
        if mode["width"] == width and mode["height"] == height and mode["fps"] != fps
    ]
    groups.append(sorted(
        same_resolution,
        key=lambda mode: (*_fps_sort_key(mode["fps"], fps), format_priority(mode)),
    ))

    for fallback_width, fallback_height, fallback_fps in (
        (1280, 720, 30),
        (640, 480, 30),
    ):
        fallback = [
            mode for mode in expanded
            if mode["width"] == fallback_width and mode["height"] == fallback_height
        ]
        groups.append(sorted(
            fallback,
            key=lambda mode: (
                *_fps_sort_key(mode["fps"], fallback_fps),
                format_priority(mode),
            ),
        ))

    def general_key(mode: dict) -> tuple[int, int, int, int, int, int]:
        area = mode["width"] * mode["height"]
        return (
            0 if area <= desired_area else 1,
            abs(area - desired_area),
            *_fps_sort_key(mode["fps"], fps),
            format_priority(mode),
        )

    groups.append(sorted(expanded, key=general_key))

    candidates: list[dict] = []
    seen: set[tuple[str, int, int, int]] = set()
    for group in groups:
        for mode in group:
            key = (mode["format"], mode["width"], mode["height"], mode["fps"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "format": mode["format"],
                "width": mode["width"],
                "height": mode["height"],
                "fps": mode["fps"],
            })

    return candidates


def select_camera_mode(
    device: str,
    width: int,
    height: int,
    fps: int,
) -> dict:
    """Return the best capture mode for a camera."""
    candidates = camera_mode_candidates(device, width, height, fps)
    if candidates:
        return candidates[0]
    return {"format": "mjpeg", "width": width, "height": height, "fps": fps}


def select_camera_capture_format(
    device: str,
    width: int,
    height: int,
    fps: int,
) -> str:
    """Return `mjpeg` or `raw` for the requested capture mode."""
    if IS_MACOS:
        return "raw"

    return select_camera_mode(device, width, height, fps)["format"]


@lru_cache(maxsize=32)
def _get_v4l2_device_info(device: str) -> str:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-D", "-d", device],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return ""
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _device_caps_text(info: str) -> str:
    marker = "Device Caps"
    if marker not in info:
        return info
    return info.split(marker, 1)[1]


def is_usable_camera_device(device: str, name: str = "") -> bool:
    """Return whether a V4L2 node is a usable physical capture camera."""
    if IS_MACOS:
        return bool(device or name)
    if not device.startswith("/dev/video"):
        return False
    if _is_virtual_camera_name(name):
        return False

    info = _get_v4l2_device_info(device)
    has_capture_info = False
    if info:
        if _is_virtual_camera_name(info):
            return False
        caps = _device_caps_text(info)
        if "Metadata Capture" in caps and "Video Capture" not in caps:
            return False
        if "Video Capture" not in caps:
            return False
        has_capture_info = True

    # Prefer cameras with known modes, but do not hide a real capture node when
    # a distro/kernel reports device info but refuses the formats query.
    return bool(list_camera_modes(device)) or has_capture_info


def resolve_camera_device(saved_device: str | None = None) -> str:
    """Return the saved camera if valid, otherwise the first usable camera."""
    if IS_MACOS:
        return saved_device or ""

    device = saved_device or ""
    if device and is_usable_camera_device(device):
        return device

    cameras = list_camera_devices()
    if cameras:
        return cameras[0]["device"]
    return device or "/dev/video0"


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
        if result.returncode != 0:
            return []
        lines = result.stdout.split("\n")
        current_name = ""
        is_loopback = False
        group_devices: list[str] = []

        def add_current_group():
            if not current_name or is_loopback or _is_virtual_camera_name(current_name):
                return
            for dev in group_devices:
                if is_usable_camera_device(dev, current_name):
                    cameras.append({"name": current_name, "device": dev})
                    break

        for line in lines:
            if line and not line.startswith("\t") and not line.startswith(" "):
                add_current_group()
                current_name = line.rstrip(":")
                is_loopback = _is_virtual_camera_name(current_name)
                group_devices = []
            elif line.strip() and not line.strip().startswith("/dev/"):
                # Continuation line (e.g. "  Broadcast (platform:v4l2loopback-010):")
                cont = line.strip().rstrip(":")
                if _is_virtual_camera_name(cont):
                    is_loopback = True
                current_name = f"{current_name} {cont}".strip()
            elif line.strip().startswith("/dev/video"):
                dev = line.strip()
                group_devices.append(dev)
        add_current_group()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return cameras

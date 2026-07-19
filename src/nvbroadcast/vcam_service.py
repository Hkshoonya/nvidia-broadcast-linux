# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Headless virtual camera service.

Runs the webcam -> effects -> v4l2loopback pipeline without a GUI,
keeping the virtual camera available for browsers and apps at all times.

Usage:
    nvbroadcast-vcam                  # Run with defaults
    nvbroadcast-vcam --device /dev/video0 --format yuy2
    nvbroadcast-vcam --format i420    # Better Firefox compatibility
"""

import signal
import sys
import argparse
import subprocess

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from nvbroadcast.core.constants import (
    VIRTUAL_CAM_DEVICE,
    VIRTUAL_CAM_LABEL,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FPS,
)
from nvbroadcast.core.config import load_config
from nvbroadcast.core.platform import get_gst_camera_caps
from nvbroadcast.video.virtual_camera import (
    camera_mode_candidates,
    ensure_virtual_camera,
    list_camera_devices,
    resolve_camera_device,
    select_camera_mode,
)


OUTPUT_FORMATS = {
    "yuy2": "YUY2",
    "i420": "I420",
    "yuv420": "I420",
    "nv12": "NV12",
}


def _strict_vcam_preference(device: str | None, explicit: bool = False) -> str | None:
    configured = (device or "").strip()
    if not configured:
        return None
    if explicit or configured != VIRTUAL_CAM_DEVICE:
        return configured
    return None


def build_pipeline(
    source_device: str,
    vcam_device: str,
    width: int,
    height: int,
    fps: int,
    output_format: str,
    capture_format: str | None = None,
) -> Gst.Pipeline:
    """Build a headless webcam -> v4l2loopback pipeline.

    Handles both MJPEG and raw camera sources automatically.
    Most USB cameras output MJPEG at HD resolutions and raw YUYV only at low res.
    """
    fmt = OUTPUT_FORMATS.get(output_format.lower(), "YUY2")

    if capture_format is None:
        selected_mode = select_camera_mode(source_device, width, height, fps)
        width = selected_mode["width"]
        height = selected_mode["height"]
        fps = selected_mode["fps"]
        capture_format = selected_mode["format"]

    camera_src = get_gst_camera_caps(
        source_device, width, height, fps, capture_format=capture_format
    )
    decoder = "jpegdec ! videoconvert" if capture_format == "mjpeg" else "videoconvert"

    pipeline_str = (
        f"{camera_src} ! "
        f"{decoder} ! "
        f"video/x-raw,format={fmt},width={width},height={height},framerate={fps}/1 ! "
        f"identity drop-allocation=true ! "
        f"v4l2sink device={vcam_device} io-mode=2 sync=false async=false"
    )

    print(f"[NVIDIA Broadcast VCam] Pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    except GLib.Error:
        pass

    alternate_format = "raw" if capture_format == "mjpeg" else "mjpeg"
    camera_src = get_gst_camera_caps(
        source_device, width, height, fps, capture_format=alternate_format
    )
    decoder = "jpegdec ! videoconvert" if alternate_format == "mjpeg" else "videoconvert"
    print(f"[NVIDIA Broadcast VCam] Trying {alternate_format} source fallback...")
    pipeline_str = (
        f"{camera_src} ! "
        f"{decoder} ! "
        f"video/x-raw,format={fmt},width={width},height={height},framerate={fps}/1 ! "
        f"identity drop-allocation=true ! "
        f"v4l2sink device={vcam_device} io-mode=2 sync=false async=false"
    )
    print(f"[NVIDIA Broadcast VCam] Pipeline: {pipeline_str}")
    pipeline = Gst.parse_launch(pipeline_str)
    return pipeline


def _start_pipeline_once(pipeline: Gst.Pipeline) -> bool:
    """Start a pipeline and catch immediate negotiation failures."""
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        return False

    try:
        state_ret, _state, _pending = pipeline.get_state(2 * Gst.SECOND)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            return False
    except Exception:
        # Some test doubles and platform backends do not expose get_state
        # cleanly. If set_state did not fail, keep previous permissive behavior.
        pass

    try:
        bus = pipeline.get_bus()
        message = bus.timed_pop_filtered(
            2 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.ASYNC_DONE,
        )
        if message and message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[NVIDIA Broadcast VCam] Startup error: {err.message}")
            if debug:
                print(f"[NVIDIA Broadcast VCam] Startup debug: {debug}")
            return False
    except Exception:
        pass

    return True


def _describe_vcam_device(device: str) -> str:
    """Return a short diagnostic string for a v4l2loopback device."""
    if not device.startswith("/dev/video"):
        return "non-v4l2"

    caps = "unknown"
    holders = "unknown"
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-D", "-d", device],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if "Video Output" in result.stdout:
            caps = "output"
        elif "Video Capture" in result.stdout:
            caps = "capture"
        elif result.returncode == 0:
            caps = "unreported"
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["fuser", "-v", device],
            capture_output=True,
            text=True,
            timeout=1,
        )
        merged = (result.stdout + " " + result.stderr).strip()
        holders = merged.replace("\n", " | ") if merged else "none"
    except Exception:
        pass

    return f"caps={caps}, holders={holders}"


def _vcam_ready_for_writer(device: str) -> bool:
    """Return whether the virtual camera can accept a writer pipeline."""
    if not device.startswith("/dev/video"):
        return True
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-D", "-d", device],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    if result.returncode != 0:
        return False
    return "Video Output" in result.stdout


def start_pipeline_with_fallback(
    source_device: str,
    vcam_device: str,
    width: int,
    height: int,
    fps: int,
    output_format: str,
) -> tuple[Gst.Pipeline | None, dict | None]:
    """Start the first working camera mode, retrying safer phone-webcam paths."""
    if not _vcam_ready_for_writer(vcam_device):
        print(
            "[NVIDIA Broadcast VCam] Virtual camera is busy or already opened "
            f"by another process ({_describe_vcam_device(vcam_device)})."
        )
        print(
            "[NVIDIA Broadcast VCam] Close video apps or stop the GUI/headless "
            "service before starting nvbroadcast-vcam."
        )
        return None, None

    candidates = camera_mode_candidates(source_device, width, height, fps)
    for index, mode in enumerate(candidates, start=1):
        if (
            mode["width"] != width
            or mode["height"] != height
            or mode["fps"] != fps
            or index > 1
        ):
            print(
                "[NVIDIA Broadcast VCam] Trying camera mode "
                f"{mode['width']}x{mode['height']}@{mode['fps']}fps "
                f"({mode['format']})"
            )

        pipeline = build_pipeline(
            source_device,
            vcam_device,
            mode["width"],
            mode["height"],
            mode["fps"],
            output_format,
            capture_format=mode["format"],
        )
        if _start_pipeline_once(pipeline):
            return pipeline, mode

        print(
            "[NVIDIA Broadcast VCam] Camera mode failed: "
            f"{mode['width']}x{mode['height']}@{mode['fps']}fps "
            f"({mode['format']})"
        )
        try:
            pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass

    return None, None


def on_bus_message(bus, message, loop):
    """Handle GStreamer bus messages."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[NVIDIA Broadcast VCam] End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[NVIDIA Broadcast VCam] Error: {err.message}")
        if debug:
            print(f"[NVIDIA Broadcast VCam] Debug: {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        print(f"[NVIDIA Broadcast VCam] Warning: {warn.message}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=f"{VIRTUAL_CAM_LABEL} Virtual Camera Service - keeps virtual camera available for apps"
    )
    parser.add_argument(
        "--device", "-d",
        help="Source camera device (default: auto-detect or from config)",
    )
    parser.add_argument(
        "--vcam",
        default=None,
        help=(
            "Virtual camera device "
            f"(default: config value, then {VIRTUAL_CAM_DEVICE})"
        ),
    )
    parser.add_argument(
        "--width", "-W", type=int, default=0,
        help=f"Video width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height", "-H", type=int, default=0,
        help=f"Video height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--fps", type=int, default=0,
        help=f"Frames per second (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--format", "-f",
        choices=list(OUTPUT_FORMATS.keys()),
        default="yuy2",
        help="Output pixel format (default: yuy2, use i420 for Firefox)",
    )
    args = parser.parse_args()

    Gst.init(None)

    # Load config for defaults
    config = load_config()
    source_device = resolve_camera_device(args.device or config.video.camera_device)
    width = args.width or config.video.width
    height = args.height or config.video.height
    fps = args.fps or config.video.fps
    requested_vcam = args.vcam or config.video.vcam_device

    # Auto-detect camera if not specified
    if not source_device or source_device == "/dev/video0":
        cameras = list_camera_devices()
        if cameras:
            source_device = cameras[0]["device"]
            print(f"[NVIDIA Broadcast VCam] Auto-detected camera: {cameras[0]['name']} ({source_device})")
        else:
            source_device = "/dev/video0"

    # Ensure virtual camera device exists
    try:
        vcam = ensure_virtual_camera(
            _strict_vcam_preference(requested_vcam, explicit=bool(args.vcam))
        )
        print(f"[NVIDIA Broadcast VCam] Virtual camera: {vcam}")
    except RuntimeError as e:
        print(f"[NVIDIA Broadcast VCam] Error: {e}", file=sys.stderr)
        sys.exit(1)

    vcam_device = vcam
    print(f"[NVIDIA Broadcast VCam] Source: {source_device} ({width}x{height}@{fps}fps)")
    print(f"[NVIDIA Broadcast VCam] Output: {vcam_device} (format: {args.format.upper()})")
    print(f"[NVIDIA Broadcast VCam] Virtual camera will be visible to browsers and apps")
    print()

    pipeline, active_mode = start_pipeline_with_fallback(
        source_device, vcam_device, width, height, fps, args.format
    )
    if pipeline is None or active_mode is None:
        print("[NVIDIA Broadcast VCam] Failed to start pipeline", file=sys.stderr)
        sys.exit(1)

    if (
        active_mode["width"] != width
        or active_mode["height"] != height
        or active_mode["fps"] != fps
    ):
        print(
            "[NVIDIA Broadcast VCam] Using compatible source mode: "
            f"{active_mode['width']}x{active_mode['height']}@"
            f"{active_mode['fps']}fps ({active_mode['format']})"
        )

    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    # Handle SIGINT/SIGTERM for clean shutdown
    def shutdown(signum, frame):
        print("\n[NVIDIA Broadcast VCam] Shutting down...")
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[NVIDIA Broadcast VCam] Streaming... (Ctrl+C to stop)")
    print(f"[NVIDIA Broadcast VCam] Open your browser or video app and select '{VIRTUAL_CAM_LABEL}'")

    try:
        loop.run()
    except Exception:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()

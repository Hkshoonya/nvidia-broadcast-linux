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

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from nvbroadcast.core.constants import VIRTUAL_CAM_DEVICE, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS
from nvbroadcast.core.config import load_config
from nvbroadcast.video.virtual_camera import ensure_virtual_camera, list_camera_devices


OUTPUT_FORMATS = {
    "yuy2": "YUY2",
    "i420": "I420",
    "yuv420": "I420",
    "nv12": "NV12",
}


def build_pipeline(
    source_device: str,
    vcam_device: str,
    width: int,
    height: int,
    fps: int,
    output_format: str,
) -> Gst.Pipeline:
    """Build a headless webcam -> v4l2loopback pipeline.

    Handles both MJPEG and raw camera sources automatically.
    Most USB cameras output MJPEG at HD resolutions and raw YUYV only at low res.
    """
    fmt = OUTPUT_FORMATS.get(output_format.lower(), "YUY2")

    # Try MJPEG source first (most cameras use this for HD),
    # fall back to raw if MJPEG isn't available
    pipeline_str = (
        f"v4l2src device={source_device} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw,format={fmt},width={width},height={height},framerate={fps}/1 ! "
        f"v4l2sink device={vcam_device}"
    )

    print(f"[NVIDIA Broadcast VCam] Pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        return pipeline
    except GLib.Error:
        pass

    # Fallback: raw source (lower resolution may be needed)
    print("[NVIDIA Broadcast VCam] MJPEG not available, trying raw source...")
    pipeline_str = (
        f"v4l2src device={source_device} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw,format={fmt},width={width},height={height},framerate={fps}/1 ! "
        f"v4l2sink device={vcam_device}"
    )
    print(f"[NVIDIA Broadcast VCam] Pipeline: {pipeline_str}")
    pipeline = Gst.parse_launch(pipeline_str)
    return pipeline


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
        description="NVIDIA Broadcast Virtual Camera Service - keeps virtual camera available for apps"
    )
    parser.add_argument(
        "--device", "-d",
        help="Source camera device (default: auto-detect or from config)",
    )
    parser.add_argument(
        "--vcam",
        default=VIRTUAL_CAM_DEVICE,
        help=f"Virtual camera device (default: {VIRTUAL_CAM_DEVICE})",
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
    source_device = args.device or config.video.camera_device
    width = args.width or config.video.width
    height = args.height or config.video.height
    fps = args.fps or config.video.fps

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
        vcam = ensure_virtual_camera()
        print(f"[NVIDIA Broadcast VCam] Virtual camera: {vcam}")
    except RuntimeError as e:
        print(f"[NVIDIA Broadcast VCam] Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[NVIDIA Broadcast VCam] Source: {source_device} ({width}x{height}@{fps}fps)")
    print(f"[NVIDIA Broadcast VCam] Output: {args.vcam} (format: {args.format.upper()})")
    print(f"[NVIDIA Broadcast VCam] Virtual camera will be visible to browsers and apps")
    print()

    pipeline = build_pipeline(source_device, args.vcam, width, height, fps, args.format)

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

    # Start pipeline
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[NVIDIA Broadcast VCam] Failed to start pipeline", file=sys.stderr)
        sys.exit(1)

    print("[NVIDIA Broadcast VCam] Streaming... (Ctrl+C to stop)")
    print("[NVIDIA Broadcast VCam] Open your browser or video app and select 'NVIDIA Broadcast Virtual Camera'")

    try:
        loop.run()
    except Exception:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Platform detection — abstracts OS differences for cross-platform support."""

import platform
import subprocess
import shutil
import ctypes
import ctypes.util

IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"
MACHINE = platform.machine().lower()
IS_ARM64 = MACHINE in ("arm64", "aarch64")
IS_X86_64 = MACHINE in ("x86_64", "amd64")


def linux_multiarch_triplet() -> str:
    """Return the Debian-style multiarch triplet for this machine."""
    if IS_ARM64:
        return "aarch64-linux-gnu"
    return "x86_64-linux-gnu"


def supports_linux_gpu_stack() -> bool:
    """Return whether the Linux CUDA/TensorRT optional stack is supported."""
    return IS_LINUX and IS_X86_64


def has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available."""
    if IS_MACOS:
        return False  # No NVIDIA on modern Macs
    if IS_LINUX and not supports_linux_gpu_stack():
        return False
    return shutil.which("nvidia-smi") is not None


def has_v4l2() -> bool:
    """Check if v4l2 tools are available (Linux only)."""
    if not IS_LINUX:
        return False
    return shutil.which("v4l2-ctl") is not None


def has_pyvirtualcam() -> bool:
    """Check if pyvirtualcam is available (cross-platform virtual camera)."""
    try:
        import pyvirtualcam  # noqa: F401
        return True
    except ImportError:
        return False


def has_tensorrt_runtime() -> bool:
    """Check whether TensorRT EP is actually runnable on this system."""
    if not supports_linux_gpu_stack():
        return False
    try:
        import onnxruntime as ort
    except Exception:
        return False

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        return False

    # Main runtime library must be loadable, otherwise ORT will advertise TRT
    # but still fail at session creation with a provider load error.
    if ctypes.util.find_library("nvinfer"):
        return True

    try:
        import importlib.util
        from pathlib import Path

        spec = importlib.util.find_spec("tensorrt_libs")
        if not spec or not spec.submodule_search_locations:
            return False

        lib_dir = Path(spec.submodule_search_locations[0])
        for so in sorted(lib_dir.glob("libnvinfer.so*")):
            try:
                ctypes.CDLL(str(so))
                return True
            except OSError:
                continue
    except Exception:
        return False

    return False


def get_default_camera_device() -> str:
    """Return the default camera device path/identifier for this OS."""
    if IS_LINUX:
        return "/dev/video0"
    if IS_MACOS:
        return ""  # macOS uses AVFoundation device index, not path
    return ""


def get_gst_camera_source() -> str:
    """Return the GStreamer camera source element for this OS."""
    if IS_MACOS:
        return "avfvideosrc"
    return "v4l2src"


def get_gst_camera_caps(device: str, width: int, height: int, fps: int) -> str:
    """Return the GStreamer camera source + caps string for this OS."""
    if IS_MACOS:
        # avfvideosrc outputs raw video directly (no MJPEG intermediate)
        dev_prop = f"device-index={device}" if device.isdigit() else ""
        return (
            f"avfvideosrc {dev_prop} ! "
            f"video/x-raw,width={width},height={height},"
            f"framerate={fps}/1"
        )
    # Linux: use MMAP + fresh timestamps so the live path favors newest frames.
    return (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        f"image/jpeg,width={width},height={height},"
        f"framerate={fps}/1"
    )


def get_onnx_providers(gpu_index: int = 0,
                       use_tensorrt: bool = False) -> list:
    """Return the ONNX Runtime execution providers for this platform."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    providers = []

    if supports_linux_gpu_stack() and use_tensorrt and 'TensorrtExecutionProvider' in available:
        from pathlib import Path
        cache_dir = str(Path(__file__).parent.parent.parent.parent / "models" / "trt_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        providers.append(('TensorrtExecutionProvider', {
            'device_id': gpu_index,
            'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
            'trt_builder_optimization_level': 3,
        }))

    if supports_linux_gpu_stack() and 'CUDAExecutionProvider' in available:
        providers.append(('CUDAExecutionProvider', {
            'device_id': gpu_index,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'HEURISTIC',
            'do_copy_in_default_stream': True,
        }))

    if IS_MACOS and 'CoreMLExecutionProvider' in available:
        providers.append('CoreMLExecutionProvider')

    providers.append('CPUExecutionProvider')
    return providers


def list_cameras_macos() -> list[dict[str, str]]:
    """List camera devices on macOS using system_profiler."""
    cameras = []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        )
        import json
        data = json.loads(result.stdout)
        for cam in data.get("SPCameraDataType", []):
            cameras.append({
                "name": cam.get("_name", "Camera"),
                "device": "0",  # avfvideosrc uses device-index
            })
    except Exception:
        # Fallback: assume at least one camera exists
        cameras.append({"name": "Default Camera", "device": "0"})
    return cameras


def get_firefox_profile_dirs() -> list[str]:
    """Return Firefox profile search dirs for this OS."""
    from pathlib import Path
    home = Path.home()
    if IS_MACOS:
        return [home / "Library" / "Application Support" / "Firefox" / "Profiles"]
    # Linux: regular, snap, flatpak
    return [
        home / ".mozilla" / "firefox",
        home / "snap" / "firefox" / "common" / ".mozilla" / "firefox",
        home / ".var" / "app" / "org.mozilla.firefox" / ".mozilla" / "firefox",
    ]

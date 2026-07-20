# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Platform detection — abstracts OS differences for cross-platform support."""

import os
import platform
import subprocess
import shutil
import sys
import ctypes
import ctypes.util
import importlib.metadata
from pathlib import Path

IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"
MACHINE = platform.machine().lower()
IS_ARM64 = MACHINE in ("arm64", "aarch64")
IS_X86_64 = MACHINE in ("x86_64", "amd64")
TENSORRT_LIB_MODULES = (
    "tensorrt_libs",
    "tensorrt_cu12_libs",
    "tensorrt_cu13_libs",
)
# NOTE: nvidia.cuda_nvrtc is intentionally NOT preloaded. Loading the CUDA 12
# nvrtc wheel RTLD_GLOBAL poisons CuPy's kernel compiler when a CUDA 13 stack
# (e.g. torch cu13 wheels) coexists in the same environment: CuPy resolves
# nvrtc 13 + CUDA 13 headers but compiles through the preloaded 12.9 library,
# and every jitify/CUB kernel build fails. ORT's CUDA EP and cuDNN do not
# require this preload (cuDNN falls back to precompiled engines).
NVIDIA_RUNTIME_LIB_MODULES = (
    "nvidia.cuda_runtime",
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.curand",
    "nvidia.cufft",
    "nvidia.nvjitlink",
)


def _coerce_version_info(
    version_info: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Normalize version_info-like values to a major/minor tuple."""
    if version_info is None:
        current = sys.version_info
        if hasattr(current, "major") and hasattr(current, "minor"):
            return (current.major, current.minor)
        return tuple(current[:2])
    if hasattr(version_info, "major") and hasattr(version_info, "minor"):
        return (version_info.major, version_info.minor)
    return tuple(version_info[:2])


def legacy_tray_enabled() -> bool:
    """Return whether the legacy GTK3 AppIndicator tray path is allowed.

    NV Broadcast is a GTK4/libadwaita application. The current tray code still
    uses a GTK3 AppIndicator bridge, which can terminate startup without a
    Python traceback on some Linux desktops. Keep it opt-in until it is
    replaced with a native GTK4/KStatusNotifier-safe path.

    Behavior:
    - explicit env override always wins
    - KDE/Plasma sessions default to disabled for safety
    - other Linux desktops keep the previous default-enabled behavior to avoid
      silently removing tray/minimize behavior for working installs
    """
    value = os.getenv("NVBROADCAST_ENABLE_LEGACY_TRAY", "").strip().lower()
    if value:
        return value in {"1", "true", "yes", "on"}

    desktop = " ".join(
        filter(
            None,
            (
                os.getenv("XDG_CURRENT_DESKTOP", ""),
                os.getenv("DESKTOP_SESSION", ""),
                "KDE" if os.getenv("KDE_FULL_SESSION") else "",
            ),
        )
    ).lower()
    if "kde" in desktop or "plasma" in desktop:
        return False
    return True


def linux_multiarch_triplet() -> str:
    """Return the Debian-style multiarch triplet for this machine."""
    if IS_ARM64:
        return "aarch64-linux-gnu"
    return "x86_64-linux-gnu"


def supports_linux_gpu_stack() -> bool:
    """Return whether the Linux CUDA/TensorRT optional stack is supported."""
    return IS_LINUX and IS_X86_64


def supports_tensorrt_python(version_info: tuple[int, int] | None = None) -> bool:
    """Return whether NVIDIA publishes TensorRT Python wheels for this version."""
    major, minor = _coerce_version_info(version_info)
    return major == 3 and 8 <= minor <= 13


def tensorrt_python_unsupported_reason(version_info: tuple[int, int] | None = None) -> str:
    """Human-readable reason when TensorRT wheels are unavailable for Python."""
    major, minor = _coerce_version_info(version_info)
    return (
        "TensorRT Python wheels are currently available only for Python 3.8-3.13 "
        f"on Linux x86_64. This system is running Python {major}.{minor}."
    )


def supports_openai_whisper_python(version_info: tuple[int, int] | None = None) -> bool:
    """Return whether openai-whisper is safe to auto-probe in the GUI process."""
    if _coerce_version_info(version_info) < (3, 14):
        return True

    try:
        from packaging.version import InvalidVersion, Version
    except Exception:
        return False

    try:
        numba_version = Version(importlib.metadata.version("numba"))
        llvmlite_version = Version(importlib.metadata.version("llvmlite"))
    except (importlib.metadata.PackageNotFoundError, InvalidVersion, Exception):
        return False

    # Numba announced Python 3.14 support in the 0.63 / 0.46 line. Until that
    # compatible pair is actually present, keep the GUI-startup auto-probe on
    # the safer faster-whisper path.
    return numba_version >= Version("0.63.0b1") and llvmlite_version >= Version("0.46.0b1")


def python_runtime_advisory(
    version_info: tuple[int, int] | None = None,
    has_trt_runtime: bool | None = None,
) -> tuple[str, str, str] | None:
    """Return a one-time advisory for Python runtimes with reduced feature paths.

    The goal is not to say the interpreter is unsupported. The goal is to tell
    users which optional premium paths are reduced today and what the app will
    do instead.
    """
    major, minor = _coerce_version_info(version_info)
    if (major, minor) < (3, 14):
        return None

    if has_trt_runtime is None:
        try:
            has_trt_runtime = has_tensorrt_runtime()
        except Exception:
            has_trt_runtime = False

    title = (
        f"Python {major}.{minor} detected: some optional runtimes use safer defaults"
    )
    lines = []
    if has_trt_runtime:
        lines.append(
            "TensorRT runtime is already installed, so Zeus and Killer can stay available."
        )
    else:
        lines.append(
            "Zeus and Killer TensorRT installs are reduced until NVIDIA ships matching Python wheels."
        )
    if supports_openai_whisper_python((major, minor)):
        lines.append(
            "Meeting transcription still works locally, and a compatible openai-whisper stack is allowed if installed."
        )
    else:
        lines.append(
            "Meeting transcription still works locally, but the app defaults to faster-whisper until a compatible numba/llvmlite stack is present."
        )
    lines.append(
        "DocZeus, CUDA modes, CPU modes, and local meeting transcription remain available."
    )
    lines.append(
        "Recommendation: Python 3.12 or 3.13 currently gives the broadest premium-feature compatibility."
    )
    lines.append(
        "As vendor libraries catch up, compatible installed runtimes will be picked up automatically."
    )
    return (f"python-runtime-{major}.{minor}", title, "\n".join(lines))


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


def get_tensorrt_lib_dirs() -> list:
    """Return candidate package directories that may contain TensorRT libs."""
    try:
        import importlib.util
        from pathlib import Path
    except Exception:
        return []

    dirs = []
    seen: set[str] = set()
    for module_name in TENSORRT_LIB_MODULES:
        spec = importlib.util.find_spec(module_name)
        if not spec or not spec.submodule_search_locations:
            continue
        root = Path(spec.submodule_search_locations[0])
        for candidate in (root, root / "lib"):
            if not candidate.is_dir():
                continue
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            dirs.append(candidate)
    return dirs


def get_trt_cache_dir(gpu_index: int) -> Path:
    """Return a per-GPU TensorRT cache directory under user config."""
    from nvbroadcast.core.constants import CONFIG_DIR

    cache_dir = CONFIG_DIR / "trt_cache" / f"gpu{gpu_index}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
        for lib_dir in get_tensorrt_lib_dirs():
            for so in sorted(lib_dir.glob("libnvinfer.so*")):
                try:
                    ctypes.CDLL(str(so))
                    return True
                except OSError:
                    continue
    except Exception:
        return False

    return False


def preload_nvidia_runtime_libs() -> None:
    """Preload pip-installed NVIDIA runtime libraries from nested wheel paths."""
    if not IS_LINUX:
        return
    try:
        import importlib.util

        # CuPy 14 uses CUDA_PATH to find the runtime headers required by
        # NVRTC. Component wheels keep those headers in a nested package
        # rather than a conventional /usr/local/cuda tree.
        if not os.environ.get("CUDA_PATH"):
            runtime_spec = importlib.util.find_spec("nvidia.cuda_runtime")
            if runtime_spec and runtime_spec.submodule_search_locations:
                runtime_root = Path(runtime_spec.submodule_search_locations[0])
                if (runtime_root / "include" / "cuda_runtime.h").is_file():
                    os.environ["CUDA_PATH"] = str(runtime_root)

        for pkg in NVIDIA_RUNTIME_LIB_MODULES:
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                lib_dir = Path(spec.submodule_search_locations[0]) / "lib"
                if not lib_dir.is_dir():
                    continue
                for so in sorted(lib_dir.glob("*.so*")):
                    try:
                        ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
    except Exception:
        pass


def has_cuda_inference_runtime() -> bool:
    """Return True when ONNX Runtime can run model inference on CUDA."""
    if not supports_linux_gpu_stack():
        return False
    preload_nvidia_runtime_libs()
    try:
        import onnxruntime as ort
    except Exception:
        return False
    try:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
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


def get_gst_camera_caps(
    device: str,
    width: int,
    height: int,
    fps: int,
    capture_format: str = "mjpeg",
) -> str:
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
    if capture_format == "raw":
        return (
            f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
            f"video/x-raw,width={width},height={height},"
            f"framerate={fps}/1"
        )
    return (
        f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
        f"image/jpeg,width={width},height={height},"
        f"framerate={fps}/1"
    )


def get_onnx_providers(gpu_index: int = 0,
                       use_tensorrt: bool = False,
                       trt_cache_path: str | None = None) -> list:
    """Return the ONNX Runtime execution providers for this platform."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    providers = []

    if supports_linux_gpu_stack() and use_tensorrt and 'TensorrtExecutionProvider' in available:
        cache_dir = trt_cache_path or str(get_trt_cache_dir(gpu_index))
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
            # GStreamer/GTK live pipelines can call inference from different
            # worker threads over time. Use one EP-level CUDA stream so ORT
            # does not keep per-thread stream/event state that can go stale.
            'use_ep_level_unified_stream': True,
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

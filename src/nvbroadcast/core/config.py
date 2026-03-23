# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""User configuration management - persists all settings across sessions."""

import tomllib
from pathlib import Path
from dataclasses import dataclass, field

from nvbroadcast.core.constants import CONFIG_DIR, CONFIG_FILE


@dataclass
class EdgeConfig:
    """Advanced edge refinement parameters - tunable per system."""
    dilate_size: int = 3          # Expand person mask (pixels, odd — smaller = less lag)
    blur_size: int = 5            # Edge softness (pixels, odd — smaller = crisper in motion)
    sigmoid_strength: float = 14.0  # Edge sharpness (higher = crisper boundary)
    sigmoid_midpoint: float = 0.45  # Edge transition center (lower = keeps more of person)


@dataclass
class VideoConfig:
    camera_device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    output_format: str = "YUY2"
    model: str = "rvm"
    quality_preset: str = "quality"
    background_removal: bool = False
    background_mode: str = "blur"
    background_image: str = ""
    blur_intensity: float = 0.7
    auto_frame: bool = False
    auto_frame_zoom: float = 1.5
    edge: EdgeConfig = field(default_factory=EdgeConfig)


@dataclass
class AudioConfig:
    mic_device: str = ""
    noise_removal: bool = False
    noise_intensity: float = 1.0
    speaker_denoise: bool = False


# Performance profiles: control where the workload runs (CPU vs GPU)
PERFORMANCE_PROFILES = {
    "max_quality": {
        "label": "Max Quality (GPU heavy, ~250% CPU)",
        "description": "Full 30fps processing, every frame, full resolution",
        "effects_fps": 30,
        "skip_interval": 1,
        "process_scale": 1.0,  # Full resolution
        "edge_dilate": 3,
        "edge_blur": 5,
        "edge_sigmoid": 14.0,
    },
    "balanced": {
        "label": "Balanced (recommended, ~120% CPU)",
        "description": "20fps effects, skip every other frame, full resolution",
        "effects_fps": 20,
        "skip_interval": 2,
        "process_scale": 1.0,
        "edge_dilate": 3,
        "edge_blur": 5,
        "edge_sigmoid": 12.0,
    },
    "performance": {
        "label": "Performance (CPU light, ~60% CPU)",
        "description": "15fps effects, half resolution processing, fast edges",
        "effects_fps": 15,
        "skip_interval": 2,
        "process_scale": 0.5,  # Half resolution for processing
        "edge_dilate": 2,
        "edge_blur": 3,
        "edge_sigmoid": 10.0,
    },
    "potato": {
        "label": "Low-End (minimal resources, ~30% CPU)",
        "description": "10fps effects, half resolution, skip 3 frames",
        "effects_fps": 10,
        "skip_interval": 3,
        "process_scale": 0.5,
        "edge_dilate": 1,
        "edge_blur": 3,
        "edge_sigmoid": 8.0,
    },
}


# Compositing backends
COMPOSITING_BACKENDS = {
    "cpu": {
        "label": "CPU (works everywhere)",
        "description": "NumPy/OpenCV compositing — compatible with all systems",
        "requires": [],
    },
    "gstreamer_gl": {
        "label": "GStreamer OpenGL (GPU — recommended)",
        "description": "GPU blur + blend via OpenGL — dramatically reduces CPU usage",
        "requires": ["glvideomixer", "gleffects_blur", "glupload"],
    },
    "cupy": {
        "label": "CuPy CUDA (GPU — maximum performance)",
        "description": "CUDA GPU arrays for compositing — requires cupy-cuda12x (~800MB)",
        "requires": ["cupy"],
    },
}


@dataclass
class AppConfig:
    compute_gpu: int = 0
    performance_profile: str = "balanced"  # max_quality, balanced, performance, potato
    compositing: str = "cpu"  # cpu, gstreamer_gl, cupy
    auto_start: bool = True
    minimize_on_close: bool = True
    first_run: bool = True  # Show setup wizard on first launch
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


def load_config() -> AppConfig:
    if not CONFIG_FILE.exists():
        return AppConfig()

    try:
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)

        config = AppConfig()
        for k in ("compute_gpu", "performance_profile", "compositing", "auto_start", "minimize_on_close", "first_run"):
            if k in data:
                setattr(config, k, data[k])
        if "video" in data:
            for k, v in data["video"].items():
                if k == "edge":
                    continue  # Handled separately below
                if hasattr(config.video, k):
                    setattr(config.video, k, v)
            if "edge" in data["video"]:
                for k, v in data["video"]["edge"].items():
                    if hasattr(config.video.edge, k):
                        setattr(config.video.edge, k, v)
        if "audio" in data:
            for k, v in data["audio"].items():
                if hasattr(config.audio, k):
                    setattr(config.audio, k, v)
        return config
    except Exception:
        return AppConfig()


def save_config(config: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        f"compute_gpu = {config.compute_gpu}",
        f'performance_profile = "{config.performance_profile}"',
        f'compositing = "{config.compositing}"',
        f"auto_start = {'true' if config.auto_start else 'false'}",
        f"minimize_on_close = {'true' if config.minimize_on_close else 'false'}",
        f"first_run = {'true' if config.first_run else 'false'}",
        "",
        "[video]",
        f'camera_device = "{config.video.camera_device}"',
        f"width = {config.video.width}",
        f"height = {config.video.height}",
        f"fps = {config.video.fps}",
        f'output_format = "{config.video.output_format}"',
        f'model = "{config.video.model}"',
        f'quality_preset = "{config.video.quality_preset}"',
        f"background_removal = {'true' if config.video.background_removal else 'false'}",
        f'background_mode = "{config.video.background_mode}"',
        f'background_image = "{config.video.background_image}"',
        f"blur_intensity = {config.video.blur_intensity}",
        f"auto_frame = {'true' if config.video.auto_frame else 'false'}",
        f"auto_frame_zoom = {config.video.auto_frame_zoom}",
        "",
        "[video.edge]",
        f"dilate_size = {config.video.edge.dilate_size}",
        f"blur_size = {config.video.edge.blur_size}",
        f"sigmoid_strength = {config.video.edge.sigmoid_strength}",
        f"sigmoid_midpoint = {config.video.edge.sigmoid_midpoint}",
        "",
        "[audio]",
        f'mic_device = "{config.audio.mic_device}"',
        f"noise_removal = {'true' if config.audio.noise_removal else 'false'}",
        f"noise_intensity = {config.audio.noise_intensity}",
        f"speaker_denoise = {'true' if config.audio.speaker_denoise else 'false'}",
    ]

    CONFIG_FILE.write_text("\n".join(lines) + "\n")


def detect_system_capabilities() -> dict:
    """Detect system hardware and recommend the best configuration."""
    import os
    import subprocess

    caps = {
        "cpu_cores": os.cpu_count() or 4,
        "gpu_name": "Unknown",
        "gpu_vram_mb": 0,
        "has_nvidia": False,
        "has_gl_compositor": False,
        "has_cupy": False,
        "recommended_mode": "cpu_quality",
    }

    # GPU detection
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        caps["gpu_name"] = parts[0]
        caps["gpu_vram_mb"] = int(parts[1])
        caps["has_nvidia"] = True
    except Exception:
        pass

    # GStreamer GL
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        caps["has_gl_compositor"] = all(
            Gst.ElementFactory.find(e) is not None
            for e in ["glvideomixer", "glupload", "gldownload"]
        )
    except Exception:
        pass

    # CuPy
    try:
        import cupy  # noqa: F401
        caps["has_cupy"] = True
    except ImportError:
        pass

    # Auto-recommend based on hardware
    if caps["has_nvidia"]:
        if caps["has_cupy"]:
            caps["recommended_mode"] = "gpu_cuda_best"
        elif caps["has_gl_compositor"]:
            caps["recommended_mode"] = "gpu_balanced"
        elif caps["gpu_vram_mb"] >= 4096:
            caps["recommended_mode"] = "gpu_balanced"  # Can install GL
        else:
            caps["recommended_mode"] = "cpu_quality"
    elif caps["cpu_cores"] >= 8:
        caps["recommended_mode"] = "cpu_quality"
    elif caps["cpu_cores"] >= 4:
        caps["recommended_mode"] = "cpu_light"
    else:
        caps["recommended_mode"] = "low_end"

    return caps


def detect_compositing_backends() -> dict[str, bool]:
    """Detect which compositing backends are available on this system."""
    available = {"cpu": True}

    # Check GStreamer GL
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        gl_ok = all(
            Gst.ElementFactory.find(e) is not None
            for e in ["glvideomixer", "glupload", "gldownload"]
        )
        available["gstreamer_gl"] = gl_ok
    except Exception:
        available["gstreamer_gl"] = False

    # Check CuPy
    try:
        import cupy  # noqa: F401
        available["cupy"] = True
    except ImportError:
        available["cupy"] = False

    return available


def apply_performance_profile(config: AppConfig, profile_name: str) -> None:
    """Apply a performance profile to the config."""
    if profile_name not in PERFORMANCE_PROFILES:
        return
    p = PERFORMANCE_PROFILES[profile_name]
    config.performance_profile = profile_name
    config.video.edge.dilate_size = p["edge_dilate"]
    config.video.edge.blur_size = p["edge_blur"]
    config.video.edge.sigmoid_strength = p["edge_sigmoid"]

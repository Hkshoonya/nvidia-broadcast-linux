# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
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
class BeautyConfig:
    """Video enhancement / beautification settings."""
    enabled: bool = False
    preset: str = "natural"
    skin_smooth: float = 0.5
    denoise: float = 0.3
    enhance: float = 0.4
    sharpen: float = 0.3
    edge_darken: float = 0.2


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
    mirror: bool = True
    eye_contact: bool = False
    eye_contact_intensity: float = 0.7
    relighting: bool = False
    relighting_intensity: float = 0.5
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    beauty: BeautyConfig = field(default_factory=BeautyConfig)


@dataclass
class AudioConfig:
    mic_device: str = ""
    noise_removal: bool = False
    noise_intensity: float = 1.0
    speaker_denoise: bool = False


# Performance profiles: control where the workload runs (CPU vs GPU)
# effects_ratio: fraction of camera fps used for effects (1.0 = every frame)
PERFORMANCE_PROFILES = {
    "max_quality": {
        "label": "Max Quality",
        "description": "Process every frame at full resolution",
        "effects_ratio": 1.0,
        "skip_interval": 1,
        "process_scale": 1.0,
        "edge_dilate": 3,
        "edge_blur": 5,
        "edge_sigmoid": 14.0,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Process 2/3 of frames, full resolution",
        "effects_ratio": 0.67,
        "skip_interval": 2,
        "process_scale": 1.0,
        "edge_dilate": 3,
        "edge_blur": 5,
        "edge_sigmoid": 12.0,
    },
    "performance": {
        "label": "Performance",
        "description": "Process half of frames, fast edges",
        "effects_ratio": 0.5,
        "skip_interval": 2,
        "process_scale": 0.5,
        "edge_dilate": 2,
        "edge_blur": 3,
        "edge_sigmoid": 10.0,
    },
    "potato": {
        "label": "Low-End",
        "description": "Process 1/3 of frames, minimal resources",
        "effects_ratio": 0.33,
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


def _load_from_toml(filepath: Path) -> AppConfig:
    """Load an AppConfig from a TOML file."""
    with open(filepath, "rb") as f:
        data = tomllib.load(f)

    config = AppConfig()
    for k in ("compute_gpu", "performance_profile", "compositing",
              "auto_start", "minimize_on_close", "first_run"):
        if k in data:
            setattr(config, k, data[k])
    if "video" in data:
        for k, v in data["video"].items():
            if k in ("edge", "beauty"):
                continue
            if hasattr(config.video, k):
                setattr(config.video, k, v)
        if "edge" in data["video"]:
            for k, v in data["video"]["edge"].items():
                if hasattr(config.video.edge, k):
                    setattr(config.video.edge, k, v)
        if "beauty" in data["video"]:
            for k, v in data["video"]["beauty"].items():
                if hasattr(config.video.beauty, k):
                    setattr(config.video.beauty, k, v)
    if "audio" in data:
        for k, v in data["audio"].items():
            if hasattr(config.audio, k):
                setattr(config.audio, k, v)
    return config


def load_config() -> AppConfig:
    if not CONFIG_FILE.exists():
        return AppConfig()
    try:
        return _load_from_toml(CONFIG_FILE)
    except Exception:
        return AppConfig()


def _bool(val: bool) -> str:
    return "true" if val else "false"


def _config_to_toml(config: AppConfig) -> str:
    """Serialize AppConfig to TOML string (complete — all fields)."""
    v = config.video
    a = config.audio
    b = v.beauty
    e = v.edge
    lines = [
        f"compute_gpu = {config.compute_gpu}",
        f'performance_profile = "{config.performance_profile}"',
        f'compositing = "{config.compositing}"',
        f"auto_start = {_bool(config.auto_start)}",
        f"minimize_on_close = {_bool(config.minimize_on_close)}",
        f"first_run = {_bool(config.first_run)}",
        "",
        "[video]",
        f'camera_device = "{v.camera_device}"',
        f"width = {v.width}",
        f"height = {v.height}",
        f"fps = {v.fps}",
        f'output_format = "{v.output_format}"',
        f'model = "{v.model}"',
        f'quality_preset = "{v.quality_preset}"',
        f"background_removal = {_bool(v.background_removal)}",
        f'background_mode = "{v.background_mode}"',
        f'background_image = "{v.background_image}"',
        f"blur_intensity = {v.blur_intensity}",
        f"auto_frame = {_bool(v.auto_frame)}",
        f"auto_frame_zoom = {v.auto_frame_zoom}",
        f"mirror = {_bool(v.mirror)}",
        f"eye_contact = {_bool(v.eye_contact)}",
        f"eye_contact_intensity = {v.eye_contact_intensity}",
        f"relighting = {_bool(v.relighting)}",
        f"relighting_intensity = {v.relighting_intensity}",
        "",
        "[video.edge]",
        f"dilate_size = {e.dilate_size}",
        f"blur_size = {e.blur_size}",
        f"sigmoid_strength = {e.sigmoid_strength}",
        f"sigmoid_midpoint = {e.sigmoid_midpoint}",
        "",
        "[video.beauty]",
        f"enabled = {_bool(b.enabled)}",
        f'preset = "{b.preset}"',
        f"skin_smooth = {b.skin_smooth}",
        f"denoise = {b.denoise}",
        f"enhance = {b.enhance}",
        f"sharpen = {b.sharpen}",
        f"edge_darken = {b.edge_darken}",
        "",
        "[audio]",
        f'mic_device = "{a.mic_device}"',
        f"noise_removal = {_bool(a.noise_removal)}",
        f"noise_intensity = {a.noise_intensity}",
        f"speaker_denoise = {_bool(a.speaker_denoise)}",
    ]
    return "\n".join(lines) + "\n"


def save_config(config: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(_config_to_toml(config))


# ─── User Profiles ───────────────────────────────────────────────────────────

from nvbroadcast.core.constants import PROFILES_DIR


def list_profiles() -> list[str]:
    """List available user profile names."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p.stem for p in PROFILES_DIR.glob("*.toml")
    )


def save_profile(name: str, config: AppConfig) -> Path:
    """Save current config as a named profile."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    filepath = PROFILES_DIR / f"{safe_name}.toml"
    filepath.write_text(_config_to_toml(config))
    return filepath


def load_profile(name: str) -> AppConfig | None:
    """Load a named profile. Returns None if not found."""
    filepath = PROFILES_DIR / f"{name}.toml"
    if not filepath.exists():
        return None
    try:
        return _load_from_toml(filepath)
    except Exception:
        return None


def delete_profile(name: str) -> bool:
    """Delete a named profile."""
    filepath = PROFILES_DIR / f"{name}.toml"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def get_builtin_profiles() -> dict[str, dict]:
    """Built-in preset profiles for common use cases."""
    return {
        "Meeting": {
            "description": "Clean look for video calls",
            "background_removal": True,
            "background_mode": "blur",
            "blur_intensity": 0.6,
            "eye_contact": True,
            "eye_contact_intensity": 0.5,
            "relighting": True,
            "relighting_intensity": 0.4,
            "beauty_enabled": True,
            "beauty_preset": "natural",
        },
        "Streaming": {
            "description": "Professional broadcast look",
            "background_removal": True,
            "background_mode": "blur",
            "blur_intensity": 0.8,
            "eye_contact": False,
            "relighting": True,
            "relighting_intensity": 0.6,
            "beauty_enabled": True,
            "beauty_preset": "broadcast",
        },
        "Presentation": {
            "description": "Minimal processing, max performance",
            "background_removal": True,
            "background_mode": "blur",
            "blur_intensity": 0.5,
            "eye_contact": True,
            "eye_contact_intensity": 0.7,
            "relighting": False,
            "beauty_enabled": False,
        },
        "Gaming": {
            "description": "Low overhead, background only",
            "background_removal": True,
            "background_mode": "replace",
            "eye_contact": False,
            "relighting": False,
            "beauty_enabled": False,
        },
        "Clean": {
            "description": "Everything off — passthrough",
            "background_removal": False,
            "eye_contact": False,
            "relighting": False,
            "beauty_enabled": False,
        },
    }


def apply_builtin_profile(config: AppConfig, name: str) -> bool:
    """Apply a built-in profile to the config."""
    profiles = get_builtin_profiles()
    if name not in profiles:
        return False
    p = profiles[name]
    config.video.background_removal = p.get("background_removal", False)
    config.video.background_mode = p.get("background_mode", "blur")
    config.video.blur_intensity = p.get("blur_intensity", 0.7)
    config.video.eye_contact = p.get("eye_contact", False)
    config.video.eye_contact_intensity = p.get("eye_contact_intensity", 0.7)
    config.video.relighting = p.get("relighting", False)
    config.video.relighting_intensity = p.get("relighting_intensity", 0.5)
    config.video.beauty.enabled = p.get("beauty_enabled", False)
    if "beauty_preset" in p:
        config.video.beauty.preset = p["beauty_preset"]
    return True


def detect_system_capabilities() -> dict:
    """Detect system hardware and recommend the best configuration."""
    import os
    import subprocess
    from nvbroadcast.core.platform import IS_MACOS, IS_LINUX

    caps = {
        "cpu_cores": os.cpu_count() or 4,
        "gpu_name": "Unknown",
        "gpu_vram_mb": 0,
        "has_nvidia": False,
        "has_apple_silicon": False,
        "has_gl_compositor": False,
        "has_cupy": False,
        "recommended_mode": "cpu_quality",
    }

    if IS_MACOS:
        # Detect Apple Silicon
        import platform as _pf
        caps["gpu_name"] = f"Apple {_pf.processor() or 'Silicon'}"
        caps["has_apple_silicon"] = _pf.machine() == "arm64"
        caps["recommended_mode"] = "cpu_quality"
        return caps

    # GPU detection (Linux — nvidia-smi)
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

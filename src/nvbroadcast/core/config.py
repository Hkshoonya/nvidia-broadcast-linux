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
class VideoConfig:
    camera_device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    output_format: str = "YUY2"
    quality_preset: str = "quality"
    background_removal: bool = False
    background_mode: str = "blur"
    background_image: str = ""
    blur_intensity: float = 0.7
    auto_frame: bool = False
    auto_frame_zoom: float = 1.5


@dataclass
class AudioConfig:
    mic_device: str = ""
    noise_removal: bool = False
    noise_intensity: float = 1.0
    speaker_denoise: bool = False


@dataclass
class AppConfig:
    compute_gpu: int = 0
    auto_start: bool = True  # Start broadcast automatically on launch
    minimize_on_close: bool = True  # Minimize to tray instead of quitting
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


def load_config() -> AppConfig:
    if not CONFIG_FILE.exists():
        return AppConfig()

    try:
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)

        config = AppConfig()
        for k in ("compute_gpu", "auto_start", "minimize_on_close"):
            if k in data:
                setattr(config, k, data[k])
        if "video" in data:
            for k, v in data["video"].items():
                if hasattr(config.video, k):
                    setattr(config.video, k, v)
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
        f"auto_start = {'true' if config.auto_start else 'false'}",
        f"minimize_on_close = {'true' if config.minimize_on_close else 'false'}",
        "",
        "[video]",
        f'camera_device = "{config.video.camera_device}"',
        f"width = {config.video.width}",
        f"height = {config.video.height}",
        f"fps = {config.video.fps}",
        f'output_format = "{config.video.output_format}"',
        f'quality_preset = "{config.video.quality_preset}"',
        f"background_removal = {'true' if config.video.background_removal else 'false'}",
        f'background_mode = "{config.video.background_mode}"',
        f'background_image = "{config.video.background_image}"',
        f"blur_intensity = {config.video.blur_intensity}",
        f"auto_frame = {'true' if config.video.auto_frame else 'false'}",
        f"auto_frame_zoom = {config.video.auto_frame_zoom}",
        "",
        "[audio]",
        f'mic_device = "{config.audio.mic_device}"',
        f"noise_removal = {'true' if config.audio.noise_removal else 'false'}",
        f"noise_intensity = {config.audio.noise_intensity}",
        f"speaker_denoise = {'true' if config.audio.speaker_denoise else 'false'}",
    ]

    CONFIG_FILE.write_text("\n".join(lines) + "\n")

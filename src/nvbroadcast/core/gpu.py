# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""GPU detection and selection utilities."""

import subprocess
import re
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    compute_capability: str
    driver_version: str


def detect_gpus() -> list[GpuInfo]:
    """Detect NVIDIA GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=int(parts[2]),
                    compute_capability=parts[3],
                    driver_version=parts[4],
                )
            )
    return gpus


def select_compute_gpu(gpus: list[GpuInfo], preferred_index: int = 1) -> GpuInfo | None:
    """Select the GPU for AI compute workloads.

    Prefers the specified index (default: device 1 / RTX 3080).
    Falls back to any available GPU.
    """
    if not gpus:
        return None

    for gpu in gpus:
        if gpu.index == preferred_index:
            return gpu

    return gpus[0]


def get_gpu_summary() -> str:
    """Return a human-readable GPU summary."""
    gpus = detect_gpus()
    if not gpus:
        return "No NVIDIA GPUs detected"

    lines = []
    for gpu in gpus:
        lines.append(
            f"  GPU {gpu.index}: {gpu.name} ({gpu.memory_total_mb} MB, CC {gpu.compute_capability})"
        )
    return "\n".join(lines)

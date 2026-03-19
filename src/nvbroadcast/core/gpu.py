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


def get_cuda_device_id(nvsmi_index: int) -> int:
    """Map an nvidia-smi GPU index to the CUDA device_id used by ONNX Runtime.

    nvidia-smi and CUDA can enumerate GPUs in different orders.
    This maps by matching UUIDs between nvidia-smi and CUDA's ordering.
    """
    try:
        # Get nvidia-smi UUID for the requested index
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        uuid_by_nvsmi = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_by_nvsmi[int(parts[0])] = parts[1]

        target_uuid = uuid_by_nvsmi.get(nvsmi_index)
        if not target_uuid:
            return nvsmi_index  # Fallback

        # Get CUDA ordering via nvidia-smi topology
        # CUDA enumerates by PCI bus ID by default
        result2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
             "--id=" + ",".join(str(i) for i in sorted(uuid_by_nvsmi.keys()))],
            capture_output=True, text=True, check=True,
        )

        # Try to determine CUDA order from PCI bus IDs
        result3 = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        pci_uuid_pairs = []
        for line in result3.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                pci_uuid_pairs.append((parts[0], parts[1]))

        # CUDA orders by PCI bus ID ascending
        pci_uuid_pairs.sort(key=lambda x: x[0])
        for cuda_id, (_, uuid) in enumerate(pci_uuid_pairs):
            if uuid == target_uuid:
                return cuda_id

        return nvsmi_index  # Fallback
    except Exception:
        return nvsmi_index  # Fallback


def select_compute_gpu(gpus: list[GpuInfo], preferred_index: int = 0) -> GpuInfo | None:
    """Select the GPU for AI compute workloads."""
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

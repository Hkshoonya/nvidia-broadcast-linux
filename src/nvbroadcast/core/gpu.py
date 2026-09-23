# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""GPU detection and selection utilities."""

import os
import subprocess
import ctypes
from dataclasses import dataclass

# cuCtxCreate flag: park CPU threads on a synchronization primitive while
# waiting for the GPU instead of the default heuristic, which busy-spins
# whenever the machine has at least as many cores as CUDA contexts.
_CU_CTX_SCHED_BLOCKING_SYNC = 0x04


def apply_cuda_blocking_sync() -> bool:
    """Make CUDA synchronization block instead of spin (opt-out via env).

    Every cupy ``synchronize()`` and ONNX Runtime IOBinding sync otherwise
    burns a CPU core for the full duration of in-flight GPU work. The flag
    only takes effect if it is set before the primary context is created,
    so this must run before the first cupy/ORT CUDA call in the process.
    Uses the driver API because cupy does not expose cudaSetDeviceFlags.
    """
    if os.getenv("NVBROADCAST_CUDA_SYNC", "blocking").strip().lower() == "spin":
        return False
    try:
        import ctypes

        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            return False
        count = ctypes.c_int(0)
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
        ok = False
        for device in range(count.value):
            if cuda.cuDevicePrimaryCtxSetFlags(
                    device, _CU_CTX_SCHED_BLOCKING_SYNC) == 0:
                ok = True
        return ok
    except Exception:
        return False


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    compute_capability: str
    driver_version: str


def _detect_gpus_with_nvidia_smi() -> list[GpuInfo]:
    """Detect NVIDIA GPUs through the host utility when it is available."""
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

    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                index = int(parts[0])
                memory_total_mb = int(parts[2])
            except ValueError:
                continue
            gpus.append(
                GpuInfo(
                    index=index,
                    name=parts[1],
                    memory_total_mb=memory_total_mb,
                    compute_capability=parts[3],
                    driver_version=parts[4],
                )
            )
    return gpus


def _cuda_driver_version(value: int) -> str:
    if value <= 0:
        return "CUDA driver API"
    return f"CUDA driver API {value // 1000}.{(value % 1000) // 10}"


def _detect_gpus_with_cuda_driver() -> list[GpuInfo]:
    """Detect GPUs through libcuda when host tools are sandboxed away."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            return []

        count = ctypes.c_int()
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return []

        driver_api = ctypes.c_int()
        if cuda.cuDriverGetVersion(ctypes.byref(driver_api)) != 0:
            driver_api.value = 0
    except Exception:
        return []

    total_mem_fn = getattr(cuda, "cuDeviceTotalMem_v2", None)
    if total_mem_fn is None:
        total_mem_fn = getattr(cuda, "cuDeviceTotalMem", None)

    gpus: list[GpuInfo] = []
    for ordinal in range(max(0, count.value)):
        try:
            device = ctypes.c_int()
            if cuda.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
                continue

            name_buffer = ctypes.create_string_buffer(256)
            if cuda.cuDeviceGetName(name_buffer, len(name_buffer), device) != 0:
                name = f"NVIDIA GPU {ordinal}"
            else:
                name = name_buffer.value.decode("utf-8", errors="replace")

            total_memory = ctypes.c_size_t()
            if total_mem_fn is None or total_mem_fn(
                ctypes.byref(total_memory), device
            ) != 0:
                total_memory.value = 0

            major = ctypes.c_int()
            minor = ctypes.c_int()
            if cuda.cuDeviceComputeCapability(
                ctypes.byref(major), ctypes.byref(minor), device
            ) != 0:
                compute_capability = "Unknown"
            else:
                compute_capability = f"{major.value}.{minor.value}"

            gpus.append(
                GpuInfo(
                    index=ordinal,
                    name=name or f"NVIDIA GPU {ordinal}",
                    memory_total_mb=int(total_memory.value >> 20),
                    compute_capability=compute_capability,
                    driver_version=_cuda_driver_version(driver_api.value),
                )
            )
        except Exception:
            continue
    return gpus


def detect_gpus() -> list[GpuInfo]:
    """Detect NVIDIA GPUs without requiring host executables in sandboxes."""
    return _detect_gpus_with_nvidia_smi() or _detect_gpus_with_cuda_driver()


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

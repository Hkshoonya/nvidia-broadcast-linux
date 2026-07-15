"""Performance monitoring — FPS, GPU usage, VRAM."""

import ctypes
import subprocess
import threading
import time


class _Nvml:
    """Minimal ctypes NVML wrapper — one in-process query instead of an
    nvidia-smi fork every poll. pynvml is not a dependency; libnvidia-ml
    ships with the driver."""

    class _Utilization(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class _Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    def __init__(self, gpu_index: int):
        self._lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self._lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        self._handle = ctypes.c_void_p()
        if self._lib.nvmlDeviceGetHandleByIndex_v2(
                gpu_index, ctypes.byref(self._handle)) != 0:
            self._lib.nvmlShutdown()
            raise OSError(f"no NVML handle for GPU {gpu_index}")

    def query(self) -> tuple[int, int, int, int]:
        """Return (gpu_util_pct, vram_used_mb, vram_total_mb, temp_c)."""
        util = self._Utilization()
        mem = self._Memory()
        temp = ctypes.c_uint()
        if self._lib.nvmlDeviceGetUtilizationRates(
                self._handle, ctypes.byref(util)) != 0:
            raise OSError("utilization query failed")
        if self._lib.nvmlDeviceGetMemoryInfo(
                self._handle, ctypes.byref(mem)) != 0:
            raise OSError("memory query failed")
        # 0 = NVML_TEMPERATURE_GPU
        if self._lib.nvmlDeviceGetTemperature(
                self._handle, 0, ctypes.byref(temp)) != 0:
            raise OSError("temperature query failed")
        return (int(util.gpu), int(mem.used >> 20), int(mem.total >> 20),
                int(temp.value))

    def close(self):
        try:
            self._lib.nvmlShutdown()
        except Exception:
            pass


class PerfMonitor:
    """Polls GPU stats and tracks FPS."""

    def __init__(self, gpu_index: int = 0):
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.monotonic()
        self._gpu_index = gpu_index
        self._gpu_util = 0
        self._vram_used = 0
        self._vram_total = 0
        self._gpu_temp = 0
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_gpu, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def tick(self):
        """Call once per processed frame to track FPS."""
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 0.5:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def gpu_util(self) -> int:
        return self._gpu_util

    @property
    def vram_used_mb(self) -> int:
        return self._vram_used

    @property
    def vram_total_mb(self) -> int:
        return self._vram_total

    @property
    def gpu_temp(self) -> int:
        return self._gpu_temp

    @property
    def gpu_index(self) -> int:
        return self._gpu_index

    def set_gpu_index(self, gpu_index: int) -> None:
        self._gpu_index = max(0, int(gpu_index))

    def format_status(self) -> str:
        """Format as a status bar string."""
        parts = [f"{self._fps:.0f} fps"]
        if self._vram_total > 0:
            parts.append(f"GPU {self._gpu_index} {self._gpu_util}%")
            parts.append(f"VRAM {self._vram_used}MB/{self._vram_total}MB")
            parts.append(f"{self._gpu_temp}°C")
        return "  |  ".join(parts)

    def _poll_gpu(self):
        """Poll GPU stats every 2 seconds — NVML in-process, nvidia-smi
        subprocess only as fallback."""
        nvml = None
        nvml_index = None
        while self._running:
            if nvml is not None and nvml_index != self._gpu_index:
                nvml.close()
                nvml = None
            if nvml is None:
                try:
                    nvml = _Nvml(self._gpu_index)
                    nvml_index = self._gpu_index
                except Exception:
                    nvml = None
            try:
                if nvml is not None:
                    (self._gpu_util, self._vram_used,
                     self._vram_total, self._gpu_temp) = nvml.query()
                else:
                    self._poll_gpu_smi()
            except Exception:
                if nvml is not None:
                    nvml.close()
                    nvml = None
            time.sleep(2)
        if nvml is not None:
            nvml.close()

    def _poll_gpu_smi(self):
        result = subprocess.run(
            ["nvidia-smi",
             f"--id={self._gpu_index}",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 4:
            self._gpu_util = int(parts[0])
            self._vram_used = int(parts[1])
            self._vram_total = int(parts[2])
            self._gpu_temp = int(parts[3])

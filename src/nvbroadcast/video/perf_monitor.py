"""Performance monitoring — FPS, GPU usage, VRAM."""

import subprocess
import threading
import time


class PerfMonitor:
    """Polls GPU stats and tracks FPS."""

    def __init__(self):
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.monotonic()
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

    def format_status(self) -> str:
        """Format as a status bar string."""
        parts = [f"{self._fps:.0f} fps"]
        if self._vram_total > 0:
            parts.append(f"GPU {self._gpu_util}%")
            parts.append(f"VRAM {self._vram_used}MB/{self._vram_total}MB")
            parts.append(f"{self._gpu_temp}\u00b0C")
        return "  |  ".join(parts)

    def _poll_gpu(self):
        """Poll nvidia-smi every 2 seconds."""
        while self._running:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
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
            except Exception:
                pass
            time.sleep(2)

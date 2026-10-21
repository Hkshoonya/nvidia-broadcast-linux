"""Optional HUSH/Maxine worker, isolated from Python's CUDA runtime.

Protocol and installation: docs/HUSH.md. No downloads occur in this module.
"""

import os
import platform
import selectors
import shutil
import struct
import subprocess
import time

import numpy as np


class HushDenoiser:
    def __init__(self):
        self._process = None
        self._pending = np.empty(0, dtype=np.float32)
        self._output = np.empty(0, dtype=np.float32)
        self._frame = 0

    @property
    def available(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _read_exact(self, size: int, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        result = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while len(result) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("HUSH worker response timed out")
                block = os.read(self._process.stdout.fileno(), size - len(result))
                if not block:
                    raise RuntimeError("HUSH worker exited or closed its output")
                result.extend(block)
        return bytes(result)

    def initialize(self) -> bool:
        if self.available:
            return True
        try:
            if platform.system() != "Linux" or platform.machine() != "x86_64":
                raise RuntimeError("HUSH requires Linux x86_64")
            executable = shutil.which(os.getenv("NVBROADCAST_HUSH_WORKER", "hush-worker"))
            if not executable:
                raise RuntimeError("install hush-worker; see docs/HUSH.md")
            self._process = subprocess.Popen(
                [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
            )
            header = self._read_exact(8, 30.0)
            self._frame = struct.unpack("<I", header[4:])[0]
            if header[:4] != b"HSH1" or not 1 <= self._frame <= 4800:
                raise RuntimeError("unsupported HUSH worker protocol")
            self._pending = np.empty(0, dtype=np.float32)
            # One frame of fixed latency permits arbitrary capture block sizes.
            self._output = np.zeros(self._frame, dtype=np.float32)
            return True
        except Exception as error:
            print(f"[NV Broadcast] HUSH initialization failed: {error}", flush=True)
            self.cleanup()
            return False

    def process_chunk(self, audio_data: np.ndarray, sample_rate: int = 48000,
                      intensity: float = 1.0) -> np.ndarray:
        if sample_rate != 48000:
            raise ValueError("HUSH requires 48 kHz mono audio")
        if not self.available:
            raise RuntimeError("HUSH worker is unavailable")
        self._pending = np.concatenate((self._pending, audio_data.astype(np.float32)))
        while len(self._pending) >= self._frame:
            packet = struct.pack("<f", intensity) + self._pending[:self._frame].astype("<f4").tobytes()
            # Nonblocking writes bound the callback even if the worker stalls.
            os.set_blocking(self._process.stdin.fileno(), False)
            deadline = time.monotonic() + 0.25
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdin, selectors.EVENT_WRITE)
                sent = 0
                while sent < len(packet):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("HUSH worker input timed out")
                    try:
                        sent += os.write(self._process.stdin.fileno(), packet[sent:])
                    except BlockingIOError:
                        continue
            cleaned = np.frombuffer(self._read_exact(self._frame * 4, 0.25), dtype="<f4")
            if not np.isfinite(cleaned).all():
                raise RuntimeError("HUSH returned non-finite audio")
            self._output = np.concatenate((self._output, cleaned))
            self._pending = self._pending[self._frame:]
        size = len(audio_data)
        result = self._output[:size].copy()
        self._output = self._output[size:]
        return result

    def cleanup(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            process.stdin.close()
            process.stdout.close()
        self._pending = np.empty(0, dtype=np.float32)
        self._output = np.empty(0, dtype=np.float32)

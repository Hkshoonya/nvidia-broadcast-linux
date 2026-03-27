# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Background installer for optional runtimes.

The app ships leaner by default and installs heavyweight optional runtimes on
first use. This manager runs installs in the background, streams status lines
back to GTK, and keeps the rest of the app usable while dependencies download.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import threading
from pathlib import Path

import gi

gi.require_version("GObject", "2.0")
from gi.repository import GObject, GLib

from nvbroadcast.core.platform import has_tensorrt_runtime


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def _has_whisper() -> bool:
    return importlib.util.find_spec("whisper") is not None


def _verify_cupy() -> bool:
    try:
        import cupy
        import numpy as np
        arr = cupy.asarray(np.ones((8, 8), dtype=np.float32))
        _ = (arr * 2.0).astype(cupy.float32)
        return True
    except Exception:
        return False


PACKAGE_SPECS = {
    "cupy": {
        "title": "CUDA Compositing Runtime",
        "subtitle": "Needed for DocZeus and CUDA modes",
        "size": "~800 MB",
        "summary": (
            "Installs CuPy and CUDA runtime wheels so GPU compositing modes can run "
            "inside the app."
        ),
        "install_args": ["install", "cupy-cuda12x", "nvidia-cuda-nvrtc-cu12"],
        "check": _has_cupy,
        "verify": _verify_cupy,
        "help": "Retry later with: .venv/bin/pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12",
    },
    "tensorrt": {
        "title": "TensorRT Runtime",
        "subtitle": "Needed for Zeus and Killer premium modes",
        "size": "~1.2 GB",
        "summary": (
            "Installs the TensorRT Python runtime used by premium inference modes "
            "for faster ONNX execution."
        ),
        "install_args": ["install", "tensorrt-cu12"],
        "check": has_tensorrt_runtime,
        "verify": has_tensorrt_runtime,
        "help": "Retry later with: .venv/bin/pip install tensorrt-cu12",
    },
    "whisper": {
        "title": "Meeting Transcription Runtime",
        "subtitle": "Needed for local meeting transcription",
        "size": "~250 MB",
        "summary": (
            "Installs Whisper so Start Meeting can record and transcribe locally "
            "without sending audio anywhere."
        ),
        "install_args": ["install", "openai-whisper"],
        "check": _has_whisper,
        "verify": _has_whisper,
        "help": "Retry later with: .venv/bin/pip install openai-whisper",
    },
}

PACKAGE_BUNDLES = {
    "premium_gpu_stack": {
        "title": "Premium GPU Runtime",
        "subtitle": "Needed for Zeus and Killer premium modes",
        "size": "~2.0 GB",
        "summary": (
            "Installs both CUDA compositing and TensorRT so premium GPU modes can "
            "run at full speed."
        ),
        "packages": ["cupy", "tensorrt"],
    },
}


class DependencyInstaller(GObject.Object):
    """Install optional runtimes without blocking the GTK main loop."""

    __gsignals__ = {
        "job-started": (GObject.SignalFlags.RUN_FIRST, None, (str, str)),
        "job-progress": (GObject.SignalFlags.RUN_FIRST, None, (str, str, float)),
        "job-completed": (GObject.SignalFlags.RUN_FIRST, None, (str, bool, str)),
    }

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._active_job_id = ""
        self._active_thread = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._active_job_id)

    def is_available(self, key: str) -> bool:
        if key in PACKAGE_BUNDLES:
            return all(self.is_available(pkg) for pkg in PACKAGE_BUNDLES[key]["packages"])
        spec = PACKAGE_SPECS.get(key)
        if spec is None:
            return False
        try:
            return bool(spec["check"]())
        except Exception:
            return False

    def describe(self, key: str) -> dict:
        if key in PACKAGE_BUNDLES:
            return PACKAGE_BUNDLES[key]
        return PACKAGE_SPECS[key]

    def missing_for_mode(self, mode_key: str) -> list[str]:
        missing: list[str] = []
        if mode_key in ("doczeus", "cuda_max", "cuda_balanced", "cuda_perf", "zeus", "killer"):
            if not self.is_available("cupy"):
                missing.append("cupy")
        if mode_key in ("zeus", "killer") and not self.is_available("tensorrt"):
            missing.append("tensorrt")
        return missing

    def install_key_for_mode(self, mode_key: str) -> str | None:
        missing = self.missing_for_mode(mode_key)
        if missing == ["cupy", "tensorrt"] or missing == ["tensorrt", "cupy"]:
            return "premium_gpu_stack"
        if len(missing) == 1:
            return missing[0]
        return None

    def start_install(self, key: str) -> bool:
        with self._lock:
            if self._active_job_id:
                return False
            self._active_job_id = key

        thread = threading.Thread(target=self._run_install, args=(key,), daemon=True)
        self._active_thread = thread
        thread.start()
        return True

    def _emit_started(self, key: str, text: str) -> bool:
        self.emit("job-started", key, text)
        return False

    def _emit_progress(self, key: str, text: str, fraction: float) -> bool:
        self.emit("job-progress", key, text, fraction)
        return False

    def _emit_completed(self, key: str, success: bool, text: str) -> bool:
        self.emit("job-completed", key, success, text)
        return False

    def _run_install(self, key: str):
        if key in PACKAGE_BUNDLES:
            bundle = PACKAGE_BUNDLES[key]
            GLib.idle_add(
                self._emit_started,
                key,
                f"{bundle['title']} download started. The app stays usable while this runs.",
            )
            ok = True
            final_msg = ""
            packages = bundle["packages"]
            for index, package_id in enumerate(packages, start=1):
                success, message = self._install_single(
                    key,
                    package_id,
                    prefix=f"Step {index}/{len(packages)}",
                )
                if not success:
                    ok = False
                    final_msg = message
                    break
            if ok:
                final_msg = f"{bundle['title']} installed and ready."
            self._finish_job(key, ok, final_msg)
            return

        spec = PACKAGE_SPECS[key]
        GLib.idle_add(
            self._emit_started,
            key,
            f"{spec['title']} download started. The app stays usable while this runs.",
        )
        success, message = self._install_single(key, key)
        self._finish_job(key, success, message)

    def _finish_job(self, key: str, success: bool, message: str):
        with self._lock:
            self._active_job_id = ""
            self._active_thread = None
        GLib.idle_add(self._emit_completed, key, success, message)

    def _install_single(self, job_key: str, package_id: str, prefix: str = "") -> tuple[bool, str]:
        spec = PACKAGE_SPECS[package_id]
        if self.is_available(package_id):
            return True, f"{spec['title']} already available."

        label_prefix = f"{prefix}: " if prefix else ""
        venv_pip = Path(sys.executable).parent / "pip"
        cmd = [str(venv_pip), *spec["install_args"], "--progress-bar", "off"]
        GLib.idle_add(
            self._emit_progress,
            job_key,
            f"{label_prefix}Installing {spec['title']} ({spec['size']})...",
            0.0,
        )
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            return False, f"{spec['title']} could not start: {exc}"

        last_line = ""
        if proc.stdout is not None:
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                last_line = line
                fraction = self._parse_progress_fraction(line)
                GLib.idle_add(
                    self._emit_progress,
                    job_key,
                    f"{label_prefix}{self._humanize_pip_line(line)}",
                    fraction,
                )

        return_code = proc.wait()
        if return_code != 0:
            msg = f"{spec['title']} install failed. {spec['help']}"
            if last_line:
                msg += f" Last output: {last_line[:160]}"
            return False, msg

        try:
            verified = bool(spec["verify"]())
        except Exception:
            verified = False

        if not verified:
            return False, f"{spec['title']} installed but verification failed. {spec['help']}"
        return True, f"{spec['title']} installed successfully."

    @staticmethod
    def _parse_progress_fraction(line: str) -> float:
        match = re.search(r"(\d{1,3})%", line)
        if not match:
            return -1.0
        return max(0.0, min(1.0, int(match.group(1)) / 100.0))

    @staticmethod
    def _humanize_pip_line(line: str) -> str:
        stripped = line.strip()
        if stripped.startswith("Collecting "):
            return stripped
        if stripped.startswith("Downloading "):
            return stripped
        if stripped.startswith("Installing collected packages"):
            return "Installing downloaded packages..."
        if stripped.startswith("Successfully installed "):
            return "Packages installed. Verifying runtime..."
        if stripped.startswith("Requirement already satisfied"):
            return stripped
        return stripped[:180]

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

import importlib
import importlib.util
import os
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import gi

gi.require_version("GObject", "2.0")
from gi.repository import GObject, GLib

from nvbroadcast.core.platform import (
    IS_ARM64,
    IS_LINUX,
    cuda_inference_probe_result,
    has_cuda_inference_runtime,
    has_tensorrt_runtime,
    preload_nvidia_runtime_libs,
    running_in_flatpak,
    supports_openai_whisper_python,
    supports_linux_gpu_stack,
    supports_tensorrt_python,
    tensorrt_inference_probe_result,
    tensorrt_python_unsupported_reason,
)
from nvbroadcast.runtime.probe import clear_runtime_probe_cache
from nvbroadcast.runtime.variants import (
    FASTER_WHISPER_REQUIREMENT,
    RuntimeVariant,
    detect_runtime_variant,
)


CUDA_RUNTIME_PACKAGES = [
    "cupy-cuda12x>=14.1.1,<15",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-curand-cu12",
    "nvidia-cufft-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-cuda-nvrtc-cu12",
]
CUDA_RUNTIME_HELP_PACKAGES = CUDA_RUNTIME_PACKAGES


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def _has_whisper() -> bool:
    try:
        if importlib.util.find_spec("faster_whisper") is not None:
            return True
    except Exception:
        pass

    # openai-whisper imports numba/llvmlite at module import time. On Python
    # 3.14+, a visible but incompatible system-site install can segfault during
    # an availability probe, so keep this check import-free and treat the
    # fallback backend as unavailable there.
    if not supports_openai_whisper_python():
        return False

    try:
        return importlib.util.find_spec("whisper") is not None
    except Exception:
        return False
    return False


def _supports_cuda_runtime() -> bool:
    return (
        supports_linux_gpu_stack()
        and detect_runtime_variant() is RuntimeVariant.CUDA
    )


def _running_in_snap() -> bool:
    return bool(os.environ.get("SNAP"))


def _running_in_native_package() -> bool:
    """Return whether Debian or RPM owns the current application environment."""
    return Path(sys.prefix) == Path("/opt/nvbroadcast/.venv")


def _cuda_runtime_unsupported_reason() -> str:
    """Explain how this installation can change its ONNX Runtime owner."""
    if not supports_linux_gpu_stack():
        return "CUDA modes are currently available only on Linux x86_64."
    if _running_in_native_package():
        return (
            "This package was installed with the CPU runtime variant. Stop "
            "NVBroadcast, make sure nvidia-smi detects the NVIDIA GPU, then "
            "reinstall or upgrade NVBroadcast through the system package manager "
            "to recreate its managed environment as CUDA."
        )
    if running_in_flatpak():
        return (
            "This Flatpak was built with the CPU runtime variant. Flatpak "
            "runtimes are immutable; update to a build that bundles CUDA or "
            "use another supported package format for CUDA modes."
        )
    return (
        "This environment uses the CPU runtime variant. Stop NVBroadcast and run "
        "./install.sh --runtime cuda, or recreate a user-owned source environment "
        "with .[cuda]."
    )


def _unsupported_reason(spec: dict, default: str) -> str:
    reason = spec.get("unsupported_reason", default)
    return reason() if callable(reason) else reason


def _runtime_install_block_reason() -> str | None:
    if _running_in_snap():
        return (
            "Optional runtimes cannot be installed inside the immutable Snap. "
            "Refresh the Snap to receive bundled runtime updates."
        )
    if running_in_flatpak():
        return (
            "Optional runtimes cannot be installed inside the immutable "
            "Flatpak. Update the Flatpak to receive bundled runtime changes."
        )

    venv_root = Path(sys.prefix)
    venv_pip = Path(sys.executable).parent / "pip"
    if (
        sys.prefix == getattr(sys, "base_prefix", sys.prefix)
        or not venv_pip.is_file()
        or not os.access(venv_pip, os.X_OK)
        or not os.access(venv_root, os.W_OK)
    ):
        return (
            "This app runtime is managed by its installer and cannot be changed "
            "from the GUI. Install optional runtimes through the same package or "
            "use a user-owned source installation."
        )
    return None


def _has_cuda_mode_runtime() -> bool:
    return _has_cupy() and has_cuda_inference_runtime()


def _supports_tensorrt_runtime() -> bool:
    return supports_linux_gpu_stack() and supports_tensorrt_python()


def _verify_cupy_result() -> tuple[bool, str]:
    try:
        preload_nvidia_runtime_libs()
        import cupy
        import numpy as np
        arr = cupy.asarray(np.ones((8, 8), dtype=np.float32))
        _ = (arr * 2.0).astype(cupy.float32)
        return True, ""
    except Exception as error:
        return (
            False,
            f"CuPy execution failed: {type(error).__name__}: {error}\n"
            f"{traceback.format_exc()}",
        )


def _verify_cupy() -> bool:
    return _verify_cupy_result()[0]


def _verify_cuda_mode_runtime_result() -> tuple[bool, str]:
    cupy_ok, cupy_detail = _verify_cupy_result()
    if not cupy_ok:
        return False, cupy_detail
    probe = cuda_inference_probe_result()
    return probe.success, probe.failure_detail


def _verify_cuda_mode_runtime() -> bool:
    return _verify_cuda_mode_runtime_result()[0]


def _verify_tensorrt_runtime_result() -> tuple[bool, str]:
    probe = tensorrt_inference_probe_result()
    return probe.success, probe.failure_detail


def _runtime_probe_failure_detail(package_id: str) -> str:
    if package_id == "cupy":
        probe = cuda_inference_probe_result()
    elif package_id == "tensorrt":
        probe = tensorrt_inference_probe_result()
    else:
        return ""
    return "" if probe.success else probe.failure_detail


def _with_probe_failure(message: str, package_id: str) -> str:
    detail = _runtime_probe_failure_detail(package_id).strip()
    if not detail:
        return message
    return f"{message}\n\nProvider probe details:\n{detail}"


PACKAGE_SPECS = {
    "cupy": {
        "title": "CUDA Mode Runtime",
        "subtitle": "Needed for DocZeus and CUDA modes",
        "size": "~2.0 GB",
        "summary": (
            "Installs CuPy and CUDA support wheels for an environment already "
            "owned by the CUDA ONNX Runtime variant."
        ),
        "install_args": ["install", "--upgrade", *CUDA_RUNTIME_PACKAGES],
        "supported": _supports_cuda_runtime,
        "check": _has_cuda_mode_runtime,
        "verify": _verify_cuda_mode_runtime_result,
        "requires_restart": True,
        "help": "Retry later with: .venv/bin/pip install --upgrade " + " ".join(CUDA_RUNTIME_HELP_PACKAGES),
        "unsupported_reason": _cuda_runtime_unsupported_reason,
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
        "supported": _supports_tensorrt_runtime,
        "check": lambda: has_tensorrt_runtime(),
        "verify": _verify_tensorrt_runtime_result,
        "requires_restart": True,
        "help": "Retry later with: .venv/bin/pip install tensorrt-cu12",
        "unsupported_reason": (
            "TensorRT premium modes are currently available only on Linux x86_64 "
            "with Python 3.8-3.13."
        ),
    },
    "whisper": {
        "title": "Meeting Transcription Runtime",
        "subtitle": "Needed for local meeting transcription",
        "size": "~350 MB",
        "summary": (
            "Installs the local meeting transcription runtime so Start Meeting "
            "can record and transcribe without sending audio anywhere."
        ),
        # Install faster-whisper without its CPU onnxruntime dependency so the
        # main app can keep the GPU ONNX Runtime package for video inference.
        # Install the rest normally so dependencies like av/httpcore/anyio are
        # resolved instead of being skipped by a broad --no-deps install.
        "install_steps": [
            ["install", "--no-deps", FASTER_WHISPER_REQUIREMENT],
            [
                "install",
                "ctranslate2",
                "huggingface-hub",
                "httpx",
                "tokenizers",
                "soundfile",
                "av",
                "tqdm",
            ],
        ],
        "supported": lambda: True,
        "check": _has_whisper,
        "verify": _has_whisper,
        "help": (
            "Retry later with: .venv/bin/pip install --no-deps "
            f"{FASTER_WHISPER_REQUIREMENT} && "
            ".venv/bin/pip install ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm"
        ),
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
        self._restart_pending_packages: set[str] = set()

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

    def is_supported(self, key: str) -> bool:
        if key in PACKAGE_BUNDLES:
            return all(self.is_supported(pkg) for pkg in PACKAGE_BUNDLES[key]["packages"])
        spec = PACKAGE_SPECS.get(key)
        if spec is None:
            return False
        supported = spec.get("supported")
        if supported is None:
            return True
        try:
            return bool(supported())
        except Exception:
            return False

    def describe(self, key: str) -> dict:
        if key in PACKAGE_BUNDLES:
            return PACKAGE_BUNDLES[key]
        return PACKAGE_SPECS[key]

    def _package_ids(self, key: str) -> tuple[str, ...]:
        if key in PACKAGE_BUNDLES:
            return tuple(PACKAGE_BUNDLES[key]["packages"])
        return (key,) if key in PACKAGE_SPECS else ()

    def restart_pending(self, key: str) -> bool:
        with self._lock:
            return any(
                package_id in self._restart_pending_packages
                for package_id in self._package_ids(key)
            )

    def _mark_restart_pending(self, package_id: str) -> None:
        if PACKAGE_SPECS[package_id].get("requires_restart"):
            with self._lock:
                self._restart_pending_packages.add(package_id)

    def _restart_pending_message(self, key: str) -> str:
        title = self.describe(key)["title"]
        return (
            f"The current app process cannot safely use {title} after an "
            "installation attempt. Restart NVBroadcast before selecting a mode "
            "that uses it."
        )

    def install_block_reason(self, key: str) -> str | None:
        if self.restart_pending(key):
            return self._restart_pending_message(key)
        if key in PACKAGE_BUNDLES:
            for package_id in PACKAGE_BUNDLES[key]["packages"]:
                reason = self.install_block_reason(package_id)
                if reason:
                    return reason
            return None

        spec = PACKAGE_SPECS.get(key)
        if spec is None:
            return "Unknown optional runtime."
        if self.is_available(key):
            return None
        if not self.is_supported(key):
            return _unsupported_reason(
                spec, f"{spec['title']} is not supported on this system."
            )
        return _runtime_install_block_reason()

    def unsupported_reason_for_mode(self, mode_key: str) -> str | None:
        pending_runtime = None
        if mode_key in (
            "doczeus",
            "cuda_max",
            "cuda_balanced",
            "cuda_perf",
            "zeus",
            "killer",
        ) and self.restart_pending("cupy"):
            pending_runtime = "cupy"
        if mode_key in ("zeus", "killer") and self.restart_pending("tensorrt"):
            pending_runtime = "tensorrt"
        if pending_runtime:
            return self._restart_pending_message(pending_runtime)
        if (
            IS_LINUX
            and IS_ARM64
            and mode_key in ("doczeus", "cuda_max", "cuda_balanced", "cuda_perf", "zeus", "killer")
        ):
            return "GPU CUDA and TensorRT modes are not available on Linux arm64 yet. Use CPU modes for now."
        if (
            _running_in_snap()
            and mode_key in ("doczeus", "cuda_max", "cuda_balanced", "cuda_perf", "zeus", "killer")
            and not _has_cuda_mode_runtime()
        ):
            return _with_probe_failure(
                (
                    "This Snap build cannot load the CUDA mode runtime on this "
                    "system. On amd64, refresh to the latest Snap; otherwise use "
                    "the .deb, .rpm, or source installer for CUDA GPU modes."
                ),
                "cupy",
            )
        if (
            running_in_flatpak()
            and mode_key in ("doczeus", "cuda_max", "cuda_balanced", "cuda_perf", "zeus", "killer")
            and not _has_cuda_mode_runtime()
        ):
            return _with_probe_failure(
                (
                    "CUDA modes are unavailable in this Flatpak build, and "
                    "Flatpak runtimes cannot be changed after deployment. "
                    "Update the Flatpak or use a mode whose runtime is bundled."
                ),
                "cupy",
            )
        if (
            mode_key in ("zeus", "killer")
            and not has_tensorrt_runtime()
            and not supports_tensorrt_python()
        ):
            return (
                f"{tensorrt_python_unsupported_reason()} "
                "Use DocZeus or the CUDA modes instead."
            )
        if (
            _running_in_snap()
            and mode_key in ("zeus", "killer")
            and not has_tensorrt_runtime()
        ):
            return _with_probe_failure(
                (
                    "TensorRT is unavailable in this Snap build, and strict Snaps "
                    "cannot install runtimes after deployment. Use DocZeus or a "
                    "CUDA mode instead."
                ),
                "tensorrt",
            )
        if (
            running_in_flatpak()
            and mode_key in ("zeus", "killer")
            and not has_tensorrt_runtime()
        ):
            return _with_probe_failure(
                (
                    "TensorRT is unavailable in this Flatpak build, and "
                    "Flatpak runtimes cannot be changed after deployment. "
                    "Use DocZeus or another bundled mode instead."
                ),
                "tensorrt",
            )
        required_runtimes: list[str] = []
        if mode_key in ("doczeus", "cuda_max", "cuda_balanced", "cuda_perf", "zeus", "killer"):
            required_runtimes.append("cupy")
        if mode_key in ("zeus", "killer"):
            required_runtimes.append("tensorrt")
        for package_id in required_runtimes:
            if self.is_available(package_id):
                continue
            block_reason = self.install_block_reason(package_id)
            if block_reason:
                return _with_probe_failure(
                    f"{block_reason} Use a mode whose runtime is already installed.",
                    package_id,
                )
        return None

    def missing_for_mode(self, mode_key: str) -> list[str]:
        if self.unsupported_reason_for_mode(mode_key):
            return []
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
        if self.install_block_reason(key):
            return False
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
                if self.restart_pending(key):
                    final_msg = (
                        f"{bundle['title']} installed; provider execution was "
                        "verified in fresh processes. Restart NVBroadcast to "
                        "activate it."
                    )
                else:
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
        if self.restart_pending(key) and "Restart NVBroadcast" not in message:
            message += (
                "\n\nA runtime installation was attempted. Restart NVBroadcast "
                "before retrying or selecting a GPU mode."
            )
        with self._lock:
            self._active_job_id = ""
            self._active_thread = None
        GLib.idle_add(self._emit_completed, key, success, message)

    def _install_single(self, job_key: str, package_id: str, prefix: str = "") -> tuple[bool, str]:
        spec = PACKAGE_SPECS[package_id]
        if self.is_available(package_id):
            return True, f"{spec['title']} already available."
        block_reason = self.install_block_reason(package_id)
        if block_reason:
            return False, block_reason

        label_prefix = f"{prefix}: " if prefix else ""
        install_steps = spec.get("install_steps") or [spec["install_args"]]
        GLib.idle_add(
            self._emit_progress,
            job_key,
            f"{label_prefix}Installing {spec['title']} ({spec['size']})...",
            0.0,
        )

        last_line = ""
        for step_index, install_args in enumerate(install_steps, start=1):
            step_prefix = ""
            if len(install_steps) > 1:
                step_prefix = f"Step {step_index}/{len(install_steps)}: "
            cmd = [sys.executable, "-m", "pip", *install_args, "--progress-bar", "off"]
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

            # A failed pip run can still leave a partially changed runtime.
            self._mark_restart_pending(package_id)
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
                        f"{label_prefix}{step_prefix}{self._humanize_pip_line(line)}",
                        fraction,
                    )

            return_code = proc.wait()
            if return_code != 0:
                msg = f"{spec['title']} install failed. {spec['help']}"
                if last_line:
                    msg += f" Last output: {last_line[:160]}"
                return False, msg

        clear_runtime_probe_cache()
        verification_detail = ""
        try:
            verification = spec["verify"]()
            if (
                isinstance(verification, tuple)
                and len(verification) == 2
            ):
                verified = bool(verification[0])
                verification_detail = str(verification[1] or "")
            else:
                verified = bool(verification)
        except Exception as error:
            verified = False
            verification_detail = (
                f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
            )

        if not verified:
            message = (
                f"{spec['title']} installed but verification failed. "
                f"{spec['help']}"
            )
            if verification_detail.strip():
                message += f"\n\nProbe details:\n{verification_detail.strip()}"
            return False, message
        if spec.get("requires_restart"):
            return (
                True,
                f"{spec['title']} installed; provider execution was verified in "
                "a fresh process. Restart NVBroadcast to activate it.",
            )
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

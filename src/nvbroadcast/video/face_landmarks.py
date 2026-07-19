# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Shared FaceLandmarker — single instance shared across all face effects.

Running 3 separate MediaPipe FaceLandmarkers (beautify, eye contact, relighting)
costs ~60-90ms per frame. Sharing one instance reduces it to ~20-30ms.
"""

import importlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from nvbroadcast.core.model_download import download_verified_model

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
_FACE_MODEL = "face_landmarker.task"
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
_MEDIAPIPE_IMPORT_ERROR: str | None = None
_MEDIAPIPE_READY: bool | None = None

# Singleton instance
_instance: Optional["SharedFaceLandmarker"] = None


def _probe_mediapipe_runtime() -> tuple[bool, str]:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    code = (
        "import mediapipe as mp\n"
        "from mediapipe.tasks.python import BaseOptions\n"
        "from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode\n"
        "print('ok')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
    except Exception as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    detail = (result.stderr or result.stdout or "").strip()
    return False, detail[:300] or f"exit code {result.returncode}"


def _load_mediapipe():
    global _MEDIAPIPE_IMPORT_ERROR, _MEDIAPIPE_READY
    if _MEDIAPIPE_READY is False:
        raise RuntimeError(_MEDIAPIPE_IMPORT_ERROR or "MediaPipe runtime unavailable")
    if _MEDIAPIPE_READY is None:
        ok, detail = _probe_mediapipe_runtime()
        if not ok:
            _MEDIAPIPE_READY = False
            _MEDIAPIPE_IMPORT_ERROR = detail or "MediaPipe import probe failed"
            raise RuntimeError(_MEDIAPIPE_IMPORT_ERROR)
        _MEDIAPIPE_READY = True

    mp = importlib.import_module("mediapipe")
    mp_python = importlib.import_module("mediapipe.tasks.python")
    mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
    return (
        mp,
        mp_python.BaseOptions,
        mp_vision.FaceLandmarker,
        mp_vision.FaceLandmarkerOptions,
        mp_vision.RunningMode,
    )


def get_shared_landmarker() -> "SharedFaceLandmarker":
    """Get or create the shared FaceLandmarker singleton."""
    global _instance
    if _instance is None:
        _instance = SharedFaceLandmarker()
    return _instance


class SharedFaceLandmarker:
    """Single FaceLandmarker shared across eye contact, relighting, and beautify.

    Call detect(bgra_frame) to get landmarks. Results are cached per frame
    (same frame pointer = cached result). Thread-safe via the GIL.
    """

    def __init__(self):
        self._landmarker = None
        self._initialized = False
        self._last_frame_id = None
        self._last_result = None
        self._frames_since_infer = 0
        self._lock = threading.Lock()
        self._detect_lock = threading.Lock()
        self._async_busy = False
        self._pending_job = None
        self._pending_frame_id = None
        self._worker_event = threading.Event()
        self._worker_stop = False
        self._worker_thread = None
        self._mp = None
        self._init()

    def _init(self):
        try:
            model_path = download_verified_model(
                _FACE_MODEL,
                _FACE_MODEL_URL,
                _FACE_MODEL_SHA256,
                bundled_dir=_MODELS_DIR,
            )
        except Exception as e:
            print(f"[FaceLandmarks] Download failed: {e}")
            return
        try:
            (
                self._mp,
                BaseOptions,
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            ) = _load_mediapipe()
            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = FaceLandmarker.create_from_options(opts)
            self._initialized = True
            self._worker_thread = threading.Thread(
                target=self._async_loop,
                name="nvbroadcast.face_landmarks",
                daemon=True,
            )
            self._worker_thread.start()
            print("[FaceLandmarks] Shared landmarker initialized")
        except Exception as e:
            print(f"[FaceLandmarks] Init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._initialized and self._landmarker is not None

    def latest(self):
        """Return the most recent landmark result without running inference."""
        with self._lock:
            return self._last_result

    def request_async(self, bgra_frame: np.ndarray, reuse_frames: int = 1):
        """Kick off background detection and return the latest available result.

        This keeps the live video path from blocking on MediaPipe while still
        refreshing landmarks in the background often enough for face effects.
        """
        if not self.ready:
            return None

        frame_id = id(bgra_frame)
        with self._lock:
            cached = self._last_result
            if frame_id == self._last_frame_id and cached is not None:
                return cached

            reuse_frames = max(1, int(reuse_frames))
            if (
                reuse_frames > 1
                and cached is not None
                and self._frames_since_infer < (reuse_frames - 1)
            ):
                self._frames_since_infer += 1
                self._last_frame_id = frame_id
                return cached

            if frame_id == self._pending_frame_id:
                return cached

        frame_copy = bgra_frame.copy()
        with self._lock:
            self._pending_job = frame_copy
            self._pending_frame_id = frame_id
            self._async_busy = True
            self._worker_event.set()
        return cached

    def detect(self, bgra_frame: np.ndarray, reuse_frames: int = 1):
        """Detect face landmarks. Returns list of landmarks or None.

        Results are cached per frame (by id) so multiple effects calling
        detect() on the same frame only run inference once.
        """
        if not self.ready:
            return None

        # Cache check — same frame object means same detection
        frame_id = id(bgra_frame)
        if frame_id == self._last_frame_id and self._last_result is not None:
            return self._last_result

        reuse_frames = max(1, int(reuse_frames))
        if (
            reuse_frames > 1
            and self._last_result is not None
            and self._frames_since_infer < (reuse_frames - 1)
        ):
            self._frames_since_infer += 1
            self._last_frame_id = frame_id
            return self._last_result

        with self._detect_lock:
            landmarks = self._run_detection(bgra_frame)
        with self._lock:
            self._last_frame_id = frame_id
            self._last_result = landmarks
            self._frames_since_infer = 0
        return landmarks

    def _async_loop(self):
        """Single background worker that always processes the newest frame."""
        while True:
            self._worker_event.wait()
            self._worker_event.clear()
            if self._worker_stop:
                return

            while True:
                with self._lock:
                    frame = self._pending_job
                    frame_id = self._pending_frame_id
                    self._pending_job = None
                    self._pending_frame_id = None
                if frame is None:
                    break

                with self._detect_lock:
                    landmarks = self._run_detection(frame)
                with self._lock:
                    self._last_frame_id = frame_id
                    self._last_result = landmarks
                    self._frames_since_infer = 0

            with self._lock:
                self._async_busy = False

    @staticmethod
    def _scale_detection_frame(bgra_frame: np.ndarray) -> np.ndarray:
        """Clamp large inference inputs without over-downscaling face crops."""
        h, w = bgra_frame.shape[:2]
        max_dim = max(w, h)
        if max_dim <= 512:
            return bgra_frame
        scale = 512.0 / float(max_dim)
        return cv2.resize(
            bgra_frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _detect_in_frame(self, bgra_frame: np.ndarray):
        scaled = self._scale_detection_frame(bgra_frame)
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGRA2RGB)
        ts = int(time.monotonic() * 1000)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect_for_video(mp_image, ts)
        except Exception:
            return None

        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def _run_detection(self, bgra_frame: np.ndarray):
        return self._detect_in_frame(bgra_frame)

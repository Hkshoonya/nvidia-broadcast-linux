# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Shared FaceLandmarker — single instance shared across all face effects.

Running 3 separate MediaPipe FaceLandmarkers (beautify, eye contact, relighting)
costs ~60-90ms per frame. Sharing one instance reduces it to ~20-30ms.
"""

import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
_FACE_MODEL = "face_landmarker.task"
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# Singleton instance
_instance: Optional["SharedFaceLandmarker"] = None


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
        self._init()

    def _init(self):
        model_path = _MODELS_DIR / _FACE_MODEL
        if not model_path.exists():
            try:
                _MODELS_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[FaceLandmarks] Downloading {_FACE_MODEL}...")
                urllib.request.urlretrieve(_FACE_MODEL_URL, str(model_path))
            except Exception as e:
                print(f"[FaceLandmarks] Download failed: {e}")
                return
        try:
            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = FaceLandmarker.create_from_options(opts)
            self._initialized = True
            print("[FaceLandmarks] Shared landmarker initialized")
        except Exception as e:
            print(f"[FaceLandmarks] Init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._initialized and self._landmarker is not None

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

        h, w = bgra_frame.shape[:2]
        if w >= 640 or h >= 360:
            scaled = cv2.resize(
                bgra_frame,
                (max(1, w // 2), max(1, h // 2)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scaled = bgra_frame

        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGRA2RGB)
        ts = int(time.monotonic() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect_for_video(mp_image, ts)
        except Exception:
            self._last_frame_id = frame_id
            self._last_result = None
            self._frames_since_infer = 0
            return None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            self._last_frame_id = frame_id
            self._last_result = landmarks
            self._frames_since_infer = 0
            return landmarks
        else:
            self._last_frame_id = frame_id
            self._last_result = None
            self._frames_since_infer = 0
            return None

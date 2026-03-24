# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Face relighting — adjusts face lighting to match background.

Uses MediaPipe FaceLandmarker (Tasks API) for face region detection,
analyzes background luminance, adjusts face brightness and warmth.
"""

import urllib.request
from pathlib import Path

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


class FaceRelighter:
    def __init__(self):
        self._enabled = False
        self._intensity = 0.5
        self._landmarker = None
        self._initialized = False
        self._bg_luminance = 128.0
        self._bg_warmth = 0.0
        self._frame_idx = 0
        self._analyze_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if value and not self._initialized:
            self._init_landmarker()

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, min(1.0, value))

    def _init_landmarker(self):
        model_path = _MODELS_DIR / _FACE_MODEL
        if not model_path.exists():
            try:
                _MODELS_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[Relighting] Downloading {_FACE_MODEL}...")
                urllib.request.urlretrieve(_FACE_MODEL_URL, str(model_path))
            except Exception as e:
                print(f"[Relighting] Model download failed: {e}")
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
            print("[Relighting] Initialized")
        except Exception as e:
            print(f"[Relighting] Init failed: {e}")

    def process_frame(self, frame: np.ndarray,
                      alpha: np.ndarray | None = None) -> np.ndarray:
        if not self._enabled or self._landmarker is None:
            return frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Analyze background every 10 frames
        self._analyze_count += 1
        if self._analyze_count % 10 == 0:
            self._analyze_background(frame, alpha)

        self._frame_idx += 1
        import time
        ts = int(time.monotonic() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            results = self._landmarker.detect_for_video(mp_image, ts)
        except Exception as e:
            if self._frame_idx <= 3:
                print(f"[Relighting] detect error: {e}")
            return frame

        if not results.face_landmarks:
            return frame

        landmarks = results.face_landmarks[0]

        # Build face mask from convex hull of all landmarks
        face_pts = np.array([
            (int(lm.x * w), int(lm.y * h))
            for lm in landmarks
        ], dtype=np.int32)

        hull = cv2.convexHull(face_pts)
        face_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(face_mask, hull, 1.0)
        face_mask = cv2.GaussianBlur(face_mask, (0, 0), sigmaX=8)

        # Calculate face luminance
        face_gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(float)
        masked_sum = (face_gray * face_mask).sum()
        mask_sum = face_mask.sum()
        if mask_sum < 1:
            return frame
        face_lum = masked_sum / mask_sum

        if face_lum < 1:
            return frame

        # Brightness adjustment — higher intensity = brighter/more matched face
        # intensity=0 → no change, intensity=1 → full match to background
        target = self._bg_luminance
        ratio = target / face_lum
        ratio = max(0.6, min(1.6, ratio))
        adj_ratio = 1.0 + (ratio - 1.0) * self._intensity

        output = frame.copy()
        adjusted = output[:, :, :3].astype(np.float32)

        for c in range(3):
            ch = adjusted[:, :, c]
            adjusted[:, :, c] = ch * (1 - face_mask) + (ch * adj_ratio) * face_mask

        # Warmth adjustment
        if abs(self._bg_warmth) > 0.02:
            warmth = self._bg_warmth * self._intensity * 15
            adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] + warmth * face_mask, 0, 255)
            adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] - warmth * 0.5 * face_mask, 0, 255)

        output[:, :, :3] = np.clip(adjusted, 0, 255).astype(np.uint8)
        return output

    def _analyze_background(self, frame: np.ndarray, alpha=None):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)

        if alpha is not None and hasattr(alpha, 'shape'):
            bg_mask = (alpha < 128).astype(np.float32)
            if bg_mask.sum() > 100:
                self._bg_luminance = float(np.average(gray, weights=bg_mask))
                bg_b = float(np.average(frame[:, :, 0].astype(float), weights=bg_mask))
                bg_r = float(np.average(frame[:, :, 2].astype(float), weights=bg_mask))
                self._bg_warmth = (bg_r - bg_b) / 255.0
                return

        # Fallback: sample frame borders
        border = min(40, h // 10, w // 10)
        regions = [gray[:border, :], gray[-border:, :],
                   gray[:, :border], gray[:, -border:]]
        self._bg_luminance = float(np.mean([r.mean() for r in regions]))

        border_r = float(np.mean([frame[:border, :, 2].mean(), frame[-border:, :, 2].mean()]))
        border_b = float(np.mean([frame[:border, :, 0].mean(), frame[-border:, :, 0].mean()]))
        self._bg_warmth = (border_r - border_b) / 255.0

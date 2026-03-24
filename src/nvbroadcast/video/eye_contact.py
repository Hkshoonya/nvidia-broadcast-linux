# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Eye contact correction — redirects gaze to look at camera.

Uses MediaPipe FaceLandmarker (Tasks API) to detect iris landmarks,
estimates gaze offset, and applies affine warp to eye regions.
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

_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]


class EyeContactCorrector:
    def __init__(self):
        self._enabled = False
        self._intensity = 0.7
        self._landmarker = None
        self._initialized = False
        self._frame_idx = 0

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
                print(f"[Eye Contact] Downloading {_FACE_MODEL}...")
                urllib.request.urlretrieve(_FACE_MODEL_URL, str(model_path))
            except Exception as e:
                print(f"[Eye Contact] Model download failed: {e}")
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
            print("[Eye Contact] Initialized")
        except Exception as e:
            print(f"[Eye Contact] Init failed: {e}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled or self._landmarker is None:
            return frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        self._frame_idx += 1
        # Use monotonic time in ms to avoid timestamp conflicts with other landmarkers
        import time
        ts = int(time.monotonic() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            results = self._landmarker.detect_for_video(mp_image, ts)
        except Exception as e:
            if self._frame_idx <= 3:
                print(f"[Eye Contact] detect error: {e}")
            return frame

        if not results.face_landmarks:
            return frame

        landmarks = results.face_landmarks[0]
        output = frame.copy()

        # Need 478 landmarks for iris (468 base + 10 iris)
        if len(landmarks) < 478:
            return frame

        for eye_idx, iris_idx in [(_LEFT_EYE, _LEFT_IRIS), (_RIGHT_EYE, _RIGHT_IRIS)]:
            output = self._correct_eye(output, landmarks, eye_idx, iris_idx, w, h)

        return output

    def _correct_eye(self, frame, landmarks, eye_indices, iris_indices,
                     img_w, img_h) -> np.ndarray:
        eye_pts = np.array([
            (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
            for i in eye_indices
        ], dtype=np.int32)

        x, y, ew, eh = cv2.boundingRect(eye_pts)
        pad = max(int(eh * 0.4), 6)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + ew + pad), min(img_h, y + eh + pad)
        if x2 - x1 < 10 or y2 - y1 < 8:
            return frame

        iris_pts = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in iris_indices
        ])
        iris_center = iris_pts.mean(axis=0)
        eye_center = eye_pts.astype(float).mean(axis=0)

        # Amplify gaze offset for more noticeable correction
        shift_x = -(iris_center[0] - eye_center[0]) * self._intensity * 1.8
        shift_y = -(iris_center[1] - eye_center[1]) * self._intensity * 0.8

        if abs(shift_x) < 0.3 and abs(shift_y) < 0.3:
            return frame

        eye_roi = frame[y1:y2, x1:x2].copy()
        roi_h, roi_w = eye_roi.shape[:2]

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped = cv2.warpAffine(eye_roi, M, (roi_w, roi_h),
                                borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        cx, cy = roi_w // 2, roi_h // 2
        axes = (max(1, roi_w // 2 - 2), max(1, roi_h // 2 - 2))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, pad * 0.4))
        mask = mask[:, :, np.newaxis]

        blended = (warped * mask + eye_roi * (1 - mask)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
        return frame

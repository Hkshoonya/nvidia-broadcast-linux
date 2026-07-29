# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Auto-frame: face tracking with smooth pan/zoom.

Uses MediaPipe Face Detection (Tasks API) for real-time face tracking,
with exponential moving average smoothing for stable framing.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2

from nvbroadcast.core.model_download import download_verified_model

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
_FACE_DETECTOR_MODEL = "blaze_face_short_range.tflite"
_FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_FACE_DETECTOR_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
_MEDIAPIPE_IMPORT_ERROR: str | None = None
_MEDIAPIPE_READY: bool | None = None
_MIN_CENTER_TRACKING_ZOOM = 1.12


def _probe_mediapipe_runtime() -> tuple[bool, str]:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    code = (
        "import mediapipe as mp\n"
        "from mediapipe.tasks import python as mp_python\n"
        "from mediapipe.tasks.python import vision as mp_vision\n"
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
    return mp, mp_python, mp_vision


class AutoFrame:
    """Face detection and auto-crop/zoom.

    Detects the primary face in each frame and smoothly pans/zooms
    to keep it centered.
    """

    def __init__(self, gpu_index: int = 1):
        self._gpu_index = gpu_index
        self._initialized = False
        self._detector = None
        self._enabled = False

        # Framing parameters
        self._mode = "center"
        self._zoom_level = 1.5
        self._smoothing = 0.85
        self._dead_zone = 0.015

        # Smoothed state
        self._smooth_cx = 0.5
        self._smooth_cy = 0.5
        self._smooth_zoom = 1.0
        self._no_face_frames = 0
        self._timestamp_ms = 0
        self._snap_to_face_on_next_frame = False
        self._mp = None
        self._mp_python = None
        self._mp_vision = None

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if value and not self._initialized:
            self.initialize()

    @property
    def zoom_level(self) -> float:
        return self._zoom_level

    @zoom_level.setter
    def zoom_level(self, value: float):
        self._zoom_level = max(1.0, min(3.0, value))

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        next_mode = value if value in ("center", "stable") else "center"
        if next_mode == self._mode:
            return
        self._mode = next_mode
        self._no_face_frames = 0
        if next_mode == "stable":
            self._snap_to_face_on_next_frame = False
            self._smooth_cx = 0.5
            self._smooth_cy = 0.5
        else:
            self._snap_to_face_on_next_frame = True

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: float):
        self._smoothing = max(0.0, min(0.99, value))

    def initialize(self) -> bool:
        """Initialize the face detection model."""
        if self._initialized:
            return True

        try:
            model_path = download_verified_model(
                _FACE_DETECTOR_MODEL,
                _FACE_DETECTOR_URL,
                _FACE_DETECTOR_SHA256,
                bundled_dir=_MODELS_DIR,
            )
        except Exception as e:
            print(f"[NV Broadcast] Failed to download face detection model: {e}")
            return False

        try:
            self._mp, self._mp_python, self._mp_vision = _load_mediapipe()
            base_options = self._mp_python.BaseOptions(
                model_asset_path=str(model_path),
            )
            options = self._mp_vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=self._mp_vision.RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
            self._detector = self._mp_vision.FaceDetector.create_from_options(options)
            self._initialized = True
            print("[NV Broadcast] Face detection initialized")
            return True
        except Exception as e:
            print(f"[NV Broadcast] Failed to initialize face detection: {e}")
            return False

    def process_frame(self, frame_data: bytes, width: int, height: int) -> bytes:
        """Detect face and apply auto-crop/zoom."""
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
        result = self.process_frame_array(frame, width, height)
        if result is frame:
            return frame_data
        return result.tobytes()

    def process_frame_array(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Detect face and apply auto-crop/zoom to a BGRA ndarray."""
        if not self._enabled or not self._initialized:
            return frame
        if not frame.flags.writeable:
            frame = frame.copy()

        face_box = self._detect_face(frame)

        if face_box is not None:
            self._no_face_frames = 0
            cx, cy = face_box

            if self._mode == "stable":
                cx, cy = 0.5, 0.5
            if self._snap_to_face_on_next_frame:
                self._smooth_cx = cx
                self._smooth_cy = cy
                self._snap_to_face_on_next_frame = False
            else:
                self._smooth_cx = self._smooth_axis(self._smooth_cx, cx)
                self._smooth_cy = self._smooth_axis(self._smooth_cy, cy)

            target_zoom = self._target_zoom()
            self._smooth_zoom = (self._smoothing * self._smooth_zoom +
                                 (1 - self._smoothing) * target_zoom)
        else:
            self._no_face_frames += 1
            if self._no_face_frames > 30:
                self._smooth_cx = self._smoothing * self._smooth_cx + (1 - self._smoothing) * 0.5
                self._smooth_cy = self._smoothing * self._smooth_cy + (1 - self._smoothing) * 0.5
                self._smooth_zoom = (self._smoothing * self._smooth_zoom +
                                     (1 - self._smoothing) * 1.0)

        return self._crop_and_zoom(frame, width, height)

    def _smooth_axis(self, current: float, target: float) -> float:
        """Smooth visible face motion while ignoring detector jitter."""
        if abs(target - current) <= self._dead_zone:
            return current
        return self._smoothing * current + (1 - self._smoothing) * target

    def _target_zoom(self) -> float:
        if self._mode == "center":
            return max(self._zoom_level, _MIN_CENTER_TRACKING_ZOOM)
        return self._zoom_level

    def _detect_face(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Detect the primary face. Returns (center_x, center_y) normalized [0,1]."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

            self._timestamp_ms += 33  # ~30fps
            result = self._detector.detect_for_video(mp_image, self._timestamp_ms)

            if not result.detections:
                return None

            # Take highest confidence detection
            best = max(result.detections, key=lambda d: d.categories[0].score)
            bbox = best.bounding_box

            # Center of face (normalized to frame dimensions)
            h, w = frame.shape[:2]
            cx = (bbox.origin_x + bbox.width / 2) / w
            cy = (bbox.origin_y + bbox.height / 2) / h

            return (cx, cy)

        except Exception:
            return None

    def _crop_and_zoom(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Crop and zoom centered on the smoothed face position."""
        zoom = max(1.0, self._smooth_zoom)

        crop_w = int(width / zoom)
        crop_h = int(height / zoom)

        center_x = int(self._smooth_cx * width)
        center_y = int(self._smooth_cy * height)

        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(width, x1 + crop_w)
        y2 = min(height, y1 + crop_h)

        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)

        cropped = frame[y1:y2, x1:x2]
        result = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        return result

    def cleanup(self):
        if self._detector:
            self._detector.close()
            self._detector = None
        self._initialized = False

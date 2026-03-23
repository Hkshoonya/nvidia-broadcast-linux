# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Auto-frame: face tracking with smooth pan/zoom.

Uses MediaPipe Face Detection (Tasks API) for real-time face tracking,
with exponential moving average smoothing for stable framing.
"""

from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


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
        self._zoom_level = 1.5
        self._smoothing = 0.85
        self._dead_zone = 0.15

        # Smoothed state
        self._smooth_cx = 0.5
        self._smooth_cy = 0.5
        self._smooth_zoom = 1.0
        self._no_face_frames = 0
        self._timestamp_ms = 0

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
    def smoothing(self) -> float:
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: float):
        self._smoothing = max(0.0, min(0.99, value))

    def initialize(self) -> bool:
        """Initialize the face detection model."""
        if self._initialized:
            return True

        model_path = _MODELS_DIR / "blaze_face_short_range.tflite"
        if not model_path.exists():
            # Try to download
            try:
                import urllib.request
                model_path.parent.mkdir(parents=True, exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
                urllib.request.urlretrieve(url, str(model_path))
            except Exception as e:
                print(f"[NVIDIA Broadcast] Failed to download face detection model: {e}")
                return False

        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_path),
            )
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
            self._detector = mp_vision.FaceDetector.create_from_options(options)
            self._initialized = True
            print("[NVIDIA Broadcast] Face detection initialized")
            return True
        except Exception as e:
            print(f"[NVIDIA Broadcast] Failed to initialize face detection: {e}")
            return False

    def process_frame(self, frame_data: bytes, width: int, height: int) -> bytes:
        """Detect face and apply auto-crop/zoom."""
        if not self._enabled or not self._initialized:
            return frame_data

        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4).copy()

        face_box = self._detect_face(frame)

        if face_box is not None:
            self._no_face_frames = 0
            cx, cy = face_box

            dx = abs(cx - self._smooth_cx)
            dy = abs(cy - self._smooth_cy)

            if dx > self._dead_zone or dy > self._dead_zone:
                self._smooth_cx = self._smoothing * self._smooth_cx + (1 - self._smoothing) * cx
                self._smooth_cy = self._smoothing * self._smooth_cy + (1 - self._smoothing) * cy

            self._smooth_zoom = (self._smoothing * self._smooth_zoom +
                                 (1 - self._smoothing) * self._zoom_level)
        else:
            self._no_face_frames += 1
            if self._no_face_frames > 30:
                self._smooth_cx = self._smoothing * self._smooth_cx + (1 - self._smoothing) * 0.5
                self._smooth_cy = self._smoothing * self._smooth_cy + (1 - self._smoothing) * 0.5
                self._smooth_zoom = (self._smoothing * self._smooth_zoom +
                                     (1 - self._smoothing) * 1.0)

        result = self._crop_and_zoom(frame, width, height)
        return result.tobytes()

    def _detect_face(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Detect the primary face. Returns (center_x, center_y) normalized [0,1]."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

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

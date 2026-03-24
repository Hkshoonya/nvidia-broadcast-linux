# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Face relighting — adjusts face lighting to match background.

Uses shared FaceLandmarker for face region detection.
Analyzes background luminance and adjusts face brightness/warmth.
"""

import numpy as np
import cv2

from nvbroadcast.video.face_landmarks import get_shared_landmarker


class FaceRelighter:
    def __init__(self):
        self._enabled = False
        self._intensity = 0.5
        self._bg_luminance = 128.0
        self._bg_warmth = 0.0
        self._analyze_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, min(1.0, value))

    def process_frame(self, frame: np.ndarray,
                      alpha: np.ndarray | None = None) -> np.ndarray:
        if not self._enabled:
            return frame

        lm = get_shared_landmarker()
        if not lm.ready:
            return frame

        h, w = frame.shape[:2]

        # Analyze background every 10 frames
        self._analyze_count += 1
        if self._analyze_count % 10 == 0:
            self._analyze_background(frame, alpha)

        landmarks = lm.detect(frame)
        if landmarks is None:
            return frame

        # Build face mask from convex hull
        face_pts = np.array([
            (int(l.x * w), int(l.y * h))
            for l in landmarks
        ], dtype=np.int32)

        hull = cv2.convexHull(face_pts)
        face_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(face_mask, hull, 1.0)
        face_mask = cv2.GaussianBlur(face_mask, (0, 0), sigmaX=8)

        # Face luminance
        face_gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(float)
        mask_sum = face_mask.sum()
        if mask_sum < 1:
            return frame
        face_lum = (face_gray * face_mask).sum() / mask_sum
        if face_lum < 1:
            return frame

        # Brightness adjustment
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

        border = min(40, h // 10, w // 10)
        regions = [gray[:border, :], gray[-border:, :],
                   gray[:, :border], gray[:, -border:]]
        self._bg_luminance = float(np.mean([r.mean() for r in regions]))

        border_r = float(np.mean([frame[:border, :, 2].mean(), frame[-border:, :, 2].mean()]))
        border_b = float(np.mean([frame[:border, :, 0].mean(), frame[-border:, :, 0].mean()]))
        self._bg_warmth = (border_r - border_b) / 255.0

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Eye contact correction — redirects gaze to look at camera.

Uses shared FaceLandmarker for efficient per-frame landmark detection.
Detects iris position, estimates gaze offset, applies affine warp.
"""

import numpy as np
import cv2

from nvbroadcast.video.face_landmarks import get_shared_landmarker

_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]


class EyeContactCorrector:
    def __init__(self):
        self._enabled = False
        self._intensity = 0.7

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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return frame

        lm = get_shared_landmarker()
        if not lm.ready:
            return frame

        landmarks = lm.detect(frame)
        if landmarks is None or len(landmarks) < 478:
            return frame

        h, w = frame.shape[:2]
        output = frame.copy()

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

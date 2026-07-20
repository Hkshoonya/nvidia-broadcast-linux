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
        self._intensity = 0.45
        self._smoothed_shifts: dict[tuple[int, ...], np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = bool(value)
        if not self._enabled:
            self._smoothed_shifts.clear()

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, min(1.0, value))
        if self._intensity == 0.0:
            self._smoothed_shifts.clear()

    def process_frame(self, frame: np.ndarray, landmarks=None) -> np.ndarray:
        if not self._enabled:
            return frame

        if landmarks is None:
            lm = get_shared_landmarker()
            if not lm.ready:
                self._smoothed_shifts.clear()
                return frame
            landmarks = lm.detect(frame, reuse_frames=2)
        if landmarks is None or len(landmarks) < 478:
            self._smoothed_shifts.clear()
            return frame

        h, w = frame.shape[:2]
        output = frame.copy()

        for eye_idx, iris_idx in [(_LEFT_EYE, _LEFT_IRIS), (_RIGHT_EYE, _RIGHT_IRIS)]:
            output = self._correct_eye(output, landmarks, eye_idx, iris_idx, w, h)

        return output

    def _correct_eye(self, frame, landmarks, eye_indices, iris_indices,
                     img_w, img_h) -> np.ndarray:
        shift_key = tuple(iris_indices)
        if frame.ndim != 3 or frame.shape[2] < 3:
            self._smoothed_shifts.pop(shift_key, None)
            return frame

        eye_pts_float = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in eye_indices
        ], dtype=np.float32)
        eye_pts = np.rint(eye_pts_float).astype(np.int32)

        x, y, ew, eh = cv2.boundingRect(eye_pts)
        eye_width = float(np.ptp(eye_pts_float[:, 0]))
        eye_height = float(np.ptp(eye_pts_float[:, 1]))
        if eye_width < 8.0 or eye_height < 2.0:
            self._smoothed_shifts.pop(shift_key, None)
            return frame

        pad = max(int(round(eye_width * 0.12)), int(round(eye_height * 0.75)), 4)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + ew + pad), min(img_h, y + eh + pad)
        if x2 - x1 < 10 or y2 - y1 < 8:
            self._smoothed_shifts.pop(shift_key, None)
            return frame

        iris_pts = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in iris_indices
        ], dtype=np.float32)
        iris_center = iris_pts.mean(axis=0)
        eye_center = np.array(
            (
                (eye_pts_float[:, 0].min() + eye_pts_float[:, 0].max()) * 0.5,
                (eye_pts_float[:, 1].min() + eye_pts_float[:, 1].max()) * 0.5,
            ),
            dtype=np.float32,
        )

        # Reject blinks and unstable detections before strengthening the gaze
        # movement. The float geometry avoids treating small, open eyes as
        # closed only because their integer bounding box is a few pixels high.
        eye_ratio = eye_height / max(eye_width, 1.0)
        if eye_ratio < 0.12:
            self._smoothed_shifts.pop(shift_key, None)
            return frame

        delta_x = iris_center[0] - eye_center[0]
        delta_y = iris_center[1] - eye_center[1]
        if abs(delta_x) > eye_width * 0.38 or abs(delta_y) > eye_height * 0.45:
            self._smoothed_shifts.pop(shift_key, None)
            return frame

        # The old path reduced vertical movement to 20% and horizontal movement
        # to 75%, then applied the user's intensity again. At the default 0.35
        # that often produced a sub-pixel no-op. Square-root response preserves
        # fine control while making the middle of the slider useful; the caps
        # keep the result within the visible eye opening.
        strength = float(np.sqrt(self._intensity))
        shift_x = float(np.clip(
            -delta_x * strength,
            -max(1.0, eye_width * 0.18),
            max(1.0, eye_width * 0.18),
        ))
        shift_y = float(np.clip(
            -delta_y * strength,
            -max(0.75, eye_height * 0.22),
            max(0.75, eye_height * 0.22),
        ))

        if abs(shift_x) < 0.2 and abs(shift_y) < 0.2:
            self._smoothed_shifts.pop(shift_key, None)
            return frame

        target_shift = np.array((shift_x, shift_y), dtype=np.float32)
        previous_shift = self._smoothed_shifts.get(shift_key)
        if previous_shift is not None:
            target_shift = previous_shift * 0.55 + target_shift * 0.45
        self._smoothed_shifts[shift_key] = target_shift
        shift_x, shift_y = map(float, target_shift)

        eye_roi = frame[y1:y2, x1:x2].copy()
        roi_h, roi_w = eye_roi.shape[:2]

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped = cv2.warpAffine(
            eye_roi[:, :, :3],
            M,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Feather the shifted pixels inside the actual eyelid contour. This
        # moves the iris and sclera together while anchoring eyelids, glasses,
        # and surrounding skin at the boundary.
        mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        local_eye_pts = np.rint(
            eye_pts_float - np.array((x1, y1), dtype=np.float32)
        ).astype(np.int32)
        hull = cv2.convexHull(local_eye_pts)
        cv2.fillConvexPoly(mask, hull, 1.0)
        sigma = max(0.6, min(2.5, eye_height * 0.18))
        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        mask = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis]

        original_color = eye_roi[:, :, :3]
        blended = (
            warped.astype(np.float32) * mask
            + original_color.astype(np.float32) * (1.0 - mask)
        ).astype(np.uint8)
        frame[y1:y2, x1:x2, :3] = blended
        return frame

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Eye contact correction - redirects gaze toward a shared camera target."""

from dataclasses import dataclass

import cv2
import numpy as np

from nvbroadcast.video.face_landmarks import get_shared_landmarker

_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]
_NOSE_TIP = 1
_EYE_CONTACT_MODES = ("natural", "gaze_lock")


@dataclass(frozen=True)
class _EyeGeometry:
    eye_points: np.ndarray
    eye_center: np.ndarray
    normalized_offset: np.ndarray
    width: float
    height: float
    bounds: tuple[int, int, int, int]
    correction_weight: float


class EyeContactCorrector:
    def __init__(self):
        self._enabled = False
        self._intensity = 0.45
        self._mode = "natural"
        self._smoothed_correction: np.ndarray | None = None
        self._previous_disparity: np.ndarray | None = None
        self._locked_camera_target: np.ndarray | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = bool(value)
        if not self._enabled:
            self._reset_tracking()

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, min(1.0, value))
        if self._intensity == 0.0:
            self._reset_tracking()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        mode = value if value in _EYE_CONTACT_MODES else "natural"
        if mode != self._mode:
            self._mode = mode
            self._reset_tracking()

    def _reset_tracking(self) -> None:
        self._smoothed_correction = None
        self._previous_disparity = None
        self._locked_camera_target = None

    def process_frame(self, frame: np.ndarray, landmarks=None) -> np.ndarray:
        if not self._enabled or self._intensity == 0.0:
            return frame
        if frame.ndim != 3 or frame.shape[2] < 3:
            self._reset_tracking()
            return frame

        if landmarks is None:
            lm = get_shared_landmarker()
            if not lm.ready:
                self._reset_tracking()
                return frame
            landmarks = lm.detect(frame, reuse_frames=1)
        if landmarks is None or len(landmarks) < 478:
            self._reset_tracking()
            return frame

        img_h, img_w = frame.shape[:2]
        eyes = [
            self._measure_eye(landmarks, eye_indices, iris_indices, img_w, img_h)
            for eye_indices, iris_indices in (
                (_LEFT_EYE, _LEFT_IRIS),
                (_RIGHT_EYE, _RIGHT_IRIS),
            )
        ]

        # An asymmetric correction during a blink or bad detection is more
        # distracting than skipping one frame, so only move a valid eye pair.
        if any(eye is None for eye in eyes):
            self._reset_tracking()
            return frame

        valid_eyes = [eye for eye in eyes if eye is not None]
        shifts = self._binocular_shifts(valid_eyes, landmarks, img_w)
        if shifts is None:
            self._reset_tracking()
            return frame

        output = frame.copy()
        for eye, shift in zip(valid_eyes, shifts):
            self._warp_eye(output, eye, shift)
        return output

    def _measure_eye(self, landmarks, eye_indices, iris_indices,
                     img_w: int, img_h: int) -> _EyeGeometry | None:
        eye_points = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in eye_indices
        ], dtype=np.float32)
        iris_points = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in iris_indices
        ], dtype=np.float32)
        if not np.isfinite(eye_points).all() or not np.isfinite(iris_points).all():
            return None

        eye_width = float(np.ptp(eye_points[:, 0]))
        eye_height = float(np.ptp(eye_points[:, 1]))
        if eye_width < 8.0 or eye_height < 2.0:
            return None

        eye_ratio = eye_height / max(eye_width, 1.0)
        if eye_ratio < 0.12:
            return None

        rounded_points = np.rint(eye_points).astype(np.int32)
        x, y, width, height = cv2.boundingRect(rounded_points)
        pad = max(
            int(round(eye_width * 0.12)),
            int(round(eye_height * 0.75)),
            4,
        )
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + width + pad), min(img_h, y + height + pad)
        if x2 - x1 < 10 or y2 - y1 < 8:
            return None

        eye_center = np.array(
            (
                (eye_points[:, 0].min() + eye_points[:, 0].max()) * 0.5,
                (eye_points[:, 1].min() + eye_points[:, 1].max()) * 0.5,
            ),
            dtype=np.float32,
        )
        iris_center = iris_points.mean(axis=0)
        normalized_offset = np.array(
            (
                (iris_center[0] - eye_center[0]) / eye_width,
                (iris_center[1] - eye_center[1]) / eye_height,
            ),
            dtype=np.float32,
        )
        correction_weight = min(
            self._smooth_falloff(abs(float(normalized_offset[0])), 0.28, 0.44),
            self._smooth_falloff(abs(float(normalized_offset[1])), 0.34, 0.55),
        )

        return _EyeGeometry(
            eye_points=eye_points,
            eye_center=eye_center,
            normalized_offset=normalized_offset,
            width=eye_width,
            height=eye_height,
            bounds=(x1, y1, x2, y2),
            correction_weight=correction_weight,
        )

    @staticmethod
    def _smooth_falloff(value: float, start: float, end: float) -> float:
        if value <= start:
            return 1.0
        if value >= end:
            return 0.0
        progress = (value - start) / (end - start)
        smoothstep = progress * progress * (3.0 - 2.0 * progress)
        return 1.0 - smoothstep

    @staticmethod
    def _head_pose_target(landmarks, eyes: list[_EyeGeometry],
                          img_w: int) -> np.ndarray:
        interocular = abs(float(eyes[1].eye_center[0] - eyes[0].eye_center[0]))
        if interocular < 8.0 or len(landmarks) <= _NOSE_TIP:
            return np.zeros(2, dtype=np.float32)

        nose_x = float(landmarks[_NOSE_TIP].x * img_w)
        eye_midpoint = float(
            (eyes[0].eye_center[0] + eyes[1].eye_center[0]) * 0.5
        )
        yaw = (nose_x - eye_midpoint) / interocular
        if not np.isfinite(yaw):
            return np.zeros(2, dtype=np.float32)

        # Ignore normal landmark asymmetry near frontal, then turn the eyes
        # opposite the measured head yaw to keep the shared target at camera.
        yaw = np.sign(yaw) * max(0.0, abs(yaw) - 0.03)
        target_x = float(np.clip(-yaw * 0.32, -0.08, 0.08))
        return np.array((target_x, 0.0), dtype=np.float32)

    def _stabilize_camera_target(self, camera_target: np.ndarray) -> np.ndarray:
        if self._locked_camera_target is None:
            self._locked_camera_target = camera_target.astype(np.float32)
        else:
            delta = camera_target - self._locked_camera_target
            deadzone = np.array((0.01, 0.015), dtype=np.float32)
            if np.any(np.abs(delta) > deadzone):
                self._locked_camera_target = (
                    self._locked_camera_target + delta * 0.45
                ).astype(np.float32)
        return self._locked_camera_target

    def _binocular_shifts(self, eyes: list[_EyeGeometry], landmarks,
                          img_w: int) -> list[np.ndarray] | None:
        pair_weight = min(eye.correction_weight for eye in eyes)
        if pair_weight <= 0.0:
            return None

        offsets = np.stack([eye.normalized_offset for eye in eyes])
        widths = np.array([eye.width for eye in eyes], dtype=np.float32)
        shared_gaze = np.average(offsets, axis=0, weights=widths)

        disparity = offsets[0] - offsets[1]
        if self._previous_disparity is None:
            agreement = 1.0
        else:
            disparity_change = np.abs(disparity - self._previous_disparity)
            instability = max(
                float(disparity_change[0]) / 0.06,
                float(disparity_change[1]) / 0.08,
            )
            agreement = float(np.clip(1.0 - instability, 0.0, 1.0))
        self._previous_disparity = disparity.astype(np.float32)

        camera_target = self._head_pose_target(landmarks, eyes, img_w)
        if self._mode == "gaze_lock":
            camera_target = self._stabilize_camera_target(camera_target)
        raw_correction = camera_target - shared_gaze
        if self._smoothed_correction is None:
            smoothed_correction = raw_correction
        else:
            # Consistent binocular motion can follow quickly. Disagreement is
            # smoothed more heavily so one noisy iris does not pull the pair.
            if self._mode == "gaze_lock":
                response = 0.30 + agreement * 0.70
            else:
                response = 0.40 + agreement * 0.40
            smoothed_correction = (
                self._smoothed_correction * (1.0 - response)
                + raw_correction * response
            )
        self._smoothed_correction = smoothed_correction.astype(np.float32)

        if self._mode == "gaze_lock":
            strength = float(np.power(self._intensity, 0.25))
        else:
            strength = float(np.sqrt(self._intensity))
        confidence = 0.65 + agreement * 0.35
        eye_midpoint = float(
            (eyes[0].eye_center[0] + eyes[1].eye_center[0]) * 0.5
        )
        shifts = []
        for eye in eyes:
            outward = -1.0 if eye.eye_center[0] < eye_midpoint else 1.0
            residual = eye.normalized_offset - shared_gaze
            # Preserve most person-specific binocular disparity. Only damp the
            # small unstable part instead of independently centering both eyes.
            normalized_shift = (
                smoothed_correction
                + np.array((outward * 0.025, 0.0), dtype=np.float32)
                - residual * 0.15
            )
            normalized_shift *= strength * pair_weight * confidence

            shift = np.array(
                (
                    np.clip(
                        normalized_shift[0] * eye.width,
                        -max(1.0, eye.width * 0.22),
                        max(1.0, eye.width * 0.22),
                    ),
                    np.clip(
                        normalized_shift[1] * eye.height,
                        -max(0.75, eye.height * 0.22),
                        max(0.75, eye.height * 0.22),
                    ),
                ),
                dtype=np.float32,
            )
            shifts.append(shift)
        return shifts

    @staticmethod
    def _warp_eye(frame: np.ndarray, eye: _EyeGeometry,
                  shift: np.ndarray) -> None:
        shift_x, shift_y = map(float, shift)
        if abs(shift_x) < 0.2 and abs(shift_y) < 0.2:
            return

        x1, y1, x2, y2 = eye.bounds
        eye_roi = frame[y1:y2, x1:x2].copy()
        roi_h, roi_w = eye_roi.shape[:2]
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped = cv2.warpAffine(
            eye_roi[:, :, :3],
            matrix,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        local_eye_points = np.rint(
            eye.eye_points - np.array((x1, y1), dtype=np.float32)
        ).astype(np.int32)
        hull = cv2.convexHull(local_eye_points)
        cv2.fillConvexPoly(mask, hull, 1.0)
        sigma = max(0.6, min(2.5, eye.height * 0.18))
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mask = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis]

        original_color = eye_roi[:, :, :3]
        blended = (
            warped.astype(np.float32) * mask
            + original_color.astype(np.float32) * (1.0 - mask)
        ).astype(np.uint8)
        frame[y1:y2, x1:x2, :3] = blended

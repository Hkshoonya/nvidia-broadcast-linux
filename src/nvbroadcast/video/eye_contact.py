"""Eye contact correction — redirects gaze to look at camera.

Uses MediaPipe face mesh (468 landmarks) to detect iris position,
estimates gaze offset, and applies affine warp to eye regions
to simulate direct eye contact with the camera.
"""

import numpy as np
import cv2

# MediaPipe eye landmark indices
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_IRIS = [468, 469, 470, 471, 472]   # MediaPipe iris landmarks (refine=True)
_RIGHT_IRIS = [473, 474, 475, 476, 477]


class EyeContactCorrector:
    """Corrects eye gaze to look directly at camera."""

    def __init__(self):
        self._enabled = False
        self._intensity = 0.7  # 0.0 = no correction, 1.0 = full
        self._face_mesh = None
        self._frame_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if value and self._face_mesh is None:
            self._init_face_mesh()

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, min(1.0, value))

    def _init_face_mesh(self):
        """Initialize MediaPipe face mesh with iris refinement."""
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # Enable iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"[Eye Contact] MediaPipe init failed: {e}")
            self._face_mesh = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply eye contact correction to a BGRA frame."""
        if not self._enabled or self._face_mesh is None:
            return frame

        h, w = frame.shape[:2]
        # Convert BGRA to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Run face mesh (every frame for smooth tracking)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return frame

        landmarks = results.multi_face_landmarks[0].landmark
        output = frame.copy()

        # Process each eye
        for eye_indices, iris_indices in [
            (_LEFT_EYE, _LEFT_IRIS),
            (_RIGHT_EYE, _RIGHT_IRIS),
        ]:
            output = self._correct_eye(
                output, landmarks, eye_indices, iris_indices, w, h
            )

        return output

    def _correct_eye(self, frame, landmarks, eye_indices, iris_indices,
                     img_w, img_h) -> np.ndarray:
        """Correct gaze for a single eye using affine warp."""
        # Get eye region bounding box with padding
        eye_pts = np.array([
            (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
            for i in eye_indices
        ], dtype=np.int32)

        x, y, ew, eh = cv2.boundingRect(eye_pts)
        pad = max(int(eh * 0.4), 6)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + ew + pad)
        y2 = min(img_h, y + eh + pad)

        if x2 - x1 < 10 or y2 - y1 < 8:
            return frame

        # Get iris center
        iris_pts = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in iris_indices
        ])
        iris_center = iris_pts.mean(axis=0)

        # Get eye center (midpoint of eye outline)
        eye_center = eye_pts.astype(float).mean(axis=0)

        # Calculate gaze offset (how far iris is from eye center)
        offset_x = iris_center[0] - eye_center[0]
        offset_y = iris_center[1] - eye_center[1]

        # Apply correction (shift iris toward center)
        shift_x = -offset_x * self._intensity * 0.6
        shift_y = -offset_y * self._intensity * 0.3  # Less vertical correction

        if abs(shift_x) < 0.5 and abs(shift_y) < 0.5:
            return frame  # Already looking at camera

        # Extract eye region
        eye_roi = frame[y1:y2, x1:x2].copy()
        roi_h, roi_w = eye_roi.shape[:2]

        # Create affine warp (translation)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped = cv2.warpAffine(
            eye_roi, M, (roi_w, roi_h),
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Create smooth blend mask (elliptical, feathered)
        mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        center = (roi_w // 2, roi_h // 2)
        axes = (roi_w // 2 - 2, roi_h // 2 - 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=pad * 0.4)
        mask = mask[:, :, np.newaxis]

        # Blend warped eye into original
        blended = (warped * mask + eye_roi * (1 - mask)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended

        return frame

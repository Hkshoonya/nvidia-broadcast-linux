"""Face relighting — adjusts face lighting to match background.

Uses MediaPipe face landmarks to identify face zones (forehead, cheeks,
chin), analyzes background luminance, and adjusts face zone brightness
and color temperature to create natural-looking lighting.
"""

import numpy as np
import cv2

# Face zone landmark indices (MediaPipe 468-point mesh)
_FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
_LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
_RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
_CHIN = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
         127, 162, 21, 54, 103, 67, 109, 10]


class FaceRelighter:
    """Adjusts face lighting to match scene/background."""

    def __init__(self):
        self._enabled = False
        self._intensity = 0.5
        self._face_mesh = None
        self._bg_luminance = 128  # Cached background brightness
        self._bg_warmth = 0.0     # Cached background color temperature shift
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
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"[Relighting] MediaPipe init failed: {e}")

    def process_frame(self, frame: np.ndarray,
                      alpha: np.ndarray | None = None) -> np.ndarray:
        """Apply face relighting to a BGRA frame.

        Args:
            frame: BGRA frame
            alpha: Optional alpha mask from background removal (0-255).
                   If provided, uses it to analyze background vs foreground.
        """
        if not self._enabled or self._face_mesh is None:
            return frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Analyze background luminance every 10 frames
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            self._analyze_background(frame, alpha)

        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return frame

        landmarks = results.multi_face_landmarks[0].landmark
        output = frame.copy()

        # Get face region mask
        face_pts = np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in range(468)
        ], dtype=np.int32)

        # Create face mask from convex hull
        hull = cv2.convexHull(face_pts)
        face_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(face_mask, hull, 1.0)
        face_mask = cv2.GaussianBlur(face_mask, (0, 0), sigmaX=8)

        # Calculate target adjustment
        face_roi = frame[:, :, :3]  # BGR
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(float)

        # Face luminance (masked average)
        face_lum = np.average(face_gray, weights=face_mask)
        if face_lum < 1:
            return frame

        # Target: bring face luminance closer to background
        target_lum = self._bg_luminance
        ratio = target_lum / face_lum
        # Clamp ratio to avoid extreme adjustments
        ratio = max(0.7, min(1.4, ratio))

        # Blend toward target based on intensity
        adj_ratio = 1.0 + (ratio - 1.0) * self._intensity

        # Apply brightness adjustment to face region
        adjusted = output[:, :, :3].astype(np.float32)
        for c in range(3):
            channel = adjusted[:, :, c]
            channel_adj = channel * adj_ratio
            # Smooth blend using face mask
            adjusted[:, :, c] = channel * (1 - face_mask) + channel_adj * face_mask

        # Apply warmth adjustment
        if abs(self._bg_warmth) > 0.02:
            warmth = self._bg_warmth * self._intensity * 15
            adjusted[:, :, 2] = np.clip(
                adjusted[:, :, 2] + warmth * face_mask, 0, 255
            )  # Red channel
            adjusted[:, :, 0] = np.clip(
                adjusted[:, :, 0] - warmth * 0.5 * face_mask, 0, 255
            )  # Blue channel

        output[:, :, :3] = np.clip(adjusted, 0, 255).astype(np.uint8)
        return output

    def _analyze_background(self, frame: np.ndarray,
                           alpha: np.ndarray | None = None):
        """Analyze background brightness and color temperature."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)

        if alpha is not None:
            # Use alpha to isolate background (where alpha < 128)
            bg_mask = (alpha < 128).astype(np.float32)
            if bg_mask.sum() > 100:
                self._bg_luminance = np.average(gray, weights=bg_mask)
                # Analyze warmth from background
                bg_b = np.average(frame[:, :, 0].astype(float), weights=bg_mask)
                bg_r = np.average(frame[:, :, 2].astype(float), weights=bg_mask)
                self._bg_warmth = (bg_r - bg_b) / 255.0
                return

        # Fallback: sample edges of frame (likely background)
        border = 40
        regions = [
            gray[:border, :],           # Top
            gray[-border:, :],          # Bottom
            gray[:, :border],           # Left
            gray[:, -border:],          # Right
        ]
        self._bg_luminance = np.mean([r.mean() for r in regions])

        # Warmth from borders
        border_r = np.mean([frame[r, :, 2].mean() for r in [
            slice(0, border), slice(-border, None)
        ]])
        border_b = np.mean([frame[r, :, 0].mean() for r in [
            slice(0, border), slice(-border, None)
        ]])
        self._bg_warmth = (border_r - border_b) / 255.0

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Face beautification — skin smoothing, denoising, edge darkening, enhancement.

Uses MediaPipe FaceLandmarker for precise face region detection.
All effects are independently togglable with intensity control.
Zero CPU cost when disabled.
"""

import urllib.request
from pathlib import Path

import numpy as np
import cv2

from nvbroadcast.video.face_landmarks import get_shared_landmarker

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
_FACE_MODEL = "face_landmarker.task"
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# Face oval landmark indices (MediaPipe face mesh)
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Forehead region (above eyes)
_FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]

# Cheek regions (left and right)
_LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
_RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]


class FaceBeautifier:
    """Real-time face beautification with per-effect intensity control.

    Effects:
    - Skin smoothing: Bilateral filter on face skin regions
    - Denoising: Fast temporal + spatial denoise on full frame
    - Edge darkening: Subtle vignette centered on face
    - Face enhancement: Brightness, contrast, warmth boost on face
    - Eye/lip sharpening: Unsharp mask on detail regions
    """

    def __init__(self, compositing: str = "cpu"):
        self._initialized = False
        self._compositing = "cpu"
        self._cupy = None
        if compositing != "cpu":
            self.set_compositing(compositing)

        # Master toggle
        self._enabled = False

        # Individual effect toggles and intensities (0.0 = off, 1.0 = max)
        self._skin_smooth = 0.0
        self._denoise = 0.0
        self._edge_darken = 0.0
        self._enhance = 0.0
        self._sharpen = 0.0

        # Cached face data
        self._face_mask = None
        self._face_bbox = None
        self._face_center = None
        self._prev_frame = None  # For temporal denoising
        self._vignette_cache = None  # Cached vignette gradient
        self._vignette_size = None
        self._frame_counter = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if value and not self._initialized:
            self.initialize()

    @property
    def skin_smooth(self) -> float:
        return self._skin_smooth

    @skin_smooth.setter
    def skin_smooth(self, v: float):
        self._skin_smooth = max(0.0, min(1.0, v))

    @property
    def denoise(self) -> float:
        return self._denoise

    @denoise.setter
    def denoise(self, v: float):
        self._denoise = max(0.0, min(1.0, v))

    @property
    def edge_darken(self) -> float:
        return self._edge_darken

    @edge_darken.setter
    def edge_darken(self, v: float):
        self._edge_darken = max(0.0, min(1.0, v))

    @property
    def enhance(self) -> float:
        return self._enhance

    @enhance.setter
    def enhance(self, v: float):
        self._enhance = max(0.0, min(1.0, v))

    @property
    def sharpen(self) -> float:
        return self._sharpen

    @sharpen.setter
    def sharpen(self, v: float):
        self._sharpen = max(0.0, min(1.0, v))

    def initialize(self) -> bool:
        """Initialize FaceLandmarker model."""
        if self._initialized:
            return True

        model_path = _MODELS_DIR / _FACE_MODEL
        if not model_path.exists():
            try:
                _MODELS_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[NV Broadcast] Downloading {_FACE_MODEL}...")
                urllib.request.urlretrieve(_FACE_MODEL_URL, str(model_path))
            except Exception as e:
                print(f"[NV Broadcast] Failed to download face model: {e}")
                return False

        try:
            landmarker = get_shared_landmarker()
            self._initialized = landmarker.ready
            if self._initialized:
                print("[NV Broadcast] Face beautifier initialized")
            return self._initialized
        except Exception as e:
            print(f"[NV Broadcast] Face beautifier init failed: {e}")
            return False

    def set_compositing(self, backend: str):
        """Set compositing backend (cpu, cupy, gstreamer_gl)."""
        self._compositing = backend
        if backend in ("cupy", "gstreamer_gl") and self._cupy is None:
            try:
                import cupy
                self._cupy = cupy
            except ImportError:
                if backend == "cupy":
                    self._compositing = "cpu"

    def process_frame(self, frame_data: bytes, width: int, height: int,
                      landmarks=None) -> bytes:
        """Apply beautification effects to a BGRA frame."""
        if not self._enabled or not self._initialized:
            return frame_data

        # Any effect active?
        if (self._skin_smooth <= 0 and self._denoise <= 0 and
                self._edge_darken <= 0 and self._enhance <= 0 and
                self._sharpen <= 0):
            return frame_data

        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
        if not frame.flags.writeable:
            frame = frame.copy()

        self._frame_counter += 1

        # Refresh mask periodically. Landmark inference itself is shared across
        # all face effects, so this only rebuilds the beautify mask.
        if self._frame_counter % 5 == 0 or self._face_mask is None:
            self._detect_face(frame, width, height, landmarks)

        # CPU-only operations first (bilateral has no GPU equivalent)
        if self._denoise > 0:
            frame = self._apply_denoise(frame)

        if self._skin_smooth > 0 and self._face_mask is not None:
            frame = self._apply_skin_smooth(frame)

        # Batch GPU-eligible operations (enhance + sharpen + vignette)
        if self._cupy is not None:
            frame = self._apply_gpu_batch(frame, width, height)
        else:
            if self._enhance > 0 and self._face_mask is not None:
                frame = self._apply_enhance(frame)
            if self._sharpen > 0 and self._face_mask is not None:
                frame = self._apply_sharpen(frame)
            if self._edge_darken > 0:
                frame = self._apply_edge_darken(frame, width, height)

        return frame.tobytes()

    def _apply_gpu_batch(self, frame: np.ndarray,
                         width: int, height: int) -> np.ndarray:
        """Batch enhance + sharpen + vignette on GPU in one transfer."""
        cp = self._cupy
        try:
            bgr = frame[:, :, :3]
            # Single upload to GPU
            f_gpu = cp.asarray(bgr, dtype=cp.float32)

            # Enhance on GPU
            if self._enhance > 0 and self._face_mask is not None:
                intensity = self._enhance
                mask_gpu = cp.asarray(self._face_mask, dtype=cp.float32)[:, :, cp.newaxis] / 255.0 * intensity

                enhanced = (f_gpu - 128) * (1.0 + intensity * 0.2) + 128 + intensity * 15
                enhanced[:, :, 2] += intensity * 8       # Red warmth
                enhanced[:, :, 1] += intensity * 8 * 0.3  # Green warmth
                cp.clip(enhanced, 0, 255, out=enhanced)

                f_gpu = f_gpu * (1 - mask_gpu) + enhanced * mask_gpu

            # Sharpen on GPU (unsharp mask approximation)
            if self._sharpen > 0 and self._face_mask is not None and self._face_bbox is not None:
                intensity = self._sharpen
                x, y, w, h = self._face_bbox
                pad = 10
                y1, y2 = max(0, y - pad), min(height, y + h + pad)
                x1, x2 = max(0, x - pad), min(width, x + w + pad)

                # Simple high-pass sharpen on ROI
                roi = f_gpu[y1:y2, x1:x2]
                # Average as blur approximation (fast on GPU)
                blurred = cp.mean(roi.reshape(-1, roi.shape[-1]), axis=0)
                amount = 0.5 + intensity * 1.5
                roi_sharp = roi + (roi - blurred) * amount * 0.1
                mask_roi = cp.asarray(self._face_mask[y1:y2, x1:x2], dtype=cp.float32)[:, :, cp.newaxis] / 255.0 * intensity
                f_gpu[y1:y2, x1:x2] = roi * (1 - mask_roi) + cp.clip(roi_sharp, 0, 255) * mask_roi

            # Vignette on GPU
            if self._edge_darken > 0:
                if self._vignette_cache is None or self._vignette_size != (width, height):
                    self._build_vignette_cache(width, height)
                vig = cp.asarray(self._vignette_cache)
                v_scaled = (1.0 - self._edge_darken) + self._edge_darken * vig
                f_gpu = f_gpu * v_scaled[:, :, cp.newaxis]

            # Single download from GPU
            frame[:, :, :3] = cp.asnumpy(cp.clip(f_gpu, 0, 255).astype(cp.uint8))
            return frame

        except Exception as e:
            if self._frame_counter <= 2:
                print(f"[NV Broadcast] GPU beautify failed, using CPU: {e}")
            self._compositing = "cpu"
            # Fallback to CPU path
            if self._enhance > 0 and self._face_mask is not None:
                frame = self._apply_enhance(frame)
            if self._sharpen > 0 and self._face_mask is not None:
                frame = self._apply_sharpen(frame)
            if self._edge_darken > 0:
                frame = self._apply_edge_darken(frame, width, height)
            return frame

    def _build_vignette_cache(self, width: int, height: int):
        """Build vignette gradient and cache it."""
        if self._face_center:
            cx, cy = self._face_center
        else:
            cx, cy = width // 2, height // 2
        Y, X = np.ogrid[:height, :width]
        dx = (X - cx) / (width * 0.5)
        dy = (Y - cy) / (height * 0.5)
        dist = np.sqrt(dx * dx + dy * dy)
        self._vignette_cache = np.clip(1.0 - (dist - 0.3) * 0.8, 0.3, 1.0).astype(np.float32)
        self._vignette_size = (width, height)

    def _detect_face(self, frame: np.ndarray, width: int, height: int,
                     landmarks=None):
        """Build a beautify mask from shared face landmarks."""
        try:
            if landmarks is None:
                shared = get_shared_landmarker()
                if not shared.ready:
                    self._face_mask = None
                    self._face_bbox = None
                    return
                landmarks = shared.detect(frame, reuse_frames=2)

            if not landmarks:
                self._face_mask = None
                self._face_bbox = None
                return

            # Build face oval mask from landmarks
            pts = np.array([
                (int(landmarks[i].x * width), int(landmarks[i].y * height))
                for i in _FACE_OVAL_INDICES
            ], dtype=np.int32)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)

            # Feather edges with Gaussian blur for smooth blending
            self._face_mask = cv2.GaussianBlur(mask, (21, 21), 0)

            # Bounding box for ROI optimization
            x, y, w, h = cv2.boundingRect(pts)
            self._face_bbox = (x, y, w, h)

            # Face center for vignette
            cx = int(np.mean([landmarks[i].x for i in _FACE_OVAL_INDICES]) * width)
            cy = int(np.mean([landmarks[i].y for i in _FACE_OVAL_INDICES]) * height)
            self._face_center = (cx, cy)

        except Exception:
            self._face_mask = None

    def _apply_skin_smooth(self, frame: np.ndarray) -> np.ndarray:
        """Bilateral filter on face ROI only — fast, preserves edges."""
        intensity = self._skin_smooth
        mask = self._face_mask
        bbox = self._face_bbox
        if bbox is None:
            return frame

        # Only process the face bounding box (not full frame)
        x, y, w, h = bbox
        pad = 20  # Padding around face
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)

        roi = frame[y1:y2, x1:x2, :3]
        roi_mask = mask[y1:y2, x1:x2]

        # Small kernel (d=5 = 2ms vs d=10 = 20ms)
        d = 5 if intensity < 0.6 else 7
        sigma = int(30 + intensity * 40)
        smoothed = cv2.bilateralFilter(roi, d, sigma, sigma)

        # Blend using face mask (ROI only)
        mask_f = (roi_mask.astype(np.float32) / 255.0 * intensity)[:, :, np.newaxis]
        frame[y1:y2, x1:x2, :3] = np.clip(
            roi.astype(np.float32) * (1 - mask_f) +
            smoothed.astype(np.float32) * mask_f, 0, 255
        ).astype(np.uint8)

        return frame

    def _apply_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Fast temporal + spatial denoising using cv2 (SIMD-optimized)."""
        intensity = self._denoise
        bgr = frame[:, :, :3]

        # Temporal denoising: cv2.addWeighted is SIMD-fast (~1ms)
        if self._prev_frame is not None:
            weight = 0.1 + intensity * 0.2
            bgr = cv2.addWeighted(bgr, 1.0 - weight, self._prev_frame, weight, 0)

        self._prev_frame = bgr.copy()

        # Spatial denoising: light blur only at higher intensities
        if intensity > 0.3:
            k = 3 if intensity < 0.7 else 5
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)

        frame[:, :, :3] = bgr
        return frame

    def _apply_edge_darken(self, frame: np.ndarray,
                           width: int, height: int) -> np.ndarray:
        """Vignette effect — cached gradient, cv2 SIMD multiply."""
        intensity = self._edge_darken

        if (self._vignette_cache is None or
                self._vignette_size != (width, height) or
                self._frame_counter % 10 == 0):
            self._build_vignette_cache(width, height)

        vignette = (1.0 - intensity) + intensity * self._vignette_cache
        v3 = cv2.merge([vignette, vignette, vignette])
        frame[:, :, :3] = cv2.multiply(
            frame[:, :, :3], v3, scale=1.0, dtype=cv2.CV_8U
        )
        return frame

    def _apply_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Brightness, contrast, and warmth boost on face ROI.

        Uses direct BGR math instead of LAB conversion (saves ~12ms).
        """
        intensity = self._enhance
        mask = self._face_mask
        bbox = self._face_bbox
        if bbox is None:
            return frame

        x, y, w, h = bbox
        pad = 20
        y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)

        roi = frame[y1:y2, x1:x2, :3].astype(np.float32)
        roi_mask = mask[y1:y2, x1:x2]

        # Brightness + contrast in one operation
        brightness = intensity * 15
        contrast = 1.0 + intensity * 0.2
        enhanced = (roi - 128) * contrast + 128 + brightness

        # Warmth: boost red slightly, boost green very slightly
        warmth = intensity * 8
        enhanced[:, :, 2] += warmth        # Red
        enhanced[:, :, 1] += warmth * 0.3  # Green (less)

        enhanced = np.clip(enhanced, 0, 255)

        # Blend using face mask (ROI only)
        mask_f = (roi_mask.astype(np.float32) / 255.0 * intensity)[:, :, np.newaxis]
        frame[y1:y2, x1:x2, :3] = np.clip(
            roi * (1 - mask_f) + enhanced * mask_f, 0, 255
        ).astype(np.uint8)

        return frame

    def _apply_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Unsharp mask on face ROI for crisper eyes and lips."""
        intensity = self._sharpen
        mask = self._face_mask
        bbox = self._face_bbox
        if bbox is None:
            return frame

        x, y, w, h = bbox
        pad = 10
        y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)

        roi = frame[y1:y2, x1:x2, :3]
        roi_mask = mask[y1:y2, x1:x2]

        # Unsharp mask on ROI only
        blurred = cv2.GaussianBlur(roi, (0, 0), 3)
        amount = 0.5 + intensity * 1.5
        sharpened = cv2.addWeighted(roi, 1 + amount, blurred, -amount, 0)

        mask_f = (roi_mask.astype(np.float32) / 255.0 * intensity)[:, :, np.newaxis]
        frame[y1:y2, x1:x2, :3] = np.clip(
            roi.astype(np.float32) * (1 - mask_f) +
            sharpened.astype(np.float32) * mask_f, 0, 255
        ).astype(np.uint8)

        return frame

    def cleanup(self):
        """Release resources."""
        self._initialized = False
        self._face_mask = None
        self._prev_frame = None

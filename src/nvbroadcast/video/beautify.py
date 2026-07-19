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

import numpy as np
import cv2

from nvbroadcast.video.face_landmarks import get_shared_landmarker

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
    - Denoising: Motion-aware temporal + spatial denoise on face ROI
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
        self._tone_mask = None
        self._face_bbox = None
        self._face_center = None
        self._face_motion_px = 0.0
        self._prev_frame = None  # For temporal denoising
        self._vignette_cache = None  # Cached vignette gradient
        self._vignette_size = None
        self._vignette_center = None
        self._vignette_rgb_cache = None
        self._vignette_rgb_intensity = None
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

    def process_frame(
        self,
        frame_data: bytes,
        width: int,
        height: int,
        landmarks=None,
        allow_inline_landmarks: bool = True,
        skip_enhance: bool = False,
        skip_edge_darken: bool = False,
        cache_prepared: bool = False,
    ) -> bytes:
        """Apply beautification effects to a BGRA frame."""
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
        result = self.process_frame_array(
            frame,
            width,
            height,
            landmarks=landmarks,
            allow_inline_landmarks=allow_inline_landmarks,
            skip_enhance=skip_enhance,
            skip_edge_darken=skip_edge_darken,
            cache_prepared=cache_prepared,
        )
        if result is frame:
            return frame_data
        return result.tobytes()

    def process_frame_array(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        landmarks=None,
        allow_inline_landmarks: bool = True,
        skip_enhance: bool = False,
        skip_edge_darken: bool = False,
        cache_prepared: bool = False,
    ) -> np.ndarray:
        """Apply beautification effects to a BGRA frame ndarray."""
        if not self._enabled or not self._initialized:
            return frame

        # Any effect active?
        if (self._skin_smooth <= 0 and self._denoise <= 0 and
                (self._edge_darken <= 0 or skip_edge_darken) and
                (self._enhance <= 0 or skip_enhance) and
                self._sharpen <= 0):
            return frame
        if not frame.flags.writeable:
            frame = frame.copy()

        if not cache_prepared:
            self._frame_counter += 1

            # Refresh mask periodically. Landmark inference itself is shared across
            # all face effects, so this only rebuilds the beautify mask.
            if self._frame_counter % 5 == 0 or self._face_mask is None:
                self._detect_face(
                    frame,
                    width,
                    height,
                    landmarks,
                    allow_inline_landmarks=allow_inline_landmarks,
                )

        # CPU-only operations first (bilateral has no GPU equivalent)
        if self._denoise > 0:
            frame = self._apply_denoise(frame)

        if self._skin_smooth > 0 and self._face_mask is not None:
            frame = self._apply_skin_smooth(frame)

        # Batch GPU-eligible operations (enhance + sharpen + vignette)
        if self._cupy is not None:
            frame = self._apply_gpu_batch(
                frame,
                width,
                height,
                apply_enhance=not skip_enhance,
                apply_edge_darken=not skip_edge_darken,
            )
        else:
            if not skip_enhance and self._enhance > 0 and self._face_mask is not None:
                frame = self._apply_enhance(frame)
            if self._sharpen > 0 and self._face_mask is not None:
                frame = self._apply_sharpen(frame)
            if not skip_edge_darken and self._edge_darken > 0:
                frame = self._apply_edge_darken(frame, width, height)

        return frame

    def prime_face_cache(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        landmarks=None,
        allow_inline_landmarks: bool = True,
    ) -> None:
        """Refresh cached face geometry without applying beautify effects."""
        if not self._enabled or not self._initialized:
            return
        self._frame_counter += 1
        if self._frame_counter % 5 == 0 or self._face_mask is None:
            self._detect_face(
                frame,
                width,
                height,
                landmarks,
                allow_inline_landmarks=allow_inline_landmarks,
            )

    def fused_overlay_inputs(self, width: int, height: int):
        """Return cached overlay data for fused background compositing."""
        if not self._enabled or self._face_mask is None:
            return None
        if self._enhance <= 0 and self._edge_darken <= 0:
            return None
        if self._edge_darken > 0:
            current_center = self._face_center or (width // 2, height // 2)
            center_shift = 0
            if self._vignette_center is not None:
                center_shift = max(
                    abs(current_center[0] - self._vignette_center[0]),
                    abs(current_center[1] - self._vignette_center[1]),
                )
            if (
                self._vignette_cache is None
                or self._vignette_size != (width, height)
                or center_shift > max(8, min(width, height) // 64)
            ):
                self._build_vignette_cache(width, height)
        brightness = self._enhance * 15.0
        contrast = self._enhance * 0.2
        warmth = self._enhance * 8.0
        tone_mask = self._tone_mask if self._tone_mask is not None else self._face_mask
        return (
            tone_mask,
            self._vignette_cache,
            float(self._enhance),
            float(self._edge_darken),
            float(brightness),
            float(contrast),
            float(warmth),
        )

    def _apply_gpu_batch(self, frame: np.ndarray,
                         width: int, height: int,
                         apply_enhance: bool = True,
                         apply_edge_darken: bool = True) -> np.ndarray:
        """Batch ROI-local enhance/sharpen on GPU; keep full-frame vignette on CPU.

        Uploading the whole frame for beautify adds a large round-trip cost at
        meeting resolutions. The actual face adjustments are local to the face
        ROI, so only that region goes to the GPU. The subtle vignette stays on
        the CPU path where cv2 multiply is already efficient.
        """
        cp = self._cupy
        try:
            bbox = self._face_bbox
            detail_mask = self._face_mask
            tone_mask = self._tone_mask if self._tone_mask is not None else detail_mask
            if bbox is None or detail_mask is None:
                if apply_edge_darken and self._edge_darken > 0:
                    return self._apply_edge_darken(frame, width, height)
                return frame

            x, y, w, h = bbox
            pad = 20
            y1, y2 = max(0, y - pad), min(height, y + h + pad)
            x1, x2 = max(0, x - pad), min(width, x + w + pad)
            if x2 <= x1 or y2 <= y1:
                if apply_edge_darken and self._edge_darken > 0:
                    return self._apply_edge_darken(frame, width, height)
                return frame

            roi = frame[y1:y2, x1:x2, :3]
            roi_gpu = cp.asarray(roi, dtype=cp.float32)
            detail_roi_mask = detail_mask[y1:y2, x1:x2]
            tone_roi_mask = tone_mask[y1:y2, x1:x2] if tone_mask is not None else detail_roi_mask

            if apply_enhance and self._enhance > 0:
                intensity = self._enhance
                mask_gpu = cp.asarray(tone_roi_mask, dtype=cp.float32)[:, :, cp.newaxis] / 255.0 * intensity
                enhanced = (roi_gpu - 128) * (1.0 + intensity * 0.2) + 128 + intensity * 15
                enhanced[:, :, 2] += intensity * 8
                enhanced[:, :, 1] += intensity * 8 * 0.3
                cp.clip(enhanced, 0, 255, out=enhanced)
                roi_gpu = roi_gpu * (1 - mask_gpu) + enhanced * mask_gpu

            if self._sharpen > 0:
                intensity = self._sharpen
                blurred = cp.mean(roi_gpu.reshape(-1, roi_gpu.shape[-1]), axis=0)
                amount = 0.5 + intensity * 1.5
                roi_sharp = roi_gpu + (roi_gpu - blurred) * amount * 0.1
                mask_gpu = cp.asarray(detail_roi_mask, dtype=cp.float32)[:, :, cp.newaxis] / 255.0 * intensity
                roi_gpu = roi_gpu * (1 - mask_gpu) + cp.clip(roi_sharp, 0, 255) * mask_gpu

            frame[y1:y2, x1:x2, :3] = cp.asnumpy(cp.clip(roi_gpu, 0, 255).astype(cp.uint8))
            if apply_edge_darken and self._edge_darken > 0:
                frame = self._apply_edge_darken(frame, width, height)
            return frame

        except Exception as e:
            if self._frame_counter <= 2:
                print(f"[NV Broadcast] GPU beautify failed, using CPU: {e}")
            self._compositing = "cpu"
            # Fallback to CPU path
            if apply_enhance and self._enhance > 0 and self._face_mask is not None:
                frame = self._apply_enhance(frame)
            if self._sharpen > 0 and self._face_mask is not None:
                frame = self._apply_sharpen(frame)
            if apply_edge_darken and self._edge_darken > 0:
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
        self._vignette_center = (cx, cy)
        self._vignette_rgb_cache = None
        self._vignette_rgb_intensity = None

    def _detect_face(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        landmarks=None,
        allow_inline_landmarks: bool = True,
    ):
        """Build a beautify mask from shared face landmarks."""
        try:
            if landmarks is None:
                if not allow_inline_landmarks:
                    return
                shared = get_shared_landmarker()
                if not shared.ready:
                    self._face_mask = None
                    self._tone_mask = None
                    self._face_bbox = None
                    return
                landmarks = shared.detect(frame, reuse_frames=2)

            if not landmarks:
                self._face_mask = None
                self._tone_mask = None
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
            # Bounding box for ROI optimization
            x, y, w, h = cv2.boundingRect(pts)
            self._face_bbox = (x, y, w, h)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = self._apply_hairline_taper(mask, y, h)
            self._face_mask = mask
            self._tone_mask = self._build_tone_mask(mask, y, h)

            # Face center for vignette
            prev_center = self._face_center
            cx = int(np.mean([landmarks[i].x for i in _FACE_OVAL_INDICES]) * width)
            cy = int(np.mean([landmarks[i].y for i in _FACE_OVAL_INDICES]) * height)
            self._face_center = (cx, cy)
            if prev_center is None:
                self._face_motion_px = 0.0
            else:
                self._face_motion_px = float(
                    np.hypot(cx - prev_center[0], cy - prev_center[1])
                )

        except Exception:
            self._face_mask = None
            self._tone_mask = None

    @staticmethod
    def _apply_hairline_taper(mask: np.ndarray, top: int, face_height: int) -> np.ndarray:
        """Fade face effects out near the upper hairline.

        The face oval and blur feathering can leak slightly into the hair above
        the forehead. Apply a vertical taper so skin-focused effects stay on the
        face and forehead while backing off before the hairline.
        """
        if mask is None or face_height < 8:
            return mask
        height = mask.shape[0]
        start = max(0, top)
        end = min(height, top + max(6, int(face_height * 0.24)))
        if end <= start:
            return mask
        tapered = mask.astype(np.float32)
        ramp = np.linspace(0.30, 1.0, end - start, dtype=np.float32)[:, np.newaxis]
        tapered[start:end, :] *= ramp
        return np.clip(tapered, 0, 255).astype(np.uint8)

    @staticmethod
    def _build_tone_mask(mask: np.ndarray, top: int, face_height: int) -> np.ndarray:
        """Build a tighter mask for tone-changing face effects.

        Brightness/warmth/smoothing should stay lower on the face than the
        broader detail mask used for sharpening. This pulls those effects away
        from the upper hairline and top side hair.
        """
        if mask is None:
            return None
        tone_mask = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tone_mask = cv2.erode(tone_mask, kernel, iterations=1)

        height, width = tone_mask.shape[:2]
        start = max(0, top)
        end = min(height, top + max(8, int(face_height * 0.38)))
        if end > start:
            tapered = tone_mask.astype(np.float32)
            ramp = np.linspace(0.0, 1.0, end - start, dtype=np.float32)[:, np.newaxis]
            tapered[start:end, :] *= ramp

            upper_end = min(height, top + max(6, int(face_height * 0.26)))
            if upper_end > start:
                x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
                side_ramp = 0.70 + 0.30 * (1.0 - np.abs(x))
                tapered[start:upper_end, :] *= side_ramp[np.newaxis, :]
            tone_mask = np.clip(tapered, 0, 255).astype(np.uint8)

        tone_mask = cv2.GaussianBlur(tone_mask, (11, 11), 0)
        return tone_mask

    def _apply_skin_smooth(self, frame: np.ndarray) -> np.ndarray:
        """Bilateral filter on face ROI only — fast, preserves edges."""
        intensity = self._skin_smooth
        mask = self._tone_mask if self._tone_mask is not None else self._face_mask
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
        # When the face is moving quickly, smoothing contributes less visible
        # value than edge freshness. Back off instead of spending the full
        # bilateral cost on motion-heavy frames.
        motion_fast = self._face_motion_px > max(10.0, min(w, h) * 0.12)
        motion_very_fast = self._face_motion_px > max(18.0, min(w, h) * 0.20)
        if motion_very_fast and intensity <= 0.45:
            return frame
        if motion_fast:
            intensity *= 0.55
            d = 3
            sigma = int(20 + intensity * 20)
        else:
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
        """Face-local temporal denoising that avoids recursive motion smear."""
        intensity = self._denoise
        raw_bgr = frame[:, :, :3].copy()

        bbox = self._face_bbox
        mask = self._tone_mask if self._tone_mask is not None else self._face_mask
        if bbox is None or mask is None:
            # Keep raw history current, but do not blur the whole composited frame.
            self._prev_frame = raw_bgr
            return frame

        x, y, w, h = bbox
        pad = 16
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)
        if x2 <= x1 or y2 <= y1:
            self._prev_frame = raw_bgr
            return frame

        roi = raw_bgr[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]
        denoised = roi.copy()

        if self._face_motion_px > max(12.0, min(w, h) * 0.14):
            self._prev_frame = raw_bgr
            return frame

        if self._prev_frame is not None and self._prev_frame.shape == raw_bgr.shape:
            prev_roi = self._prev_frame[y1:y2, x1:x2]
            diff = cv2.absdiff(roi, prev_roi)
            motion = float(diff.mean()) * (1.0 / 255.0)
            motion_gate = float(np.clip(1.0 - motion * 8.0, 0.15, 1.0))
            weight = (0.06 + intensity * 0.16) * motion_gate
            if weight > 0.01:
                denoised = cv2.addWeighted(roi, 1.0 - weight, prev_roi, weight, 0)

        if intensity > 0.35:
            k = 3 if intensity < 0.7 else 5
            denoised = cv2.GaussianBlur(denoised, (k, k), 0)

        mask_f = (roi_mask.astype(np.float32) / 255.0 * intensity)[:, :, np.newaxis]
        frame[y1:y2, x1:x2, :3] = np.clip(
            roi.astype(np.float32) * (1.0 - mask_f)
            + denoised.astype(np.float32) * mask_f,
            0,
            255,
        ).astype(np.uint8)

        # Store the raw frame, never the already-denoised output, so motion
        # does not compound blur across frames.
        self._prev_frame = raw_bgr
        return frame

    def _apply_edge_darken(self, frame: np.ndarray,
                           width: int, height: int) -> np.ndarray:
        """Vignette effect — cached gradient, cv2 SIMD multiply."""
        intensity = self._edge_darken
        current_center = self._face_center or (width // 2, height // 2)
        center_shift = 0
        if self._vignette_center is not None:
            center_shift = max(
                abs(current_center[0] - self._vignette_center[0]),
                abs(current_center[1] - self._vignette_center[1]),
            )
        if (
            self._vignette_cache is None
            or self._vignette_size != (width, height)
            or center_shift > max(8, min(width, height) // 64)
        ):
            self._build_vignette_cache(width, height)

        if self._vignette_rgb_cache is None or self._vignette_rgb_intensity != intensity:
            vignette = (1.0 - intensity) + intensity * self._vignette_cache
            self._vignette_rgb_cache = cv2.merge([vignette, vignette, vignette])
            self._vignette_rgb_intensity = intensity

        frame[:, :, :3] = cv2.multiply(
            frame[:, :, :3], self._vignette_rgb_cache, scale=1.0, dtype=cv2.CV_8U
        )
        return frame

    def _apply_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Brightness, contrast, and warmth boost on face ROI.

        Uses direct BGR math instead of LAB conversion (saves ~12ms).
        """
        intensity = self._enhance
        mask = self._tone_mask if self._tone_mask is not None else self._face_mask
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
        self._tone_mask = None
        self._prev_frame = None
        self._vignette_cache = None
        self._vignette_rgb_cache = None

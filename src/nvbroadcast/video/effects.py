# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Video effects with multi-model segmentation backend.

Supported models:
- RVM (RobustVideoMatting): Person-only matting with recurrent temporal state
- BiRefNet-lite: General object segmentation (chairs, desks, people — anything)
- RMBG-2.0: Best quality general segmentation (non-commercial license)
"""

import os
# Force CUDA device ordering to match nvidia-smi (PCI bus ID order)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import ctypes
import threading
from pathlib import Path

import numpy as np
import cv2


def _preload_cuda_libs():
    """Pre-load pip-installed NVIDIA libs for ONNX Runtime CUDA."""
    try:
        import importlib.util
        for pkg in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cudnn",
                    "nvidia.curand", "nvidia.cufft", "nvidia.cusparse",
                    "nvidia.cusolver", "nvidia.nvjitlink"):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                lib_dir = Path(spec.submodule_search_locations[0]) / "lib"
                if lib_dir.is_dir():
                    for so in sorted(lib_dir.glob("*.so*")):
                        try:
                            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass
    except Exception:
        pass


_preload_cuda_libs()
import onnxruntime as ort

from nvbroadcast.core.constants import COMPUTE_GPU_INDEX

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"

# ─── Model Registry ─────────────────────────────────────────────────────────

MODELS = {
    "rvm": {
        "name": "RVM - Person Matting",
        "description": "Fast person-only matting with temporal consistency",
        "license": "GPL-3.0",
        "type": "recurrent",
        "skip_interval": 2,
    },
    "isnet": {
        "name": "IS-Net - General Objects",
        "description": "Segments foreground objects with high edge precision",
        "license": "Apache 2.0",
        "type": "single_frame",
        "model": "isnet-general-use.onnx",
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        "input_size": 1024,
        "mean": [0.5, 0.5, 0.5],
        "std": [1.0, 1.0, 1.0],
        "skip_interval": 2,  # Run every 2nd frame to maintain ~30fps display
    },
    "birefnet": {
        "name": "BiRefNet - Best Quality",
        "description": "Highest quality edges (requires 8GB+ free VRAM)",
        "license": "MIT",
        "type": "single_frame",
        "model": "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "input_size": 1024,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "skip_interval": 3,  # Heavy model, skip more frames
    },
}

QUALITY_PRESETS = {
    "performance": {
        "model": "rvm_mobilenetv3_fp32.onnx",
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        "downsample": 0.25,
        "label": "Performance (fastest, good edges)",
    },
    "balanced": {
        "model": "rvm_mobilenetv3_fp32.onnx",
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        "downsample": 0.5,
        "label": "Balanced (fast, better edges)",
    },
    "quality": {
        "model": "rvm_resnet50_fp32.onnx",
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx",
        "downsample": 0.375,
        "label": "Quality (detailed edges)",
    },
    "ultra": {
        "model": "rvm_resnet50_fp32.onnx",
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx",
        "downsample": 0.5,
        "label": "Ultra (best quality, sharpest edges)",
    },
}


# ─── Model Backends ──────────────────────────────────────────────────────────

def _create_session(model_path: str, gpu_index: int) -> ort.InferenceSession:
    """Create an ONNX Runtime session with CUDA fallback.

    CUDA_DEVICE_ORDER=PCI_BUS_ID ensures gpu_index matches nvidia-smi ordering.
    Uses arena_extend_strategy=1 (kSameAsRequested) to avoid pre-allocating
    large VRAM blocks — only allocates what the model actually needs.
    """
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': gpu_index,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB max for model
            'cudnn_conv_algo_search': 'HEURISTIC',  # Fast algo selection, no extra VRAM
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path), opts, providers=providers)


def _download_model(filename: str, url: str) -> Path:
    """Download a model file if not present."""
    model_path = _MODELS_DIR / filename
    if model_path.exists():
        return model_path
    import urllib.request
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[NV Broadcast] Downloading {filename}...")
    urllib.request.urlretrieve(url, str(model_path))
    print(f"[NV Broadcast] Downloaded {filename}")
    return model_path


def _get_device_name(session: ort.InferenceSession, gpu_index: int) -> str:
    """Get a human-readable device name from session."""
    active = session.get_providers()[0]
    if "CUDA" in active:
        try:
            from nvbroadcast.core.gpu import detect_gpus
            gpus = detect_gpus()
            name = gpus[gpu_index].name if gpu_index < len(gpus) else f"GPU {gpu_index}"
            return f"GPU ({name})"
        except Exception:
            return "GPU"
    return "CPU"


def _release_session(session):
    """Force-release an ONNX Runtime session and its GPU memory."""
    if session is None:
        return
    del session
    import gc
    gc.collect()


class _RVMBackend:
    """RobustVideoMatting — person-only with recurrent temporal states."""

    def __init__(self, gpu_index: int):
        self._gpu_index = gpu_index
        self.session = None
        self._r1 = self._r2 = self._r3 = self._r4 = None
        self._downsample_ratio = None

    def load(self, quality: str) -> str:
        preset = QUALITY_PRESETS[quality]
        model_path = _download_model(preset["model"], preset["url"])
        self.session = _create_session(model_path, self._gpu_index)
        self._downsample_ratio = np.array([preset["downsample"]], dtype=np.float32)
        self._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        device = _get_device_name(self.session, self._gpu_index)
        return f"RVM loaded on {device} | {preset['label']}"

    def infer(self, frame: np.ndarray, width: int, height: int) -> np.ndarray | None:
        # Fast normalize: BGRA→RGB + /255 + HWC→NCHW in minimal ops
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        src = rgb.astype(np.float32) * (1.0 / 255.0)
        src = src.transpose(2, 0, 1)[np.newaxis]  # HWC -> 1xCxHxW

        outputs = self.session.run(None, {
            'src': src,
            'r1i': self._r1, 'r2i': self._r2,
            'r3i': self._r3, 'r4i': self._r4,
            'downsample_ratio': self._downsample_ratio,
        })

        alpha = outputs[1][0, 0]
        if alpha.shape[0] != height or alpha.shape[1] != width:
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

        self._r1, self._r2 = outputs[2], outputs[3]
        self._r3, self._r4 = outputs[4], outputs[5]
        return alpha

    def reset_state(self):
        self._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)

    def cleanup(self):
        _release_session(self.session)
        self.session = None
        self._r1 = self._r2 = self._r3 = self._r4 = None


class _SingleFrameBackend:
    """Backend for single-frame models (BiRefNet, RMBG-2.0, IS-Net).

    These models have no recurrent state — each frame is independent.
    Temporal smoothing (EMA) is applied to reduce flicker.
    """

    def __init__(self, gpu_index: int, model_key: str):
        self._gpu_index = gpu_index
        self._model_key = model_key
        self._info = MODELS[model_key]
        self.session = None
        self._input_size = self._info["input_size"]
        self._mean = np.array(self._info["mean"], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(self._info["std"], dtype=np.float32).reshape(1, 1, 3)
        self._prev_alpha = None
        self._ema_weight = 0.15  # Temporal smoothing for flicker reduction

    def load(self, quality: str = "") -> str:
        model_path = _download_model(self._info["model"], self._info["url"])
        self.session = _create_session(model_path, self._gpu_index)
        device = _get_device_name(self.session, self._gpu_index)
        return f"{self._info['name']} loaded on {device}"

    def infer(self, frame: np.ndarray, width: int, height: int) -> np.ndarray | None:
        # Preprocess: resize to model input, normalize with model-specific mean/std
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, (self._input_size, self._input_size),
                             interpolation=cv2.INTER_LINEAR)

        # Normalize: (pixel / 255 - mean) / std
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - self._mean) / self._std
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = blob[np.newaxis, ...]  # Add batch dim: 1xCxHxW

        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})

        # Get alpha from output (sigmoid already applied in most models)
        raw = outputs[0]
        if raw.ndim == 4:
            alpha_small = raw[0, 0]  # 1x1xHxW -> HxW
        elif raw.ndim == 3:
            alpha_small = raw[0]  # 1xHxW -> HxW
        else:
            alpha_small = raw

        # Sigmoid if output is logits (values outside 0-1)
        if alpha_small.min() < -0.1 or alpha_small.max() > 1.1:
            alpha_small = 1.0 / (1.0 + np.exp(-alpha_small))

        alpha_small = np.clip(alpha_small, 0, 1).astype(np.float32)

        # Resize back to frame size
        alpha = cv2.resize(alpha_small, (width, height), interpolation=cv2.INTER_LINEAR)

        # Temporal smoothing (EMA) to reduce flicker
        if self._prev_alpha is not None:
            alpha = self._ema_weight * self._prev_alpha + (1.0 - self._ema_weight) * alpha
        self._prev_alpha = alpha

        return alpha

    def reset_state(self):
        self._prev_alpha = None

    def cleanup(self):
        _release_session(self.session)
        self.session = None
        self._prev_alpha = None


# ─── Main VideoEffects Class ─────────────────────────────────────────────────

class VideoEffects:
    def __init__(self, gpu_index: int = COMPUTE_GPU_INDEX, edge_config=None):
        self._gpu_index = gpu_index
        self._initialized = False
        self._lock = threading.Lock()
        self._quality = "quality"
        self._model_type = "rvm"
        self._backend = None
        self._edge_config = edge_config

        # Effect state
        self._bg_removal_enabled = False
        self._bg_mode = "blur"
        self._bg_image = None
        self._bg_image_path = ""
        self._blur_strength = 21
        self._intensity = 0.7
        self._frame_size = None
        self._resized_bg = None
        self._green_bg = None
        self._frame_counter = 0
        self._cached_alpha = None
        self._skip_interval = 1
        self._cached_blur_bg = None  # Cache blurred background between frames

        # Alpha refinement
        self._apply_edge_config(edge_config)

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def enabled(self) -> bool:
        return self._bg_removal_enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._bg_removal_enabled = value
        if value and not self._initialized:
            self.initialize()

    @property
    def mode(self) -> str:
        return self._bg_mode

    @mode.setter
    def mode(self, value: str):
        if value in ("blur", "replace", "remove"):
            self._bg_mode = value

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def quality(self) -> str:
        return self._quality

    @quality.setter
    def quality(self, value: str):
        if self._model_type != "rvm":
            return  # Quality presets only apply to RVM
        if value not in QUALITY_PRESETS or value == self._quality:
            return
        old = self._quality
        self._quality = value
        if self._initialized:
            old_model = QUALITY_PRESETS[old]["model"]
            new_model = QUALITY_PRESETS[value]["model"]
            if old_model != new_model:
                # Different model file — full reload
                self._cleanup_backend()
                self.initialize()
            else:
                # Same model, different downsample — just update ratio
                with self._lock:
                    if isinstance(self._backend, _RVMBackend):
                        self._backend._downsample_ratio = np.array(
                            [QUALITY_PRESETS[value]["downsample"]], dtype=np.float32
                        )

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, min(1.0, value))
        k = int(5 + value * 94)
        self._blur_strength = k if k % 2 == 1 else k + 1

    def set_model(self, model_type: str):
        """Switch segmentation model."""
        if model_type not in MODELS or model_type == self._model_type:
            return
        self._model_type = model_type
        # Apply per-model frame skip interval
        model_info = MODELS[model_type]
        self._skip_interval = model_info.get("skip_interval", 1)
        if self._initialized:
            self._cleanup_backend()
            self._cached_alpha = None
            self.initialize()

    def set_background_image(self, image_path: str) -> bool:
        if not image_path or not os.path.exists(image_path):
            self._bg_image = None
            self._bg_image_path = ""
            return False
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        self._bg_image = img
        self._bg_image_path = image_path
        self._frame_size = None
        return True

    def initialize(self) -> bool:
        """Initialize the active model backend."""
        if self._initialized:
            return True

        try:
            if self._model_type == "rvm":
                backend = _RVMBackend(self._gpu_index)
                msg = backend.load(self._quality)
            else:
                backend = _SingleFrameBackend(self._gpu_index, self._model_type)
                msg = backend.load()

            with self._lock:
                self._backend = backend
                self._initialized = True
                self._cached_alpha = None

            print(f"[NV Broadcast] {msg}")
            return True

        except Exception as e:
            print(f"[NV Broadcast] Failed to initialize model: {e}")
            return False

    def process_frame(self, frame_data: bytes, width: int, height: int) -> bytes:
        if not self._bg_removal_enabled or not self._initialized:
            return frame_data

        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
        if not frame.flags.writeable:
            frame = frame.copy()

        self._frame_counter += 1

        if self._skip_interval <= 1 or self._frame_counter % self._skip_interval == 0 or self._cached_alpha is None:
            alpha = self._run_inference(frame, width, height)
            if alpha is not None:
                self._cached_alpha = alpha
            elif self._cached_alpha is None:
                return frame_data
        alpha = self._cached_alpha

        if self._bg_mode == "blur":
            result = self._apply_blur(frame, alpha)
        elif self._bg_mode == "remove":
            result = self._apply_green_screen(frame, alpha, width, height)
        else:
            result = self._apply_replace(frame, alpha, width, height)

        return result.tobytes()

    def _run_inference(self, frame: np.ndarray, width: int, height: int) -> np.ndarray | None:
        """Run the active backend's inference and refine the alpha."""
        with self._lock:
            backend = self._backend
            if backend is None:
                return None

        try:
            alpha = backend.infer(frame, width, height)
            if alpha is not None:
                alpha = self._refine_alpha(alpha)
            return alpha
        except Exception as e:
            print(f"[NV Broadcast] Inference error: {e}")
            return None

    # ─── Alpha Refinement ────────────────────────────────────────────────

    def _apply_edge_config(self, edge_config=None):
        if edge_config:
            self._dilate_size = edge_config.dilate_size
            self._blur_size = edge_config.blur_size
            self._sigmoid_strength = edge_config.sigmoid_strength
            self._sigmoid_midpoint = edge_config.sigmoid_midpoint
        else:
            self._dilate_size = 5
            self._blur_size = 9
            self._sigmoid_strength = 12.0
            self._sigmoid_midpoint = 0.5
        ds = self._dilate_size if self._dilate_size % 2 == 1 else self._dilate_size + 1
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ds, ds))
        bs = self._blur_size if self._blur_size % 2 == 1 else self._blur_size + 1
        self._blur_ksize = (bs, bs)

    def update_edge_params(self, dilate_size=None, blur_size=None,
                           sigmoid_strength=None, sigmoid_midpoint=None):
        if dilate_size is not None:
            self._dilate_size = int(dilate_size)
            ds = self._dilate_size if self._dilate_size % 2 == 1 else self._dilate_size + 1
            self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ds, ds))
        if blur_size is not None:
            self._blur_size = int(blur_size)
            bs = self._blur_size if self._blur_size % 2 == 1 else self._blur_size + 1
            self._blur_ksize = (bs, bs)
        if sigmoid_strength is not None:
            self._sigmoid_strength = float(sigmoid_strength)
        if sigmoid_midpoint is not None:
            self._sigmoid_midpoint = float(sigmoid_midpoint)

    def _refine_alpha(self, alpha: np.ndarray) -> np.ndarray:
        # Stay in uint8 for morphological ops (skip float conversion)
        a8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        if self._dilate_size > 0:
            a8 = cv2.dilate(a8, self._dilate_kernel, iterations=1)
        if self._blur_size > 1:
            a8 = cv2.GaussianBlur(a8, self._blur_ksize, 0)
        if self._sigmoid_strength > 0:
            # Sigmoid in float32
            t = a8.astype(np.float32) * (1.0 / 255.0)
            np.subtract(t, self._sigmoid_midpoint, out=t)
            np.multiply(t, -self._sigmoid_strength, out=t)
            np.exp(t, out=t)
            np.add(t, 1.0, out=t)
            np.reciprocal(t, out=t)
            return t
        return a8.astype(np.float32) * (1.0 / 255.0)

    # ─── Compositing ─────────────────────────────────────────────────────

    @staticmethod
    def _blend(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Alpha blend using cv2 for SIMD-optimized per-pixel operations."""
        # Build 4-channel alpha mask (cv2 needs matching channels)
        a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        a4 = cv2.merge([a8, a8, a8, a8])
        ia4 = cv2.bitwise_not(a4)
        # fg * alpha + bg * (1-alpha) using cv2 multiply (SIMD optimized)
        fg_part = cv2.multiply(fg, a4, scale=1.0 / 255.0, dtype=cv2.CV_8U)
        bg_part = cv2.multiply(bg, ia4, scale=1.0 / 255.0, dtype=cv2.CV_8U)
        return cv2.add(fg_part, bg_part)

    def _apply_blur(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        # Only recompute blur when alpha updated (skip frames reuse cached blur)
        if self._frame_counter % max(self._skip_interval, 2) == 0 or self._cached_blur_bg is None:
            self._cached_blur_bg = cv2.GaussianBlur(
                frame, (self._blur_strength, self._blur_strength), 0
            )
        return self._blend(frame, self._cached_blur_bg, alpha)

    def _apply_green_screen(self, frame: np.ndarray, alpha: np.ndarray,
                            width: int, height: int) -> np.ndarray:
        if self._green_bg is None or self._green_bg.shape[:2] != (height, width):
            self._green_bg = np.zeros((height, width, 4), dtype=np.uint8)
            self._green_bg[:, :, 1] = 255
            self._green_bg[:, :, 3] = 255
        return self._blend(frame, self._green_bg, alpha)

    def _apply_replace(self, frame: np.ndarray, alpha: np.ndarray,
                       width: int, height: int) -> np.ndarray:
        if self._bg_image is None:
            return self._apply_blur(frame, alpha)
        if self._frame_size != (width, height):
            self._resized_bg = self._resize_bg(self._bg_image, width, height)
            self._frame_size = (width, height)
        return self._blend(frame, self._resized_bg, alpha)

    def _resize_bg(self, bg: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        bg_h, bg_w = bg.shape[:2]
        scale = max(target_w / bg_w, target_h / bg_h)
        new_w, new_h = int(bg_w * scale), int(bg_h * scale)
        resized = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        x = (new_w - target_w) // 2
        y = (new_h - target_h) // 2
        cropped = resized[y:y + target_h, x:x + target_w]
        if cropped.shape[2] == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        return cropped

    # ─── Lifecycle ───────────────────────────────────────────────────────

    def _cleanup_backend(self):
        with self._lock:
            if self._backend:
                self._backend.cleanup()
            self._backend = None
            self._initialized = False

    def cleanup(self):
        self._cleanup_backend()

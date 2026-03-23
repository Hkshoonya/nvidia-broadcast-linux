# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
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
    """Pre-load pip-installed NVIDIA libs for ONNX Runtime CUDA + TensorRT."""
    try:
        import importlib.util
        # CUDA runtime libs
        for pkg in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cudnn",
                    "nvidia.curand", "nvidia.cufft", "nvidia.cusparse",
                    "nvidia.cusolver", "nvidia.nvjitlink", "nvidia.cuda_nvrtc"):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                lib_dir = Path(spec.submodule_search_locations[0]) / "lib"
                if lib_dir.is_dir():
                    for so in sorted(lib_dir.glob("*.so*")):
                        try:
                            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass
        # TensorRT libs (Zeus/Killer modes)
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            lib_dir = Path(spec.submodule_search_locations[0])
            # Load main libs first, then builders
            for pattern in ("libnvinfer.so*", "libnvinfer_plugin.so*",
                            "libnvonnxparser.so*", "libnvinfer_builder*.so*"):
                for so in sorted(lib_dir.glob(pattern)):
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
        "skip_interval": 1,  # Every frame — RVM is fast and has temporal state
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

def _create_session(model_path: str, gpu_index: int,
                    use_tensorrt: bool = False) -> ort.InferenceSession:
    """Create an ONNX Runtime session.

    use_tensorrt=True enables TensorRT EP (Zeus/Killer modes) for 3-5x faster inference.
    First run builds the TRT engine (~30s), cached for instant subsequent loads.
    """
    providers = []
    if use_tensorrt and 'TensorrtExecutionProvider' in ort.get_available_providers():
        cache_dir = str(_MODELS_DIR / "trt_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        providers.append(('TensorrtExecutionProvider', {
            'device_id': gpu_index,
            'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
            'trt_builder_optimization_level': 3,
        }))
    providers.append(('CUDAExecutionProvider', {
        'device_id': gpu_index,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': True,
    }))
    providers.append('CPUExecutionProvider')
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path), opts, providers=providers)


# ─── Fused CUDA Kernel (DocZeus/Killer modes) ──────────────────────────────────

_FUSED_COMPOSITE_KERNEL = r'''
extern "C" __global__ void fused_composite(
    const unsigned char* fg, const unsigned char* bg,
    const float* alpha, const unsigned char* face_mask,
    const float* vignette, unsigned char* output,
    int total_pixels,
    float enhance_i, float vignette_i, float brightness,
    float contrast, float warmth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    int px = idx * 4;
    float a = alpha[idx];
    float ia = 1.0f - a;

    // Alpha blend
    float b = (float)fg[px]   * a + (float)bg[px]   * ia;
    float g = (float)fg[px+1] * a + (float)bg[px+1] * ia;
    float r = (float)fg[px+2] * a + (float)bg[px+2] * ia;

    // Enhance on face region (brightness + contrast + warmth)
    if (enhance_i > 0.0f && face_mask != NULL) {
        float fm = (float)face_mask[idx] / 255.0f * enhance_i;
        if (fm > 0.01f) {
            float er = (r - 128.0f) * (1.0f + fm * contrast) + 128.0f + fm * brightness;
            float eg = (g - 128.0f) * (1.0f + fm * contrast) + 128.0f + fm * brightness;
            float eb = (b - 128.0f) * (1.0f + fm * contrast) + 128.0f + fm * brightness;
            er += fm * warmth;
            eg += fm * warmth * 0.3f;
            r = r * (1.0f - fm) + er * fm;
            g = g * (1.0f - fm) + eg * fm;
            b = b * (1.0f - fm) + eb * fm;
        }
    }

    // Vignette
    if (vignette != NULL && vignette_i > 0.0f) {
        float v = (1.0f - vignette_i) + vignette_i * vignette[idx];
        r *= v; g *= v; b *= v;
    }

    output[px]   = (unsigned char)fminf(fmaxf(b, 0.0f), 255.0f);
    output[px+1] = (unsigned char)fminf(fmaxf(g, 0.0f), 255.0f);
    output[px+2] = (unsigned char)fminf(fmaxf(r, 0.0f), 255.0f);
    output[px+3] = fg[px+3];
}
'''

_fused_kernel = None

def _get_fused_kernel():
    """Lazy-load the fused CUDA kernel."""
    global _fused_kernel
    if _fused_kernel is None:
        try:
            import cupy as cp
            _fused_kernel = cp.RawKernel(_FUSED_COMPOSITE_KERNEL, 'fused_composite')
        except Exception as e:
            print(f"[NV Broadcast] Fused CUDA kernel failed: {e}")
    return _fused_kernel


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

    def load(self, quality: str, use_tensorrt: bool = False) -> str:
        preset = QUALITY_PRESETS[quality]
        model_path = _download_model(preset["model"], preset["url"])
        # Use shape-inferred model for TensorRT
        if use_tensorrt:
            trt_path = model_path.with_name(
                model_path.stem + "_trt" + model_path.suffix
            )
            if trt_path.exists():
                model_path = trt_path
        self.session = _create_session(model_path, self._gpu_index,
                                       use_tensorrt=use_tensorrt)
        self._downsample_ratio = np.array([preset["downsample"]], dtype=np.float32)
        self._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        device = _get_device_name(self.session, self._gpu_index)
        return f"RVM loaded on {device} | {preset['label']}"

    # Max input resolution for preprocessing — above this, downsample first.
    # 720p is the sweet spot: fast preprocessing, model quality maintained.
    _MAX_INFER_HEIGHT = 720

    def infer(self, frame: np.ndarray, width: int, height: int) -> np.ndarray | None:
        # Pre-downsample large frames to reduce preprocessing + inference cost.
        # E.g. 1080p → 720p input saves ~50% time; alpha is upscaled back.
        if height > self._MAX_INFER_HEIGHT:
            scale = self._MAX_INFER_HEIGHT / height
            infer_w = int(width * scale) & ~1  # Even dimensions
            infer_h = self._MAX_INFER_HEIGHT & ~1
            small = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame
            infer_w, infer_h = width, height

        # Fast normalize: BGRA→RGB + /255 + HWC→NCHW
        rgb = cv2.cvtColor(small, cv2.COLOR_BGRA2RGB)
        src = rgb.astype(np.float32) * (1.0 / 255.0)
        src = src.transpose(2, 0, 1)[np.newaxis]  # HWC -> 1xCxHxW

        outputs = self.session.run(None, {
            'src': src,
            'r1i': self._r1, 'r2i': self._r2,
            'r3i': self._r3, 'r4i': self._r4,
            'downsample_ratio': self._downsample_ratio,
        })

        alpha = outputs[1][0, 0]
        self._r1, self._r2 = outputs[2], outputs[3]
        self._r3, self._r4 = outputs[4], outputs[5]

        # For aggressive downsample modes (Zeus/Killer): refine at low-res
        # then upscale with cubic. Stronger sigmoid compensates for low-res edges.
        if self._MAX_INFER_HEIGHT < 720 and (alpha.shape[0] != height or alpha.shape[1] != width):
            alpha = np.clip(alpha, 0, 1)
            t = (alpha - 0.45) * -16.0  # Stronger sigmoid for crisp edges
            np.exp(t, out=t)
            np.add(t, 1.0, out=t)
            np.reciprocal(t, out=t)
            # Dilate slightly to avoid cutting into person silhouette
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            t_u8 = (t * 255).astype(np.uint8)
            t_u8 = cv2.dilate(t_u8, kernel, iterations=1)
            t = t_u8.astype(np.float32) / 255.0
            # Cubic upscale preserves edge sharpness better than linear
            alpha = cv2.resize(t, (width, height), interpolation=cv2.INTER_CUBIC)
            alpha = np.clip(alpha, 0, 1)
            self._lowres_refined = True
        else:
            if alpha.shape[0] != height or alpha.shape[1] != width:
                alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
            self._lowres_refined = False

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


# ─── Edge Refinement (Zeus/Killer quality boost) ────────────────────────────

class EdgeRefiner:
    """Neural edge refinement using a second RVM pass at full 720p.

    Runs every Nth frame (configurable) to refine the coarse alpha from
    Zeus (480p) or Killer (360p) modes. Blends refined edges into the
    coarse alpha, keeping sharp interior/exterior from the fast pass.

    Cost: ~30ms per refinement frame. At skip=3 (default), adds ~10ms avg.
    """

    def __init__(self, gpu_index: int):
        self._gpu_index = gpu_index
        self._backend = None
        self._initialized = False
        self._skip = 2  # Refine every 2nd frame
        self._counter = 0
        self._cached_refined = None
        self._reset_interval = 30  # Reset recurrent state every N refines (~2s at skip=2, 30fps)

    def initialize(self, quality: str = "quality"):
        """Load same-quality RVM at full 720p for edge refinement.

        Uses the SAME model class as main inference but at full resolution.
        This ensures the refiner is at least as good as the main pass.
        """
        if self._initialized:
            return
        self._backend = _RVMBackend(self._gpu_index)
        self._backend.load(quality)  # Same model as main (resnet50 for quality)
        self._backend._MAX_INFER_HEIGHT = 720  # Always full resolution
        self._initialized = True
        print(f"[NV Broadcast] Edge refiner initialized (720p {quality})")

    def refine(self, frame: np.ndarray, coarse_alpha: np.ndarray,
               width: int, height: int) -> np.ndarray:
        """Refine alpha using full-quality 720p inference.

        On refine frames: run 720p inference, use that alpha entirely.
        On skip frames: blend 80% cached refined + 20% coarse for tracking.
        This gives quality edges from the refiner + position updates from fast pass.
        """
        if not self._initialized or self._backend is None:
            return coarse_alpha

        self._counter += 1
        # Reset recurrent state periodically to prevent temporal divergence
        if self._counter % (self._skip * self._reset_interval) == 0:
            self._backend.reset_state()
        if self._counter % self._skip == 0 or self._cached_refined is None:
            try:
                fine_alpha = self._backend.infer(frame, width, height)
                if fine_alpha is not None:
                    self._cached_refined = fine_alpha
                    return fine_alpha
            except Exception:
                pass

        if self._cached_refined is None:
            return coarse_alpha

        # Between refine frames: mostly use cached quality alpha,
        # but blend in coarse for position tracking on movement
        return (0.8 * self._cached_refined + 0.2 * coarse_alpha).astype(np.float32)

    def reset(self):
        if self._backend:
            self._backend.reset_state()
        self._cached_refined = None

    def cleanup(self):
        if self._backend:
            self._backend.cleanup()
            self._backend = None
        self._initialized = False
        self._cached_refined = None


# ─── Main VideoEffects Class ─────────────────────────────────────────────────

class VideoEffects:
    def __init__(self, gpu_index: int = COMPUTE_GPU_INDEX, edge_config=None,
                 compositing: str = "cpu"):
        self._gpu_index = gpu_index
        self._initialized = False
        self._lock = threading.Lock()
        self._quality = "quality"
        self._model_type = "rvm"
        self._backend = None
        self._edge_config = edge_config
        self._use_tensorrt = False   # Zeus/Killer mode
        self._use_fused_kernel = False  # DocZeus/Killer mode
        self._edge_refine_enabled = False  # Edge refinement toggle
        self._edge_refiner = EdgeRefiner(gpu_index)
        self._compositing = "cpu"
        self._cupy = None  # Lazy-loaded cupy module
        if compositing != "cpu":
            self.set_compositing(compositing)

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

        # SVG support via librsvg (renders to raster at high quality)
        if image_path.lower().endswith('.svg') or image_path.lower().endswith('.svgz'):
            img = self._load_svg(image_path)
        else:
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

    @staticmethod
    def _load_svg(path: str) -> np.ndarray | None:
        """Render SVG to BGRA numpy array using GdkPixbuf."""
        try:
            import gi
            gi.require_version('GdkPixbuf', '2.0')
            from gi.repository import GdkPixbuf

            # Load SVG at high resolution
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, 1920, -1, True)
            if pixbuf is None:
                return None

            w = pixbuf.get_width()
            h = pixbuf.get_height()
            channels = pixbuf.get_n_channels()
            rowstride = pixbuf.get_rowstride()
            data = pixbuf.get_pixels()

            # GdkPixbuf gives RGBA, convert to BGRA for OpenCV
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, rowstride // channels, channels)
            img = img[:h, :w, :].copy()
            if channels >= 3:
                img[:, :, [0, 2]] = img[:, :, [2, 0]]  # RGB → BGR
            if channels == 3:
                # Add alpha channel
                alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha], axis=2)
            return img
        except Exception as e:
            print(f"[NV Broadcast] SVG load failed: {e}")
            return None

    def initialize(self) -> bool:
        """Initialize the active model backend."""
        if self._initialized:
            return True

        try:
            if self._model_type == "rvm":
                backend = _RVMBackend(self._gpu_index)
                msg = backend.load(self._quality, use_tensorrt=self._use_tensorrt)
                if self._use_tensorrt:
                    active = backend.session.get_providers()[0]
                    if "Tensorrt" in active:
                        msg += " [TensorRT]"
                    else:
                        msg += " [TRT fallback to CUDA]"
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

        # Fused CUDA kernel path (DocZeus/Killer) — single GPU pass for composite
        if self._use_fused_kernel and self._cupy is not None:
            result = self._composite_fused(frame, alpha, width, height)
            if result is not None:
                return result.tobytes()

        # Standard compositing path
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
                # Skip full-res refinement if already done at low-res (Zeus/Killer)
                if not getattr(backend, '_lowres_refined', False):
                    alpha = self._refine_alpha(alpha)
                # Edge refine: second pass at 720p for Zeus/Killer modes
                if self._edge_refine_enabled:
                    if not self._edge_refiner._initialized:
                        self._edge_refiner.initialize(self._quality)
                    raw_refined = self._edge_refiner.refine(frame, alpha, width, height)
                    # Apply same refinement to the refiner output for consistency
                    alpha = self._refine_alpha(raw_refined)
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

    # ─── Fused CUDA Kernel (DocZeus/Killer) ─────────────────────────────

    def _composite_fused(self, frame: np.ndarray, alpha: np.ndarray,
                         width: int, height: int) -> np.ndarray | None:
        """Single-pass GPU composite: blend + enhance + vignette in one kernel."""
        kernel = _get_fused_kernel()
        if kernel is None:
            return None
        try:
            cp = self._cupy
            total = width * height

            # Build background
            if self._bg_mode == "blur":
                bg = cv2.GaussianBlur(frame, (self._blur_strength, self._blur_strength), 0)
            elif self._bg_mode == "remove":
                bg = np.zeros_like(frame)
                bg[:, :, 1] = 255
                bg[:, :, 3] = 255
            else:
                bg = self._get_bg_image(frame, width, height)

            # Upload all to GPU in one batch
            fg_gpu = cp.asarray(frame)
            bg_gpu = cp.asarray(bg)
            alpha_gpu = cp.asarray(alpha, dtype=cp.float32)
            output_gpu = cp.empty_like(fg_gpu)

            # Zero-filled arrays for unused features (no NULL pointers in CuPy)
            face_mask_gpu = cp.zeros((height, width), dtype=cp.uint8)
            vignette_gpu = cp.ones((height, width), dtype=cp.float32)

            threads = 256
            blocks = (total + threads - 1) // threads
            kernel((blocks,), (threads,), (
                fg_gpu, bg_gpu, alpha_gpu,
                face_mask_gpu, vignette_gpu, output_gpu,
                cp.int32(total),
                cp.float32(0.0),   # enhance (handled by beautifier)
                cp.float32(0.0),   # vignette (0 = no darkening)
                cp.float32(0.0),   # brightness
                cp.float32(0.0),   # contrast
                cp.float32(0.0),   # warmth
            ))
            cp.cuda.Stream.null.synchronize()

            return cp.asnumpy(output_gpu)
        except Exception as e:
            print(f"[NV Broadcast] Fused kernel error: {e}")
            return None

    def _get_bg_image(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Get background image resized to match frame."""
        if self._bg_image is None:
            bg = np.full_like(frame, 128)
            bg[:, :, 3] = 255
            return bg
        if self._frame_size != (width, height):
            self._bg_resized = cv2.resize(self._bg_image, (width, height))
            self._frame_size = (width, height)
        return self._bg_resized

    # ─── Compositing ─────────────────────────────────────────────────────

    def set_engine_mode(self, use_tensorrt: bool, use_fused_kernel: bool):
        """Set inference/compositing engine.

        Zeus/Killer modes use aggressive 480p/360p pre-downsampling for faster inference.
        DocZeus/Killer modes use a fused CUDA kernel for single-pass compositing.
        """
        self._use_tensorrt = use_tensorrt
        self._use_fused_kernel = use_fused_kernel
        if self._backend and hasattr(self._backend, '_MAX_INFER_HEIGHT'):
            if use_tensorrt and use_fused_kernel:
                new_h = 360   # Killer
            elif use_tensorrt:
                new_h = 480   # Zeus
            else:
                new_h = 720   # Standard/DocZeus
            # ALWAYS reset state on mode change to prevent shape mismatch
            with self._lock:
                self._backend._MAX_INFER_HEIGHT = new_h
                self._backend.reset_state()
                self._cached_alpha = None
            self._edge_refiner.reset()

    def set_compositing(self, backend: str):
        """Switch compositing backend (cpu, gstreamer_gl, cupy)."""
        self._compositing = backend
        if backend in ("cupy", "gstreamer_gl") and self._cupy is None:
            try:
                import cupy
                self._cupy = cupy
                print("[NV Broadcast] CuPy GPU compositing enabled")
            except ImportError:
                if backend == "cupy":
                    print("[NV Broadcast] CuPy not installed, falling back to CPU")
                    self._compositing = "cpu"

    def _blend(self, fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Alpha blend — uses CuPy GPU when available, regardless of mode."""
        if self._cupy is not None:
            return self._blend_cupy(fg, bg, alpha)
        return self._blend_cpu(fg, bg, alpha)

    @staticmethod
    def _blend_cpu(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """CPU blend using cv2 SIMD-optimized operations."""
        a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        a4 = cv2.merge([a8, a8, a8, a8])
        ia4 = cv2.bitwise_not(a4)
        fg_part = cv2.multiply(fg, a4, scale=1.0 / 255.0, dtype=cv2.CV_8U)
        bg_part = cv2.multiply(bg, ia4, scale=1.0 / 255.0, dtype=cv2.CV_8U)
        return cv2.add(fg_part, bg_part)

    def _blend_cupy(self, fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """GPU blend using CuPy CUDA arrays — near-zero CPU usage."""
        try:
            cp = self._cupy
            fg_gpu = cp.asarray(fg)
            bg_gpu = cp.asarray(bg)
            a_gpu = cp.asarray(alpha, dtype=cp.float32)[:, :, cp.newaxis]
            result = (fg_gpu.astype(cp.float32) * a_gpu +
                      bg_gpu.astype(cp.float32) * (1.0 - a_gpu))
            return cp.asnumpy(result.astype(cp.uint8))
        except Exception as e:
            # Fallback to CPU if CuPy fails (missing nvrtc, OOM, etc.)
            if self._frame_counter <= 2:
                print(f"[NV Broadcast] CuPy blend failed, falling back to CPU: {e}")
            self._compositing = "cpu"
            return self._blend_cpu(fg, bg, alpha)

    def _apply_blur(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (self._blur_strength, self._blur_strength), 0)
        return self._blend(frame, blurred, alpha)

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
        self._edge_refiner.cleanup()

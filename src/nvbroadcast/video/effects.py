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
import platform as _platform
import json

# Force CUDA device ordering to match nvidia-smi (PCI bus ID order) — Linux only
if _platform.system() != "Darwin":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import ctypes
import threading
from pathlib import Path

import numpy as np
import cv2


def _preload_cuda_libs():
    """Pre-load pip-installed NVIDIA libs for ONNX Runtime CUDA + TensorRT.

    Skipped on macOS where CUDA is not available.
    """
    if _platform.system() == "Darwin":
        return
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
_LEARNED_REFINER_MODELS = {
    "replace": _MODELS_DIR / "edge_refiner_replace.onnx",
    "remove": _MODELS_DIR / "edge_refiner_remove.onnx",
}

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
                    use_tensorrt: bool = False,
                    cpu_only: bool = False) -> ort.InferenceSession:
    """Create an ONNX Runtime session.

    use_tensorrt=True enables TensorRT EP (Zeus/Killer modes) for 3-5x faster inference.
    First run builds the TRT engine (~30s), cached for instant subsequent loads.

    On macOS: uses CoreML EP (Apple Silicon) or CPU EP as fallback.
    """
    if cpu_only:
        providers = ["CPUExecutionProvider"]
    else:
        from nvbroadcast.core.platform import get_onnx_providers
        providers = get_onnx_providers(gpu_index, use_tensorrt)
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
    if "CUDA" in active or "Tensorrt" in active:
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
        self._active_trt = False
        self._base_model_path = None
        self._trt_fallback_logged = False

    def load(self, quality: str, use_tensorrt: bool = False) -> str:
        preset = QUALITY_PRESETS[quality]
        base_model_path = _download_model(preset["model"], preset["url"])
        self._base_model_path = base_model_path
        model_path = base_model_path
        # Use shape-inferred model for TensorRT
        if use_tensorrt:
            trt_path = model_path.with_name(
                model_path.stem + "_trt" + model_path.suffix
            )
            if trt_path.exists():
                model_path = trt_path

        self.session = _create_session(
            model_path, self._gpu_index, use_tensorrt=use_tensorrt
        )
        active = self.session.get_providers()[0]
        self._active_trt = "TensorrtExecutionProvider" in active

        # TensorRT parser/build can fail internally and leave us on CPU. In that
        # case, explicitly reload the session on CUDA so premium modes remain
        # functional instead of silently becoming a broken CPU path.
        if use_tensorrt and "CPUExecutionProvider" in active:
            _release_session(self.session)
            self.session = _create_session(base_model_path, self._gpu_index, use_tensorrt=False)
            active = self.session.get_providers()[0]
            self._active_trt = False

        self._downsample_ratio = np.array([preset["downsample"]], dtype=np.float32)
        self._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        device = _get_device_name(self.session, self._gpu_index)
        msg = f"RVM loaded on {device} | {preset['label']}"
        if use_tensorrt:
            if self._active_trt:
                msg += " [TensorRT]"
            else:
                msg += " [TRT fallback to CUDA]"
        return msg

    def _fallback_to_cuda(self):
        """Recreate the session on CUDA after TRT runtime/build failure."""
        if not self._active_trt or self._base_model_path is None:
            return
        _release_session(self.session)
        self.session = _create_session(self._base_model_path, self._gpu_index, use_tensorrt=False)
        self._active_trt = False
        self.reset_state()
        if not self._trt_fallback_logged:
            print("[NV Broadcast] TensorRT runtime failed during inference - falling back to CUDA", flush=True)
            self._trt_fallback_logged = True

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

        inputs = {
            'src': src,
            'r1i': self._r1, 'r2i': self._r2,
            'r3i': self._r3, 'r4i': self._r4,
            'downsample_ratio': self._downsample_ratio,
        }
        try:
            outputs = self.session.run(None, inputs)
        except Exception:
            if self._active_trt:
                self._fallback_to_cuda()
                outputs = self.session.run(None, inputs)
            else:
                # Shape mismatch from resolution change — reset and retry
                self.reset_state()
                inputs['r1i'] = self._r1
                inputs['r2i'] = self._r2
                inputs['r3i'] = self._r3
                inputs['r4i'] = self._r4
                outputs = self.session.run(None, inputs)

        alpha = outputs[1][0, 0]
        self._r1, self._r2 = outputs[2], outputs[3]
        self._r3, self._r4 = outputs[4], outputs[5]

        # Upscale to frame resolution if needed (low-res inference modes)
        if alpha.shape[0] != height or alpha.shape[1] != width:
            alpha = np.clip(alpha, 0, 1)
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_CUBIC)
            alpha = np.clip(alpha, 0, 1)
        # Always let _refine_alpha handle edge processing — it has the
        # improved pipeline (morph close, bilateral, matte hardening).
        self._lowres_refined = False

        return alpha

    def reset_state(self):
        """Reset recurrent states for resolution change.
        Zero (1,1,1,1) tensors trigger RVM's built-in shape auto-detection.
        """
        self._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        print("[NV Broadcast] RVM recurrent states reset", flush=True)

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
        self._model_path = None
        self._input_size = self._info["input_size"]
        self._input_size_current = self._input_size
        self._fixed_input_size = self._input_size
        self._mean = np.array(self._info["mean"], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(self._info["std"], dtype=np.float32).reshape(1, 1, 3)
        self._prev_alpha = None
        self._ema_weight = 0.15  # Temporal smoothing for flicker reduction
        self._fallback_logged = False
        self._cpu_fallback_active = False

    def load(self, quality: str = "") -> str:
        self._model_path = _download_model(self._info["model"], self._info["url"])
        try:
            self.session = _create_session(self._model_path, self._gpu_index)
            self._cpu_fallback_active = False
        except Exception as exc:
            self.session = self._load_cpu_fallback(exc)
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self._fixed_input_size = int(input_shape[2])
            self._input_size_current = self._fixed_input_size
        else:
            self._fixed_input_size = None
        device = _get_device_name(self.session, self._gpu_index)
        suffix = " [CPU fallback]" if self._cpu_fallback_active else ""
        return f"{self._info['name']} loaded on {device}{suffix}"

    def _load_cpu_fallback(self, exc: Exception) -> ort.InferenceSession:
        if self._model_path is None:
            raise exc
        session = _create_session(self._model_path, self._gpu_index, cpu_only=True)
        self._cpu_fallback_active = True
        if not self._fallback_logged:
            print(
                f"[NV Broadcast] {self._info['name']} GPU path unavailable, using CPU fallback: {exc}",
                flush=True,
            )
            self._fallback_logged = True
        return session

    def _fallback_to_cpu(self, exc: Exception) -> None:
        if self._cpu_fallback_active:
            raise exc
        _release_session(self.session)
        self.session = self._load_cpu_fallback(exc)
        self.reset_state()

    def _select_input_size(self, width: int, height: int) -> int:
        """Pick a square input size without upscaling far beyond the source frame."""
        if self._fixed_input_size is not None:
            return self._fixed_input_size
        longest = max(width, height)
        target = min(self._input_size_current, longest)
        target = max(512, target)
        target = int(np.ceil(target / 32.0) * 32)
        return min(target, self._input_size_current)

    @staticmethod
    def _candidate_input_sizes(start: int) -> list[int]:
        sizes = [start]
        for candidate in (896, 768, 640, 512):
            if candidate < start:
                sizes.append(candidate)
        deduped: list[int] = []
        for size in sizes:
            if size not in deduped:
                deduped.append(size)
        return deduped

    def infer(self, frame: np.ndarray, width: int, height: int) -> np.ndarray | None:
        # Preprocess: resize to model input, normalize with model-specific mean/std
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        input_name = self.session.get_inputs()[0].name
        outputs = None
        last_error = None
        requested_size = self._select_input_size(width, height)

        for infer_size in self._candidate_input_sizes(requested_size):
            resized = cv2.resize(rgb, (infer_size, infer_size),
                                 interpolation=cv2.INTER_LINEAR)

            # Normalize: (pixel / 255 - mean) / std
            blob = resized.astype(np.float32) / 255.0
            blob = (blob - self._mean) / self._std
            blob = blob.transpose(2, 0, 1)  # HWC -> CHW
            blob = blob[np.newaxis, ...]  # Add batch dim: 1xCxHxW

            try:
                outputs = self.session.run(None, {input_name: blob})
                self._input_size_current = infer_size
                break
            except Exception as exc:
                if infer_size == requested_size and not self._cpu_fallback_active:
                    message = str(exc).lower()
                    if any(token in message for token in ("memory", "cuda", "cudnn", "bfcarena", "allocate")):
                        self._fallback_to_cpu(exc)
                        outputs = self.session.run(None, {input_name: blob})
                        self._input_size_current = infer_size
                        break
                last_error = exc
                continue

        if outputs is None:
            if not self._fallback_logged and last_error is not None:
                print(
                    f"[NV Broadcast] {self._info['name']} inference failed at <= {requested_size}px: {last_error}",
                    flush=True,
                )
                self._fallback_logged = True
            raise last_error

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

        # Light temporal smoothing at inference level — just enough to reduce
        # model noise, NOT enough to cause visible lag during movement.
        # Heavy flicker reduction is handled by bilateral filter + compositing layer.
        if self._prev_alpha is not None:
            dropping = alpha < self._prev_alpha
            diff = np.abs(alpha - self._prev_alpha).mean()
            # During motion, almost no smoothing. When still, light smoothing.
            motion_scale = max(0.0, 1.0 - diff * 8.0)
            weight = np.where(
                dropping, 0.02,
                0.12 * motion_scale,
            ).astype(np.float32)
            alpha = weight * self._prev_alpha + (1.0 - weight) * alpha
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


class _LearnedMatteRefiner:
    """Optional tiny learned post-refiner for replace/remove mattes.

    Loaded lazily from a locally trained ONNX file if present. This lets the
    runtime benefit from pseudo-label training without making the app depend on
    PyTorch or changing behavior when no trained model exists.
    """

    def __init__(self, gpu_index: int, variant: str):
        self._gpu_index = gpu_index
        self._variant = variant
        self._model_path = _LEARNED_REFINER_MODELS[variant]
        self._meta_path = self._model_path.with_suffix(".json")
        self._enabled = os.getenv("NVBROADCAST_ENABLE_LEARNED_REFINER", "").lower() in {"1", "true", "yes", "on"}
        self._session = None
        self._input_name = None
        self._output_name = None
        self._load_attempted = False
        self._failed = False
        self._target_size = 512
        self._input_channels = 4
        env_cap = os.getenv("NVBROADCAST_LEARNED_REFINER_MAX_SIZE", "").strip()
        try:
            self._runtime_target_size = max(128, int(env_cap)) if env_cap else 256
        except ValueError:
            self._runtime_target_size = 256

    @property
    def available(self) -> bool:
        return self._enabled and self._model_path.exists()

    def _ensure_session(self) -> bool:
        if self._failed:
            return False
        if self._session is not None:
            return True
        if not self.available:
            return False
        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            if self._meta_path.exists():
                try:
                    meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                    self._target_size = int(meta.get("size", self._target_size))
                    self._input_channels = int(meta.get("input_channels", self._input_channels))
                    self._runtime_target_size = min(self._target_size, self._runtime_target_size)
                except Exception:
                    pass
            self._session = _create_session(str(self._model_path), self._gpu_index, use_tensorrt=False)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            shape = self._session.get_inputs()[0].shape
            if len(shape) >= 2 and isinstance(shape[1], int):
                self._input_channels = shape[1]
            print(f"[NV Broadcast] Learned {self._variant} refiner loaded from {self._model_path.name}")
            return True
        except Exception as e:
            self._failed = True
            print(f"[NV Broadcast] Learned {self._variant} refiner unavailable: {e}")
            return False

    def _build_trimap(self, matte: np.ndarray) -> np.ndarray:
        fg_threshold = 0.97 if self._variant == "replace" else 0.98
        bg_threshold = 0.03 if self._variant == "replace" else 0.02
        radius = 3 if self._variant == "replace" else 2
        trimap = np.full(matte.shape, 0.5, dtype=np.float32)
        fg_seed = (matte >= fg_threshold).astype(np.uint8)
        bg_seed = (matte <= bg_threshold).astype(np.uint8)
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            fg_seed = cv2.erode(fg_seed, kernel, iterations=1)
            bg_seed = cv2.erode(bg_seed, kernel, iterations=1)
        trimap[bg_seed > 0] = 0.0
        trimap[fg_seed > 0] = 1.0
        return trimap

    def _build_band(self, matte: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        coarse_band = (matte > 0.02) & (matte < 0.98)
        trimap_band = (trimap > 0.25) & (trimap < 0.75)
        band = coarse_band | trimap_band
        if band.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            band = cv2.dilate(band.astype(np.uint8), kernel, iterations=1) > 0
        return band.astype(np.float32)

    def refine(self, frame: np.ndarray, matte: np.ndarray) -> np.ndarray:
        if not self._ensure_session():
            return matte

        transition = ((matte > 0.02) & (matte < 0.98)).astype(np.uint8)
        if transition.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            band = cv2.dilate(transition, kernel, iterations=1) > 0
        else:
            return matte

        height, width = matte.shape[:2]
        ys, xs = np.where(band)
        if ys.size == 0 or xs.size == 0:
            return matte
        pad = 28 if self._variant == "replace" else 24
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(width, int(xs.max()) + pad + 1)
        y1 = min(height, int(ys.max()) + pad + 1)

        frame_roi = frame[y0:y1, x0:x1]
        matte_roi = matte[y0:y1, x0:x1]
        band_roi = band[y0:y1, x0:x1]

        roi_h, roi_w = matte_roi.shape[:2]
        target_size = max(128, self._runtime_target_size)
        scale = min(1.0, float(target_size) / float(max(roi_h, roi_w)))
        if scale < 1.0:
            work_w = max(16, int(round(roi_w * scale)))
            work_h = max(16, int(round(roi_h * scale)))
            frame_work = cv2.resize(frame_roi, (work_w, work_h), interpolation=cv2.INTER_AREA)
            matte_work = cv2.resize(matte_roi, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_work = frame_roi
            matte_work = matte_roi

        rgb = cv2.cvtColor(frame_work, cv2.COLOR_BGRA2RGB).astype(np.float32) * (1.0 / 255.0)
        channels = [rgb.transpose(2, 0, 1), matte_work[None, :, :]]
        if self._input_channels >= 6:
            trimap_work = self._build_trimap(matte_work)
            band_work = self._build_band(matte_work, trimap_work)
            channels.extend([trimap_work[None, :, :], band_work[None, :, :]])
        tensor = np.concatenate(channels, axis=0)[None, ...].astype(np.float32)

        try:
            output = self._session.run([self._output_name], {self._input_name: tensor})[0]
        except Exception as e:
            print(f"[NV Broadcast] Learned {self._variant} refiner failed: {e}")
            return matte

        refined = np.clip(output[0, 0], 0.0, 1.0).astype(np.float32)
        if refined.shape != matte_roi.shape:
            refined = cv2.resize(refined, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        max_delta = 0.12 if self._variant == "replace" else 0.10
        clipped = np.clip(refined, matte_roi - max_delta, matte_roi + max_delta)
        blend = 0.55 if self._variant == "replace" else 0.60
        result = matte.copy()
        result_roi = result[y0:y1, x0:x1]
        result_roi[band_roi] = matte_roi[band_roi] * (1.0 - blend) + clipped[band_roi] * blend
        result[result < 0.02] = 0.0
        result[result > 0.985] = 1.0
        return result.astype(np.float32)

    def cleanup(self):
        _release_session(self._session)
        self._session = None


# ─── Main VideoEffects Class ─────────────────────────────────────────────────

class VideoEffects:
    def __init__(self, gpu_index: int = COMPUTE_GPU_INDEX, edge_config=None,
                 compositing: str = "cpu"):
        self._gpu_index = gpu_index
        self._initialized = False
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._quality = "quality"
        self._model_type = "rvm"
        self._backend = None
        self._edge_config = edge_config
        self._use_tensorrt = False   # Zeus/Killer mode
        self._use_fused_kernel = False  # DocZeus/Killer mode
        self._edge_refine_enabled = False  # Edge refinement toggle
        self._edge_refiner = EdgeRefiner(gpu_index)
        self._learned_refiners = {
            "replace": _LearnedMatteRefiner(gpu_index, "replace"),
            "remove": _LearnedMatteRefiner(gpu_index, "remove"),
        }
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
        self._bg_resized = None  # Used by fused kernel path (_get_bg_image)
        self._green_bg = None
        self._frame_counter = 0
        self._cached_alpha = None
        self._prev_alpha = None  # Previous frame's alpha for temporal smoothing
        self._stable_alpha = None  # Pre-tightened alpha for replacement smoothing
        self._matte_version = 0
        self._skip_interval = 1
        self._temporal_strength = 0.34  # EMA weight for temporal smoothing

        # Alpha refinement
        self._apply_edge_config(edge_config)
        self._refresh_temporal_strength()

    def reset_cached_mattes(self):
        """Reset cached mattes after mode, resolution, or backend changes."""
        with self._state_lock:
            self._matte_version += 1
            self._cached_alpha = None
            self._prev_alpha = None
            self._stable_alpha = None

    def _matte_snapshot(self) -> tuple[np.ndarray | None, int]:
        """Return the current cached alpha and its generation."""
        with self._state_lock:
            return self._cached_alpha, self._matte_version

    def _commit_alpha(self, alpha: np.ndarray, matte_version: int) -> bool:
        """Store an alpha mask only if matte state has not been invalidated."""
        with self._state_lock:
            if matte_version != self._matte_version:
                return False
            self._cached_alpha = alpha
            return True

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def enabled(self) -> bool:
        return self._bg_removal_enabled

    @enabled.setter
    def enabled(self, value: bool):
        if value != self._bg_removal_enabled:
            self.reset_cached_mattes()
        self._bg_removal_enabled = value
        if value and not self._initialized:
            self.initialize()

    @property
    def mode(self) -> str:
        return self._bg_mode

    @mode.setter
    def mode(self, value: str):
        if value in ("blur", "replace", "remove"):
            if value != self._bg_mode:
                self._bg_mode = value
                self._refresh_temporal_strength()
                self.reset_cached_mattes()

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
        self._refresh_temporal_strength()
        if self._initialized:
            self._cleanup_backend()
            self.initialize()

    def set_background_image(self, image_path: str) -> bool:
        if not image_path or not os.path.exists(image_path):
            self._bg_image = None
            self._bg_image_path = ""
            self._resized_bg = None
            self._bg_resized = None
            with self._state_lock:
                self._stable_alpha = None
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
        self._resized_bg = None
        self._bg_resized = None
        with self._state_lock:
            self._stable_alpha = None
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
            else:
                backend = _SingleFrameBackend(self._gpu_index, self._model_type)
                msg = backend.load()

            with self._lock:
                self._backend = backend
                self._initialized = True

            print(f"[NV Broadcast] {msg}")
            return True

        except Exception as e:
            print(f"[NV Broadcast] Failed to initialize model: {e}")
            return False

    def update_alpha(self, frame_data: bytes, width: int, height: int) -> None:
        """Run inference to update the cached alpha mask. Call from background thread."""
        if not self._bg_removal_enabled or not self._initialized:
            return
        _, matte_version = self._matte_snapshot()

        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
        if not frame.flags.writeable:
            frame = frame.copy()

        alpha = self._run_inference(frame, width, height, matte_version)
        if alpha is not None:
            self._commit_alpha(alpha, matte_version)

    def _temporal_smooth(self, alpha: np.ndarray, matte_version: int | None = None) -> np.ndarray:
        """Motion-adaptive temporal EMA to eliminate edge jitter.

        Still: strong smoothing eliminates flicker completely.
        Moving: light smoothing so edges track movement.
        Hair/edges: always get extra smoothing since they jitter most.
        """
        with self._state_lock:
            if matte_version is not None and matte_version != self._matte_version:
                return alpha
            prev = self._prev_alpha
        if prev is None or prev.shape != alpha.shape:
            with self._state_lock:
                if matte_version is not None and matte_version != self._matte_version:
                    return alpha
                self._prev_alpha = alpha.copy()
            return alpha

        # Per-pixel motion and global motion
        diff = np.abs(alpha - prev)
        global_motion = diff.mean()

        # Motion gate: inversely proportional to movement
        # Still → 1.0, fast motion → 0.15 (never zero — always some smoothing)
        motion_gate = np.clip(1.0 - global_motion * 20.0, 0.15, 1.0)
        base_w = self._temporal_strength * motion_gate

        if self._bg_mode == "remove":
            base_scale = 0.12
            edge_scale = 1.6
            fringe_scale = 2.2
            max_weight = 0.55
            fringe_min, fringe_max = 0.05, 0.22
        elif self._bg_mode == "replace":
            base_scale = 0.16
            edge_scale = 2.0
            fringe_scale = 2.8
            max_weight = 0.62
            fringe_min, fringe_max = 0.03, 0.26
        else:
            base_scale = 0.2
            edge_scale = 2.5
            fringe_scale = 3.5
            max_weight = 0.7
            fringe_min, fringe_max = 0.03, 0.30

        # Three-tier smoothing weights:
        # Edge/transition (0.03-0.97): strongest — these jitter most
        # Core (>0.97): minimal — solid person
        # Background (<0.03): minimal — stable already
        edge_mask = (alpha > 0.03) & (alpha < 0.97)
        weight = np.full_like(alpha, base_w * base_scale, dtype=np.float32)
        weight[edge_mask] = base_w * edge_scale
        # Thin fringe (0.03-0.3): extra heavy smoothing — most jittery zone
        thin_fringe = (alpha > fringe_min) & (alpha < fringe_max)
        weight[thin_fringe] = base_w * fringe_scale
        # Cap weight to prevent ghosting
        np.clip(weight, 0, max_weight, out=weight)

        # Asymmetric: dropping alpha (person leaving an area) gets less
        # smoothing to avoid ghost trails
        dropping = alpha < prev - 0.05
        weight[dropping] *= 0.12 if self._bg_mode == "remove" else 0.2

        result = weight * prev + (1.0 - weight) * alpha
        with self._state_lock:
            if matte_version is not None and matte_version != self._matte_version:
                return alpha
            self._prev_alpha = result.copy()
        return result

    def _refresh_temporal_strength(self) -> None:
        """Tune temporal smoothing for the active model/engine/effect mode."""
        if self._model_type == "rvm":
            if self._use_tensorrt and self._use_fused_kernel:
                strength = 0.42  # Killer: fastest, needs extra stabilization
            elif self._use_tensorrt:
                strength = 0.39  # Zeus: low-res inference, still needs help
            elif self._use_fused_kernel:
                strength = 0.33  # DocZeus: quality path, keep edges tighter
            else:
                strength = 0.34
        else:
            strength = 0.20  # Single-frame models already have their own EMA

        if self._bg_mode == "remove":
            strength -= 0.07
        elif self._bg_mode == "replace":
            strength -= 0.02

        self._temporal_strength = float(np.clip(strength, 0.12, 0.45))

    @staticmethod
    def _fill_small_internal_holes(mask_u8: np.ndarray,
                                   binary_threshold: int,
                                   fill_cutoff: int,
                                   fill_value: int,
                                   max_area_ratio: float,
                                   max_span_ratio: float) -> np.ndarray:
        """Fill only tiny interior holes while preserving real body gaps.

        Large openings like under-arm gaps or spaces between hair strands and
        shoulders are legitimate structure, not defects. Only fill compact hole
        components that are small relative to the frame and silhouette span.
        """
        _, binary = cv2.threshold(mask_u8, binary_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_u8

        largest = max(contours, key=cv2.contourArea)
        interior = np.zeros_like(mask_u8)
        cv2.drawContours(interior, [largest], -1, 255, cv2.FILLED)

        holes = ((interior == 255) & (binary == 0)).astype(np.uint8)
        if not holes.any():
            return mask_u8

        h, w = mask_u8.shape[:2]
        max_area = max(6, int(h * w * max_area_ratio))
        max_w = max(2, int(w * max_span_ratio))
        max_h = max(2, int(h * max_span_ratio))

        num, labels, stats, _centroids = cv2.connectedComponentsWithStats(holes, connectivity=8)
        filled = mask_u8.copy()
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            span_w = int(stats[idx, cv2.CC_STAT_WIDTH])
            span_h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if area > max_area or span_w > max_w or span_h > max_h:
                continue
            component = labels == idx
            component &= filled < fill_cutoff
            filled[component] = fill_value
        return filled

    @staticmethod
    def _preserve_large_internal_holes(mask_u8: np.ndarray,
                                       binary_threshold: int,
                                       min_area_ratio: float,
                                       min_span_ratio: float) -> np.ndarray:
        """Return a mask of meaningful interior openings that should stay open."""
        _, binary = cv2.threshold(mask_u8, binary_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask_u8, dtype=bool)

        largest = max(contours, key=cv2.contourArea)
        interior = np.zeros_like(mask_u8)
        cv2.drawContours(interior, [largest], -1, 255, cv2.FILLED)

        holes = ((interior == 255) & (binary == 0)).astype(np.uint8)
        if not holes.any():
            return np.zeros_like(mask_u8, dtype=bool)

        h, w = mask_u8.shape[:2]
        min_area = max(6, int(h * w * min_area_ratio))
        min_w = max(3, int(w * min_span_ratio))
        min_h = max(3, int(h * min_span_ratio))

        preserve = np.zeros_like(mask_u8, dtype=bool)
        num, labels, stats, _centroids = cv2.connectedComponentsWithStats(holes, connectivity=8)
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            span_w = int(stats[idx, cv2.CC_STAT_WIDTH])
            span_h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if area < min_area and span_w < min_w and span_h < min_h:
                continue
            preserve |= labels == idx
        return preserve

    def _replacement_matte(self, alpha: np.ndarray,
                           matte_version: int | None = None) -> np.ndarray:
        """Stabilize and slightly tighten the matte for image replacement.

        Blur mode can tolerate a softer edge because the source and destination
        backgrounds are visually similar. Replacement mode cannot; the fringe is
        immediately visible as a halo. This path keeps replacement mattes more
        stable without smearing motion.
        """
        with self._state_lock:
            if matte_version is not None and matte_version != self._matte_version:
                return alpha
            prev = self._stable_alpha
        if prev is None or prev.shape != alpha.shape:
            stable = alpha.copy()
        else:
            diff = np.abs(alpha - prev)
            global_motion = diff.mean()
            motion_gate = np.clip(1.0 - global_motion * 18.0, 0.2, 1.0)

            # Keep replace-mode stabilization lighter than blur/remove so
            # narrow hair gaps and under-arm openings can reopen quickly.
            weight = np.full_like(alpha, 0.07 * motion_gate, dtype=np.float32)
            edge_mask = (alpha > 0.02) & (alpha < 0.98)
            fringe_mask = (alpha > 0.02) & (alpha < 0.35)
            weight[edge_mask] = 0.15 * motion_gate
            weight[fringe_mask] = 0.21 * motion_gate

            dropping = alpha < prev - 0.04
            weight[dropping] *= 0.18
            np.clip(weight, 0.0, 0.30, out=weight)

            stable = weight * prev + (1.0 - weight) * alpha

        matte = np.clip((stable - 0.05) / 0.95, 0.0, 1.0)
        matte[matte < 0.012] = 0.0
        matte[matte > 0.985] = 1.0

        matte_u8 = np.clip(matte * 255.0, 0, 255).astype(np.uint8)
        preserve_holes = self._preserve_large_internal_holes(
            matte_u8,
            binary_threshold=74,
            min_area_ratio=0.00010,
            min_span_ratio=0.030,
        )
        matte_u8 = self._fill_small_internal_holes(
            matte_u8,
            binary_threshold=78,
            fill_cutoff=58,
            fill_value=184,
            max_area_ratio=0.00008,
            max_span_ratio=0.024,
        )

        if min(matte_u8.shape[:2]) >= 8:
            matte_u8 = cv2.GaussianBlur(matte_u8, (3, 3), 0)
        matte = matte_u8.astype(np.float32) * (1.0 / 255.0)

        mid = (matte > 0.04) & (matte < 0.55)
        matte[mid] = np.clip((matte[mid] - 0.05) / 0.95, 0.0, 1.0)
        fringe = (matte > 0.0) & (matte < 0.22)
        matte[fringe] *= 0.72
        solid = stable > 0.86
        matte[solid] = np.maximum(matte[solid], 0.995)
        if preserve_holes.any():
            matte[preserve_holes] = np.minimum(matte[preserve_holes], alpha[preserve_holes])
        matte[matte < 0.12] = 0.0

        with self._state_lock:
            if matte_version is not None and matte_version != self._matte_version:
                return alpha
            self._stable_alpha = stable.copy()
        return matte

    def _edge_aware_replace_matte(self, frame: np.ndarray, matte: np.ndarray) -> np.ndarray:
        """Sharpen replace-mode transitions where the camera frame has a real edge.

        The model output is intentionally smoothed for stability, which leaves a
        soft halo around hair, glasses, and shoulders when compositing onto a
        very different background. Use the frame luminance gradient as a local
        confidence signal and only harden the transition band where the image
        itself supports a boundary.
        """
        if matte is None or min(matte.shape[:2]) < 8:
            return matte

        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY).astype(np.float32) * (1.0 / 255.0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad = cv2.GaussianBlur(grad, (3, 3), 0)
        edge_strength = np.clip((grad - 0.03) / 0.20, 0.0, 1.0)

        transition = (matte > 0.05) & (matte < 0.95)
        if not transition.any():
            return matte

        focus = edge_strength * transition.astype(np.float32)
        if float(focus.max()) < 0.08:
            return matte

        eps = 1e-4
        clipped = np.clip(matte, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped))
        sharpened = 1.0 / (1.0 + np.exp(-(logits * (1.0 + 1.75 * focus))))

        blend = np.clip(focus * 0.80, 0.0, 0.80)
        result = matte * (1.0 - blend) + sharpened * blend

        supported_fine_fringe = (matte > 0.05) & (matte < 0.18) & (edge_strength >= 0.10)
        result[supported_fine_fringe] = np.maximum(
            result[supported_fine_fringe],
            matte[supported_fine_fringe] * 0.78,
        )

        weak_edge_fringe = (result > 0.0) & (result < 0.12) & (edge_strength < 0.12)
        result[weak_edge_fringe] *= 0.82
        result[(result < 0.06) & (edge_strength < 0.10)] = 0.0
        result[result > 0.985] = 1.0
        return result.astype(np.float32)

    def _greenscreen_matte(self, frame: np.ndarray, alpha: np.ndarray,
                           matte_version: int | None = None) -> np.ndarray:
        """Build a tighter matte for green-screen output.

        Remove mode is effectively a chroma-key source. It should sacrifice a
        little feathering if needed to avoid a muddy dark halo against pure
        green. Start from the replace matte, then harden weak fringe regions
        more aggressively.
        """
        matte = self._replacement_matte(alpha, matte_version)
        matte = self._edge_aware_replace_matte(frame, matte)

        matte_u8 = np.clip(matte * 255.0, 0, 255).astype(np.uint8)
        if min(matte_u8.shape[:2]) >= 8:
            matte_u8 = cv2.GaussianBlur(matte_u8, (3, 3), 0)
        matte = matte_u8.astype(np.float32) * (1.0 / 255.0)

        soft = (matte > 0.0) & (matte < 0.55)
        matte[soft] = np.clip((matte[soft] - 0.08) / 0.92, 0.0, 1.0)

        weak = (matte > 0.0) & (matte < 0.22)
        matte[weak] *= 0.45
        matte[matte < 0.12] = 0.0
        matte[matte > 0.96] = 1.0
        return matte.astype(np.float32)

    def _apply_learned_refiner(self, variant: str, frame: np.ndarray, matte: np.ndarray) -> np.ndarray:
        refiner = self._learned_refiners.get(variant)
        if refiner is None or not refiner.available:
            return matte
        return refiner.refine(frame, matte)

    def _final_matte(self, frame: np.ndarray, alpha: np.ndarray,
                     matte_version: int | None = None) -> np.ndarray:
        if self._bg_mode == "replace":
            matte = self._replacement_matte(alpha, matte_version)
            matte = self._edge_aware_replace_matte(frame, matte)
            return self._apply_learned_refiner("replace", frame, matte)
        if self._bg_mode == "remove":
            matte = self._greenscreen_matte(frame, alpha, matte_version)
            return self._apply_learned_refiner("remove", frame, matte)
        return alpha

    def composite_only(self, frame_data: bytes, width: int, height: int) -> bytes:
        """Composite current frame with cached alpha. Fast — no inference."""
        if not self._bg_removal_enabled or not self._initialized:
            return frame_data
        alpha, matte_version = self._matte_snapshot()
        if alpha is None:
            return frame_data

        # Resolution changed — alpha is wrong size, resize it
        if alpha.shape[0] != height or alpha.shape[1] != width:
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
        if not frame.flags.writeable:
            frame = frame.copy()

        return self._composite(frame, alpha, width, height, matte_version)

    def _composite(self, frame: np.ndarray, alpha: np.ndarray,
                   width: int, height: int,
                   matte_version: int | None = None) -> bytes:
        """Apply alpha mask to frame — shared by process_frame and composite_only."""
        # Ensure alpha matches frame dimensions
        if alpha.shape[0] != height or alpha.shape[1] != width:
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

        alpha = self._final_matte(frame, alpha, matte_version)

        # Fused CUDA kernel path (DocZeus/Killer) — single GPU pass
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

    def process_frame(self, frame_data: bytes, width: int, height: int) -> bytes:
        if not self._bg_removal_enabled or not self._initialized:
            return frame_data

        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
        if not frame.flags.writeable:
            frame = frame.copy()

        self._frame_counter += 1
        alpha, matte_version = self._matte_snapshot()

        run_inference = (
            self._skip_interval <= 1
            or self._frame_counter % self._skip_interval == 0
            or alpha is None
        )
        if run_inference:
            alpha = self._run_inference(frame, width, height, matte_version)
            if alpha is not None:
                if not self._commit_alpha(alpha, matte_version):
                    alpha, matte_version = self._matte_snapshot()
            elif alpha is None:
                alpha, matte_version = self._matte_snapshot()
            if alpha is None:
                return frame_data
        elif alpha is None:
            alpha, matte_version = self._matte_snapshot()
            if alpha is None:
                return frame_data

        return self._composite(frame, alpha, width, height, matte_version)

    def _run_inference(self, frame: np.ndarray, width: int, height: int,
                       matte_version: int | None = None) -> np.ndarray | None:
        """Run the active backend's inference and refine the alpha."""
        with self._lock:
            backend = self._backend
            if backend is None:
                return None

        try:
            alpha = backend.infer(frame, width, height)
            if alpha is not None:
                # Refine first, then temporal smooth on the FINAL output.
                # RVM's raw edges jitter 6-8% — smoothing the final result
                # directly stabilizes what the user sees.
                if not getattr(backend, '_lowres_refined', False):
                    alpha = self._refine_alpha(alpha)
                # Edge refine: second pass at 720p for Zeus/Killer modes
                if self._edge_refine_enabled:
                    if not self._edge_refiner._initialized:
                        self._edge_refiner.initialize(self._quality)
                    raw_refined = self._edge_refiner.refine(frame, alpha, width, height)
                    alpha = self._refine_alpha(raw_refined)
                alpha = self._temporal_smooth(alpha, matte_version)
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
        a8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        is_replace = self._bg_mode == "replace"
        preserve_holes = None
        if is_replace:
            preserve_holes = self._preserve_large_internal_holes(
                a8,
                binary_threshold=70,
                min_area_ratio=0.00012,
                min_span_ratio=0.032,
            )

        # 1. Small close: fill tiny holes in hair/fine detail
        close_sm_size = 3 if is_replace else 5
        close_sm = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_sm_size, close_sm_size)
        )
        a8 = cv2.morphologyEx(a8, cv2.MORPH_CLOSE, close_sm)

        # 2. Half-resolution close: keep replacement tighter, keep blur/remove
        #    more forgiving where wide feathering hides seams.
        h, w = a8.shape[:2]
        small = cv2.resize(a8, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        close_size = 3 if is_replace else 25
        close_lg = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, close_lg)
        a8 = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. Fill interior holes. Replacement mode is intentionally more
        #    conservative so we do not inflate the subject outline.
        threshold = 72 if is_replace else 30
        fill_cutoff = 46 if is_replace else 100
        fill_value = 160 if is_replace else 220
        a8 = self._fill_small_internal_holes(
            a8,
            binary_threshold=threshold,
            fill_cutoff=fill_cutoff,
            fill_value=fill_value,
            max_area_ratio=0.00009 if is_replace else 0.0007,
            max_span_ratio=0.026 if is_replace else 0.07,
        )

        if is_replace:
            # 4. Keep replacement narrow. Inflating the silhouette is what
            #    creates shoulder and ear halos against a swapped background.
            a8 = cv2.GaussianBlur(a8, (5, 5), 0)
            a8 = cv2.GaussianBlur(a8, (3, 3), 0)

            # 5. Moderate-strong sigmoid
            t = a8.astype(np.float32) * (1.0 / 255.0)
            sig = self._sigmoid_strength * 0.60
            mid = self._sigmoid_midpoint
            if sig > 0:
                t = 1.0 / (1.0 + np.exp(-sig * (t - mid)))

            # 6. Core solidification
            result = t
            core = result > 0.78
            result[core] = 1.0 - (1.0 - result[core]) ** 2.2
            result[result < 0.03] = 0.0
            if preserve_holes is not None and preserve_holes.any():
                result[preserve_holes] = np.minimum(result[preserve_holes], alpha[preserve_holes])

            # 7. Final feathering
            r_u8 = np.clip(result * 255, 0, 255).astype(np.uint8)
            r_u8 = cv2.GaussianBlur(r_u8, (3, 3), 0)
            result = r_u8.astype(np.float32) * (1.0 / 255.0)
        else:
            # 4. Wide dilate for blur/remove: 2 passes with 7x7
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            a8 = cv2.dilate(a8, dilate_k, iterations=2)

            # 5. Three-pass Gaussian: wide, smooth gradient
            a8 = cv2.GaussianBlur(a8, (17, 17), 0)
            a8 = cv2.GaussianBlur(a8, (11, 11), 0)
            a8 = cv2.GaussianBlur(a8, (7, 7), 0)

            # 6. Moderate sigmoid
            t = a8.astype(np.float32) * (1.0 / 255.0)
            sig = self._sigmoid_strength * 0.45
            mid = self._sigmoid_midpoint
            if sig > 0:
                t = 1.0 / (1.0 + np.exp(-sig * (t - mid)))

            # 7. Core solidification + noise suppression
            result = t
            core = result > 0.75
            result[core] = 1.0 - (1.0 - result[core]) ** 2.0
            result[result < 0.02] = 0.0

            # 8. Final feathering
            r_u8 = np.clip(result * 255, 0, 255).astype(np.uint8)
            r_u8 = cv2.GaussianBlur(r_u8, (11, 11), 0)
            result = r_u8.astype(np.float32) * (1.0 / 255.0)

        return result

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
        if self._resized_bg is None or self._frame_size != (width, height):
            self._resized_bg = self._resize_bg(self._bg_image, width, height)
            self._frame_size = (width, height)
        return self._resized_bg

    # ─── Compositing ─────────────────────────────────────────────────────

    def set_engine_mode(self, use_tensorrt: bool, use_fused_kernel: bool):
        """Set inference/compositing engine.

        Zeus/Killer modes use aggressive 480p/360p pre-downsampling for faster inference.
        DocZeus/Killer modes use a fused CUDA kernel for single-pass compositing.
        """
        self._use_tensorrt = use_tensorrt
        self._use_fused_kernel = use_fused_kernel
        self._refresh_temporal_strength()
        if self._backend and hasattr(self._backend, '_MAX_INFER_HEIGHT'):
            if use_tensorrt and use_fused_kernel:
                new_h = 360   # Killer
            elif use_tensorrt:
                new_h = 480   # Zeus
            else:
                new_h = 720   # Standard/DocZeus
            with self._lock:
                old_h = self._backend._MAX_INFER_HEIGHT
                self._backend._MAX_INFER_HEIGHT = new_h
                # Only reset recurrent states if resolution actually changed
                if old_h != new_h:
                    self._backend.reset_state()
                    self.reset_cached_mattes()
                    print(f"[NV Broadcast] Inference resolution: {old_h}p → {new_h}p")

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
        frame = self._prepare_greenscreen_foreground(frame, alpha)
        return self._blend(frame, self._green_bg, alpha)

    def _clean_color_reference(self, fg: np.ndarray, alpha: np.ndarray,
                               solid_threshold: float = 0.88) -> np.ndarray:
        """Estimate clean foreground colors from solid subject pixels."""
        # Create a "clean color" reference by blurring only solid person pixels.
        # Process at half resolution for speed (32ms → ~4ms).
        h, w = fg.shape[:2]
        hw, hh = w // 2, h // 2
        fg_small = cv2.resize(fg, (hw, hh), interpolation=cv2.INTER_AREA)
        alpha_small = cv2.resize(alpha, (hw, hh), interpolation=cv2.INTER_AREA)
        solid_2d = (alpha_small > solid_threshold).astype(np.float32)
        solid_3d = solid_2d[:, :, np.newaxis]
        weighted_fg = fg_small.astype(np.float32) * solid_3d
        weighted_sum = cv2.GaussianBlur(weighted_fg, (11, 11), 0)
        wt_2d = cv2.GaussianBlur(solid_2d, (11, 11), 0)
        wt = np.maximum(wt_2d[:, :, np.newaxis], 0.001)
        clean_small = (weighted_sum / wt).astype(np.uint8)
        return cv2.resize(clean_small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _despill_fringe(self, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Remove webcam background color bleeding into hair/edge fringe.

        Fringe pixels are a blend of person + webcam background. The lower
        the alpha, the more contaminated. Strategy: pull fringe pixel colors
        toward nearby solid person pixels using a weighted blur.
        """
        fringe = (alpha > 0.03) & (alpha < 0.35)
        if not fringe.any():
            return fg
        result = fg.copy()
        clean_color = self._clean_color_reference(fg, alpha, solid_threshold=0.88)

        # Blend fringe pixels toward clean color based on how contaminated they are
        # alpha=0.03 → ~45% clean color, alpha=0.35 → 0%
        color_delta = np.mean(
            np.abs(
                result[:, :, :3].astype(np.int16) - clean_color[:, :, :3].astype(np.int16)
            ),
            axis=2,
        )
        contamination = color_delta > 10.0
        fringe &= contamination
        if not fringe.any():
            return result

        blend = np.clip((0.35 - alpha) / 0.32, 0.0, 1.0) * 0.45
        blend_4ch = blend[:, :, np.newaxis].astype(np.float32)
        fringe_4ch = fringe[:, :, np.newaxis]

        result = np.where(
            fringe_4ch,
            np.clip(
                result.astype(np.float32) * (1.0 - blend_4ch) + clean_color.astype(np.float32) * blend_4ch,
                0, 255
            ).astype(np.uint8),
            result,
        )
        return result

    def _prepare_greenscreen_foreground(self, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Clean fringe colors more aggressively for green-screen output."""
        result = self._despill_fringe(fg, alpha)
        fringe = (alpha > 0.02) & (alpha < 0.55)
        if not fringe.any():
            return result

        clean_color = self._clean_color_reference(result, alpha, solid_threshold=0.9)
        color_delta = np.mean(
            np.abs(
                result[:, :, :3].astype(np.int16) - clean_color[:, :, :3].astype(np.int16)
            ),
            axis=2,
        )
        contamination = color_delta > 6.0
        fringe &= contamination
        if not fringe.any():
            return result

        result_f = result.astype(np.float32)
        clean_f = clean_color.astype(np.float32)
        alpha_f = alpha.astype(np.float32)
        luma = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        clean_luma = cv2.cvtColor(clean_color[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)

        blend = np.clip((0.55 - alpha_f) / 0.50, 0.0, 1.0) * 0.78
        dark_halo = fringe & (luma + 10.0 < clean_luma)
        blend[dark_halo] = np.maximum(blend[dark_halo], 0.88)

        blend_4ch = blend[:, :, np.newaxis]
        fringe_4ch = fringe[:, :, np.newaxis]
        repaired = np.clip(result_f * (1.0 - blend_4ch) + clean_f * blend_4ch, 0, 255)

        return np.where(fringe_4ch, repaired.astype(np.uint8), result)

    def _apply_replace(self, frame: np.ndarray, alpha: np.ndarray,
                       width: int, height: int) -> np.ndarray:
        if self._bg_image is None:
            return self._apply_blur(frame, alpha)
        if self._frame_size != (width, height):
            self._resized_bg = self._resize_bg(self._bg_image, width, height)
            self._frame_size = (width, height)
        # Despill: remove webcam background color from hair fringe pixels
        frame = self._despill_fringe(frame, alpha)
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
        self.reset_cached_mattes()

    def cleanup(self):
        self._cleanup_backend()
        self._edge_refiner.cleanup()
        for refiner in self._learned_refiners.values():
            refiner.cleanup()

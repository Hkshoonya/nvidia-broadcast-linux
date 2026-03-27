#!/usr/bin/env python3
"""Offline background-quality harness for matting/compositing evaluation.

Run a saved frame or a live camera capture through the current VideoEffects
pipeline and save debug artifacts for comparison across code changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nvbroadcast.video.effects import VideoEffects  # noqa: E402


MODE_PRESETS: dict[str, dict[str, object]] = {
    "killer": {"quality": "quality", "compositing": "cupy", "tensorrt": True, "fused": True, "edge_refine": True},
    "zeus": {"quality": "quality", "compositing": "cupy", "tensorrt": True, "fused": False, "edge_refine": True},
    "doczeus": {"quality": "quality", "compositing": "cupy", "tensorrt": False, "fused": True, "edge_refine": False},
    "cuda_max": {"quality": "quality", "compositing": "cupy", "tensorrt": False, "fused": False, "edge_refine": False},
    "cuda_balanced": {"quality": "balanced", "compositing": "cupy", "tensorrt": False, "fused": False, "edge_refine": False},
    "cuda_perf": {"quality": "performance", "compositing": "cupy", "tensorrt": False, "fused": False, "edge_refine": False},
    "cpu_quality": {"quality": "quality", "compositing": "cpu", "tensorrt": False, "fused": False, "edge_refine": False},
    "cpu_light": {"quality": "performance", "compositing": "cpu", "tensorrt": False, "fused": False, "edge_refine": False},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to an input image/frame")
    source.add_argument("--camera", help="Video device to capture one frame from, e.g. /dev/video0")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--mode", default="replace", choices=["replace", "blur", "remove"])
    parser.add_argument(
        "--quality", default="quality",
        choices=["performance", "balanced", "quality", "ultra"],
        help="VideoEffects quality preset",
    )
    parser.add_argument("--model", choices=["rvm", "isnet", "birefnet"], help="Override segmentation model")
    parser.add_argument("--preset", choices=sorted(MODE_PRESETS), help="Apply an app-style engine preset")
    parser.add_argument("--compositing", choices=["cpu", "cupy", "gstreamer_gl"], help="Override compositing backend")
    parser.add_argument("--tensorrt", action="store_true", help="Enable TensorRT inference")
    parser.add_argument("--fused", action="store_true", help="Enable fused CUDA compositing")
    parser.add_argument("--background", help="Optional replacement background image")
    parser.add_argument("--out", default="artifacts/quality_eval", help="Output directory")
    parser.add_argument("--edge-refine", action="store_true", help="Enable premium edge refiner")
    parser.add_argument("--warmup", type=int, default=1, help="Number of unmeasured warmup inferences")
    parser.add_argument("--repeat", type=int, default=3, help="Number of measured inferences to average")
    return parser.parse_args()


def _capture_frame(device: str, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera device {device}")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame from {device}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    finally:
        cap.release()


def _load_frame(path: Path) -> np.ndarray:
    rgba = np.array(Image.open(path).convert("RGBA"))
    return rgba[:, :, [2, 1, 0, 3]].copy()


def _load_background(path: str | None, width: int, height: int) -> np.ndarray:
    if path:
        rgba = np.array(Image.open(path).convert("RGBA"))
        bg = rgba[:, :, [2, 1, 0, 3]].copy()
        if bg.shape[1] != width or bg.shape[0] != height:
            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return bg

    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    bg = np.zeros((height, width, 4), dtype=np.uint8)
    bg[..., 0] = np.clip(30 + 160 * x, 0, 255).astype(np.uint8)
    bg[..., 1] = np.clip(110 + 100 * y, 0, 255).astype(np.uint8)
    bg[..., 2] = np.clip(220 - 130 * x, 0, 255).astype(np.uint8)
    bg[..., 3] = 255
    return bg


def _save_rgba(path: Path, bgra: np.ndarray) -> None:
    rgba = bgra[:, :, [2, 1, 0, 3]]
    Image.fromarray(rgba).save(path)


def _save_mask(path: Path, alpha: np.ndarray) -> None:
    mask = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(path)


def _metric_dict(alpha: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(alpha.mean()),
        "min": float(alpha.min()),
        "max": float(alpha.max()),
        "nonzero_fraction": float((alpha > 0.0).mean()),
        "mid_fraction": float(((alpha > 0.05) & (alpha < 0.95)).mean()),
        "low_mid_fraction": float(((alpha > 0.05) & (alpha < 0.40)).mean()),
    }


def main() -> int:
    args = _parse_args()

    if args.input:
        frame = _load_frame(Path(args.input))
    else:
        frame = _capture_frame(args.camera, args.width, args.height)

    height, width = frame.shape[:2]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "model": args.model or "rvm",
        "quality": args.quality,
        "compositing": args.compositing or "cpu",
        "tensorrt": args.tensorrt,
        "fused": args.fused,
        "edge_refine": args.edge_refine,
    }
    if args.preset:
        settings.update(MODE_PRESETS[args.preset])
    if args.compositing:
        settings["compositing"] = args.compositing
    if args.tensorrt:
        settings["tensorrt"] = True
    if args.fused:
        settings["fused"] = True
    if args.edge_refine:
        settings["edge_refine"] = True

    fx = VideoEffects()
    fx.mode = args.mode
    fx.set_model(str(settings["model"]))
    fx.quality = str(settings["quality"])
    fx.set_compositing(str(settings["compositing"]))
    fx.set_engine_mode(bool(settings["tensorrt"]), bool(settings["fused"]))
    fx._edge_refine_enabled = bool(settings["edge_refine"])
    if args.mode == "replace":
        fx._bg_image = _load_background(args.background, width, height)
    t0 = time.perf_counter()
    if not fx.initialize():
        raise RuntimeError("VideoEffects initialization failed")
    timings = {"initialize_ms": (time.perf_counter() - t0) * 1000.0}
    fx.set_engine_mode(bool(settings["tensorrt"]), bool(settings["fused"]))

    for _ in range(max(0, args.warmup)):
        warm_alpha = fx._run_inference(frame, width, height)
        if warm_alpha is None:
            raise RuntimeError("warmup inference returned no alpha")

    infer_durations: list[float] = []
    alpha = None
    for _ in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        alpha = fx._run_inference(frame, width, height)
        infer_durations.append((time.perf_counter() - t0) * 1000.0)
    timings["infer_ms"] = sum(infer_durations) / len(infer_durations)
    timings["infer_runs"] = len(infer_durations)
    if alpha is None:
        raise RuntimeError("inference returned no alpha")

    metrics: dict[str, dict[str, float]] = {"alpha": _metric_dict(alpha)}
    _save_rgba(out_dir / "raw.png", frame)
    _save_mask(out_dir / "alpha.png", alpha)

    if args.mode == "replace":
        t0 = time.perf_counter()
        replacement_matte = fx._replacement_matte(alpha)
        timings["replacement_matte_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        edge_aware_matte = fx._edge_aware_replace_matte(frame, replacement_matte)
        timings["edge_aware_matte_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        final_matte = fx._final_matte(frame, alpha)
        timings["final_matte_ms"] = (time.perf_counter() - t0) * 1000.0
        metrics["replacement_matte"] = _metric_dict(replacement_matte)
        metrics["edge_aware_matte"] = _metric_dict(edge_aware_matte)
        metrics["final_matte"] = _metric_dict(final_matte)
        _save_mask(out_dir / "replacement_matte.png", replacement_matte)
        _save_mask(out_dir / "edge_aware_matte.png", edge_aware_matte)
        _save_mask(out_dir / "final_matte.png", final_matte)
    elif args.mode == "remove":
        t0 = time.perf_counter()
        greenscreen_matte = fx._greenscreen_matte(frame, alpha)
        timings["greenscreen_matte_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        final_matte = fx._final_matte(frame, alpha)
        timings["final_matte_ms"] = (time.perf_counter() - t0) * 1000.0
        metrics["greenscreen_matte"] = _metric_dict(greenscreen_matte)
        metrics["final_matte"] = _metric_dict(final_matte)
        _save_mask(out_dir / "greenscreen_matte.png", greenscreen_matte)
        _save_mask(out_dir / "final_matte.png", final_matte)

    t0 = time.perf_counter()
    composite = np.frombuffer(
        fx._composite(frame.copy(), alpha, width, height),
        dtype=np.uint8,
    ).reshape(height, width, 4)
    timings["composite_ms"] = (time.perf_counter() - t0) * 1000.0
    _save_rgba(out_dir / "composite.png", composite)

    payload = {"settings": settings, "timings_ms": timings, "metrics": metrics}
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

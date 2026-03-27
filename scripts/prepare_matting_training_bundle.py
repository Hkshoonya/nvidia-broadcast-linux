#!/usr/bin/env python3
"""Prepare a pseudo-labeled matting bundle from a captured dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
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
    parser.add_argument("--dataset", required=True, help="Directory containing frame_*.png files")
    parser.add_argument("--out", default="artifacts/training_bundle", help="Output root directory")
    parser.add_argument(
        "--label",
        default="",
        help="Optional suffix for the generated bundle directory",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODE_PRESETS),
        default="doczeus",
        help="Engine preset used while generating pseudo-labels",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["birefnet", "isnet", "rvm"],
        choices=["rvm", "isnet", "birefnet"],
        help="Segmentation model priority order for pseudo-label generation",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["replace", "remove"],
        choices=["replace", "remove"],
        help="Mask variants to export",
    )
    parser.add_argument(
        "--coarse-preset",
        choices=sorted(MODE_PRESETS),
        default="doczeus",
        help="Engine preset used to generate the live coarse matte input",
    )
    parser.add_argument(
        "--coarse-model",
        choices=["rvm", "isnet", "birefnet"],
        default="rvm",
        help="Model used to generate the live coarse matte input",
    )
    parser.add_argument("--teacher-gpu-index", type=int, default=0, help="GPU index used for teacher model inference")
    parser.add_argument("--coarse-gpu-index", type=int, default=0, help="GPU index used for coarse model inference")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Fraction of frames assigned to train split")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for train/val split")
    parser.add_argument("--trimap-radius", type=int, default=5, help="Unknown-band radius in pixels")
    parser.add_argument("--fg-threshold", type=float, default=0.97, help="Foreground threshold for trimap seeds")
    parser.add_argument("--bg-threshold", type=float, default=0.03, help="Background threshold for trimap seeds")
    return parser.parse_args()


def _load_frame(path: Path) -> np.ndarray:
    rgba = np.array(Image.open(path).convert("RGBA"))
    return rgba[:, :, [2, 1, 0, 3]].copy()


def _save_rgb(path: Path, bgra: np.ndarray) -> None:
    rgba = bgra[:, :, [2, 1, 0, 3]]
    Image.fromarray(rgba[:, :, :3]).save(path)


def _save_mask(path: Path, alpha: np.ndarray) -> None:
    Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def _save_trimap(path: Path, trimap: np.ndarray) -> None:
    Image.fromarray(trimap, mode="L").save(path)


def _metric_dict(alpha: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(alpha.mean()),
        "min": float(alpha.min()),
        "max": float(alpha.max()),
        "nonzero_fraction": float((alpha > 0.0).mean()),
        "mid_fraction": float(((alpha > 0.05) & (alpha < 0.95)).mean()),
        "low_mid_fraction": float(((alpha > 0.05) & (alpha < 0.40)).mean()),
    }


def _build_trimap(alpha: np.ndarray, fg_threshold: float, bg_threshold: float, radius: int) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    trimap = np.full(alpha.shape, 128, dtype=np.uint8)

    fg_seed = (alpha >= fg_threshold).astype(np.uint8)
    bg_seed = (alpha <= bg_threshold).astype(np.uint8)
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        fg_seed = cv2.erode(fg_seed, kernel, iterations=1)
        bg_seed = cv2.erode(bg_seed, kernel, iterations=1)

    trimap[bg_seed > 0] = 0
    trimap[fg_seed > 0] = 255
    return trimap


def _configure_effects(model: str, preset_name: str, gpu_index: int) -> VideoEffects:
    settings = MODE_PRESETS[preset_name]
    fx = VideoEffects(gpu_index=gpu_index)
    fx.mode = "replace"
    fx.set_model(model)
    fx.quality = str(settings["quality"])
    fx.set_compositing(str(settings["compositing"]))
    fx.set_engine_mode(bool(settings["tensorrt"]), bool(settings["fused"]))
    fx._edge_refine_enabled = bool(settings["edge_refine"])
    if not fx.initialize():
        raise RuntimeError(f"failed to initialize VideoEffects for model={model}")
    fx.set_engine_mode(bool(settings["tensorrt"]), bool(settings["fused"]))
    return fx


def _select_alpha(
    frame: np.ndarray,
    models: list[str],
    effects_cache: dict[str, VideoEffects],
    preset_name: str,
    gpu_index: int,
) -> tuple[str, np.ndarray, float]:
    height, width = frame.shape[:2]
    last_error: Exception | None = None

    for model in models:
        try:
            fx = effects_cache.get(model)
            if fx is None:
                fx = _configure_effects(model, preset_name, gpu_index)
                effects_cache[model] = fx

            t0 = time.perf_counter()
            alpha = fx._run_inference(frame, width, height)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            if alpha is None:
                raise RuntimeError(f"{model} returned no alpha")

            alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
            if not np.isfinite(alpha).all():
                raise RuntimeError(f"{model} produced non-finite alpha")
            return model, alpha, infer_ms
        except Exception as exc:  # pragma: no cover - exercised in live environment
            last_error = exc
            continue

    raise RuntimeError(f"all pseudo-label models failed; last error: {last_error}")


def main() -> int:
    args = _parse_args()
    dataset_dir = Path(args.dataset).resolve()
    frame_paths = sorted(dataset_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"no frame_*.png files found in {dataset_dir}")

    bundle_name = dataset_dir.name
    if args.label:
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in args.label).strip("-_")
        if safe_label:
            bundle_name = f"{bundle_name}-{safe_label}"

    out_root = Path(args.out).resolve() / bundle_name
    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    coarse_dir = out_root / "coarse"
    trimaps_dir = out_root / "trimaps"
    images_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        (masks_dir / variant).mkdir(parents=True, exist_ok=True)
        (coarse_dir / variant).mkdir(parents=True, exist_ok=True)
        (trimaps_dir / variant).mkdir(parents=True, exist_ok=True)

    source_manifest_path = dataset_dir / "manifest.json"
    source_manifest = None
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    teacher_effects_cache: dict[str, VideoEffects] = {}
    coarse_effects_cache: dict[str, VideoEffects] = {}
    variant_entries: dict[str, list[dict[str, object]]] = {variant: [] for variant in args.variants}
    model_counter: Counter[str] = Counter()
    coarse_counter: Counter[str] = Counter()

    print("frame teacher_model coarse_model teacher_ms coarse_ms replace_mid remove_mid")
    for frame_path in frame_paths:
        frame = _load_frame(frame_path)
        selected_model, alpha, infer_ms = _select_alpha(
            frame, list(args.models), teacher_effects_cache, args.preset, args.teacher_gpu_index
        )
        model_counter[selected_model] += 1
        coarse_model, coarse_alpha, coarse_infer_ms = _select_alpha(
            frame,
            [args.coarse_model],
            coarse_effects_cache,
            args.coarse_preset,
            args.coarse_gpu_index,
        )
        coarse_counter[coarse_model] += 1
        frame_id = frame_path.stem
        _save_rgb(images_dir / frame_path.name, frame)

        replace_mid = "-"
        remove_mid = "-"
        teacher_effects = teacher_effects_cache[selected_model]
        coarse_effects = coarse_effects_cache[coarse_model]

        for variant in args.variants:
            if variant == "replace":
                matte = teacher_effects._edge_aware_replace_matte(frame, teacher_effects._replacement_matte(alpha))
                coarse_matte = coarse_effects._edge_aware_replace_matte(frame, coarse_effects._replacement_matte(coarse_alpha))
            else:
                matte = teacher_effects._greenscreen_matte(frame, alpha)
                coarse_matte = coarse_effects._greenscreen_matte(frame, coarse_alpha)

            trimap = _build_trimap(matte, args.fg_threshold, args.bg_threshold, args.trimap_radius)
            mask_rel = Path("masks") / variant / frame_path.name
            coarse_rel = Path("coarse") / variant / frame_path.name
            trimap_rel = Path("trimaps") / variant / frame_path.name
            image_rel = Path("images") / frame_path.name

            _save_mask(out_root / mask_rel, matte)
            _save_mask(out_root / coarse_rel, coarse_matte)
            _save_trimap(out_root / trimap_rel, trimap)

            entry = {
                "id": f"{frame_id}:{variant}",
                "variant": variant,
                "image": str(image_rel),
                "coarse_alpha": str(coarse_rel),
                "mask": str(mask_rel),
                "trimap": str(trimap_rel),
                "source_frame": frame_path.name,
                "selected_model": selected_model,
                "coarse_model": coarse_model,
                "infer_ms": infer_ms,
                "coarse_infer_ms": coarse_infer_ms,
                "metrics": _metric_dict(matte),
                "coarse_metrics": _metric_dict(coarse_matte),
            }
            variant_entries[variant].append(entry)
            if variant == "replace":
                replace_mid = f"{entry['metrics']['mid_fraction']:.6f}"
            else:
                remove_mid = f"{entry['metrics']['mid_fraction']:.6f}"

        print(
            f"{frame_path.name} {selected_model} {coarse_model} "
            f"{infer_ms:.2f} {coarse_infer_ms:.2f} {replace_mid} {remove_mid}"
        )

    randomizer = random.Random(args.seed)
    frame_names = [path.name for path in frame_paths]
    shuffled_frames = frame_names[:]
    randomizer.shuffle(shuffled_frames)
    train_count = int(round(len(shuffled_frames) * max(0.0, min(1.0, args.train_ratio))))
    if len(shuffled_frames) > 1:
        train_count = max(1, min(len(shuffled_frames) - 1, train_count))
    train_frames = set(shuffled_frames[:train_count])
    val_frames = set(shuffled_frames[train_count:])

    train_entries: list[dict[str, object]] = []
    val_entries: list[dict[str, object]] = []
    for variant in args.variants:
        for entry in variant_entries[variant]:
            target = train_entries if entry["source_frame"] in train_frames else val_entries
            target.append(entry)

    (out_root / "train.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in train_entries),
        encoding="utf-8",
    )
    (out_root / "val.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in val_entries),
        encoding="utf-8",
    )

    summary = {
        "dataset": str(dataset_dir),
        "bundle": str(out_root),
        "preset": args.preset,
        "coarse_preset": args.coarse_preset,
        "models_priority": list(args.models),
        "coarse_model": args.coarse_model,
        "teacher_gpu_index": args.teacher_gpu_index,
        "coarse_gpu_index": args.coarse_gpu_index,
        "variants": list(args.variants),
        "source_manifest": source_manifest,
        "splits": {
            "train_frames": sorted(train_frames),
            "val_frames": sorted(val_frames),
            "train_entries": len(train_entries),
            "val_entries": len(val_entries),
        },
        "model_usage": dict(model_counter),
        "coarse_usage": dict(coarse_counter),
        "entries_by_variant": {
            variant: {
                "count": len(variant_entries[variant]),
                "mid_fraction_mean": sum(item["metrics"]["mid_fraction"] for item in variant_entries[variant]) / max(1, len(variant_entries[variant])),
                "low_mid_fraction_mean": sum(item["metrics"]["low_mid_fraction"] for item in variant_entries[variant]) / max(1, len(variant_entries[variant])),
                "coarse_mid_fraction_mean": sum(item["coarse_metrics"]["mid_fraction"] for item in variant_entries[variant]) / max(1, len(variant_entries[variant])),
                "coarse_low_mid_fraction_mean": sum(item["coarse_metrics"]["low_mid_fraction"] for item in variant_entries[variant]) / max(1, len(variant_entries[variant])),
            }
            for variant in args.variants
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(out_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a patch-based edge-delta matte refiner from a prepared bundle."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Bundle directory from prepare_matting_training_bundle.py")
    parser.add_argument("--variant", default="replace", choices=["replace", "remove"])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=192, help="Square patch size for training and inference")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel width for the patch refiner")
    parser.add_argument("--samples-per-image", type=int, default=6, help="Number of transition-band patches per frame")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device index when training on GPU")
    parser.add_argument("--install", action="store_true", help="Copy the exported ONNX into models/edge_refiner_<variant>.onnx")
    parser.add_argument("--out", default="artifacts/training_runs", help="Output root directory")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) * (1.0 / 255.0)


def _transition_band(coarse: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    coarse_band = (coarse > 0.02) & (coarse < 0.98)
    trimap_band = (trimap > 0.25) & (trimap < 0.75)
    band = coarse_band | trimap_band
    if band.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        band = cv2.dilate(band.astype(np.uint8), kernel, iterations=1) > 0
    return band


def _trimap_unknown(trimap: torch.Tensor) -> torch.Tensor:
    return ((trimap > 0.25) & (trimap < 0.75)).float()


def _crop_patch(
    rgb: np.ndarray,
    coarse: np.ndarray,
    target: np.ndarray,
    trimap: np.ndarray,
    center_x: int,
    center_y: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = patch_size // 2
    height, width = coarse.shape[:2]
    x0 = max(0, min(width - patch_size, center_x - half))
    y0 = max(0, min(height - patch_size, center_y - half))
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    rgb_patch = rgb[y0:y1, x0:x1]
    coarse_patch = coarse[y0:y1, x0:x1]
    target_patch = target[y0:y1, x0:x1]
    trimap_patch = trimap[y0:y1, x0:x1]

    if rgb_patch.shape[0] != patch_size or rgb_patch.shape[1] != patch_size:
        pad_h = patch_size - rgb_patch.shape[0]
        pad_w = patch_size - rgb_patch.shape[1]
        rgb_patch = cv2.copyMakeBorder(rgb_patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        coarse_patch = cv2.copyMakeBorder(coarse_patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        target_patch = cv2.copyMakeBorder(target_patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        trimap_patch = cv2.copyMakeBorder(trimap_patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    return rgb_patch, coarse_patch, target_patch, trimap_patch


@dataclass
class FrameSample:
    image: Path
    coarse_alpha: Path
    mask: Path
    trimap: Path
    sample_id: str


class EdgePatchDataset(Dataset):
    def __init__(
        self,
        bundle_dir: Path,
        split_file: str,
        variant: str,
        patch_size: int,
        samples_per_image: int,
        augment: bool,
        seed: int,
    ):
        self.bundle_dir = bundle_dir
        self.variant = variant
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image
        self.augment = augment
        self.random = random.Random(seed)
        self.frames: list[FrameSample] = []

        with (bundle_dir / split_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("variant") != variant:
                    continue
                self.frames.append(
                    FrameSample(
                        image=bundle_dir / item["image"],
                        coarse_alpha=bundle_dir / item["coarse_alpha"],
                        mask=bundle_dir / item["mask"],
                        trimap=bundle_dir / item["trimap"],
                        sample_id=item["id"],
                    )
                )
        if not self.frames:
            raise RuntimeError(f"no {variant} entries found in {bundle_dir / split_file}")

    def __len__(self) -> int:
        return len(self.frames) * self.samples_per_image

    def __getitem__(self, index: int):
        frame = self.frames[index % len(self.frames)]
        rgb = _load_rgb(frame.image)
        coarse = _load_mask(frame.coarse_alpha)
        target = _load_mask(frame.mask)
        trimap = _load_mask(frame.trimap)
        band = _transition_band(coarse, trimap)

        ys, xs = np.where(band)
        if ys.size:
            choice = self.random.randrange(ys.size)
            center_y = int(ys[choice])
            center_x = int(xs[choice])
        else:
            height, width = coarse.shape[:2]
            center_y = height // 2
            center_x = width // 2

        rgb_patch, coarse_patch, target_patch, trimap_patch = _crop_patch(
            rgb, coarse, target, trimap, center_x, center_y, self.patch_size
        )

        if self.augment and self.random.random() < 0.5:
            rgb_patch = np.ascontiguousarray(rgb_patch[:, ::-1])
            coarse_patch = np.ascontiguousarray(coarse_patch[:, ::-1])
            target_patch = np.ascontiguousarray(target_patch[:, ::-1])
            trimap_patch = np.ascontiguousarray(trimap_patch[:, ::-1])

        if self.augment and self.random.random() < 0.15:
            rgb_patch = np.clip(rgb_patch.astype(np.float32) * self.random.uniform(0.92, 1.08), 0, 255).astype(np.uint8)

        rgb_t = torch.from_numpy(rgb_patch.transpose(2, 0, 1)).float() * (1.0 / 255.0)
        coarse_t = torch.from_numpy(coarse_patch[None, ...]).float()
        target_t = torch.from_numpy(target_patch[None, ...]).float()
        trimap_t = torch.from_numpy((trimap_patch[None, ...]).astype(np.float32) * (1.0 / 255.0)).float()
        band_t = torch.from_numpy(_transition_band(coarse_patch, trimap_patch).astype(np.float32)[None, ...])
        return torch.cat([rgb_t, coarse_t, trimap_t, band_t], dim=0), coarse_t, target_t, trimap_t, band_t


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PatchEdgeRefiner(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(6, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        coarse = x[:, 3:4]
        trimap = x[:, 4:5]
        band = x[:, 5:6]
        x1 = self.enc1(x)
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.enc3(F.avg_pool2d(x2, 2))
        y2 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        unknown = torch.clamp(1.0 - torch.abs(trimap - 0.5) * 2.0, 0.0, 1.0)
        refine_gate = torch.clamp(torch.maximum(band, unknown), 0.0, 1.0)
        delta = 0.08 * torch.tanh(self.out(y1)) * refine_gate
        return torch.clamp(coarse + delta, 0.0, 1.0)


def _gradient_magnitude(t: torch.Tensor) -> torch.Tensor:
    gx = t[:, :, :, 1:] - t[:, :, :, :-1]
    gy = t[:, :, 1:, :] - t[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _compute_loss(
    pred: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    trimap: torch.Tensor,
    band: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    delta_target = target - coarse
    delta_pred = pred - coarse
    unknown = _trimap_unknown(trimap)
    focus = torch.clamp(torch.maximum(band, unknown), 0.0, 1.0)
    outside = 1.0 - focus
    weights = 1.0 + focus * 5.0
    delta_loss = (weights * (delta_pred - delta_target).abs()).mean()
    alpha_loss = (weights * (pred - target).abs()).mean()
    edge_loss = (focus * (_gradient_magnitude(pred) - _gradient_magnitude(target)).abs()).mean()
    identity_loss = (outside * (pred - coarse).abs()).mean()
    loss = alpha_loss + 0.6 * delta_loss + 0.2 * edge_loss + 0.8 * identity_loss
    return loss, {
        "delta": float(delta_loss.detach()),
        "edge": float(edge_loss.detach()),
        "alpha": float(alpha_loss.detach()),
        "identity": float(identity_loss.detach()),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_delta = 0.0
    total_batches = 0

    for inputs, coarse, target, trimap, band in loader:
        inputs = inputs.to(device, non_blocking=True)
        coarse = coarse.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        trimap = trimap.to(device, non_blocking=True)
        band = band.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(inputs)
            loss, parts = _compute_loss(pred, coarse, target, trimap, band)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach())
        total_delta += parts["delta"]
        total_batches += 1

    denom = max(1, total_batches)
    return {"loss": total_loss / denom, "delta": total_delta / denom}


def _export_onnx(model: nn.Module, out_path: Path, patch_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.randn(1, 6, patch_size, patch_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["alpha"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "alpha": {2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )


def main() -> int:
    args = _parse_args()
    _seed_everything(args.seed)

    bundle_dir = Path(args.bundle).resolve()
    run_dir = Path(args.out).resolve() / bundle_dir.name / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device_index}")

    train_ds = EdgePatchDataset(
        bundle_dir, "train.jsonl", args.variant, args.patch_size, args.samples_per_image, True, args.seed
    )
    val_ds = EdgePatchDataset(
        bundle_dir, "val.jsonl", args.variant, args.patch_size, max(2, args.samples_per_image // 2), False, args.seed + 13
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.workers)

    model = PatchEdgeRefiner(base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val = math.inf
    history: list[dict[str, float]] = []
    best_ckpt = run_dir / "best.pt"

    print("epoch train_loss val_loss train_delta val_delta")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_delta": train_metrics["delta"],
            "val_delta": val_metrics["delta"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"{epoch} "
            f"{record['train_loss']:.6f} "
            f"{record['val_loss']:.6f} "
            f"{record['train_delta']:.6f} "
            f"{record['val_delta']:.6f}"
        )

        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": args.variant,
                    "patch_size": args.patch_size,
                    "base_channels": args.base_channels,
                    "history": history,
                },
                best_ckpt,
            )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    onnx_path = run_dir / f"edge_refiner_{args.variant}.onnx"
    _export_onnx(model, onnx_path, args.patch_size, device)
    metadata = {
        "variant": args.variant,
        "size": args.patch_size,
        "patch_size": args.patch_size,
        "base_channels": args.base_channels,
        "best_val_loss": best_val,
        "model_kind": "patch_edge_delta",
        "input_channels": 6,
    }
    metadata_path = run_dir / f"edge_refiner_{args.variant}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    installed_path = None
    installed_metadata_path = None
    if args.install:
        models_dir = Path(__file__).resolve().parents[1] / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        installed_path = models_dir / f"edge_refiner_{args.variant}.onnx"
        installed_metadata_path = models_dir / f"edge_refiner_{args.variant}.json"
        shutil.copy2(onnx_path, installed_path)
        shutil.copy2(metadata_path, installed_metadata_path)

    summary = {
        "bundle": str(bundle_dir),
        "variant": args.variant,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patch_size": args.patch_size,
        "base_channels": args.base_channels,
        "samples_per_image": args.samples_per_image,
        "device_index": args.device_index if device.type == "cuda" else None,
        "best_val_loss": best_val,
        "history": history,
        "checkpoint": str(best_ckpt),
        "onnx": str(onnx_path),
        "onnx_metadata": str(metadata_path),
        "installed_onnx": str(installed_path) if installed_path is not None else None,
        "installed_onnx_metadata": str(installed_metadata_path) if installed_metadata_path is not None else None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(run_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

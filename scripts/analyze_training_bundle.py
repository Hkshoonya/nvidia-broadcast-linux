#!/usr/bin/env python3
"""Analyze coarse-vs-teacher gaps inside a prepared training bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Training bundle directory")
    parser.add_argument("--variant", default="replace", choices=["replace", "remove"])
    parser.add_argument("--top-k", type=int, default=8, help="Number of worst examples to print")
    parser.add_argument("--out", default="", help="Optional path for JSON output")
    return parser.parse_args()


def _load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) * (1.0 / 255.0)


def _gradient(alpha: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(alpha)
    gy = np.zeros_like(alpha)
    gx[:, :-1] = np.abs(alpha[:, 1:] - alpha[:, :-1])
    gy[:-1, :] = np.abs(alpha[1:, :] - alpha[:-1, :])
    return np.sqrt(gx * gx + gy * gy)


def _load_entries(bundle_dir: Path, variant: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for split_name in ("train.jsonl", "val.jsonl"):
        path = bundle_dir / split_name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("variant") == variant:
                    item["split"] = split_name.removesuffix(".jsonl")
                    entries.append(item)
    if not entries:
        raise RuntimeError(f"no {variant} entries found in {bundle_dir}")
    return entries


def main() -> int:
    args = _parse_args()
    bundle_dir = Path(args.bundle).resolve()
    entries = _load_entries(bundle_dir, args.variant)

    results: list[dict[str, object]] = []
    for item in entries:
        coarse = _load_mask(bundle_dir / item["coarse_alpha"])
        teacher = _load_mask(bundle_dir / item["mask"])
        abs_err = np.abs(coarse - teacher)
        grad_teacher = _gradient(teacher)
        edge_band = grad_teacher > 0.03
        if edge_band.any():
            edge_abs = float(abs_err[edge_band].mean())
        else:
            edge_abs = 0.0

        results.append(
            {
                "id": item["id"],
                "split": item["split"],
                "source_frame": item["source_frame"],
                "coarse_model": item.get("coarse_model"),
                "teacher_model": item.get("selected_model"),
                "global_l1": float(abs_err.mean()),
                "edge_l1": edge_abs,
                "teacher_mid_fraction": float(item["metrics"]["mid_fraction"]),
                "coarse_mid_fraction": float(item["coarse_metrics"]["mid_fraction"]),
                "teacher_low_mid_fraction": float(item["metrics"]["low_mid_fraction"]),
                "coarse_low_mid_fraction": float(item["coarse_metrics"]["low_mid_fraction"]),
                "image": item["image"],
                "coarse_alpha": item["coarse_alpha"],
                "mask": item["mask"],
                "trimap": item["trimap"],
            }
        )

    results.sort(key=lambda row: (row["edge_l1"], row["global_l1"]), reverse=True)
    summary = {
        "bundle": str(bundle_dir),
        "variant": args.variant,
        "count": len(results),
        "global_l1_mean": float(sum(row["global_l1"] for row in results) / len(results)),
        "edge_l1_mean": float(sum(row["edge_l1"] for row in results) / len(results)),
        "worst": results[: args.top_k],
    }

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = bundle_dir / f"analysis_{args.variant}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(out_path)
    print("rank edge_l1 global_l1 frame split")
    for idx, row in enumerate(summary["worst"], start=1):
        print(
            f"{idx} "
            f"{row['edge_l1']:.6f} "
            f"{row['global_l1']:.6f} "
            f"{row['source_frame']} "
            f"{row['split']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

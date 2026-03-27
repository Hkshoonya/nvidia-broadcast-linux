#!/usr/bin/env python3
"""Run the background-quality harness across a captured dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
EVAL_SCRIPT = REPO_ROOT / "scripts/eval_background_quality.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Directory containing captured PNG frames")
    parser.add_argument("--out", default="artifacts/dataset_eval", help="Output root directory")
    parser.add_argument("--effect-mode", default="replace", choices=["replace", "remove", "blur"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["doczeus", "cuda_max", "cuda_balanced", "zeus", "killer"],
        help="Preset modes to evaluate",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per frame")
    parser.add_argument("--repeat", type=int, default=2, help="Measured runs per frame")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = _parse_args()
    dataset_dir = Path(args.dataset).resolve()
    frame_paths = sorted(p for p in dataset_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"no frame_*.png files found in {dataset_dir}")

    out_root = Path(args.out).resolve() / args.effect_mode / dataset_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "dataset": str(dataset_dir),
        "effect_mode": args.effect_mode,
        "frames": [p.name for p in frame_paths],
        "modes": {},
    }

    for mode in args.modes:
        mode_root = out_root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        frame_results: list[dict[str, object]] = []
        infer_values: list[float] = []
        composite_values: list[float] = []
        edge_mid_values: list[float] = []
        edge_low_mid_values: list[float] = []

        for frame_path in frame_paths:
            frame_out = mode_root / frame_path.stem
            cmd = [
                str(PYTHON), str(EVAL_SCRIPT),
                "--input", str(frame_path),
                "--preset", mode,
                "--mode", args.effect_mode,
                "--out", str(frame_out),
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            item: dict[str, object] = {
                "frame": frame_path.name,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-1200:],
                "stderr_tail": proc.stderr[-1200:],
            }

            metrics_path = frame_out / "metrics.json"
            if metrics_path.exists():
                payload = json.loads(metrics_path.read_text())
                item["payload"] = payload
                timings = payload.get("timings_ms", {})
                metrics = payload.get("metrics", {})
                edge = (
                    metrics.get("final_matte")
                    or metrics.get("greenscreen_matte")
                    or metrics.get("edge_aware_matte")
                    or metrics.get("replacement_matte")
                    or metrics.get("alpha", {})
                )
                infer_values.append(float(timings.get("infer_ms", 0.0)))
                composite_values.append(float(timings.get("composite_ms", 0.0)))
                edge_mid_values.append(float(edge.get("mid_fraction", 0.0)))
                edge_low_mid_values.append(float(edge.get("low_mid_fraction", 0.0)))

            frame_results.append(item)

        aggregate = {
            "infer_ms_mean": _mean(infer_values),
            "composite_ms_mean": _mean(composite_values),
            "edge_mid_fraction_mean": _mean(edge_mid_values),
            "edge_low_mid_fraction_mean": _mean(edge_low_mid_values),
        }
        summary["modes"][mode] = {
            "aggregate": aggregate,
            "frames": frame_results,
        }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    print("mode infer_ms_mean composite_ms_mean edge_mid_mean edge_low_mid_mean")
    for mode in args.modes:
        aggregate = summary["modes"][mode]["aggregate"]
        print(
            f"{mode} "
            f"{aggregate['infer_ms_mean']:.2f} "
            f"{aggregate['composite_ms_mean']:.2f} "
            f"{aggregate['edge_mid_fraction_mean']:.6f} "
            f"{aggregate['edge_low_mid_fraction_mean']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

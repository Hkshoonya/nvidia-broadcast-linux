#!/usr/bin/env python3
"""Compare segmentation models on a single saved frame."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
EVAL_SCRIPT = REPO_ROOT / "scripts/eval_background_quality.py"
MODELS = ["rvm", "isnet", "birefnet"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an input image/frame")
    parser.add_argument("--preset", default="doczeus", help="App-style engine preset to keep fixed")
    parser.add_argument("--effect-mode", default="replace", choices=["replace", "remove", "blur"])
    parser.add_argument("--out", default="artifacts/model_sweep", help="Output directory")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "input": str(Path(args.input).resolve()),
        "preset": args.preset,
        "effect_mode": args.effect_mode,
        "models": {},
    }

    print("model infer_ms composite_ms edge_mid edge_low_mid")
    for model in MODELS:
        out_dir = out_root / model
        cmd = [
            str(PYTHON), str(EVAL_SCRIPT),
            "--input", args.input,
            "--preset", args.preset,
            "--model", model,
            "--mode", args.effect_mode,
            "--out", str(out_dir),
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        item: dict[str, object] = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-1500:],
        }
        metrics_path = out_dir / "metrics.json"
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
            print(
                f"{model} "
                f"{float(timings.get('infer_ms', 0.0)):.2f} "
                f"{float(timings.get('composite_ms', 0.0)):.2f} "
                f"{float(edge.get('mid_fraction', 0.0)):.6f} "
                f"{float(edge.get('low_mid_fraction', 0.0)):.6f}"
            )
        summary["models"][model] = item

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

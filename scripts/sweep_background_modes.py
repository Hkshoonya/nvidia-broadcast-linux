#!/usr/bin/env python3
"""Compare multiple app-style background modes on the same saved frame."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
EVAL_SCRIPT = REPO_ROOT / "scripts/eval_background_quality.py"


DEFAULT_MODES = [
    "cpu_quality",
    "cuda_balanced",
    "cuda_max",
    "doczeus",
    "zeus",
    "killer",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an input image/frame")
    parser.add_argument("--out", default="artifacts/mode_sweep", help="Output directory")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="Preset names to evaluate")
    parser.add_argument("--effect-mode", default="replace", choices=["replace", "remove", "blur"])
    parser.add_argument("--warmup", type=int, default=1, help="Warmup inferences per mode")
    parser.add_argument("--repeat", type=int, default=3, help="Measured inferences per mode")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"input": str(Path(args.input).resolve()), "modes": {}}
    table: list[tuple[str, float, float, float, float]] = []

    for mode in args.modes:
        out_dir = out_root / mode
        cmd = [
            str(PYTHON), str(EVAL_SCRIPT),
            "--input", args.input,
            "--preset", mode,
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
            table.append((
                mode,
                float(timings.get("infer_ms", -1.0)),
                float(timings.get("composite_ms", -1.0)),
                float(edge.get("mid_fraction", -1.0)),
                float(edge.get("low_mid_fraction", -1.0)),
            ))
        summary["modes"][mode] = item

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(summary_path)
    print("mode infer_ms composite_ms edge_mid edge_low_mid")
    for mode, infer_ms, composite_ms, edge_mid, edge_low_mid in table:
        print(f"{mode} {infer_ms:.2f} {composite_ms:.2f} {edge_mid:.6f} {edge_low_mid:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Capture a small local camera dataset for background-quality evaluation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0", help="Video device to capture from")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height")
    parser.add_argument("--count", type=int, default=8, help="Number of frames to capture")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between captures")
    parser.add_argument("--warmup-frames", type=int, default=12, help="Frames to discard before capture")
    parser.add_argument("--label", default="", help="Optional short label for the capture session")
    parser.add_argument("--note", default="", help="Optional free-form note stored in the manifest")
    parser.add_argument("--out", default="artifacts/quality_dataset", help="Output root directory")
    return parser.parse_args()


def _save_bgr(path: Path, frame_bgr) -> None:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


def main() -> int:
    args = _parse_args()
    session_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.label:
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in args.label).strip("-_")
        if safe_label:
            session_name = f"{session_name}-{safe_label}"
    out_dir = Path(args.out).resolve() / session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera device {args.camera}")

    manifest: dict[str, object] = {
        "label": args.label,
        "note": args.note,
        "camera": args.camera,
        "requested_width": args.width,
        "requested_height": args.height,
        "count": args.count,
        "interval_s": args.interval,
        "warmup_frames": args.warmup_frames,
        "frames": [],
    }

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        for _ in range(max(0, args.warmup_frames)):
            ok, _frame = cap.read()
            if not ok:
                raise RuntimeError("failed during camera warmup")

        for idx in range(args.count):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {idx + 1}")

            timestamp = datetime.now().isoformat(timespec="seconds")
            filename = f"frame_{idx + 1:03d}.png"
            _save_bgr(out_dir / filename, frame)
            manifest["frames"].append({
                "index": idx + 1,
                "filename": filename,
                "captured_at": timestamp,
                "actual_width": int(frame.shape[1]),
                "actual_height": int(frame.shape[0]),
            })
            print(f"captured {filename} at {timestamp}", flush=True)

            if idx + 1 < args.count and args.interval > 0:
                time.sleep(args.interval)
    finally:
        cap.release()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

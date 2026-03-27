#!/usr/bin/env python3
"""Combine multiple captured quality datasets into one training-ready session."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset directories to combine")
    parser.add_argument("--label", default="combined", help="Label suffix for the combined dataset directory")
    parser.add_argument("--out", default="artifacts/quality_dataset", help="Output root directory")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_dirs = [Path(item).resolve() for item in args.datasets]
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            raise RuntimeError(f"dataset directory not found: {dataset_dir}")

    session_name = "__".join(dataset_dir.name for dataset_dir in dataset_dirs)
    if args.label:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in args.label).strip("-_")
        if safe:
            session_name = f"{session_name}--{safe}"

    out_dir = Path(args.out).resolve() / session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "label": args.label,
        "datasets": [str(dataset_dir) for dataset_dir in dataset_dirs],
        "frames": [],
    }

    next_index = 1
    for dataset_dir in dataset_dirs:
        source_manifest_path = dataset_dir / "manifest.json"
        source_manifest = None
        if source_manifest_path.exists():
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

        frame_paths = sorted(dataset_dir.glob("frame_*.png"))
        if not frame_paths:
            continue

        source_tag = dataset_dir.name
        for frame_path in frame_paths:
            filename = f"frame_{next_index:03d}.png"
            target_path = out_dir / filename
            shutil.copy2(frame_path, target_path)
            manifest["frames"].append(
                {
                    "index": next_index,
                    "filename": filename,
                    "source_dataset": source_tag,
                    "source_frame": frame_path.name,
                    "source_manifest_label": source_manifest.get("label") if isinstance(source_manifest, dict) else None,
                }
            )
            next_index += 1

    manifest["count"] = len(manifest["frames"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

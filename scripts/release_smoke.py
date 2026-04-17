#!/usr/bin/env python3
"""Run the local release smoke checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DIST_DIR = REPO_ROOT / "dist" / "release-smoke"
REQUIRED_WHEEL_PATHS = [
    "nvbroadcast/ui/style.css",
    ".data/data/share/applications/com.doczeus.NVBroadcast.desktop",
    ".data/data/share/metainfo/com.doczeus.NVBroadcast.metainfo.xml",
    ".data/data/share/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg",
]


def _run(cmd: list[str], label: str) -> None:
    print(f"[release-smoke] {label}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _wheel_members(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as zf:
        return zf.namelist()


def _check_wheel_contents(wheel_path: Path) -> dict[str, str]:
    members = _wheel_members(wheel_path)
    missing = {}
    for required in REQUIRED_WHEEL_PATHS:
        if not any(required in member for member in members):
            missing[required] = "missing"
    if missing:
        raise RuntimeError(f"wheel missing required assets: {json.dumps(missing, indent=2)}")
    return {
        "wheel": str(wheel_path),
        "asset_checks": "passed",
    }


def main() -> int:
    if not PYTHON.exists():
        raise RuntimeError(f"expected venv python at {PYTHON}")

    _run(["python3", "-m", "compileall", "src", "scripts", "tests"], "compileall")
    _run(
        [
            str(PYTHON), "-m", "unittest", "-v",
            "tests.test_updates",
            "tests.test_audio_devices",
            "tests.test_background_overlay",
            "tests.test_training_bundle",
        ],
        "unit tests",
    )

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    for old_wheel in DIST_DIR.glob("*.whl"):
        old_wheel.unlink()
    _run([str(PYTHON), "-m", "pip", "wheel", "--no-deps", ".", "-w", str(DIST_DIR)], "build wheel")

    wheels = sorted(DIST_DIR.glob("*.whl"))
    if not wheels:
        raise RuntimeError("wheel build produced no files")

    payload = _check_wheel_contents(wheels[-1])
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

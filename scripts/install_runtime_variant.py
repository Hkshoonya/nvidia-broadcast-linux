#!/usr/bin/env python3
"""Install one source runtime variant, then validate its ownership."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run_pip(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", *arguments],
        check=True,
    )


def install(project: Path, variant: str, meeting_backends: str) -> None:
    extras = [variant]
    if meeting_backends != "none":
        extras.append("meeting-support")
    if meeting_backends == "all":
        extras.append("meeting")

    selected_extras = ",".join(extras)
    run_pip("install", "--upgrade", f"{project}[{selected_extras}]")
    if meeting_backends != "none":
        # Keep backend installation outside dependency resolution so its
        # onnxruntime requirement cannot replace the selected runtime owner.
        run_pip("install", "--no-deps", "faster-whisper")
    subprocess.run(
        [sys.executable, "-m", "nvbroadcast.runtime", "--variant", variant],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--meeting-backends",
        choices=("none", "faster", "all"),
        default="none",
        help="install no backend, faster-whisper only, or both supported backends",
    )
    arguments = parser.parse_args()
    install(
        arguments.project.resolve(), arguments.variant, arguments.meeting_backends
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

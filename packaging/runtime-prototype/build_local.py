#!/usr/bin/env python3
"""Build the three local wheels from pinned inputs, in a network-disabled image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile

import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--image", required=True, help="inspected local builder image ID")
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error("build as an ordinary user")
    if not args.image.startswith("sha256:") or len(args.image) != 71:
        parser.error("use an inspected immutable local image ID")
    os.umask(0o022)
    work = args.directory.resolve(strict=True)
    pins = json.loads((prepare.HERE / "inputs.json").read_text())
    for name in ("builder-python", "app-source", "wheels"):
        if (work / name).exists():
            parser.error(f"{name} exists; use a fresh work directory for a repeat build")
    for item in [pins["python"], pins["uv"], *pins["bindings"]]:
        prepare.fetch(work / "inputs", item, offline=True)
    prepare.fetch_lock(prepare.HERE / "pylock.build.toml", work / "bootstrap", offline=True)
    prepare.unpack_python(work / "inputs" / prepare.filename(pins["python"]),
                          pins["python"], work / "builder-python")
    repo = prepare.HERE.parents[1]
    source_archive = work / "app-source.tar"
    with source_archive.open("wb") as stream:
        subprocess.run(["git", "archive", "--format=tar", pins["application"]["revision"],
                        "src", "data", "pyproject.toml", "LICENSE", "NOTICE", "CONTRIBUTORS.md", "README.md"],
                       cwd=repo, stdout=stream, check=True)
    with tarfile.open(source_archive) as archive:
        archive.extractall(work / "app-source", filter=prepare.safe_member)
    source_archive.unlink()
    with (work / "binding-build.log").open("w") as log:
        subprocess.run([
            "docker", "run", "--rm", "--init", "--pull=never", "--network=none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", f"SOURCE_DATE_EPOCH={pins['application']['source_date_epoch']}",
            "-v", f"{work / 'builder-python/runtime'}:/opt/nvbroadcast/.venv",
            "-v", f"{work}:/work", "-v", f"{prepare.HERE}:/prototype:ro",
            args.image, "bash", "/prototype/build-bindings.sh",
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
    print((work / "built-wheel-sha256.txt").read_text(), end="")


if __name__ == "__main__":
    main()

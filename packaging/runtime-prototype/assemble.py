#!/usr/bin/env python3
"""Assemble a non-release payload with no online installation/resolution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import prepare


def inventory(root: Path) -> dict:
    """Describe content and permissions, independent of build timestamps."""
    result = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        entry = {"mode": stat.S_IMODE(mode)}
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"runtime symlink escapes payload: {path}")
            entry["symlink"] = os.readlink(path)
        elif path.is_file():
            entry.update(sha256=prepare.digest(path), size=path.stat().st_size)
        elif not path.is_dir():
            raise ValueError(f"unexpected runtime entry: {path}")
        result[str(path.relative_to(root))] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="prepared wheel/input cache")
    parser.add_argument("--output", type=Path, required=True, help="new destination, never overwritten")
    parser.add_argument("--image", required=True, help="local builder image ID, sha256:...")
    parser.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error("assemble as an ordinary user")
    if not args.image.startswith("sha256:") or len(args.image) != 71:
        parser.error("use an inspected immutable local image ID")
    os.umask(0o022)
    work = args.directory.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output exists; use a new directory")
    lock = prepare.HERE / f"pylock.linux-x86_64-cp313-{args.variant}.toml"
    prepare.fetch_lock(lock, work / "application", work / "wheels", offline=True)
    pins = json.loads((prepare.HERE / "inputs.json").read_text())
    prepare.unpack_python(work / "inputs" / prepare.filename(pins["python"]), pins["python"], output)
    runtime = output / "runtime"
    prefix = "/opt/nvbroadcast/.venv"
    provenance = runtime / "share/nvbroadcast-runtime-provenance"
    with (output / "assembly.log").open("w") as log:
        subprocess.run([
            "docker", "run", "--rm", "--init", "--network=none", "--pull=never",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{runtime}:{prefix}", "-v", f"{work / 'application'}:/inputs:ro",
            args.image, f"{prefix}/bin/python", "-I", "-B", "-m", "pip", "--isolated",
            "install", "--no-index", "--no-cache-dir", "--no-deps", "--no-compile",
            "--require-hashes", "--only-binary=:all:", "--find-links=/inputs/wheels",
            "-r", "/inputs/requirements.txt",
            f"--report={prefix}/share/nvbroadcast-runtime-provenance/install-report.json",
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copyfile(lock, provenance / lock.name)
    shutil.copyfile(prepare.HERE / "inputs.json", provenance / "inputs.json")
    (provenance / "selection.json").write_text(json.dumps({
        "target": f"linux-x86_64-cp313-{args.variant}",
        "lock_sha256": prepare.digest(lock),
        "inputs_sha256": prepare.digest(prepare.HERE / "inputs.json"),
    }, indent=2, sort_keys=True) + "\n")
    manifest = inventory(runtime)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runtime": str(runtime), "manifest_entries": len(manifest),
                      "file_bytes": sum(p.get("size", 0) for p in manifest.values()),
                      "manifest_sha256": prepare.digest(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build native test images, then probe one read-only private runtime offline.

This is a feasibility harness, not a support declaration or package installer.
Run from any directory. Docker daemon access and Python 3.12+ are required.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from assemble import inventory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFIX = "/opt/nvbroadcast/.venv"


def source_hashes(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*.py"))}


def run_cell(cell: dict, directory: Path, build: bool, runtime: Path,
             variant: str = "cpu", gpu: int | None = None) -> dict:
    tag = f"nvb-private-runtime-test:{cell['name']}-20261005"
    result = {**cell, "tag": tag, "variant": variant, "gpu": gpu}
    if build:
        with (directory / f"image-{cell['name']}.log").open("w") as log:
            completed = subprocess.run(
                ["docker", "build", "--build-arg", f"BASE_IMAGE={cell['base']}",
                 "-f", str(HERE / f"Dockerfile.runtime.{cell['family']}"),
                 "-t", tag, str(HERE)], stdout=log, stderr=subprocess.STDOUT,
            )
        result["build_exit"] = completed.returncode
        if completed.returncode:
            return {**result, "status": "image-build-failed"}
    inspected = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", tag],
                               capture_output=True, text=True)
    if inspected.returncode:
        return {**result, "status": "image-missing"}
    result["image_id"] = inspected.stdout.strip()
    inventory_command = ["dpkg-query", "-W"] if cell["family"] == "deb" else ["rpm", "-qa"]
    inventory = subprocess.run(
        ["docker", "run", "--rm", "--network=none", result["image_id"], *inventory_command],
        capture_output=True, text=True, check=True,
    ).stdout
    (directory / f"packages-{cell['name']}.txt").write_text(inventory)

    name = f"nvb-runtime-probe-{uuid.uuid4().hex}"
    devices = ["--gpus", f"device={gpu}", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"] if gpu is not None else []
    command = ["docker", "run", "--rm", "--init", "--name", name, "--network=none",
               "--user", "1000:1000", "--read-only",
               "--tmpfs", "/tmp:rw,mode=1777",
               "-e", "XDG_CACHE_HOME=/tmp/cache", "-e", "XDG_CONFIG_HOME=/tmp/config",
               # -I must ignore these deliberate host-Python contamination paths.
               "-e", "PYTHONPATH=/usr/lib/python3/dist-packages", "-e", "PYTHONHOME=/usr",
               "-e", "OPENBLAS_NUM_THREADS=2",
               "-v", f"{runtime}:{PREFIX}:ro", "-v", f"{HERE}:/prototype:ro", *devices,
               result["image_id"], "bash", "/prototype/desktop_probe.sh",
               f"{PREFIX}/bin/python", "-I", "-B", "-u", "/prototype/probe.py",
               "--runtime", PREFIX, "--python-version",
               json.loads((HERE / "inputs.json").read_text())["python"]["version"], "--window",
               "--variant", variant]
    if variant == "cuda" and gpu is None:
        command += ["--cuda-unavailable"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
        output = completed.stdout + completed.stderr
        result["probe_exit"] = completed.returncode
        result["status"] = "probe-failed"
        if completed.returncode == 0:
            reports = [line[7:] for line in completed.stdout.splitlines() if line.startswith("RESULT=")]
            if len(reports) == 1:
                result["probe"] = json.loads(reports[0])
                expected = source_hashes(REPO / "src/nvbroadcast")
                result["status"] = ("pass" if expected == result["probe"]["app_python_hashes"]
                                    else "source-mismatch")
            else:
                result["status"] = "missing-probe-report"
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"").decode(errors="replace") + (error.stderr or b"").decode(errors="replace")
        result["status"] = "probe-timeout"
    finally:
        # Killing docker's client on a timeout does not stop its container.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    (directory / f"probe-{cell['name']}.log").write_text(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--build-images", action="store_true")
    parser.add_argument("--cells", nargs="+", help="defaults to every recorded cell, including Rocky 9")
    parser.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, help="expose one NVIDIA GPU for one CUDA matrix cell")
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error("run the probe as an ordinary user, not root")
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.runtime.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if inventory(args.runtime) != manifest:
        parser.error("runtime does not match its assembly manifest")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    matrix = json.loads((HERE / "matrix.json").read_text())
    if args.cells:
        unknown = set(args.cells) - {c["name"] for c in matrix}
        if unknown:
            parser.error(f"unknown cells: {sorted(unknown)}")
        matrix = [c for c in matrix if c["name"] in args.cells]
    if args.gpu is not None and (args.gpu < 0 or args.variant != "cuda" or len(matrix) != 1):
        parser.error("--gpu requires one selected cell, the CUDA variant, and a nonnegative device")
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_cell, cell, args.directory.resolve(), args.build_images,
                               args.runtime.resolve(strict=True), args.variant, args.gpu) for cell in matrix]
        for future in futures:
            result = future.result()
            result["runtime_manifest_sha256"] = manifest_digest
            results.append(result)
            print(f"{result['name']}: {result['status']}", flush=True)
            (args.directory / "matrix-results.json").write_text(json.dumps(results, indent=2) + "\n")
    unchanged = inventory(args.runtime) == manifest
    (args.directory / "payload-integrity.json").write_text(json.dumps({
        "manifest_sha256": manifest_digest, "unchanged_after_probes": unchanged,
    }, indent=2) + "\n")
    raise SystemExit(0 if unchanged and all(r["status"] == "pass" for r in results) else 1)


if __name__ == "__main__":
    main()

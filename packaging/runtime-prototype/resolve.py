#!/usr/bin/env python3
"""Explicitly refresh a runtime lock from canonical app metadata; uses the network.

This is a maintainer operation, never part of assembly or installation. Review
new artifacts and hashes before replacing the checked-in lock.
"""

from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import prepare


def cuda_metadata(work: Path, version: str) -> dict:
    """Remove only faster-whisper's CPU ORT edge from verified wheel metadata.

    The wheel itself stays byte-identical. All other requirements and markers
    remain inputs to the resolver, including future additions to this version.
    """
    baseline = tomllib.loads((prepare.HERE / "pylock.linux-x86_64-cp313-cpu.toml").read_text())
    package = next(p for p in baseline["packages"] if p["name"] == "faster-whisper")
    if package["version"] != version or len(package["wheels"]) != 1:
        raise ValueError("CPU baseline must pin the same faster-whisper wheel")
    wheel = package["wheels"][0]
    (work / "metadata-inputs").mkdir(exist_ok=True)
    artifact = prepare.fetch(work / "metadata-inputs", {
        "url": wheel["url"], "sha256": wheel["hashes"]["sha256"],
    })
    with zipfile.ZipFile(artifact) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("expected one faster-whisper METADATA")
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    if metadata["Name"] != "faster-whisper" or metadata["Version"] != version:
        raise ValueError("faster-whisper wheel identity mismatch")
    requirements = metadata.get_all("Requires-Dist", [])
    omitted = [r for r in requirements if canonicalize_name(Requirement(r).name) == "onnxruntime"]
    if len(omitted) != 1 or Requirement(omitted[0]).marker:
        raise ValueError("expected one unconditional faster-whisper CPU ORT dependency")
    override = {"name": "faster-whisper", "version": version,
                "requires-dist": [r for r in requirements if r not in omitted],
                "requires-python": metadata["Requires-Python"],
                "provides-extra": metadata.get_all("Provides-Extra", [])}
    (work / "cuda-metadata-override.json").write_text(json.dumps({
        "wheel_sha256": prepare.digest(artifact), "omitted": omitted,
        "original_requires_dist": requirements, "override": override,
    }, indent=2) + "\n")
    return override


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    work = args.directory.resolve(strict=True)
    pins = json.loads((prepare.HERE / "inputs.json").read_text())
    tree = ast.parse((work / "app-source/src/nvbroadcast/runtime/variants.py").read_text())
    version = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "FASTER_WHISPER_VERSION" for t in n.targets))
    requirements = []
    for name, pattern in ((f"nvbroadcast[{args.variant},meeting-support]", "nvbroadcast-*.whl"),
                          ("pycairo", "pycairo-*.whl"), ("PyGObject", "pygobject-*.whl")):
        wheels = list((work / "wheels").glob(pattern))
        if len(wheels) != 1:
            raise ValueError(f"expected one local wheel for {name}")
        requirements.append(f"{name} @ {wheels[0].as_uri()}")
    requirements.append(f"faster-whisper=={version}")
    options = []
    if args.variant == "cuda":
        override = cuda_metadata(work, version)
        config = work / "uv-cuda.toml"
        config.write_text("[pip]\ndependency-metadata = [{ " + ", ".join(
            f"{key} = {json.dumps(value)}" for key, value in override.items()) + " }]\n")
        # Hold shared CPU/CUDA dependencies fixed so this experiment isolates
        # variant ownership, rather than also updating unrelated libraries.
        baseline = tomllib.loads((prepare.HERE / "pylock.linux-x86_64-cp313-cpu.toml").read_text())
        constraints = work / "cuda-shared-constraints.txt"
        constraints.write_text("\n".join(f"{p['name']}=={p['version']}" for p in baseline["packages"]
                                        if p["name"] != "onnxruntime") + "\n")
        options = ["--config-file", str(config), "--constraint", constraints.as_uri()]
    source = work / "application-requirements.in"
    source.write_text("\n".join(requirements) + "\n")
    # Re-extract the verified uv archive so an altered cached executable is not
    # implicitly trusted during a lock update.
    with tempfile.TemporaryDirectory(prefix="resolver-", dir=work) as temporary:
        resolver = Path(temporary) / "verified"
        prepare.unpack_uv(work / "inputs" / prepare.filename(pins["uv"]), pins["uv"], resolver)
        subprocess.run([
            str(resolver / "uv-x86_64-unknown-linux-gnu/uv"), "pip", "compile", str(source),
            "--python", str(work / "builder-python/runtime/bin/python"),
            "--python-version", pins["python"]["version"], "--no-python-downloads",
            "--python-platform", "x86_64-manylinux_2_28", "--only-binary", ":all:",
            "--format", "pylock.toml", "-o", str(args.output), *options,
        ], check=True)
    packages = {p["name"] for p in tomllib.loads(args.output.read_text())["packages"]}
    expected_owner = "onnxruntime-gpu" if args.variant == "cuda" else "onnxruntime"
    if packages & {"onnxruntime", "onnxruntime-gpu"} != {expected_owner}:
        raise ValueError("resolved lock has ambiguous or incorrect ONNX Runtime ownership")
    # PEP 751 local paths should be portable between workspace locations.
    content = args.output.read_text()
    for wheel in (work / "wheels").glob("*.whl"):
        content = content.replace(json.dumps(str(wheel)),
                                  json.dumps(os.path.relpath(wheel, args.output.resolve().parent)))
    # Do not publish machine-specific workspace paths in uv's command comment.
    content = "# Generated by resolve.py using the pinned uv release.\n" + "\n".join(
        line for line in content.splitlines() if not line.startswith("#")
    ) + "\n"
    args.output.write_text(content)


if __name__ == "__main__":
    main()

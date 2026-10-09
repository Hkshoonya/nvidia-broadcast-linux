#!/usr/bin/env python3
"""Validate a package-owned runtime before completing its native transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


def validate(root: Path, manifest: dict, *, owner: int = 0) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("runtime root must be a real directory")
    paths = {str(p.relative_to(root)): p for p in root.rglob("*")}
    if paths.keys() != manifest.keys():
        raise ValueError("runtime file set differs from package manifest")
    for relative, path in paths.items():
        expected = manifest[relative]
        info = path.lstat()
        if info.st_uid != owner or info.st_gid != owner:
            raise ValueError(f"unexpected ownership: {relative}")
        if stat.S_IMODE(info.st_mode) != expected["mode"]:
            raise ValueError(f"unexpected permissions: {relative}")
        if "symlink" in expected:
            if (not path.is_symlink() or os.readlink(path) != expected["symlink"]
                    or not path.resolve().is_relative_to(root.resolve())):
                raise ValueError(f"unexpected link: {relative}")
        elif "sha256" in expected:
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected["size"]:
                raise ValueError(f"unexpected regular file: {relative}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected["sha256"]:
                raise ValueError(f"runtime checksum mismatch: {relative}")
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"unexpected directory: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/usr/lib/nvbroadcast/runtime"))
    parser.add_argument("--manifest", type=Path, default=Path("/usr/lib/nvbroadcast/runtime-manifest.json"))
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        parser.error("validation requires isolated Python with bytecode disabled")
    validate(args.root, json.loads(args.manifest.read_text()))
    from nvbroadcast.runtime.artifact import ArtifactEnvironment
    from nvbroadcast.runtime.variants import detect_runtime_variant, RuntimeVariant
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    environment = ArtifactEnvironment.current()
    if detect_runtime_variant() != RuntimeVariant(args.variant):
        raise ValueError("installed runtime variant does not match package")
    if any(len(values) != 1 for values in environment.installed.values()):
        raise ValueError("duplicate distribution ownership")
    substitutions = {}
    if args.variant == "cuda":
        for distribution in environment.distributions:
            for raw in distribution.requires or ():
                requirement = Requirement(raw)
                if requirement.marker and not requirement.marker.evaluate(environment.markers):
                    continue
                if (canonicalize_name(requirement.name) == "onnxruntime"
                        and canonicalize_name(distribution.metadata["Name"]) != "faster-whisper"):
                    raise ValueError("unreviewed CUDA dependency substitution")
        substitutions["onnxruntime"] = "onnxruntime-gpu"
    problems = environment.dependency_closure_problems(substitutions)
    if problems:
        raise ValueError("incomplete runtime: " + "; ".join(problems))
    print(json.dumps({"status": "pass", "variant": args.variant,
                      "distributions": len(environment.distributions)}))


if __name__ == "__main__":
    main()

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
    root_info = root.stat()
    if (root_info.st_uid != owner or root_info.st_gid != owner
            or stat.S_IMODE(root_info.st_mode) != 0o755):
        raise ValueError("runtime root must have trusted ownership and mode 0755")
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


def validate_dependencies(environment, variant: str) -> None:
    from nvbroadcast.runtime.variants import FASTER_WHISPER_VERSION
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    extras = {variant, "meeting-support"}
    if any(len(values) != 1 for values in environment.installed.values()):
        raise ValueError("duplicate distribution ownership")
    substitutions = {}
    if variant == "cuda":
        for distribution in environment.distributions:
            name = canonicalize_name(distribution.metadata["Name"])
            selected_extras = extras if name == "nvbroadcast" else set(distribution.metadata.get_all("Provides-Extra", []))
            for raw in distribution.requires or ():
                requirement = Requirement(raw)
                if requirement.marker and not any(requirement.marker.evaluate({**environment.markers, "extra": extra})
                                                   for extra in {"", *selected_extras}):
                    continue
                if (canonicalize_name(requirement.name) == "onnxruntime"
                        and name != "faster-whisper"):
                    raise ValueError("unreviewed CUDA dependency substitution")
        substitutions["onnxruntime"] = "onnxruntime-gpu"
    problems = environment.dependency_closure_problems(substitutions, root_extras={"nvbroadcast": extras})
    if environment.installed.get("faster-whisper") != (FASTER_WHISPER_VERSION,):
        problems.append("managed faster-whisper backend is missing or has the wrong version")
    if problems:
        raise ValueError("incomplete runtime: " + "; ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/usr/lib/nvbroadcast/runtime"))
    parser.add_argument("--manifest", type=Path, default=Path("/usr/lib/nvbroadcast/runtime-manifest.json"))
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        parser.error("validation requires isolated Python with bytecode disabled")
    if Path(sys.prefix).resolve() != args.root.resolve() or Path(sys.base_prefix).resolve() != args.root.resolve():
        parser.error("validation must run with the package's private interpreter")
    validate(args.root, json.loads(args.manifest.read_text()))
    from nvbroadcast.runtime.artifact import ArtifactEnvironment
    from nvbroadcast.runtime.variants import detect_runtime_variant, RuntimeVariant
    environment = ArtifactEnvironment.current()
    if detect_runtime_variant() != RuntimeVariant(args.variant):
        raise ValueError("installed runtime variant does not match package")
    validate_dependencies(environment, args.variant)
    print(json.dumps({"status": "pass", "variant": args.variant,
                      "distributions": len(environment.distributions)}))


if __name__ == "__main__":
    main()

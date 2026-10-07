#!/usr/bin/env python3
"""Verify installed bytes, ownership, and private runtime contents in a container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

PREFIX = Path("/usr/lib/nvbroadcast/runtime")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        entry = {"mode": stat.S_IMODE(path.lstat().st_mode)}
        if path.is_symlink():
            assert path.resolve().is_relative_to(root), path
            entry["symlink"] = os.readlink(path)
        elif path.is_file():
            entry.update(sha256=sha(path), size=path.stat().st_size)
        else:
            assert path.is_dir(), path
        assert path.lstat().st_uid == 0 and path.lstat().st_gid == 0, path
        result[str(path.relative_to(root))] = entry
    return result


def verify(family: str, shape: str, revision: int) -> dict:
    manifest = json.loads(Path("/payload-manifest.json").read_text())
    assert inventory(PREFIX) == manifest, "installed runtime differs from verified payload"
    packages = json.loads(Path("/artifacts/packages.json").read_text())
    kinds = {"self"} if shape == "self" else {"app", "runtime"}
    selected = [p for p in packages if p["family"] == family and p["revision"] == revision and p["kind"] in kinds]
    ownership = {}
    for package in selected:
        if family == "deb":
            command = ["dpkg-query", "-L", package["name"]]
        else:
            command = ["rpm", "-ql", package["name"]]
        owned = set(subprocess.check_output(command, text=True).splitlines())
        expected = json.loads((Path("/artifacts") / package["content"]).read_text())
        regular = set()
        for name, entry in expected.items():
            path = Path(name)
            if "sha256" in entry or "symlink" in entry:
                assert name in owned, (package["name"], name, "unowned")
                regular.add(name)
                if "symlink" in entry:
                    assert path.is_symlink() and os.readlink(path) == entry["symlink"], path
                else:
                    assert sha(path) == entry["sha256"], path
                    assert stat.S_IMODE(path.stat().st_mode) == entry["mode"], path
                assert path.lstat().st_uid == 0 and path.lstat().st_gid == 0, path
        ownership[package["name"]] = regular
    if shape == "split":
        assert not set.intersection(*ownership.values()), "split packages share file ownership"
    subprocess.run(["/usr/lib/nvbroadcast/check-packages"], check=True)
    assert not Path("/usr/lib/nvbroadcast/.transaction").exists()
    return {"status": "pass", "runtime_entries": len(manifest),
            "owned_files": {k: len(v) for k, v in ownership.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("deb", "rpm"))
    parser.add_argument("shape", choices=("self", "split"))
    parser.add_argument("revision", type=int)
    args = parser.parse_args()
    print("RESULT=" + json.dumps(verify(args.family, args.shape, args.revision), sort_keys=True))

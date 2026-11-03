#!/usr/bin/env python3
"""Preserve verified RPM unpack remnants before replaying an interrupted prototype.

This is a maintainer recovery tool for the bounded container experiment. It
never deletes a file. A recognized partial payload is moved to a fresh recovery
directory, with its original path, bytes, mode and digest recorded. Unrecognized
paths stop recovery without being moved. Native reinstall and full integrity
verification are still required after this step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT_OWNER = (0, 0)
TEMPORARY = re.compile(r"^(?P<path>.+);(?P<tid>[0-9a-f]{8})$")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_parents(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink in recovery directory path: {path}")


def prefix_matches(partial: Path, source: Path) -> bool:
    with partial.open("rb") as left, source.open("rb") as right:
        while chunk := left.read(1024 * 1024):
            if chunk != right.read(len(chunk)):
                return False
    return True


def plan(runtime: Path, payload: Path, manifest: dict, baseline: dict,
         tid_min: int, tid_max: int, native_source: Path | None = None,
         native_manifest: dict | None = None, native_baseline: dict | None = None) -> list[dict]:
    check_parents(runtime)
    check_parents(payload)
    if not (0 <= tid_min <= tid_max < 2**32) or tid_max - tid_min > 300:
        raise ValueError("invalid recorded interruption time window")
    marker = runtime.parent / ".transaction"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("interrupted native transaction marker required")
    native_manifest, native_baseline = native_manifest or {}, native_baseline or {}
    paths = set(runtime.rglob("*"))
    if native_manifest:
        if native_source is None:
            raise ValueError("verified native integration source required")
        check_parents(native_source)
        if any(not Path(name).is_absolute() or ".." in Path(name).parts for name in native_manifest):
            raise ValueError("invalid native manifest path")
        paths.update(runtime.parent.iterdir())
        for name, entry in native_manifest.items():
            path = Path(name)
            if not path.is_relative_to(runtime) and ({"sha256", "symlink"} & entry.keys()):
                paths.update(path.parent.glob(path.name + ";*"))
    result = []
    for path in sorted(paths):
        runtime_file = path.is_relative_to(runtime) and path != runtime
        relative = str(path.relative_to(runtime)) if runtime_file else str(path)
        selected = manifest if runtime_file else native_manifest
        prior = baseline if runtime_file else native_baseline
        if relative in selected or relative in prior or path in (marker, runtime.parent / ".legacy-runtime"):
            continue
        match = TEMPORARY.fullmatch(relative)
        if not match or not tid_min <= int(match["tid"], 16) <= tid_max:
            raise ValueError(f"unrecognized file in interrupted runtime: {relative}")
        expected = selected.get(match["path"])
        if not expected or not ({"sha256", "symlink"} & expected.keys()):
            raise ValueError(f"temporary path has no selected payload file: {relative}")
        check_parents(path.parent)
        source = (payload / match["path"] if runtime_file else native_source / match["path"].lstrip("/"))
        check_parents(source.parent)
        info = path.lstat()
        if (info.st_uid, info.st_gid) != ROOT_OWNER or info.st_nlink != 1:
            raise ValueError(f"unexpected owner or hardlinks: {relative}")
        entry = {"original": relative, "payload_path": match["path"], "tid": match["tid"],
                 "mode": stat.S_IMODE(info.st_mode), "scope": "runtime" if runtime_file else "native",
                 "stored_as": relative if runtime_file else "native/" + relative.lstrip("/")}
        if "symlink" in expected:
            if not path.is_symlink() or os.readlink(path) != expected["symlink"]:
                raise ValueError(f"temporary symlink differs from selected payload: {relative}")
            if not source.is_symlink() or os.readlink(source) != expected["symlink"]:
                raise ValueError(f"selected payload symlink changed: {match['path']}")
            entry["symlink"] = os.readlink(path)
        else:
            if not stat.S_ISREG(info.st_mode) or source.is_symlink() or not source.is_file():
                raise ValueError(f"unexpected temporary file type: {relative}")
            expected_size = expected.get("size", source.stat().st_size)
            if sha(source) != expected["sha256"] or source.stat().st_size != expected_size:
                raise ValueError(f"selected payload changed: {match['path']}")
            if info.st_size > expected_size or not prefix_matches(path, source):
                raise ValueError(f"temporary bytes differ from selected payload: {relative}")
            entry.update(size=info.st_size, sha256=sha(path))
        result.append(entry)
    return result


def recover(runtime: Path, payload: Path, manifest: dict, baseline: dict,
            tid_min: int, tid_max: int, quarantine: Path, native_source: Path | None = None,
            native_manifest: dict | None = None, native_baseline: dict | None = None) -> dict:
    check_parents(runtime)
    check_parents(payload)
    check_parents(quarantine)
    runtime, payload, quarantine = runtime.resolve(), payload.resolve(), quarantine.resolve()
    if quarantine.exists() or quarantine.is_relative_to(runtime) or runtime.is_relative_to(quarantine):
        raise ValueError("recovery directory must be new and outside the runtime")
    entries = plan(runtime, payload, manifest, baseline, tid_min, tid_max,
                   native_source, native_manifest, native_baseline)
    quarantine.mkdir(mode=0o700, parents=True)
    report = {"status": "moving", "runtime": str(runtime), "tid_min": tid_min,
              "tid_max": tid_max, "preserved": []}
    report_path = quarantine / "recovery.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for entry in entries:
        source = runtime / entry["original"] if entry["scope"] == "runtime" else Path(entry["original"])
        destination = quarantine / "files" / entry["stored_as"]
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.rename(source, destination)
        report["preserved"].append(entry)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "preserved"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--tid-min", type=int, required=True)
    parser.add_argument("--tid-max", type=int, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--native-baseline", type=Path)
    args = parser.parse_args()
    report = recover(args.runtime.absolute(), args.payload.absolute(), json.loads(args.manifest.read_text()),
                     json.loads(args.baseline_manifest.read_text()), args.tid_min, args.tid_max,
                     args.quarantine.absolute(), args.native_source,
                     json.loads(args.native_manifest.read_text()) if args.native_manifest else None,
                     json.loads(args.native_baseline.read_text()) if args.native_baseline else None)
    print("RESULT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

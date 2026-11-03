#!/usr/bin/env python3
"""Select and verify fixed macOS supplier wheels without installing or resolving."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import urlsplit
import zipfile

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from wheel_target import rank_wheels
from runtime import MINIMUM_PYTHON


PYTHON_VERSION = ".".join(map(str, MINIMUM_PYTHON))
MARKERS = {
    "implementation_name": "cpython", "implementation_version": PYTHON_VERSION,
    "os_name": "posix", "platform_machine": "arm64", "platform_release": "23.0.0",
    "platform_system": "Darwin", "platform_version": "", "python_full_version": PYTHON_VERSION,
    "platform_python_implementation": "CPython", "python_version": "3.13", "sys_platform": "darwin",
}
REQUIRED = {"onnxruntime", "faster-whisper", "pip", "setuptools", "wheel", "packaging"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_input(package: dict, wheel: dict) -> dict:
    """Reject alternate artifact types and validate names before any download."""
    if set(wheel) - {"url", "hashes", "size", "upload-time"}:
        raise ValueError("Supplier input must be an HTTPS wheel, not a local archive")
    url = urlsplit(wheel["url"])
    filename = url.path.rsplit("/", 1)[-1]
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename)):
        raise ValueError("Supplier input must have a simple HTTPS wheel URL")
    if set(wheel["hashes"]) != {"sha256"} or not re.fullmatch(r"[0-9a-f]{64}", wheel["hashes"]["sha256"]):
        raise ValueError("Supplier wheel needs exactly one SHA-256 pin")
    name, version, _, _ = parse_wheel_filename(filename)
    if name != package["name"] or version != Version(package["version"]):
        raise ValueError("Supplier filename does not match its locked name/version")
    result = {"name": package["name"], "version": package["version"], "file": filename,
              "url": wheel["url"], "sha256": wheel["hashes"]["sha256"]}
    if "size" in wheel:
        if type(wheel["size"]) is not int or wheel["size"] <= 0:
            raise ValueError("Invalid supplier wheel size")
        result["size"] = wheel["size"]
    return result


def supplier_selection(lock: dict, *, refreshing: bool = False) -> list[dict]:
    if (lock.get("lock-version") != "1.0" or lock.get("requires-python") != ">=3.13"
            or set(lock) - {"lock-version", "created-by", "requires-python", "packages"}):
        raise ValueError("Unsupported fixed supplier lock contract")
    result, owners = [], set()
    for package in lock["packages"]:
        name = package["name"]
        if (not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or name != canonicalize_name(name) or name in owners):
            raise ValueError("Invalid or duplicate supplier owner")
        owners.add(name)
        if refreshing and name == "nvbroadcast":
            # uv resolves the current locally rebuilt app, but it must never
            # become a supplier pin that could preserve an older source wheel.
            if set(package) != {"name", "version", "archive"}:
                raise ValueError("Expected one rebuilt application archive")
            continue
        allowed = {"name", "version", "wheels", "marker"} if refreshing else {"name", "version", "wheels"}
        if name == "nvbroadcast" or set(package) - allowed or not {"name", "version", "wheels"} <= set(package):
            raise ValueError("Supplier lock may contain only external wheel packages")
        if refreshing and package.get("marker") and not Marker(package["marker"]).evaluate({**MARKERS, "extra": ""}):
            continue
        wheels = [wheel_input(package, wheel) for wheel in package["wheels"]]
        if not wheels or (not refreshing and len(wheels) != 1):
            raise ValueError("Fixed supplier lock needs exactly one wheel per owner")
        ranks = rank_wheels([wheel["file"] for wheel in wheels])
        compatible = [wheel for wheel in wheels if ranks[wheel["file"]] is not None]
        if not compatible:
            raise ValueError(f"No macOS 14 arm64 CPython 3.13 wheel for {name}")
        result.append(min(compatible, key=lambda wheel: (ranks[wheel["file"]], wheel["file"])))
    selected_owners = {item["name"] for item in result}
    if not REQUIRED <= selected_owners or "onnxruntime-gpu" in selected_owners:
        raise ValueError("Supplier lock requires the CPU owner and all runtime/bootstrap roots")
    return sorted(result, key=lambda item: item["name"])


def read_selection(path: Path, expected_sha256: str) -> list[dict]:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError("Reviewed supplier lock SHA-256 mismatch")
    return supplier_selection(tomllib.loads(path.read_text()))


def render_lock(selection: list[dict]) -> str:
    lines = ['# Reviewed external Python wheels; the application is rebuilt from committed source.',
             'lock-version = "1.0"', 'created-by = "uv"', 'requires-python = ">=3.13"']
    for item in selection:
        wheel = 'url = ' + json.dumps(item["url"])
        if "size" in item:
            wheel += ', size = ' + str(item["size"])
        wheel += ', hashes = { sha256 = ' + json.dumps(item["sha256"]) + ' }'
        lines += ['', '[[packages]]', 'name = ' + json.dumps(item["name"]),
                  'version = ' + json.dumps(item["version"]), 'wheels = [{ ' + wheel + ' }]']
    return '\n'.join(lines) + '\n'


def propose_lock(resolved_path: Path, app_wheel: Path, output: Path) -> list[dict]:
    lock = tomllib.loads(resolved_path.read_text())
    applications = [package for package in lock["packages"] if package["name"] == "nvbroadcast"]
    if len(applications) != 1:
        raise ValueError("Resolver must use exactly one current application wheel")
    archive = applications[0].get("archive", {})
    if (Path(archive.get("path", "")).name != app_wheel.name
            or archive.get("hashes") != {"sha256": sha256(app_wheel)}):
        raise ValueError("Resolver application differs from the rebuilt source wheel")
    selection = supplier_selection(lock, refreshing=True)
    output.write_text(render_lock(selection))
    return selection


def wheel_metadata(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular wheel")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist()
                 if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("A wheel must have exactly one top-level METADATA")
        info = BytesParser().parsebytes(archive.read(names[0]))
    name, version, _, _ = parse_wheel_filename(path.name)
    if (canonicalize_name(info.get("Name", "")) != name
            or info.get("Version") != str(version)):
        raise ValueError("Wheel metadata differs from its filename")
    if rank_wheels([path.name])[path.name] is None:
        raise ValueError("Wheel has incompatible macOS/Python/architecture tags")
    if PYTHON_VERSION not in SpecifierSet(info.get("Requires-Python", "")):
        raise ValueError(f"{name} does not support the CPython 3.13 deployment floor")
    return info


def verify_wheelhouse(selection: list[dict], directory: Path, app_wheel: Path,
                      faster_whisper: str, bootstrap: list[dict]) -> dict:
    expected = {item["file"] for item in selection}
    if len(expected) != len(selection) or {path.name for path in directory.iterdir()} != expected:
        raise ValueError("Supplier wheelhouse has missing, extra or duplicate files")
    metadata = {}
    for item in selection:
        path = directory / item["file"]
        if (path.is_symlink() or not path.is_file() or sha256(path) != item["sha256"]
                or ("size" in item and path.stat().st_size != item["size"])):
            raise ValueError(f"Supplier wheel hash/size mismatch: {path.name}")
        info = wheel_metadata(path)
        if info["Name"] is None or canonicalize_name(info["Name"]) != item["name"] or info["Version"] != item["version"]:
            raise ValueError("Supplier metadata differs from reviewed selection")
        metadata[item["name"]] = info
    app = wheel_metadata(app_wheel)
    if canonicalize_name(app["Name"]) != "nvbroadcast":
        raise ValueError("Application wheel must belong to nvbroadcast")
    metadata["nvbroadcast"] = app
    if metadata["faster-whisper"]["Version"] != faster_whisper:
        raise ValueError("Managed faster-whisper version differs from supplier lock")
    by_name = {item["name"]: item for item in selection}
    for item in bootstrap:
        if any(by_name[item["name"]][key] != item[key] for key in ("version", "url", "sha256")):
            raise ValueError("Builder bootstrap differs from supplier lock")

    # Inspect wheel METADATA, never install/execute a target wheel on the builder.
    # All selected roots' base dependencies are checked; requested extras follow
    # dependency edges rather than leaking the app's extras into other packages.
    extras = {"nvbroadcast": {"cpu", "meeting-support"}}
    pending, visited = set(metadata), {}
    while pending:
        owner = pending.pop()
        selected_extras = frozenset(extras.get(owner, ()))
        if visited.get(owner) == selected_extras:
            continue
        visited[owner] = selected_extras
        declared = {canonicalize_name(extra) for extra in metadata[owner].get_all("Provides-Extra", [])}
        if not selected_extras <= declared:
            raise ValueError(f"{owner} does not provide requested extras: {sorted(selected_extras - declared)}")
        for raw in metadata[owner].get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate({**MARKERS, "extra": extra})
                                               for extra in {"", *selected_extras}):
                continue
            name = canonicalize_name(requirement.name)
            if requirement.url:
                raise ValueError(f"{owner} requires an unreviewed direct URL: {requirement}")
            if name not in metadata or metadata[name]["Version"] not in requirement.specifier:
                raise ValueError(f"Unsatisfied locked dependency: {owner} requires {requirement}")
            extras.setdefault(name, set()).update(canonicalize_name(extra) for extra in requirement.extras)
            pending.add(name)
    return {"status": "pass", "target": "macos-arm64-cp313-cpu", "minimum_macos": "14.0",
            "minimum_python": PYTHON_VERSION,
            "supplier_count": len(selection), "application_sha256": sha256(app_wheel),
            "application_version": app["Version"], "dependency_closure": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "propose", "verify"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--app-wheel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--faster-whisper")
    args = parser.parse_args()
    if args.command == "propose":
        result = propose_lock(args.lock, args.app_wheel, args.output)
    else:
        result = read_selection(args.lock, args.sha256)
        if args.command == "verify":
            result = verify_wheelhouse(result, args.wheelhouse, args.app_wheel,
                                       args.faster_whisper, json.loads(args.pins.read_text())["bootstrap"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

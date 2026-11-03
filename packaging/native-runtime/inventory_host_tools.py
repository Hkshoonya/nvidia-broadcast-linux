"""Bounded, source-pinned schema tooling for the optional Ubuntu evidence job.

This is a host-only CPython 3.12 Linux x86_64 closure, never a target package
dependency. No installer, resolver, wheel entry point or .pth file is executed.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PINS = Path(__file__).with_name("python-inventory-tools.json")
MAX_WHEEL = 2 * 1024**2
MAX_DOWNLOAD = 8 * 1024**2
MAX_EXPANDED = 32 * 1024**2
MAX_MEMBERS = 5000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_directory(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and stat.S_ISDIR(path.lstat().st_mode),
            "host tool directory is redirected")
    return path


def inputs():
    pins = json.loads(PINS.read_text())
    require(pins["schema_version"] == 1 and pins["host"] == {
        "system": "Linux", "machine": "x86_64", "implementation": "cpython", "python": [3, 12]},
        "unsupported evidence host policy")
    require(platform.system() == "Linux" and platform.machine() == "x86_64"
            and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12),
            "schema evidence tooling requires the pinned CPython3.12 Linux x86_64 host")
    packages = pins["packages"]
    require(len(packages) == 6 and len({p["name"] for p in packages}) == 6
            and {p["name"] for p in packages} == {"attrs", "jsonschema", "jsonschema-specifications",
                                                 "referencing", "rpds-py", "typing-extensions"},
            "schema host closure changed")
    total = 0
    for item in packages:
        url = urlsplit(item["url"])
        require(url.scheme == "https" and url.netloc == "files.pythonhosted.org" and not url.query
                and not url.fragment and url.path.endswith("/" + item["filename"])
                and re.fullmatch(r"[A-Za-z0-9_.-]+\.whl", item["filename"])
                and re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
                and type(item["bytes"]) is int and 0 < item["bytes"] <= MAX_WHEEL,
                "invalid pinned schema wheel")
        total += item["bytes"]
    require(total <= MAX_DOWNLOAD, "schema download aggregate exceeds bound")
    return packages


def wheel_files(wheels, packages):
    """Verify the complete original wheels before interpreting any member."""
    files, total, count = {}, 0, 0
    for item in packages:
        wheel = wheels / item["filename"]
        require(wheel.resolve(strict=True) == wheel and stat.S_ISREG(wheel.lstat().st_mode)
                and wheel.stat().st_size == item["bytes"] and sha(wheel) == item["sha256"],
                "schema wheel differs from source pin")
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                count += 1
                path = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                require(count <= MAX_MEMBERS and not path.is_absolute() and str(path) == member.filename.rstrip("/")
                        and all(part not in ("", ".", "..") for part in path.parts)
                        and not any(ord(char) < 32 for char in member.filename) and "\\" not in member.filename
                        and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and not (member.flag_bits & 1) and not stat.S_ISLNK(mode)
                        and (not stat.S_IFMT(mode) or stat.S_ISDIR(mode) or stat.S_ISREG(mode)),
                        "unsafe schema wheel member")
                if member.is_dir():
                    continue
                require(path.suffix.lower() != ".pth" and not any(part.endswith(".data") for part in path.parts)
                        and str(path) not in files and 0 <= member.file_size <= 8 * 1024**2,
                        "unsupported or duplicate schema wheel member")
                total += member.file_size
                require(total <= MAX_EXPANDED, "expanded schema closure exceeds bound")
                data = archive.read(member)
                require(len(data) == member.file_size, "schema wheel member size differs")
                files[str(path)] = data
    return files


def prepare(output, *, cache=None):
    packages = inputs()
    output = Path(output).absolute()
    require(output.parent.resolve(strict=True) == output.parent and not output.exists() and not output.is_symlink(),
            "schema host output must be fresh and nonredirected")
    output.mkdir(mode=0o700)
    wheels = output / "wheels"
    wheels.mkdir()
    started = time.monotonic()
    for item in packages:
        target = wheels / item["filename"]
        if cache is not None:
            source = checked_directory(cache) / item["filename"]
            require(source.resolve(strict=True) == source and stat.S_ISREG(source.lstat().st_mode)
                    and source.stat().st_size == item["bytes"], "invalid cached schema wheel")
            data = source.read_bytes()
        else:
            remaining = 120 - (time.monotonic() - started)
            require(remaining > 0, "schema download deadline exceeded")
            with urlopen(item["url"], timeout=min(30, remaining)) as stream:
                require(stream.geturl() == item["url"], "schema wheel redirect is forbidden")
                data = stream.read(item["bytes"] + 1)
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                "downloaded schema wheel differs from pin")
        target.write_bytes(data)
    site = output / "site"
    site.mkdir()
    for name, data in wheel_files(wheels, packages).items():
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return verified_site(output)


def verified_site(root):
    """Re-read exact wheel bytes and every extracted member before imports."""
    root = checked_directory(root)
    packages = inputs()
    require({p.name for p in root.iterdir()} == {"wheels", "site"}, "unexpected schema tool root member")
    wheels, site = checked_directory(root / "wheels"), checked_directory(root / "site")
    require({p.name for p in wheels.iterdir()} == {p["filename"] for p in packages}, "schema wheel set differs")
    expected = wheel_files(wheels, packages)
    expected_dirs = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    actual_files, actual_dirs = set(), set()
    for count, path in enumerate(site.rglob("*"), 1):
        mode, name = path.lstat().st_mode, path.relative_to(site).as_posix()
        require(count <= 2 * MAX_MEMBERS and (stat.S_ISREG(mode) or stat.S_ISDIR(mode)),
                "redirected or excessive extracted schema tools")
        if stat.S_ISDIR(mode):
            actual_dirs.add(name)
        else:
            actual_files.add(name)
            require(name in expected and path.stat().st_size == len(expected[name])
                    and path.read_bytes() == expected[name], "extracted schema tool changed")
    require(actual_files == set(expected) and actual_dirs == expected_dirs, "extracted schema tool set differs")
    return site, {"input_manifest_sha256": sha(PINS), "host_python": sys.version,
                  "wheels": {p["filename"]: {"bytes": p["bytes"], "sha256": p["sha256"]} for p in packages},
                  "site_sha256": hashlib.sha256(json.dumps(
                      {name: hashlib.sha256(data).hexdigest() for name, data in expected.items()},
                      sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def validate_imports(site, packages):
    """Check the evidence-only dependency closure after selecting verified paths."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    distributions = list(metadata.distributions(path=[str(site)]))
    actual = {}
    for dist in distributions:
        name = canonicalize_name(dist.metadata["Name"])
        require(name not in actual, "duplicate schema host owner")
        actual[name] = dist.version
    require(actual == {p["name"]: p["version"] for p in packages}, "schema host metadata differs from pins")
    environment = dict(default_environment(), extra="")
    for dist in distributions:
        require(SpecifierSet(dist.metadata.get("Requires-Python", "")).contains(platform.python_version()),
                "schema host Python requirement is unsatisfied")
        for value in dist.requires or ():
            dependency = Requirement(value)
            if dependency.marker is None or dependency.marker.evaluate(environment):
                name = canonicalize_name(dependency.name)
                require(not dependency.url and name in actual and dependency.specifier.contains(actual[name]),
                        "schema host dependency closure is unsatisfied")
    import attrs
    import jsonschema
    import jsonschema_specifications
    import referencing
    import rpds
    import typing_extensions
    for module in (attrs, jsonschema, jsonschema_specifications, referencing, rpds, typing_extensions):
        require(Path(module.__file__).resolve().is_relative_to(site), "ambient schema tool import is forbidden")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:
        parser.error("run with the trusted host interpreter -I -S")
    try:
        _, receipt = prepare(args.output, cache=args.cache)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Schema evidence tools failed: {type(error).__name__}: {error}\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

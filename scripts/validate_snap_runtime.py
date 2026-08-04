#!/usr/bin/env python3
"""Validate the Python dependency closure inside an extracted Snap."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
from pathlib import Path
import re
import sys

import packaging
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


REQUIRED_RUNTIME = {
    "packaging": SpecifierSet(">=26.0"),
    "setuptools": SpecifierSet(">=83.0.0"),
    "opencv-python-headless": SpecifierSet(">=4.8,<5"),
}
IMPORT_PROBES = ("packaging", "setuptools")
ARCHITECTURES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}


def discover_package_roots(snap_root: Path) -> list[Path]:
    patterns = (
        "lib/python*/site-packages",
        "lib/python*/dist-packages",
        "usr/lib/python*/site-packages",
        "usr/lib/python*/dist-packages",
        "usr/local/lib/python*/site-packages",
        "usr/local/lib/python*/dist-packages",
    )
    roots = {
        path.resolve()
        for pattern in patterns
        for path in snap_root.glob(pattern)
        if path.is_dir()
    }
    return sorted(roots)


def _python_environment(
    package_roots: list[Path], platform_machine: str
) -> dict[str, str]:
    environment = default_environment()
    environment.update(
        os_name="posix",
        platform_machine=ARCHITECTURES[platform_machine],
        platform_system="Linux",
        sys_platform="linux",
        extra="",
    )

    versions = {
        match.groups()
        for root in package_roots
        if (match := re.search(r"python(\d+)\.(\d+)", str(root)))
    }
    if len(versions) == 1:
        major, minor = versions.pop()
        environment["python_version"] = f"{major}.{minor}"
        if (int(major), int(minor)) != sys.version_info[:2]:
            environment["python_full_version"] = f"{major}.{minor}.0"
    return environment


def _distribution_index(
    package_roots: list[Path],
) -> tuple[list[metadata.Distribution], dict[str, list[str]]]:
    distributions = list(
        metadata.distributions(path=[str(path) for path in package_roots])
    )
    installed: dict[str, list[str]] = {}
    for distribution in distributions:
        name = distribution.metadata.get("Name")
        if not name:
            continue
        installed.setdefault(canonicalize_name(name), []).append(distribution.version)
    return distributions, installed


def dependency_problems(
    snap_root: Path, platform_machine: str
) -> tuple[int, list[str]]:
    package_roots = discover_package_roots(snap_root)
    if not package_roots:
        return 0, ["no Python package roots were found"]

    distributions, installed = _distribution_index(package_roots)
    environment = _python_environment(package_roots, platform_machine)
    problems: list[str] = []

    for package_name, specifier in REQUIRED_RUNTIME.items():
        versions = installed.get(canonicalize_name(package_name), [])
        if not versions:
            problems.append(f"required runtime package is missing: {package_name}{specifier}")
        elif len(versions) != 1:
            problems.append(
                f"required runtime package has multiple owners: "
                f"{package_name} ({', '.join(versions)})"
            )
        elif not any(version in specifier for version in versions):
            problems.append(
                f"{package_name} versions {', '.join(versions)} do not satisfy {specifier}"
            )

    opencv_owners = {
        name: versions for name, versions in installed.items() if name.startswith("opencv-")
    }
    if set(opencv_owners) != {"opencv-python-headless"}:
        rendered = ", ".join(
            f"{name} ({', '.join(versions)})"
            for name, versions in sorted(opencv_owners.items())
        )
        problems.append(f"Snap must have exactly one OpenCV owner; found: {rendered}")

    for distribution in distributions:
        owner = distribution.metadata.get("Name", "<unknown>")
        for raw_requirement in distribution.requires or ():
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement as error:
                problems.append(f"{owner} has invalid requirement {raw_requirement!r}: {error}")
                continue
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue

            versions = installed.get(canonicalize_name(requirement.name), [])
            if not versions:
                problems.append(f"{owner} requires missing package {requirement}")
            elif requirement.specifier and not any(
                version in requirement.specifier for version in versions
            ):
                problems.append(
                    f"{owner} requires {requirement}, found {', '.join(versions)}"
                )

    return len(installed), sorted(set(problems))


def import_problems(package_roots: list[Path]) -> list[str]:
    problems = []
    for module_name in IMPORT_PROBES:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # pragma: no cover - reported by artifact CI
            problems.append(f"cannot import {module_name}: {error}")
            continue
        module_path = Path(module.__file__).resolve()
        if not any(module_path.is_relative_to(root) for root in package_roots):
            problems.append(f"{module_name} resolved outside the Snap: {module_path}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snap_root", type=Path)
    parser.add_argument(
        "--platform-machine",
        required=True,
        choices=tuple(ARCHITECTURES),
    )
    args = parser.parse_args()

    snap_root = args.snap_root.resolve()
    package_roots = discover_package_roots(snap_root)
    count, problems = dependency_problems(snap_root, args.platform_machine)
    problems.extend(import_problems(package_roots))

    if problems:
        for problem in sorted(set(problems)):
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"Validated {count} Python distributions for "
        f"{ARCHITECTURES[args.platform_machine]}; packaging {packaging.__version__}, "
        "setuptools and OpenCV runtime constraints are satisfied."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

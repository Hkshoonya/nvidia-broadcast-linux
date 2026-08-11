"""Inspect Python distributions stored inside a packaged runtime artifact."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import metadata
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from nvbroadcast.runtime.variants import canonical_distribution_paths


ARCHITECTURES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}


def _installed_versions(
    distributions: tuple[metadata.Distribution, ...],
) -> dict[str, tuple[str, ...]]:
    installed: dict[str, list[str]] = {}
    for distribution in distributions:
        name = distribution.metadata.get("Name")
        if name:
            installed.setdefault(canonicalize_name(name), []).append(
                distribution.version
            )
    return {name: tuple(versions) for name, versions in installed.items()}


def discover_package_roots(artifact_root: Path) -> list[Path]:
    """Return every Python package root contained by an extracted artifact."""
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
        for path in artifact_root.glob(pattern)
        if path.is_dir()
    }
    return sorted(roots)


def _marker_environment(
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


@dataclass(frozen=True)
class ArtifactEnvironment:
    """Installed-distribution view of a current or extracted environment."""

    package_roots: tuple[Path, ...]
    distributions: tuple[metadata.Distribution, ...]
    installed: dict[str, tuple[str, ...]]
    markers: dict[str, str]

    @classmethod
    def current(cls) -> "ArtifactEnvironment":
        """Inspect distributions visible to the running interpreter."""
        distributions = tuple(
            metadata.distributions(path=canonical_distribution_paths(sys.path))
        )
        markers = default_environment()
        markers["extra"] = ""
        return cls(
            package_roots=(),
            distributions=distributions,
            installed=_installed_versions(distributions),
            markers=markers,
        )

    @classmethod
    def inspect(
        cls, artifact_root: Path, platform_machine: str
    ) -> "ArtifactEnvironment":
        package_roots = discover_package_roots(artifact_root)
        distributions = tuple(
            metadata.distributions(path=[str(path) for path in package_roots])
        )
        return cls(
            package_roots=tuple(package_roots),
            distributions=distributions,
            installed=_installed_versions(distributions),
            markers=_marker_environment(package_roots, platform_machine),
        )

    def dependency_closure_problems(
        self,
        substitutions: Mapping[str, str] | None = None,
        *,
        roots: Iterable[str] | None = None,
    ) -> list[str]:
        """Return unsatisfied or malformed active distribution requirements."""
        substitutions = {
            canonicalize_name(name): canonicalize_name(provider)
            for name, provider in (substitutions or {}).items()
        }
        problems: list[str] = []

        distributions_by_name: dict[str, list[metadata.Distribution]] = {}
        for distribution in self.distributions:
            name = distribution.metadata.get("Name")
            if name:
                distributions_by_name.setdefault(
                    canonicalize_name(name), []
                ).append(distribution)

        if roots is None:
            selected_distributions = self.distributions
        else:
            selected: list[metadata.Distribution] = []
            pending = {canonicalize_name(root) for root in roots}
            missing_roots = sorted(
                root for root in pending if root not in distributions_by_name
            )
            problems.extend(
                f"required dependency root is missing: {root}"
                for root in missing_roots
            )
            visited: set[str] = set()
            while pending:
                distribution_name = pending.pop()
                if distribution_name in visited:
                    continue
                visited.add(distribution_name)
                distributions = distributions_by_name.get(distribution_name, ())
                selected.extend(distributions)
                for distribution in distributions:
                    for raw_requirement in distribution.requires or ():
                        try:
                            requirement = Requirement(raw_requirement)
                        except InvalidRequirement:
                            continue
                        if requirement.marker and not requirement.marker.evaluate(
                            self.markers
                        ):
                            continue
                        requirement_name = canonicalize_name(requirement.name)
                        provided_name = substitutions.get(
                            requirement_name, requirement_name
                        )
                        if self.installed.get(provided_name):
                            pending.add(provided_name)
            selected_distributions = tuple(selected)

        for distribution in selected_distributions:
            owner = distribution.metadata.get("Name", "<unknown>")
            for raw_requirement in distribution.requires or ():
                try:
                    requirement = Requirement(raw_requirement)
                except InvalidRequirement as error:
                    problems.append(
                        f"{owner} has invalid requirement {raw_requirement!r}: {error}"
                    )
                    continue
                if requirement.marker and not requirement.marker.evaluate(self.markers):
                    continue

                requirement_name = canonicalize_name(requirement.name)
                provided_name = substitutions.get(
                    requirement_name, requirement_name
                )
                versions = self.installed.get(provided_name, ())
                if not versions:
                    problems.append(f"{owner} requires missing package {requirement}")
                elif requirement.specifier and not any(
                    version in requirement.specifier for version in versions
                ):
                    problems.append(
                        f"{owner} requires {requirement}, found {', '.join(versions)}"
                    )
        return sorted(set(problems))

    def import_problems(self, module_names: tuple[str, ...]) -> list[str]:
        """Verify imports resolve from this artifact rather than the host."""
        problems = []
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except Exception as error:  # pragma: no cover - reported by artifact CI
                problems.append(f"cannot import {module_name}: {error}")
                continue
            module_path = Path(module.__file__).resolve()
            if not any(
                module_path.is_relative_to(root) for root in self.package_roots
            ):
                problems.append(
                    f"{module_name} resolved outside the artifact: {module_path}"
                )
        return problems

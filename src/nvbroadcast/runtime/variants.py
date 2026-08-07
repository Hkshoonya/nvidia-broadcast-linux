"""Define and validate mutually exclusive ONNX Runtime variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
import sys
from typing import Iterable, Mapping


def canonical_distribution_paths(paths: Iterable[str]) -> list[str]:
    """Return unique metadata search paths after resolving filesystem aliases."""
    return list(dict.fromkeys(str(Path(path or ".").resolve()) for path in paths))


class RuntimeVariant(StrEnum):
    """Supported ONNX Runtime ownership choices."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class RuntimeContract:
    """Distribution and provider invariants for one runtime variant."""

    distribution: str
    required_providers: frozenset[str]
    forbidden_providers: frozenset[str] = frozenset()


RUNTIME_CONTRACTS = {
    RuntimeVariant.CPU: RuntimeContract(
        distribution="onnxruntime",
        required_providers=frozenset({"CPUExecutionProvider"}),
        forbidden_providers=frozenset({"CUDAExecutionProvider"}),
    ),
    RuntimeVariant.CUDA: RuntimeContract(
        distribution="onnxruntime-gpu",
        required_providers=frozenset(
            {"CPUExecutionProvider", "CUDAExecutionProvider"}
        ),
    ),
}

def _canonicalize_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


RUNTIME_DISTRIBUTIONS = frozenset(
    _canonicalize_name(contract.distribution)
    for contract in RUNTIME_CONTRACTS.values()
)


def current_distribution_inventory() -> dict[str, tuple[str, ...]]:
    """Return installed versions of runtime-owning distributions."""
    installed: dict[str, list[str]] = {}
    for distribution in metadata.distributions(
        path=canonical_distribution_paths(sys.path)
    ):
        name = distribution.metadata.get("Name")
        canonical_name = _canonicalize_name(name) if name else ""
        if canonical_name in RUNTIME_DISTRIBUTIONS:
            installed.setdefault(canonical_name, []).append(distribution.version)
    return {name: tuple(versions) for name, versions in installed.items()}


def detect_runtime_variant(
    installed: Mapping[str, tuple[str, ...]] | None = None,
) -> RuntimeVariant | None:
    """Return variant only when exactly one runtime distribution owns environment."""
    inventory = installed if installed is not None else current_distribution_inventory()
    owners = {
        _canonicalize_name(name)
        for name, versions in inventory.items()
        if versions and _canonicalize_name(name) in RUNTIME_DISTRIBUTIONS
    }
    if len(owners) != 1:
        return None
    owner = owners.pop()
    for variant, contract in RUNTIME_CONTRACTS.items():
        if owner == _canonicalize_name(contract.distribution):
            return variant
    return None


def runtime_ownership_problems(
    variant: RuntimeVariant,
    installed: Mapping[str, tuple[str, ...]],
    providers: Iterable[str],
) -> list[str]:
    """Return violations of selected runtime's ownership/provider contract."""
    contract = RUNTIME_CONTRACTS[variant]
    expected_owner = _canonicalize_name(contract.distribution)
    runtime_inventory = {
        _canonicalize_name(name): tuple(versions)
        for name, versions in installed.items()
        if _canonicalize_name(name) in RUNTIME_DISTRIBUTIONS and versions
    }
    problems: list[str] = []

    expected_versions = runtime_inventory.get(expected_owner, ())
    if len(expected_versions) != 1:
        problems.append(
            f"expected exactly one {contract.distribution} distribution, "
            f"found {len(expected_versions)}"
        )

    unexpected = sorted(set(runtime_inventory) - {expected_owner})
    if unexpected:
        problems.append(
            "unexpected runtime distribution(s): " + ", ".join(unexpected)
        )

    available_providers = set(providers)
    missing = sorted(contract.required_providers - available_providers)
    if missing:
        problems.append("missing execution provider(s): " + ", ".join(missing))
    forbidden = sorted(contract.forbidden_providers & available_providers)
    if forbidden:
        problems.append("forbidden execution provider(s): " + ", ".join(forbidden))
    return problems


def validate_current_runtime(variant: RuntimeVariant) -> list[str]:
    """Validate current interpreter against selected runtime contract."""
    import onnxruntime

    return runtime_ownership_problems(
        variant,
        current_distribution_inventory(),
        onnxruntime.get_available_providers(),
    )

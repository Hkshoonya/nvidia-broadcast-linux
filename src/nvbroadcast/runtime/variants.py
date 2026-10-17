"""Define and validate mutually exclusive ONNX Runtime variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping


FASTER_WHISPER_VERSION = "1.2.1"
FASTER_WHISPER_REQUIREMENT = f"faster-whisper=={FASTER_WHISPER_VERSION}"


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
    return re.sub(r"[-_.]+", "-", name).lower()


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


def _live_available_providers() -> frozenset[str]:
    # GUI probe must never crash; fall back to CPU when providers are unknown.
    try:
        import onnxruntime

        return frozenset(onnxruntime.get_available_providers())
    except Exception:
        return frozenset()


def _is_cuda_build_with_cpu_dist_name(
    installed: Mapping[str, tuple[str, ...]],
    providers: Iterable[str] | None,
) -> bool:
    """CUDA build sold under the plain onnxruntime name (e.g. Arch opt-cuda)."""
    canonical = {
        _canonicalize_name(name): versions
        for name, versions in installed.items()
    }
    versions = canonical.get("onnxruntime", ())
    if set(canonical) != {"onnxruntime"} or len(versions) != 1:
        return False
    return "CUDAExecutionProvider" in set(providers or ())


def detect_runtime_variant(
    installed: Mapping[str, tuple[str, ...]] | None = None,
    providers: Iterable[str] | None = None,
) -> RuntimeVariant | None:
    """Single owner wins; onnxruntime with CUDA provider counts as CUDA."""
    inventory = installed if installed is not None else current_distribution_inventory()
    owners = [
        _canonicalize_name(name)
        for name, versions in inventory.items()
        if _canonicalize_name(name) in RUNTIME_DISTRIBUTIONS
        for _version in versions
    ]
    if len(owners) != 1:
        return None
    owner = owners.pop()
    if owner == _canonicalize_name(RUNTIME_CONTRACTS[RuntimeVariant.CPU].distribution):
        live_providers = providers
        if live_providers is None and installed is None:
            live_providers = _live_available_providers()
        if _is_cuda_build_with_cpu_dist_name(inventory, live_providers):
            return RuntimeVariant.CUDA
    for variant, contract in RUNTIME_CONTRACTS.items():
        if owner == _canonicalize_name(contract.distribution):
            return variant
    return None


def _runtime_distribution_problems(
    variant: RuntimeVariant, installed: Mapping[str, tuple[str, ...]]
) -> list[str]:
    """Check distribution ownership without importing the shared runtime tree."""
    contract = RUNTIME_CONTRACTS[variant]
    expected_owner = _canonicalize_name(contract.distribution)
    runtime_inventory: dict[str, list[str]] = {}
    for name, versions in installed.items():
        canonical_name = _canonicalize_name(name)
        if canonical_name in RUNTIME_DISTRIBUTIONS and versions:
            runtime_inventory.setdefault(canonical_name, []).extend(versions)
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
    return problems


def runtime_ownership_problems(
    variant: RuntimeVariant,
    installed: Mapping[str, tuple[str, ...]],
    providers: Iterable[str],
) -> list[str]:
    """Return violations of selected runtime's ownership/provider contract."""
    contract = RUNTIME_CONTRACTS[variant]
    if variant is RuntimeVariant.CUDA and _is_cuda_build_with_cpu_dist_name(
        installed, providers
    ):
        available_providers = set(providers)
        missing = sorted(contract.required_providers - available_providers)
        if missing:
            return ["missing execution provider(s): " + ", ".join(missing)]
        return []
    problems = _runtime_distribution_problems(variant, installed)

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
    inventory = current_distribution_inventory()
    inventory_names = {_canonicalize_name(name) for name in inventory}
    possible_cpu_named_cuda = (
        variant is RuntimeVariant.CUDA
        and inventory_names == {"onnxruntime"}
        and sum(len(versions) for versions in inventory.values()) == 1
    )
    if not possible_cpu_named_cuda:
        ownership_problems = _runtime_distribution_problems(variant, inventory)
        if ownership_problems:
            return ownership_problems

    try:
        import onnxruntime
    except Exception as error:
        return [f"cannot import onnxruntime: {type(error).__name__}: {error}"]

    try:
        providers = onnxruntime.get_available_providers()
    except Exception as error:
        return [
            "cannot query onnxruntime execution providers: "
            f"{type(error).__name__}: {error}"
        ]

    return runtime_ownership_problems(
        variant,
        inventory,
        providers,
    )

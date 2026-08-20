"""Command-line validation for installed runtime ownership."""

from __future__ import annotations

import argparse
import sys

from nvbroadcast.runtime.probe import ProbeProvider, probe_execution_provider
from nvbroadcast.runtime.variants import RuntimeVariant, validate_current_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(RuntimeVariant))
    parser.add_argument(
        "--provider",
        choices=tuple(ProbeProvider),
        help=(
            "execution provider to probe; defaults to CPU for the CPU variant "
            "and CUDA for the CUDA variant"
        ),
    )
    arguments = parser.parse_args()

    variant = RuntimeVariant(arguments.variant)
    provider = (
        ProbeProvider(arguments.provider)
        if arguments.provider
        else (
            ProbeProvider.CPU
            if variant is RuntimeVariant.CPU
            else ProbeProvider.CUDA
        )
    )
    if variant is RuntimeVariant.CPU and provider is not ProbeProvider.CPU:
        parser.error("the CPU runtime variant can probe only the CPU provider")
    if variant is RuntimeVariant.CUDA and provider is ProbeProvider.CPU:
        parser.error("the CUDA runtime variant probes CUDA or TensorRT")

    problems = validate_current_runtime(variant)
    if problems:
        for problem in problems:
            print(f"runtime validation failed: {problem}", file=sys.stderr)
        return 1

    probe = probe_execution_provider(provider, use_cache=False)
    if not probe.success:
        detail = probe.failure_detail or "no diagnostic was returned"
        print(
            f"runtime execution probe failed for {provider.ort_name}:\n{detail}",
            file=sys.stderr,
        )
        return 1
    print(
        f"runtime variant validated: {variant}; "
        f"{provider.ort_name} executed pinned model {probe.model_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

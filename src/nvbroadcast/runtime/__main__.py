"""Command-line validation for installed runtime ownership."""

from __future__ import annotations

import argparse

from nvbroadcast.runtime.variants import RuntimeVariant, validate_current_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(RuntimeVariant))
    arguments = parser.parse_args()

    variant = RuntimeVariant(arguments.variant)
    problems = validate_current_runtime(variant)
    if problems:
        for problem in problems:
            print(f"runtime validation failed: {problem}")
        return 1
    print(f"runtime variant validated: {variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

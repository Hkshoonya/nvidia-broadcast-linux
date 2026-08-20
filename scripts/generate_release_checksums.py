#!/usr/bin/env python3
"""Generate a deterministic SHA-256 manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path


_SAFE_RELEASE_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_READ_SIZE = 1024 * 1024


class ManifestError(ValueError):
    """Raised when release inputs cannot produce an unambiguous manifest."""


def _sha256_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot open release artifact {path}: {exc}") from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManifestError(f"release artifact is not a regular file: {path}")

        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as artifact:
            descriptor = -1
            for chunk in iter(lambda: artifact.read(_READ_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def generate_manifest(artifacts: Iterable[Path], output: Path) -> tuple[str, ...]:
    """Hash release artifacts and atomically write sorted manifest lines."""

    artifact_paths = tuple(Path(path) for path in artifacts)
    if not artifact_paths:
        raise ManifestError("at least one release artifact is required")

    output = Path(output)
    output_resolved = output.resolve(strict=False)
    entries: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    for path in artifact_paths:
        if path.is_symlink():
            raise ManifestError(f"release artifact must not be a symlink: {path}")
        if not path.exists():
            raise ManifestError(f"release artifact does not exist: {path}")
        if not path.is_file():
            raise ManifestError(f"release artifact is not a regular file: {path}")
        if path.resolve(strict=True) == output_resolved:
            raise ManifestError("the checksum manifest cannot hash itself")

        name = path.name
        if not _SAFE_RELEASE_NAME.fullmatch(name):
            raise ManifestError(f"unsafe release artifact name: {name!r}")
        if name in seen_names:
            raise ManifestError(f"duplicate release artifact name: {name}")

        seen_names.add(name)
        entries.append((name, _sha256_regular_file(path)))

    lines = tuple(
        f"{digest}  {name}"
        for name, digest in sorted(entries, key=lambda entry: entry[0])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as manifest:
            manifest.write("\n".join(lines))
            manifest.write("\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("artifacts", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        lines = generate_manifest(args.artifacts, args.output)
    except (ManifestError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Wrote {args.output} with {len(lines)} SHA-256 entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

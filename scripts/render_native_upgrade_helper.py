#!/usr/bin/env python3
"""Render the native-package upgrader with exact release artifact digests."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile


_READ_SIZE = 1024 * 1024
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_REVISION_PATTERN = re.compile(r"^[1-9][0-9]*$")


class RenderError(ValueError):
    """Raised when a release helper cannot be rendered safely."""


def _sha256_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RenderError(f"cannot open release artifact {path}: {error}") from error

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RenderError(f"release artifact is not a regular file: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as artifact:
            descriptor = -1
            for chunk in iter(lambda: artifact.read(_READ_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def render_helper(
    template: Path,
    deb: Path,
    rpm: Path,
    version: str,
    revision: str,
    output: Path,
) -> str:
    """Render an executable helper bound to one exact DEB and RPM."""
    if not _VERSION_PATTERN.fullmatch(version):
        raise RenderError(f"unsafe package version: {version!r}")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise RenderError(f"unsafe package revision: {revision!r}")

    expected_names = (
        (deb, f"nvbroadcast_{version}-{revision}_all.deb"),
        (rpm, f"nvbroadcast-{version}-{revision}.noarch.rpm"),
    )
    for artifact, expected_name in expected_names:
        if artifact.name != expected_name:
            raise RenderError(
                f"unexpected release artifact name {artifact.name!r}; "
                f"expected {expected_name!r}"
            )

    if template.is_symlink() or not template.is_file():
        raise RenderError(f"upgrade helper template is not a regular file: {template}")
    try:
        content = template.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise RenderError(
            f"cannot read upgrade helper template {template}: {error}"
        ) from error

    replacements = {
        "@TARGET_VERSION@": version,
        "@TARGET_REVISION@": revision,
        "@DEB_SHA256@": _sha256_regular_file(deb),
        "@RPM_SHA256@": _sha256_regular_file(rpm),
    }
    for placeholder, value in replacements.items():
        if content.count(placeholder) != 1:
            raise RenderError(
                f"upgrade helper template must contain {placeholder!r} exactly once"
            )
        content = content.replace(placeholder, value)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o755)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--deb", required=True, type=Path)
    parser.add_argument("--rpm", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        render_helper(
            arguments.template,
            arguments.deb,
            arguments.rpm,
            arguments.version,
            arguments.revision,
            arguments.output,
        )
    except (OSError, RenderError) as error:
        parser.error(str(error))
    print(f"Rendered {arguments.output} for NV Broadcast {arguments.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

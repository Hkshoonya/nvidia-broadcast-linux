#!/usr/bin/env python3
"""Detect NVBroadcast processes using an installer-owned source environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


TARGET_MODULES = frozenset(("nvbroadcast", "nvbroadcast.vcam_service"))
PYTHON_EXECUTABLE = re.compile(r"python(?:\d+(?:\.\d+)?[a-z]*)?")


class ProcessInspectionError(RuntimeError):
    """Raised when process state cannot be checked safely."""


@dataclass(frozen=True)
class SourceProcess:
    pid: int
    module: str


def _decode_nul_separated(data: bytes) -> list[str]:
    return [os.fsdecode(item) for item in data.split(b"\0") if item]


def _decode_environment(data: bytes) -> dict[str, str]:
    environment = {}
    for item in _decode_nul_separated(data):
        name, separator, value = item.partition("=")
        if separator:
            environment[name] = value
    return environment


def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(path)))


def _source_module(
    arguments: Sequence[str],
    *,
    venv: Path,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    if len(arguments) < 3 or arguments[1] != "-m":
        return None

    module = arguments[2]
    if module not in TARGET_MODULES:
        return None

    executable = Path(arguments[0])
    if not PYTHON_EXECUTABLE.fullmatch(executable.name):
        return None

    expected_venv = _normalized_path(venv)
    if executable.is_absolute():
        executable = _normalized_path(executable)
        return module if executable.parent == expected_venv / "bin" else None

    if executable.parent != Path("."):
        if cwd is None:
            raise ProcessInspectionError(
                "cannot resolve a relative Python executable without process cwd"
            )
        executable = _normalized_path(cwd / executable)
        return module if executable.parent == expected_venv / "bin" else None

    virtual_env = (environment or {}).get("VIRTUAL_ENV")
    if virtual_env and _normalized_path(Path(virtual_env)) == expected_venv:
        return module
    return None


def _read_process_file(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ProcessInspectionError(
            f"cannot inspect {description} at {path}: {error}"
        ) from error


def find_source_processes(
    venv: Path,
    *,
    proc_root: Path = Path("/proc"),
    uid: int | None = None,
) -> list[SourceProcess]:
    """Return same-user app processes whose Python belongs to ``venv``."""
    if not proc_root.is_dir():
        raise ProcessInspectionError(f"process filesystem is unavailable at {proc_root}")

    current_uid = os.getuid() if uid is None else uid
    processes = []
    try:
        process_directories = sorted(proc_root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise ProcessInspectionError(
            f"cannot enumerate process filesystem at {proc_root}: {error}"
        ) from error

    for process_dir in process_directories:
        if not process_dir.name.isdigit():
            continue

        try:
            if process_dir.stat().st_uid != current_uid:
                continue
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ProcessInspectionError(
                f"cannot inspect process owner at {process_dir}: {error}"
            ) from error

        try:
            arguments = _decode_nul_separated(
                _read_process_file(process_dir / "cmdline", "process command line")
            )
        except FileNotFoundError:
            continue
        if len(arguments) < 3 or arguments[1] != "-m":
            continue
        if arguments[2] not in TARGET_MODULES:
            continue

        executable = Path(arguments[0])
        cwd = None
        environment = None
        try:
            if not executable.is_absolute() and executable.parent != Path("."):
                cwd = (process_dir / "cwd").resolve(strict=True)
            elif not executable.is_absolute():
                environment = _decode_environment(
                    _read_process_file(
                        process_dir / "environ", "process environment"
                    )
                )
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ProcessInspectionError(
                f"cannot inspect process path at {process_dir}: {error}"
            ) from error

        module = _source_module(
            arguments,
            venv=venv,
            cwd=cwd,
            environment=environment,
        )
        if module is not None:
            processes.append(SourceProcess(int(process_dir.name), module))
    return processes


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    options = parser.parse_args(arguments)

    try:
        processes = find_source_processes(options.venv)
    except ProcessInspectionError as error:
        print(
            f"ERROR: Cannot verify whether the source environment is idle: {error}",
            file=sys.stderr,
        )
        return 2

    if not processes:
        return 0

    print(
        f"ERROR: {options.venv} is in use by running NVBroadcast processes:",
        file=sys.stderr,
    )
    for process in processes:
        print(f"  PID {process.pid}: python -m {process.module}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

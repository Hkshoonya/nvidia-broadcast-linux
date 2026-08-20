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
TARGET_CONSOLE_SCRIPTS = {
    "nvbroadcast": "nvbroadcast",
    "nvbroadcast-vcam": "nvbroadcast.vcam_service",
}
PYTHON_EXECUTABLE = re.compile(r"python(?:\d+(?:\.\d+)?[a-z]*)?")
PYTHON_FLAG_OPTIONS = frozenset("bBdEiIOPqRsSuvx")
PYTHON_EXIT_OPTIONS = frozenset("h?V")
PYTHON_VALUE_OPTIONS = frozenset("WX")
HASH_PYC_MODES = frozenset(("always", "default", "never"))


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


def _target_invocation(arguments: Sequence[str]) -> tuple[str, Path | None] | None:
    if len(arguments) < 2:
        return None
    if not PYTHON_EXECUTABLE.fullmatch(Path(arguments[0]).name):
        return None

    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--check-hash-based-pycs":
            if index + 1 >= len(arguments):
                return None
            if arguments[index + 1] not in HASH_PYC_MODES:
                return None
            index += 2
            continue
        if argument == "--" or argument == "-":
            return None
        if argument.startswith("--"):
            return None
        if not argument.startswith("-"):
            script = Path(argument)
            module = TARGET_CONSOLE_SCRIPTS.get(script.name)
            return (module, script) if module is not None else None

        options = argument[1:]
        if not options:
            return None
        option_index = 0
        while option_index < len(options):
            option = options[option_index]
            remainder = options[option_index + 1 :]
            if option in PYTHON_EXIT_OPTIONS or option == "c":
                return None
            if option == "m":
                if remainder:
                    module = remainder
                elif index + 1 < len(arguments):
                    module = arguments[index + 1]
                else:
                    return None
                return (module, None) if module in TARGET_MODULES else None
            if option in PYTHON_VALUE_OPTIONS:
                if not remainder:
                    if index + 1 >= len(arguments):
                        return None
                    index += 1
                break
            if option not in PYTHON_FLAG_OPTIONS:
                return None
            option_index += 1
        index += 1
    return None


def _source_module(
    arguments: Sequence[str],
    *,
    venv: Path,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    invocation = _target_invocation(arguments)
    if invocation is None:
        return None
    module, console_script = invocation

    executable = Path(arguments[0])
    expected_venv = _normalized_path(venv)
    if executable.is_absolute():
        executable = _normalized_path(executable)
        if executable.parent != expected_venv / "bin":
            return None
    elif executable.parent != Path("."):
        if cwd is None:
            raise ProcessInspectionError(
                "cannot resolve a relative Python executable without process cwd"
            )
        executable = _normalized_path(cwd / executable)
        if executable.parent != expected_venv / "bin":
            return None
    else:
        virtual_env = (environment or {}).get("VIRTUAL_ENV")
        if not virtual_env or _normalized_path(Path(virtual_env)) != expected_venv:
            return None

    if console_script is not None:
        if console_script.is_absolute():
            console_script = _normalized_path(console_script)
        else:
            if cwd is None:
                raise ProcessInspectionError(
                    "cannot resolve a relative console script without process cwd"
                )
            console_script = _normalized_path(cwd / console_script)
        if console_script.parent != expected_venv / "bin":
            return None
    return module


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
        invocation = _target_invocation(arguments)
        if invocation is None:
            continue

        executable = Path(arguments[0])
        console_script = invocation[1]
        cwd = None
        environment = None
        try:
            if (
                not executable.is_absolute()
                and executable.parent != Path(".")
            ) or (console_script is not None and not console_script.is_absolute()):
                cwd = (process_dir / "cwd").resolve(strict=True)
            if not executable.is_absolute() and executable.parent == Path("."):
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

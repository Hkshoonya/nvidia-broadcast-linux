#!/usr/bin/env python3
# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
"""Build and check opt-in CPython 3.15 CPU runtime wheels."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import venv
import zipfile

ORT_COMMIT = "2d924974ef147392ced8409d36bd6d2e7fcc8a74"  # v1.24.4
AV_SOURCE = (
    "av @ https://files.pythonhosted.org/packages/source/a/av/av-16.1.0.tar.gz"
    "#sha256=a094b4fd87a3721dacf02794d3d2c82b8d712c85b9534437e82a8a978c175ffd"
)
SOURCE_VERSIONS = {"av": "16.1.0", "onnxruntime": "1.24.4"}


def compatible_tag(tag: str) -> bool:
    parts = tag.split("-")
    if len(parts) != 3:
        return False
    interpreter, abi, target = parts
    native_abi = interpreter == "cp315" and abi == "cp315"
    stable_abi = (
        re.fullmatch(r"cp3\d+", interpreter) is not None
        and 311 <= int(interpreter[2:]) <= 315
        and abi == "abi3"
    )
    linux = target == "linux_x86_64" or (
        target.startswith("manylinux") and target.endswith("_x86_64")
    )
    return (native_abi or stable_abi) and linux


def require_python315() -> None:
    if (
        platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 15)
        or sysconfig.get_config_var("Py_GIL_DISABLED")
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
    ):
        raise RuntimeError(
            "Experimental runtime requires regular CPython 3.15 on Linux x86_64 (not 3.15t)."
        )


def check_wheelhouse(directory: Path) -> list[Path]:
    """Check local wheel metadata without importing pip or third-party modules.

    pip still validates platform compatibility and the complete dependency
    graph in the candidate environment before the existing inference probes.
    """
    directory = directory.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Wheelhouse is not a directory: {directory}")
    selected = []
    for name, version in SOURCE_VERSIONS.items():
        matches = []
        for path in directory.glob(f"{name}-{version}-*.whl"):
            try:
                with zipfile.ZipFile(path) as archive:
                    prefix = f"{name}-{version}.dist-info/"
                    metadata = Parser().parsestr(
                        archive.read(prefix + "METADATA").decode()
                    )
                    wheel = Parser().parsestr(archive.read(prefix + "WHEEL").decode())
                    tags = wheel.get_all("Tag", [])
                    if metadata["Name"] != name or metadata["Version"] != version:
                        continue
                    if any(compatible_tag(tag) for tag in tags):
                        matches.append(path)
            except (OSError, KeyError, UnicodeError, zipfile.BadZipFile) as error:
                raise RuntimeError(f"Invalid runtime wheel {path}: {error}") from error
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one CPython 3.15 Linux x86_64 wheel for {name}=={version} "
                f"in {directory}, found {len(matches)}. "
                "Build it with python3.15 scripts/python315_runtime.py build; see docs/PYTHON315.md."
            )
        selected.append(matches[0])
    return selected


def run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build(build_dir: Path, wheelhouse: Path, jobs: int) -> None:
    """Build the existing pinned dependencies, keeping the system Python intact."""
    for tool in ("git", "cmake", "c++", "pkg-config"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Missing build tool: {tool}; see docs/PYTHON315.md.")
    try:
        codec_version = subprocess.check_output(
            ["pkg-config", "--modversion", "libavcodec"], text=True
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Install FFmpeg 8 development headers; see docs/PYTHON315.md."
        ) from error
    if codec_version.split(".")[0] != "62":
        raise RuntimeError(
            f"AV 16 needs FFmpeg 8 (libavcodec 62); found {codec_version}."
        )

    build_dir.mkdir(parents=True, exist_ok=True)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    environment = build_dir / "venv"
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = str(environment / "bin/python")
    run(
        python,
        "-c",
        "import sys, sysconfig; assert sys.version_info[:2] == (3, 15) "
        "and not sysconfig.get_config_var('Py_GIL_DISABLED')",
    )
    run(
        python,
        "-m",
        "pip",
        "install",
        "pip>=26.2.2",
        "setuptools>=84.1.0",
        "wheel",
        "numpy",
        "packaging",
        "sympy",
        "flatbuffers",
        "protobuf>=6.33.5,<7",
        "onnx>=1.22.0",
    )
    run(
        python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-binary=av",
        "--wheel-dir",
        str(wheelhouse),
        AV_SOURCE,
    )

    source = build_dir / "onnxruntime"
    if not source.exists():
        run(
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "v1.24.4",
            "https://github.com/microsoft/onnxruntime.git",
            str(source),
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if commit != ORT_COMMIT:
        raise RuntimeError(
            f"Expected ONNX Runtime source {ORT_COMMIT}, found {commit}."
        )
    run("git", "diff", "--exit-code", "HEAD", cwd=source)
    # CMake fetches the CPU dependencies using deps.txt's hashes; avoid the
    # unrelated Emscripten/fuzzer Git submodule histories.
    run(
        python,
        "tools/ci_build/build.py",
        "--config",
        "Release",
        "--update",
        "--build",
        "--build_wheel",
        "--skip_tests",
        "--skip_submodule_sync",
        "--compile_no_warning_as_error",
        "--parallel",
        str(jobs),
        "--build_dir",
        str(build_dir / "ort-build"),
        "--cmake_extra_defines",
        "onnxruntime_BUILD_UNIT_TESTS=OFF",
        "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER",
        cwd=source,
    )
    wheels = list(
        (build_dir / "ort-build/Release/dist").glob("onnxruntime-1.24.4-*.whl")
    )
    if len(wheels) != 1:
        raise RuntimeError("ONNX Runtime did not produce exactly one 1.24.4 wheel.")
    shutil.copy2(wheels[0], wheelhouse / wheels[0].name)
    selected = check_wheelhouse(wheelhouse)
    # Run actual native imports in the build venv before handing wheels to the installer.
    run(
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        *(str(path) for path in selected),
    )
    run(
        python,
        "-c",
        """
import av.option
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
tensor = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
graph = helper.make_graph([helper.make_node('Add', ['x', 'x'], ['y'])],
                          'python315-smoke', [tensor('x')], [tensor('y')])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
value = np.array([3.15], dtype=np.float32)
np.testing.assert_array_equal(session.run(None, {'x': value})[0], value * 2)
print('AV 16 import and ONNX Runtime CPU inference passed on Python 3.15')
""",
    )
    print(
        f"Built experimental CPU wheels in {wheelhouse}. See docs/PYTHON315.md to install and run inference validation."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "check"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--wheelhouse", type=Path, default=Path("build/python315-wheels")
        )
        if command == "build":
            subparser.add_argument(
                "--build-dir", type=Path, default=Path("build/python315")
            )
            subparser.add_argument("--jobs", type=int, default=2)
    arguments = parser.parse_args()
    try:
        require_python315()
        if arguments.command == "check":
            check_wheelhouse(arguments.wheelhouse)
        else:
            if arguments.jobs < 1:
                parser.error("--jobs must be positive")
            build(
                arguments.build_dir.resolve(),
                arguments.wheelhouse.resolve(),
                arguments.jobs,
            )
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

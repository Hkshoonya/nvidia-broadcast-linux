#!/usr/bin/env python3
"""Install one source runtime variant, then validate its ownership."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Mapping


def run_pip(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", *arguments],
        check=True,
    )


def runtime_owner_inventory(project: Path) -> dict[str, tuple[str, ...]]:
    source_root = str(project / "src")
    added_source_root = source_root not in sys.path
    if added_source_root:
        sys.path.insert(0, source_root)

    try:
        from nvbroadcast.runtime.variants import current_distribution_inventory
    finally:
        if added_source_root:
            sys.path.remove(source_root)

    return current_distribution_inventory()


def selected_runtime_owner(project: Path, variant: str) -> str:
    source_root = str(project / "src")
    added_source_root = source_root not in sys.path
    if added_source_root:
        sys.path.insert(0, source_root)

    try:
        from nvbroadcast.runtime.variants import RUNTIME_CONTRACTS, RuntimeVariant
    finally:
        if added_source_root:
            sys.path.remove(source_root)

    return RUNTIME_CONTRACTS[RuntimeVariant(variant)].distribution


def transition_problem(
    inventory: Mapping[str, tuple[str, ...]], selected_owner: str
) -> str | None:
    if not inventory:
        return None
    if set(inventory) == {selected_owner} and len(inventory[selected_owner]) == 1:
        return None

    found = ", ".join(
        f"{name} ({', '.join(versions)})"
        for name, versions in sorted(inventory.items())
    )
    return (
        f"selected owner is {selected_owner}, but environment contains {found}"
    )


def recovery_guidance(source_venv: Path | None) -> str:
    if source_venv is not None:
        return (
            "Stop NVBroadcast and the virtual-camera service, remove "
            f"{source_venv}, then recreate it with the requested setup target."
        )
    return (
        "Stop NVBroadcast and the virtual-camera service, recreate the target "
        "virtual environment, then rerun the installer."
    )


def preflight_runtime_owner(
    project: Path, variant: str, source_venv: Path | None
) -> None:
    selected_owner = selected_runtime_owner(project, variant)
    problem = transition_problem(runtime_owner_inventory(project), selected_owner)
    if problem is not None:
        raise RuntimeError(
            "Refusing runtime-owner transition before modifying the environment: "
            f"{problem}. {recovery_guidance(source_venv)}"
        )


def guard_source_environment(project: Path, source_venv: Path) -> None:
    guard = project / "scripts" / "check_source_venv_processes.py"
    if not guard.is_file():
        raise RuntimeError(
            f"Cannot guard source environment: missing process guard {guard}."
        )

    result = subprocess.run(
        [sys.executable, str(guard), "--venv", str(source_venv)],
        check=False,
    )
    if result.returncode == 1:
        raise RuntimeError(
            "Source environment is in use. Stop NVBroadcast and the "
            "virtual-camera service, then rerun the setup target."
        )
    if result.returncode != 0:
        raise RuntimeError(
            "Cannot verify that the source environment is idle. Resolve the "
            "process-inspection error above, then rerun the setup target."
        )


def preflight_runtime_install(
    project: Path, variant: str, source_venv: Path | None
) -> None:
    preflight_runtime_owner(project, variant, source_venv)
    if source_venv is not None:
        guard_source_environment(project, source_venv)


def validate_selected_owner(project: Path, variant: str) -> None:
    selected_owner = selected_runtime_owner(project, variant)
    inventory = runtime_owner_inventory(project)
    if set(inventory) != {selected_owner} or len(inventory[selected_owner]) != 1:
        found = ", ".join(
            f"{name} ({', '.join(versions)})"
            for name, versions in sorted(inventory.items())
        ) or "none"
        raise RuntimeError(
            "Runtime ownership validation failed after installation: expected "
            f"exactly one {selected_owner} distribution, found {found}. "
            f"{recovery_guidance(None)}"
        )


def validate_meeting_dependencies(variant: str, meeting_backends: str) -> None:
    from nvbroadcast.runtime.artifact import ArtifactEnvironment
    from nvbroadcast.runtime.variants import FASTER_WHISPER_VERSION

    environment = ArtifactEnvironment.current()
    roots = {"nvbroadcast", "faster-whisper"}
    if meeting_backends == "all" and sys.version_info < (3, 14):
        roots.add("openai-whisper")
    problems = environment.dependency_closure_problems(
        {"onnxruntime": "onnxruntime-gpu"} if variant == "cuda" else None,
        roots=roots,
    )
    backend_versions = environment.installed.get("faster-whisper", ())
    if backend_versions != (FASTER_WHISPER_VERSION,):
        found = ", ".join(backend_versions) if backend_versions else "none"
        problems.append(
            f"faster-whisper must be {FASTER_WHISPER_VERSION}, found {found}"
        )
    if problems:
        details = "\n".join(f"- {problem}" for problem in sorted(set(problems)))
        raise RuntimeError(f"Meeting backend dependency check failed:\n{details}")


def install(
    project: Path,
    variant: str,
    meeting_backends: str,
    *,
    development: bool = False,
    editable: bool = False,
    source_venv: Path | None = None,
) -> None:
    preflight_runtime_install(project, variant, source_venv)

    extras = ["dev"] if development else []
    extras.append(variant)
    if meeting_backends != "none":
        extras.append("meeting-support")
    if meeting_backends == "all":
        extras.append("meeting")

    selected_extras = ",".join(extras)
    install_arguments = ["install", "--upgrade"]
    if editable:
        install_arguments.append("--editable")
    install_arguments.append(f"{project}[{selected_extras}]")
    run_pip(*install_arguments)
    if meeting_backends != "none":
        from nvbroadcast.runtime.variants import FASTER_WHISPER_REQUIREMENT

        # Keep backend installation outside dependency resolution so its
        # onnxruntime requirement cannot replace the selected runtime owner.
        run_pip("install", "--no-deps", FASTER_WHISPER_REQUIREMENT)
        validate_meeting_dependencies(variant, meeting_backends)
    validate_selected_owner(project, variant)
    subprocess.run(
        [sys.executable, "-m", "nvbroadcast.runtime", "--variant", variant],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--editable", action="store_true")
    parser.add_argument("--source-venv", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--meeting-backends",
        choices=("none", "faster", "all"),
        default="none",
        help="install no backend, faster-whisper only, or both supported backends",
    )
    arguments = parser.parse_args()
    project = arguments.project.resolve()
    source_venv = (
        arguments.source_venv.resolve() if arguments.source_venv else None
    )
    try:
        if arguments.preflight_only:
            preflight_runtime_install(project, arguments.variant, source_venv)
        else:
            install(
                project,
                arguments.variant,
                arguments.meeting_backends,
                development=arguments.development,
                editable=arguments.editable,
                source_venv=source_venv,
            )
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

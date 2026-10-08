#!/usr/bin/env python3
"""Select verified local source-install generations without moving a venv.

This is a user-owned source installer, not a signed runtime-pack downloader.
Generations still depend on the selected host Python and distro GTK/GStreamer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import uuid

# The launcher uses -I so Python ignores ambient PYTHONPATH and user packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_venv_processes import find_source_processes  # noqa: E402


STORE = ".nvbroadcast-runtimes"
MARKER = ".nvbroadcast-source-generation.json"
GENERATION = re.compile(r"gen-[0-9a-f]{32}\Z")
MODULES = {"nvbroadcast", "nvbroadcast.vcam_service"}


class SourceRuntimeError(RuntimeError):
    pass


def owned_directory(path: Path, *, private: bool = False) -> None:
    info = path.lstat()
    # A source checkout or legacy venv commonly uses the user's shared-group
    # umask. New selection metadata and generations themselves stay private.
    unsafe_mode = 0o077 if private else 0o002
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & unsafe_mode
    ):
        raise SourceRuntimeError(f"Unsafe runtime directory: {path}")


def read_json(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o022
        ):
            raise SourceRuntimeError(f"Unsafe runtime metadata: {path}")
        data = stream.read(65537)
    if len(data) > 65536:
        raise SourceRuntimeError(f"Runtime metadata is too large: {path}")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise SourceRuntimeError(f"Invalid runtime metadata: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        except OSError as error:
            raise SourceRuntimeError(
                f"Selection may already have changed; directory sync failed at {path}. "
                "Inspect the active runtime before retrying."
            ) from error
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def state_digest(state: dict | None) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def verify_runtime(path: Path, variant: str | None, meeting: str) -> str:
    """Use the candidate interpreter, never inference imported by the installer."""
    verification = r"""
import importlib, importlib.util, json, pathlib, sys
import gi
for name, version in (("Gtk", "4.0"), ("Adw", "1"), ("Gst", "1.0")):
    gi.require_version(name, version)
    importlib.import_module("gi.repository." + name)
for name in ("numpy", "cv2", "onnxruntime", "PIL", "psutil", "onnx", "mediapipe", "av.option", "pyrnnoise.rnnoise"):
    importlib.import_module(name)
import nvbroadcast.app
from nvbroadcast.video.effects import VideoEffects
# Include startup after a saved GPU preference reaches a CPU-only install.
for compositing in ("cpu", "cupy", "gstreamer_gl"):
    effects = VideoEffects(compositing=compositing)
from nvbroadcast.runtime.variants import detect_runtime_variant
# Rollback generations can predate the checker's root_extras API. Load only
# the current checkout's verifier under a private module name; adding src to
# sys.path would incorrectly substitute checkout modules for candidate imports.
spec = importlib.util.spec_from_file_location("_nvbroadcast_dependency_verifier", sys.argv[3])
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)
variant = detect_runtime_variant()
if variant is None or (sys.argv[1] and variant.value != sys.argv[1]):
    raise RuntimeError("Missing, duplicate or unexpected ONNX runtime owner")
roots = {"nvbroadcast"}
extras = {variant.value}
if sys.argv[2] != "none":
    import faster_whisper
    roots.add("faster-whisper")
    extras.add("meeting-support")
    if sys.argv[2] == "all":
        extras.add("meeting")
    if sys.argv[2] == "all" and sys.version_info < (3, 14):
        import whisper
        roots.add("openai-whisper")
problems = checker.ArtifactEnvironment.current().dependency_closure_problems(
    {"onnxruntime": "onnxruntime-gpu"} if variant.value == "cuda" else None,
    roots=roots,
    root_extras={"nvbroadcast": extras},
)
if problems:
    raise RuntimeError("Dependency validation failed: " + "; ".join(problems))
print("NVB_RUNTIME=" + variant.value)
"""
    python = str(path / "bin/python")
    checker = Path(__file__).resolve().parents[1] / "src/nvbroadcast/runtime/artifact.py"
    result = subprocess.run(
        [python, "-I", "-c", verification, variant or "", meeting, str(checker)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    selected = next(
        (
            line.removeprefix("NVB_RUNTIME=")
            for line in result.stdout.splitlines()
            if line.startswith("NVB_RUNTIME=")
        ),
        None,
    )
    if selected not in {"cpu", "cuda"}:
        raise SourceRuntimeError("Runtime verification returned no variant")
    subprocess.run(
        [python, "-I", "-m", "nvbroadcast.runtime", "--variant", selected],
        check=True,
        stdout=sys.stderr,
        timeout=300,
    )
    return selected


class SourceRuntimeStore:
    def __init__(self, project: Path):
        self.project = project.resolve(strict=True)
        owned_directory(self.project)
        if any(
            self.project == root or root in self.project.parents
            for root in map(Path, ("/opt", "/snap", "/app", "/nix/store"))
        ):
            raise SourceRuntimeError(
                "Package-owned runtimes cannot use source activation"
            )
        self.root = self.project / STORE
        self.state_file = self.root / "selection.json"

    def initialize(self) -> None:
        self.root.mkdir(mode=0o700, exist_ok=True)
        owned_directory(self.root, private=True)

    @contextmanager
    def locked(self):
        self.initialize()
        fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise SourceRuntimeError("Unsafe runtime lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)

    def path(self, entry: dict) -> Path:
        name = entry.get("path")
        if name == "legacy":
            path = self.project / ".venv"
        elif isinstance(name, str) and GENERATION.fullmatch(name):
            path = self.root / name
        else:
            raise SourceRuntimeError("Invalid runtime selection path")
        if entry.get("variant") not in {None, "cpu", "cuda"}:
            raise SourceRuntimeError("Invalid runtime variant")
        if entry.get("meeting", "none") not in {"none", "faster", "all"}:
            raise SourceRuntimeError("Invalid meeting runtime selection")
        owned_directory(path)
        return path

    def state(self) -> dict | None:
        if not self.root.exists() and not self.root.is_symlink():
            return None
        owned_directory(self.root, private=True)
        try:
            state = read_json(self.state_file)
        except FileNotFoundError:
            return None
        if state.get("schema") != 1 or not isinstance(state.get("active"), dict):
            raise SourceRuntimeError("Invalid runtime selection")
        self.path(state["active"])
        previous = state.get("previous")
        if previous is not None:
            if not isinstance(previous, dict):
                raise SourceRuntimeError("Invalid previous runtime selection")
            self.path(previous)
        return state

    def active(self) -> Path:
        state = self.state()
        return self.path(state["active"]) if state else self.project / ".venv"

    def features(self) -> dict:
        """Carry optional features forward without importing their ML runtimes."""
        python = self.active() / "bin/python"
        if not python.exists():
            if self.state():
                raise SourceRuntimeError(
                    "Selected interpreter is missing; cannot inspect installed features"
                )
            return {"meeting": "none", "tensorrt": False}
        probe = """
from importlib.metadata import distributions
import json
names = {d.metadata.get("Name", "").lower().replace("_", "-") for d in distributions()}
meeting = "all" if "openai-whisper" in names else "faster" if "faster-whisper" in names else "none"
print(json.dumps({"meeting": meeting, "tensorrt": "tensorrt-cu12-libs" in names}))
"""
        result = subprocess.run(
            [str(python), "-I", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    def guard(self) -> None:
        paths = {self.active(), self.project / ".venv"}
        for path in paths:
            if find_source_processes(path):
                raise SourceRuntimeError(
                    "Stop NVBroadcast and any audio or virtual-camera service, "
                    "then retry source activation."
                )

    def prepare(self) -> Path:
        with self.locked():
            state = self.state()
            candidate = self.root / f"gen-{uuid.uuid4().hex}"
            candidate.mkdir(mode=0o700)
            write_json(candidate / MARKER, {"expected_selection": state_digest(state)})
            return candidate

    def activate(self, candidate: Path, variant: str, meeting: str) -> None:
        entry = {"path": candidate.name, "variant": variant, "meeting": meeting}
        if candidate.absolute() != self.root / candidate.name:
            raise SourceRuntimeError("Candidate is outside this source runtime store")
        with self.locked():
            self.path(entry)
            marker = read_json(candidate / MARKER)
            state = self.state()
            if marker.get("expected_selection") != state_digest(state):
                raise SourceRuntimeError(
                    "Runtime selection changed during installation; retry"
                )
            self.guard()
            verify_runtime(candidate, variant, meeting)
            previous = state["active"] if state else None
            if previous is None and (self.project / ".venv/bin/python").is_file():
                previous = {"path": "legacy", "variant": None, "meeting": "none"}
                self.path(previous)
            self.guard()
            write_json(
                self.state_file, {"schema": 1, "active": entry, "previous": previous}
            )

    def rollback(self) -> None:
        with self.locked():
            state = self.state()
            if not state or not state.get("previous"):
                raise SourceRuntimeError("No previous source runtime is available")
            self.guard()
            previous = dict(state["previous"])
            previous["variant"] = verify_runtime(
                self.path(previous),
                previous.get("variant"),
                previous.get("meeting", "none"),
            )
            self.guard()
            write_json(
                self.state_file,
                {"schema": 1, "active": previous, "previous": state["active"]},
            )

    def discard(self, candidate: Path) -> None:
        with self.locked():
            entry = {"path": candidate.name}
            if candidate.absolute() != self.path(entry):
                raise SourceRuntimeError(
                    "Candidate is outside this source runtime store"
                )
            read_json(candidate / MARKER)
            state = self.state()
            if state and any(
                item and item["path"] == candidate.name
                for item in (state["active"], state.get("previous"))
            ):
                raise SourceRuntimeError(
                    "Cannot discard a selected or previous runtime"
                )
            shutil.rmtree(candidate)

    def remove(self) -> None:
        with self.locked():
            self.guard()
            generations = [
                path for path in self.root.iterdir() if GENERATION.fullmatch(path.name)
            ]
            for path in generations:
                owned_directory(path)
                if find_source_processes(path):
                    raise SourceRuntimeError(f"Source runtime is still in use: {path}")
            for path in generations:
                shutil.rmtree(path)
            self.state_file.unlink(missing_ok=True)

    def launch(self, module: str, arguments: list[str]) -> None:
        if module not in MODULES:
            raise SourceRuntimeError("Unsupported source launcher module")
        python = self.active() / "bin/python"
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.execv(str(python), [str(python), "-I", "-m", module, *arguments])

    def launchers(self, prefix: Path) -> None:
        """Legacy installs keep launching .venv until the selection commit succeeds."""
        destination = prefix / "bin"
        destination.mkdir(parents=True, exist_ok=True)
        for name, module in (
            ("nvbroadcast", "nvbroadcast"),
            ("nvbroadcast-vcam", "nvbroadcast.vcam_service"),
        ):
            interpreter = os.path.realpath(
                getattr(sys, "_base_executable", sys.executable)
            )
            command = [
                interpreter,
                "-I",
                str(Path(__file__).resolve()),
                "--project",
                str(self.project),
                "launch",
                module,
            ]
            temporary = destination / f".{name}-{uuid.uuid4().hex}"
            try:
                temporary.write_text(
                    "#!/usr/bin/env bash\nexec " + shlex.join(command) + ' "$@"\n'
                )
                temporary.chmod(0o755)
                os.replace(temporary, destination / name)
            finally:
                temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("active", "prepare", "rollback", "guard", "remove", "features"):
        commands.add_parser(name)
    commands.add_parser("discard").add_argument("candidate", type=Path)
    activate = commands.add_parser("activate")
    activate.add_argument("candidate", type=Path)
    activate.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    activate.add_argument(
        "--meeting", choices=("none", "faster", "all"), default="none"
    )
    launch = commands.add_parser("launch")
    launch.add_argument("module", choices=sorted(MODULES))
    launch.add_argument("arguments", nargs=argparse.REMAINDER)
    launchers = commands.add_parser("launchers")
    launchers.add_argument("prefix", type=Path)
    options = parser.parse_args()
    try:
        store = SourceRuntimeStore(options.project)
        if options.command == "active":
            print(store.active())
        elif options.command == "features":
            features = store.features()
            print(features["meeting"] + "\t" + str(features["tensorrt"]).lower())
        elif options.command == "prepare":
            print(store.prepare())
        elif options.command == "activate":
            store.activate(options.candidate, options.variant, options.meeting)
        elif options.command == "rollback":
            store.rollback()
            print(f"Selected previous verified runtime: {store.active()}")
        elif options.command == "guard":
            store.guard()
        elif options.command == "remove":
            store.remove()
        elif options.command == "discard":
            store.discard(options.candidate)
        elif options.command == "launchers":
            store.launchers(options.prefix)
        elif options.command == "launch":
            store.launch(options.module, options.arguments)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(
                f"ERROR: Runtime verification failed (exit {error.returncode}).",
                file=sys.stderr,
            )
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        else:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

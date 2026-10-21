#!/usr/bin/env python3
"""Build/install the pinned HUSH worker and this NV Broadcast fork on Linux.

Run from a clone of fixtse/nvidia-broadcast-linux, branch feature/hush-denoiser.
Provision the Maxine SDK/model with HUSH first; this script never downloads them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib

HUSH_REVISION = "c8a2a171ddcd702568f604170a315c17de47fff3"
HUSH_REPOSITORY = "https://github.com/fixtse/hush.git"
PROJECT = Path(__file__).resolve().parents[1]


def run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def output(*args: str) -> str:
    return run(*args, capture_output=True, text=True).stdout.strip()


def atomic_write(path: Path, content: str, mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def worker_launcher(binary: Path, sdk: Path, model: Path, gpu: str) -> str:
    libraries = ":".join(str(sdk / part) for part in (
        "nvafx/lib", "external/cuda/lib", "features/denoiser/lib"))
    return "\n".join((
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"export NVAFX_MODEL={shlex.quote(str(model))}",
        "export NVAFX_VERSION=2",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}",
        # Keep Maxine's CUDA/TensorRT libraries out of the Python app process.
        f'export LD_LIBRARY_PATH={shlex.quote(libraries)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}',
        f'exec {shlex.quote(str(binary))} "$@"', ""))


def stop_hush() -> None:
    # Do not uninstall packages or remove SDKs, models, or user settings.
    for service in (("hush.service", "hushd.service") if shutil.which("systemctl") else ()):
        state = subprocess.run(["systemctl", "--user", "show", service,
                                "--property=LoadState", "--value"],
                               capture_output=True, text=True)
        if state.returncode == 0 and state.stdout.strip() == "loaded":
            run("systemctl", "--user", "disable", "--now", service)
    for name in ("hush", "hushd"):
        result = subprocess.run(["pgrep", "-u", str(os.getuid()), "-x", name],
                                capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"Cannot inspect running {name} processes")
        for value in result.stdout.split():
            try:
                os.kill(int(value), signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(1)
    for name in ("hush", "hushd"):
        result = subprocess.run(["pgrep", "-u", str(os.getuid()), "-x", name],
                                capture_output=True, text=True)
        if result.returncode == 0:
            raise RuntimeError(f"{name} is still running; close it and retry")

    if shutil.which("pactl"):
        # PulseAudio/PipeWire JSON module listings may omit the module ID.
        for line in output("pactl", "list", "short", "modules").splitlines():
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            identifier, name, argument = fields[:3]
            if name == "module-null-sink" and "sink_name=HUSH" in shlex.split(argument):
                run("pactl", "unload-module", identifier)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, help="Maxine SDK root (defaults to HUSH's installed SDK)")
    parser.add_argument("--model", type=Path, help="trusted v2 model matching the selected GPU")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--python", default=sys.executable, help="desktop-capable Python for install.sh")
    parser.add_argument("--runtime", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse-runtime", action="store_true",
                        help="upgrade only the app in an existing verified source runtime, in place")
    parser.add_argument("--stop-hush", action="store_true",
                        help="stop the old GUI/daemon and disable its known user services")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64" or os.getuid() == 0:
        raise RuntimeError("Run as your desktop user on Linux x86_64")
    for command in ("git", "cargo", "pkg-config", "nvidia-smi", "pgrep"):
        if not shutil.which(command):
            raise RuntimeError(f"Missing prerequisite: {command}; see docs/HUSH.md")
    run("pkg-config", "--exists", "libpipewire-0.3")
    data = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    sdk = (args.sdk or Path(os.getenv("NVAFX_SDK", data / "hush/sdk/2.1.0"))).expanduser().resolve()
    if not (sdk / "nvafx/lib/libnv_audiofx.so.2").is_file():
        raise RuntimeError("Maxine SDK missing. Complete HUSH setup or pass --sdk.")
    gpus = output("nvidia-smi", "--query-gpu=uuid,compute_cap", "--format=csv,noheader").splitlines()
    if not 0 <= args.gpu_index < len(gpus):
        raise RuntimeError("Invalid --gpu-index")
    gpu, capability = (part.strip() for part in gpus[args.gpu_index].split(","))
    sm = capability.replace(".", "")
    if sm not in {"75", "80", "86", "89", "90", "100", "120"}:
        raise RuntimeError(f"Maxine does not support sm_{sm}")
    model = (args.model or cache / f"hush/models/sm_{sm}/denoiser_v2_48k.trtpkg").expanduser().resolve()
    if not model.is_file():
        raise RuntimeError("Matching model missing. Complete HUSH setup or pass --model.")
    config = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "nvbroadcast/config.toml"
    if config.exists():
        tomllib.loads(config.read_text())  # Never silently discard corrupt user settings.
    helper = PROJECT / "scripts/source_runtime.py"
    run(args.python, helper, "--project", PROJECT, "guard")
    source = cache / "nvbroadcast-hush" / HUSH_REVISION
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", HUSH_REPOSITORY, source)
    if output("git", "-C", source, "status", "--porcelain"):
        raise RuntimeError(f"Build checkout has uncommitted edits: {source}")
    run("git", "-C", source, "checkout", "--detach", HUSH_REVISION)
    # Verify automatically selected cached models against the pinned HUSH manifest.
    if args.model is None:
        import re
        manifest = (source / "crates/hush-core/src/model.rs").read_text()
        entry = re.search(rf'sm: {sm},\s*version: 2,\s*sha256: "([a-f0-9]+)",\s*size: ([\d_]+)', manifest)
        if not entry or model.stat().st_size != int(entry[2].replace("_", "")):
            raise RuntimeError("Cached model size does not match the pinned HUSH manifest")
        with model.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry[1]:
            raise RuntimeError("Cached model checksum does not match the pinned HUSH manifest")
    environment = {**os.environ, "NVAFX_SDK": str(sdk)}
    environment.pop("NVAFX_LINK_DIR", None)
    run("cargo", "build", "--locked", "--release", "--target-dir", source / "target",
        "-p", "hush-engine", "--bin", "hush-worker",
        cwd=source, env=environment)
    prefix = Path.home() / ".local"
    installed = prefix / "lib/nvbroadcast-hush" / HUSH_REVISION / "hush-worker"
    installed.parent.mkdir(parents=True, exist_ok=True)
    temporary_binary = installed.with_suffix(".new")
    shutil.copy2(source / "target/release/hush-worker", temporary_binary)
    temporary_binary.chmod(0o755)
    os.replace(temporary_binary, installed)
    launcher_text = worker_launcher(installed, sdk, model, gpu)
    # Validate the exact launcher before changing launchers or preferences.
    with tempfile.TemporaryDirectory(prefix="nvbroadcast-hush-test-") as directory:
        launcher = Path(directory) / "hush-worker"
        atomic_write(launcher, launcher_text)
        if args.reuse_runtime:
            active = Path(output(args.python, helper, "--project", PROJECT, "active"))
            python = active / "bin/python"
            # Verify the existing generation before an explicitly requested in-place upgrade.
            verification = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts'))
from source_runtime import verify_runtime
state = json.loads((Path(sys.argv[1]) / '.nvbroadcast-runtimes/selection.json').read_text())
entry = state['active']
verify_runtime(Path(sys.argv[2]), entry['variant'], entry.get('meeting', 'none'))
"""
            run(args.python, "-c", verification, PROJECT, active)
        else:
            run("bash", PROJECT / "install.sh", "--runtime", args.runtime, "--python", args.python)
            active = Path(output(args.python, helper, "--project", PROJECT, "active"))
            python = active / "bin/python"
        # Test this checkout's adapter against the real GPU before app upgrade.
        smoke = """
import numpy as np
from nvbroadcast.audio.hush import HushDenoiser
worker = HushDenoiser()
try:
    assert worker.initialize(), 'HUSH initialization failed'
    for _ in range(10):
        result = worker.process_chunk(np.zeros(960, dtype=np.float32))
        assert len(result) == 960 and np.isfinite(result).all()
finally:
    worker.cleanup()
print('HUSH GPU worker smoke test passed')
"""
        run(python, "-c", smoke, env={**os.environ, "PYTHONPATH": str(PROJECT / "src"),
            "NVBROADCAST_HUSH_WORKER": str(launcher), "NVAFX_MODEL": str(model)})
    if args.reuse_runtime:
        # Explicit opt-in to an in-place app-only upgrade; dependency owner is untouched.
        run(args.python, helper, "--project", PROJECT, "guard")
        run(python, "-m", "pip", "install", "--no-deps", PROJECT)
    # Confirm the installed app, rather than PYTHONPATH, contains the adapter.
    run(python, "-I", "-c", "from nvbroadcast.audio.hush import HushDenoiser")
    if args.stop_hush:
        stop_hush()
    atomic_write(prefix / "bin/hush-worker", launcher_text)
    # A stable dedicated launcher makes desktop launches independent of shell PATH/env.
    app_launcher = prefix / "bin/nvbroadcast-hush"
    atomic_write(app_launcher, "\n".join((
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"export NVBROADCAST_HUSH_WORKER={shlex.quote(str(prefix / 'bin/hush-worker'))}",
        f"exec {shlex.quote(args.python)} -I {shlex.quote(str(helper))} --project "
        f"{shlex.quote(str(PROJECT))} launch nvbroadcast \"$@\"", "")))
    # Standard NV Broadcast launchers also discover hush-worker via PATH.
    run(args.python, helper, "--project", PROJECT, "launchers", prefix)
    config_program = """
from datetime import datetime, timezone
from pathlib import Path
import shutil, tomllib
from nvbroadcast.core.config import load_config, save_config
from nvbroadcast.core.constants import CONFIG_FILE
if CONFIG_FILE.exists():
    tomllib.loads(CONFIG_FILE.read_text())
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    shutil.copy2(CONFIG_FILE, CONFIG_FILE.with_name('config.toml.pre-hush-' + suffix))
config = load_config()
config.audio.noise_engine = 'hush'
config.audio.noise_removal = True
save_config(config)
"""
    run(python, "-I", "-c", config_program)
    desktop = data / "applications/nvbroadcast-hush.desktop"
    # Desktop Exec has its own quoting rules (not shell quoting).
    escaped_launcher = str(app_launcher).replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%')
    atomic_write(desktop, "\n".join((
        "[Desktop Entry]", "Type=Application", "Name=NV Broadcast (HUSH fork)",
        'Exec="' + escaped_launcher + '"', "Icon=com.doczeus.NVBroadcast",
        "Terminal=false", "Categories=AudioVideo;Audio;Video;", "")), 0o644)
    print(f"Installed. Launch {app_launcher} or 'NV Broadcast (HUSH fork)' in your app menu.")
    print("The existing HUSH package and its SDK/model data were retained.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"HUSH fork installation failed: {error}", file=sys.stderr)
        raise SystemExit(1)

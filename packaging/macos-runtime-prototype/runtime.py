#!/usr/bin/env python3
"""Install/qualify the experimental Mac wheelhouse without dependency resolution.

The supplied Homebrew CPython 3.13/GI/GStreamer ABI remains a prerequisite.
This program contains no downloader and never executes as Installer root.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import zipfile


BREW = Path("/opt/homebrew")
TARGET = "macos-arm64-cp313-cpu"
MINIMUM_MACOS = "14.0"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def tree_digest(root: Path) -> str:
    result = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Redirected package payload: {path}")
        if not path.is_file() or path == root / "macos-runtime-id":
            continue
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        result.update(len(relative).to_bytes(8, "big"))
        result.update(relative)
        result.update(stat.S_IMODE(path.stat().st_mode).to_bytes(4, "big"))
        result.update(len(payload).to_bytes(8, "big"))
        result.update(payload)
    return result.hexdigest()


def wheel_record(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", path.name):
        raise ValueError("Expected a regular wheel with a simple filename")
    with zipfile.ZipFile(path) as archive:
        # pip and other tools legitimately vendor dependencies with nested
        # .dist-info records; only the wheel's top-level record identifies it.
        names = [name for name in archive.namelist()
                 if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("A wheel must have exactly one METADATA")
        info = BytesParser().parsebytes(archive.read(names[0]))
    name, version = info["Name"], info["Version"]
    if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("Invalid wheel project name")
    if not version or not re.fullmatch(r"[A-Za-z0-9.!+_-]+", version):
        raise ValueError("Invalid wheel version")
    return {"name": canonical(name), "version": version,
            "file": path.name, "sha256": digest(path)}


def requirements(packages: list[dict]) -> str:
    return "".join(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n"
                   for item in sorted(packages, key=lambda item: item["name"]))


def validate_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("target") != TARGET
            or manifest.get("minimum_macos") != MINIMUM_MACOS):
        raise ValueError("Unsupported wheelhouse contract")
    expected = manifest["packages"]
    if not expected or len({item["name"] for item in expected}) != len(expected):
        raise ValueError("Missing or duplicate wheel owners")
    directory = bundle / "wheels"
    if directory.is_symlink() or any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError("Wheelhouse may contain only regular wheels")
    actual = sorted((wheel_record(path) for path in directory.iterdir()), key=lambda item: item["name"])
    if sorted(expected, key=lambda item: item["name"]) != actual:
        raise ValueError("Wheelhouse bytes/metadata differ from the manifest")
    owners = {item["name"] for item in actual}
    if owners & {"onnxruntime", "onnxruntime-gpu"} != {"onnxruntime"}:
        raise ValueError("Expected exactly one CPU ONNX Runtime owner")
    if not {"nvbroadcast", "faster-whisper", "pip", "setuptools", "wheel", "packaging"} <= owners:
        raise ValueError("Incomplete application/bootstrap wheelhouse")
    if (bundle / "requirements.txt").read_text() != requirements(actual):
        raise ValueError("Offline requirements differ from the verified wheel inventory")
    return manifest


def native_probe() -> dict:
    if (platform.system(), platform.machine(), sys.version_info[:2]) != ("Darwin", "arm64", (3, 13)):
        raise RuntimeError("Candidate requires macOS arm64 Homebrew CPython 3.13")
    if tuple(map(int, platform.mac_ver()[0].split(".")[:2])) < (14, 0):
        raise RuntimeError("Candidate requires macOS 14 or newer for the pinned PyAV wheel ABI")
    base = Path(sys.base_prefix).resolve()
    if not base.is_relative_to(BREW) or not Path(sys.executable).exists():
        raise RuntimeError("Candidate requires Apple Silicon Homebrew Python")
    import cairo
    import cairo._cairo
    import gi
    import gi._gi
    for namespace, version in (("Gtk", "4.0"), ("Adw", "1"), ("Gst", "1.0"),
                               ("GstApp", "1.0"), ("GstVideo", "1.0")):
        gi.require_version(namespace, version)
    from gi.repository import Adw, Gst, GstApp, GstVideo, Gtk
    Gst.init([])
    native_paths = {"gi": Path(gi.__file__).resolve(), "gi_extension": Path(gi._gi.__file__).resolve(),
                    "cairo": Path(cairo.__file__).resolve(), "cairo_extension": Path(cairo._cairo.__file__).resolve()}
    if any(not path.is_relative_to(BREW) for path in native_paths.values()):
        raise RuntimeError(f"GI/Pycairo must come from the required Homebrew prefix: {native_paths}")
    required = ("avfvideosrc", "osxaudiosrc", "osxaudiosink", "videoconvert", "audioconvert",
                "audioresample", "x264enc", "h264parse", "mp4mux")
    plugins = {}
    for name in required:
        factory = Gst.ElementFactory.find(name)
        if factory is None:
            raise RuntimeError(f"Missing native GStreamer element: {name}")
        plugin = factory.get_plugin()
        plugins[name] = {"version": plugin.get_version(), "file": plugin.get_filename()}
    if Gst.DeviceProviderFactory.find("osxaudiodeviceprovider") is None:
        raise RuntimeError("Missing osxaudiodeviceprovider")
    if not any(Gst.ElementFactory.find(name) for name in ("avenc_aac", "voaacenc")):
        raise RuntimeError("Missing native AAC encoder")
    # `list --versions` reads installed receipts, without formula/API refresh.
    brew = run(BREW / "bin/brew", "list", "--versions", capture_output=True, text=True,
               env={**os.environ, "HOMEBREW_NO_AUTO_UPDATE": "1"}).stdout
    return {"python": sys.version, "base_prefix": str(base), "platform": platform.platform(),
            "bindings": {key: {"file": str(path), "sha256": digest(path)}
                         for key, path in native_paths.items()},
            "gi_version": gi.__version__, "cairo_version": cairo.version,
            "gtk": [Gtk.get_major_version(), Gtk.get_minor_version(), Gtk.get_micro_version()],
            "adwaita": [Adw.get_major_version(), Adw.get_minor_version(), Adw.get_micro_version()],
            "gstreamer": Gst.version_string(), "plugins": plugins,
            "homebrew_installed_versions": brew.splitlines()}


def pip_install(python: Path, bundle: Path, requirement_file: Path) -> None:
    # Isolated ignores user config and PIP_* injection; no deps forbids resolver
    # work, hash-checking binds every wheel, and no-index prohibits index use.
    run(python, "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
        "--require-hashes", "--only-binary=:all:", "--force-reinstall", "--no-cache-dir",
        "--disable-pip-version-check", "--find-links", bundle / "wheels", "-r", requirement_file)


def pinned_distribution_problems(packages: list[dict], prefix: Path, search_paths: list[str]) -> list[str]:
    prefix = prefix.resolve()
    local_paths = sorted({str(Path(path).resolve()) for path in search_paths
                          if path and Path(path).resolve().is_relative_to(prefix)})
    owners = {}
    for distribution in metadata.distributions(path=local_paths):
        name = distribution.metadata.get("Name")
        if name:
            owners.setdefault(canonical(name), []).append(distribution.version)
    return [f"Expected one local {item['name']}=={item['version']} owner, found {owners.get(item['name'], [])}"
            for item in packages if owners.get(item["name"], []) != [item["version"]]]


def verify_runtime(bundle: Path, runtime: Path) -> dict:
    manifest = validate_bundle(bundle)
    if Path(sys.prefix).resolve() != (runtime / ".venv").resolve():
        raise RuntimeError("Qualification must use the candidate's private environment")
    from nvbroadcast.runtime.artifact import ArtifactEnvironment
    from nvbroadcast.runtime.variants import validate_current_runtime
    problems = validate_current_runtime("cpu")
    problems += ArtifactEnvironment.current().dependency_closure_problems(
        roots=("nvbroadcast", "faster-whisper"), root_extras={"nvbroadcast": ("cpu", "meeting-support")})
    problems += pinned_distribution_problems(manifest["packages"], Path(sys.prefix), sys.path)
    for item in manifest["packages"]:
        dist = metadata.distribution(item["name"])
        if dist.version != item["version"] or not Path(dist.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()):
            problems.append(f"Wheel not installed locally at its pinned version: {item['name']}")
    if problems:
        raise RuntimeError("Invalid runtime closure: " + "; ".join(problems))
    module_records = {}
    for name in ("nvbroadcast", "numpy", "PIL", "cv2", "mediapipe", "av", "onnx", "onnxruntime",
                 "pyvirtualcam", "scipy", "psutil", "ctranslate2", "faster_whisper", "tokenizers", "soundfile"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"Application dependency imported outside the private environment: {name}: {path}")
        module_records[name] = {"file": str(path), "version": getattr(module, "__version__", None)}
    native = native_probe()
    run(sys.executable, "-m", "pip", "--isolated", "check")
    result_path = runtime / "generated-media.json"
    run(sys.executable, bundle / "validate_macos_runtime.py", "--output", result_path)
    return {"manifest_sha256": digest(bundle / "manifest.json"), "native": native,
            "media": json.loads(result_path.read_text()), "dependency_closure": "pass",
            "distribution_inventory": manifest["packages"], "application_module_imports": module_records}


def install(bundle: Path, runtime: Path) -> None:
    if os.geteuid() == 0:
        raise RuntimeError("Run setup as the logged-in user, without sudo")
    native_probe()  # Fail before creating anything when the Homebrew ABI is wrong.
    validate_bundle(bundle)
    identity = (bundle / "macos-runtime-id").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", identity) or tree_digest(bundle) != identity:
        raise ValueError("Package runtime identity does not match its payload")
    for parent in [runtime, *runtime.parents]:
        if parent.is_symlink():
            raise RuntimeError(f"Refusing redirected user runtime path: {parent}")
    ready = runtime / "runtime-ready"
    if runtime.exists():
        if not ready.is_file() or ready.read_text().strip() != identity:
            raise RuntimeError(f"Existing candidate runtime is incomplete; review it before retrying: {runtime}")
    else:
        runtime.mkdir(parents=True, mode=0o700)
        run(sys.executable, "-m", "venv", "--system-site-packages", runtime / ".venv")
        pip_install(runtime / ".venv/bin/python", bundle, bundle / "requirements.txt")
    run(runtime / ".venv/bin/python", bundle / "runtime.py", "verify",
        "--bundle", bundle, "--runtime", runtime)
    ready.write_text(identity + "\n")
    print(f"Offline candidate runtime ready: {runtime}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "verify", "probe"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "probe":
        args.output.write_text(json.dumps(native_probe(), indent=2) + "\n")
    elif args.command == "install":
        install(args.bundle.resolve(strict=True), args.runtime.absolute())
    else:
        result = verify_runtime(args.bundle.resolve(strict=True), args.runtime.resolve(strict=True))
        (args.runtime / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

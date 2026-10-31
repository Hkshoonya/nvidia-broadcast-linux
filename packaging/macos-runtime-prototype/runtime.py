#!/usr/bin/env python3
"""Install/qualify the experimental Mac wheelhouse without dependency resolution.

Homebrew CPython 3.13 and native libraries remain prerequisites. GI/Cairo
bindings are installed from the authenticated private binding wheel inputs.
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
import struct
import subprocess
import sys
import sysconfig
from urllib.parse import urlsplit
import zipfile


BREW = Path("/opt/homebrew")
TARGET = "macos-arm64-cp313-cpu"
MINIMUM_MACOS = "14.0"
MINIMUM_PYTHON = (3, 13, 18)
BINDING_INPUT_FILE = "binding-source-inputs.json"
BINDING_INVENTORY_FILE = "binding-build-inventory.json"
BINDING_MANIFEST_FILE = "native-bindings.json"
BINDING_NAMES = {"pycairo", "pygobject"}


def source_license_path(path: Path) -> bool:
    return (path.name.upper().startswith(("COPYING", "LICENSE", "LICENCE", "NOTICE", "AUTHORS", "COPYRIGHT"))
            or path.name == "REUSE.toml"
            or any(part.upper() in {"LICENSE", "LICENSES", "LICENCE", "LICENCES"} for part in path.parts[:-1]))


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


def _regular_json(path: Path, expected: str) -> dict:
    if (not re.fullmatch(r"[0-9a-f]{64}", expected) or path.is_symlink()
            or not path.is_file() or path.stat().st_size > (1 << 20) or digest(path) != expected):
        raise ValueError(f"Authenticated binding input changed: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Binding input must contain a JSON object")
    return value


def validate_binding_sources(inputs: dict) -> dict[str, dict]:
    if (inputs.get("schema_version") != 1 or inputs.get("target") != TARGET
            or set(inputs) != {"schema_version", "target", "bootstrap", "bindings"}):
        raise ValueError("Unsupported private binding source contract")
    owners = {}
    for group, names in (("bootstrap", {"meson", "meson-python", "pyproject-metadata", "packaging"}),
                         ("bindings", BINDING_NAMES)):
        selected = {}
        for item in inputs[group]:
            if set(item) != {"name", "version", "url", "sha256"}:
                raise ValueError("Binding sources need exact name/version/URL/hash records")
            name = canonical(item["name"])
            url = urlsplit(item["url"])
            if (name not in names or name in selected
                    or not re.fullmatch(r"[A-Za-z0-9.!+_-]+", item["version"])
                    or url.scheme != "https" or url.netloc != "files.pythonhosted.org"
                    or url.query or url.fragment or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("Invalid, duplicate or unpinned private binding source")
            selected[name] = item
        if set(selected) != names:
            raise ValueError("Missing or undeclared private binding source owner")
        if group == "bindings":
            owners = selected
    return owners


def _native_object(data: bytes) -> dict:
    # Only the thin arm64 extensions built by this preparation are admitted.
    if len(data) < 32 or data[:4] != b"\xcf\xfa\xed\xfe":
        raise ValueError("Expected a native arm64 Mach-O binding extension")
    cpu, _, kind, commands, command_bytes, _, _ = struct.unpack_from("<7I", data, 4)
    if cpu != 0x0100000c or kind not in (6, 8) or commands > 4096 or command_bytes > len(data) - 32:
        raise ValueError("Incompatible native binding architecture or Mach-O header")
    position, versions = 32, []
    for _ in range(commands):
        if position + 8 > 32 + command_bytes:
            raise ValueError("Truncated native binding load commands")
        command, size = struct.unpack_from("<2I", data, position)
        if size < 8 or size % 4 or position + size > 32 + command_bytes:
            raise ValueError("Invalid native binding load command")
        if command == 0x32:
            if size < 24 or struct.unpack_from("<I", data, position + 8)[0] != 1:
                raise ValueError("Binding extension must target macOS")
            versions.append(struct.unpack_from("<I", data, position + 12)[0])
        elif command == 0x24:
            if size < 16:
                raise ValueError("Invalid native binding deployment command")
            versions.append(struct.unpack_from("<I", data, position + 8)[0])
        position += size
    if position != 32 + command_bytes or len(versions) != 1:
        raise ValueError("Missing or ambiguous native binding deployment floor")
    version = (versions[0] >> 16, (versions[0] >> 8) & 255, versions[0] & 255)
    if not (11, 0, 0) <= version <= (14, 0, 0):
        raise ValueError("Native binding raises the declared macOS deployment floor")
    return {"architecture": "arm64", "minimum_macos": ".".join(map(str, version))}


def validate_binding_wheel(path: Path, source: dict, source_licenses: dict) -> dict:
    record = wheel_record(path)
    name = canonical(source["name"])
    if record["name"] != name or record["version"] != source["version"] or name not in BINDING_NAMES:
        raise ValueError("Built binding wheel differs from its pinned source owner/version")
    fields = path.name[:-4].split("-")
    if len(fields) != 5 or canonical(fields[0]) != name or fields[1] != source["version"]:
        raise ValueError("Unexpected native binding wheel filename")
    python, abi, platform_tag = fields[-3:]
    platform_match = re.fullmatch(r"macosx_([0-9]+)_([0-9]+)_arm64", platform_tag)
    python_match = re.fullmatch(r"cp3([0-9]+)", python)
    if (not platform_match or not python_match
            or (abi == "cp313" and python != "cp313")
            or abi not in ("cp313", "abi3")
            or int(python_match[1]) > 13
            or not (11, 0) <= tuple(map(int, platform_match.groups())) <= (14, 0)):
        raise ValueError("Binding wheel has incompatible Python/architecture/deployment tags")
    package = "cairo" if name == "pycairo" else "gi"
    module_files, licenses, objects = {}, {}, {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) > 10000 or len({item.filename for item in members}) != len(members)
                or sum(item.file_size for item in members) > (64 << 20)):
            raise ValueError("Invalid or oversized binding wheel inventory")
        wheel_metadata = [item.filename for item in members
                          if item.filename.count("/") == 1 and item.filename.endswith(".dist-info/WHEEL")]
        if len(wheel_metadata) != 1:
            raise ValueError("Binding wheel must have one WHEEL metadata record")
        wheel_info = BytesParser().parsebytes(archive.read(wheel_metadata[0]))
        if (wheel_info.get("Wheel-Version") != "1.0" or wheel_info.get("Root-Is-Purelib", "").lower() != "false"
                or wheel_info.get_all("Tag", []) != ["-".join(fields[-3:])]):
            raise ValueError("Binding wheel tags differ from its native filename")
        for member in members:
            relative = Path(member.filename)
            mode = member.external_attr >> 16
            if (relative.is_absolute() or ".." in relative.parts or member.flag_bits & 1
                    or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
                raise ValueError("Unsafe private binding wheel member")
            if member.is_dir():
                continue
            payload = archive.read(member)
            if member.filename.startswith(package + "/") and relative.suffix in (".py", ".so"):
                module_files[member.filename] = hashlib.sha256(payload).hexdigest()
            if relative.suffix == ".so":
                if not member.filename.startswith(package + "/"):
                    raise ValueError("Unexpected native library bundled with binding")
                objects[member.filename] = _native_object(payload)
                if tuple(map(int, objects[member.filename]["minimum_macos"].split("."))) > (*map(int, platform_match.groups()), 0):
                    raise ValueError("Native binding deployment command exceeds its wheel tag")
            if ".dist-info/" in member.filename and (
                    "/licenses/" in member.filename.lower()
                    or relative.name.upper().startswith(("LICENSE", "COPYING"))):
                licenses[member.filename] = hashlib.sha256(payload).hexdigest()
        required = ("_cairo",) if name == "pycairo" else ("_gi", "_gi_cairo")
        if any(not any(Path(item).name.startswith(module + ".") for item in objects) for module in required):
            raise ValueError("Binding wheel is missing required native modules")
        if package + "/__init__.py" not in module_files or not licenses:
            raise ValueError("Binding wheel is missing package or licence material")
        if not source_licenses:
            raise ValueError("Missing authenticated source licence inventory")
        for source_path, item in source_licenses.items():
            relative = Path(source_path)
            if (relative.is_absolute() or ".." in relative.parts
                    or not source_license_path(relative)
                    or set(item) != {"file", "sha256"}
                    or ".dist-info/licenses/nvbroadcast-source/" not in item["file"]
                    or not item["file"].endswith("/" + source_path)
                    or licenses.get(item["file"]) != item["sha256"]):
                raise ValueError("Binding wheel does not retain its authenticated source licences")
    return {**record, "size": path.stat().st_size, "module_files": module_files,
            "licenses": licenses, "native_objects": objects, "source_licenses": source_licenses}


def validate_binding_inputs(directory: Path, expected_sha256: str, *, exact_wheels: bool = True) -> dict:
    receipt = _regular_json(directory / BINDING_MANIFEST_FILE, expected_sha256)
    if (set(receipt) != {"schema_version", "target", "source_revision", "source_inputs", "build_inventory", "wheels"}
            or receipt["schema_version"] != 1 or receipt["target"] != TARGET
            or not re.fullmatch(r"[0-9a-f]{40}", receipt["source_revision"])):
        raise ValueError("Unsupported prepared binding input contract")
    for key, filename in (("source_inputs", BINDING_INPUT_FILE), ("build_inventory", BINDING_INVENTORY_FILE)):
        ref = receipt[key]
        if set(ref) != {"file", "sha256"} or ref["file"] != filename:
            raise ValueError("Unexpected binding provenance path")
    sources = _regular_json(directory / BINDING_INPUT_FILE, receipt["source_inputs"]["sha256"])
    source_owners = validate_binding_sources(sources)
    inventory = _regular_json(directory / BINDING_INVENTORY_FILE, receipt["build_inventory"]["sha256"])
    owners, actual = set(), []
    wheel_directory = directory / "wheels"
    if wheel_directory.is_symlink() or not wheel_directory.is_dir():
        raise ValueError("Expected a regular prepared binding wheel directory")
    for item in receipt["wheels"]:
        name = item.get("name")
        filename = item.get("file", "")
        if name not in BINDING_NAMES or name in owners or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename):
            raise ValueError("Invalid, duplicate or undeclared prepared binding wheel")
        owners.add(name)
        record = validate_binding_wheel(wheel_directory / filename, source_owners[name], item.get("source_licenses", {}))
        if record != item:
            raise ValueError("Prepared binding wheel bytes or metadata changed")
        actual.append(record)
    if owners != BINDING_NAMES or (exact_wheels and {item.name for item in wheel_directory.iterdir()} != {item["file"] for item in actual}):
        raise ValueError("Missing or undeclared prepared binding wheel")
    return {"receipt": receipt, "sources": sources, "inventory": inventory, "wheels": actual}


def binding_parameters(records: list[dict]) -> tuple[dict, dict]:
    files, versions = {}, {}
    for record in records:
        if record["name"] in versions or set(files) & set(record["module_files"]):
            raise ValueError("Duplicate private binding owners or module paths")
        files.update(record["module_files"])
        versions[record["name"]] = record["version"]
    if set(versions) != BINDING_NAMES:
        raise ValueError("Both private binding owners are required")
    return files, versions


def validate_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest.get("schema_version") != 2 or manifest.get("target") != TARGET
            or manifest.get("minimum_macos") != MINIMUM_MACOS
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("native_bindings_manifest_sha256", ""))
            or not isinstance(manifest.get("native_bindings"), dict)):
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
    bindings = validate_binding_inputs(bundle, manifest["native_bindings_manifest_sha256"], exact_wheels=False)
    if not BINDING_NAMES <= owners:
        raise ValueError("Private binding wheels are missing from the package inventory")
    if (bindings["receipt"] != manifest["native_bindings"]
            or bindings["receipt"]["source_revision"] != manifest.get("source_revision")):
        raise ValueError("Binding provenance differs from the package manifest")
    return manifest


def validate_python_version(version: tuple[int, ...]) -> None:
    if version[:2] != (3, 13):
        raise RuntimeError("Candidate requires Homebrew CPython 3.13 and matching native bindings")
    if version[:3] < MINIMUM_PYTHON:
        raise RuntimeError("Candidate requires Homebrew CPython 3.13.18 or newer in the 3.13 series; "
                           "update python@3.13 before building or running this candidate")


def base_probe() -> dict:
    validate_python_version(sys.version_info)
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("Candidate requires macOS arm64 Homebrew CPython 3.13")
    if tuple(map(int, platform.mac_ver()[0].split(".")[:2])) < (14, 0):
        raise RuntimeError("Candidate requires macOS 14 or newer for the pinned PyAV wheel ABI")
    base = Path(sys.base_prefix).resolve()
    if not base.is_relative_to(BREW) or not Path(sys.executable).exists():
        raise RuntimeError("Candidate requires Apple Silicon Homebrew Python")
    executable = Path(sys.executable).resolve()
    return {"python": sys.version, "base_prefix": str(base), "platform": platform.platform(),
            "python_executable": {"file": str(executable), "sha256": digest(executable)}}


def verify_binding_modules(paths: dict[str, Path], prefix: Path, site: Path,
                           binding_files: dict | None) -> None:
    prefix, site = prefix.resolve(), site.resolve()
    resolved = {name: path.resolve() for name, path in paths.items()}
    if any(not path.is_relative_to(prefix) for path in resolved.values()):
        raise RuntimeError(f"GI/Pycairo must come from the required private prefix: {resolved}")
    if binding_files is not None:
        for path in resolved.values():
            if not path.is_relative_to(site) or digest(path) != binding_files.get(path.relative_to(site).as_posix()):
                raise RuntimeError(f"Native binding module differs from its packaged wheel: {path}")


def native_probe(expected_prefix: Path | None = None, binding_files: dict | None = None,
                 binding_versions: dict | None = None) -> dict:
    base = base_probe()
    prefix = (expected_prefix or Path(sys.prefix)).resolve()
    if prefix != Path(sys.prefix).resolve() or prefix == Path(sys.base_prefix).resolve():
        raise RuntimeError("Native binding qualification requires the private environment")
    import cairo
    import cairo._cairo
    import gi
    import gi._gi
    gi.require_foreign("cairo")
    import gi._gi_cairo
    for namespace, version in (("Gtk", "4.0"), ("Adw", "1"), ("Gst", "1.0"),
                               ("GstApp", "1.0"), ("GstVideo", "1.0")):
        gi.require_version(namespace, version)
    from gi.repository import Adw, Gst, GstApp, GstVideo, Gtk
    Gst.init([])
    native_paths = {"gi": Path(gi.__file__).resolve(), "gi_extension": Path(gi._gi.__file__).resolve(),
                    "gi_cairo_extension": Path(gi._gi_cairo.__file__).resolve(),
                    "cairo": Path(cairo.__file__).resolve(), "cairo_extension": Path(cairo._cairo.__file__).resolve()}
    verify_binding_modules(native_paths, prefix, Path(sysconfig.get_path("platlib")), binding_files)
    if binding_versions is not None:
        problems = pinned_distribution_problems(
            [{"name": name, "version": version} for name, version in binding_versions.items()], prefix, sys.path)
        if problems:
            raise RuntimeError("Invalid private binding owners: " + "; ".join(problems))
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
    return {**base,
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
    run(python, "-I", "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
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
    validate_python_version(sys.version_info)
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
    binding_files, binding_versions = binding_parameters(manifest["native_bindings"]["wheels"])
    native = native_probe(runtime / ".venv", binding_files, binding_versions)
    run(sys.executable, "-m", "pip", "--isolated", "check")
    result_path = runtime / "generated-media.json"
    run(sys.executable, bundle / "validate_macos_runtime.py", "--output", result_path)
    return {"manifest_sha256": digest(bundle / "manifest.json"), "native": native,
            "media": json.loads(result_path.read_text()), "dependency_closure": "pass",
            "distribution_inventory": manifest["packages"], "application_module_imports": module_records}


def install(bundle: Path, runtime: Path) -> None:
    if os.geteuid() == 0:
        raise RuntimeError("Run setup as the logged-in user, without sudo")
    # Base Python/native-library provisioning is external. The authenticated
    # binding wheels are checked before they enter a fresh private environment.
    base_probe()
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
        run(sys.executable, "-m", "venv", runtime / ".venv")
        pip_install(runtime / ".venv/bin/python", bundle, bundle / "requirements.txt")
    run(runtime / ".venv/bin/python", bundle / "runtime.py", "verify",
        "--bundle", bundle, "--runtime", runtime)
    ready.write_text(identity + "\n")
    print(f"Offline candidate runtime ready: {runtime}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "verify", "probe", "python-policy"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--binding-inputs", type=Path)
    parser.add_argument("--binding-inputs-sha256")
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "python-policy":
        validate_python_version(sys.version_info)
    elif args.command == "probe":
        if bool(args.binding_inputs) != bool(args.binding_inputs_sha256):
            parser.error("binding inputs require their explicit expected SHA-256")
        files = versions = None
        if args.binding_inputs:
            prepared = validate_binding_inputs(args.binding_inputs, args.binding_inputs_sha256)
            files, versions = binding_parameters(prepared["wheels"])
        args.output.write_text(json.dumps(native_probe(args.prefix, files, versions), indent=2) + "\n")
    elif args.command == "install":
        install(args.bundle.resolve(strict=True), args.runtime.absolute())
    else:
        result = verify_runtime(args.bundle.resolve(strict=True), args.runtime.resolve(strict=True))
        (args.runtime / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare explicit private binding wheel inputs; never run on a destination.

Acquisition authenticates committed source/tool inputs. Compilation and native
qualification run with networking denied on the native Mac builder. Consumers
must supply the resulting manifest's expected SHA-256 separately from its cache.
"""
from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
import errno
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys
import tarfile
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import runtime

spec = importlib.util.spec_from_file_location("binding_input_fetch", ROOT / "packaging/runtime-prototype/prepare.py")
prepare_inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_inputs)


def clean_environment(python: Path) -> dict:
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith(("PYTHON", "PIP_", "PKG_CONFIG", "MESON", "CMAKE")) or key in {
            "CC", "CXX", "CFLAGS", "CPPFLAGS", "CXXFLAGS", "LDFLAGS", "SDKROOT",
        }:
            del environment[key]
    environment.update(
        PATH=f"{python.parent}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        MACOSX_DEPLOYMENT_TARGET=runtime.MINIMUM_MACOS,
        PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/opt/libffi/lib/pkgconfig",
        GST_PLUGIN_PATH="/opt/homebrew/lib/gstreamer-1.0",
        GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0",
        CC="/usr/bin/clang", CXX="/usr/bin/clang++",
    )
    return environment


def tool_inputs(pins: dict, sources: dict) -> list[dict]:
    selected = {}
    for item in [*pins["bootstrap"], *sources["bootstrap"]]:
        name = runtime.canonical(item["name"])
        if name in selected and selected[name] != item:
            raise ValueError("Conflicting private binding bootstrap inputs")
        selected[name] = item
    if set(selected) != {"pip", "setuptools", "wheel", "packaging", "meson", "meson-python", "pyproject-metadata"}:
        raise ValueError("Unexpected private binding build tools")
    return sorted(selected.values(), key=lambda item: item["name"])


def retain_source_licenses(wheel: Path, archive: Path, source: dict) -> dict:
    """Retain all upstream licence texts before sealing our generated wheel."""
    prepare_inputs.verify(archive, source)
    licenses = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            if not member.isfile() or not runtime.source_license_path(Path(member.name)):
                continue
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.size > (1 << 20):
                raise ValueError("Unsafe or oversized binding source licence")
            if member.name in licenses:
                raise ValueError("Duplicate binding source licence")
            licenses[member.name] = stream.extractfile(member).read()
    if not licenses or sum(map(len, licenses.values())) > (4 << 20):
        raise ValueError("Missing or oversized authenticated binding source licences")
    with zipfile.ZipFile(wheel) as source_wheel:
        members = source_wheel.infolist()
        if (len(members) > 10000 or len({item.filename for item in members}) != len(members)
                or sum(item.file_size for item in members) > (64 << 20)):
            raise ValueError("Invalid or oversized generated binding wheel inventory")
        metadata = [item.filename for item in members
                    if item.filename.count("/") == 1 and item.filename.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Generated binding wheel needs one METADATA record")
        info = metadata[0].rsplit("/", 1)[0]
        record_path = info + "/RECORD"
        if any(item.filename in (record_path + ".jws", record_path + ".p7s") for item in members):
            raise ValueError("Cannot rewrite an already signed binding wheel")
        payloads = {item.filename: (item, source_wheel.read(item)) for item in members if not item.is_dir()}
    records = {}
    for name, payload in licenses.items():
        filename = info + "/licenses/nvbroadcast-source/" + name
        member = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
        member.external_attr = 0o100644 << 16
        payloads[filename] = (member, payload)
        records[name] = {"file": filename, "sha256": hashlib.sha256(payload).hexdigest()}
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, (_, payload) in sorted(payloads.items()):
        if name != record_path:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + encoded, len(payload)))
    writer.writerow((record_path, "", ""))
    member = zipfile.ZipInfo(record_path, date_time=(1980, 1, 1, 0, 0, 0))
    member.external_attr = 0o100644 << 16
    payloads[record_path] = (member, output.getvalue().encode())
    descriptor, filename = tempfile.mkstemp(prefix=".binding-licences-", suffix=".whl", dir=wheel.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as result:
            for _, (member, payload) in sorted(payloads.items()):
                result.writestr(member, payload)
        temporary.replace(wheel)
    finally:
        temporary.unlink(missing_ok=True)
    return records


def compile_bindings(directory: Path, cache: Path, source_sha256: str,
                     tools_sha256: str, revision: str, epoch: str) -> None:
    if os.geteuid() == 0:
        raise RuntimeError("Build private bindings as the logged-in user, without sudo")
    runtime.base_probe()
    for address in (("127.0.0.1", 9), ("1.1.1.1", 443)):
        with socket.socket() as client:
            client.settimeout(2)
            try:
                client.connect(address)
            except OSError as error:
                if error.errno not in (errno.EPERM, errno.EACCES):
                    raise RuntimeError("Binding compilation requires enforced network denial") from error
            else:
                raise RuntimeError("Binding compilation requires enforced network denial")
    sources = runtime._regular_json(directory / runtime.BINDING_INPUT_FILE, source_sha256)
    owners = runtime.validate_binding_sources(sources)
    tools = runtime._regular_json(directory / "binding-tool-inputs.json", tools_sha256)["tools"]
    # The caller already acquired every input. This phase cannot repair a cache.
    for item in [*tools, *sources["bindings"]]:
        prepare_inputs.fetch(cache, item, offline=True)
    python = Path(sys.executable)
    environment = clean_environment(python)
    environment["SOURCE_DATE_EPOCH"] = epoch
    requirements = directory / "binding-tool-requirements.txt"
    records = [runtime.wheel_record(cache / prepare_inputs.filename(item)) for item in tools]
    if any(record["name"] != runtime.canonical(item["name"]) or record["version"] != item["version"]
           for record, item in zip(records, tools)):
        raise ValueError("Binding build tool metadata does not match its pin")
    requirements.write_text(runtime.requirements(records))
    runtime.run(python, "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
                "--require-hashes", "--only-binary=:all:", "--force-reinstall", "--no-cache-dir",
                "--disable-pip-version-check", "--find-links", cache, "-r", requirements, env=environment)
    wheel_directory = directory / "wheels"
    wheel_directory.mkdir()
    built = []
    for name in ("pycairo", "pygobject"):
        item = owners[name]
        archive = prepare_inputs.fetch(cache, item, offline=True)
        requirement = directory / f"{name}-source-requirement.txt"
        requirement.write_text(f'{name} @ {archive.as_uri()} --hash=sha256:{item["sha256"]}\n')
        command = [python, "-m", "pip", "--isolated", "wheel", "--no-index", "--no-deps",
                   "--no-build-isolation", "--require-hashes", "--no-cache-dir",
                   "--disable-pip-version-check", "--wheel-dir", wheel_directory,
                   "--config-settings=setup-args=-Dbuildtype=release"]
        if name == "pygobject":
            command += ["--config-settings=setup-args=-Dpycairo=enabled",
                        "--config-settings=setup-args=-Dtests=false"]
        runtime.run(*command, "-r", requirement, env=environment)
        candidates = [path for path in wheel_directory.iterdir()
                      if runtime.wheel_record(path)["name"] == name]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one newly built binding wheel")
        source_licenses = retain_source_licenses(candidates[0], archive, item)
        record = runtime.validate_binding_wheel(candidates[0], item, source_licenses)
        built.append(record)
        # PyGObject's build needs the exact private Pycairo headers/metadata.
        install_requirement = directory / f"{name}-wheel-requirement.txt"
        install_requirement.write_text(runtime.requirements([record]))
        runtime.run(python, "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
                    "--require-hashes", "--only-binary=:all:", "--force-reinstall", "--no-cache-dir",
                    "--disable-pip-version-check", "--find-links", wheel_directory,
                    "-r", install_requirement, env=environment)
    files, versions = runtime.binding_parameters(built)
    native = runtime.native_probe(Path(sys.prefix), files, versions)
    tools_inventory = {}
    for name, arguments in (("clang", ["/usr/bin/clang", "--version"]),
                            ("ninja", ["/opt/homebrew/bin/ninja", "--version"]),
                            ("pkg-config", ["/opt/homebrew/bin/pkg-config", "--version"])):
        executable = Path(arguments[0]).resolve(strict=True)
        result = runtime.run(*arguments, capture_output=True, text=True, env=environment)
        tools_inventory[name] = {"file": str(executable), "sha256": runtime.digest(executable),
                                 "version": result.stdout.strip()[:2000]}
    inventory = {"schema_version": 1, "source_revision": revision, "source_date_epoch": epoch,
                 "native": native, "build_tools": tools, "external_tools": tools_inventory,
                 "compilation_network": "denied by sandbox-exec",
                 "limits": ["Homebrew Python and native libraries remain external",
                            "Prepared binding wheels are reused as authenticated build inputs; independent native recompilation reproducibility is not claimed"]}
    (directory / runtime.BINDING_INVENTORY_FILE).write_text(json.dumps(inventory, indent=2) + "\n")
    receipt = {"schema_version": 1, "target": runtime.TARGET, "source_revision": revision,
               "source_inputs": {"file": runtime.BINDING_INPUT_FILE, "sha256": source_sha256},
               "build_inventory": {"file": runtime.BINDING_INVENTORY_FILE,
                                   "sha256": runtime.digest(directory / runtime.BINDING_INVENTORY_FILE)},
               "wheels": sorted(built, key=lambda item: item["name"])}
    path = directory / runtime.BINDING_MANIFEST_FILE
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    runtime.validate_binding_inputs(directory, runtime.digest(path))


def prepare(directory: Path, cache: Path) -> str:
    if os.geteuid() == 0:
        raise RuntimeError("Build private bindings as the logged-in user, without sudo")
    runtime.base_probe()
    if runtime.run("git", "status", "--porcelain", cwd=ROOT, capture_output=True, text=True).stdout:
        raise RuntimeError("Binding preparation requires a committed clean source checkout")
    revision = runtime.run("git", "rev-parse", "HEAD", cwd=ROOT, capture_output=True, text=True).stdout.strip()
    epoch = runtime.run("git", "show", "-s", "--format=%ct", "HEAD", cwd=ROOT,
                        capture_output=True, text=True).stdout.strip()
    source_path = ROOT / "packaging/macos-runtime/inputs.json"
    sources = json.loads(source_path.read_text())
    runtime.validate_binding_sources(sources)
    pins = json.loads((HERE / "inputs.json").read_text())
    tools = tool_inputs(pins, sources)
    environment = clean_environment(Path(sys.executable))
    runtime.run("/opt/homebrew/bin/pkg-config", "--exists", "cairo", "girepository-2.0",
                "gtk4", "libadwaita-1", "gstreamer-1.0", env=environment)
    for path in (Path("/usr/bin/clang"), Path("/opt/homebrew/bin/ninja"), Path("/usr/bin/sandbox-exec")):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"Missing native binding build prerequisite: {path}")
    if cache.is_symlink():
        raise ValueError("Refusing a redirected binding input cache")
    cache.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / runtime.BINDING_INPUT_FILE).write_bytes(source_path.read_bytes())
    tool_path = directory / "binding-tool-inputs.json"
    tool_path.write_text(json.dumps({"tools": tools}, indent=2) + "\n")
    for item in [*tools, *sources["bindings"]]:
        prepare_inputs.fetch(cache, item)
    compiler = directory / "compiler"
    runtime.run(sys.executable, "-m", "venv", compiler, env=environment)
    python = compiler / "bin/python"
    runtime.run("/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)",
                python, "-I", "-B", HERE / "prepare_bindings.py", "_compile",
                "--directory", directory, "--cache", cache,
                "--source-sha256", runtime.digest(directory / runtime.BINDING_INPUT_FILE),
                "--tools-sha256", runtime.digest(tool_path), "--revision", revision, "--epoch", epoch,
                env=clean_environment(python))
    expected = runtime.digest(directory / runtime.BINDING_MANIFEST_FILE)
    runtime.validate_binding_inputs(directory, expected)
    print(json.dumps({"directory": str(directory), "manifest": runtime.BINDING_MANIFEST_FILE,
                      "sha256": expected, "target": runtime.TARGET}), flush=True)
    return expected


def write_expected_digest(path: Path, expected: str, directory: Path, cache: Path) -> None:
    if (not re.fullmatch(r"[0-9a-f]{64}", expected)
            or any(path.resolve().is_relative_to(root.resolve()) for root in (directory, cache))):
        raise ValueError("The validated preparation SHA-256 must be kept outside its input directories")
    # Never overwrite an unrelated file or follow an existing leaf symlink.
    with path.open("x") as stream:
        stream.write(expected + "\n")


def verify_closure(directory: Path, expected_sha256: str, supplier_wheels: Path, app_wheel: Path) -> dict:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    from supplier_lock import MARKERS

    prepared = runtime.validate_binding_inputs(directory, expected_sha256)
    metadata = {}
    paths = [*supplier_wheels.glob("*.whl"), app_wheel,
             *(directory / "wheels" / item["file"] for item in prepared["wheels"])]
    for path in paths:
        record = runtime.wheel_record(path)
        if record["name"] in metadata:
            raise ValueError("Duplicate native/upstream wheel owner in binding closure")
        with zipfile.ZipFile(path) as archive:
            entries = [name for name in archive.namelist()
                       if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            info = BytesParser().parsebytes(archive.read(entries[0]))
        metadata[record["name"]] = info
    extras = {name: set() for name in runtime.BINDING_NAMES}
    pending, visited = set(runtime.BINDING_NAMES), {}
    while pending:
        owner = pending.pop()
        selected = frozenset(extras.get(owner, ()))
        if visited.get(owner) == selected:
            continue
        visited[owner] = selected
        info = metadata[owner]
        if Version(".".join(map(str, runtime.MINIMUM_PYTHON))) not in SpecifierSet(info.get("Requires-Python", "")):
            raise ValueError("Native binding closure raises the supported Python floor")
        declared = {runtime.canonical(name) for name in info.get_all("Provides-Extra", [])}
        if not selected <= declared:
            raise ValueError("Native binding dependency requests an undefined extra")
        for raw in info.get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate({**MARKERS, "extra": extra})
                                               for extra in {"", *selected}):
                continue
            name = runtime.canonical(requirement.name)
            if requirement.url or name not in metadata or metadata[name]["Version"] not in requirement.specifier:
                raise ValueError(f"Unsatisfied private binding dependency: {owner} requires {requirement}")
            extras.setdefault(name, set()).update(runtime.canonical(extra) for extra in requirement.extras)
            pending.add(name)
    return {"status": "pass", "target": runtime.TARGET, "binding_manifest_sha256": expected_sha256,
            "binding_owners": sorted(runtime.BINDING_NAMES), "combined_owner_count": len(metadata)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "_compile", "verify-closure"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--tools-sha256")
    parser.add_argument("--revision")
    parser.add_argument("--epoch")
    parser.add_argument("--sha256")
    parser.add_argument("--sha256-output", type=Path)
    parser.add_argument("--supplier-wheels", type=Path)
    parser.add_argument("--app-wheel", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "prepare":
        if args.cache is None or args.sha256_output is None:
            parser.error("preparation needs an explicit cache and an expected-digest output outside it")
        directory, cache = args.directory.absolute(), args.cache.absolute()
        output = args.sha256_output.absolute()
        if output.exists() or output.is_symlink() or any(
                output.resolve().is_relative_to(root.resolve()) for root in (directory, cache)):
            parser.error("expected-digest output must be a new file outside both input directories")
        expected = prepare(directory, cache)
        write_expected_digest(output, expected, directory, cache)
    elif args.command == "_compile":
        if not all((args.cache, args.source_sha256, args.tools_sha256, args.revision, args.epoch)):
            parser.error("internal compilation needs its authenticated source/tool identities")
        compile_bindings(args.directory.absolute(), args.cache.absolute(), args.source_sha256,
                         args.tools_sha256, args.revision, args.epoch)
    else:
        if not all((args.sha256, args.supplier_wheels, args.app_wheel)):
            parser.error("closure verification needs the expected manifest SHA and both wheel selections")
        result = verify_closure(args.directory, args.sha256, args.supplier_wheels, args.app_wheel)
        print(json.dumps(result, indent=2))

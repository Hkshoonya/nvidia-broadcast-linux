#!/usr/bin/env python3
"""Build a NON-DEFAULT unsigned arm64 Mac candidate with an audited wheelhouse.

Normal builds consume the reviewed supplier lock without resolving. Explicit
refresh mode proposes a new lock without packaging; current source is rebuilt.
No signing credentials, production package identifiers or release uploads.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tomllib

import runtime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("runtime_prepare", ROOT / "packaging/runtime-prototype/prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def root_preinstall() -> str:
    """Reuse the shipping installer's tested privilege boundary, no default edit."""
    source = (ROOT / "build-packages.sh").read_text().split("build_pkg() {", 1)[1]
    script = source.split("<< 'PREINST'\n", 1)[1].split("\nPREINST", 1)[0]
    script = script.replace("/opt/nvbroadcast", "/opt/nvbroadcast-offline-candidate")
    script = script.replace("/usr/local/bin/nvbroadcast", "/usr/local/bin/nvbroadcast-offline-candidate")
    script = script.replace('"${BASH_REMATCH[1]}" -lt 13', '"${BASH_REMATCH[1]}" -lt 14')
    script = script.replace("macOS 13 (Ventura)", "macOS 14 (Sonoma)")
    # Our payload differs from the source installer. Check EVERY existing
    # candidate descendant; no legacy venv exemption is needed at this path.
    start = script.index("# Existing package subtrees")
    return script[:start] + '''# Refuse any redirected or user-writable candidate payload path.
if [[ -e /opt/nvbroadcast-offline-candidate ]]; then
    while IFS= read -r path; do
        check_destination "$path"
    done < <(/usr/bin/find -P /opt/nvbroadcast-offline-candidate -print)
fi
'''


def select_suppliers(python: Path, work: Path, inputs: Path, pins: dict, app_wheel: Path,
                     records: list[dict], faster: str, *, refresh_lock: bool) -> tuple[list[dict], Path, str]:
    supplier_lock_path = work / "supplier-lock.toml"
    if refresh_lock:
        requirement_input = work / "requirements.in"
        requirement_input.write_text(f"nvbroadcast[cpu,meeting-support] @ {app_wheel.as_uri()}\nfaster-whisper=={faster}\n"
                                     + "".join(f"{item['name']}=={item['version']}\n" for item in records))
        resolver_archive = prepare.fetch(inputs, pins["uv"])
        prepare.unpack_uv(resolver_archive, pins["uv"], work / "resolver")
        resolved = work / "resolved-pylock.toml"
        runtime.run(work / "resolver/uv-aarch64-apple-darwin/uv", "pip", "compile", requirement_input,
                    "--python", sys.executable, "--python-version", "3.13", "--no-python-downloads",
                    "--python-platform", "aarch64-apple-darwin", "--only-binary", ":all:",
                    "--format", "pylock.toml", "--output-file", resolved,
                    env={**os.environ, "MACOSX_DEPLOYMENT_TARGET": runtime.MINIMUM_MACOS, "UV_NO_CONFIG": "1"})
        selection_result = runtime.run(python, HERE / "supplier_lock.py", "propose",
            "--lock", resolved, "--app-wheel", app_wheel, "--output", supplier_lock_path,
            capture_output=True, text=True)
        lock_sha256 = runtime.digest(supplier_lock_path)
    else:
        reviewed = pins["supplier_lock"]
        if reviewed["file"] != "pylock.macos-arm64-cp313-cpu.toml":
            raise ValueError("Unexpected supplier lock path")
        shutil.copyfile(HERE / reviewed["file"], supplier_lock_path)
        lock_sha256 = reviewed["sha256"]
        selection_result = runtime.run(python, HERE / "supplier_lock.py", "select",
            "--lock", supplier_lock_path, "--sha256", lock_sha256, capture_output=True, text=True)
    return json.loads(selection_result.stdout), supplier_lock_path, lock_sha256


def build(work: Path, *, refresh_lock: bool = False, cache: Path | None = None, offline: bool = False):
    if refresh_lock and offline:
        raise ValueError("Refreshing the supplier lock requires explicit online resolution")
    if os.geteuid() == 0:
        raise RuntimeError("Build as an ordinary user")
    native = runtime.native_probe()
    if runtime.run("git", "status", "--porcelain", cwd=ROOT, capture_output=True, text=True).stdout:
        raise RuntimeError("Build requires a committed clean source checkout")
    work.mkdir(parents=True, exist_ok=False)
    (work / "native-build-inventory.json").write_text(json.dumps(native, indent=2) + "\n")
    pins = json.loads((HERE / "inputs.json").read_text())
    if (pins["minimum_macos"] != runtime.MINIMUM_MACOS
            or pins["minimum_python"] != ".".join(map(str, runtime.MINIMUM_PYTHON))):
        raise ValueError("Builder and runtime deployment floors disagree")
    inputs = cache or work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    bootstrap = work / "bootstrap"
    (bootstrap / "wheels").mkdir(parents=True)
    records = []
    for item in pins["bootstrap"]:
        path = prepare.fetch(inputs, item, offline=offline)
        shutil.copyfile(path, bootstrap / "wheels" / path.name)
        record = runtime.wheel_record(path)
        if record["name"] != item["name"] or record["version"] != item["version"]:
            raise ValueError("Bootstrap wheel metadata mismatch")
        records.append(record)
    (bootstrap / "requirements.txt").write_text(runtime.requirements(records))
    runtime.run(sys.executable, "-m", "venv", work / "builder")
    python = work / "builder/bin/python"
    runtime.pip_install(python, bootstrap, bootstrap / "requirements.txt")

    # Build from git's committed bytes, excluding ambient egg-info/build files.
    source = work / "source"
    source.mkdir()
    archive = work / "source.tar"
    runtime.run("git", "archive", "--format=tar", "--output", archive, "HEAD", cwd=ROOT)
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter=prepare.safe_member)
    wheels = work / "app-wheel"
    wheels.mkdir()
    revision = runtime.run("git", "rev-parse", "HEAD", cwd=ROOT, capture_output=True, text=True).stdout.strip()
    epoch = runtime.run("git", "show", "-s", "--format=%ct", "HEAD", cwd=ROOT, capture_output=True, text=True).stdout.strip()
    runtime.run(python, "-m", "pip", "--isolated", "wheel", "--no-index", "--no-deps",
                "--no-build-isolation", "--no-cache-dir", "--wheel-dir", wheels, source,
                env={**os.environ, "SOURCE_DATE_EPOCH": epoch})
    app_wheel, = wheels.glob("*.whl")
    tree = ast.parse((source / "src/nvbroadcast/runtime/variants.py").read_text())
    faster = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "FASTER_WHISPER_VERSION" for target in node.targets))
    selection, supplier_lock_path, lock_sha256 = select_suppliers(
        python, work, inputs, pins, app_wheel, records, faster, refresh_lock=refresh_lock)
    (work / "supplier-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    supplier_wheels = work / "supplier-wheels"
    supplier_wheels.mkdir()
    for item in selection:
        downloaded = prepare.fetch(inputs, item, offline=offline)
        shutil.copyfile(downloaded, supplier_wheels / item["file"])
    # Verify the complete current-source dependency closure before pkgbuild.
    # The helper runs using pinned packaging tools and only reads target wheels.
    closure = runtime.run(python, HERE / "supplier_lock.py", "verify",
        "--lock", supplier_lock_path, "--sha256", lock_sha256, "--app-wheel", app_wheel,
        "--wheelhouse", supplier_wheels, "--faster-whisper", faster, "--pins", HERE / "inputs.json",
        capture_output=True, text=True)
    (work / "supplier-closure.json").write_text(closure.stdout)
    if refresh_lock:
        (work / "supplier-lock-update.json").write_text(json.dumps({
            "file": "pylock.macos-arm64-cp313-cpu.toml", "sha256": lock_sha256,
            "source_revision": revision, "status": "proposed; review before committing"}, indent=2) + "\n")
        print(f"Proposed supplier lock (no package built): {supplier_lock_path}")
        return

    lock_path = work / "pylock.macos-arm64-cp313-cpu.toml"
    lock_path.write_text(supplier_lock_path.read_text() +
        '\n[[packages]]\nname = "nvbroadcast"\nversion = ' + json.dumps(runtime.wheel_record(app_wheel)["version"]) +
        '\narchive = { path = ' + json.dumps("app-wheel/" + app_wheel.name) +
        ', hashes = { sha256 = ' + json.dumps(runtime.digest(app_wheel)) + ' } }\n')
    payload_root = work / "root"
    bundle = payload_root / "opt/nvbroadcast-offline-candidate"
    bundle.mkdir(parents=True)
    shutil.copytree(supplier_wheels, bundle / "wheels")
    shutil.copyfile(app_wheel, bundle / "wheels" / app_wheel.name)
    packages = sorted((runtime.wheel_record(path) for path in (bundle / "wheels").iterdir()), key=lambda item: item["name"])
    if len({item["name"] for item in packages}) != len(packages):
        raise ValueError("Target lock has more than one wheel per distribution; select an exact target before packaging")
    (bundle / "requirements.txt").write_text(runtime.requirements(packages))
    manifest = {"schema_version": 1, "target": runtime.TARGET, "minimum_macos": runtime.MINIMUM_MACOS,
                "source_revision": revision, "source_date_epoch": epoch, "packages": packages,
                "minimum_python": pins["minimum_python"],
                "resolver": pins["uv"], "native_build_inventory": native,
                "supplier_lock_sha256": lock_sha256, "dependency_resolution": "reviewed-fixed-inputs",
                "limits": ["Homebrew native dependencies and CPython are external prerequisites",
                           "Models are not bundled; first model acquisition still needs networking",
                           "Unsigned non-default candidate; no physical hardware qualification"]}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(lock_path, bundle / lock_path.name)
    for name in ("supplier-lock.toml", "supplier-selection.json", "supplier-closure.json"):
        shutil.copyfile(work / name, bundle / name)
    for filename in ("runtime.py", "setup.sh"):
        shutil.copyfile(HERE / filename, bundle / filename)
    shutil.copyfile(ROOT / "scripts/validate_macos_runtime.py", bundle / "validate_macos_runtime.py")
    for filename in ("LICENSE", "NOTICE", "CONTRIBUTORS.md"):
        shutil.copyfile(ROOT / filename, bundle / filename)
    shutil.copytree(ROOT / "data", bundle / "data")
    launcher = payload_root / "usr/local/bin/nvbroadcast-offline-candidate"
    launcher.parent.mkdir(parents=True)
    shutil.copyfile(HERE / "launcher.sh", launcher)
    for path in payload_root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    launcher.chmod(0o755)
    (bundle / "setup.sh").chmod(0o755)
    runtime.validate_bundle(bundle)
    (bundle / "macos-runtime-id").write_text(runtime.tree_digest(bundle) + "\n")
    (bundle / "macos-runtime-id").chmod(0o644)
    scripts = work / "installer-scripts"
    scripts.mkdir()
    (scripts / "preinstall").write_text(root_preinstall())
    (scripts / "preinstall").chmod(0o755)
    output = work / "NVBroadcast-offline-candidate-arm64.pkg"
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    runtime.run("/usr/bin/pkgbuild", "--root", payload_root, "--ownership", "recommended",
                "--identifier", "com.doczeus.nvbroadcast.offline-candidate", "--version", version,
                "--scripts", scripts, "--install-location", "/", output)
    (work / "package.json").write_text(json.dumps({"file": output.name, "sha256": runtime.digest(output),
        "source_revision": revision, "runtime_id": runtime.tree_digest(bundle), "signed": False}, indent=2) + "\n")
    print(f"Unsigned experimental candidate: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--refresh-lock", action="store_true", help="propose a new supplier lock; do not build a package")
    parser.add_argument("--cache", type=Path, help="cache of verified build and supplier inputs")
    parser.add_argument("--offline", action="store_true", help="require all input archives in --cache; no downloads")
    args = parser.parse_args()
    build(args.directory.absolute(), refresh_lock=args.refresh_lock,
          cache=args.cache.absolute() if args.cache else None, offline=args.offline)

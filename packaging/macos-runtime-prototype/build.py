#!/usr/bin/env python3
"""Build a NON-DEFAULT unsigned arm64 Mac candidate with an audited wheelhouse.

Resolution is an explicit build-only experiment. The generated PEP 751 lock,
wheel hashes, app source revision and native inventory travel with the artifact.
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


def build(work: Path):
    if os.geteuid() == 0:
        raise RuntimeError("Build as an ordinary user")
    native = runtime.native_probe()
    if runtime.run("git", "status", "--porcelain", cwd=ROOT, capture_output=True, text=True).stdout:
        raise RuntimeError("Build requires a committed clean source checkout")
    work.mkdir(parents=True, exist_ok=False)
    (work / "native-build-inventory.json").write_text(json.dumps(native, indent=2) + "\n")
    pins = json.loads((HERE / "inputs.json").read_text())
    inputs = work / "inputs"
    inputs.mkdir()
    bootstrap = work / "bootstrap"
    (bootstrap / "wheels").mkdir(parents=True)
    records = []
    for item in pins["bootstrap"]:
        path = prepare.fetch(bootstrap / "wheels", item)
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
                "--no-build-isolation", "--wheel-dir", wheels, source,
                env={**os.environ, "SOURCE_DATE_EPOCH": epoch})
    app_wheel, = wheels.glob("*.whl")
    tree = ast.parse((source / "src/nvbroadcast/runtime/variants.py").read_text())
    faster = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "FASTER_WHISPER_VERSION" for target in node.targets))
    requirement_input = work / "requirements.in"
    requirement_input.write_text(f"nvbroadcast[cpu,meeting-support] @ {app_wheel.as_uri()}\nfaster-whisper=={faster}\n"
                                 + "".join(f"{item['name']}=={item['version']}\n" for item in records))
    resolver_archive = prepare.fetch(inputs, pins["uv"])
    prepare.unpack_uv(resolver_archive, pins["uv"], work / "resolver")
    lock_path = work / "pylock.macos-arm64-cp313-cpu.toml"
    runtime.run(work / "resolver/uv-aarch64-apple-darwin/uv", "pip", "compile", requirement_input,
                "--python", sys.executable, "--python-version", "3.13", "--no-python-downloads",
                "--python-platform", "aarch64-apple-darwin", "--only-binary", ":all:",
                "--format", "pylock.toml", "--output-file", lock_path,
                env={**os.environ, "MACOSX_DEPLOYMENT_TARGET": "13.0", "UV_NO_CONFIG": "1"})
    # Keep the lock portable; all other wheels retain supplier URLs and hashes.
    lock_text = lock_path.read_text().replace(json.dumps(str(app_wheel)), json.dumps("app-wheel/" + app_wheel.name))
    lock_path.write_text("# Generated on macOS arm64 by the pinned resolver.\n" + "\n".join(
        line for line in lock_text.splitlines() if not line.startswith("#")) + "\n")
    payload_root = work / "root"
    bundle = payload_root / "opt/nvbroadcast-offline-candidate"
    bundle.mkdir(parents=True)
    (bundle / "wheels").mkdir()
    # uv's platform lock can contain several compatible wheel alternatives.
    # Select exactly one using explicit macOS 13 arm64 CPython 3.13 tags;
    # record the complete original lock as well as the selected wheel hashes.
    lock = tomllib.loads(lock_path.read_text())
    selection = runtime.run(python, HERE / "wheel_target.py",
        input=json.dumps([prepare.filename({"url": wheel["url"]})
                      for package in lock["packages"] for wheel in package.get("wheels", [])]),
        capture_output=True, text=True)
    ranks = json.loads(selection.stdout)
    for package in lock["packages"]:
        if "archive" in package:
            if package["name"] != "nvbroadcast" or package["archive"]["hashes"]["sha256"] != runtime.digest(app_wheel):
                raise ValueError("Unexpected local wheel in runtime lock")
            shutil.copyfile(app_wheel, bundle / "wheels" / app_wheel.name)
            continue
        choices = [wheel for wheel in package.get("wheels", [])
                   if ranks[prepare.filename({"url": wheel["url"]})] is not None]
        if not choices:
            raise ValueError(f"No macOS 13 arm64 CPython 3.13 wheel for {package['name']}")
        selected = min(choices, key=lambda wheel: ranks[prepare.filename({"url": wheel["url"]})])
        prepare.fetch(bundle / "wheels", {"url": selected["url"], "sha256": selected["hashes"]["sha256"]})
    packages = sorted((runtime.wheel_record(path) for path in (bundle / "wheels").iterdir()), key=lambda item: item["name"])
    if len({item["name"] for item in packages}) != len(packages):
        raise ValueError("Target lock has more than one wheel per distribution; select an exact target before packaging")
    (bundle / "requirements.txt").write_text(runtime.requirements(packages))
    manifest = {"schema_version": 1, "target": runtime.TARGET, "minimum_macos": "13.0",
                "source_revision": revision, "source_date_epoch": epoch, "packages": packages,
                "resolver": pins["uv"], "native_build_inventory": native,
                "limits": ["Homebrew native dependencies and CPython are external prerequisites",
                           "Models are not bundled; first model acquisition still needs networking",
                           "Unsigned non-default candidate; no physical hardware qualification"]}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(lock_path, bundle / lock_path.name)
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
    build(parser.parse_args().directory.absolute())

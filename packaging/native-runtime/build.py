#!/usr/bin/env python3
"""Build complete, locked, offline native Linux runtimes and packages.

Only input preparation downloads. Wheel construction, assembly and native
wrapping run without a network. No package hook runs an installer or resolver.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOTYPE = HERE.parent / "runtime-prototype"
sys.path.insert(0, str(PROTOTYPE))
import prepare
from assemble import inventory

spec = importlib.util.spec_from_file_location("native_primitives", HERE.parent / "native-prototype/build.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def run(command: list[str], log: Path, **kwargs) -> None:
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, **kwargs)


def docker(image: str, mounts: dict[Path, str], command: list[str], log: Path, epoch: int) -> None:
    args = ["docker", "run", "--rm", "--init", "--network=none", "--pull=never",
            "--user", f"{os.getuid()}:{os.getgid()}", "-e", f"SOURCE_DATE_EPOCH={epoch}"]
    for source, target in mounts.items():
        args += ["-v", f"{source}:{target}"]
    run(args + [image, *command], log)


def source_identity(revision: str) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"], cwd=REPO, text=True).strip()
    project = tomllib.loads(subprocess.check_output(["git", "show", f"{commit}:pyproject.toml"], cwd=REPO, text=True))
    version = project["project"]["version"]
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", version):
        raise ValueError("native release version must have three numeric components")
    return {"revision": commit, "version": version,
            "source_date_epoch": int(subprocess.check_output(["git", "show", "-s", "--format=%ct", commit], cwd=REPO))}


def verified_images(path: Path, provided: dict[str, str]) -> dict:
    pins = json.loads(path.read_text())
    for role, value in provided.items():
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", value) or pins[role]["image"] != value:
            raise ValueError(f"{role} image differs from reviewed builder pin")
        actual = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", value], text=True).strip()
        if actual != value:
            raise ValueError(f"{role} local image identity differs")
    return pins


def verify_application_source(runtime: Path, revision: str) -> None:
    roots = list(runtime.glob("lib/python*/site-packages/nvbroadcast"))
    if len(roots) != 1:
        raise ValueError("expected exactly one installed application tree")
    root = roots[0]
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision,
                                     "--", "src/nvbroadcast"], cwd=REPO, text=True).splitlines()
    expected = {path.removeprefix("src/nvbroadcast/"): path for path in files if path.endswith(".py")}
    if {str(path.relative_to(root)) for path in root.rglob("*.py")} != expected.keys():
        raise ValueError("application Python file set differs from source")
    for relative, source_path in expected.items():
        source_bytes = subprocess.check_output(["git", "show", f"{revision}:{source_path}"], cwd=REPO)
        if (root / relative).read_bytes() != source_bytes:
            raise ValueError(f"application differs from source: {relative}")


def assemble(args, output: Path, source: dict) -> Path:
    cache = args.cache.resolve()
    for name in ("inputs", "bootstrap", "application", "wheels"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    pins = json.loads((PROTOTYPE / "inputs.json").read_text())
    for item in (pins["python"], pins["uv"], *pins["bindings"]):
        prepare.fetch(cache / "inputs", item, offline=args.offline)
    prepare.fetch_lock(PROTOTYPE / "pylock.build.toml", cache / "bootstrap", offline=args.offline)
    work = output / "assembly"
    work.mkdir()
    prepare.unpack_python(cache / "inputs" / prepare.filename(pins["python"]), pins["python"], work / "builder")
    archive_path = work / "source.tar"
    with archive_path.open("wb") as stream:
        subprocess.run(["git", "archive", source["revision"], "src", "data", "pyproject.toml", "LICENSE",
                        "NOTICE", "CONTRIBUTORS.md", "README.md"], cwd=REPO, stdout=stream, check=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(work / "app-source", filter=prepare.safe_member)
    archive_path.unlink()
    # A separate private builder prevents build tooling contaminating the payload.
    # All binding wheel hashes must still match the committed dependency locks.
    script = (PROTOTYPE / "build-bindings.sh").read_text().replace(
        '"$NVB_PYTHON" -I /prototype/prepare.py fetch --offline --directory /work\n', "")
    script = script.replace("# setuptools writes build metadata beside the source.",
        f"export SOURCE_DATE_EPOCH={source['source_date_epoch']}\n# setuptools writes build metadata beside the source.")
    (work / "build.sh").write_text(script)
    for name in ("inputs", "bootstrap"):
        (work / name).symlink_to(cache / name, target_is_directory=True)
    docker(args.bindings_image, {work: "/work", cache: str(cache) + ":ro",
           work / "builder/runtime": "/opt/nvbroadcast/.venv"},
           ["bash", "/work/build.sh"], work / "wheel-build.log", pins["application"]["source_date_epoch"])
    lock_text = (HERE / f"pylock.linux-x86_64-cp313-{args.variant}.toml").read_text()
    app = work / "wheels" / f"nvbroadcast-{source['version']}-py3-none-any.whl"
    lock_text += ('\n[[packages]]\nname = "nvbroadcast"\nversion = ' + json.dumps(source["version"]) +
                  '\narchive = { path = ' + json.dumps(app.name) + ', hashes = { sha256 = "' + prepare.digest(app) + '" } }\n')
    lock = work / "pylock.toml"
    lock.write_text(lock_text)
    application = work / "application"
    (application / "wheels").mkdir(parents=True)
    # Reuse verified download bytes without rewriting a previous build's lock
    # materialization. The application's same-version wheel may have new bytes.
    for package in tomllib.loads(lock_text)["packages"]:
        for wheel in package.get("wheels", []):
            item = {"url": wheel["url"], "sha256": wheel["hashes"]["sha256"]}
            cached = cache / "application/wheels" / prepare.filename(item)
            if cached.is_file():
                prepare.verify(cached, item)
                os.link(cached, application / "wheels" / cached.name)
    prepare.fetch_lock(lock, application, work / "wheels", offline=args.offline)
    # The complete wheelhouse is content-verified once again before assembly.
    prepare.fetch_lock(lock, application, work / "wheels", offline=True)
    payload = output / "payload"
    prepare.unpack_python(cache / "inputs" / prepare.filename(pins["python"]), pins["python"], payload)
    runtime = payload / "runtime"
    docker(args.bindings_image, {runtime: "/opt/nvbroadcast/.venv", application: "/inputs:ro"},
           ["/opt/nvbroadcast/.venv/bin/python", "-I", "-B", "-m", "pip", "--isolated", "install", "--no-index",
            "--no-cache-dir", "--no-deps", "--no-compile", "--require-hashes", "--only-binary=:all:",
            "--find-links=/inputs/wheels", "-r", "/inputs/requirements.txt"], work / "assemble.log", source["source_date_epoch"])
    provenance = runtime / "share/nvbroadcast-runtime-provenance"
    shutil.copyfile(lock, provenance / "pylock.toml")
    pins["application"] = source
    (provenance / "inputs.json").write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(HERE / "builders.json", provenance / "builders.json")
    (provenance / "selection.json").write_text(json.dumps({"target": f"linux-x86_64-cp313-{args.variant}",
        "lock_sha256": prepare.digest(lock), "source": source}, indent=2, sort_keys=True) + "\n")
    (payload / "manifest.json").write_text(json.dumps(inventory(runtime), indent=2, sort_keys=True) + "\n")
    return runtime


def hook(action: str, version: str, variant: str, family: str) -> str:
    script = (HERE / "package-hooks.sh").read_text().replace("@VERSION@", version).replace("@VARIANT@", variant)
    # Detect a real legacy package before unpacking, not an arbitrary /opt tree.
    legacy = ""
    if action == "prepare":
        command = ("dpkg-query -S /opt/nvbroadcast/scripts/install_runtime_variant.py" if family == "deb"
                   else "rpm -qf --qf '%{NAME}: /opt/nvbroadcast/scripts/install_runtime_variant.py' /opt/nvbroadcast/scripts/install_runtime_variant.py")
        legacy = (f"owner=$({command} 2>/dev/null || true)\n"
                  "if [ \"$owner\" = 'nvbroadcast: /opt/nvbroadcast/scripts/install_runtime_variant.py' ]; then\n"
                  "    touch /usr/lib/nvbroadcast/.legacy-runtime\nfi\n")
    return "#!/bin/sh\nset -eu\nset -- " + action + "\n" + script + "\n" + legacy


def wrap(args, runtime: Path, output: Path, source: dict, family: str) -> dict:
    manifest = json.loads((runtime.parent / "manifest.json").read_text())
    if inventory(runtime) != manifest:
        raise ValueError("runtime differs from assembly manifest")
    native.verify_owner(runtime, args.variant)
    native.app_files(runtime, manifest)
    verify_application_source(runtime, source["revision"])
    version = f"{source['version']}-{args.package_revision}.{args.variant}"
    work = output / family
    stage = work / "stage"
    native.populate(runtime, manifest, set(manifest), stage)
    native.integration(stage, family, "self", version, args.variant)
    base = stage / "usr/lib/nvbroadcast"
    # Keep the existing public package name across migration and variant changes.
    check = native.version_check(family, "self", version, args.variant).replace(f"nvbroadcast-{args.variant}", "nvbroadcast").replace(" prototype:", ":")
    native.write(base / "check-packages", check, 0o755)
    for name in ("nvbroadcast", "nvbroadcast-vcam"):
        path = stage / "usr/bin" / name
        path.write_text(path.read_text().replace(" prototype:", ":"))
    path = stage / "usr/lib/systemd/user/nvbroadcast-vcam.service"
    path.write_text(path.read_text().replace(" prototype virtual camera", " virtual camera"))
    shutil.copyfile(HERE / "validate_install.py", base / "validate-install.py")
    shutil.copyfile(runtime.parent / "manifest.json", base / "runtime-manifest.json")
    native.write(base / "package.json", json.dumps({"schema_version": 1, "variant": args.variant,
        "version": version, "source": source, "runtime_manifest_sha256": prepare.digest(runtime.parent / "manifest.json")}, sort_keys=True) + "\n")
    dependency = native.DEPENDENCIES[family]
    if family == "deb":
        dependency += ", gstreamer1.0-plugins-bad, gstreamer1.0-plugins-ugly, v4l-utils, pulseaudio-utils, pipewire-bin | pipewire-utils"
        native.write(stage / "DEBIAN/control", f"Package: nvbroadcast\nVersion: {version}\nArchitecture: amd64\n"
            "Maintainer: doczeus <harshit@kshoonya.com>\nSection: video\nPriority: optional\n"
            f"Depends: {dependency}\nRecommends: v4l2loopback-dkms, gir1.2-ayatanaappindicator3-0.1\n"
            "Homepage: https://nvbroadcast.com\nDescription: NVBroadcast complete offline application runtime\n"
            f" Private Python and hash-locked {args.variant.upper()} dependencies.\n")
        native.write(stage / "DEBIAN/preinst", hook("prepare", version, args.variant, family), 0o755)
        native.write(stage / "DEBIAN/prerm", '#!/bin/sh\nset -eu\ncase "$1" in remove|deconfigure)\n' + hook("prepare", version, args.variant, family) + "\n;; esac\n", 0o755)
        native.write(stage / "DEBIAN/postinst", '#!/bin/sh\nset -eu\n[ "$1" != configure ] || {\n' + hook("finish", version, args.variant, family) + "\n}\n", 0o755)
        native.write(stage / "DEBIAN/postrm", '#!/bin/sh\nset -eu\ncase "$1" in remove|purge)\n' + hook("remove", version, args.variant, family) + "\n;; esac\n", 0o755)
        artifact = work / f"nvbroadcast_{version}_amd64.deb"
        command = ["dpkg-deb", "--root-owner-group", "-Zzstd", "-z8", "--threads-max=2", "--build", "/stage", f"/work/{artifact.name}"]
        image = args.deb_image
    else:
        dependency += ", gstreamer1-plugins-bad-free, v4l-utils, pulseaudio-utils, pipewire-utils"
        release = f"{args.package_revision}.{args.variant}"
        spec_text = (f"Name: nvbroadcast\nVersion: {source['version']}\nRelease: {release}\n"
            "Summary: NVBroadcast complete offline application runtime\nLicense: GPL-3.0-or-later AND LicenseRef-Bundled-Dependencies\n"
            f"BuildArch: x86_64\nAutoReqProv: no\nRequires: {dependency}\n"
            "%global __os_install_post %{nil}\n%global _build_id_links none\n%global debug_package %{nil}\n"
            "%global _binary_payload w8.zstdio\n%description\nPrivate Python and hash-locked application dependencies.\n"
            "%install\nmkdir -p %{buildroot}\ncp -a /stage/. %{buildroot}/\n"
            "%pretrans\n" + hook("prepare", version, args.variant, family) + "\n%posttrans\n" + hook("finish", version, args.variant, family) +
            '\n%preun\nif [ "$1" -eq 0 ]; then\n' + hook("prepare", version, args.variant, family) +
            '\nfi\n%postun\nif [ "$1" -eq 0 ]; then\n' + hook("remove", version, args.variant, family) + "\nfi\n%files -f /work/files.txt\n%defattr(-,root,root,-)\n")
        spec_text = spec_text.replace("'%s\\n'", "'%%s\\n'").replace("'%{NAME}", "'%%{NAME}")
        native.write(work / "package.spec", spec_text)
        native.write(work / "files.txt", "\n".join(("%dir " if p.is_dir() and not p.is_symlink() else "") + json.dumps("/" + str(p.relative_to(stage))).replace("%", "%%")
            for p in sorted(stage.rglob("*")) if not (p.is_dir() and len(p.relative_to(stage).parts) < 3)) + "\n")
        artifact = work / f"build/RPMS/x86_64/nvbroadcast-{version}.x86_64.rpm"
        command = ["rpmbuild", "-bb", "--define", "_topdir /work/build", "--define", "_buildhost nvbroadcast.invalid",
                   "--define", "use_source_date_epoch_as_buildtime 1", "--define", "clamp_mtime_to_source_date_epoch 1", "/work/package.spec"]
        image = args.rpm_image
    for path in (stage, *stage.rglob("*")):
        os.utime(path, (source["source_date_epoch"],) * 2, follow_symlinks=False)
    docker(image, {stage: "/stage:ro", work: "/work"}, command, work / "build.log", source["source_date_epoch"])
    return {"family": family, "variant": args.variant, "version": version, "artifact": str(artifact.relative_to(output)),
            "sha256": prepare.digest(artifact), "bytes": artifact.stat().st_size, "builder_image": image,
            "runtime_manifest_sha256": prepare.digest(runtime.parent / "manifest.json"), "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source", default="HEAD")
    parser.add_argument("--package-revision", default="2")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime", type=Path, help="reuse an assembled payload after source and manifest verification")
    parser.add_argument("--manifest-sha256", help="externally verified assembly manifest identity; required with --runtime")
    parser.add_argument("--families", nargs="+", choices=("deb", "rpm"), default=("deb", "rpm"))
    for role in ("bindings", "deb", "rpm"):
        parser.add_argument(f"--{role}-image", required=True)
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error("build as an ordinary user")
    if not re.fullmatch(r"[1-9][0-9]*", args.package_revision):
        parser.error("package revision must be a positive integer")
    if args.runtime and (not args.manifest_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256)):
        parser.error("--runtime requires an externally verified --manifest-sha256")
    source = source_identity(args.source)
    verified_images(HERE / "builders.json", {role: getattr(args, role + "_image") for role in ("bindings", "deb", "rpm")})
    os.umask(0o022)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = args.runtime.resolve(strict=True) if args.runtime else assemble(args, output, source)
    if args.runtime and prepare.digest(runtime.parent / "manifest.json") != args.manifest_sha256:
        raise ValueError("runtime manifest differs from supplied trusted identity")
    recorded = json.loads((runtime / "share/nvbroadcast-runtime-provenance/inputs.json").read_text())
    if recorded["application"] != source:
        raise ValueError("runtime application differs from requested source")
    results = []
    for family in args.families:
        result = wrap(args, runtime, output, source, family)
        results.append(result)
        (output / "packages.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

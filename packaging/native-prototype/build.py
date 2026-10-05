#!/usr/bin/env python3
"""Wrap a verified complete CPU runtime in NON-RELEASE DEB/RPM prototypes.

Both layouts consume the same bytes. No dependency resolution, native build,
stripping, Python byte-compilation, or destination-time pip is permitted here.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME_TOOLS = HERE.parent / "runtime-prototype"
sys.path.insert(0, str(RUNTIME_TOOLS))
from assemble import inventory
from prepare import digest

PREFIX = Path("/usr/lib/nvbroadcast/runtime")
APP = "nvbroadcast"
SELF = "nvbroadcast-cpu"
RUNTIME = "nvbroadcast-runtime-cpu"
CAPABILITY = "nvbroadcast-runtime"
DEPENDENCIES = {
    "deb": ("ca-certificates, gir1.2-gtk-4.0, gir1.2-adw-1, gir1.2-gstreamer-1.0, "
            "gir1.2-gst-plugins-base-1.0, gstreamer1.0-plugins-base, "
            "gstreamer1.0-plugins-good, libgomp1, libsndfile1, libportaudio2, "
            "libgirepository-1.0-1, libgl1, libgles2"),
    "rpm": ("ca-certificates, gtk4, libadwaita, gstreamer1-plugins-base, "
            "gstreamer1-plugins-good, libgomp, libsndfile, portaudio, "
            "gobject-introspection, mesa-libGL, libglvnd-gles"),
}


def app_files(runtime: Path, manifest: dict) -> set[str]:
    """Partition using the installed wheel's RECORD, rejecting escapes and lies."""
    records = list(runtime.glob("lib/python*/site-packages/nvbroadcast-*.dist-info/RECORD"))
    if len(records) != 1:
        raise ValueError("expected exactly one application RECORD")
    record = records[0]
    result = set()
    with record.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 3 or Path(row[0]).is_absolute():
                raise ValueError("invalid RECORD entry")
            path = (record.parent.parent / row[0]).resolve(strict=True)
            if not path.is_relative_to(runtime.resolve()):
                raise ValueError("RECORD path escapes runtime")
            relative = str(path.relative_to(runtime.resolve()))
            if relative in result or relative not in manifest or not path.is_file():
                raise ValueError("duplicate or missing RECORD file")
            if path != record:
                expected = "sha256=" + base64.urlsafe_b64encode(
                    hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
                if row[1] != expected or row[2] != str(path.stat().st_size):
                    raise ValueError("RECORD hash or size mismatch")
            result.add(relative)
    actual_app = {p for p in manifest if (p.startswith(str(record.parent.relative_to(runtime)) + "/")
                  or p.startswith(str(record.parent.parent.relative_to(runtime)) + "/nvbroadcast/"))
                  and "sha256" in manifest[p]}
    if not actual_app <= result:
        raise ValueError("application files missing from RECORD")
    return result


def versions(family: str, revision: int) -> str:
    return f"1.5.2-900{'~' if family == 'deb' else '.'}prototype{revision}"


def write(path: Path, value: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    path.chmod(mode)


def hooks(family: str, kind: str, version: str, action: str) -> str:
    template = (HERE / "package-hooks.sh").read_text()
    for key, value in {"FAMILY": family, "KIND": kind, "VERSION": version}.items():
        template = template.replace(f"@{key}@", value)
    return "set -- " + action + "\n" + template


def version_check(family: str, kind: str, version: str) -> str:
    names = [SELF] if kind == "self" else [APP, RUNTIME]
    body = "#!/bin/sh\nset -eu\n"
    for name in names:
        if family == "deb":
            query = f"dpkg-query -W -f='${{Status}} ${{Version}}' {name}"
            expected = f"install ok installed {version}"
        else:
            query = f"rpm -q --qf '%{{VERSION}}-%{{RELEASE}}' {name}"
            expected = version
        body += f"actual=$({query} 2>/dev/null || true)\n"
        if family == "deb" and name == names[0]:
            # dpkg has not committed this app's configured state while its
            # postinst is executing. Runtime dependencies must already be ready.
            body += (f"if [ \"${{1:-}}\" = --configure ] && [ \"$actual\" = 'install ok half-configured {version}' ]; then\n"
                     f"    actual='{expected}'\nfi\n")
        body += (
                 f"if [ \"$actual\" != '{expected}' ]; then\n"
                 f"    echo 'NVBroadcast prototype: incomplete or mismatched {name}; repair the native transaction.' >&2\n"
                 "    exit 78\nfi\n")
    return body


def populate(runtime: Path, manifest: dict, selected: set[str], stage: Path) -> None:
    prefix = stage / PREFIX.relative_to("/")
    prefix.mkdir(parents=True)
    for relative in sorted(selected):
        entry = manifest[relative]
        source, target = runtime / relative, prefix / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if "symlink" in entry:
            target.symlink_to(entry["symlink"])
        elif "sha256" in entry:
            shutil.copyfile(source, target)
            target.chmod(entry["mode"])
        else:
            target.mkdir(exist_ok=True)
            target.chmod(entry["mode"])


def integration(stage: Path, family: str, kind: str, version: str) -> None:
    base = stage / "usr/lib/nvbroadcast"
    write(base / "check-packages", version_check(family, kind, version), 0o755)
    for name, module in (("nvbroadcast", "nvbroadcast"), ("nvbroadcast-vcam", "nvbroadcast.vcam_service")):
        write(stage / "usr/bin" / name,
              "#!/bin/sh\nset -eu\n"
              "if [ -e /usr/lib/nvbroadcast/.transaction ]; then\n"
              "    echo 'NVBroadcast prototype: interrupted native transaction; reinstall the exact package set.' >&2\n"
              "    exit 78\nfi\n"
              "/usr/lib/nvbroadcast/check-packages\n"
              f'exec {PREFIX}/bin/python -I -B -m {module} "$@"\n', 0o755)
    for path in ("share/applications/com.doczeus.NVBroadcast.desktop",
                 "share/metainfo/com.doczeus.NVBroadcast.metainfo.xml",
                 "share/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg",
                 "share/doc/nvbroadcast/NOTICE", "share/doc/nvbroadcast/CONTRIBUTORS.md"):
        target = stage / "usr" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(PREFIX / path)
    write(stage / "usr/lib/systemd/user/nvbroadcast-vcam.service",
          "[Unit]\nDescription=NVBroadcast prototype virtual camera\nAfter=graphical-session.target\n"
          "[Service]\nExecStart=/usr/bin/nvbroadcast-vcam\nRestart=on-failure\nRestartSec=3\n"
          "[Install]\nWantedBy=graphical-session.target\n")


def control(kind: str, version: str, size: int) -> str:
    name = {"self": SELF, "app": APP, "runtime": RUNTIME}[kind]
    lines = [f"Package: {name}", f"Version: {version}", "Architecture: amd64",
             "Maintainer: NVBroadcast prototype <harshit@kshoonya.com>",
             f"Installed-Size: {(size + 1023) // 1024}", "Section: video", "Priority: optional"]
    if kind == "app":
        lines += [f"Depends: {RUNTIME} (= {version})"]
    else:
        lines += [f"Depends: {DEPENDENCIES['deb']}", f"Provides: {CAPABILITY} (= {version})" +
                  (f", {APP} (= {version})" if kind == "self" else ""),
                  f"Conflicts: {CAPABILITY}" +
                  (f", {APP} (<< 1.5.2-900~prototype1)" if kind == "self" else "")]
    if kind == "self":
        lines += [f"Replaces: {APP} (<< 1.5.2-900~prototype1)"]
    lines += ["Description: NON-RELEASE offline CPU package lifecycle prototype",
              " For isolated maintainer tests only. Not a supported distribution artifact."]
    return "\n".join(lines) + "\n"


def rpm_spec(kind: str, version: str) -> str:
    name = {"self": SELF, "app": APP, "runtime": RUNTIME}[kind]
    app_version, release = version.split("-", 1)
    text = (f"Name: {name}\nVersion: {app_version}\nRelease: {release}\n"
            "Summary: NON-RELEASE offline CPU package lifecycle prototype\n"
            "License: LicenseRef-Unreviewed-Prototype\n"
            "BuildArch: x86_64\nAutoReqProv: no\n")
    if kind == "app":
        text += f"Requires: {RUNTIME} = {version}\n"
    else:
        text += f"Requires: {DEPENDENCIES['rpm']}\nProvides: {CAPABILITY} = {version}\nConflicts: {CAPABILITY}\n"
    if kind == "self":
        text += f"Provides: {APP} = {version}\nObsoletes: {APP} < 1.5.2-900\n"
    # This tests byte-preserving wrapping, not RPM distribution policy or ELF QA.
    text += ("%global __os_install_post %{nil}\n%global _build_id_links none\n"
             "%global debug_package %{nil}\n%global _binary_payload w8.zstdio\n"
             "%description\nFor isolated maintainer tests only; not a release artifact.\n"
             "%install\nmkdir -p %{buildroot}\ncp -a /stage/. %{buildroot}/\n"
             "%pretrans\n" + hooks("rpm", kind, version, "prepare") +
             "\n%posttrans\n" + hooks("rpm", kind, version, "finish") +
             "\n%postun\nif [ \"$1\" -eq 0 ]; then\n" + hooks("rpm", kind, version, "remove") +
             "\nfi\n%files -f /work/files.txt\n%defattr(-,root,root,-)\n")
    # '%' inside the generated shell queries is RPM syntax; preserve its printf
    # and query strings while allowing the actual spec macros above to expand.
    return text.replace("'%s\\n'", "'%%s\\n'").replace("'%{NAME}'", "'%%{NAME}'")


def build_one(family: str, kind: str, revision: int, runtime: Path, manifest: dict,
              application: set[str], directory: Path, image: str, epoch: int) -> dict:
    os.umask(0o022)
    name = {"self": SELF, "app": APP, "runtime": RUNTIME}[kind]
    version = versions(family, revision)
    work = directory / family / f"{kind}-{revision}"
    stage = work / "stage"
    selected = set(manifest) if kind == "self" else (application if kind == "app" else set(manifest) - application)
    populate(runtime, manifest, selected, stage)
    if kind != "runtime":
        integration(stage, family, kind, version)
    # A real adapter version changes across upgrade/rollback; Python payloads
    # are deliberately identical so this cannot claim a runtime-content upgrade.
    write(stage / f"usr/lib/nvbroadcast/{kind}.json", json.dumps({
        "prototype": True, "family": family, "package": name, "version": version,
        "runtime_manifest_sha256": digest(runtime.parent / "manifest.json"),
    }, sort_keys=True) + "\n")
    content = {}
    for p in sorted(stage.rglob("*")):
        content["/" + str(p.relative_to(stage))] = (
            {"symlink": os.readlink(p)} if p.is_symlink() else
            {"sha256": digest(p), "mode": stat.S_IMODE(p.stat().st_mode)} if p.is_file() else
            {"mode": stat.S_IMODE(p.stat().st_mode)})
    (work / "content.json").write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file() and not p.is_symlink())
    if family == "deb":
        write(stage / "DEBIAN/control", control(kind, version, size))
        write(stage / "DEBIAN/preinst", "#!/bin/sh\n" + hooks(family, kind, version, "prepare"), 0o755)
        write(stage / "DEBIAN/postinst", "#!/bin/sh\n[ \"$1\" != configure ] || {\n" +
              hooks(family, kind, version, "finish") + "\n}\n", 0o755)
        write(stage / "DEBIAN/postrm", "#!/bin/sh\ncase \"$1\" in remove|purge)\n" +
              hooks(family, kind, version, "remove") + "\n;; esac\n", 0o755)
        artifact = work / f"{name}_{version}_amd64.deb"
        command = ["dpkg-deb", "--root-owner-group", "-Zzstd", "-z8", "--threads-max=2",
                   "--build", "/stage", f"/work/{artifact.name}"]
    else:
        write(work / "package.spec", rpm_spec(kind, version))
        write(work / "files.txt", "\n".join(
            ("%dir " if p.is_dir() and not p.is_symlink() else "") +
            json.dumps("/" + str(p.relative_to(stage))).replace("%", "%%")
            for p in sorted(stage.rglob("*")) if str(p.relative_to(stage)) not in
            {"usr", "usr/bin", "usr/lib", "usr/share", "usr/share/applications", "usr/share/metainfo",
             "usr/share/icons", "usr/share/icons/hicolor", "usr/share/icons/hicolor/scalable",
             "usr/share/icons/hicolor/scalable/apps", "usr/share/doc", "usr/lib/systemd", "usr/lib/systemd/user"}
        ) + "\n")
        artifact = work / f"build/RPMS/x86_64/{name}-{version}.x86_64.rpm"
        command = ["rpmbuild", "-bb", "--define", "_topdir /work/build",
                   "--define", "_buildhost prototype.invalid", "--define", "use_source_date_epoch_as_buildtime 1",
                   "--define", "clamp_mtime_to_source_date_epoch 1", "/work/package.spec"]
    for path in [stage, *stage.rglob("*")]:
        os.utime(path, (epoch, epoch), follow_symlinks=False)
    with (work / "build.log").open("w") as log:
        subprocess.run(["docker", "run", "--rm", "--init", "--network=none", "--pull=never",
                        "--user", f"{os.getuid()}:{os.getgid()}", "-e", f"SOURCE_DATE_EPOCH={epoch}",
                        "-v", f"{stage}:/stage:ro", "-v", f"{work}:/work", image, *command],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"family": family, "kind": kind, "revision": revision, "name": name, "version": version,
            "artifact": str(artifact.relative_to(directory)), "sha256": digest(artifact),
            "bytes": artifact.stat().st_size, "installed_file_bytes": size,
            "content": str((work / "content.json").relative_to(directory)), "builder_image": image}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deb-image", required=True)
    parser.add_argument("--rpm-image", required=True)
    parser.add_argument("--families", nargs="+", choices=("deb", "rpm"), default=("deb", "rpm"))
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error("build as an ordinary user")
    for value in (args.deb_image, args.rpm_image):
        if not value.startswith("sha256:") or len(value) != 71:
            parser.error("use inspected immutable local builder image IDs")
    runtime = args.runtime.resolve(strict=True)
    manifest_path = runtime.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != args.manifest_sha256 or inventory(runtime) != manifest:
        parser.error("runtime or manifest does not match the supplied identity")
    application = app_files(runtime, manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.umask(0o022)
    epoch = json.loads((RUNTIME_TOOLS / "inputs.json").read_text())["application"]["source_date_epoch"]
    results = []
    for family, image in (("deb", args.deb_image), ("rpm", args.rpm_image)):
        if family not in args.families:
            continue
        for revision in (1, 2):
            for kind in ("self", "app", "runtime"):
                result = build_one(family, kind, revision, runtime, manifest, application, output, image, epoch)
                results.append(result)
                (output / "packages.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{family} {kind} {revision}: {result['bytes']} bytes", flush=True)
    if inventory(runtime) != manifest:
        raise ValueError("input runtime changed during wrapping")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch and inspect one explicitly approved hosted CPU artifact, without install.

The Actions API is the transport/source trust anchor, not an application release
signature. Package bytes are decoded in pinned, offline, read-only containers.
No package scriptlet or interpreter from the artifact is executed.

Run this tool from the trusted finalization checkout after builder_delivery.py
has authenticated and loaded its exact builders. Producer files are compared
with git show only; they never supply validator or signing code. The approved
digest must agree with the live Actions metadata before any artifact download.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import subprocess
import tarfile
import tempfile
import threading
import tomllib
import uuid
import zipfile

REPO = "Hkshoonya/nvidia-broadcast-linux"
REPOSITORY_ID = 1184001936
WORKFLOW = ".github/workflows/native-cpu-candidate.yml"
GIB = 1024**3
MIB = 1024**2
PREFIX = "/usr/lib/nvbroadcast/runtime/"
IMAGES = {
    "deb": "sha256:1275fea2392e54fa240ca3e3917b5a050cbcd081797f1208ec7a9feff9204e4b",
    "rpm": "sha256:e494d493b7f5a50fbcf741b86c46ca6f1c25a9bb90016068ff76bcfb4c78708c",
}
ADAPTER_PATHS = ["packaging/native-runtime/" + name for name in (
    "build.py", "builders.json", "package-hooks.sh", "validate_install.py", "recovery_bootstrap.py",
    "pylock.linux-x86_64-cp313-cpu.toml", "pylock.linux-x86_64-cp313-cuda.toml")]
ADAPTER_PATHS += ["packaging/runtime-prototype/" + name for name in (
    "prepare.py", "assemble.py", "inputs.json", "pylock.build.toml", "build-bindings.sh")]
ADAPTER_PATHS += ["packaging/native-prototype/build.py", "scripts/recover_native_rpm.py",
                  "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def bounded(command, *, cwd=None, limit=4*MIB, timeout=60):
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE)
    timer = threading.Timer(timeout, process.kill)
    timer.daemon = True
    timer.start()
    try:
        data = process.stdout.read(limit + 1)
        require(len(data) <= limit, "command output exceeds evidence bound")
        require(process.wait(timeout=30) == 0, "bounded command failed or timed out")
        return data
    finally:
        timer.cancel()
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)


class BoundedTarStream:
    """Check raw 512-byte headers before tarfile allocates extension payloads."""
    def __init__(self, stream):
        self.stream = stream
        self.buffer = bytearray()
        self.data_blocks = self.headers = self.total = 0
        self.eof = False
        self.terminal_padding = False

    def read(self, size):
        require(0 <= size <= MIB, "oversize tar stream read")
        while len(self.buffer) < size and not self.eof:
            block = self.stream.read(512)
            if not block:
                self.eof = True
                break
            require(len(block) == 512, "truncated raw tar block")
            self.total += 512
            require(self.total <= 5*GIB, "decoded tar exceeds total byte bound")
            if self.data_blocks:
                self.data_blocks -= 1
            elif any(block):
                require(not self.terminal_padding, "nonzero data after tar terminal padding")
                self.headers += 1
                require(self.headers <= 100000, "excess raw tar headers")
                header = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
                require(0 <= header.size <= GIB, "raw tar member exceeds bound")
                if header.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
                    require(header.size <= MIB, "tar extension exceeds allocation bound")
                self.data_blocks = (header.size + 511) // 512
            else:
                self.terminal_padding = True
            self.buffer.extend(block)
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data


def git_bytes(root, source, path):
    return bounded(["git", "--no-replace-objects", "show", f"{source}:{path}"], cwd=root)


def source_identity(repo, source):
    """Bind comparisons to a real ancestor of the trusted tool checkout."""
    require(isinstance(source, str) and re.fullmatch(r"[a-f0-9]{40}", source), "source must be an exact commit")
    trusted = bounded(["git", "--no-replace-objects", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    bounded(["git", "--no-replace-objects", "merge-base", "--is-ancestor", source, trusted], cwd=repo)
    tree = bounded(["git", "--no-replace-objects", "rev-parse", source + "^{tree}"], cwd=repo).decode().strip()
    return {"trusted_checkout": trusted, "source_tree": tree,
            "source_workflow_sha256": digest(git_bytes(repo, source, WORKFLOW))}


def path_name(name, *, absolute=False):
    require(isinstance(name, str) and name and "\x00" not in name and "\\" not in name,
            "invalid archive path")
    name = name.removeprefix("./").rstrip("/")
    require(name and not name.startswith("/") and all(p not in ("", ".", "..") for p in name.split("/")),
            "noncanonical archive path")
    return "/" + name if absolute else name


def live_metadata(args):
    def api(path):
        return parse_json(bounded([args.gh, "api", "--hostname", "github.com", f"repos/{REPO}/{path}"]))
    run = api(f"actions/runs/{args.run_id}/attempts/{args.attempt}")
    require(all(type(value) is int for value in (run["id"], run["run_attempt"],
                run["repository"]["id"], run["head_repository"]["id"]))
            and run["id"] == args.run_id and run["run_attempt"] == args.attempt
            and run["head_sha"] == args.source and run["head_branch"] == "main"
            and run["repository"]["full_name"] == REPO and run["head_repository"]["full_name"] == REPO
            and run["repository"]["id"] == run["head_repository"]["id"] == REPOSITORY_ID
            and run["event"] == "workflow_dispatch" and run["path"] == WORKFLOW
            and run["status"] == "completed" and run["conclusion"] == "success", "run identity or success differs")
    artifact = api(f"actions/artifacts/{args.artifact_id}")
    require(all(type(value) is int for value in (artifact["id"], artifact["workflow_run"]["id"],
                artifact["workflow_run"]["repository_id"], artifact["workflow_run"]["head_repository_id"]))
            and artifact["id"] == args.artifact_id and artifact["expired"] is False
            and artifact["name"] == f"native-cpu-candidate-{args.source}-{args.run_id}-{args.attempt}"
            and artifact["workflow_run"]["id"] == args.run_id
            and artifact["workflow_run"]["head_sha"] == args.source
            and artifact["workflow_run"]["head_branch"] == "main"
            and artifact["workflow_run"]["repository_id"] == artifact["workflow_run"]["head_repository_id"] == REPOSITORY_ID
            and type(artifact["size_in_bytes"]) is int and 0 < artifact["size_in_bytes"] <= 2*GIB
            and re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["digest"]), "artifact identity differs")
    require(artifact["digest"] == args.expected_digest, "artifact digest differs from explicit approval")
    return run, artifact


def download_artifact(args, metadata, destination, *, timeout=600):
    """Stream to one new private file, enforcing size/hash before use or retention."""
    expected_bytes = metadata["size_in_bytes"]
    require(type(expected_bytes) is int and 0 < expected_bytes <= 2*GIB, "invalid download size")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", args.expected_digest)
            and metadata["digest"] == args.expected_digest, "download digest is not approved")
    command = [args.gh, "api", "--hostname", "github.com",
               f"repos/{REPO}/actions/artifacts/{args.artifact_id}/zip"]
    process = timer = None
    created = False
    complete = False
    try:
        with destination.open("xb") as target:
            created = True
            os.fchmod(target.fileno(), 0o600)
            # Do not log request diagnostics or binary responses. The token is
            # used only by gh for its fixed github.com API request.
            environment = dict(os.environ)
            environment.pop("GH_DEBUG", None)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment)
            timer = threading.Timer(timeout, process.kill)
            timer.daemon = True
            timer.start()
            size, hashed = 0, hashlib.sha256()
            while chunk := process.stdout.read1(min(MIB, expected_bytes - size + 1)):
                size += len(chunk)
                require(size <= expected_bytes, "download exceeds approved byte count")
                hashed.update(chunk)
                target.write(chunk)
            require(process.wait(timeout=30) == 0, "artifact download failed or timed out")
            require(size == expected_bytes, "artifact download is truncated")
            require("sha256:" + hashed.hexdigest() == args.expected_digest, "download digest differs from approval")
            target.flush()
            os.fsync(target.fileno())
        destination.chmod(0o400)
        complete = True
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        if created and not complete:
            destination.unlink(missing_ok=True)


@contextmanager
def create_member(output, relative):
    """Create an exclusive regular file beneath real directories, without links."""
    relative = path_name(relative)
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        for name in parts[:-1]:
            try:
                os.mkdir(name, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            os.fchmod(stream.fileno(), 0o400)
    finally:
        os.close(directory)


def unpack_zip(path, metadata, output, version):
    names = {
        f"dist/native-cpu/deb/nvbroadcast_{version}_amd64.deb",
        f"dist/native-cpu/rpm/build/RPMS/x86_64/nvbroadcast-{version}.x86_64.rpm",
        *["dist/native-cpu/" + n for n in ("packages.json", "build-provenance.json", "payload/manifest.json",
                                         "deb/content.json", "rpm/content.json", "assembly/pylock.toml")],
        *["dist/builder-delivery/" + n for n in ("authenticated-builders.json", "capacity.json", "host-tools.json")],
        "packaging/native-runtime/builder-delivery.json", "packaging/native-runtime/builders.json",
    }
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size == metadata["size_in_bytes"], "ZIP type/size differs")
        require("sha256:" + hashlib.file_digest(stream, "sha256").hexdigest() == metadata["digest"], "ZIP digest differs from live API")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            require(len(members) == len(names) and {m.filename for m in members} == names, "unexpected ZIP file set")
            evidence = 0
            for member in members:
                name = path_name(member.filename)
                mode = member.external_attr >> 16
                require(not member.flag_bits & 1 and not member.is_dir()
                        and stat.S_IFMT(mode) in (0, stat.S_IFREG), "encrypted or special ZIP entry")
                require(member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                        "unsupported ZIP compression")
                package = name.endswith((".deb", ".rpm"))
                require(0 < member.file_size <= (GIB if package else 16*MIB), "ZIP entry exceeds bound")
                if not package:
                    evidence += member.file_size
            require(evidence <= 32*MIB, "aggregate provenance exceeds bound")
            for member in members:
                with archive.open(member) as source, create_member(output, member.filename) as sink:
                    total = 0
                    while chunk := source.read(MIB):
                        total += len(chunk)
                        require(total <= member.file_size, "ZIP size changed")
                        sink.write(chunk)
                    require(total == member.file_size, "truncated ZIP entry")
        require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) ==
                tuple(getattr(os.fstat(stream.fileno()), n) for n in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
                "ZIP changed during validation")
    return {"files": len(names), "provenance_bytes": evidence}


def docker_command(package, family, name, command):
    require("," not in str(package), "unsupported mount path")
    return ["docker", "run", "--name", name, "--rm", "--init", "--network=none", "--pull=never", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=1g", "--cpus=2",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777",
            "--user=65534:65534", "--mount", f"type=bind,source={package},target=/package.{family},readonly",
            IMAGES[family], *command]


def docker_query(package, family, command):
    name = "nvb-artifact-query-" + uuid.uuid4().hex[:12]
    try:
        return bounded(docker_command(package, family, name, command))
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        require(not bounded(["docker", "ps", "-aq", "--filter", "name=^/" + name + "$" ]).strip(), "decoder query cleanup failed")


def decoded_inventory(package, family, *, control=False):
    name = "nvb-artifact-decode-" + uuid.uuid4().hex[:12]
    command = (["dpkg-deb", "--ctrl-tarfile" if control else "--fsys-tarfile", "/package.deb"] if family == "deb"
               else ["rpm2archive", "--nocompression", "/package.rpm"])
    inventory, modes, kept = {}, {}, {}
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(docker_command(package, family, name, command), stdout=subprocess.PIPE, stderr=errors)
        stream = BoundedTarStream(process.stdout)
        def stop():
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            if process.poll() is None:
                process.kill()
        timer = threading.Timer(600, stop)
        timer.daemon = True
        timer.start()
        total = 0
        try:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for entry in archive:
                    if entry.name in (".", "./") and entry.isdir():
                        continue
                    path = path_name(entry.name, absolute=True)
                    require(path not in inventory and len(inventory) < 50000, "duplicate/excess payload entry")
                    require(entry.uid == 0 and entry.gid == 0
                            and (entry.issym() or not entry.mode & 0o6022), "unsafe payload owner/mode")
                    modes[path] = entry.mode & 0o7777
                    if entry.isdir():
                        item = {"directory": True}
                    elif entry.issym():
                        require(entry.linkname and "\x00" not in entry.linkname, "invalid symbolic link")
                        if path.startswith(PREFIX):
                            require(not entry.linkname.startswith("/")
                                    and posixpath.normpath(posixpath.join(posixpath.dirname(path), entry.linkname)).startswith(PREFIX),
                                    "runtime symlink escapes private prefix")
                        item = {"symlink": entry.linkname}
                    elif entry.isfile():
                        require(0 <= entry.size <= GIB, "oversize payload member")
                        total += entry.size
                        require(total <= 4*GIB, "payload expands beyond bound")
                        member_stream = archive.extractfile(entry)
                        h, size, chunks = hashlib.sha256(), 0, []
                        retain = control or path in {
                            "/usr/bin/nvbroadcast-recover-rpm", "/usr/lib/nvbroadcast/package.json",
                            "/usr/lib/nvbroadcast/runtime-manifest.json", "/usr/lib/nvbroadcast/validate-install.py",
                            PREFIX + "share/nvbroadcast-runtime-provenance/pylock.toml",
                            PREFIX + "share/nvbroadcast-runtime-provenance/inputs.json",
                            PREFIX + "share/nvbroadcast-runtime-provenance/builders.json",
                            PREFIX + "share/nvbroadcast-runtime-provenance/selection.json"}
                        require(not retain or entry.size <= 16*MIB, "oversize retained member")
                        while chunk := member_stream.read(MIB):
                            size += len(chunk)
                            require(size <= entry.size, "member exceeds declared size")
                            h.update(chunk)
                            if retain:
                                chunks.append(chunk)
                        require(size == entry.size, "truncated payload member")
                        item = {"sha256": h.hexdigest(), "size": size}
                        if retain:
                            kept[path] = b"".join(chunks)
                    else:
                        raise ValueError("unsupported special or hardlink payload entry")
                    inventory[path] = item
            padding = 0
            while chunk := stream.read(MIB):
                padding += len(chunk)
                require(padding <= MIB and not any(chunk), "unexpected trailing payload data")
            require(process.wait(timeout=30) == 0, "archive decoder failed")
            return inventory, modes, kept, total
        finally:
            timer.cancel()
            process.stdout.close()
            if process.poll() is None:
                stop()
                process.wait(timeout=30)
            errors.seek(0)
            error = errors.read(4096)
            if process.returncode:
                print(error.decode(errors="replace"))
            require(not bounded(["docker", "ps", "-aq", "--filter", "name=^/" + name + "$" ]).strip(), "decoder cleanup failed")


def expected_hook(git, action, version, family, helper):
    script = git("packaging/native-runtime/package-hooks.sh").decode().replace("@VERSION@", version).replace("@VARIANT@", "cpu")
    bootstrap = ""
    if family == "rpm" and action in {"prepare", "finish"}:
        bootstrap = ("recovery_bootstrap() {\n/usr/bin/python3 -IB - \"$1\" <<'NVB_BOOTSTRAP_PY'\n"
                     + git("packaging/native-runtime/recovery_bootstrap.py").decode()
                     + "\nprint('NV Broadcast recovery copy:', bootstrap(sys.argv[1], " + repr(helper) + "))\nNVB_BOOTSTRAP_PY\n}\n")
    script = script.replace("@RPM_BOOTSTRAP_FUNCTION@", bootstrap)
    script = script.replace("@RPM_BOOTSTRAP_PREPARE@", "recovery_bootstrap prepare" if bootstrap else ":")
    script = script.replace("@RPM_BOOTSTRAP_FINISH@", "recovery_bootstrap finish" if bootstrap else ":")
    legacy = ""
    if action == "prepare":
        query = ("dpkg-query -S /opt/nvbroadcast/scripts/install_runtime_variant.py" if family == "deb" else
                 "rpm -qf --qf '%{NAME}: /opt/nvbroadcast/scripts/install_runtime_variant.py' /opt/nvbroadcast/scripts/install_runtime_variant.py")
        legacy = (f"owner=$({query} 2>/dev/null || true)\n"
                  "if [ \"$owner\" = 'nvbroadcast: /opt/nvbroadcast/scripts/install_runtime_variant.py' ]; then\n"
                  "    touch /usr/lib/nvbroadcast/.legacy-runtime\nfi\n")
    return "#!/bin/sh\nset -eu\nset -- " + action + "\n" + script + "\n" + legacy


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run-id", "attempt", "artifact-id"):
        parser.add_argument("--" + option, required=True, type=int)
    for option in ("source", "expected-digest", "package-revision"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--zip", type=Path, help="Inspect this already-downloaded ZIP instead of fetching it")
    for option in ("repo", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args(argv)
    require(re.fullmatch(r"[a-f0-9]{40}", args.source)
            and re.fullmatch(r"[1-9][0-9]*", args.package_revision), "invalid source/revision")
    require(all(type(value) is int and value > 0 for value in (args.run_id, args.attempt, args.artifact_id)),
            "run, attempt and artifact IDs must be positive integers")
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", args.expected_digest), "expected digest must be an explicit SHA-256")
    return args


def validate(args):
    # The trusted workflow checks out its own main revision, never producer
    # tools. Only an ancestor can supply the source blobs used for comparison.
    identity = source_identity(args.repo, args.source)
    git = lambda path: git_bytes(args.repo, args.source, path)
    project_version = tomllib.loads(git("pyproject.toml").decode())["project"]["version"]
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", project_version), "invalid project version")
    version = project_version + "-" + args.package_revision + ".cpu"
    run, artifact = live_metadata(args)
    require(not args.output.exists() and not args.output.is_symlink(), "output must be a new directory")
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "running", "driver_sha256": file_digest(Path(__file__)), "source": args.source,
              "repository": REPO, "repository_id": REPOSITORY_ID,
              "expected_artifact_digest": args.expected_digest,
              **identity,
              "run_id": args.run_id, "run_attempt": args.attempt, "artifact_id": args.artifact_id,
              "artifact_name": artifact["name"], "artifact_bytes": artifact["size_in_bytes"], "artifact_digest": artifact["digest"],
              "version": version, "packages": {}, "scope": "Read-only artifact/source validation; unsigned CPU candidates, no install or physical-device claim"}
    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    save()
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    try:
        archive = args.zip
        if archive is None:
            archive = output / "candidate.zip"
            download_artifact(args, artifact, archive)
        result["zip"] = unpack_zip(archive, artifact, output, version)
        base = output / "dist/native-cpu"
        read = lambda path: (base / path).read_bytes()
        packages = parse_json(read("packages.json"))
        require(len(packages) == 2 and {p["family"] for p in packages} == {"deb", "rpm"}, "invalid package set")
        source = {"revision": args.source, "version": project_version,
                  "source_date_epoch": int(bounded(["git", "--no-replace-objects", "show", "-s", "--format=%ct", args.source], cwd=args.repo))}
        adapter = {"revision": args.source,
                   "tree": result["source_tree"],
                   "files": {path: digest(git(path)) for path in ADAPTER_PATHS}}
        require(parse_json(read("build-provenance.json")) == {"application": source, "adapter": adapter}, "build provenance differs from source")
        for name in ("builder-delivery.json", "builders.json"):
            require((output / "packaging/native-runtime" / name).read_bytes() == git("packaging/native-runtime/" + name), "builder lock differs from source")
        pins = parse_json(git("packaging/native-runtime/builders.json"))
        delivery = parse_json(git("packaging/native-runtime/builder-delivery.json"))
        require(delivery["status"] == "promoted", "builder delivery is not promoted")
        report = parse_json((output / "dist/builder-delivery/authenticated-builders.json").read_bytes())
        require(report == {"status": "authenticated_import_delivery", "archive_sha256": delivery["archive"]["sha256"],
                "archive_bytes": delivery["archive"]["bytes"], "blob_count": 23,
                "builder_images": {role: pin["image"] for role, pin in pins.items()},
                "origin": "maintainer-qualified-historical-snapshot", "original_build_provenance_authenticated": False,
                "installed_inventory": {role: {"image": pin["image"], "package_count": len(pin["packages"])} for role, pin in pins.items()}},
                "builder authentication/inventory report differs")
        host_tools = parse_json((output / "dist/builder-delivery/host-tools.json").read_bytes())
        build_lock = tomllib.loads(git("packaging/runtime-prototype/pylock.build.toml").decode())
        packaging = next(p for p in build_lock["packages"] if p["name"] == "packaging")
        require(host_tools["interpreter"] == "/usr/bin/python3" and host_tools["isolated"] is True
                and host_tools["site_disabled"] is True and host_tools["packaging_version"] == packaging["version"]
                and host_tools["wheel_sha256"] == packaging["wheels"][0]["hashes"]["sha256"], "isolated host tools differ")
        capacity = parse_json((output / "dist/builder-delivery/capacity.json").read_bytes())
        require(capacity["required_free_bytes"] == 12*GIB
                and set(capacity["observed_free_bytes"]) == {"GITHUB_WORKSPACE", "RUNNER_TEMP", "HOME"}
                and all(n >= 12*GIB for n in capacity["observed_free_bytes"].values()), "runner capacity gate differs")
        manifest_bytes, lock_bytes = read("payload/manifest.json"), read("assembly/pylock.toml")
        manifest, lock = parse_json(manifest_bytes), tomllib.loads(lock_bytes.decode())
        base_lock_bytes = git("packaging/native-runtime/pylock.linux-x86_64-cp313-cpu.toml")
        require(lock_bytes.startswith(base_lock_bytes), "assembled dependency lock prefix differs")
        base_lock = tomllib.loads(base_lock_bytes.decode())
        require({k: v for k, v in lock.items() if k != "packages"} == {k: v for k, v in base_lock.items() if k != "packages"}
                and lock["packages"][:-1] == base_lock["packages"], "dependency lock differs")
        app_lock = lock["packages"][-1]
        require(app_lock["name"] == "nvbroadcast" and app_lock["version"] == project_version
                and set(app_lock) == {"name", "version", "archive"}
                and app_lock["archive"]["path"] == f"nvbroadcast-{project_version}-py3-none-any.whl"
                and re.fullmatch(r"[a-f0-9]{64}", app_lock["archive"]["hashes"]["sha256"]), "invalid application wheel identity")
        key = git("packaging/keys/RPM-GPG-KEY-nvbroadcast.asc")
        helper_source = git("scripts/recover_native_rpm.py").decode()
        require(helper_source.count('PUBLIC_ARMOR = b""') == 1 and digest(key) in helper_source, "recovery public key differs")
        helper = helper_source.replace('PUBLIC_ARMOR = b""', 'PUBLIC_ARMOR = ' + repr(key)).encode()
        app_paths = bounded(["git", "--no-replace-objects", "ls-tree", "-r", "--name-only", args.source, "--", "src/nvbroadcast"], cwd=args.repo).decode().splitlines()
        app_expected = {p.removeprefix("src/nvbroadcast/"): digest(git(p)) for p in app_paths if p.endswith(".py")}
        for package in packages:
            family = package["family"]
            expected_relative = (f"deb/nvbroadcast_{version}_amd64.deb" if family == "deb" else
                                 f"rpm/build/RPMS/x86_64/nvbroadcast-{version}.x86_64.rpm")
            require(package["artifact"] == expected_relative and package["content"] == f"{family}/content.json"
                    and package["version"] == version and package["variant"] == "cpu"
                    and package["source"] == source and package["adapter"] == adapter
                    and package["builder_image"] == pins[family]["image"] == IMAGES[family]
                    and package["runtime_manifest_sha256"] == digest(manifest_bytes)
                    and package["content_sha256"] == digest(read(package["content"])), "package provenance differs")
            path = base / expected_relative
            require(path.stat().st_size == package["bytes"] and file_digest(path) == package["sha256"], "package bytes differ")
            # Decoder container runs unprivileged and can read this exact file only.
            path.chmod(0o444)
            require(bounded(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGES[family]]).decode().strip() == IMAGES[family], "decoder pin differs")
            content = parse_json(read(package["content"]))
            expected_content = {p: entry for p, entry in content.items()
                                if family != "rpm" or not (entry.get("directory") and len(PurePosixPath(p).parts) < 4)}
            actual, modes, kept, total = decoded_inventory(path, family)
            require(actual == expected_content, "full package payload differs from content inventory")
            actual_runtime = {p.removeprefix(PREFIX): dict({"mode": modes[p]}, **{k: v for k, v in item.items() if k != "directory"})
                              for p, item in actual.items() if p.startswith(PREFIX)}
            require(actual_runtime == manifest, "complete runtime content/modes differ")
            require(kept["/usr/lib/nvbroadcast/runtime-manifest.json"] == manifest_bytes
                    and kept[PREFIX + "share/nvbroadcast-runtime-provenance/pylock.toml"] == lock_bytes
                    and kept[PREFIX + "share/nvbroadcast-runtime-provenance/builders.json"] == git("packaging/native-runtime/builders.json")
                    and kept["/usr/lib/nvbroadcast/validate-install.py"] == git("packaging/native-runtime/validate_install.py"), "installed manifest/lock/validator differs")
            expected_inputs = parse_json(git("packaging/runtime-prototype/inputs.json"))
            expected_inputs["application"] = source
            require(parse_json(kept[PREFIX + "share/nvbroadcast-runtime-provenance/inputs.json"]) == expected_inputs
                    and parse_json(kept[PREFIX + "share/nvbroadcast-runtime-provenance/selection.json"]) ==
                    {"target": "linux-x86_64-cp313-cpu", "lock_sha256": digest(lock_bytes), "source": source}, "installed input selection differs")
            require(parse_json(kept["/usr/lib/nvbroadcast/package.json"]) == {
                "schema_version": 1, "variant": "cpu", "version": version, "source": source, "adapter": adapter,
                "runtime_manifest_sha256": digest(manifest_bytes)}, "installed package identity differs")
            app_actual = {p.split("/site-packages/nvbroadcast/", 1)[1]: item["sha256"]
                          for p, item in actual.items() if "/site-packages/nvbroadcast/" in p and p.endswith(".py")}
            require(app_actual == app_expected, "installed application source differs")
            hook = lambda action: expected_hook(git, action, version, family, helper)
            if family == "rpm":
                require(kept["/usr/bin/nvbroadcast-recover-rpm"] == helper and modes["/usr/bin/nvbroadcast-recover-rpm"] == 0o755, "recovery helper differs")
                expected_scripts = {"PREIN": hook("prepare"), "POSTTRANS": hook("finish"),
                    "PREUN": 'if [ "$1" -eq 0 ]; then\n' + hook("prepare-remove") + '\nfi\n',
                    "POSTUN": 'if [ "$1" -eq 0 ]; then\n' + hook("remove") + '\nfi\n'}
                for tag, expected in expected_scripts.items():
                    actual_script = docker_query(path, family,
                        ["rpm", "-qp", "--qf", "%{" + tag + "}", "/package.rpm"]).decode()
                    # RPM strips only trailing LF from the spec's script body.
                    require(actual_script.rstrip("\n") == expected.rstrip("\n"), f"RPM {tag} differs from rendered source")
                scripts = ("PREIN", "POSTTRANS", "PREUN", "POSTUN")
                absent_scripts = ("POSTIN", "PRETRANS", "PREUNTRANS", "POSTUNTRANS", "VERIFYSCRIPT")
                expected_metadata = {tag + "PROG": "/bin/sh" for tag in scripts}
                expected_metadata.update({tag + "FLAGS": "(none)" for tag in scripts})
                expected_metadata.update({tag + suffix: "(none)" for tag in absent_scripts for suffix in ("", "PROG", "FLAGS")})
                expected_metadata.update({tag + suffix: "(none)" for tag in ("TRIGGER", "FILETRIGGER", "TRANSFILETRIGGER")
                                          for suffix in ("SCRIPTS", "SCRIPTPROG", "SCRIPTFLAGS")})
                query = "".join(tag + "\t%{" + tag + "}\n" for tag in expected_metadata)
                actual_metadata = docker_query(path, family, ["rpm", "-qp", "--qf", query, "/package.rpm"]).decode()
                require(actual_metadata == "".join(tag + "\t" + value + "\n" for tag, value in expected_metadata.items()),
                        "unexpected RPM scriptlet, trigger, interpreter or script flags")
                identity = docker_query(path, family, ["rpm", "-qp", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}", "/package.rpm"]).decode()
                require(identity == f"nvbroadcast\t{version}\tx86_64", "RPM header identity differs")
                requirements = docker_query(path, family, ["rpm", "-qp", "--qf", "[%{REQUIRENAME}\t%{REQUIREFLAGS}\t%{REQUIREVERSION}\n]", "/package.rpm"]).decode()
                pre_requires = {parts[0]: (int(parts[1]), parts[2]) for line in requirements.splitlines()
                                if len(parts := line.split("\t")) == 3 and int(parts[1]) & 512}
                require(pre_requires.get("python3") == (524, "3.11")
                        and pre_requires.get("util-linux") == (512, "")
                        and pre_requires.get("coreutils") == (512, ""), "RPM bootstrap dependencies differ")
            else:
                control_inventory, _, controls, _ = decoded_inventory(path, family, control=True)
                require(set(control_inventory) == {"/control", "/preinst", "/prerm", "/postinst", "/postrm"}
                        and set(controls) == set(control_inventory), "unexpected DEB control member")
                expected_scripts = {"preinst": hook("prepare"),
                    "prerm": '#!/bin/sh\nset -eu\ncase "$1" in remove|deconfigure)\n' + hook("prepare") + '\n;; esac\n',
                    "postinst": '#!/bin/sh\nset -eu\n[ "$1" != configure ] || {\n' + hook("finish") + '\n}\n',
                    "postrm": '#!/bin/sh\nset -eu\ncase "$1" in remove|purge)\n' + hook("remove") + '\n;; esac\n'}
                for tag, expected in expected_scripts.items():
                    require(controls["/" + tag] == expected.encode(), f"DEB {tag} differs from rendered source")
                identity = docker_query(path, family, ["dpkg-deb", "--show", "--showformat=${Package}\t${Version}\t${Architecture}", "/package.deb"]).decode()
                require(identity == f"nvbroadcast\t{version}\tamd64", "DEB control identity differs")
            result["packages"][family] = {"path": str(path), "sha256": package["sha256"], "bytes": package["bytes"],
                "relative_path": path.relative_to(output).as_posix(),
                "payload_entries": len(actual), "payload_regular_bytes": total, "runtime_entries": len(manifest),
                "runtime_manifest_sha256": digest(manifest_bytes), "application_python_files": len(app_actual),
                "scriptlet_source_match": True, "recovery_helper_sha256": digest(helper) if family == "rpm" else None}
            save()
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return result


def main(argv=None):
    result = validate(parse_arguments(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

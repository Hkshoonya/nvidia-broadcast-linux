#!/usr/bin/python3 -IB
"""Inspect/preserve interrupted native RPM fragments using a signed package.

This independent system-Python utility never runs the damaged private runtime,
installs a package, removes a transaction marker, or discards a fragment. The
preserved, authenticated package is replayed separately through RPM.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

PRIMARY = "A691DCA6C031FEFD2C34F5AAFE33D0C20C9451D1"
PUBLIC_SHA256 = "bf9371c4fc83c41f8b1dbc3607c25ffa261b1e5de99fe799cf0a863db16cbd8a"
BASE = "/usr/lib/nvbroadcast"
MARKER = BASE + "/.transaction"
LOCK = "/var/lib/nvbroadcast/runtime.lock"
RECOVERY = "/usr/lib/nvbroadcast-recovery"
PUBLIC_ARMOR = b""  # Filled from the reviewed public key by the native adapter.
FRAGMENT = re.compile(r"^(.*);([0-9a-f]{8})$")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+-[1-9][0-9]*\.(cpu|cuda)")
EXTERNAL = {
    "/usr/bin/nvbroadcast", "/usr/bin/nvbroadcast-vcam", "/usr/bin/nvbroadcast-recover-rpm",
    "/usr/lib/systemd/user/nvbroadcast-vcam.service",
    "/usr/share/applications/com.doczeus.NVBroadcast.desktop",
    "/usr/share/metainfo/com.doczeus.NVBroadcast.metainfo.xml",
    "/usr/share/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg",
    "/usr/share/doc/nvbroadcast/NOTICE", "/usr/share/doc/nvbroadcast/CONTRIBUTORS.md",
}
ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C", "LANG": "C"}
CHUNK = 1024 * 1024
STREAM_TIMEOUT = 600


class RecoveryError(ValueError):
    """Recovery could not establish the required trust or transaction state."""


def tool(arguments: list[str], home: Path) -> str:
    process = subprocess.run(arguments, env=dict(ENV, HOME=str(home)), capture_output=True,
                             timeout=600, check=False)
    if process.returncode:
        raise RecoveryError(f"{arguments[0]} failed (exit {process.returncode}); no package replay performed")
    return process.stdout.decode("utf-8", errors="strict")


def regular(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RecoveryError(f"not a regular file: {path}")
    return os.fdopen(descriptor, "rb")


def digest(path: Path) -> str:
    with regular(path) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(source: Path, destination: Path) -> str:
    result = hashlib.sha256()
    with regular(source) as original, destination.open("xb") as copy:
        for block in iter(lambda: original.read(CHUNK), b""):
            copy.write(block)
            result.update(block)
        copy.flush()
        os.fsync(copy.fileno())
    destination.chmod(0o600)
    return result.hexdigest()


def authenticate(package: Path, public_key: Path | None, work: Path) -> dict:
    """Verify only a fixed-name private snapshot with the pinned public key."""
    work.mkdir(mode=0o700)
    candidate = work / "package.rpm"
    package_hash = snapshot(package, candidate)
    key = work / "public.asc"
    if public_key is None:
        key.write_bytes(PUBLIC_ARMOR)
        key_hash = digest(key)
    else:
        key_hash = snapshot(public_key, key)
    if key_hash != PUBLIC_SHA256:
        raise RecoveryError("public key differs from the pinned project key")
    database = work / "rpmdb"
    database.mkdir(mode=0o700)
    tool(["rpmkeys", "--dbpath", str(database), "--import", str(key)], work)
    verification = tool(["rpmkeys", "--dbpath", str(database), "--checksig", "--verbose", str(candidate)], work)
    # Unsigned RPMs can return success for digests alone; require a signature.
    if not re.search(r"\bSignature\b[^\n]*:\s*OK\s*$", verification, re.MULTILINE | re.IGNORECASE):
        raise RecoveryError("RPM has no verified project OpenPGP signature")
    metadata = tool(["rpm", "-qp", "--qf", "%{NAME}\n%{VERSION}-%{RELEASE}\n%{ARCH}\n", str(candidate)], work).splitlines()
    if len(metadata) != 3 or metadata[0] != "nvbroadcast" or metadata[2] != "x86_64" or not VERSION.fullmatch(metadata[1]):
        raise RecoveryError("expected an nvbroadcast x86_64 CPU/CUDA native-runtime package")
    return {"path": candidate, "sha256": package_hash, "public_key": key,
            "version": metadata[1], "fingerprint": PRIMARY}


def member_path(name: str) -> str:
    if name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or name.startswith("/") or str(path) != name or any(c in name for c in "\n\r\t\x00") or ".." in path.parts:
        raise RecoveryError("unsafe or noncanonical RPM archive path")
    absolute = "/" + name
    if absolute == BASE or absolute.startswith(BASE + "/") or absolute in EXTERNAL:
        return absolute
    # RPM packages include parent directories shared with other system packages.
    if any(PurePosixPath(absolute) in PurePosixPath(p).parents for p in (BASE, *EXTERNAL)):
        return absolute
    raise RecoveryError(f"unexpected path in native RPM: {absolute}")


class Filesystem:
    """A fixed system root in the CLI; an isolated fixture root in unit tests."""
    def __init__(self, root: Path = Path("/"), owner: int = 0):
        self.root, self.owner = root, owner

    def path(self, absolute: str) -> Path:
        if not absolute.startswith("/") or ".." in PurePosixPath(absolute).parts:
            raise RecoveryError("invalid system path")
        return self.root / absolute.lstrip("/")

    def parents(self, path: Path) -> None:
        if not path.is_relative_to(self.root):
            raise RecoveryError("path escapes the system root")
        for parent in (path, *path.parents):
            info = parent.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != self.owner or info.st_gid != self.owner or info.st_mode & 0o022:
                raise RecoveryError(f"unsafe parent directory: {parent}")
            if parent == self.root:
                break

    def info(self, path: Path, *, link: bool = False) -> os.stat_result:
        self.parents(path.parent)
        info = path.lstat()
        if (info.st_uid != self.owner or info.st_gid != self.owner or info.st_nlink != 1
                or (not link and (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022))):
            raise RecoveryError(f"unsafe ownership, type, permissions or hardlinks: {path}")
        return info

    @contextmanager
    def locked(self, rpm_lock: str):
        """Use RPM's fcntl lock and the launcher's flock; never replace inodes."""
        descriptors = []
        try:
            for absolute, native in ((rpm_lock, True), (LOCK, False)):
                path = self.path(absolute)
                before = self.info(path)
                descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
                descriptors.append(descriptor)
                after = os.fstat(descriptor)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise RecoveryError("lock identity changed")
                if native:
                    fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    if after.st_size:
                        raise RecoveryError("runtime lock is not empty")
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def marker(self, version: str) -> os.stat_result:
        path = self.path(MARKER)
        info = self.info(path)
        if info.st_size > 128 or path.read_text() != version + "\n":
            raise RecoveryError("transaction marker does not identify the signed target version")
        if info.st_mtime > time.time() + 1:
            raise RecoveryError("transaction marker is in the future")
        return info


def signature(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def archive_inventory(archive, filesystem: Filesystem, marker: os.stat_result | None) -> tuple[dict, list]:
    """Read members without extracting or executing any signed package content."""
    inventory, fragments = {}, []
    total_size = 0
    siblings = {}
    for item in archive:
        name = member_path(item.name)
        if name in inventory or len(inventory) >= 100000 or item.uid or item.gid:
            raise RecoveryError("duplicate, excessive or non-root RPM members")
        if not (item.isfile() or item.isdir() or item.issym()) or (not item.issym() and item.mode & 0o022):
            raise RecoveryError(f"unsupported RPM member type or permissions: {name}")
        if name not in EXTERNAL and not (name == BASE or name.startswith(BASE + "/")) and not item.isdir():
            raise RecoveryError("RPM parent path must be a directory")
        if item.issym():
            # The authenticated package may link only into its own runtime.
            target = os.path.normpath(os.path.join(str(PurePosixPath(name).parent), item.linkname))
            if not target.startswith(BASE + "/runtime/"):
                raise RecoveryError(f"RPM symlink escapes its runtime: {name}")
        total_size += item.size
        if item.size < 0 or total_size > 20 * 1024**3:
            raise RecoveryError("RPM payload exceeds recovery size bounds")
        entry = {"kind": "file" if item.isfile() else "link" if item.issym() else "directory", "size": item.size}
        inventory[name] = entry
        if marker is None or item.isdir():
            continue
        original = filesystem.path(name)
        if not original.parent.exists():
            continue
        filesystem.parents(original.parent)
        if original.parent not in siblings:
            siblings[original.parent] = [p for p in original.parent.iterdir() if ";" in p.name]
        candidates = [p for p in siblings[original.parent] if p.name.startswith(original.name + ";")]
        if not candidates:
            continue
        reads = []
        try:
            for candidate in candidates:
                absolute = "/" + str(candidate.relative_to(filesystem.root))
                match = FRAGMENT.fullmatch(absolute)
                if not match or match[1] != name or not int(marker.st_mtime) - 300 <= int(match[2], 16) <= int(marker.st_mtime) + 1:
                    raise RecoveryError(f"fragment is outside this transaction: {absolute}")
                info = filesystem.info(candidate, link=item.issym())
                record = {"original": absolute, "payload_path": name, "tid": match[2], "identity": signature(info)}
                if item.issym():
                    if not stat.S_ISLNK(info.st_mode) or os.readlink(candidate) != item.linkname:
                        raise RecoveryError(f"temporary symlink differs from signed payload: {absolute}")
                    record["symlink"] = item.linkname
                else:
                    if info.st_size > item.size:
                        raise RecoveryError(f"fragment exceeds signed payload size: {absolute}")
                    stream = regular(candidate)
                    if signature(os.fstat(stream.fileno())) != signature(info):
                        stream.close()
                        raise RecoveryError("fragment changed while opening")
                    reads.append((stream, hashlib.sha256(), info.st_size, record))
                fragments.append(record)
            if item.isfile():
                content = archive.extractfile(item)
                with content:
                    offset = 0
                    for block in iter(lambda: content.read(CHUNK), b""):
                        for stream, digest_, length, record in reads:
                            chunk = block[:max(0, min(len(block), length - offset))]
                            if stream.read(len(chunk)) != chunk:
                                raise RecoveryError(f"fragment bytes differ from signed payload: {record['original']}")
                            digest_.update(chunk)
                        offset += len(block)
                    if offset != item.size:
                        raise RecoveryError("truncated RPM member")
                for stream, digest_, length, record in reads:
                    if signature(os.fstat(stream.fileno())) != record["identity"]:
                        raise RecoveryError("fragment changed during comparison")
                    record.update(size=length, sha256=digest_.hexdigest())
        finally:
            for stream, *_ in reads:
                stream.close()
    return inventory, fragments


def package_inventory(package: dict, filesystem: Filesystem, marker=None) -> tuple[dict, list]:
    # rpm2archive emits a tar stream and supports individual files over 4 GiB.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["rpm2archive", "--nocompression", str(package["path"])],
                                   env=dict(ENV, HOME=str(package["path"].parent)), stdout=subprocess.PIPE, stderr=errors)
        def stop_producer():
            try:
                process.kill()
            except ProcessLookupError:
                pass
        # Bound blocked reads as well as wait(): a hung decoder cannot hold the
        # package manager and application locks indefinitely.
        timer = threading.Timer(STREAM_TIMEOUT, stop_producer)
        timer.daemon = True
        timer.start()
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                result = archive_inventory(archive, filesystem, marker)
            # Consume tar's trailing padding before waiting for the producer.
            while process.stdout.read(CHUNK):
                pass
            if process.wait(timeout=600):
                raise RecoveryError("cannot decode authenticated RPM payload")
            return result
        finally:
            timer.cancel()
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


def plan(filesystem: Filesystem, target: dict, previous: dict | None = None) -> list[dict]:
    marker = filesystem.marker(target["version"])
    inventory, fragments = package_inventory(target, filesystem, marker)
    baseline = package_inventory(previous, filesystem)[0] if previous else {}
    required = {BASE + "/runtime", BASE + "/runtime-manifest.json", BASE + "/validate-install.py",
                BASE + "/package.json", BASE + "/check-packages", "/usr/bin/nvbroadcast"}
    if not required <= inventory.keys():
        raise RecoveryError("signed RPM lacks the complete native-runtime interface")
    recognized = inventory.keys() | baseline.keys() | {MARKER, BASE + "/.legacy-runtime"}
    temporary = {entry["original"] for entry in fragments}
    # Never descend through symlinks, and inspect every private-prefix path.
    for directory, dirs, files in os.walk(filesystem.path(BASE), followlinks=False):
        filesystem.parents(Path(directory))
        for name in dirs + files:
            path = Path(directory) / name
            absolute = "/" + str(path.relative_to(filesystem.root))
            if absolute not in recognized | temporary:
                raise RecoveryError(f"unrecognized private-prefix path: {absolute}")
            expected = inventory.get(absolute, baseline.get(absolute, {}))
            if expected.get("kind") == "directory" and path.is_symlink():
                raise RecoveryError(f"redirected package directory: {absolute}")
            if path.is_symlink() and name in dirs:
                dirs.remove(name)
    if len({entry["tid"] for entry in fragments}) > 1:
        raise RecoveryError("fragments belong to more than one RPM transaction")
    if signature(filesystem.marker(target["version"])) != signature(marker):
        raise RecoveryError("transaction marker changed during inspection")
    return fragments


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".new")
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def preserve(filesystem: Filesystem, package: dict, entries: list[dict]) -> dict:
    """Journal all intentions before moves; never overwrite or delete fragments."""
    if not entries:
        raise RecoveryError("no interrupted unpack fragments found; no preservation required")
    filesystem.parents(filesystem.path("/usr/lib"))
    parent = filesystem.path(RECOVERY)
    parent.mkdir(mode=0o700, exist_ok=True)
    filesystem.parents(parent)
    for entry in entries:
        source = filesystem.path(entry["original"])
        info = filesystem.info(source, link="symlink" in entry)
        if signature(info) != entry["identity"] or info.st_dev != parent.stat().st_dev:
            raise RecoveryError("fragment changed or recovery crosses filesystem boundaries")
        if "sha256" in entry and digest(source) != entry["sha256"]:
            raise RecoveryError("fragment bytes changed after inspection")
        if "symlink" in entry and os.readlink(source) != entry["symlink"]:
            raise RecoveryError("fragment symlink changed after inspection")
    destination = parent / uuid.uuid4().hex
    destination.mkdir(mode=0o700)
    sync_directory(parent.parent)
    sync_directory(parent)
    candidate = destination / "package.rpm"
    if snapshot(package["path"], candidate) != package["sha256"]:
        raise RecoveryError("authenticated package snapshot changed")
    key = destination / "public.asc"
    if snapshot(package["public_key"], key) != PUBLIC_SHA256:
        raise RecoveryError("authenticated project key snapshot changed")
    report = {"status": "preserving", "package_sha256": package["sha256"], "version": package["version"],
              "fingerprint": PRIMARY, "package": str(candidate), "entries": entries, "preserved": []}
    report_path = destination / "recovery.json"
    write_report(report_path, report)
    for entry in entries:
        source = filesystem.path(entry["original"])
        if signature(filesystem.info(source, link="symlink" in entry)) != entry["identity"]:
            raise RecoveryError(f"fragment changed before preservation; see {report_path}")
        saved = destination / "files" / entry["original"].lstrip("/")
        saved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if "sha256" in entry:
            with regular(source) as stream:
                if signature(os.fstat(stream.fileno())) != entry["identity"]:
                    raise RecoveryError("fragment changed while syncing")
                os.fsync(stream.fileno())
        os.rename(source, saved)
        sync_directory(source.parent)
        for directory in (saved.parent, *saved.parent.parents):
            sync_directory(directory)
            if directory == destination:
                break
        report["preserved"].append(entry["original"])
        write_report(report_path, report)
    report["status"] = "preserved; package replay and validation still required"
    report["key_import_command"] = "sudo rpm --import " + shlex.quote(str(key))
    report["replay_command"] = "sudo rpm --upgrade --replacepkgs --oldpackage " + shlex.quote(str(candidate))
    report["validation_commands"] = [
        "test ! -e /usr/lib/nvbroadcast/.transaction",
        "/usr/lib/nvbroadcast/check-packages",
        "sudo /usr/lib/nvbroadcast/runtime/bin/python -I -B /usr/lib/nvbroadcast/validate-install.py --variant "
        + package["version"].rsplit(".", 1)[1],
    ]
    write_report(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="project-signed target CPU/CUDA RPM")
    parser.add_argument("--previous-package", type=Path, help="signed previous RPM, if an interrupted upgrade left removed paths")
    parser.add_argument("--public-key", type=Path, help="source-tree use: project public key; its exact SHA-256 is pinned")
    parser.add_argument("--preserve", action="store_true", help="preserve matching fragments; default only inspects")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("run through sudo; this utility uses system Python, never the interrupted runtime")
    os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="nvb-rpm-recovery-", dir="/var/tmp") as temporary:
            work = Path(temporary)
            key = args.public_key.absolute() if args.public_key else None
            target = authenticate(args.package.absolute(), key, work / "target")
            previous = authenticate(args.previous_package.absolute(), key, work / "previous") if args.previous_package else None
            rpm_lock = tool(["rpm", "--eval", "%{_rpmlock_path}"], work).strip()
            if not rpm_lock.startswith("/") or "\n" in rpm_lock or "%" in rpm_lock:
                raise RecoveryError("cannot determine RPM transaction lock")
            filesystem = Filesystem()
            with filesystem.locked(rpm_lock):
                entries = plan(filesystem, target, previous)
                report = preserve(filesystem, target, entries) if args.preserve else {
                    "status": "inspected; no installed files changed", "package_sha256": target["sha256"],
                    "version": target["version"], "fingerprint": PRIMARY, "fragments": entries}
            print(json.dumps(report, indent=2))
    except (RecoveryError, OSError, subprocess.SubprocessError, tarfile.TarError, UnicodeError) as error:
        parser.exit(1, f"Recovery refused: {error}\nThe transaction marker remains; do not launch the partial runtime.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


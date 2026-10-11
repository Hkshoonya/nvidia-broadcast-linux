"""Prepare a signed recovery copy before RPM payload staging can hide /usr/bin.

This code is embedded in the authenticated RPM's ordinary %pre/%posttrans
scriptlets. It uses only system Python and never imports the private runtime.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile


def trusted_directory(path: Path, root: Path, owner: int) -> None:
    if not path.is_relative_to(root):
        raise ValueError("bootstrap path escapes the system root")
    for directory in (path, *path.parents):
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != owner or info.st_gid != owner
                or info.st_mode & 0o022):
            raise ValueError(f"unsafe recovery bootstrap directory: {directory}")
        if directory == root:
            break


def sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_file(path: Path, expected: bytes, owner: int, mode: int, *, durable: bool = False) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or (info.st_uid, info.st_gid, info.st_nlink) != (owner, owner, 1)
                or stat.S_IMODE(info.st_mode) != mode or info.st_size != len(expected)
                or stream.read() != expected):
            raise ValueError(f"unsafe or changed recovery bootstrap file: {path}")
        if durable:
            os.fsync(stream.fileno())


def bootstrap(action: str, payload: bytes, *, root: Path = Path("/"), owner: int = 0) -> Path:
    if sys.version_info < (3, 11):
        raise ValueError("RPM recovery requires system Python 3.11 or newer")
    if action not in {"prepare", "finish"} or not payload or len(payload) > 128 * 1024:
        raise ValueError("invalid recovery bootstrap operation or payload")
    state = root / "var/lib/nvbroadcast"
    directory = state / "recovery-tools"
    trusted_directory(state, root, owner)
    if action == "prepare":
        directory.mkdir(mode=0o700, exist_ok=True)
    elif not directory.exists() and not directory.is_symlink():
        return directory
    trusted_directory(directory, root, owner)
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise ValueError("recovery bootstrap directory must have mode 0700")
    target = directory / ("recover-" + hashlib.sha256(payload).hexdigest() + ".py")
    if action == "finish":
        installed = root / "usr/bin/nvbroadcast-recover-rpm"
        trusted_directory(installed.parent, root, owner)
        verify_file(installed, payload, owner, 0o755)
        if target.exists() or target.is_symlink():
            verify_file(target, payload, owner, 0o600)
            target.unlink()
            sync(directory)
        # Preserve unfamiliar state or copies from earlier failed transactions.
        # Only remove this exact authenticated copy after complete validation.
        if not any(directory.iterdir()):
            directory.rmdir()
            sync(state)
        return target
    if target.exists() or target.is_symlink():
        verify_file(target, payload, owner, 0o600, durable=True)
        sync(directory)
        sync(state)
        return target
    descriptor, temporary = tempfile.mkstemp(prefix=".prepare-", dir=directory)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        trusted_directory(directory, root, owner)
        # RPM holds its transaction lock and prepare holds Broadcast's lock.
        # A root-only directory prevents another user creating/replacing target.
        if target.exists() or target.is_symlink():
            verify_file(target, payload, owner, 0o600)
        else:
            os.rename(temporary_path, target)
        sync(directory)
        sync(state)
        verify_file(target, payload, owner, 0o600)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


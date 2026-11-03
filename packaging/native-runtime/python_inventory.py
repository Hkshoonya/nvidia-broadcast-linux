"""Bound a METADATA-only projection for optional native Python inventory evidence.

The native validator must first authenticate the archive and compare its complete
payload with the source-bound runtime manifest. This helper does not authenticate
those inputs, extract an archive, or execute target code. Final SPDX generation
and archive-bound evidence are separate from package construction and signing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib


PREFIX = "/usr/lib/nvbroadcast/runtime/"
MAX_METADATA_BYTES = 4 * 1024**2
MAX_TOTAL_METADATA_BYTES = 64 * 1024**2
MAX_DISTRIBUTIONS = 1000
PACKAGE_ROOT = re.compile(
    r"^((?:usr/local/|usr/)?lib/python[^/]*/(?:site|dist)-packages)(?:/([^/]+)(?:/(.*))?)?$"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def layout(path):
    require(isinstance(path, str) and path.startswith("/") and "\\" not in path and
            all(part not in ("", ".", "..") for part in path[1:].split("/")) and
            not any(ord(character) < 32 for character in path), "noncanonical payload path")
    if not path.startswith(PREFIX):
        return None
    return PACKAGE_ROOT.fullmatch(path.removeprefix(PREFIX))


class MetadataCollector:
    """Reserve bounds before a decoder reads bytes; verify all rows afterward."""

    def __init__(self):
        self.roots = set()
        self.distributions = set()
        self.sizes = {}
        self.data = {}
        self.total_bytes = 0

    def select(self, path, kind, size=0):
        match = layout(path)
        if match is None:
            return False
        root, child, remainder = match.groups()
        self.roots.add(root)
        if child is None:
            require(kind == "directory", "redirected or invalid Python package root")
            return False
        lower = child.lower()
        require(not lower.endswith((".egg-info", ".egg")), "unsupported egg metadata")
        if not lower.endswith(".dist-info"):
            return False
        distribution = root + "/" + child
        self.distributions.add(distribution)
        require(len(self.distributions) <= MAX_DISTRIBUTIONS, "too many Python distributions")
        if remainder is None:
            require(kind == "directory", "redirected or invalid distribution directory")
            return False
        if remainder != "METADATA":
            return False
        require(kind == "file", "METADATA must be a regular file")
        require(type(size) is int and 0 < size <= MAX_METADATA_BYTES, "METADATA size exceeds bound")
        require(path not in self.sizes, "duplicate METADATA input")
        require(self.total_bytes + size <= MAX_TOTAL_METADATA_BYTES, "aggregate METADATA exceeds bound")
        self.sizes[path] = size
        self.total_bytes += size
        return True

    def accept(self, path, data):
        require(path in self.sizes and path not in self.data, "unreserved or duplicate METADATA bytes")
        require(isinstance(data, bytes) and len(data) == self.sizes[path], "METADATA size changed during decode")
        self.data[path] = data

    def verified_records(self, inventory, runtime_manifest):
        """Require an exact projection, including nonredirected source parents."""
        expected = {PREFIX + directory + "/METADATA" for directory in self.distributions}
        require(expected and expected == set(self.sizes) == set(self.data),
                "missing or extra distribution METADATA")
        # Re-enumerate the complete decoded inventory so an omitted select()
        # call cannot silently discard another metadata directory or egg owner.
        observed = set()
        for path, record in inventory.items():
            match = layout(path)
            if match is None:
                continue
            root, child, remainder = match.groups()
            if child is None:
                require(record.get("directory") is True, "redirected Python package root")
                continue
            require(not child.lower().endswith((".egg-info", ".egg")), "unsupported egg metadata")
            if child.lower().endswith(".dist-info"):
                observed.add(PREFIX + root + "/" + child + "/METADATA")
        require(observed == expected, "projection differs from complete metadata inventory")
        declared_metadata = set()
        for relative in runtime_manifest:
            match = layout(PREFIX + relative)
            if match is None:
                continue
            root, child, _remainder = match.groups()
            if child is not None:
                require(not child.lower().endswith((".egg-info", ".egg")), "unsupported egg metadata")
                if child.lower().endswith(".dist-info"):
                    declared_metadata.add(PREFIX + root + "/" + child + "/METADATA")
        require(declared_metadata == expected, "projection differs from complete runtime manifest metadata")
        records = {}
        for path in sorted(expected):
            relative = path.removeprefix(PREFIX)
            for parent in PurePosixPath(path).parents:
                name = str(parent)
                if name != PREFIX.rstrip("/") and not name.startswith(PREFIX):
                    continue
                require(inventory.get(name, {}).get("directory") is True,
                        "METADATA parent is missing or redirected")
            actual, declared = inventory.get(path, {}), runtime_manifest.get(relative, {})
            value = self.data[path]
            digest = hashlib.sha256(value).hexdigest()
            require(actual.get("sha256") == declared.get("sha256") == digest and
                    actual.get("size") == declared.get("size") == len(value) and
                    "symlink" not in actual and "symlink" not in declared,
                    "METADATA differs from authenticated runtime manifest")
            records[relative] = {"bytes": len(value), "sha256": digest}
        return records


def write_projection(directory, collector, inventory, runtime_manifest, *, lock_bytes,
                     source_commit, source_date_epoch, archive_identity, runtime_manifest_sha256):
    """Write fresh regular metadata files only after complete-payload verification.

    Archive/source authentication belongs to validate_candidate. The returned
    receipt must be hash-bound by that validator before transfer to retention.
    No caller-provided archive path is opened or rewritten here.
    """
    records = collector.verified_records(inventory, runtime_manifest)
    require(isinstance(source_commit, str) and re.fullmatch(r"[a-f0-9]{40}", source_commit), "invalid source context")
    require(type(source_date_epoch) is int and source_date_epoch >= 0, "invalid source epoch")
    require(isinstance(runtime_manifest_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", runtime_manifest_sha256),
            "invalid runtime manifest identity")
    require(set(archive_identity) == {"name", "bytes", "sha256"} and
            isinstance(archive_identity["name"], str) and
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.deb", archive_identity["name"]) and
            type(archive_identity["bytes"]) is int and archive_identity["bytes"] > 0 and
            isinstance(archive_identity["sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", archive_identity["sha256"]),
            "invalid validated DEB identity")
    require(isinstance(lock_bytes, bytes) and 0 < len(lock_bytes) <= MAX_METADATA_BYTES, "invalid assembled lock size")
    lock = tomllib.loads(lock_bytes.decode())
    require(lock.get("lock-version") == "1.0" and isinstance(lock.get("packages"), list) and lock["packages"],
            "expected assembled PEP751 lock")
    directory = Path(directory).absolute()
    require(not directory.exists() and not directory.is_symlink() and directory.resolve() == directory,
            "projection requires a fresh nonredirected output directory")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    files = {}
    for relative, record in records.items():
        target = directory / "runtime" / relative
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(collector.data[PREFIX + relative])
        files["runtime/" + relative] = record
    (directory / "context").mkdir(mode=0o700)
    with (directory / "context/assembled-pylock.toml").open("xb") as stream:
        stream.write(lock_bytes)
    files["context/assembled-pylock.toml"] = {"bytes": len(lock_bytes), "sha256": hashlib.sha256(lock_bytes).hexdigest()}
    receipt = {
        "schema_version": 1, "source_commit": source_commit, "source_date_epoch": source_date_epoch,
        "architecture": "amd64", "archive": dict(archive_identity),
        "runtime_manifest_sha256": runtime_manifest_sha256, "files": files,
        "metadata_count": len(records), "metadata_bytes": collector.total_bytes,
        "scope": ("Partial app-owned Python METADATA projection only; archive/source authentication "
                  "must be supplied by native validation. No target-code execution, complete native/license "
                  "inventory or independent release qualification."),
    }
    with (directory / "projection.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt

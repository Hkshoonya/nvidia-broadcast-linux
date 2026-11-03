#!/usr/bin/env python3
"""Inventory app-owned installed Python metadata as an opt-in SPDX 2.3 document.

This reads an already extracted runtime without importing its packages. Hashes
identify the inputs; they do not authenticate the publisher, extraction, source
commit, or supplied lock context. Native libraries and model weights are outside
this package-level inventory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import stat
import tomllib
from urllib.parse import quote

from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version

from nvbroadcast.runtime.artifact import (
    ARCHITECTURES,
    ArtifactEnvironment,
    discover_package_roots,
)


MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_TOTAL_METADATA_BYTES = 64 * 1024 * 1024
MAX_DISTRIBUTIONS = 1000
SCOPE = (
    "Partial package-level inventory of installed app-owned Python distribution "
    "METADATA only. Package files are not exhaustively analyzed. Embedded native "
    "libraries/codecs, the CPython interpreter and its non-distribution components, OS/content runtimes, "
    "drivers, model weights and complete license/dependency analysis are excluded. "
    "Source commit and supplied locks are caller-provided context; their use by "
    "the build and the extracted runtime's relationship to the archive are not "
    "authenticated or verified by this tool. No security or release qualification "
    "is implied."
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular_file(path: Path, *, root: Path | None = None, bounded=False) -> bytes:
    """Read only a regular, nonredirected input, with a bound for metadata."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular input file: {path.name}")
    if root is not None and not path.resolve().is_relative_to(root):
        raise ValueError("Python metadata resolves outside the extracted runtime")
    if bounded and info.st_size > MAX_METADATA_BYTES:
        raise ValueError(f"Oversized metadata input: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(MAX_METADATA_BYTES + 1) if bounded else stream.read()
    if bounded and len(data) > MAX_METADATA_BYTES:
        raise ValueError(f"Oversized metadata input: {path.name}")
    return data


def installed_metadata(root: Path, architecture: str) -> list[dict]:
    """Cross-check metadata paths against the shared artifact inventory."""
    package_roots = discover_package_roots(root)
    if not package_roots or any(not path.is_relative_to(root) for path in package_roots):
        raise ValueError("Expected contained app-owned Python package roots")
    records, total_bytes = {}, 0
    for package_root in package_roots:
        directories = []
        for entry in package_root.iterdir():
            if entry.name.lower().endswith((".egg-info", ".egg")):
                # importlib.metadata can read these before a later inventory
                # mismatch. Reject them before shared discovery touches them.
                raise ValueError("Unsupported egg metadata in the extracted runtime")
            if entry.name.lower().endswith(".dist-info"):
                directories.append(entry)
                if len(records) + len(directories) > MAX_DISTRIBUTIONS:
                    raise ValueError("Python distribution inventory exceeds its bound")
        for directory in sorted(directories):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Redirected or invalid distribution metadata directory")
            path = directory / "METADATA"
            data = regular_file(path, root=root, bounded=True)
            total_bytes += len(data)
            if total_bytes > MAX_TOTAL_METADATA_BYTES:
                raise ValueError("Aggregate Python metadata exceeds its bound")
            metadata = BytesParser().parsebytes(data)
            if len(metadata.get_all("Name", [])) != 1 or len(metadata.get_all("Version", [])) != 1:
                raise ValueError("Distribution metadata needs one Name and Version")
            name, version = metadata["Name"], metadata["Version"]
            if not isinstance(name, str):
                raise ValueError("Invalid installed distribution name")
            try:
                name = canonicalize_name(name, validate=True)
            except InvalidName as error:
                raise ValueError("Invalid installed distribution name") from error
            if not isinstance(version, str) or not 0 < len(version) <= 100:
                raise ValueError("Invalid installed distribution version")
            try:
                Version(version)
            except InvalidVersion as error:
                raise ValueError("Invalid installed distribution version") from error
            if name in records:
                raise ValueError(f"Duplicate installed distribution: {name}")
            records[name] = {
                "name": name, "version": version,
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "sha1": hashlib.sha1(data, usedforsecurity=False).hexdigest(),
            }
            if len(records) > MAX_DISTRIBUTIONS:
                raise ValueError("Python distribution inventory exceeds its bound")
    # Path-only metadata discovery does not add the artifact to sys.path or run
    # its .pth files, packages, entry points or target interpreter.
    environment = ArtifactEnvironment.inspect(root, architecture)
    expected = {name: (item["version"],) for name, item in records.items()}
    if not records or environment.installed != expected or len(environment.distributions) != len(records):
        raise ValueError("Artifact inventory differs from contained wheel metadata")
    return [records[name] for name in sorted(records)]


def lock_context(paths: tuple[Path, ...]) -> list[dict]:
    if len(paths) > 16:
        raise ValueError("Lock context exceeds its input bound")
    records = []
    names = set()
    for path in paths:
        if path.name in names:
            raise ValueError("Lock context labels must be unique")
        names.add(path.name)
        data = regular_file(path, bounded=True)
        lock = tomllib.loads(data.decode("utf-8"))
        if lock.get("lock-version") != "1.0" or not isinstance(lock.get("packages"), list) or not lock["packages"]:
            raise ValueError("Expected a nonempty PEP 751 version 1.0 lock")
        records.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest()})
    return sorted(records, key=lambda item: item["name"])


def build_document(
    runtime_root: Path,
    architecture: str,
    artifact: Path,
    source_commit: str,
    source_date_epoch: int,
    locks: tuple[Path, ...] = (),
) -> dict:
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise ValueError("Expected an extracted runtime directory, not a symlink")
    root = runtime_root.resolve(strict=True)
    if architecture not in ARCHITECTURES:
        raise ValueError("Unsupported artifact architecture")
    if not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", source_commit):
        raise ValueError("Source context requires a full lowercase Git commit ID")
    if type(source_date_epoch) is not int or source_date_epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be a nonnegative integer")
    created = datetime.fromtimestamp(source_date_epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact_info = artifact.lstat()
    if not stat.S_ISREG(artifact_info.st_mode) or artifact_info.st_size == 0:
        raise ValueError("Expected a nonempty regular artifact archive, not a symlink")
    artifact_sha256 = sha256(artifact)
    installed = installed_metadata(root, architecture)
    context = {
        "source_commit": source_commit,
        "architecture": ARCHITECTURES[architecture],
        "locks": lock_context(locks),
        "authentication": "caller-provided context; not verified",
    }
    artifact_id = "SPDXRef-Artifact-" + artifact_sha256
    packages = [{
        "SPDXID": artifact_id, "name": artifact.name,
        "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
        "checksums": [{"algorithm": "SHA256", "checksumValue": artifact_sha256}],
        "licenseConcluded": "NOASSERTION", "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "sourceInfo": "Caller-provided source context: " + json.dumps(context, sort_keys=True),
        "comment": SCOPE,
    }]
    files, relationships = [], [{
        "spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES",
        "relatedSpdxElement": artifact_id,
    }]
    for item in installed:
        identity = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        package_id, file_id = "SPDXRef-Python-" + identity, "SPDXRef-Metadata-" + identity
        packages.append({
            "SPDXID": package_id, "name": item["name"], "versionInfo": item["version"],
            "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION", "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                              "referenceLocator": f"pkg:pypi/{item['name']}@{quote(item['version'], safe='')}"}],
            "comment": "Identity from installed METADATA; archive origin, dependency closure and licenses are not inferred.",
        })
        files.append({
            "SPDXID": file_id, "fileName": "./" + item["path"], "fileTypes": ["TEXT"],
            "checksums": [{"algorithm": "SHA1", "checksumValue": item["sha1"]},
                          {"algorithm": "SHA256", "checksumValue": item["sha256"]}],
            "licenseConcluded": "NOASSERTION", "licenseInfoInFiles": ["NOASSERTION"],
            "copyrightText": "NOASSERTION",
        })
        relationships.extend([
            {"spdxElementId": artifact_id, "relationshipType": "CONTAINS", "relatedSpdxElement": package_id,
             "comment": "Observed in the supplied extracted root; archive/extraction binding is caller-provided."},
            {"spdxElementId": file_id, "relationshipType": "METAFILE_OF", "relatedSpdxElement": package_id},
        ])
    document = {
        "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
        "name": "NVBroadcast app-owned Python metadata inventory: " + artifact.name,
        "creationInfo": {"creators": ["Tool: NVBroadcast-Python-Supplier-Inventory-1"], "created": created},
        "comment": SCOPE, "packages": packages, "files": files, "relationships": relationships,
    }
    # Include every emitted field in the identity, while avoiding a self-hash.
    identity = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    document["documentNamespace"] = "https://github.com/Hkshoonya/nvidia-broadcast-linux/sbom/python-inventory/" + identity
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True, help="already extracted app-owned runtime, never executed")
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--artifact", type=Path, required=True, help="regular archive file to hash; no extraction or authentication")
    parser.add_argument("--source-commit", required=True, help="caller-provided source commit, not verified provenance")
    parser.add_argument("--source-date-epoch", type=int, required=True, help="explicit reproducible metadata timestamp")
    parser.add_argument("--lock", type=Path, action="append", default=[], help="optional PEP 751 source context to hash, not closure verification")
    parser.add_argument("--output", type=Path, required=True, help="new SPDX JSON file, outside the extracted runtime")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.runtime_root.resolve()):
        parser.error("output must be outside the inspected runtime")
    try:
        document = build_document(args.runtime_root, args.architecture, args.artifact, args.source_commit,
                                  args.source_date_epoch, tuple(args.lock))
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(document, sort_keys=True, indent=2) + "\n")
    except (OSError, ValueError, OverflowError) as error:
        parser.exit(1, f"Python inventory failed: {error}\n")
    print(f"Wrote partial Python inventory: {len(document['packages']) - 1} distributions; no release qualification")


if __name__ == "__main__":
    main()

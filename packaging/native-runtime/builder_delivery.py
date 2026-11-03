#!/usr/bin/env python3
"""Verify exact historical OCI builder snapshots without claiming a source rebuild."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
import uuid


HERE = Path(__file__).resolve().parent
REPOSITORY = "Hkshoonya/nvidia-broadcast-linux"
SIGNER = REPOSITORY + "/.github/workflows/native-builders-promote.yml"
PREDICATE = "https://github.com/" + REPOSITORY + "/attestations/native-builder-import/v1"
INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
LEAVES = {CONFIG, "application/vnd.oci.empty.v1+json", "application/vnd.in-toto+json",
          "application/vnd.oci.image.layer.v1.tar", "application/vnd.oci.image.layer.v1.tar+gzip"}
DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
BLOB = re.compile(r"blobs/sha256/[a-f0-9]{64}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=unique_object)


def regular_stream(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        stream.close()
        raise ValueError("expected a regular input file")
    return stream


def hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


@contextmanager
def private_snapshot(archive, expected_bytes):
    """Use the same private bytes for hashing, attestation verification and loading."""
    require(type(expected_bytes) is int and 0 < expected_bytes <= 16 * 1024**3, "invalid archive size pin")
    with tempfile.TemporaryDirectory(prefix="nvbroadcast-builder-delivery-") as temporary:
        snapshot = Path(temporary) / "builders.tar"
        with regular_stream(archive) as source, snapshot.open("xb") as target:
            os.fchmod(target.fileno(), 0o600)
            remaining = expected_bytes
            while remaining:
                block = source.read(min(remaining, 1024 * 1024))
                require(block, "archive is truncated")
                target.write(block)
                remaining -= len(block)
            require(not source.read(1), "archive exceeds its size pin")
        yield snapshot


def inspect_archive(archive: Path, pins: dict, expected_hash: str, expected_bytes: int) -> dict:
    """Check every blob and both OCI/Docker views; never extract or execute bytes."""
    require(re.fullmatch(r"[a-f0-9]{64}", expected_hash), "invalid archive digest pin")
    require(type(expected_bytes) is int and 0 < expected_bytes <= 16 * 1024**3, "invalid archive size pin")
    require(set(pins) == {"bindings", "deb", "rpm"}, "expected all three builder roles")
    roots = {entry["image"] for entry in pins.values()}
    require(len(roots) == 3 and all(DIGEST.fullmatch(root) for root in roots), "invalid builder image pins")
    with regular_stream(archive) as stream:
        require(os.fstat(stream.fileno()).st_size == expected_bytes, "archive size differs")
        require(hash_stream(stream) == expected_hash, "archive hash differs")
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r:") as saved:
            members = {}
            for member in saved:
                require(len(members) < 4096, "archive has too many members")
                require(member.name not in members, "duplicate archive member")
                if member.isdir():
                    require(member.name in {"blobs", "blobs/sha256"}, "unexpected archive directory")
                else:
                    require(member.isfile(), "archive links and special files are forbidden")
                    require(member.name in {"index.json", "oci-layout", "manifest.json"}
                            or BLOB.fullmatch(member.name), "unexpected archive path")
                members[member.name] = member
            blobs = {name for name, member in members.items() if BLOB.fullmatch(name) and member.isfile()}
            for name in blobs:
                with saved.extractfile(members[name]) as blob:
                    require(hash_stream(blob) == name.rsplit("/", 1)[1], "OCI blob hash differs")

            def document(name):
                require(name in members and members[name].isfile(), "missing OCI document")
                require(members[name].size <= 4 * 1024**2, "oversized OCI metadata")
                return parse_json(saved.extractfile(members[name]).read())

            require(document("oci-layout") == {"imageLayoutVersion": "1.0.0"}, "unsupported OCI layout")
            top = document("index.json")
            require(isinstance(top, dict) and top.get("schemaVersion") == 2 and top.get("mediaType") == INDEX,
                    "invalid top-level OCI index")
            descriptors = top.get("manifests", [])
            require(isinstance(descriptors, list) and len(descriptors) == 3
                    and all(isinstance(item, dict) for item in descriptors)
                    and {item.get("digest") for item in descriptors} == roots,
                    "archive roots differ from reviewed builder pins")
            visited = set()
            runtime_pairs = []

            def visit(descriptor, depth=0):
                require(depth <= 5, "OCI graph exceeds depth bound")
                require(isinstance(descriptor, dict) and DIGEST.fullmatch(descriptor.get("digest", "")),
                        "invalid OCI descriptor digest")
                require("urls" not in descriptor, "external OCI content is forbidden")
                name = "blobs/sha256/" + descriptor["digest"].split(":", 1)[1]
                require(name in blobs, "OCI descriptor content is missing")
                require(type(descriptor.get("size")) is int and descriptor["size"] == members[name].size,
                        "OCI descriptor size differs")
                media = descriptor.get("mediaType")
                require(media in LEAVES | {INDEX, MANIFEST}, "unsupported OCI media type")
                if "data" in descriptor:
                    # BuildKit's attestation manifests embed the two-byte empty
                    # config as well as shipping its blob. Accept only that
                    # exact representation, never a second payload source.
                    require(media == "application/vnd.oci.empty.v1+json" and descriptor["size"] == 2
                            and descriptor["data"] == "e30="
                            and descriptor["digest"] == "sha256:" + hashlib.sha256(b"{}").hexdigest(),
                            "unexpected inline OCI content")
                visited.add(name)
                if media == INDEX:
                    value = document(name)
                    require(isinstance(value, dict) and value.get("schemaVersion") == 2 and value.get("mediaType") == INDEX,
                            "OCI index descriptor/content mismatch")
                    children = value.get("manifests", [])
                    require(isinstance(children, list) and 0 < len(children) <= 16
                            and all(isinstance(child, dict) for child in children), "invalid OCI child set")
                    active = [child for child in children if child.get("platform") == {"architecture": "amd64", "os": "linux"}]
                    require(len(active) == 1, "expected exactly one Linux amd64 image in each builder index")
                    for child in children:
                        visit(child, depth + 1)
                elif media == MANIFEST:
                    value = document(name)
                    require(isinstance(value, dict) and value.get("schemaVersion") == 2 and value.get("mediaType") == MANIFEST,
                            "OCI manifest descriptor/content mismatch")
                    config = value.get("config", {})
                    layers = value.get("layers", [])
                    require(isinstance(layers, list) and len(layers) <= 256, "invalid OCI layer set")
                    visit(config, depth + 1)
                    for layer in layers:
                        visit(layer, depth + 1)
                    if config.get("mediaType") == CONFIG:
                        config_path = "blobs/sha256/" + config["digest"].split(":", 1)[1]
                        configuration = document(config_path)
                        require(isinstance(configuration, dict) and configuration.get("os") == "linux"
                                and configuration.get("architecture") == "amd64",
                                "builder config platform differs")
                        runtime_pairs.append((config_path, tuple("blobs/sha256/" + layer["digest"].split(":", 1)[1]
                                                                for layer in layers)))

            for descriptor in descriptors:
                require(descriptor.get("mediaType") == INDEX, "builder pin must retain its OCI index")
                require("org.opencontainers.image.ref.name" not in descriptor.get("annotations", {}),
                        "unexpected mutable tag in archive")
                visit(descriptor)
            require(visited == blobs, "archive contains unreferenced OCI blobs")
            legacy = document("manifest.json")
            require(isinstance(legacy, list) and len(legacy) == 3, "Docker compatibility manifest differs")
            legacy_pairs = []
            for item in legacy:
                require(isinstance(item, dict) and set(item) == {"Config", "RepoTags", "Layers"}
                        and item["RepoTags"] in (None, []) and isinstance(item["Layers"], list)
                        and isinstance(item["Config"], str) and all(isinstance(layer, str) for layer in item["Layers"]),
                        "Docker compatibility manifest may not introduce tags or extra sources")
                legacy_pairs.append((item["Config"], tuple(item["Layers"])))
            require(len(runtime_pairs) == 3 and sorted(runtime_pairs) == sorted(legacy_pairs),
                    "Docker and OCI payload mappings differ")
    return {"status": "integrity_verified_not_authenticated", "archive_sha256": expected_hash,
            "archive_bytes": expected_bytes, "blob_count": len(blobs),
            "builder_images": {role: entry["image"] for role, entry in pins.items()},
            "origin": "maintainer-qualified-historical-snapshot",
            "original_build_provenance_authenticated": False}


def qualify_local_images(pins):
    """Read-only offline checks; do not rebuild or accept equivalent image IDs."""
    driver = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .DriverStatus}}"], text=True, timeout=30))
    require(["driver-type", "io.containerd.snapshotter.v1"] in driver,
            "containerd image store is required to preserve approved OCI index IDs")
    result = {}
    for role, entry in pins.items():
        image = entry["image"]
        inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True, timeout=30))
        require(len(inspected) == 1 and inspected[0]["Id"] == image
                and inspected[0]["Os"] == "linux" and inspected[0]["Architecture"] == "amd64",
                "loaded builder image identity/platform differs")
        query = (["rpm", "-qa", "--qf", "%{NAME} %{VERSION}-%{RELEASE}.%{ARCH}\n"] if role == "rpm"
                 else ["dpkg-query", "-W", "-f=${Package} ${Version} ${Architecture}\n"])
        container = "nvbroadcast-builder-inventory-" + uuid.uuid4().hex
        try:
            output = subprocess.check_output(["docker", "run", "--rm", "--name", container,
                "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--entrypoint", query[0], image, *query[1:]], text=True, timeout=120)
        finally:
            # A timed-out Docker client does not stop its daemon-side container.
            # Remove only this uniquely named inventory query, then prove absence.
            subprocess.run(["docker", "rm", "--force", container], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=30)
            remaining = subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=^/" + container + "$"],
                                                text=True, timeout=30)
            require(not remaining.strip(), "inventory container cleanup could not be verified")
        require(sorted(output.splitlines()) == sorted(entry["packages"]), "native builder inventory differs")
        result[role] = {"image": image, "package_count": len(entry["packages"])}
    return result


def verify_attestation(archive, lock, *, gh="gh"):
    policy = lock.get("attestation")
    require(lock.get("status") == "promoted" and isinstance(policy, dict), "builder delivery has not been authenticated/promoted")
    source = policy.get("source_commit", "")
    require(re.fullmatch(r"[a-f0-9]{40}", source), "missing reviewed promotion source")
    require(policy.get("workflow") == SIGNER and policy.get("predicate_type") == PREDICATE,
            "unreviewed attestation authority")
    subprocess.run([gh, "attestation", "verify", str(archive), "--hostname", "github.com",
        "--repo", REPOSITORY, "--signer-workflow", SIGNER, "--signer-digest", source,
        "--source-ref", "refs/heads/main", "--source-digest", source,
        "--predicate-type", PREDICATE, "--deny-self-hosted-runners"], check=True, timeout=300)


def fetch_candidate(destination, lock, asset_id, *, gh="gh"):
    """Fetch one reviewed infrastructure asset; its hash alone is not provenance."""
    require(type(asset_id) is int and asset_id > 0, "invalid candidate asset ID")
    source = lock["source"]
    require(type(source.get("asset_id")) is int and asset_id == source["asset_id"], "candidate asset ID is not reviewed/pinned")
    require(source["repository"] == REPOSITORY and source["tag"] == "infra-native-builders-20261011"
            and source["asset"] == "nvbroadcast-native-builders-linux-amd64-20261011.oci.tar",
            "unexpected candidate source")
    command = [gh, "api", "--hostname", "github.com"]
    release = parse_json(subprocess.check_output(command + [f"repos/{REPOSITORY}/releases/tags/{source['tag']}"], timeout=60))
    require(release.get("tag_name") == source["tag"] and release.get("prerelease") is True
            and type(release.get("draft")) is bool, "expected a separate infrastructure prerelease")
    assets = [asset for asset in release["assets"] if asset.get("name") == source["asset"]]
    require(len(assets) == 1 and assets[0].get("id") == asset_id and assets[0].get("state") == "uploaded"
            and assets[0].get("size") == lock["archive"]["bytes"]
            and assets[0].get("digest") == "sha256:" + lock["archive"]["sha256"],
            "candidate asset metadata differs from the reviewed lock")
    with destination.open("xb") as output:
        subprocess.run(command + ["-H", "Accept: application/octet-stream", f"repos/{REPOSITORY}/releases/assets/{asset_id}"],
                       stdout=output, check=True, timeout=600)


def process_archive(operation, archive, lock, pins, *, load=False, gh="gh"):
    require(not load or operation in {"qualify", "verify"}, "inspection never loads images")
    require(operation in {"inspect", "qualify", "verify"}, "unknown verification operation")
    metadata = lock["archive"]
    with private_snapshot(archive, metadata["bytes"]) as snapshot:
        report = inspect_archive(snapshot, pins, metadata["sha256"], metadata["bytes"])
        if operation == "verify":
            verify_attestation(snapshot, lock, gh=gh)
            report["status"] = "authenticated_import_delivery"
        if load:
            subprocess.run(["docker", "image", "load", "--input", str(snapshot)], check=True, timeout=600)
        if operation != "inspect":
            report["installed_inventory"] = qualify_local_images(pins)
        return report


def import_predicate(lock, pins, report, run_id, source_commit, *, gh="gh"):
    """Bind successful reviewed-main qualification; do not execute image contents."""
    require(type(run_id) is int and run_id > 0, "invalid qualification run ID")
    require(isinstance(source_commit, str) and re.fullmatch(r"[a-f0-9]{40}", source_commit),
            "invalid promotion source commit")
    require(lock.get("status") == "candidate" and lock.get("attestation") is None,
            "promotion must start from the reviewed candidate lock")
    command = [gh, "api", "--hostname", "github.com"]
    run = parse_json(subprocess.check_output(command + [f"repos/{REPOSITORY}/actions/runs/{run_id}"], timeout=60))
    require(run.get("id") == run_id and run.get("status") == "completed" and run.get("conclusion") == "success"
            and run.get("event") == "workflow_dispatch" and run.get("head_branch") == "main"
            and run.get("head_sha") == source_commit and run.get("repository", {}).get("full_name") == REPOSITORY
            and run.get("head_repository", {}).get("full_name") == REPOSITORY
            and run.get("path") == ".github/workflows/native-builders-qualify.yml",
            "qualification is not a successful approved-main workflow at this source commit")
    attempt = run.get("run_attempt")
    require(type(attempt) is int and attempt > 0, "invalid qualification run attempt")
    artifacts = parse_json(subprocess.check_output(command + [f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100"], timeout=60))
    require(artifacts.get("total_count", 101) <= 100, "qualification artifact list exceeds bound")
    name = f"historical-native-builder-qualification-{run_id}-{attempt}"
    selected = [item for item in artifacts["artifacts"] if item.get("name") == name]
    require(len(selected) == 1, "missing or ambiguous qualification artifact")
    artifact = selected[0]
    artifact_id, size, digest = artifact.get("id"), artifact.get("size_in_bytes"), artifact.get("digest", "")
    require(type(artifact_id) is int and artifact_id > 0 and type(size) is int and 0 < size <= 1024**2
            and DIGEST.fullmatch(digest) and artifact.get("expired") is False
            and artifact.get("workflow_run", {}).get("id") == run_id
            and artifact.get("workflow_run", {}).get("head_sha") == source_commit,
            "qualification artifact identity, size or digest is invalid")
    with tempfile.TemporaryDirectory(prefix="nvbroadcast-builder-evidence-") as temporary:
        archive = Path(temporary) / "evidence.zip"
        with archive.open("xb") as output:
            subprocess.run(command + [f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"],
                           stdout=output, check=True, timeout=120)
        with regular_stream(archive) as stream:
            require(os.fstat(stream.fileno()).st_size == size and "sha256:" + hash_stream(stream) == digest,
                    "qualification evidence transport digest differs")
            stream.seek(0)
            with zipfile.ZipFile(stream) as evidence:
                records = evidence.infolist()
                require(len(records) == 2 and {item.filename for item in records}
                        == {"archive-integrity.json", "qualification.json"}
                        and all(0 < item.file_size <= 1024**2 and not item.flag_bits & 1 for item in records),
                        "unexpected qualification evidence archive")
                integrity = parse_json(evidence.read("archive-integrity.json"))
                qualified = parse_json(evidence.read("qualification.json"))
    require(integrity == report, "qualification used different archive bytes, pins or verifier output")
    inventories = {role: {"image": entry["image"], "package_count": len(entry["packages"])} for role, entry in pins.items()}
    require(qualified == dict(report, installed_inventory=inventories), "qualification inventories differ")
    return {
        "schema": 1, "claim": "historical-builder-import", "origin": report["origin"],
        "original_build_provenance_authenticated": False,
        "scope": "Exact historical snapshot adoption and delivery; no claim of original source/dependency build provenance.",
        "archive": lock["archive"], "source_asset": lock["source"],
        "builder_images": report["builder_images"], "installed_inventory": inventories,
        "builder_inventory_sha256": hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "qualification": {"run_id": run_id, "run_attempt": attempt, "source_commit": source_commit,
                          "artifact_id": artifact_id, "artifact_digest": digest},
        "promotion": {"repository": REPOSITORY, "workflow": SIGNER, "source_commit": source_commit,
                      "source_ref": "refs/heads/main"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fetch", "inspect", "qualify", "verify", "predicate"))
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--lock", type=Path, default=HERE / "builder-delivery.json")
    parser.add_argument("--pins", type=Path, default=HERE / "builders.json")
    parser.add_argument("--load", action="store_true", help="Import after verification; intended for disposable CI runners")
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--asset-id", type=int, help="Exact candidate release asset ID, checked against the reviewed tag/name/hash")
    parser.add_argument("--qualification-run", type=int)
    parser.add_argument("--source-commit", help="Promotion workflow's actual GITHUB_SHA, never caller-supplied workflow input")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        lock = parse_json(args.lock.read_bytes())
        require(lock.get("schema") == 1, "unsupported delivery lock")
        pins = parse_json(args.pins.read_bytes())
        if args.operation == "fetch":
            require(not args.load, "fetch never loads images")
            fetch_candidate(args.archive, lock, args.asset_id, gh=args.gh)
        report = process_archive("inspect" if args.operation in {"fetch", "predicate"} else args.operation,
                                 args.archive, lock, pins, load=args.load, gh=args.gh)
        if args.operation == "predicate":
            report = import_predicate(lock, pins, report, args.qualification_run, args.source_commit, gh=args.gh)
        text = json.dumps(report, indent=2) + "\n"
        if args.report:
            args.report.write_text(text)
        print(text, end="")
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"builder delivery refused: {error}\n")


if __name__ == "__main__":
    main()

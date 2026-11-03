#!/usr/bin/env python3
"""Verify retained GitHub artifacts before resuming macOS notarization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid
import zipfile
import zlib


MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_ARTIFACTS = 1000
WORKFLOW_PATH = ".github/workflows/build-packages.yml"


class PreparationError(RuntimeError):
    """Retained artifacts failed a required binding or integrity check."""


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{" + str(length) + r"}", value) is not None


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    path.chmod(0o600)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gh_json(endpoint: str) -> dict:
    try:
        result = subprocess.run(
            ["gh", "api", "--method", "GET", endpoint], capture_output=True,
            text=True, encoding="utf-8", timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreparationError("GitHub metadata request did not finish") from error
    if result.returncode:
        # Never include CLI stderr, tokens or redirected download URLs in output.
        raise PreparationError(f"GitHub metadata request failed with exit {result.returncode}")
    try:
        value = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise PreparationError("GitHub metadata response is not valid JSON") from error
    if not isinstance(value, dict):
        raise PreparationError("GitHub metadata response is not an object")
    return value


def _check_run(run: dict, repo: str, run_id: int, attempt: int, source_sha: str) -> int:
    repository = run.get("repository")
    head_repository = run.get("head_repository")
    if not isinstance(repository, dict) or not isinstance(head_repository, dict):
        raise PreparationError("Source run repository metadata is missing")
    repository_id = repository.get("id")
    if (not _positive(repository_id)
            or str(repository.get("full_name", "")).casefold() != repo.casefold()
            or head_repository.get("id") != repository_id
            or str(head_repository.get("full_name", "")).casefold() != repo.casefold()):
        raise PreparationError("Source run must belong to the requested repository")
    if (not _positive(run.get("id")) or not _positive(run.get("run_attempt"))
            or run["id"] != run_id or run["run_attempt"] != attempt):
        raise PreparationError("Source run or attempt does not match the request")
    if run.get("path") != WORKFLOW_PATH or run.get("event") not in {"workflow_dispatch", "push"}:
        raise PreparationError("Source run is not an eligible Build Packages run")
    if not _hex(run.get("head_sha"), 40) or run["head_sha"].lower() != source_sha:
        raise PreparationError("Source run commit does not match the expected source SHA")
    return repository_id


def _artifacts(endpoint: str, output: Path) -> list[dict]:
    collected = []
    total = None
    for page in range(1, MAX_ARTIFACTS // 100 + 1):
        response = _gh_json(f"{endpoint}?per_page=100&page={page}")
        _write_json(output / f"artifact-page-{page:03d}.json", response)
        count, entries = response.get("total_count"), response.get("artifacts")
        if type(count) is not int or not 0 <= count <= MAX_ARTIFACTS or not isinstance(entries, list):
            raise PreparationError("Artifact listing is invalid or exceeds its bound")
        if total is not None and count != total:
            raise PreparationError("Artifact listing changed during preparation")
        total = count
        if any(not isinstance(item, dict) for item in entries):
            raise PreparationError("Artifact listing contains invalid entries")
        collected.extend(entries)
        if len(collected) == total:
            identifiers = [item.get("id") for item in collected]
            if any(not _positive(value) for value in identifiers) or len(set(identifiers)) != len(identifiers):
                raise PreparationError("Artifact listing contains duplicate or invalid IDs")
            return collected
        if not entries or len(collected) > total:
            raise PreparationError("Artifact listing is incomplete or inconsistent")
    raise PreparationError("Artifact listing exceeds its page bound")


def _select(artifacts: list[dict], name: str, run_id: int, repository_id: int, source_sha: str) -> dict:
    matches = [item for item in artifacts if item.get("name") == name]
    if len(matches) != 1:
        raise PreparationError(f"Expected exactly one retained artifact named {name}")
    artifact = matches[0]
    binding = artifact.get("workflow_run")
    if (not isinstance(binding, dict) or binding.get("id") != run_id
            or binding.get("repository_id") != repository_id
            or binding.get("head_repository_id") != repository_id
            or not _hex(binding.get("head_sha"), 40)
            or binding["head_sha"].lower() != source_sha):
        raise PreparationError(f"Artifact {name} is not bound to the expected source run")
    if artifact.get("expired") is not False:
        raise PreparationError(f"Artifact {name} is expired or has no expiry metadata")
    size = artifact.get("size_in_bytes")
    digest = artifact.get("digest")
    if not _positive(size) or size > MAX_ARCHIVE_BYTES:
        raise PreparationError(f"Artifact {name} size exceeds its bound or is invalid")
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest) is None:
        raise PreparationError(f"Artifact {name} has no valid GitHub SHA-256 digest")
    return artifact


def _download(repo: str, artifact: dict, path: Path) -> None:
    with path.open("xb") as output:
        path.chmod(0o600)
        try:
            result = subprocess.run(
                ["gh", "api", "--method", "GET",
                 f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
                stdout=output, stderr=subprocess.PIPE, timeout=120, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PreparationError("GitHub artifact download did not finish") from error
    if result.returncode:
        raise PreparationError(f"GitHub artifact download failed with exit {result.returncode}")
    if path.stat().st_size != artifact["size_in_bytes"]:
        raise PreparationError(f"Artifact {artifact['name']} ZIP size does not match GitHub metadata")
    if _sha256(path) != artifact["digest"].split(":", 1)[1].lower():
        raise PreparationError(f"Artifact {artifact['name']} ZIP SHA-256 does not match GitHub metadata")


def _extract(archive: Path, output: Path, *, checkpoint: bool) -> list[Path]:
    try:
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            names = [member.filename for member in members]
            if (len(names) != len(set(names)) or not members or len(members) > 2
                    or (checkpoint and set(names) != {"checkpoint.json", "signed-upload.pkg"})
                    or (not checkpoint and (len(names) != 1 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.pkg", names[0]) is None))):
                raise PreparationError("ZIP must contain only the expected unique regular files")
            total = 0
            for member in members:
                mode = member.external_attr >> 16
                if (member.orig_filename != member.filename or "/" in member.filename
                        or "\\" in member.filename or member.is_dir()
                        or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
                        or member.flag_bits & 1
                        or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}):
                    raise PreparationError("ZIP contains an unsafe path, link, type or encoding")
                limit = MAX_MANIFEST_BYTES if member.filename == "checkpoint.json" else MAX_ARCHIVE_BYTES
                if not 0 < member.file_size <= limit:
                    raise PreparationError("ZIP member size is invalid or exceeds its bound")
                total += member.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise PreparationError("ZIP expanded size exceeds its bound")
            output.mkdir(mode=0o700)
            paths = []
            for member in members:
                path = output / member.filename
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, "wb") as target, source.open(member) as content:
                    size = 0
                    for block in iter(lambda: content.read(1024 * 1024), b""):
                        size += len(block)
                        if size > member.file_size:
                            raise PreparationError("ZIP member exceeds its declared size")
                        target.write(block)
                    if size != member.file_size:
                        raise PreparationError("ZIP member does not match its declared size")
                paths.append(path)
            return paths
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as error:
        if isinstance(error, PreparationError):
            raise
        raise PreparationError("Artifact is not a valid supported ZIP") from error


def _checkpoint(directory: Path, source: Path, source_hash: str, signed_hash: str) -> dict:
    try:
        manifest = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise PreparationError("Checkpoint manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise PreparationError("Checkpoint schema version is unsupported")
    if manifest.get("state") not in {"pending", "accepted"}:
        raise PreparationError("Checkpoint has no resumable submission")
    submission = manifest.get("submission_id")
    try:
        identifier = uuid.UUID(submission) if isinstance(submission, str) else None
    except ValueError:
        identifier = None
    if identifier is None or not identifier.int or str(identifier) != submission:
        raise PreparationError("Checkpoint submission ID is not a canonical UUID")
    signed = directory / "signed-upload.pkg"
    if _sha256(source) != source_hash or manifest.get("source_sha256") != source_hash:
        raise PreparationError("Original unsigned package SHA-256 does not match its independent pin")
    if _sha256(signed) != signed_hash or manifest.get("signed_upload_sha256") != signed_hash:
        raise PreparationError("Signed upload SHA-256 does not match its independent pin")
    for key, path in (("source_bytes", source), ("signed_upload_bytes", signed)):
        if not _positive(manifest.get(key)) or manifest[key] != path.stat().st_size:
            raise PreparationError("Checkpoint package byte count does not match the retained file")
    return manifest


def prepare_resume(*, repo: str, run_id: int, attempt: int, expected_source_sha: str,
                   source_sha256: str, signed_sha256: str, output_dir: Path) -> dict:
    if (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo) is None
            or not _positive(run_id) or not _positive(attempt)
            or not _hex(expected_source_sha, 40) or not _hex(source_sha256, 64) or not _hex(signed_sha256, 64)):
        raise PreparationError("Repository, run, attempt or independent SHA pins are invalid")
    expected_source_sha, source_sha256, signed_sha256 = (
        value.lower() for value in (expected_source_sha, source_sha256, signed_sha256)
    )
    if ".." in output_dir.parts:
        raise PreparationError("Output directory cannot contain parent traversal")
    output = output_dir.absolute()
    for parent in output.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise PreparationError("Output directory parents must be existing directories without symlinks")
    output.mkdir(mode=0o700)  # Existing files, directories and symlinks fail closed.
    output.chmod(0o700)
    base = f"repos/{repo}/actions/runs/{run_id}"
    run = _gh_json(f"{base}/attempts/{attempt}")
    _write_json(output / "source-run.json", run)
    repository_id = _check_run(run, repo, run_id, attempt, expected_source_sha)
    artifacts = _artifacts(f"{base}/artifacts", output)
    unsigned = _select(artifacts, "macos-packages", run_id, repository_id, expected_source_sha)
    checkpoint = _select(artifacts, f"macos-notarization-checkpoint-attempt-{attempt}", run_id, repository_id, expected_source_sha)
    for label, artifact in (("unsigned", unsigned), ("checkpoint", checkpoint)):
        _write_json(output / f"{label}-artifact.json", artifact)
        _download(repo, artifact, output / f"{label}-artifact.zip")
    source = _extract(output / "unsigned-artifact.zip", output / "unsigned", checkpoint=False)[0]
    checkpoint_dir = output / "checkpoint"
    _extract(output / "checkpoint-artifact.zip", checkpoint_dir, checkpoint=True)
    manifest = _checkpoint(checkpoint_dir, source, source_sha256, signed_sha256)
    provenance = {
        "source": str(source), "checkpoint_directory": str(checkpoint_dir),
        "source_run_id": run_id, "source_attempt": attempt, "source_sha": expected_source_sha,
        "repository": repo, "repository_id": repository_id,
        "source_sha256": source_sha256, "signed_sha256": signed_sha256,
        "submission_id": manifest["submission_id"],
        "unsigned_artifact": unsigned, "checkpoint_artifact": checkpoint,
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--signed-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = prepare_resume(**vars(arguments))
    except (PreparationError, OSError) as error:
        print(f"macOS resume preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

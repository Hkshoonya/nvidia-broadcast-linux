#!/usr/bin/env python3
"""Sign, notarize and verify an installer without installing or publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Optional, Sequence
import uuid


class SigningError(RuntimeError):
    """A package failed a required distribution gate."""


def _sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SigningError(f"Not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
            return digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _copy_regular(source: Path, output: Path) -> None:
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SigningError("Input snapshot requires a regular file")
        with os.fdopen(descriptor, "rb") as original:
            descriptor = -1
            with output.open("xb") as snapshot:
                shutil.copyfileobj(original, snapshot)
        output.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _manifest(directory: Path, summary: dict) -> None:
    _write_json(directory / "evidence-manifest.json", {
        "status": summary["status"],
        "artifact": ({
            "name": Path(summary["output"]).name, "sha256": summary["final_sha256"],
            "bytes": summary["final_bytes"],
        } if summary["status"] == "verified" else None),
        "evidence": [
            {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(directory.iterdir()) if path.name != "evidence-manifest.json"
        ],
    })


class _Evidence:
    def __init__(self, directory: Path):
        self.directory = directory
        self.sequence = 0

    def run(
        self, label: str, arguments: Sequence[str], *, timeout: int = 300,
        required: bool = True,
    ) -> subprocess.CompletedProcess:
        self.sequence += 1
        base = self.directory / f"{self.sequence:02d}-{label}"
        record = {"argv": list(arguments), "timeout_seconds": timeout}
        _write_json(base.with_suffix(".json"), record)
        environment = dict(os.environ, LC_ALL="C", LANG="C")
        try:
            result = subprocess.run(
                list(arguments), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                check=False, env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(error=str(error), returncode=None)
            if isinstance(error, subprocess.TimeoutExpired):
                for stream in ("stdout", "stderr"):
                    content = getattr(error, stream) or ""
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="replace")
                    base.with_suffix(f".{stream}.txt").write_text(content, encoding="utf-8")
            _write_json(base.with_suffix(".json"), record)
            raise SigningError(f"{label} did not finish: {error}") from error
        record["returncode"] = result.returncode
        _write_json(base.with_suffix(".json"), record)
        base.with_suffix(".stdout.txt").write_text(result.stdout, encoding="utf-8")
        base.with_suffix(".stderr.txt").write_text(result.stderr, encoding="utf-8")
        if required and result.returncode != 0:
            raise SigningError(f"{label} failed with exit {result.returncode}; see {base.name} evidence")
        return result


def _tree(directory: Path) -> dict:
    """Describe every expanded member without following payload symlinks."""
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise SigningError("Expanded package is not a directory")
    entries = {}
    pending = [directory]
    while pending:
        parent = pending.pop()
        for member in sorted(parent.iterdir()):
            metadata = member.lstat()
            entry = {"mode": stat.S_IMODE(metadata.st_mode)}
            if stat.S_ISLNK(metadata.st_mode):
                entry.update(type="symlink", target=os.readlink(member))
            elif stat.S_ISDIR(metadata.st_mode):
                entry["type"] = "directory"
                pending.append(member)
            elif stat.S_ISREG(metadata.st_mode):
                entry.update(type="file", size=metadata.st_size, sha256=_sha256(member))
            else:
                raise SigningError(f"Unsupported expanded member type: {member}")
            entries[member.relative_to(directory).as_posix()] = entry
    if not entries:
        raise SigningError("Expanded package contains no members")
    return entries


def _expand(evidence: _Evidence, package: Path, output: Path, label: str) -> dict:
    evidence.run(label, ["/usr/sbin/pkgutil", "--expand-full", str(package), str(output)])
    inventory = _tree(output)
    _write_json(evidence.directory / f"{label}-inventory.json", inventory)
    return inventory


def _signature(output: str, team_id: str) -> str:
    trusted = {
        "signed by a certificate trusted by macOS",
        "signed by a developer certificate issued by Apple for distribution",
    }
    status = re.search(r"^\s*Status:\s*(.+?)\s*$", output, re.MULTILINE)
    signer = re.search(r"^\s*1\.\s+(.+?)\s*$", output, re.MULTILINE)
    if status is None or status.group(1) not in trusted:
        raise SigningError("Installer signature is not trusted by macOS")
    if not re.search(r"^\s*Signed with a trusted timestamp on:\s*\S", output, re.MULTILINE):
        raise SigningError("Installer signature has no trusted timestamp")
    if signer is None or not re.fullmatch(
        r"Developer ID Installer: .+ \(" + re.escape(team_id) + r"\)", signer.group(1)
    ):
        raise SigningError("First signer is not the expected Developer ID Installer team")
    return signer.group(1)


def _submission_id(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    return str(parsed) if parsed.int and str(parsed) == value.lower() else None


def _json_object(content: str, label: str) -> dict:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise SigningError(f"{label} did not contain valid JSON") from error
    if not isinstance(value, dict):
        raise SigningError(f"{label} did not contain a JSON object")
    return value


def _new_path(path: Path, *, suffix: Optional[str] = None) -> Path:
    if os.path.lexists(path):
        raise SigningError(f"Refusing to overwrite existing path: {path}")
    if suffix is not None and path.suffix != suffix:
        raise SigningError(f"Expected a {suffix} path: {path}")
    return path.parent.resolve(strict=True) / path.name


def _timeout_response(stdout: str, stderr: str) -> dict:
    """Timeout JSON may be emitted on stderr rather than stdout."""
    responses = []
    for content in (stdout, stderr):
        try:
            value = _json_object(content, "notarization timeout response")
        except SigningError:
            continue
        if _submission_id(value.get("id")) is not None:
            responses.append(value)
    identifiers = {_submission_id(value["id"]) for value in responses}
    if len(identifiers) != 1:
        return {"id": None, "status": None}
    response = responses[0]
    return {"id": next(iter(identifiers)), "status": response.get("status")}


def _checkpoint_write(directory: Path, checkpoint: dict) -> None:
    temporary = directory / ".checkpoint.json.tmp"
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(checkpoint, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.chmod(0o600)
    os.replace(temporary, directory / "checkpoint.json")


def _checkpoint_submission(
    directory: Optional[Path], checkpoint: dict, submission: dict, *, timed_out: bool = False,
) -> None:
    if directory is None:
        return
    checkpoint.update(
        submission_id=_submission_id(submission.get("id")),
        notarization_status=submission.get("status"),
        notarization_wait_status="timed_out" if timed_out else None,
    )
    status = submission.get("status")
    checkpoint["state"] = (
        "rejected" if status in {"Invalid", "Rejected"} else
        "accepted" if status == "Accepted" else "pending"
    )
    _checkpoint_write(directory, checkpoint)


def _complete_verification(
    evidence: _Evidence, signed: Path, staging: Path, output: Path, summary: dict,
    original_tree: dict, unchanged_source: Callable[[], bool], authentication: Sequence[str],
    submission: dict, returncode: int, *, require_log_hash: bool = False,
) -> None:
    submission_id = _submission_id(submission.get("id"))
    log = None
    if submission_id is not None:
        log_path = evidence.directory / "notarization-log.json"
        log_result = evidence.run("notary-log", [
            "/usr/bin/xcrun", "notarytool", "log", submission_id,
            *authentication, str(log_path),
        ], required=False)
        if log_result.returncode == 0:
            log = _json_object(log_path.read_text(encoding="utf-8"), "notarization log")
            log_path.chmod(0o600)
    if returncode != 0 or submission_id is None or submission.get("status") != "Accepted":
        raise SigningError("Notarization did not finish successfully with Accepted and a valid submission ID")
    if log is None or _submission_id(log.get("jobId")) != submission_id or log.get("status") != "Accepted":
        raise SigningError("Notarization log does not confirm this accepted submission")
    if require_log_hash and not isinstance(log.get("sha256"), str):
        raise SigningError("Resumed notarization log must identify the uploaded archive SHA-256")
    if "sha256" in log and log["sha256"] != summary["signed_upload_sha256"]:
        raise SigningError("Notarization log identifies different archive bytes")
    evidence.run("stapler-staple", ["/usr/bin/xcrun", "stapler", "staple", str(signed)])
    evidence.run("stapler-validate", ["/usr/bin/xcrun", "stapler", "validate", str(signed)])
    assessment = evidence.run("gatekeeper", [
        "/usr/sbin/spctl", "--assess", "--type", "install", "--verbose=4", str(signed),
    ])
    assessment_output = assessment.stdout + "\n" + assessment.stderr
    if not re.search(r": accepted\s*$", assessment_output, re.MULTILINE) or not re.search(
        r"^source=Notarized Developer ID\s*$", assessment_output, re.MULTILINE
    ):
        raise SigningError("Gatekeeper did not confirm accepted Notarized Developer ID")
    final_signature = evidence.run("signature-final", [
        "/usr/sbin/pkgutil", "--check-signature", str(signed),
    ])
    if _signature(final_signature.stdout, summary["expected_team_id"]) != summary["signer"]:
        raise SigningError("Stapling changed the expected signing identity")
    if _expand(evidence, signed, staging / "final-expanded", "expand-final") != original_tree:
        raise SigningError("Stapling changed expanded package content, modes or symlinks")
    if not unchanged_source():
        raise SigningError("Original input package changed during signing")
    signed.chmod(0o644)
    summary.update(
        status="verified", final_sha256=_sha256(signed), final_bytes=signed.stat().st_size,
        expanded_members=len(original_tree), original_unchanged=True,
    )
    _write_json(evidence.directory / "summary.json", summary)
    _manifest(evidence.directory, summary)
    # An exclusive link cannot overwrite an artifact created during verification.
    os.link(signed, output, follow_symlinks=False)


def sign_package(
    source: Path, output: Path, *, identity: str, team_id: str, keychain: Path,
    notary_profile: str, evidence_directory: Path, checkpoint_directory: Optional[Path] = None,
) -> dict:
    """Only expose the final archive after native Apple verification succeeds."""
    if platform.system() != "Darwin":
        raise SigningError("Signing and notarization require macOS")
    if not re.fullmatch(r"[A-Z0-9]{10}", team_id):
        raise SigningError("Expected a 10-character Apple Team ID")
    if not re.fullmatch(r"[0-9A-Fa-f]{40}", identity) and not re.fullmatch(
        r"Developer ID Installer: [^\r\n]+ \(" + re.escape(team_id) + r"\)", identity
    ):
        raise SigningError("Use an Installer identity SHA-1 or the expected Developer ID Installer label")
    if not notary_profile.strip() or any(ord(c) < 32 for c in notary_profile):
        raise SigningError("A stored notary profile name is required")
    if source.suffix != ".pkg" or source.is_symlink() or not source.is_file():
        raise SigningError("Input must be a regular .pkg file, not a symlink")
    if keychain.is_symlink() or not keychain.is_file():
        raise SigningError("Dedicated keychain must be a regular file, not a symlink")
    source = source.resolve(strict=True)
    keychain = keychain.resolve(strict=True)
    output = _new_path(output, suffix=".pkg")
    evidence_directory = _new_path(evidence_directory)
    if output == evidence_directory or evidence_directory == source or output == keychain:
        raise SigningError("Artifact and evidence paths must be distinct")
    if checkpoint_directory is not None:
        checkpoint_directory = _new_path(checkpoint_directory)
        if checkpoint_directory in (source, output, keychain, evidence_directory) or (
            checkpoint_directory.is_relative_to(evidence_directory) or
            evidence_directory.is_relative_to(checkpoint_directory) or
            output.is_relative_to(checkpoint_directory) or source.is_relative_to(checkpoint_directory) or
            keychain.is_relative_to(checkpoint_directory)
        ):
            raise SigningError("Checkpoint, artifact, credentials and evidence paths must be separate")
    original_hash = _sha256(source)
    original_identity = (source.stat().st_dev, source.stat().st_ino)
    evidence_directory.mkdir(mode=0o700)
    evidence = _Evidence(evidence_directory)
    summary = {
        "status": "started", "source": str(source), "source_sha256": original_hash,
        "output": str(output), "expected_team_id": team_id, "identity": identity,
        "keychain_profile": notary_profile,
    }
    checkpoint = {}
    if checkpoint_directory is not None:
        summary["checkpoint_directory"] = str(checkpoint_directory)

    def unchanged_source() -> bool:
        metadata = source.lstat()
        return (
            stat.S_ISREG(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == original_identity
            and _sha256(source) == original_hash
        )

    try:
        with tempfile.TemporaryDirectory(prefix=".nvbroadcast-signing-", dir=output.parent) as temporary:
            staging = Path(temporary)
            unsigned = staging / "unsigned.pkg"
            signed = staging / output.name
            _copy_regular(source, unsigned)
            if _sha256(unsigned) != original_hash:
                raise SigningError("Input changed while creating its private snapshot")
            original_tree = _expand(evidence, unsigned, staging / "input-expanded", "expand-input")
            evidence.run("productsign", [
                "/usr/bin/productsign", "--sign", identity, "--keychain", str(keychain),
                "--timestamp", str(unsigned), str(signed),
            ], timeout=600)
            signature = evidence.run("signature-signed", [
                "/usr/sbin/pkgutil", "--check-signature", str(signed),
            ])
            summary["signer"] = _signature(signature.stdout, team_id)
            if _expand(evidence, signed, staging / "signed-expanded", "expand-signed") != original_tree:
                raise SigningError("Signing changed expanded package content, modes or symlinks")
            upload_hash = _sha256(signed)
            summary["signed_upload_sha256"] = upload_hash
            if checkpoint_directory is not None:
                checkpoint_directory.mkdir(mode=0o700)
                retained = checkpoint_directory / "signed-upload.pkg"
                _copy_regular(signed, retained)
                if _sha256(retained) != upload_hash:
                    raise SigningError("Checkpoint copy does not match the signed upload")
                checkpoint = {
                    "schema_version": 1, "state": "prepared", "submission_id": None,
                    "notarization_status": None, "source_sha256": original_hash,
                    "source_bytes": source.stat().st_size, "signed_upload_sha256": upload_hash,
                    "signed_upload_bytes": retained.stat().st_size, "expected_team_id": team_id,
                    "signer": summary["signer"], "identity": identity,
                }
                _checkpoint_write(checkpoint_directory, checkpoint)
            authentication = ["--keychain-profile", notary_profile, "--keychain", str(keychain)]
            try:
                response = evidence.run("notary-submit", [
                    "/usr/bin/xcrun", "notarytool", "submit", str(signed), *authentication,
                    "--wait", "--timeout", "30m", "--output-format", "json",
                ], timeout=2100, required=False)
            except SigningError as error:
                if isinstance(error.__cause__, subprocess.TimeoutExpired):
                    partial = error.__cause__
                    streams = [
                        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                        for value in (partial.stdout, partial.stderr)
                    ]
                    submission = _timeout_response(*streams)
                    summary.update(
                        submission_id=submission["id"], notarization_status=submission["status"],
                        notarization_wait_status="timed_out",
                    )
                    _checkpoint_submission(checkpoint_directory, checkpoint, submission, timed_out=True)
                raise
            submission = (
                _timeout_response(response.stdout, response.stderr) if response.returncode == 124 else
                _json_object(response.stdout, "notarization response")
            )
            submission_id = _submission_id(submission.get("id"))
            summary.update(submission_id=submission_id, notarization_status=submission.get("status"))
            _checkpoint_submission(checkpoint_directory, checkpoint, submission, timed_out=response.returncode == 124)
            if response.returncode == 124:
                summary["notarization_wait_status"] = "timed_out"
                raise SigningError("Notarization wait timed out; the submission remains unverified")
            _complete_verification(
                evidence, signed, staging, output, summary, original_tree,
                unchanged_source, authentication, submission, response.returncode,
            )
        return summary
    except Exception as error:
        summary.update(status="failed", error=str(error))
        try:
            summary["original_unchanged"] = unchanged_source()
        except OSError:
            summary["original_unchanged"] = False
        _write_json(evidence_directory / "summary.json", summary)
        _manifest(evidence_directory, summary)
        raise


def resume_package(
    source: Path, output: Path, *, checkpoint_directory: Path, source_sha256: str,
    signed_sha256: str, team_id: str, keychain: Path, notary_profile: str,
    evidence_directory: Path,
) -> dict:
    """Verify the retained upload against its existing Apple submission."""
    if platform.system() != "Darwin":
        raise SigningError("Signing and notarization require macOS")
    if not re.fullmatch(r"[A-Z0-9]{10}", team_id):
        raise SigningError("Expected a 10-character Apple Team ID")
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (source_sha256, signed_sha256)):
        raise SigningError("Resume requires independently pinned lowercase source and signed SHA-256 hashes")
    if not notary_profile.strip() or any(ord(c) < 32 for c in notary_profile):
        raise SigningError("A stored notary profile name is required")
    if source.suffix != ".pkg" or source.is_symlink() or not source.is_file():
        raise SigningError("Input must be a regular .pkg file, not a symlink")
    if keychain.is_symlink() or not keychain.is_file():
        raise SigningError("Dedicated keychain must be a regular file, not a symlink")
    if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
        raise SigningError("Checkpoint must be a directory, not a symlink")
    source = source.resolve(strict=True)
    keychain = keychain.resolve(strict=True)
    checkpoint_directory = checkpoint_directory.resolve(strict=True)
    output = _new_path(output, suffix=".pkg")
    evidence_directory = _new_path(evidence_directory)
    if output in (source, keychain) or evidence_directory in (source, keychain, output) or any(
        path.is_relative_to(checkpoint_directory) or checkpoint_directory.is_relative_to(path)
        for path in (source, keychain, output, evidence_directory)
    ):
        raise SigningError("Checkpoint, artifact, credentials and evidence paths must be separate")
    checkpoint_path = checkpoint_directory / "checkpoint.json"
    descriptor = os.open(checkpoint_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, encoding="utf-8") as checkpoint_file:
        if not stat.S_ISREG(os.fstat(checkpoint_file.fileno()).st_mode):
            raise SigningError("Checkpoint metadata must be a regular file")
        checkpoint_content = checkpoint_file.read()
    checkpoint = _json_object(checkpoint_content, "checkpoint")
    if type(checkpoint.get("schema_version")) is not int or checkpoint["schema_version"] != 1:
        raise SigningError("Unsupported checkpoint schema")
    if checkpoint.get("state") not in {"prepared", "pending", "accepted"}:
        raise SigningError("Checkpoint is not a resumable submission")
    if checkpoint.get("expected_team_id") != team_id:
        raise SigningError("Checkpoint does not match the independently expected team")
    if checkpoint.get("source_sha256") != source_sha256 or checkpoint.get("signed_upload_sha256") != signed_sha256:
        raise SigningError("Checkpoint does not match the independently pinned archive hashes")
    submission_id = _submission_id(checkpoint.get("submission_id"))
    if submission_id is None:
        raise SigningError("Checkpoint has no valid submission ID; query Apple status separately")
    retained = checkpoint_directory / "signed-upload.pkg"
    if retained.is_symlink() or not retained.is_file():
        raise SigningError("Retained signed upload must be a regular file, not a symlink")
    for path, hash_value, size_key in (
        (source, source_sha256, "source_bytes"), (retained, signed_sha256, "signed_upload_bytes"),
    ):
        if type(checkpoint.get(size_key)) is not int or checkpoint[size_key] != path.stat().st_size:
            raise SigningError("Checkpoint archive size does not match")
        if _sha256(path) != hash_value:
            raise SigningError("Archive does not match its independently pinned SHA-256")
    if not isinstance(checkpoint.get("signer"), str) or not re.fullmatch(
        r"Developer ID Installer: [^\r\n]+ \(" + re.escape(team_id) + r"\)", checkpoint["signer"],
    ):
        raise SigningError("Checkpoint has no expected Installer signer")
    original_identity = (source.stat().st_dev, source.stat().st_ino)
    retained_identity = (retained.stat().st_dev, retained.stat().st_ino)
    evidence_directory.mkdir(mode=0o700)
    evidence = _Evidence(evidence_directory)
    summary = {
        "status": "started", "source": str(source), "source_sha256": source_sha256,
        "output": str(output), "expected_team_id": team_id, "keychain_profile": notary_profile,
        "submission_id": submission_id, "signed_upload_sha256": signed_sha256,
        "checkpoint_directory": str(checkpoint_directory),
        "checkpoint_sha256": hashlib.sha256(checkpoint_content.encode("utf-8")).hexdigest(),
        "resumed": True,
    }

    def unchanged_source() -> bool:
        for path, identity, hash_value in (
            (source, original_identity, source_sha256), (retained, retained_identity, signed_sha256),
        ):
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != identity or _sha256(path) != hash_value:
                return False
        return True

    try:
        with tempfile.TemporaryDirectory(prefix=".nvbroadcast-signing-", dir=output.parent) as temporary:
            staging = Path(temporary)
            unsigned = staging / "unsigned.pkg"
            signed = staging / output.name
            _copy_regular(source, unsigned)
            _copy_regular(retained, signed)
            if _sha256(unsigned) != source_sha256 or _sha256(signed) != signed_sha256:
                raise SigningError("Archive changed while creating its private snapshot")
            original_tree = _expand(evidence, unsigned, staging / "input-expanded", "expand-input")
            signature = evidence.run("signature-signed", ["/usr/sbin/pkgutil", "--check-signature", str(signed)])
            summary["signer"] = _signature(signature.stdout, team_id)
            if summary["signer"] != checkpoint["signer"]:
                raise SigningError("Retained upload has a different Installer signer")
            if _expand(evidence, signed, staging / "signed-expanded", "expand-signed") != original_tree:
                raise SigningError("Signing changed expanded package content, modes or symlinks")
            authentication = ["--keychain-profile", notary_profile, "--keychain", str(keychain)]
            response = evidence.run("notary-info", [
                "/usr/bin/xcrun", "notarytool", "info", submission_id, *authentication,
                "--output-format", "json",
            ], required=False)
            submission = _json_object(response.stdout, "existing notarization response")
            summary["notarization_status"] = submission.get("status")
            if response.returncode != 0 or _submission_id(submission.get("id")) != submission_id or submission.get("status") != "Accepted":
                raise SigningError("Existing submission is not confirmed Accepted")
            _complete_verification(
                evidence, signed, staging, output, summary, original_tree,
                unchanged_source, authentication, submission, response.returncode,
                require_log_hash=True,
            )
        return summary
    except Exception as error:
        summary.update(status="failed", error=str(error))
        try:
            summary["original_unchanged"] = unchanged_source()
        except OSError:
            summary["original_unchanged"] = False
        _write_json(evidence_directory / "summary.json", summary)
        _manifest(evidence_directory, summary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity")
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--keychain", required=True, type=Path)
    parser.add_argument("--notary-profile", required=True)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, help="Retain the exact upload privately for later verification")
    parser.add_argument("--resume-from", type=Path, help="Verify a retained upload without signing or submitting again")
    parser.add_argument("--source-sha256", help="Independent original unsigned archive hash required for resume")
    parser.add_argument("--signed-sha256", help="Independent pre-staple signed upload hash required for resume")
    args = parser.parse_args()
    if args.resume_from is not None:
        if args.identity is not None or args.checkpoint_dir is not None or not args.source_sha256 or not args.signed_sha256:
            parser.error("--resume-from requires --source-sha256 and --signed-sha256, without --identity or --checkpoint-dir")
    elif args.identity is None or args.source_sha256 is not None or args.signed_sha256 is not None:
        parser.error("Signing requires --identity; hash pins are reserved for --resume-from")
    try:
        if args.resume_from is not None:
            summary = resume_package(
                args.input, args.output, checkpoint_directory=args.resume_from,
                source_sha256=args.source_sha256, signed_sha256=args.signed_sha256,
                team_id=args.team_id, keychain=args.keychain, notary_profile=args.notary_profile,
                evidence_directory=args.evidence_dir,
            )
        else:
            summary = sign_package(
                args.input, args.output, identity=args.identity, team_id=args.team_id,
                keychain=args.keychain, notary_profile=args.notary_profile,
                evidence_directory=args.evidence_dir, checkpoint_directory=args.checkpoint_dir,
            )
    except (SigningError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Verified {args.output}: SHA-256 {summary['final_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from typing import Optional, Sequence
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


def sign_package(
    source: Path, output: Path, *, identity: str, team_id: str, keychain: Path,
    notary_profile: str, evidence_directory: Path,
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
    original_hash = _sha256(source)
    original_identity = (source.stat().st_dev, source.stat().st_ino)
    evidence_directory.mkdir(mode=0o700)
    evidence = _Evidence(evidence_directory)
    summary = {
        "status": "started", "source": str(source), "source_sha256": original_hash,
        "output": str(output), "expected_team_id": team_id, "identity": identity,
        "keychain_profile": notary_profile,
    }

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
            authentication = ["--keychain-profile", notary_profile, "--keychain", str(keychain)]
            response = evidence.run("notary-submit", [
                "/usr/bin/xcrun", "notarytool", "submit", str(signed), *authentication,
                "--wait", "--timeout", "30m", "--output-format", "json",
            ], timeout=2100, required=False)
            submission = _json_object(response.stdout, "notarization response")
            submission_id = _submission_id(submission.get("id"))
            summary.update(submission_id=submission_id, notarization_status=submission.get("status"))
            log = None
            if submission_id is not None:
                log_path = evidence_directory / "notarization-log.json"
                log_result = evidence.run("notary-log", [
                    "/usr/bin/xcrun", "notarytool", "log", submission_id,
                    *authentication, str(log_path),
                ], required=False)
                if log_result.returncode == 0:
                    log = _json_object(log_path.read_text(encoding="utf-8"), "notarization log")
                    log_path.chmod(0o600)
            if response.returncode != 0 or submission_id is None or submission.get("status") != "Accepted":
                raise SigningError("Notarization did not finish successfully with Accepted and a valid submission ID")
            if log is None or _submission_id(log.get("jobId")) != submission_id or log.get("status") != "Accepted":
                raise SigningError("Notarization log does not confirm this accepted submission")
            if "sha256" in log and log["sha256"] != upload_hash:
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
            if _signature(final_signature.stdout, team_id) != summary["signer"]:
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
            _write_json(evidence_directory / "summary.json", summary)
            _manifest(evidence_directory, summary)
            # An exclusive link cannot overwrite an artifact created during verification.
            os.link(signed, output, follow_symlinks=False)
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
    parser.add_argument("--identity", required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--keychain", required=True, type=Path)
    parser.add_argument("--notary-profile", required=True)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = sign_package(
            args.input, args.output, identity=args.identity, team_id=args.team_id,
            keychain=args.keychain, notary_profile=args.notary_profile,
            evidence_directory=args.evidence_dir,
        )
    except (SigningError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Verified {args.output}: SHA-256 {summary['final_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

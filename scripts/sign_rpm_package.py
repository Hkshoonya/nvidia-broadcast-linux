#!/usr/bin/env python3
"""Sign an RPM copy and verify it against a single pinned public key.

Private key material is read from NVB_RPM_PRIVATE_KEY, never an argument.
NVB_RPM_KEY_PASSPHRASE may contain its passphrase. Neither is retained in the
output, command diagnostics, or the user's GnuPG/RPM databases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


class SigningError(RuntimeError):
    """A signing or verification requirement was not met."""


def regular_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SigningError(f"Not a regular file: {path}")
        return stream.read()


def snapshot(source: Path, destination: Path) -> str:
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as original:
        if not stat.S_ISREG(os.fstat(original.fileno()).st_mode):
            raise SigningError(f"Not a regular file: {source}")
        with destination.open("xb") as output:
            for block in iter(lambda: original.read(1024 * 1024), b""):
                output.write(block)
                digest.update(block)
    destination.chmod(0o600)
    return digest.hexdigest()


def payload_digest(path: Path, environment: dict[str, str]) -> str:
    # Bound tool execution without keeping multi-GB CUDA payloads in memory.
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as payload:
        process = subprocess.run(["rpm2cpio", str(path)], env=environment,
                                 stdout=payload, stderr=subprocess.DEVNULL,
                                 timeout=300, check=False)
        if process.returncode:
            raise SigningError("Cannot verify the RPM payload")
        payload.seek(0)
        for block in iter(lambda: payload.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(arguments: list[str], environment: dict[str, str], *, data: bytes | None = None) -> str:
    # In particular, never include GnuPG import diagnostics in an exception:
    # a malformed key or tool failure must not echo supplied secret material.
    completed = subprocess.run(arguments, input=data, env=environment,
                               capture_output=True, timeout=300, check=False)
    if completed.returncode:
        raise SigningError(f"{Path(arguments[0]).name} failed (exit {completed.returncode})")
    return completed.stdout.decode("utf-8", errors="replace")


def primary_fingerprints(listing: str) -> list[str]:
    result = []
    primary = False
    for line in listing.splitlines():
        fields = line.split(":")
        if fields[0] in {"pub", "sec"}:
            if len(fields) < 12 or fields[1] in {"r", "e", "d"}:
                raise SigningError("Signing key is revoked, expired, disabled, or malformed")
            primary = True
        elif fields[0] in {"sub", "ssb"}:
            primary = False
        elif fields[0] == "fpr" and primary:
            if len(fields) <= 9:
                raise SigningError("Missing primary key fingerprint")
            result.append(fields[9].upper())
            primary = False
    return result


def verify_rpm(package: Path, public_key: Path, fingerprint: str,
               environment: dict[str, str]) -> str:
    """Require a real signature, using private snapshots and only our key."""
    # rpmkeys includes its input filename in stdout. Never let an untrusted
    # filename inject a fake "Signature: OK" line into the verification result.
    with tempfile.TemporaryDirectory(prefix="nvb-rpm-verify-", dir="/tmp") as temporary:
        directory = Path(temporary)
        candidate = directory / "package.rpm"
        snapshot(package, candidate)
        key = directory / "public.asc"
        key.write_bytes(regular_bytes(public_key))
        gpg_home = directory / "public-gnupg"
        gpg_home.mkdir(mode=0o700)
        env = {name: value for name, value in environment.items()
               if name not in {"NVB_RPM_PRIVATE_KEY", "NVB_RPM_KEY_PASSPHRASE",
                               "GNUPGHOME", "GPG_AGENT_INFO"}}
        env.update(GNUPGHOME=str(gpg_home), LC_ALL="C", LANG="C")
        try:
            run(["gpg", "--batch", "--import"], env, data=key.read_bytes())
            secrets = run(["gpg", "--batch", "--with-colons", "--list-secret-keys"], env)
            if any(line.startswith("sec:") for line in secrets.splitlines()):
                raise SigningError("The public-key input must not contain private key material")
            listing = run(["gpg", "--batch", "--with-colons", "--fingerprint", "--list-keys"], env)
            if primary_fingerprints(listing) != [fingerprint]:
                raise SigningError("Public key does not match the pinned primary fingerprint")
            database = directory / "rpmdb"
            database.mkdir(mode=0o700)
            run(["rpmkeys", "--dbpath", str(database), "--import", str(key)], env)
            output = run(["rpmkeys", "--dbpath", str(database), "--checksig", "--verbose",
                          str(candidate)], env)
            # rpmkeys can return zero for an unsigned package with valid digests.
            # Require native verification of an actual signature as well.
            if not re.search(r"\bSignature\b[^\n]*:\s*OK\s*$", output, re.MULTILINE | re.IGNORECASE):
                raise SigningError("RPM has no verified OpenPGP signature")
            return output
        finally:
            subprocess.run(["gpgconf", "--homedir", str(gpg_home), "--kill", "gpg-agent"],
                           env=env, capture_output=True, check=False, timeout=30)


def sign(package: Path, public_key: Path, fingerprint: str, output: Path) -> dict:
    fingerprint = fingerprint.upper()
    if not re.fullmatch(r"[0-9A-F]{40}|[0-9A-F]{64}", fingerprint):
        raise SigningError("A complete OpenPGP fingerprint is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.rpm", package.name):
        raise SigningError("Expected a safe RPM package basename")
    private = os.environ.get("NVB_RPM_PRIVATE_KEY", "")
    if not private:
        raise SigningError("NVB_RPM_PRIVATE_KEY is required")
    if output.exists() or output.is_symlink():
        raise SigningError("Output directory already exists")
    # Snapshot public inputs before invoking tools; the unsigned source remains
    # unchanged, and only the verified copy becomes a distribution artifact.
    public = regular_bytes(public_key)
    if (not public.startswith(b"-----BEGIN PGP PUBLIC KEY BLOCK-----")
            or b"PRIVATE KEY BLOCK" in public or b"SECRET KEY BLOCK" in public):
        raise SigningError("Expected an armored public key without private key material")
    environment = {name: value for name, value in os.environ.items()
                   if name not in {"NVB_RPM_PRIVATE_KEY", "NVB_RPM_KEY_PASSPHRASE",
                                   "GNUPGHOME", "GPG_AGENT_INFO"}}
    environment.update(LC_ALL="C", LANG="C")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep paths interpolated into RPM macros free of shell/macro metacharacters.
    with tempfile.TemporaryDirectory(prefix="nvb-rpm-sign-", dir="/tmp") as temporary:
        scratch = Path(temporary)
        home = scratch / "gnupg"
        home.mkdir(mode=0o700)
        env = dict(environment, GNUPGHOME=str(home))
        (scratch / "signed").mkdir(mode=0o700)
        candidate = scratch / "signed" / package.name
        unsigned = scratch / "input.rpm"
        original_sha = snapshot(package, unsigned)
        snapshot(unsigned, candidate)
        key = scratch / "public.asc"
        key.write_bytes(public)
        password = scratch / "passphrase"
        password.write_text(os.environ.get("NVB_RPM_KEY_PASSPHRASE", "") + "\n")
        password.chmod(0o600)
        try:
            run(["gpg", "--batch", "--import"], env, data=private.encode())
            listing = run(["gpg", "--batch", "--with-colons", "--fingerprint",
                           "--list-secret-keys"], env)
            if primary_fingerprints(listing) != [fingerprint]:
                raise SigningError("Private key does not match the pinned primary fingerprint")
            # Force selection by full fingerprint; never select by an ambiguous
            # UID. Both macro names support RPM 4.x and current RPM 6.x.
            macros = ["--define", f"_gpg_name {fingerprint}",
                      "--define", f"_openpgp_sign_id {fingerprint}",
                      "--define", "_openpgp_sign gpg",
                      "--define", f"_gpg_path {home}",
                      "--define", "_gpg_digest_algo sha256",
                      "--define", f"_gpg_sign_cmd_extra_args --batch --pinentry-mode loopback --passphrase-file {password}"]
            run(["rpmsign", *macros, "--addsign", str(candidate)], env)
            verification = verify_rpm(candidate, key, fingerprint, environment)
            # The native signature changes the signature header only. Compare
            # the complete immutable main header and unpacked payload through
            # rpm2cpio plus package metadata queries before exposing the output.
            query = ["rpm", "-qp", "--qf", "%{HEADERIMMUTABLE:base64}"]
            if run(query + [str(unsigned)], environment) != run(query + [str(candidate)], environment):
                raise SigningError("Signing changed the immutable package header")
            payload_sha = payload_digest(unsigned, environment)
            if payload_digest(candidate, environment) != payload_sha:
                raise SigningError("Signing changed the package payload")
            output.mkdir(mode=0o755)
            signed_sha = snapshot(candidate, output / package.name)
            report = {"status": "verified", "fingerprint": fingerprint,
                      "input_sha256": original_sha,
                      "signed_sha256": signed_sha,
                      "signed_bytes": candidate.stat().st_size, "payload_sha256": payload_sha,
                      "public_key_sha256": hashlib.sha256(public).hexdigest(),
                      "signature_verification": verification}
            (output / "RPM-GPG-KEY-nvbroadcast.asc").write_bytes(public)
            (output / "rpm-signing.json").write_text(json.dumps(report, indent=2) + "\n")
            for path in output.iterdir():
                path.chmod(0o644)
            return report
        finally:
            # Terminate only the agent for this disposable keyring.
            subprocess.run(["gpgconf", "--homedir", str(home), "--kill", "gpg-agent"],
                           env=environment, capture_output=True, check=False, timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpm", required=True, type=Path)
    parser.add_argument("--public-key", required=True, type=Path)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        report = sign(arguments.rpm, arguments.public_key, arguments.fingerprint, arguments.output)
    except (SigningError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Verified RPM signature with key {report['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

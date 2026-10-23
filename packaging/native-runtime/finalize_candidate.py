#!/usr/bin/env python3
"""Bind reviewed CPU artifacts, publisher signing, and isolated qualification.

No command installs a package or runs its interpreter. Signing receives secrets
only in its dedicated job; verify and complete need public evidence alone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = "A691DCA6C031FEFD2C34F5AAFE33D0C20C9451D1"
PUBLIC_SHA = "bf9371c4fc83c41f8b1dbc3607c25ffa261b1e5de99fe799cf0a863db16cbd8a"
MAX_PACKAGE = 1024**3
SMOKE_STEPS = {
    "confirm-no-existing-package", "sentinels", "record-system-python-state",
    "install-exact-hosted-package", "installed-source-adapter-manifest-identity",
    "complete-installed-integrity", "native-package-state", "ordinary-user-launcher",
    "ordinary-user-runtime-media-gtk", "ordinary-user-generated-recording",
    "remove-native-package", "package-uninstalled", "preserve-user-system-and-stable-lock",
    "preserve-system-python-state",
}
RPM_SMOKE_STEPS = {
    "fresh-project-verification-keyring", "import-project-verification-key",
    "verify-finalized-project-signature", "import-project-installation-key",
    "system-python-recovery-interface",
}
RECOVERY_STEPS = {
    "confirm-clean-fixture", "clean-private-paths", "system-python",
    "private-verification-keyring", "import-only-project-public-key", "authenticate-signed-target",
    "signed-rpm-metadata", "authenticated-payload-bindings", "import-install-public-key",
    "sigkill-observed-first-install", "fresh-launcher-not-committed", "bootstrap-identity",
    "bootstrap-uses-system-python", "inspect-authenticated-fragments", "preserve-authenticated-fragments",
    "audit-preserved-bytes-and-marker", "import-preserved-public-key", "replay-exact-preserved-package",
    "configured-marker-and-bootstrap-cleared", "installed-payload-bindings", "native-package-state",
    "complete-installed-integrity", "launcher-after-repair",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def json_bytes(data):
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def read_json(path, maximum=16 * 1024**2):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= maximum, "invalid JSON file")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=pairs)


def relative_file(root, name):
    require(isinstance(name, str) and name and "\\" not in name, "invalid relative file")
    parts = name.split("/")
    require(not PurePosixPath(name).is_absolute() and all(p not in ("", ".", "..") for p in parts),
            "noncanonical relative file")
    target = root
    for index, part in enumerate(parts):
        target /= part
        mode = target.lstat().st_mode
        require(stat.S_ISREG(mode) if index == len(parts) - 1 else stat.S_ISDIR(mode),
                "file path contains a link or special entry")
    return target


def verified_file(root, name, record, maximum=MAX_PACKAGE):
    path = relative_file(root, name)
    require(type(record.get("bytes")) is int and 0 < record["bytes"] <= maximum,
            "invalid recorded file size")
    require(path.stat().st_size == record["bytes"] and sha(path) == digest(record["sha256"]),
            "file differs from reviewed identity")
    return path


def source_identity(source):
    require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source), "invalid source revision")
    subprocess.run(["git", "--no-replace-objects", "merge-base", "--is-ancestor", source, "HEAD"], cwd=ROOT,
                   check=True, timeout=30)


def checked_receipt(directory, expected):
    directory = directory.resolve(strict=True)
    path = relative_file(directory, "result.json")
    require(sha(path) == digest(expected), "validation receipt differs from approved job output")
    report = read_json(path)
    require(report.get("status") == "pass", "candidate did not pass independent validation")
    source_identity(report["source"])
    version = report.get("version")
    require(isinstance(version, str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+-[1-9][0-9]*\.cpu", version),
            "invalid CPU package version")
    require(set(report.get("packages", {})) == {"deb", "rpm"}, "incomplete validated packages")
    for key in ("run_id", "run_attempt", "artifact_id"):
        require(type(report.get(key)) is int and report[key] > 0, "invalid producer identity")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", report.get("artifact_digest", "")),
            "invalid approved artifact digest")
    names = {"deb": f"nvbroadcast_{version}_amd64.deb", "rpm": f"nvbroadcast-{version}.x86_64.rpm"}
    paths = {}
    for family, record in report["packages"].items():
        require(record.get("scriptlet_source_match") is True, "package scripts were not source-verified")
        path = verified_file(directory, record["relative_path"], record)
        require(path.name == names[family], "unexpected native package name")
        paths[family] = path
        digest(record["runtime_manifest_sha256"])
    require(report["packages"]["deb"]["runtime_manifest_sha256"] ==
            report["packages"]["rpm"]["runtime_manifest_sha256"], "package runtimes differ")
    digest(report["packages"]["rpm"]["recovery_helper_sha256"])
    return report, paths, directory / "result.json"


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def record_file(path):
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def sign_candidate(directory, expected, output):
    report, paths, receipt_path = checked_receipt(directory, expected)
    public = ROOT / "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc"
    require(sha(public) == PUBLIC_SHA and
            (ROOT / "packaging/keys/RPM-GPG-KEY-nvbroadcast.fingerprint").read_text().strip() == FINGERPRINT,
            "publisher identity differs")
    output.mkdir(parents=True, exist_ok=False)
    # Only trusted checkout modules run here. No candidate interpreter, hook,
    # helper or application module is executed in the signing job.
    signer = module("candidate_rpm_signer", "scripts/sign_rpm_package.py")
    signing = signer.sign(paths["rpm"], public, FINGERPRINT, output / "rpm")
    require(signing["input_sha256"] == report["packages"]["rpm"]["sha256"] and
            signing["fingerprint"] == FINGERPRINT and signing["public_key_sha256"] == PUBLIC_SHA,
            "signer output differs from reviewed inputs")
    signed = output / "rpm" / paths["rpm"].name
    require(sha(signed) == signing["signed_sha256"], "signed bytes differ from signing receipt")
    (output / "deb").mkdir()
    deb = output / "deb" / paths["deb"].name
    shutil.copyfile(paths["deb"], deb)
    require(sha(deb) == report["packages"]["deb"]["sha256"], "DEB changed during transfer")
    for name in ("RPM-GPG-KEY-nvbroadcast.asc", "rpm-signing.json"):
        (output / "rpm" / name).rename(output / name)
    shutil.copyfile(receipt_path, output / "validation.json")
    version, revision = report["version"].rsplit("-", 1)
    renderer = module("candidate_upgrade_renderer", "scripts/render_native_upgrade_helper.py")
    renderer.render_helper(ROOT / "scripts/native_package_upgrade.sh.in", deb, signed,
                           version, revision, output / "nvbroadcast-native-upgrade")
    files = {str(path.relative_to(output)): record_file(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"schema_version": 1, "status": "signed-awaiting-qualification", "version": report["version"],
                "producer": {key: report[key] for key in ("source", "run_id", "run_attempt", "artifact_id", "artifact_digest")},
                "validation_receipt_sha256": expected, "publisher_fingerprint": FINGERPRINT,
                "public_key_sha256": PUBLIC_SHA,
                "runtime_manifest_sha256": report["packages"]["rpm"]["runtime_manifest_sha256"],
                "recovery_helper_sha256": report["packages"]["rpm"]["recovery_helper_sha256"],
                "finalizer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "packages": {family: {"path": str(path.relative_to(output)), **record_file(path)}
                             for family, path in (("deb", deb), ("rpm", signed))}, "files": files}
    (output / "candidate.json").write_bytes(json_bytes(manifest))
    checksums = {**files, "candidate.json": record_file(output / "candidate.json")}
    (output / "SHA256SUMS").write_text("".join(f"{record['sha256']}  {name}\n" for name, record in sorted(checksums.items())))
    for family, path in paths.items():
        require(sha(path) == report["packages"][family]["sha256"], "original package changed")
    return {"status": manifest["status"], "manifest_sha256": sha(output / "candidate.json")}


def verify_bundle(directory, expected):
    directory = directory.resolve(strict=True)
    path = relative_file(directory, "candidate.json")
    require(sha(path) == digest(expected), "candidate manifest differs from approved job output")
    manifest = read_json(path)
    require(manifest.get("schema_version") == 1 and manifest.get("status") == "signed-awaiting-qualification",
            "invalid signed candidate state")
    version = manifest.get("version", "")
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+-[1-9][0-9]*\.cpu", version), "invalid CPU candidate")
    names = {"deb": f"deb/nvbroadcast_{version}_amd64.deb", "rpm": f"rpm/nvbroadcast-{version}.x86_64.rpm"}
    expected_files = {*names.values(), "RPM-GPG-KEY-nvbroadcast.asc", "rpm-signing.json", "validation.json", "nvbroadcast-native-upgrade"}
    require(set(manifest.get("files", {})) == expected_files and set(manifest.get("packages", {})) == {"deb", "rpm"},
            "unexpected signed bundle file set")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if not p.is_dir() or p.is_symlink()}
    require(actual == expected_files | {"candidate.json", "SHA256SUMS"}, "extra or missing bundle entries")
    for name, record in manifest["files"].items():
        verified_file(directory, name, record)
    for family, name in names.items():
        require(manifest["packages"][family] == {"path": name, **manifest["files"][name]}, "package mapping differs")
    require(manifest.get("publisher_fingerprint") == FINGERPRINT and manifest.get("public_key_sha256") == PUBLIC_SHA
            and manifest["files"]["RPM-GPG-KEY-nvbroadcast.asc"]["sha256"] == PUBLIC_SHA, "publisher differs")
    require(manifest["files"]["validation.json"]["sha256"] == manifest["validation_receipt_sha256"], "validation binding differs")
    signing = read_json(directory / "rpm-signing.json")
    validated = read_json(directory / "validation.json")
    require(validated.get("status") == "pass" and validated.get("version") == version, "validation state differs")
    producer_keys = {"source", "run_id", "run_attempt", "artifact_id", "artifact_digest"}
    require(set(manifest.get("producer", {})) == producer_keys and
            {key: validated[key] for key in producer_keys} == manifest["producer"], "producer binding differs")
    require(manifest["runtime_manifest_sha256"] == validated["packages"]["rpm"]["runtime_manifest_sha256"]
            == validated["packages"]["deb"]["runtime_manifest_sha256"] and
            manifest["recovery_helper_sha256"] == validated["packages"]["rpm"]["recovery_helper_sha256"],
            "runtime or recovery helper binding differs")
    require(signing.get("status") == "verified" and signing.get("signed_sha256") == manifest["packages"]["rpm"]["sha256"]
            and signing.get("input_sha256") == validated["packages"]["rpm"]["sha256"]
            and signing.get("fingerprint") == FINGERPRINT and signing.get("public_key_sha256") == PUBLIC_SHA,
            "signing binding differs")
    require(manifest["packages"]["deb"]["sha256"] == validated["packages"]["deb"]["sha256"], "DEB binding differs")
    checksums = {**manifest["files"], "candidate.json": record_file(path)}
    require(relative_file(directory, "SHA256SUMS").read_text() == "".join(
        f"{r['sha256']}  {n}\n" for n, r in sorted(checksums.items())), "checksum file differs")
    return manifest


def check_execution(path, manifest, family, recovery=False):
    report = read_json(path)
    require(report.get("status") == "pass", "qualification did not pass")
    require(report.get("source_revision" if recovery else "source") == manifest["producer"]["source"],
            "qualification source differs")
    require(report.get("package_sha256" if recovery else "artifact_sha256") == manifest["packages"][family]["sha256"],
            "qualification package differs")
    require(report.get("runtime_manifest_sha256") == manifest["runtime_manifest_sha256"], "qualified runtime differs")
    require(report.get("container_removed") is True, "qualification container remains")
    for field in (("original_package_unchanged", "snapshots_unchanged") if recovery else ("input_unchanged", "snapshot_unchanged")):
        require(report.get(field) is True, "qualification package changed")
    steps = report.get("steps")
    require(isinstance(steps, list) and steps, "missing execution steps")
    names = set()
    rejected = {"confirm-clean-fixture"} if recovery else {"confirm-no-existing-package", "package-uninstalled"}
    for step in steps:
        require(step["name"] not in names, "duplicate qualification step")
        names.add(step["name"])
        require(type(step.get("expected_failure")) is bool and type(step.get("exit_code")) is int
                and step["expected_failure"] == (step["name"] in rejected)
                and step["exit_code"] == (1 if step["name"] in rejected else 0), "unexpected execution exit code")
        log = relative_file(path.parent, step["log"])
        require(log.stat().st_size <= 8 * 1024**2 and sha(log) == digest(step.get("log_sha256", step.get("sha256"))),
                "execution log differs")
    expected_steps = RECOVERY_STEPS if recovery else SMOKE_STEPS | (RPM_SMOKE_STEPS if family == "rpm" else set())
    require(names == expected_steps, "incomplete or unexpected qualification steps")
    if recovery:
        require(report.get("authenticated") is True and report.get("helper_sha256") == manifest["recovery_helper_sha256"],
                "recovery authentication differs")
        interruption = report.get("interruption", {})
        require(interruption.get("signal") == "SIGKILL" and interruption.get("fresh_install") is True
                and interruption.get("manager_exit") == 9
                and interruption.get("observed_bytes", 0) > 1024**2, "actual fresh-install interruption missing")
        require(sha(relative_file(path.parent, "interrupted-manager.log")) ==
                digest(interruption["manager_log_sha256"]), "interrupted manager log differs")
        preservation_path = relative_file(path.parent, "preservation.json")
        require(sha(preservation_path) == digest(report["preservation_sha256"]), "preservation receipt differs")
        preservation = read_json(preservation_path)
        require(preservation.get("package_sha256") == manifest["packages"]["rpm"]["sha256"] and
                preservation.get("version") == manifest["version"] and
                preservation.get("fingerprint") == FINGERPRINT, "preserved package identity differs")
    else:
        require(report.get("family") == family and report.get("version") == manifest["version"], "qualified package identity differs")
        require(report.get("runtime_probe", {}).get("status") == "pass" and report.get("recording", {}).get("status") == "pass",
                "runtime or recording failed")
        if family == "rpm":
            require(report.get("rpm_authentication", {}).get("status") == "verified"
                    and report["rpm_authentication"].get("publisher_fingerprint") == FINGERPRINT,
                    "final RPM was not authenticated")
    return {"sha256": sha(path), "steps": len(steps), "image": report["image"], "status": "pass"}


def evidence_files(directory, family):
    """Retain only bounded receipts, referenced logs, and fixture inventory."""
    allowed = {"smoke/result.json", "fixture.json", "fixture-packages.txt"}
    if family == "rpm":
        allowed |= {"recovery/result.json", "recovery/preservation.json", "recovery/interrupted-manager.log"}
    for part in (("smoke", "recovery") if family == "rpm" else ("smoke",)):
        report = read_json(relative_file(directory, part + "/result.json"))
        allowed.update(part + "/" + step["log"] for step in report["steps"])
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if not p.is_dir() or p.is_symlink()}
    require(actual == allowed, "unexpected qualification evidence file set")
    require(len(actual) <= 128, "too many evidence files")
    total = 0
    for name in actual:
        path = relative_file(directory, name)
        size = path.stat().st_size
        require(size <= 16 * 1024**2, "oversize evidence file")
        total += size
    require(total <= 128 * 1024**2, "oversize evidence bundle")
    fixture = read_json(directory / "fixture.json")
    require(fixture.get("family") == family and re.fullmatch(r"sha256:[0-9a-f]{64}", fixture.get("image", "")),
            "invalid fixture identity")
    require(sha(directory / "fixture-packages.txt") == digest(fixture["inventory_sha256"]), "fixture inventory differs")
    return fixture


def complete(directory, expected, deb_evidence, rpm_evidence, output):
    manifest = verify_bundle(directory, expected)
    results = {"deb": check_execution(deb_evidence / "smoke/result.json", manifest, "deb"),
               "rpm": check_execution(rpm_evidence / "smoke/result.json", manifest, "rpm"),
               "recovery": check_execution(rpm_evidence / "recovery/result.json", manifest, "rpm", recovery=True)}
    fixtures = {"deb": evidence_files(deb_evidence, "deb"), "rpm": evidence_files(rpm_evidence, "rpm")}
    for family, fixture in fixtures.items():
        require(fixture["image"] == results[family]["image"], "qualified fixture image differs")
    require(fixtures["rpm"]["image"] == results["recovery"]["image"], "recovery fixture image differs")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(directory, output / "packages")
    shutil.copytree(deb_evidence, output / "evidence/deb")
    shutil.copytree(rpm_evidence, output / "evidence/rpm")
    report = {"schema_version": 1, "status": "qualified-development-candidate", "candidate_manifest_sha256": expected,
              "producer": manifest["producer"], "packages": manifest["packages"], "qualification": results, "fixtures": fixtures,
              "scope": "Disconnected Ubuntu/Fedora x86_64 CPU containers; no physical-device, CUDA, clean-VM or release promotion claim"}
    (output / "qualification.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sign = sub.add_parser("sign")
    sign.add_argument("--validated", type=Path, required=True)
    sign.add_argument("--expected-receipt", required=True)
    sign.add_argument("--output", type=Path, required=True)
    for name in ("verify", "complete"):
        command = sub.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--expected-manifest", required=True)
        if name == "complete":
            command.add_argument("--deb-evidence", type=Path, required=True)
            command.add_argument("--rpm-evidence", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sign":
        result = sign_candidate(args.validated, args.expected_receipt, args.output)
    elif args.command == "verify":
        result = verify_bundle(args.bundle, args.expected_manifest)
    else:
        result = complete(args.bundle, args.expected_manifest, args.deb_evidence, args.rpm_evidence, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

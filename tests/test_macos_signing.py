"""Exercise package-signing gates without calling Apple tools or services."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock


_SPEC = importlib.util.spec_from_file_location(
    "sign_macos_package", Path(__file__).parents[1] / "scripts/sign_macos_package.py"
)
signing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(signing)

TEAM_ID = "ABC123DE45"
OTHER_TEAM = "ZZZ123DE45"
IDENTITY = "0123456789ABCDEF0123456789ABCDEF01234567"
SUBMISSION_ID = "7c8aaf20-6e8a-4bbf-aaf1-4a6af19c5633"
OTHER_SUBMISSION = "6074645d-2fe7-49f2-86bc-dc7b2a05205d"


class _MacTools:
    """Model native command results and real filesystem changes for each gate."""

    def __init__(self):
        self.calls = []
        self.signer = f"Developer ID Installer: Example Developer ({TEAM_ID})"
        self.final_signer = None
        self.trusted = True
        self.timestamp = True
        self.status = "Accepted"
        self.submission_id = SUBMISSION_ID
        self.submit_returncode = 0
        self.submit_json = None
        self.log_status = None
        self.log_id = None
        self.log_hash = None
        self.omit_log = False
        self.fail = None
        self.timeout = None
        self.altered_phase = None
        self.alteration = None
        self.stapled = False
        self.assessment_disabled = False
        self.before_assess = None
        self.upload_hash = None

    def signature(self):
        status = (
            "signed by a developer certificate issued by Apple for distribution"
            if self.trusted else "signed by an untrusted certificate"
        )
        timestamp = "  Signed with a trusted timestamp on: 2026-10-07 12:00:00 +0000\n" if self.timestamp else ""
        signer = self.final_signer if self.stapled and self.final_signer else self.signer
        return (
            f'Package "example.pkg":\n  Status: {status}\n{timestamp}'
            f"  Certificate Chain:\n   1. {signer}\n"
            "   2. Developer ID Certification Authority\n   3. Apple Root CA\n"
        )

    def expand(self, output, phase):
        output.mkdir(mode=0o700)
        contents = {
            "Distribution": (b"<installer-gui-script/>\n", 0o644),
            "component.pkg/PackageInfo": (b"package metadata", 0o644),
            "component.pkg/Bom": (b"bom metadata", 0o644),
            "component.pkg/Payload/opt/nvbroadcast/app.py": (b"print('application')\n", 0o644),
            "component.pkg/Payload/usr/local/bin/nvbroadcast": (b"#!/bin/bash\nexit 0\n", 0o755),
            "component.pkg/Scripts/postinstall": (b"#!/bin/bash\nexit 0\n", 0o755),
        }
        for relative, (content, mode) in contents.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode)
        for path in output.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        link = output / "component.pkg/Payload/opt/nvbroadcast/app-link"
        link.symlink_to("app.py")
        if phase != self.altered_phase:
            return
        target = output / "component.pkg/Payload/opt/nvbroadcast/app.py"
        if self.alteration == "bytes":
            target.write_bytes(b"different application")
        elif self.alteration == "mode":
            target.chmod(0o755)
        elif self.alteration == "symlink":
            link.unlink()
            link.symlink_to("another.py")
        elif self.alteration == "missing":
            target.unlink()
        elif self.alteration == "extra":
            (target.parent / "extra.py").write_bytes(b"unexpected")
        elif self.alteration == "special-file":
            target.unlink()
            os.mkfifo(target)

    def __call__(self, arguments, **kwargs):
        self.calls.append((list(arguments), kwargs))
        executable = Path(arguments[0]).name
        if executable == "productsign":
            step = "productsign"
        elif executable == "pkgutil":
            step = "expand" if arguments[1] == "--expand-full" else "signature"
        elif executable == "spctl":
            step = "gatekeeper"
        else:
            step = "-".join(arguments[1:3])
        if self.timeout == step:
            raise subprocess.TimeoutExpired(arguments, kwargs["timeout"], output=b"partial output", stderr=b"timed out")
        if self.fail == step:
            return subprocess.CompletedProcess(arguments, 1, "", "controlled failure")
        output = ""
        errors = ""
        returncode = 0
        if step == "productsign":
            Path(arguments[-1]).write_bytes(Path(arguments[-2]).read_bytes() + b".signed")
        elif step == "expand":
            phase = "input" if Path(arguments[2]).name == "unsigned.pkg" else ("final" if self.stapled else "signed")
            self.expand(Path(arguments[3]), phase)
        elif step == "signature":
            output = self.signature()
        elif step == "notarytool-submit":
            self.upload_hash = hashlib.sha256(Path(arguments[3]).read_bytes()).hexdigest()
            output = self.submit_json if self.submit_json is not None else json.dumps({
                "id": self.submission_id, "status": self.status,
            })
            returncode = self.submit_returncode
        elif step == "notarytool-log":
            if not self.omit_log:
                Path(arguments[-1]).write_text(json.dumps({
                    "jobId": self.log_id or self.submission_id,
                    "status": self.log_status or self.status,
                    "sha256": self.log_hash or self.upload_hash,
                    "issues": [{"severity": "warning", "message": "retained warning"}],
                }))
        elif step == "stapler-staple":
            package = Path(arguments[3])
            package.write_bytes(package.read_bytes() + b".ticket")
            self.stapled = True
        elif step == "gatekeeper":
            if self.before_assess:
                self.before_assess()
            errors = (
                "assessments disabled\n" if self.assessment_disabled else
                f"{arguments[-1]}: accepted\nsource=Notarized Developer ID\norigin={self.signer}\n"
            )
        return subprocess.CompletedProcess(arguments, returncode, output, errors)


class MacOSSigningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="macOS signing tests ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "unsigned.pkg"
        self.source.write_bytes(b"reviewed unsigned installer")
        self.source.chmod(0o640)
        self.original = self.source.read_bytes()
        self.output = self.root / "verified.pkg"
        self.evidence = self.root / "evidence"
        self.keychain = self.root / "release.keychain-db"
        self.keychain.write_bytes(b"placeholder, never read by orchestration")
        self.tools = _MacTools()
        run_patch = mock.patch.object(signing.subprocess, "run", side_effect=self.tools)
        self.run = run_patch.start()
        self.addCleanup(run_patch.stop)
        platform_patch = mock.patch.object(signing.platform, "system", return_value="Darwin")
        platform_patch.start()
        self.addCleanup(platform_patch.stop)

    def sign(self, **overrides):
        arguments = dict(
            source=self.source, output=self.output, identity=IDENTITY, team_id=TEAM_ID,
            keychain=self.keychain, notary_profile="release notary profile", evidence_directory=self.evidence,
        )
        arguments.update(overrides)
        return signing.sign_package(**arguments)

    def assert_failed(self):
        with self.assertRaises((signing.SigningError, OSError)):
            self.sign()
        self.assertFalse(os.path.lexists(self.output))
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(stat.S_IMODE(self.source.stat().st_mode), 0o640)
        summary = json.loads((self.evidence / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["original_unchanged"])
        manifest = json.loads((self.evidence / "evidence-manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNone(manifest["artifact"])
        self.assertFalse(list(self.root.glob(".nvbroadcast-signing-*")))

    def test_success_checks_every_gate_and_only_hashes_the_stapled_artifact(self):
        summary = self.sign()
        self.assertEqual(summary["status"], "verified")
        self.assertEqual(summary["submission_id"], SUBMISSION_ID)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.output.read_bytes(), self.original + b".signed.ticket")
        self.assertEqual(summary["final_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertNotEqual(summary["signed_upload_sha256"], summary["final_sha256"])
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o644)
        self.assertEqual(stat.S_IMODE(self.evidence.stat().st_mode), 0o700)
        self.assertFalse(list(self.root.glob(".nvbroadcast-signing-*")))
        calls = [arguments for arguments, _ in self.tools.calls]
        self.assertEqual(len(calls), 11)
        signing_call = next(arguments for arguments in calls if Path(arguments[0]).name == "productsign")
        self.assertIn("--timestamp", signing_call)
        self.assertEqual(signing_call[signing_call.index("--keychain") + 1], str(self.keychain))
        notary_call = next(arguments for arguments in calls if arguments[1:3] == ["notarytool", "submit"])
        self.assertEqual(notary_call[notary_call.index("--timeout") + 1], "30m")
        self.assertIn("--wait", notary_call)
        self.assertEqual(notary_call[notary_call.index("--keychain-profile") + 1], "release notary profile")
        self.assertEqual(notary_call[notary_call.index("--keychain") + 1], str(self.keychain))
        for _, kwargs in self.tools.calls:
            self.assertNotIn("shell", kwargs)
            self.assertEqual(kwargs["env"]["LC_ALL"], "C")
        inventories = [json.loads((self.evidence / f"expand-{phase}-inventory.json").read_text()) for phase in ("input", "signed", "final")]
        self.assertEqual(inventories[0], inventories[1])
        self.assertEqual(inventories[0], inventories[2])
        manifest = json.loads((self.evidence / "evidence-manifest.json").read_text())
        self.assertEqual(manifest["artifact"]["sha256"], summary["final_sha256"])
        for member in manifest["evidence"]:
            self.assertEqual(member["sha256"], hashlib.sha256((self.evidence / member["name"]).read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.evidence / "notarization-log.json").read_text())["issues"][0]["severity"], "warning")

    def test_accepts_the_expected_installer_identity_label(self):
        self.assertEqual(self.sign(identity=self.tools.signer)["status"], "verified")

    def test_rejects_wrong_team_first_signer(self):
        self.tools.signer = f"Developer ID Installer: Example Developer ({OTHER_TEAM})"
        self.assert_failed()
        self.assertFalse(any(arguments[1:3] == ["notarytool", "submit"] for arguments, _ in self.tools.calls))

    def test_rejects_application_certificate_even_when_its_team_matches(self):
        self.tools.signer = f"Developer ID Application: Example Developer ({TEAM_ID})"
        self.assert_failed()

    def test_rejects_untrusted_signature(self):
        self.tools.trusted = False
        self.assert_failed()

    def test_rejects_missing_trusted_timestamp(self):
        self.tools.timestamp = False
        self.assert_failed()

    def test_rejects_signing_failure(self):
        self.tools.fail = "productsign"
        self.assert_failed()

    def test_rejects_invalid_notarization_and_keeps_the_service_log(self):
        self.tools.status = "Invalid"
        self.assert_failed()
        self.assertTrue((self.evidence / "notarization-log.json").is_file())
        self.assertFalse(self.tools.stapled)

    def test_rejects_in_progress_notarization(self):
        self.tools.status = "In Progress"
        self.assert_failed()
        self.assertFalse(self.tools.stapled)

    def test_nonzero_notary_exit_does_not_accept_an_accepted_json_response(self):
        self.tools.submit_returncode = 1
        self.assert_failed()
        self.assertFalse(self.tools.stapled)

    def test_rejects_malformed_notary_json(self):
        self.tools.submit_json = "not a JSON response"
        self.assert_failed()

    def test_rejects_non_object_notary_json(self):
        self.tools.submit_json = "[]"
        self.assert_failed()

    def test_rejects_invalid_submission_identifier(self):
        self.tools.submission_id = "not-a-uuid"
        self.assert_failed()
        self.assertFalse(any(arguments[1:3] == ["notarytool", "log"] for arguments, _ in self.tools.calls))

    def test_rejects_missing_submission_identifier(self):
        self.tools.submit_json = json.dumps({"status": "Accepted"})
        self.assert_failed()

    def test_rejects_log_for_another_submission(self):
        self.tools.log_id = OTHER_SUBMISSION
        self.assert_failed()

    def test_rejects_log_for_different_package_bytes(self):
        self.tools.log_hash = "0" * 64
        self.assert_failed()

    def test_rejects_log_that_does_not_confirm_acceptance(self):
        self.tools.log_status = "Invalid"
        self.assert_failed()

    def test_rejects_missing_notarization_log(self):
        self.tools.omit_log = True
        self.assert_failed()

    def test_rejects_failed_notarization_log_download(self):
        self.tools.fail = "notarytool-log"
        self.assert_failed()

    def test_rejects_stapling_failure(self):
        self.tools.fail = "stapler-staple"
        self.assert_failed()

    def test_rejects_stapled_ticket_validation_failure(self):
        self.tools.fail = "stapler-validate"
        self.assert_failed()

    def test_rejects_gatekeeper_failure(self):
        self.tools.fail = "gatekeeper"
        self.assert_failed()

    def test_rejects_disabled_gatekeeper_even_with_a_zero_exit_code(self):
        self.tools.assessment_disabled = True
        self.assert_failed()

    def test_rechecks_signer_after_stapling(self):
        self.tools.final_signer = f"Developer ID Application: Example Developer ({TEAM_ID})"
        self.assert_failed()

    def test_compares_every_expanded_member_before_and_after_notarization(self):
        for phase in ("signed", "final"):
            for alteration in ("bytes", "mode", "symlink", "missing", "extra", "special-file"):
                with self.subTest(phase=phase, alteration=alteration):
                    self.tools.altered_phase = phase
                    self.tools.alteration = alteration
                    self.evidence = self.root / f"evidence-{phase}-{alteration}"
                    self.tools.stapled = False
                    self.assert_failed()

    def test_command_timeout_preserves_partial_evidence(self):
        self.tools.timeout = "notarytool-submit"
        self.assert_failed()
        self.assertTrue(any("partial output" in path.read_text() for path in self.evidence.glob("*.stdout.txt")))

    def test_original_changed_during_signing_prevents_final_artifact(self):
        self.tools.before_assess = lambda: self.source.write_bytes(b"changed input")
        with self.assertRaisesRegex(signing.SigningError, "Original input package changed"):
            self.sign()
        self.assertFalse(self.output.exists())
        self.assertFalse(json.loads((self.evidence / "summary.json").read_text())["original_unchanged"])

    def test_racing_output_creation_is_preserved_and_rejected(self):
        self.tools.before_assess = lambda: self.output.write_bytes(b"another process's file")
        with self.assertRaises(FileExistsError):
            self.sign()
        self.assertEqual(self.output.read_bytes(), b"another process's file")
        self.assertEqual(json.loads((self.evidence / "summary.json").read_text())["status"], "failed")
        self.assertIsNone(json.loads((self.evidence / "evidence-manifest.json").read_text())["artifact"])

    def test_refuses_existing_output_before_any_apple_command(self):
        self.output.write_bytes(b"existing artifact")
        with self.assertRaises(signing.SigningError):
            self.sign()
        self.assertEqual(self.output.read_bytes(), b"existing artifact")
        self.run.assert_not_called()

    def test_refuses_dangling_output_symlink_without_touching_its_target(self):
        target = self.root / "must-not-create.pkg"
        self.output.symlink_to(target)
        with self.assertRaises(signing.SigningError):
            self.sign()
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())
        self.run.assert_not_called()

    def test_refuses_input_symlink(self):
        link = self.root / "linked.pkg"
        link.symlink_to(self.source)
        with self.assertRaises(signing.SigningError):
            self.sign(source=link)
        self.run.assert_not_called()

    def test_refuses_keychain_symlink_without_reading_it(self):
        link = self.root / "linked.keychain-db"
        link.symlink_to(self.keychain)
        with self.assertRaises(signing.SigningError):
            self.sign(keychain=link)
        self.run.assert_not_called()

    def test_refuses_existing_evidence_directory(self):
        self.evidence.mkdir()
        sentinel = self.evidence / "keep.txt"
        sentinel.write_text("existing evidence")
        with self.assertRaises(signing.SigningError):
            self.sign()
        self.assertEqual(sentinel.read_text(), "existing evidence")
        self.run.assert_not_called()

    def test_rejects_wrong_identity_type_before_commands(self):
        with self.assertRaises(signing.SigningError):
            self.sign(identity=f"Developer ID Application: Example ({TEAM_ID})")
        self.run.assert_not_called()

    def test_rejects_wrong_team_identity_label_before_commands(self):
        with self.assertRaises(signing.SigningError):
            self.sign(identity=f"Developer ID Installer: Example ({OTHER_TEAM})")
        self.run.assert_not_called()

    def test_rejects_invalid_team_id_before_commands(self):
        with self.assertRaises(signing.SigningError):
            self.sign(team_id="invalid")
        self.run.assert_not_called()

    def test_rejects_non_macos_before_commands(self):
        with mock.patch.object(signing.platform, "system", return_value="Linux"):
            with self.assertRaises(signing.SigningError):
                self.sign()
        self.run.assert_not_called()

    def test_metacharacters_in_paths_remain_literal_argv_members(self):
        unusual = self.root / "input $(touch nope); with spaces.pkg"
        self.source.rename(unusual)
        summary = self.sign(source=unusual)
        self.assertEqual(summary["source"], str(unusual))
        self.assertFalse((self.root / "nope").exists())


if __name__ == "__main__":
    unittest.main()

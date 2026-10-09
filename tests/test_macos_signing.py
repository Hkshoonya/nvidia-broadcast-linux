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
        self.submit_stderr = ""
        self.info_status = None
        self.info_id = None
        self.info_returncode = 0
        self.info_json = None
        self.log_status = None
        self.log_id = None
        self.log_hash = None
        self.omit_log = False
        self.omit_log_hash = False
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
            errors = self.submit_stderr
        elif step == "notarytool-info":
            output = self.info_json if self.info_json is not None else json.dumps({
                "id": self.info_id if self.info_id is not None else self.submission_id,
                "status": self.info_status if self.info_status is not None else self.status,
            })
            returncode = self.info_returncode
        elif step == "notarytool-log":
            if not self.omit_log:
                log = {
                    "jobId": self.log_id or self.submission_id,
                    "status": self.log_status or self.status,
                    "sha256": self.log_hash or self.upload_hash,
                    "issues": [{"severity": "warning", "message": "retained warning"}],
                }
                if self.omit_log_hash:
                    del log["sha256"]
                Path(arguments[-1]).write_text(json.dumps(log))
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

    def prepare_checkpoint(self):
        self.checkpoint = self.root / "checkpoint"
        self.tools.submit_returncode = 124
        self.tools.submit_json = ""
        self.tools.submit_stderr = json.dumps({
            "id": SUBMISSION_ID,
            "message": "Timeout of 1800 second(s) was reached before processing completed.",
        })
        with self.assertRaisesRegex(signing.SigningError, "wait timed out"):
            self.sign(checkpoint_directory=self.checkpoint)
        self.source_pin = hashlib.sha256(self.original).hexdigest()
        self.signed_pin = hashlib.sha256(self.original + b".signed").hexdigest()
        self.tools.submit_returncode = 0
        self.tools.submit_json = None
        self.tools.submit_stderr = ""
        self.tools.calls.clear()
        self.evidence = self.root / "resume-evidence"

    def resume(self, **overrides):
        arguments = dict(
            source=self.source, output=self.output, checkpoint_directory=self.checkpoint,
            source_sha256=self.source_pin, signed_sha256=self.signed_pin,
            team_id=TEAM_ID, keychain=self.keychain, notary_profile="release notary profile",
            evidence_directory=self.evidence,
        )
        arguments.update(overrides)
        return signing.resume_package(**arguments)

    def assert_no_sign_or_submit(self):
        for arguments, _ in self.tools.calls:
            self.assertNotEqual(Path(arguments[0]).name, "productsign")
            self.assertNotEqual(arguments[1:3], ["notarytool", "submit"])

    def assert_resume_failed(self):
        retained = (self.checkpoint / "signed-upload.pkg").read_bytes()
        checkpoint = (self.checkpoint / "checkpoint.json").read_bytes()
        with self.assertRaises((signing.SigningError, OSError)):
            self.resume()
        self.assertFalse(os.path.lexists(self.output))
        self.assertEqual((self.checkpoint / "signed-upload.pkg").read_bytes(), retained)
        self.assertEqual((self.checkpoint / "checkpoint.json").read_bytes(), checkpoint)
        self.assertFalse(list(self.root.glob(".nvbroadcast-signing-*")))
        self.assert_no_sign_or_submit()

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

    def test_empty_stdout_timeout_retains_stderr_uuid_without_inventing_apple_status(self):
        self.tools.submit_returncode = 124
        self.tools.submit_json = ""
        self.tools.submit_stderr = json.dumps({
            "id": SUBMISSION_ID,
            "message": "Timeout of 1800 second(s) was reached before processing completed.",
        })
        self.assert_failed()
        summary = json.loads((self.evidence / "summary.json").read_text())
        self.assertEqual(summary["submission_id"], SUBMISSION_ID)
        self.assertIsNone(summary["notarization_status"])
        self.assertEqual(summary["notarization_wait_status"], "timed_out")
        self.assertIn("wait timed out", summary["error"])
        self.assertFalse(self.tools.stapled)

    def test_checkpoint_retains_exact_timeout_upload_with_private_permissions(self):
        self.prepare_checkpoint()
        checkpoint = json.loads((self.checkpoint / "checkpoint.json").read_text())
        self.assertEqual((self.checkpoint / "signed-upload.pkg").read_bytes(), self.original + b".signed")
        self.assertEqual(checkpoint["source_sha256"], self.source_pin)
        self.assertEqual(checkpoint["signed_upload_sha256"], self.signed_pin)
        self.assertEqual(checkpoint["submission_id"], SUBMISSION_ID)
        self.assertEqual(checkpoint["state"], "pending")
        self.assertIsNone(checkpoint["notarization_status"])
        self.assertEqual(checkpoint["notarization_wait_status"], "timed_out")
        self.assertEqual(stat.S_IMODE(self.checkpoint.stat().st_mode), 0o700)
        for name in ("checkpoint.json", "signed-upload.pkg"):
            self.assertEqual(stat.S_IMODE((self.checkpoint / name).stat().st_mode), 0o600)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".nvbroadcast-signing-*")))

    def test_outer_timeout_retains_upload_without_guessing_submission_id(self):
        checkpoint = self.root / "checkpoint"
        self.tools.timeout = "notarytool-submit"
        with self.assertRaises(signing.SigningError):
            self.sign(checkpoint_directory=checkpoint)
        metadata = json.loads((checkpoint / "checkpoint.json").read_text())
        self.assertIsNone(metadata["submission_id"])
        self.assertEqual(metadata["notarization_wait_status"], "timed_out")
        self.assertEqual((checkpoint / "signed-upload.pkg").read_bytes(), self.original + b".signed")
        self.assertFalse(self.output.exists())
        self.assertTrue(any("partial output" in path.read_text() for path in self.evidence.glob("*.stdout.txt")))

    def test_outer_timeout_can_record_a_complete_stderr_uuid_but_cannot_release(self):
        checkpoint = self.root / "checkpoint"

        def interrupted(arguments, **kwargs):
            if arguments[1:3] == ["notarytool", "submit"]:
                raise subprocess.TimeoutExpired(arguments, kwargs["timeout"], output=b"", stderr=json.dumps({
                    "id": SUBMISSION_ID, "message": "Processing has not completed.",
                }).encode())
            return self.tools(arguments, **kwargs)

        self.run.side_effect = interrupted
        with self.assertRaises(signing.SigningError):
            self.sign(checkpoint_directory=checkpoint)
        metadata = json.loads((checkpoint / "checkpoint.json").read_text())
        self.assertEqual(metadata["submission_id"], SUBMISSION_ID)
        self.assertEqual(metadata["notarization_wait_status"], "timed_out")
        self.assertIsNone(metadata["notarization_status"])
        self.assertFalse(self.output.exists())

    def test_conflicting_timeout_identifiers_are_not_guessed(self):
        self.tools.submit_returncode = 124
        self.tools.submit_json = json.dumps({"id": SUBMISSION_ID, "status": "In Progress"})
        self.tools.submit_stderr = json.dumps({"id": OTHER_SUBMISSION})
        self.assert_failed()
        self.assertIsNone(json.loads((self.evidence / "summary.json").read_text())["submission_id"])

    def test_checkpoint_is_not_created_before_signature_and_payload_gates(self):
        checkpoint = self.root / "checkpoint"
        self.tools.trusted = False
        with self.assertRaises(signing.SigningError):
            self.sign(checkpoint_directory=checkpoint)
        self.assertFalse(checkpoint.exists())
        self.assertFalse(self.output.exists())

    def test_resume_accepted_submission_checks_all_gates_without_signing_or_uploading(self):
        self.prepare_checkpoint()
        retained = (self.checkpoint / "signed-upload.pkg").read_bytes()
        metadata = (self.checkpoint / "checkpoint.json").read_bytes()
        summary = self.resume()
        self.assertTrue(summary["resumed"])
        self.assertEqual(summary["status"], "verified")
        self.assertEqual(summary["submission_id"], SUBMISSION_ID)
        self.assertEqual(summary["signed_upload_sha256"], self.signed_pin)
        self.assertEqual(summary["final_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertNotEqual(summary["signed_upload_sha256"], summary["final_sha256"])
        self.assertEqual(self.output.read_bytes(), self.original + b".signed.ticket")
        self.assertEqual((self.checkpoint / "signed-upload.pkg").read_bytes(), retained)
        self.assertEqual((self.checkpoint / "checkpoint.json").read_bytes(), metadata)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assert_no_sign_or_submit()
        calls = [arguments for arguments, _ in self.tools.calls]
        info = next(arguments for arguments in calls if arguments[1:3] == ["notarytool", "info"])
        self.assertEqual(info[3], SUBMISSION_ID)
        self.assertEqual(info[info.index("--keychain") + 1], str(self.keychain))
        for phase in ("signed", "final"):
            self.assertEqual(
                json.loads((self.evidence / "expand-input-inventory.json").read_text()),
                json.loads((self.evidence / f"expand-{phase}-inventory.json").read_text()),
            )
        manifest = json.loads((self.evidence / "evidence-manifest.json").read_text())
        self.assertEqual(manifest["artifact"]["sha256"], summary["final_sha256"])
        for member in manifest["evidence"]:
            self.assertEqual(member["sha256"], hashlib.sha256((self.evidence / member["name"]).read_bytes()).hexdigest())

    def test_resume_refuses_unaccepted_missing_or_mismatched_info_and_logs(self):
        self.prepare_checkpoint()
        scenarios = (
            {"info_status": "In Progress"}, {"info_status": "Invalid"},
            {"info_returncode": 1}, {"info_id": OTHER_SUBMISSION},
            {"info_json": "[]"}, {"info_json": "not JSON"},
            {"info_json": json.dumps({"status": "Accepted"})},
            {"info_json": json.dumps({"id": "invalid", "status": "Accepted"})},
            {"fail": "notarytool-info"}, {"fail": "notarytool-log"},
            {"log_id": OTHER_SUBMISSION}, {"log_status": "Invalid"},
            {"log_hash": "0" * 64}, {"omit_log": True}, {"omit_log_hash": True},
        )
        for index, settings in enumerate(scenarios):
            with self.subTest(settings=settings):
                self.tools = _MacTools()
                self.tools.upload_hash = self.signed_pin
                self.run.side_effect = self.tools
                for name, value in settings.items():
                    setattr(self.tools, name, value)
                self.evidence = self.root / f"resume-evidence-{index}"
                self.assert_resume_failed()
                self.assertFalse(self.tools.stapled)

    def test_resume_preserves_all_native_verification_failure_gates(self):
        self.prepare_checkpoint()
        scenarios = (
            {"trusted": False}, {"timestamp": False},
            {"signer": f"Developer ID Installer: Other Developer ({TEAM_ID})"},
            {"signer": f"Developer ID Installer: Example ({OTHER_TEAM})"},
            {"signer": f"Developer ID Application: Example ({TEAM_ID})"},
            {"fail": "stapler-staple"}, {"fail": "stapler-validate"},
            {"fail": "gatekeeper"}, {"assessment_disabled": True},
            {"final_signer": f"Developer ID Application: Example ({TEAM_ID})"},
        ) + tuple(
            {"altered_phase": phase, "alteration": alteration}
            for phase in ("signed", "final")
            for alteration in ("bytes", "mode", "symlink", "missing", "extra", "special-file")
        )
        for index, settings in enumerate(scenarios):
            with self.subTest(settings=settings):
                self.tools = _MacTools()
                self.tools.upload_hash = self.signed_pin
                self.run.side_effect = self.tools
                for name, value in settings.items():
                    setattr(self.tools, name, value)
                self.evidence = self.root / f"native-resume-evidence-{index}"
                self.assert_resume_failed()

    def test_resume_requires_independent_source_and_signed_hash_pins_before_native_calls(self):
        self.prepare_checkpoint()
        for overrides in ({"source_sha256": "0" * 64}, {"signed_sha256": "0" * 64},
                          {"source_sha256": "invalid"}, {"team_id": OTHER_TEAM}):
            with self.subTest(overrides=overrides), self.assertRaises(signing.SigningError):
                self.resume(**overrides)
        self.assertFalse(self.tools.calls)

    def test_resume_rejects_changed_source_or_upload_before_native_calls(self):
        self.prepare_checkpoint()
        for path in (self.source, self.checkpoint / "signed-upload.pkg"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"x" * len(original))
                with self.assertRaises(signing.SigningError):
                    self.resume()
                self.assertFalse(self.tools.calls)
                path.write_bytes(original)

    def test_resume_rejects_tampered_metadata_even_if_its_hashes_are_changed_together(self):
        self.prepare_checkpoint()
        path = self.checkpoint / "checkpoint.json"
        checkpoint = json.loads(path.read_text())
        checkpoint["signed_upload_sha256"] = "0" * 64
        path.write_text(json.dumps(checkpoint))
        (self.checkpoint / "signed-upload.pkg").write_bytes(b"tampered archive")
        with self.assertRaises(signing.SigningError):
            self.resume()
        self.assertFalse(self.tools.calls)

    def test_resume_rejects_invalid_checkpoint_schema_uuid_state_and_size_before_commands(self):
        self.prepare_checkpoint()
        path = self.checkpoint / "checkpoint.json"
        original = json.loads(path.read_text())
        for key, value in (("schema_version", True), ("schema_version", 2),
                           ("submission_id", None), ("submission_id", "invalid"),
                           ("state", "rejected"), ("source_bytes", 1),
                           ("signed_upload_bytes", True), ("signer", "wrong signer")):
            with self.subTest(key=key, value=value):
                path.write_text(json.dumps(dict(original, **{key: value})))
                with self.assertRaises(signing.SigningError):
                    self.resume()
                self.assertFalse(self.tools.calls)

    def test_resume_refuses_symlinked_checkpoint_files_and_existing_output_before_commands(self):
        self.prepare_checkpoint()
        for name in ("checkpoint.json", "signed-upload.pkg"):
            path = self.checkpoint / name
            target = self.root / f"retained-{name}"
            path.rename(target)
            path.symlink_to(target)
            with self.subTest(name=name), self.assertRaises((signing.SigningError, OSError)):
                self.resume()
            self.assertFalse(self.tools.calls)
            path.unlink()
            target.rename(path)
        linked = self.root / "linked-checkpoint"
        linked.symlink_to(self.checkpoint, target_is_directory=True)
        with self.assertRaises(signing.SigningError):
            self.resume(checkpoint_directory=linked)
        for path in (self.source, self.keychain):
            target = path.with_name("original-" + path.name)
            path.rename(target)
            path.symlink_to(target)
            with self.subTest(path=path), self.assertRaises(signing.SigningError):
                self.resume()
            self.assertFalse(self.tools.calls)
            path.unlink()
            target.rename(path)
        self.output.write_bytes(b"existing output")
        with self.assertRaises(signing.SigningError):
            self.resume()
        self.assertEqual(self.output.read_bytes(), b"existing output")
        self.assertFalse(self.tools.calls)

    def test_resume_rechecks_original_source_and_retained_upload_before_output_creation(self):
        self.prepare_checkpoint()
        self.tools.before_assess = lambda: (self.checkpoint / "signed-upload.pkg").write_bytes(b"changed upload")
        with self.assertRaises(signing.SigningError):
            self.resume()
        self.assertFalse(self.output.exists())
        self.assertFalse(json.loads((self.evidence / "summary.json").read_text())["original_unchanged"])
        self.assert_no_sign_or_submit()

    def test_resume_rechecks_original_unsigned_source_before_output_creation(self):
        self.prepare_checkpoint()
        self.tools.before_assess = lambda: self.source.write_bytes(b"changed input")
        with self.assertRaises(signing.SigningError):
            self.resume()
        self.assertFalse(self.output.exists())
        self.assertFalse(json.loads((self.evidence / "summary.json").read_text())["original_unchanged"])
        self.assert_no_sign_or_submit()

    def test_resume_preserves_a_racing_output_and_does_not_mark_it_verified(self):
        self.prepare_checkpoint()
        self.tools.before_assess = lambda: self.output.write_bytes(b"another process's file")
        with self.assertRaises(FileExistsError):
            self.resume()
        self.assertEqual(self.output.read_bytes(), b"another process's file")
        self.assertEqual(json.loads((self.evidence / "summary.json").read_text())["status"], "failed")
        self.assertIsNone(json.loads((self.evidence / "evidence-manifest.json").read_text())["artifact"])
        self.assert_no_sign_or_submit()

    def test_resume_cli_requires_both_pins_and_does_not_require_an_installer_identity(self):
        self.prepare_checkpoint()
        arguments = [
            "sign_macos_package.py", "--input", str(self.source), "--output", str(self.output),
            "--team-id", TEAM_ID, "--keychain", str(self.keychain), "--notary-profile", "release notary profile",
            "--evidence-dir", str(self.evidence), "--resume-from", str(self.checkpoint),
            "--source-sha256", self.source_pin,
        ]
        with mock.patch.object(signing.sys, "argv", arguments), mock.patch.object(signing.sys, "stderr"):
            with self.assertRaises(SystemExit) as error:
                signing.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.tools.calls)
        arguments.extend(["--signed-sha256", self.signed_pin])
        with mock.patch.object(signing.sys, "argv", arguments), mock.patch.object(signing.sys, "stdout"):
            self.assertEqual(signing.main(), 0)
        self.assertEqual(self.output.read_bytes(), self.original + b".signed.ticket")
        self.assert_no_sign_or_submit()

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

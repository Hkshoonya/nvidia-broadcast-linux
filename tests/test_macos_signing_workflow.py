"""Exercise credential cleanup and status queries without Apple tools."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ("build-packages.yml", "macos-notarization-recovery.yml")


def run_body(workflow_name, step_name):
    workflow = (ROOT / ".github/workflows" / workflow_name).read_text()
    step = workflow.split(f"      - name: {step_name}\n", 1)[1]
    body = step.split("        run: |\n", 1)[1]
    body = body.split("\n      - ", 1)[0].split("\n  #", 1)[0]
    return textwrap.dedent(body)


class SigningWorkflowCleanupTests(unittest.TestCase):
    def cleanup_script(self, security_tool, workflow_name):
        body = run_body(workflow_name, "Remove signing material")
        return body.replace("/usr/bin/security", shlex.quote(str(security_tool)))

    def test_material_removed_even_when_keychain_deletion_fails(self):
        for workflow_name, failure in ((name, code) for name in WORKFLOWS for code in (0, 7)):
            with self.subTest(workflow=workflow_name, security_exit=failure), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp)
                staging = parent / "nvbroadcast-signing.abc123"
                staging.mkdir()
                for name in ("installer.keychain-db", "certificate.p12", "certificate-password"):
                    (staging / name).write_text("fixture sentinel; no secret")
                tool = parent / "security-fixture"
                tool.write_text(f"#!/bin/bash\nexit {failure}\n")
                tool.chmod(0o755)
                result = subprocess.run(
                    ["bash", "-c", self.cleanup_script(tool, workflow_name)],
                    env={"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(parent),
                         "MACOS_SIGNING_TEMP": str(staging)},
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, failure, result.stderr)
                self.assertFalse(staging.exists())

    def test_cleanup_refuses_redirected_directory(self):
        for workflow_name in WORKFLOWS:
            with self.subTest(workflow=workflow_name), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp)
                original = parent / "keep"
                original.mkdir()
                sentinel = original / "certificate-password"
                sentinel.write_text("keep this fixture")
                staging = parent / "nvbroadcast-signing.abc123"
                staging.symlink_to(original, target_is_directory=True)
                result = subprocess.run(
                    ["bash", "-c", self.cleanup_script("/usr/bin/false", workflow_name)],
                    env={"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(parent),
                         "MACOS_SIGNING_TEMP": str(staging)},
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(sentinel.read_text(), "keep this fixture")
                self.assertTrue(staging.is_symlink())


class NotarizationStatusTests(unittest.TestCase):
    submission_id = "59a39df1-f83b-4e0f-83d0-298b16ea4e52"

    @staticmethod
    def status_script():
        body = run_body("macos-notarization-recovery.yml", "Query the existing Apple submission")
        return body.split("<<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]

    def execute_query(self, root, response, *, returncode=0, native_error=None):
        result = subprocess.CompletedProcess([], returncode, json.dumps(response), "fixture diagnostic")
        environment = {
            "SUBMISSION_ID": self.submission_id,
            "MACOS_SIGNING_KEYCHAIN": str(root / "isolated.keychain-db"),
            "GITHUB_STEP_SUMMARY": str(root / "step-summary.md"),
        }
        previous = Path.cwd()
        try:
            os.chdir(root)
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                "subprocess.run", return_value=result, side_effect=native_error,
            ) as native, mock.patch("builtins.print"):
                exec(compile(self.status_script(), "status-workflow", "exec"), {})
                arguments = native.call_args.args[0]
                self.assertEqual(arguments[1:4], ["notarytool", "info", self.submission_id])
                self.assertNotIn("submit", arguments)
                self.assertFalse(native.call_args.kwargs["check"])
        finally:
            os.chdir(previous)

    def test_pending_and_terminal_statuses_do_not_verify_a_package(self):
        for status in ("In Progress", "Accepted", "Invalid"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.execute_query(root, {"id": self.submission_id, "status": status})
                record = json.loads((root / "dist/macos-recovery/status/command.json").read_text())
                self.assertFalse(record["distribution_verified"])
                self.assertIn(status, (root / "step-summary.md").read_text())
                self.assertFalse((root / "dist/pkg-signed").exists())

    def test_wrong_submission_and_failed_native_query_retain_evidence_and_fail(self):
        cases = (
            ({"id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "status": "Accepted"}, 0),
            ({"id": self.submission_id, "status": "unknown"}, 0),
            ({"id": self.submission_id, "status": "Accepted"}, 1),
        )
        for response, code in cases:
            with self.subTest(response=response, code=code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaises(SystemExit):
                    self.execute_query(root, response, returncode=code)
                evidence = root / "dist/macos-recovery/status"
                self.assertEqual(json.loads((evidence / "notary-info.stdout.json").read_text()), response)
                self.assertEqual((evidence / "notary-info.stderr.txt").read_text(), "fixture diagnostic")
                self.assertFalse((root / "step-summary.md").exists())

    def test_status_query_timeout_retains_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            error = subprocess.TimeoutExpired(["notarytool", "info"], 300,
                                             output=b"partial status", stderr=b"partial diagnostic")
            with self.assertRaises(SystemExit):
                self.execute_query(root, {}, native_error=error)
            evidence = root / "dist/macos-recovery/status"
            self.assertEqual((evidence / "notary-info.stdout.json").read_text(), "partial status")
            self.assertEqual((evidence / "notary-info.stderr.txt").read_text(), "partial diagnostic")
            record = json.loads((evidence / "command.json").read_text())
            self.assertIsNone(record["returncode"])
            self.assertIn("timed out", record["error"])
            self.assertFalse(record["distribution_verified"])


if __name__ == "__main__":
    unittest.main()

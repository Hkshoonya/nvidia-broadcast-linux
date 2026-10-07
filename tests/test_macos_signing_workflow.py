"""Exercise credential cleanup failures without credentials or Apple tools."""

from pathlib import Path
import shlex
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SigningWorkflowCleanupTests(unittest.TestCase):
    def cleanup_script(self, security_tool):
        workflow = (ROOT / ".github/workflows/build-packages.yml").read_text()
        step = workflow.split("      - name: Remove signing material\n", 1)[1]
        body = step.split("        run: |\n", 1)[1].split("\n  #", 1)[0]
        return textwrap.dedent(body).replace("/usr/bin/security", shlex.quote(str(security_tool)))

    def test_material_removed_even_when_keychain_deletion_fails(self):
        for failure in (0, 7):
            with self.subTest(security_exit=failure), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp)
                staging = parent / "nvbroadcast-signing.abc123"
                staging.mkdir()
                for name in ("installer.keychain-db", "certificate.p12", "certificate-password"):
                    (staging / name).write_text("fixture sentinel; no secret")
                tool = parent / "security-fixture"
                tool.write_text(f"#!/bin/bash\nexit {failure}\n")
                tool.chmod(0o755)
                result = subprocess.run(
                    ["bash", "-c", self.cleanup_script(tool)],
                    env={"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(parent),
                         "MACOS_SIGNING_TEMP": str(staging)},
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, failure, result.stderr)
                self.assertFalse(staging.exists())

    def test_cleanup_refuses_redirected_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            original = parent / "keep"
            original.mkdir()
            sentinel = original / "certificate-password"
            sentinel.write_text("keep this fixture")
            staging = parent / "nvbroadcast-signing.abc123"
            staging.symlink_to(original, target_is_directory=True)
            result = subprocess.run(
                ["bash", "-c", self.cleanup_script("/usr/bin/false")],
                env={"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(parent),
                     "MACOS_SIGNING_TEMP": str(staging)},
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "keep this fixture")
            self.assertTrue(staging.is_symlink())


if __name__ == "__main__":
    unittest.main()

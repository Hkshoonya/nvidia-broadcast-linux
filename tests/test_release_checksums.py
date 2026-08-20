import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_release_checksums.py"
SPEC = importlib.util.spec_from_file_location("generate_release_checksums", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKSUMS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKSUMS)


class ReleaseChecksumTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_manifest_is_sorted_and_uses_release_basenames(self):
        rpm = self.root / "nested" / "nvbroadcast-1.4.0-1.noarch.rpm"
        deb = self.root / "nvbroadcast_1.4.0-1_all.deb"
        rpm.parent.mkdir()
        rpm.write_bytes(b"rpm payload")
        deb.write_bytes(b"deb payload")
        output = self.root / "release" / "SHA256SUMS.packages"

        lines = CHECKSUMS.generate_manifest((rpm, deb), output)

        expected = (
            f"{hashlib.sha256(rpm.read_bytes()).hexdigest()}  {rpm.name}",
            f"{hashlib.sha256(deb.read_bytes()).hexdigest()}  {deb.name}",
        )
        self.assertEqual(lines, expected)
        self.assertEqual(output.read_text(encoding="ascii"), "\n".join(expected) + "\n")
        self.assertEqual(output.stat().st_mode & 0o777, 0o644)

    def test_cli_writes_manifest_and_reports_input_errors(self):
        artifact = self.root / "package.pkg"
        artifact.write_bytes(b"payload")
        output = self.root / "SHA256SUMS.packages"

        completed = subprocess.run(
            (
                sys.executable,
                str(SCRIPT_PATH),
                "--output",
                str(output),
                str(artifact),
            ),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("with 1 SHA-256 entries", completed.stdout)
        self.assertEqual(
            output.read_text(encoding="ascii"),
            f"{hashlib.sha256(b'payload').hexdigest()}  {artifact.name}\n",
        )

        failed = subprocess.run(
            (
                sys.executable,
                str(SCRIPT_PATH),
                "--output",
                str(output),
                str(self.root / "missing.snap"),
            ),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("error: release artifact does not exist", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)

    def test_duplicate_basenames_do_not_replace_existing_manifest(self):
        first = self.root / "one" / "package.rpm"
        second = self.root / "two" / "package.rpm"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        output = self.root / "SHA256SUMS.packages"
        output.write_text("existing manifest\n", encoding="ascii")

        with self.assertRaisesRegex(
            CHECKSUMS.ManifestError,
            "duplicate release artifact name",
        ):
            CHECKSUMS.generate_manifest((first, second), output)

        self.assertEqual(output.read_text(encoding="ascii"), "existing manifest\n")

    def test_symlink_artifact_is_rejected(self):
        target = self.root / "package.deb"
        target.write_bytes(b"payload")
        link = self.root / "linked.deb"
        link.symlink_to(target)

        with self.assertRaisesRegex(CHECKSUMS.ManifestError, "must not be a symlink"):
            CHECKSUMS.generate_manifest((link,), self.root / "SHA256SUMS")

    def test_unsafe_release_name_is_rejected(self):
        artifact = self.root / "package\nforged.rpm"
        artifact.write_bytes(b"payload")

        with self.assertRaisesRegex(CHECKSUMS.ManifestError, "unsafe release artifact name"):
            CHECKSUMS.generate_manifest((artifact,), self.root / "SHA256SUMS")

    def test_missing_directory_and_manifest_inputs_are_rejected(self):
        output = self.root / "SHA256SUMS"
        directory = self.root / "artifact-directory"
        directory.mkdir()
        regular = self.root / "package.pkg"
        regular.write_bytes(b"payload")

        cases = (
            ((), "at least one release artifact"),
            ((self.root / "missing.snap",), "does not exist"),
            ((directory,), "not a regular file"),
            ((output,), "does not exist"),
        )
        for artifacts, message in cases:
            with self.subTest(artifacts=artifacts):
                with self.assertRaisesRegex(CHECKSUMS.ManifestError, message):
                    CHECKSUMS.generate_manifest(artifacts, output)

        output.write_text("old\n", encoding="ascii")
        with self.assertRaisesRegex(CHECKSUMS.ManifestError, "cannot hash itself"):
            CHECKSUMS.generate_manifest((output, regular), output)


if __name__ == "__main__":
    unittest.main()

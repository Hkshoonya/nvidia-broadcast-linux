"""Signing trust boundaries, plus real RPM/GnuPG tests when tools exist."""

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sign_rpm_package", ROOT / "scripts/sign_rpm_package.py")
signing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(signing)
FINGERPRINT = "A" * 40


class SigningBoundaryTests(unittest.TestCase):
    def test_fingerprint_is_complete_and_primary_only(self):
        listing = "pub:u:2048:1:AAA:0:0:::::sc:\nfpr:::::::::" + FINGERPRINT + ":\n"
        listing += "sub:u:2048:1:BBB:0:0:::::s:\nfpr:::::::::" + "B" * 40 + ":\n"
        self.assertEqual(signing.primary_fingerprints(listing), [FINGERPRINT])
        for state in ("r", "e", "d"):
            with self.subTest(state=state), self.assertRaises(signing.SigningError):
                signing.primary_fingerprints(listing.replace("pub:u:", "pub:" + state + ":"))

    def test_short_key_ids_are_rejected_before_any_tool(self):
        with mock.patch.object(signing, "run") as tool:
            with self.assertRaisesRegex(signing.SigningError, "complete OpenPGP fingerprint"):
                signing.sign(Path("a.rpm"), Path("key"), "12345678", Path("out"))
            tool.assert_not_called()

    def test_unsafe_package_names_are_rejected_before_any_tool(self):
        for name in ("a\nSignature: OK\na.rpm", "$(id).rpm", "%{lua:print(1)}.rpm", "-options.rpm"):
            with self.subTest(name=name), mock.patch.object(signing, "run") as tool:
                with self.assertRaisesRegex(signing.SigningError, "safe RPM package basename"):
                    signing.sign(Path(name), Path("key"), FINGERPRINT, Path("out"))
                tool.assert_not_called()

    def test_snapshot_refuses_symlink_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "source.rpm"
            original.write_bytes(b"original")
            link = root / "link.rpm"
            link.symlink_to(original)
            with self.assertRaises(OSError):
                signing.snapshot(link, root / "rejected")
            digest = signing.snapshot(original, root / "copy")
            self.assertEqual(digest, hashlib.sha256(b"original").hexdigest())
            self.assertEqual(original.read_bytes(), b"original")
            self.assertEqual((root / "copy").stat().st_mode & 0o777, 0o600)

    def test_failed_tool_diagnostics_do_not_expose_key_material(self):
        failed = subprocess.CompletedProcess(["gpg"], 2, b"", b"PRIVATE-SECRET-SENTINEL")
        with mock.patch.object(signing.subprocess, "run", return_value=failed):
            with self.assertRaises(signing.SigningError) as error:
                signing.run(["gpg", "--import"], {}, data=b"PRIVATE-SECRET-SENTINEL")
        self.assertNotIn("PRIVATE-SECRET", str(error.exception))


TOOLS = ("gpg", "gpgconf", "rpm", "rpmkeys", "rpmsign", "rpm2cpio", "rpmbuild")


@unittest.skipUnless(all(shutil.which(tool) for tool in TOOLS), "RPM and GnuPG tools required")
class RealRPMSigningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="nvb-signing-tests-")
        cls.root = Path(cls.temporary.name)
        cls.home = cls.root / "keys"
        cls.home.mkdir(mode=0o700)
        cls.tool_environment = dict(os.environ, GNUPGHOME=str(cls.home), LC_ALL="C")
        # Disposable test-only key, generated inside the test's private keyring.
        cls.password = "nvb-disposable-signature-test"
        subprocess.run(["gpg", "--batch", "--pinentry-mode", "loopback", "--passphrase-fd", "0",
                        "--quick-generate-key", "NV Broadcast Disposable Test", "rsa2048", "cert", "1d"],
                       input=cls.password.encode(), env=cls.tool_environment, check=True, capture_output=True)
        listing = subprocess.check_output(["gpg", "--batch", "--with-colons", "--fingerprint", "--list-secret-keys"],
                                          env=cls.tool_environment, text=True)
        cls.fingerprint = signing.primary_fingerprints(listing)[0]
        subprocess.run(["gpg", "--batch", "--pinentry-mode", "loopback", "--passphrase-fd", "0",
                        "--quick-add-key", cls.fingerprint, "rsa2048", "sign", "1d"],
                       input=cls.password.encode(), env=cls.tool_environment, check=True, capture_output=True)
        # CI receives only the signing subkey; the certification primary stays
        # in the maintainer's encrypted backup.
        cls.private = subprocess.check_output(["gpg", "--batch", "--pinentry-mode", "loopback", "--passphrase-fd", "0",
                                              "--armor", "--export-secret-subkeys", cls.fingerprint],
                                             input=cls.password.encode(), env=cls.tool_environment).decode()
        cls.public = cls.root / "public.asc"
        cls.public.write_bytes(subprocess.check_output(["gpg", "--batch", "--armor", "--export", cls.fingerprint],
                                                       env=cls.tool_environment))
        cls.spec = cls.root / "fixture.spec"
        cls.spec.write_text('''Name: nvb-signature-fixture
Version: 1.0
Release: 1
Summary: Disposable signature regression fixture
License: MIT
BuildArch: noarch
%description
No hooks or host configuration changes.
%install
mkdir -p %{buildroot}/usr/share/nvb-signature-fixture
printf 'signature-test-payload\\n' > %{buildroot}/usr/share/nvb-signature-fixture/example
%files
/usr/share/nvb-signature-fixture/example
''')
        subprocess.run(["rpmbuild", "--define", f"_topdir {cls.root / 'build'}", "-bb", str(cls.spec)],
                       check=True, capture_output=True)
        cls.package = next((cls.root / "build/RPMS").rglob("*.rpm"))

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["gpgconf", "--homedir", str(cls.home), "--kill", "gpg-agent"],
                       capture_output=True, check=False)
        cls.temporary.cleanup()

    def environment(self):
        return mock.patch.dict(os.environ, {"NVB_RPM_PRIVATE_KEY": self.private,
                                           "NVB_RPM_KEY_PASSPHRASE": self.password})

    def test_protected_key_signs_and_tampered_payload_is_rejected(self):
        original = self.package.read_bytes()
        output = self.root / "signed"
        with self.environment():
            report = signing.sign(self.package, self.public, self.fingerprint, output)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(self.package.read_bytes(), original)
        self.assertNotEqual(report["input_sha256"], report["signed_sha256"])
        self.assertEqual(sorted(p.name for p in output.iterdir()),
                         sorted([self.package.name, "RPM-GPG-KEY-nvbroadcast.asc", "rpm-signing.json"]))
        self.assertNotIn(self.password, (output / "rpm-signing.json").read_text())
        damaged = self.root / "tampered.rpm"
        content = bytearray((output / self.package.name).read_bytes())
        content[-10] ^= 1
        damaged.write_bytes(content)
        with self.assertRaises(signing.SigningError):
            signing.verify_rpm(damaged, self.public, self.fingerprint, dict(os.environ, LC_ALL="C"))

    def test_unsigned_rpm_with_valid_digests_is_rejected(self):
        with self.assertRaises(signing.SigningError):
            signing.verify_rpm(self.package, self.public, self.fingerprint, dict(os.environ, LC_ALL="C"))

    def test_unsigned_rpm_filename_cannot_spoof_signature_output(self):
        deceptive = self.root / "spoof\nSignature: OK\nunsigned.rpm"
        deceptive.write_bytes(self.package.read_bytes())
        with self.assertRaisesRegex(signing.SigningError, "no verified OpenPGP signature"):
            signing.verify_rpm(deceptive, self.public, self.fingerprint, dict(os.environ))

    def test_wrong_pinned_key_rejected_without_output(self):
        output = self.root / "wrong-key"
        with self.environment(), self.assertRaisesRegex(signing.SigningError, "Private key does not match"):
            signing.sign(self.package, self.public, FINGERPRINT, output)
        self.assertFalse(output.exists())

    def test_private_key_cannot_be_published_as_the_public_key(self):
        secret_file = self.root / "not-public.asc"
        secret_file.write_text(self.private)
        output = self.root / "private-key-rejected"
        with self.environment(), self.assertRaisesRegex(signing.SigningError, "public key"):
            signing.sign(self.package, secret_file, self.fingerprint, output)
        self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "existing-output"
        output.mkdir()
        (output / "sentinel").write_text("preserve")
        with self.environment(), self.assertRaisesRegex(signing.SigningError, "already exists"):
            signing.sign(self.package, self.public, self.fingerprint, output)
        self.assertEqual((output / "sentinel").read_text(), "preserve")

    def test_wrong_passphrase_does_not_reuse_other_keyring_agent(self):
        output = self.root / "wrong-passphrase"
        with self.environment(), mock.patch.dict(os.environ, {"NVB_RPM_KEY_PASSPHRASE": "wrong"}):
            with self.assertRaises(signing.SigningError):
                signing.sign(self.package, self.public, self.fingerprint, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

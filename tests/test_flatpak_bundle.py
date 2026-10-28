import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flatpak_bundle", ROOT / "packaging/flatpak/export_development_bundle.py"
)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class FlatpakBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.build = self.root / "build"
        self.repo = self.root / "repo"
        self.output = self.root / "output"
        self.source.mkdir()
        (self.source / "pyproject.toml").write_text(
            '[tool.setuptools.package-data]\n"nvbroadcast.ai" = ["trust.json"]\n'
            '[tool.setuptools.data-files]\n'
            '"share/doc/nvbroadcast" = ["NOTICE", "CONTRIBUTORS.md"]\n'
        )
        for name in (
            "src/nvbroadcast/__main__.py", "src/nvbroadcast/ai/trust.json",
            "LICENSE", "NOTICE", "CONTRIBUTORS.md",
            "packaging/flatpak/com.nvbroadcast.NVBroadcast.yml",
            "packaging/flatpak/python3-flatpak-requirements.yaml",
            "packaging/flatpak/requirements.txt",
            *bundle.BUILD_INPUTS[4:],
        ):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        for name in bundle.FLATPAK_METADATA.values():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.site = self.build / "files/lib/python3.13/site-packages"
        license_path = self.site / "nvbroadcast-1.5.3.dist-info/licenses/LICENSE"
        license_path.parent.mkdir(parents=True)
        license_path.write_text("LICENSE")
        (self.build / "metadata").write_text(
            f"[Application]\nname={bundle.APP_ID}\n"
            "runtime=org.gnome.Platform/x86_64/50\n"
            "[Environment]\nPYTHONPATH=/app/lib/python3.13/site-packages:/usr/lib/python3.13/site-packages\n"
            "SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem\n"
            "SSL_CERT_DIR=/etc/pki/tls/certs\nPYTHONNOUSERSITE=1\n"
        )
        self.commit = "c" * 64
        self.imported_commit = self.commit
        self.bad_payload = False
        self.bad_private_payload = False
        self.fail_import = False
        self.commands = []
        self.payloads = bundle.source_payloads(self.source, self.build)
        self.runtime_inputs = {"status": "pass", "base_image": "example/builder@sha256:" + "b" * 64,
                               "lock_sha256": "a" * 64}
        self.private_runtime = {"python": {"version": "3.13.18"},
                                "ssl_fixture": {"path": "lib/python3.13/test/certdata/distrusted.zip"},
                                "license_files": {"LICENSE.cpython.txt": "a" * 64}}
        self.private_payloads = bundle.private_python_payloads(self.build / 'files', self.private_runtime)
        for name, path in self.private_payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        self.preparation = self.source / "dist/flatpak-runtime/prepared-image.json"
        self.preparation.parent.mkdir(parents=True)
        self.preparation.write_text(json.dumps({
            "schema_version": 1, "status": "pass", "base_image": self.runtime_inputs["base_image"],
            "base_image_id": "sha256:" + "b" * 64, "prepared_image_id": "sha256:" + "e" * 64,
            "base_rootfs_layers": ["sha256:" + "c" * 64],
            "prepared_rootfs_layers": ["sha256:" + "c" * 64, "sha256:" + "d" * 64],
            "effective_environment": {"FLATPAK_SYSTEM_DIR": "/var/lib/nvbroadcast-flatpak",
                                      "NVIDIA_VISIBLE_DEVICES": "void"},
            "runtime_inputs": self.runtime_inputs, "lock_sha256": self.runtime_inputs["lock_sha256"],
        }))
        subprocess.run(("git", "init", "--quiet", str(self.source)), check=True)
        subprocess.run(("git", "-C", str(self.source), "add", "."), check=True)
        subprocess.run(
            ("git", "-C", str(self.source), "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture"),
            check=True,
        )
        self.revision = subprocess.check_output(
            ("git", "-C", str(self.source), "rev-parse", "HEAD"), text=True
        ).strip()

    def fake_command(self, *args):
        self.commands.append(args)
        if "rev-parse" in args:
            value = self.imported_commit if "imported-repo" in args[1] else self.commit
            return (value + "\n").encode()
        if args[:2] == ("flatpak", "build-bundle"):
            Path(args[4]).write_bytes(b"test bundle")
        if args[:2] == ("flatpak", "build-import-bundle") and self.fail_import:
            raise subprocess.CalledProcessError(1, args)
        if "cat" in args:
            if args[-1] == "metadata":
                return (self.build / "metadata").read_bytes()
            name = args[-1].removeprefix("files/")
            if self.bad_payload and name.endswith("trust.json"):
                return b"different model trust data"
            if name in self.private_payloads:
                if self.bad_private_payload and name.endswith('LICENSE.cpython.txt'):
                    return b'different native interpreter notice'
                return self.private_payloads[name].read_bytes()
            return self.payloads[name].read_bytes()
        return b""

    def export(self, revision=None):
        revision = self.revision if revision is None else revision
        with patch.object(bundle, "command", side_effect=self.fake_command), \
                patch.object(bundle.runtime_policy, "inspect_runtime", return_value=self.runtime_inputs), \
                patch.object(bundle.private_python, "verify_prefix_provenance", return_value=(self.private_runtime, "a" * 64)):
            return bundle.export_bundle(
                self.source, self.build, self.repo, self.output, revision,
                "example/builder@sha256:" + "b" * 64,
                "sha256:" + "e" * 64,
            )

    def test_flatpak_metadata_cannot_drift_from_the_named_commit(self):
        path = self.source / bundle.FLATPAK_METADATA[
            "share/metainfo/com.nvbroadcast.NVBroadcast.metainfo.xml"
        ]
        path.write_text("unreviewed metadata")
        with self.assertRaisesRegex(ValueError, "Source differs from named Git revision"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_success_records_bundle_digest_source_files_and_import_proof(self):
        evidence = self.export()
        self.assertEqual(evidence["application_commit"], self.commit)
        self.assertEqual(evidence["prepared_builder_image"], "sha256:" + "e" * 64)
        self.assertEqual(evidence["runtime_inputs"]["status"], "pass")
        self.assertEqual(evidence["builder_preparation"]["runtime_inputs"], evidence["runtime_inputs"])
        self.assertEqual(evidence["builder_preparation_sha256"], bundle.sha256(self.preparation))
        self.assertEqual(evidence["private_python"], self.private_runtime)
        self.assertEqual(evidence["private_python_sha256"], "a" * 64)
        self.assertFalse(evidence["public_distribution_qualified"])
        self.assertFalse(evidence["host_installation_performed"])
        self.assertEqual(len(evidence["verification"]["source_payloads"]), 9)
        self.assertEqual(len(evidence["verification"]["private_python_payloads"]), 9)
        self.assertEqual(
            evidence["bundle"]["sha256"], hashlib.sha256(b"test bundle").hexdigest()
        )
        saved = json.loads((self.output / "bundle-provenance.json").read_text())
        self.assertEqual(saved, evidence)
        self.assertEqual(len((self.output / "SHA256SUMS").read_text().splitlines()), 2)
        self.assertTrue(any("fsck" in args for args in self.commands))
        self.assertFalse(any("install" in args for args in self.commands))

    def test_wrong_committed_image_or_runtime_proof_prevents_export(self):
        original = self.preparation.read_text()
        for field, value in (("prepared_image_id", "sha256:" + "f" * 64),
                             ("base_rootfs_layers", ["sha256:" + "f" * 64]),
                             ("effective_environment", {}), ("runtime_inputs", {})):
            with self.subTest(field=field):
                proof = json.loads(original)
                proof[field] = value
                self.preparation.write_text(json.dumps(proof))
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())
                self.assertFalse(any(args[:2] == ("flatpak", "build-bundle") for args in self.commands))

    def test_missing_committed_image_inspection_prevents_export(self):
        self.preparation.unlink()
        with self.assertRaisesRegex(ValueError, "Missing, redirected"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_private_interpreter_provenance_failure_prevents_bundle_creation(self):
        with patch.object(bundle, "command", side_effect=self.fake_command), \
                patch.object(bundle.runtime_policy, "inspect_runtime", return_value=self.runtime_inputs), \
                patch.object(bundle.private_python, "verify_prefix_provenance", side_effect=ValueError("private input mismatch")):
            with self.assertRaisesRegex(ValueError, "private input mismatch"):
                bundle.export_bundle(self.source, self.build, self.repo, self.output, self.revision,
                                     "example/builder@sha256:" + "b" * 64, "sha256:" + "e" * 64)
        self.assertFalse(self.output.exists())
        self.assertFalse(any(args[:2] == ("flatpak", "build-bundle") for args in self.commands))

    def test_missing_or_shared_startup_path_prevents_export(self):
        metadata = self.build / "metadata"
        original = metadata.read_text()
        for key, path in (("PYTHONPATH", ""), ("PYTHONPATH", "/usr/lib/python3.13/site-packages"),
                          ("pythonpath", "/app/lib/python3.13/site-packages:/usr/lib/python3.13/site-packages")):
            with self.subTest(key=key, path=path):
                metadata.write_text(original.replace("PYTHONPATH=/app/lib/python3.13/site-packages:/usr/lib/python3.13/site-packages",
                                                     f"{key}={path}"))
                with self.assertRaisesRegex(ValueError, "before startup"):
                    self.export()
                self.assertFalse(self.output.exists())
                self.assertFalse(any(args[:2] == ("flatpak", "build-bundle") for args in self.commands))

    def test_final_metadata_cannot_omit_or_mis_case_private_trust_environment(self):
        metadata = self.build / 'metadata'
        original = metadata.read_text()
        for key in ('SSL_CERT_FILE', 'SSL_CERT_DIR', 'PYTHONNOUSERSITE'):
            with self.subTest(key=key):
                metadata.write_text(original.replace(key + '=', key.lower() + '='))
                with self.assertRaisesRegex(ValueError, 'trust environment'):
                    self.export()
                self.assertFalse(self.output.exists())
                self.assertEqual(self.commands, [])

    def test_import_failure_keeps_output_absent(self):
        self.fail_import = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_wrong_imported_commit_keeps_output_absent(self):
        self.imported_commit = "d" * 64
        with self.assertRaisesRegex(ValueError, "commit differs"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_changed_model_trust_data_keeps_output_absent(self):
        self.bad_payload = True
        with self.assertRaisesRegex(ValueError, "does not match source"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_imported_private_interpreter_notice_must_match_verified_prefix(self):
        self.bad_private_payload = True
        with self.assertRaisesRegex(ValueError, 'does not match source'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_missing_project_license_stops_before_export(self):
        for path in self.site.rglob("LICENSE"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "project LICENSE"):
            self.export()
        self.assertEqual(self.commands[-1][2], "rev-parse")
        self.assertFalse(self.output.exists())

    def test_wrong_runtime_stops_before_export(self):
        path = self.build / "metadata"
        path.write_text(path.read_text().replace("x86_64/50", "aarch64/50"))
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.export()
        self.assertEqual(self.commands, [])

    def test_short_source_revision_stops_before_export(self):
        with self.assertRaisesRegex(ValueError, "full checked-out Git"):
            self.export("abcdef")
        self.assertEqual(self.commands, [])

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        sentinel = self.output / "existing.flatpak"
        sentinel.write_bytes(b"keep existing artifact")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.export()
        self.assertEqual(sentinel.read_bytes(), b"keep existing artifact")
        self.assertEqual(self.commands, [])

    def test_staged_source_edit_is_rejected_before_bundle_export(self):
        path = self.source / "src/nvbroadcast/__main__.py"
        path.write_text("staged changed app")
        subprocess.run(("git", "-C", str(self.source), "add", str(path)), check=True)
        with self.assertRaisesRegex(ValueError, "Source differs from named Git"):
            self.export()
        self.assertFalse(any("build-bundle" in args for args in self.commands))
        self.assertFalse(self.output.exists())

    def test_unstaged_dependency_input_edit_is_rejected(self):
        (self.source / "packaging/flatpak/requirements.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Source differs from named Git"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_runtime_lock_edits_are_bound_to_named_source(self):
        (self.source / "packaging/flatpak/runtime-lock.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Source differs from named Git"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_export_rechecks_actual_runtime_and_fails_before_bundle_creation(self):
        with patch.object(bundle, "command", side_effect=self.fake_command), \
                patch.object(bundle.runtime_policy, "inspect_runtime", side_effect=ValueError("stale runtime")):
            with self.assertRaisesRegex(ValueError, "stale runtime"):
                bundle.export_bundle(self.source, self.build, self.repo, self.output,
                                     self.revision, "example/builder@sha256:" + "b" * 64,
                                     "sha256:" + "e" * 64)
        self.assertFalse(any('build-bundle' in args for args in self.commands))
        self.assertFalse(self.output.exists())

    def test_untracked_application_file_is_rejected(self):
        (self.source / "src/nvbroadcast/untracked.py").write_text("not committed")
        with self.assertRaises(subprocess.CalledProcessError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_source_symlink_cannot_claim_the_target_files_git_identity(self):
        path = self.source / "src/nvbroadcast/__main__.py"
        path.unlink()
        path.symlink_to(self.source / "NOTICE")
        with self.assertRaisesRegex(ValueError, "Symlinks are not supported"):
            self.export()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

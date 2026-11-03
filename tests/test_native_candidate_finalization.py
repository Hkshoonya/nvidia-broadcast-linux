"""Native candidate signing and qualification trust boundaries, without secrets.

The signer is replaced with a deterministic file-producing stub. These tests
exercise receipt checks, real bundle/helper construction, and retained evidence;
they neither sign packages nor execute a downloaded package or container.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "native_candidate_finalization", ROOT / "packaging/native-runtime/finalize_candidate.py")
finalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalize)
SOURCE = "1" * 40
RUNTIME = "2" * 64
RECOVERY = "3" * 64


class CandidateFixture(unittest.TestCase):
    variant = "cpu"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nvb-finalization-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.validated = self.root / "validated"
        self.validated.mkdir()
        self.version = "1.5.3-4." + self.variant
        self.report = {
            "status": "pass", "source": SOURCE, "version": self.version, "variant": self.variant,
            "run_id": 11, "run_attempt": 2, "artifact_id": 13,
            "artifact_digest": "sha256:" + "4" * 64, "packages": {},
        }
        for family, name in (("deb", f"nvbroadcast_{self.version}_amd64.deb"),
                             ("rpm", f"nvbroadcast-{self.version}.x86_64.rpm")):
            path = self.validated / family / name
            path.parent.mkdir()
            path.write_bytes(("unsigned-" + family).encode())
            self.report["packages"][family] = {
                "relative_path": str(path.relative_to(self.validated)),
                **finalize.record_file(path), "runtime_manifest_sha256": RUNTIME,
                "scriptlet_source_match": True,
            }
        self.report["packages"]["rpm"]["recovery_helper_sha256"] = RECOVERY
        self.receipt = self.validated / "result.json"
        self.save_receipt()
        self.output = self.root / "signed"
        self.signer = mock.Mock(side_effect=self.fake_sign)
        original_module = finalize.module
        self.module_loader = mock.patch.object(
            finalize, "module", side_effect=lambda name, relative:
            types.SimpleNamespace(sign=self.signer) if relative == "scripts/sign_rpm_package.py"
            else original_module(name, relative))
        self.module_loader.start()
        self.addCleanup(self.module_loader.stop)
        self.ancestry = mock.patch.object(finalize, "source_identity")
        self.source_identity = self.ancestry.start()
        self.addCleanup(self.ancestry.stop)

    def save_receipt(self):
        self.receipt.write_bytes(finalize.json_bytes(self.report))
        self.receipt_hash = finalize.sha(self.receipt)

    def fake_sign(self, package, public, fingerprint, output):
        output.mkdir()
        signed = output / package.name
        signed.write_bytes(package.read_bytes() + b"-signed")
        (output / "RPM-GPG-KEY-nvbroadcast.asc").write_bytes(public.read_bytes())
        report = {"status": "verified", "input_sha256": finalize.sha(package),
                  "signed_sha256": finalize.sha(signed), "fingerprint": fingerprint,
                  "public_key_sha256": finalize.sha(public)}
        (output / "rpm-signing.json").write_bytes(finalize.json_bytes(report))
        return report

    def build_bundle(self):
        result = finalize.sign_candidate(self.validated, self.receipt_hash, self.output, variant=self.variant)
        self.manifest_hash = result["manifest_sha256"]
        self.manifest = finalize.read_json(self.output / "candidate.json")
        return result

    def rewrite_bundle(self):
        """Rebind fixture files to test semantic checks beneath the outer hash."""
        for name in self.manifest["files"]:
            self.manifest["files"][name] = finalize.record_file(self.output / name)
        for package in self.manifest["packages"].values():
            package.update(self.manifest["files"][package["path"]])
        manifest_path = self.output / "candidate.json"
        manifest_path.write_bytes(finalize.json_bytes(self.manifest))
        self.manifest_hash = finalize.sha(manifest_path)
        checksums = {**self.manifest["files"], "candidate.json": finalize.record_file(manifest_path)}
        (self.output / "SHA256SUMS").write_text("".join(
            f"{item['sha256']}  {name}\n" for name, item in sorted(checksums.items())))

    def runtime_probe(self):
        if self.variant == "cpu":
            return {"status": "pass"}
        return {
            "status": "pass", "variant": "cuda", "cuda_unavailable_expected": True,
            "cpu_execution": {"success": True, "provider": "cpu", "executed_providers": ["CPUExecutionProvider"],
                              "output": [1.0, 4.0, 9.0, 16.0], "output_shape": [1, 4]},
            "cuda_execution": {"success": False, "provider": "cuda", "executed_providers": [],
                               "error": "CUDA device unavailable in isolated fixture"},
            "cupy_execution": None,
        }

    def execution(self, family, recovery=False):
        output = self.root / (family + "-evidence") / ("recovery" if recovery else "smoke")
        output.mkdir(parents=True, exist_ok=True)
        names = (["confirm-clean-fixture", "clean-private-paths", "system-python",
                  "private-verification-keyring", "import-only-project-public-key",
                  "authenticate-signed-target", "signed-rpm-metadata", "authenticated-payload-bindings",
                  "import-install-public-key", "sigkill-observed-first-install",
                  "fresh-launcher-not-committed", "bootstrap-identity", "bootstrap-uses-system-python",
                  "inspect-authenticated-fragments", "preserve-authenticated-fragments",
                  "audit-preserved-bytes-and-marker", "import-preserved-public-key",
                  "replay-exact-preserved-package", "configured-marker-and-bootstrap-cleared",
                  "installed-payload-bindings", "native-package-state", "complete-installed-integrity",
                  "launcher-after-repair"] if recovery else
                 ["confirm-no-existing-package"] +
                 (["fresh-project-verification-keyring", "import-project-verification-key",
                   "verify-finalized-project-signature", "import-project-installation-key"] if family == "rpm" else []) +
                 ["sentinels", "record-system-python-state", "install-exact-hosted-package",
                  "installed-source-adapter-manifest-identity", "complete-installed-integrity",
                  "native-package-state", "ordinary-user-launcher"] +
                 (["system-python-recovery-interface"] if family == "rpm" else []) +
                 ["ordinary-user-runtime-media-gtk", "ordinary-user-generated-recording",
                  "remove-native-package", "package-uninstalled", "preserve-user-system-and-stable-lock",
                  "preserve-system-python-state"])
        if recovery and self.variant == "cuda":
            names.append("recovered-runtime-cpu-fallback")
        report = {"status": "pass", "variant": self.variant, "image": "sha256:" + "5" * 64,
                  "runtime_manifest_sha256": RUNTIME, "container_removed": True, "steps": []}
        if self.variant == "cuda":
            report.update(container_runtime="runc", gpu_device=None, nvidia_visible_devices="void")
        if recovery:
            report.update(source_revision=SOURCE, package_sha256=self.manifest["packages"][family]["sha256"],
                          original_package_unchanged=True, snapshots_unchanged=True, authenticated=True,
                          helper_sha256=RECOVERY,
                          interruption={"signal": "SIGKILL", "fresh_install": True, "observed_bytes": 2 * 1024**2,
                                        "manager_exit": 9})
            manager = output / "interrupted-manager.log"
            manager.write_text("observed manager SIGKILL fixture\n")
            report["interruption"]["manager_log_sha256"] = finalize.sha(manager)
            preservation = output / "preservation.json"
            preservation.write_bytes(finalize.json_bytes({
                "package_sha256": self.manifest["packages"][family]["sha256"],
                "version": self.version, "fingerprint": finalize.FINGERPRINT,
                "entries": [{"original": "/usr/bin/nvbroadcast;1234", "size": 42,
                             "sha256": "7" * 64}],
                "preserved": ["/usr/bin/nvbroadcast;1234"],
            }))
            report["preservation_sha256"] = finalize.sha(preservation)
            if self.variant == "cuda":
                report["runtime_probe"] = self.runtime_probe()
        else:
            report.update(source=SOURCE, artifact_sha256=self.manifest["packages"][family]["sha256"],
                          input_unchanged=True, snapshot_unchanged=True, probes_unchanged=True,
                          family=family, version=self.version,
                          runtime_probe=self.runtime_probe(), recording={"status": "pass"})
            if family == "rpm":
                report["rpm_authentication"] = {"status": "verified", "publisher_fingerprint": finalize.FINGERPRINT}
        expected_failures = {"confirm-clean-fixture", "confirm-no-existing-package", "package-uninstalled"}
        for index, name in enumerate(names):
            log = output / f"{index:02d}-{name}.log"
            log.write_text("fixture log: " + name + "\n")
            report["steps"].append({"name": name, "exit_code": 1 if name in expected_failures else 0,
                                    "expected_failure": name in expected_failures,
                                    "log": log.name, "log_sha256": finalize.sha(log)})
        path = output / "result.json"
        path.write_bytes(finalize.json_bytes(report))
        inventory = output.parent / "fixture-packages.txt"
        inventory.write_text("fixture-dependency\t1.0\tx86_64\n")
        (output.parent / "fixture.json").write_bytes(finalize.json_bytes({
            "family": family, "image": report["image"], "inventory_sha256": finalize.sha(inventory),
            "status": "pass", "inventory_container_removed": True,
        }))
        return path, report

    def reject_before_signer(self):
        with self.assertRaises((ValueError, OSError, subprocess.CalledProcessError)):
            finalize.sign_candidate(self.validated, self.receipt_hash, self.output)
        self.signer.assert_not_called()
        self.assertFalse(self.output.exists())


class SigningPreconditionTests(CandidateFixture):
    def test_validated_candidate_produces_verifiable_portable_bundle(self):
        result = self.build_bundle()
        self.assertEqual(result["status"], "signed-awaiting-qualification")
        self.assertEqual(finalize.verify_bundle(self.output, self.manifest_hash), self.manifest)
        self.source_identity.assert_called_once_with(SOURCE)
        self.signer.assert_called_once()
        helper = (self.output / "nvbroadcast-native-upgrade").read_text()
        for package in self.manifest["packages"].values():
            self.assertIn(package["sha256"], helper)
        self.assertEqual(self.manifest["producer"]["run_attempt"], 2)

    def test_changed_receipt_rejected_before_signing(self):
        self.receipt.write_bytes(self.receipt.read_bytes() + b" ")
        self.reject_before_signer()

    def test_failed_validation_rejected_before_signing(self):
        self.report["status"] = "fail"
        self.save_receipt()
        self.reject_before_signer()

    def test_nonancestor_source_rejected_before_signing(self):
        self.source_identity.side_effect = subprocess.CalledProcessError(1, "git")
        self.reject_before_signer()

    def test_bad_producer_identities_rejected_before_signing(self):
        original = copy.deepcopy(self.report)
        for key, value in (("run_id", True), ("run_attempt", 0), ("artifact_id", -1),
                           ("artifact_digest", "sha256:" + "A" * 64)):
            with self.subTest(field=key):
                self.report = copy.deepcopy(original)
                self.report[key] = value
                self.save_receipt()
                self.reject_before_signer()

    def test_changed_package_rejected_before_signing(self):
        package = self.validated / self.report["packages"]["rpm"]["relative_path"]
        package.write_bytes(b"tampered")
        self.reject_before_signer()

    def test_package_symlink_rejected_before_signing(self):
        package = self.validated / self.report["packages"]["rpm"]["relative_path"]
        moved = self.root / package.name
        package.rename(moved)
        package.symlink_to(moved)
        self.reject_before_signer()

    def test_unsafe_package_relative_path_rejected_before_signing(self):
        original = self.report["packages"]["rpm"]["relative_path"]
        for name in ("../" + original, "/" + original, "rpm//" + Path(original).name,
                     "rpm/./" + Path(original).name, "rpm\\file.rpm"):
            with self.subTest(path=name):
                self.report["packages"]["rpm"]["relative_path"] = name
                self.save_receipt()
                self.reject_before_signer()

    def test_unverified_scriptlet_rejected_before_signing(self):
        self.report["packages"]["rpm"]["scriptlet_source_match"] = False
        self.save_receipt()
        self.reject_before_signer()

    def test_mixed_runtime_rejected_before_signing(self):
        self.report["packages"]["deb"]["runtime_manifest_sha256"] = "6" * 64
        self.save_receipt()
        self.reject_before_signer()

    def test_unexpected_package_name_rejected_before_signing(self):
        record = self.report["packages"]["rpm"]
        package = self.validated / record["relative_path"]
        package.rename(package.with_name("other.rpm"))
        record["relative_path"] = "rpm/other.rpm"
        self.save_receipt()
        self.reject_before_signer()

    def test_signer_identity_mismatch_rejected(self):
        def mismatched(*args):
            report = self.fake_sign(*args)
            report["fingerprint"] = "F" * 40
            return report
        self.signer.side_effect = mismatched
        with self.assertRaisesRegex(ValueError, "signer output"):
            self.build_bundle()

    def test_duplicate_json_key_rejected(self):
        self.receipt.write_bytes(b'{"status":"pass","status":"pass"}')
        self.receipt_hash = finalize.sha(self.receipt)
        self.reject_before_signer()

    def test_cpu_default_preserves_historical_receipt_without_variant(self):
        self.report.pop("variant")
        self.save_receipt()
        self.build_bundle()
        self.assertEqual(self.manifest["variant"], "cpu")
        self.assertEqual(finalize.verify_bundle(self.output, self.manifest_hash), self.manifest)

    def test_wrong_explicit_variant_rejected_before_signing(self):
        for variant in ("cuda", "other", None, True):
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    finalize.sign_candidate(self.validated, self.receipt_hash, self.output, variant=variant)
                self.signer.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_receipt_variant_cannot_disagree_with_cpu_version(self):
        for variant in ("cuda", None, True):
            with self.subTest(variant=variant):
                self.report["variant"] = variant
                self.save_receipt()
                self.reject_before_signer()


class BundleBindingTests(CandidateFixture):
    def setUp(self):
        super().setUp()
        self.build_bundle()

    def test_manifest_tamper_rejected(self):
        path = self.output / "candidate.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_each_bound_file_tamper_rejected(self):
        for name in self.manifest["files"]:
            with self.subTest(file=name):
                path = self.output / name
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    finalize.verify_bundle(self.output, self.manifest_hash)
                path.write_bytes(original)

    def test_extra_bundle_file_rejected(self):
        (self.output / "unexpected").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "extra or missing"):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_bundle_directory_symlink_rejected(self):
        directory = self.output / "rpm"
        moved = self.root / "moved-rpm"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValueError):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_checksum_tamper_rejected(self):
        (self.output / "SHA256SUMS").write_text("wrong\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_producer_mismatch_rejected_beneath_outer_digest(self):
        self.manifest["producer"]["run_attempt"] += 1
        self.rewrite_bundle()
        with self.assertRaisesRegex(ValueError, "producer"):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_missing_producer_binding_rejected(self):
        self.manifest["producer"] = {}
        self.rewrite_bundle()
        with self.assertRaises(ValueError):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_runtime_manifest_mismatch_rejected(self):
        self.manifest["runtime_manifest_sha256"] = "6" * 64
        self.rewrite_bundle()
        with self.assertRaises(ValueError):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_recovery_helper_mismatch_rejected(self):
        self.manifest["recovery_helper_sha256"] = "6" * 64
        self.rewrite_bundle()
        with self.assertRaises(ValueError):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_signing_receipt_mismatch_rejected_beneath_outer_digest(self):
        path = self.output / "rpm-signing.json"
        report = finalize.read_json(path)
        report["signed_sha256"] = "6" * 64
        path.write_bytes(finalize.json_bytes(report))
        self.rewrite_bundle()
        with self.assertRaisesRegex(ValueError, "signing"):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_explicit_cuda_does_not_accept_cpu_bundle(self):
        with self.assertRaises(ValueError):
            finalize.verify_bundle(self.output, self.manifest_hash, variant="cuda")

    def test_manifest_variant_change_rejected_beneath_outer_digest(self):
        for variant in ("cuda", None, True):
            with self.subTest(variant=variant):
                self.manifest["variant"] = variant
                self.rewrite_bundle()
                with self.assertRaises(ValueError):
                    finalize.verify_bundle(self.output, self.manifest_hash)

    def test_nested_validation_variant_change_rejected_beneath_outer_digest(self):
        path = self.output / "validation.json"
        original = finalize.read_json(path)
        for variant in ("cuda", None, True):
            with self.subTest(variant=variant):
                report = copy.deepcopy(original)
                report["variant"] = variant
                path.write_bytes(finalize.json_bytes(report))
                self.manifest["validation_receipt_sha256"] = finalize.sha(path)
                self.rewrite_bundle()
                with self.assertRaises(ValueError):
                    finalize.verify_bundle(self.output, self.manifest_hash)

    def test_historical_cpu_manifest_without_variant_remains_verifiable(self):
        self.manifest.pop("variant")
        self.rewrite_bundle()
        self.assertEqual(finalize.verify_bundle(self.output, self.manifest_hash), self.manifest)


class QualificationEvidenceTests(CandidateFixture):
    def setUp(self):
        super().setUp()
        self.build_bundle()

    def test_all_successful_steps_complete_candidate(self):
        deb, _ = self.execution("deb")
        rpm, _ = self.execution("rpm")
        self.execution("rpm", recovery=True)
        result = finalize.complete(self.output, self.manifest_hash, deb.parent.parent,
                                   rpm.parent.parent, self.root / "qualified")
        self.assertEqual(result["status"], "qualified-development-candidate")
        self.assertEqual({key: value["steps"] for key, value in result["qualification"].items()},
                         {"deb": 14, "rpm": 19, "recovery": 23})
        self.assertEqual(finalize.sha(self.root / "qualified/packages/candidate.json"), self.manifest_hash)

    def test_execution_identity_or_cleanup_mismatch_rejected(self):
        path, base = self.execution("rpm")
        changes = {"status": "fail", "source": "9" * 40, "artifact_sha256": "9" * 64,
                   "runtime_manifest_sha256": "9" * 64, "family": "deb", "version": "1.5.3-5.cpu",
                   "container_removed": False, "input_unchanged": False, "snapshot_unchanged": False}
        for key, value in changes.items():
            with self.subTest(field=key):
                report = copy.deepcopy(base)
                report[key] = value
                path.write_bytes(finalize.json_bytes(report))
                with self.assertRaises(ValueError):
                    finalize.check_execution(path, self.manifest, "rpm")

    def test_changed_or_missing_probe_preservation_never_completes(self):
        deb, _ = self.execution("deb")
        rpm, _ = self.execution("rpm")
        self.execution("rpm", True)
        target = self.root / "qualified"
        for path in (deb, rpm):
            original = finalize.read_json(path)
            for value in (False, None, 1, "true"):
                with self.subTest(family=original["family"], probes_unchanged=value):
                    changed = copy.deepcopy(original)
                    if value is None:
                        del changed["probes_unchanged"]
                    else:
                        changed["probes_unchanged"] = value
                    path.write_bytes(finalize.json_bytes(changed))
                    with self.assertRaises(ValueError):
                        finalize.complete(self.output, self.manifest_hash, deb.parent.parent,
                                          rpm.parent.parent, target)
                    self.assertFalse(target.exists())
                    path.write_bytes(finalize.json_bytes(original))

    def test_failed_or_unremoved_fixture_never_completes(self):
        deb, _ = self.execution("deb")
        rpm, _ = self.execution("rpm")
        self.execution("rpm", True)
        target = self.root / "qualified"
        for directory in (deb.parent.parent, rpm.parent.parent):
            path = directory / "fixture.json"
            original = finalize.read_json(path)
            for field, value in (("status", "fail"), ("status", None),
                                 ("inventory_container_removed", False),
                                 ("inventory_container_removed", None),
                                 ("inventory_container_removed", 1),
                                 ("inventory_container_removed", "true")):
                with self.subTest(family=original["family"], field=field, value=value):
                    changed = copy.deepcopy(original)
                    if value is None:
                        del changed[field]
                    else:
                        changed[field] = value
                    path.write_bytes(finalize.json_bytes(changed))
                    with self.assertRaises(ValueError):
                        finalize.complete(self.output, self.manifest_hash, deb.parent.parent,
                                          rpm.parent.parent, target)
                    self.assertFalse(target.exists())
                    path.write_bytes(finalize.json_bytes(original))

    def test_finalized_rpm_requires_authentication(self):
        path, report = self.execution("rpm")
        report["rpm_authentication"]["publisher_fingerprint"] = "9" * 40
        path.write_bytes(finalize.json_bytes(report))
        with self.assertRaises(ValueError):
            finalize.check_execution(path, self.manifest, "rpm")

    def test_positive_integrity_step_cannot_be_relabelled_expected_failure(self):
        for recovery in (False, True):
            with self.subTest(recovery=recovery):
                path, report = self.execution("rpm", recovery)
                step = next(step for step in report["steps"] if step["name"] == "complete-installed-integrity")
                step.update(exit_code=1, expected_failure=True)
                path.write_bytes(finalize.json_bytes(report))
                with self.assertRaises(ValueError):
                    finalize.check_execution(path, self.manifest, "rpm", recovery)

    def test_missing_required_step_rejected(self):
        path, report = self.execution("deb")
        report["steps"] = [s for s in report["steps"] if s["name"] != "ordinary-user-generated-recording"]
        path.write_bytes(finalize.json_bytes(report))
        with self.assertRaises(ValueError):
            finalize.check_execution(path, self.manifest, "deb")

    def test_duplicate_step_rejected(self):
        path, report = self.execution("deb")
        report["steps"].append(copy.deepcopy(report["steps"][-1]))
        path.write_bytes(finalize.json_bytes(report))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            finalize.check_execution(path, self.manifest, "deb")

    def test_changed_execution_log_rejected(self):
        path, report = self.execution("deb")
        (path.parent / report["steps"][0]["log"]).write_text("modified")
        with self.assertRaisesRegex(ValueError, "log differs"):
            finalize.check_execution(path, self.manifest, "deb")

    def test_log_symlink_rejected(self):
        path, report = self.execution("deb")
        log = path.parent / report["steps"][0]["log"]
        moved = self.root / "outside.log"
        log.rename(moved)
        log.symlink_to(moved)
        with self.assertRaises(ValueError):
            finalize.check_execution(path, self.manifest, "deb")

    def test_recording_or_runtime_failure_rejected(self):
        path, base = self.execution("deb")
        for field in ("recording", "runtime_probe"):
            with self.subTest(field=field):
                report = copy.deepcopy(base)
                report[field]["status"] = "fail"
                path.write_bytes(finalize.json_bytes(report))
                with self.assertRaises(ValueError):
                    finalize.check_execution(path, self.manifest, "deb")

    def test_recovery_requires_actual_fresh_interruption(self):
        path, base = self.execution("rpm", True)
        for field, value in (("signal", "SIGTERM"), ("fresh_install", False),
                             ("observed_bytes", 0), ("manager_exit", 0)):
            with self.subTest(field=field):
                report = copy.deepcopy(base)
                report["interruption"][field] = value
                path.write_bytes(finalize.json_bytes(report))
                with self.assertRaises(ValueError):
                    finalize.check_execution(path, self.manifest, "rpm", True)

    def test_interrupted_manager_log_tamper_rejected(self):
        path, _ = self.execution("rpm", True)
        (path.parent / "interrupted-manager.log").write_text("changed manager transcript")
        with self.assertRaises(ValueError):
            finalize.check_execution(path, self.manifest, "rpm", True)

    def test_preservation_bytes_tamper_rejected(self):
        path, _ = self.execution("rpm", True)
        preservation = path.parent / "preservation.json"
        preservation.write_bytes(preservation.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            finalize.check_execution(path, self.manifest, "rpm", True)

    def test_preservation_identity_mismatch_rejected_even_with_matching_digest(self):
        path, base = self.execution("rpm", True)
        preservation = path.parent / "preservation.json"
        original = finalize.read_json(preservation)
        for field, value in (("package_sha256", "9" * 64), ("version", "1.5.3-5.cpu"),
                             ("fingerprint", "9" * 40)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                preservation.write_bytes(finalize.json_bytes(changed))
                report = copy.deepcopy(base)
                report["preservation_sha256"] = finalize.sha(preservation)
                path.write_bytes(finalize.json_bytes(report))
                with self.assertRaises(ValueError):
                    finalize.check_execution(path, self.manifest, "rpm", True)

    def test_failed_evidence_never_creates_final_retention(self):
        deb, report = self.execution("deb")
        rpm, _ = self.execution("rpm")
        self.execution("rpm", True)
        report["status"] = "fail"
        deb.write_bytes(finalize.json_bytes(report))
        target = self.root / "qualified"
        with self.assertRaises(ValueError):
            finalize.complete(self.output, self.manifest_hash, deb.parent.parent, rpm.parent.parent, target)
        self.assertFalse(target.exists())

    def test_unlisted_evidence_is_not_retained(self):
        deb, _ = self.execution("deb")
        rpm, _ = self.execution("rpm")
        self.execution("rpm", True)
        (deb.parent.parent / "unrelated-credential.txt").write_text("fixture-only unrelated file")
        target = self.root / "qualified"
        with self.assertRaisesRegex(ValueError, "evidence file set"):
            finalize.complete(self.output, self.manifest_hash, deb.parent.parent, rpm.parent.parent, target)
        self.assertFalse(target.exists())

    def test_evidence_link_cannot_copy_external_file(self):
        path, _ = self.execution("deb")
        inventory = path.parent.parent / "fixture-packages.txt"
        moved = self.root / "outside-inventory"
        inventory.rename(moved)
        inventory.symlink_to(moved)
        with self.assertRaises(ValueError):
            finalize.evidence_files(path.parent.parent, "deb")

    def test_fixture_inventory_tamper_rejected(self):
        path, _ = self.execution("deb")
        (path.parent.parent / "fixture-packages.txt").write_text("changed inventory")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            finalize.evidence_files(path.parent.parent, "deb")

    def test_fixture_image_must_match_executed_image(self):
        deb, _ = self.execution("deb")
        rpm, _ = self.execution("rpm")
        self.execution("rpm", True)
        fixture = deb.parent.parent / "fixture.json"
        report = finalize.read_json(fixture)
        report["image"] = "sha256:" + "9" * 64
        fixture.write_bytes(finalize.json_bytes(report))
        target = self.root / "qualified"
        with self.assertRaisesRegex(ValueError, "fixture image differs"):
            finalize.complete(self.output, self.manifest_hash, deb.parent.parent, rpm.parent.parent, target)
        self.assertFalse(target.exists())


class CudaCandidateTests(CandidateFixture):
    variant = "cuda"

    def test_cuda_requires_explicit_choice_before_signing(self):
        with self.assertRaises(ValueError):
            finalize.sign_candidate(self.validated, self.receipt_hash, self.output)
        self.signer.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cuda_receipt_must_carry_matching_variant_before_signing(self):
        original = copy.deepcopy(self.report)
        for value in (None, "cpu", True, "cuda "):
            with self.subTest(value=value):
                self.report = copy.deepcopy(original)
                if value is None:
                    self.report.pop("variant")
                else:
                    self.report["variant"] = value
                self.save_receipt()
                with self.assertRaises(ValueError):
                    finalize.sign_candidate(self.validated, self.receipt_hash, self.output, variant="cuda")
                self.signer.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_explicit_cuda_sign_verify_complete_with_authenticated_variant(self):
        self.build_bundle()
        manifest = finalize.verify_bundle(self.output, self.manifest_hash, variant="cuda")
        self.assertEqual(manifest["variant"], "cuda")
        self.assertEqual(manifest["version"], self.version)
        deb, _ = self.execution("deb")
        rpm, _ = self.execution("rpm")
        self.execution("rpm", recovery=True)
        target = self.root / "qualified"
        with self.assertRaises(ValueError):
            finalize.complete(self.output, self.manifest_hash, deb.parent.parent, rpm.parent.parent, target)
        self.assertFalse(target.exists())
        result = finalize.complete(self.output, self.manifest_hash, deb.parent.parent,
                                   rpm.parent.parent, target, variant="cuda")
        self.assertEqual(result["variant"], "cuda")
        self.assertEqual(result["qualification"]["recovery"]["steps"], 24)
        self.assertIn("CPU execution only", result["scope"])
        self.assertEqual(finalize.sha(target / "packages/candidate.json"), self.manifest_hash)

    def test_cuda_bundle_cannot_silently_use_cpu_default(self):
        self.build_bundle()
        with self.assertRaises(ValueError):
            finalize.verify_bundle(self.output, self.manifest_hash)

    def test_cuda_bundle_variant_is_mandatory_beneath_outer_hash(self):
        self.build_bundle()
        original = copy.deepcopy(self.manifest)
        for value in (None, "cpu", True):
            with self.subTest(value=value):
                self.manifest = copy.deepcopy(original)
                if value is None:
                    self.manifest.pop("variant")
                else:
                    self.manifest["variant"] = value
                self.rewrite_bundle()
                with self.assertRaises(ValueError):
                    finalize.verify_bundle(self.output, self.manifest_hash, variant="cuda")

    def test_cuda_cannot_rebind_cpu_validation(self):
        self.build_bundle()
        path = self.output / "validation.json"
        report = finalize.read_json(path)
        report["variant"] = "cpu"
        path.write_bytes(finalize.json_bytes(report))
        self.manifest["validation_receipt_sha256"] = finalize.sha(path)
        self.rewrite_bundle()
        with self.assertRaises(ValueError):
            finalize.verify_bundle(self.output, self.manifest_hash, variant="cuda")

    def test_cuda_smoke_and_recovery_require_runtime_variant_and_no_gpu(self):
        self.build_bundle()
        for recovery in (False, True):
            path, original = self.execution("rpm", recovery=recovery)
            changes = {
                "variant": "cpu", "runtime_probe": {"status": "pass"},
                "container_runtime": "nvidia", "gpu_device": "GPU-test",
                "nvidia_visible_devices": "all",
            }
            for field, value in changes.items():
                with self.subTest(recovery=recovery, field=field):
                    report = copy.deepcopy(original)
                    report[field] = value
                    path.write_bytes(finalize.json_bytes(report))
                    with self.assertRaises(ValueError):
                        finalize.check_execution(path, self.manifest, "rpm", recovery=recovery, variant="cuda")
            for field in ("variant", "runtime_probe", "gpu_device"):
                with self.subTest(recovery=recovery, missing=field):
                    report = copy.deepcopy(original)
                    report.pop(field)
                    path.write_bytes(finalize.json_bytes(report))
                    with self.assertRaises(ValueError):
                        finalize.check_execution(path, self.manifest, "rpm", recovery=recovery, variant="cuda")

    def test_cuda_runtime_cannot_claim_gpu_execution_or_false_cpu_success(self):
        self.build_bundle()
        path, original = self.execution("deb")
        mutations = (
            lambda p: p.update(variant="cpu"),
            lambda p: p.update(cuda_unavailable_expected=False),
            lambda p: p["cpu_execution"].update(success=False),
            lambda p: p["cpu_execution"].update(executed_providers=["CUDAExecutionProvider"]),
            lambda p: p["cuda_execution"].update(success=True),
            lambda p: p["cuda_execution"].update(executed_providers=["CUDAExecutionProvider"]),
            lambda p: p.update(cupy_execution={"device": "unexpected GPU"}),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                report = copy.deepcopy(original)
                mutate(report["runtime_probe"])
                path.write_bytes(finalize.json_bytes(report))
                with self.assertRaises(ValueError):
                    finalize.check_execution(path, self.manifest, "deb", variant="cuda")

    def test_cuda_recovery_requires_its_extra_step(self):
        self.build_bundle()
        path, report = self.execution("rpm", recovery=True)
        report["steps"] = [s for s in report["steps"] if s["name"] != "recovered-runtime-cpu-fallback"]
        path.write_bytes(finalize.json_bytes(report))
        with self.assertRaisesRegex(ValueError, "qualification steps"):
            finalize.check_execution(path, self.manifest, "rpm", recovery=True, variant="cuda")

    def test_signed_cuda_at_release_ceiling_never_creates_candidate_manifest(self):
        def oversized(*args):
            report = self.fake_sign(*args)
            with (args[-1] / args[0].name).open("r+b") as stream:
                stream.truncate(2 * 1024**3)
            return report
        self.signer.side_effect = oversized
        with self.assertRaisesRegex(ValueError, "signed package.*size bound"):
            self.build_bundle()
        self.assertFalse((self.output / "candidate.json").exists())


class VariantPackageBoundsTests(CandidateFixture):
    def test_package_size_boundaries_are_variant_specific_without_reading_sparse_data(self):
        path = self.validated / "sparse.rpm"
        for variant, allowed in (("cpu", 1024**3), ("cuda", 2 * 1024**3 - 1)):
            maximum = finalize.policy.policy_for(variant).package_max_bytes
            self.assertEqual(maximum, allowed)
            for size, accepted in ((allowed, True), (allowed + 1, False)):
                with self.subTest(variant=variant, size=size):
                    with path.open("wb") as stream:
                        stream.truncate(size)
                    record = {"bytes": size, "sha256": "8" * 64}
                    with mock.patch.object(finalize, "sha", return_value="8" * 64) as hashed:
                        if accepted:
                            self.assertEqual(finalize.verified_file(self.validated, path.name, record,
                                                                   maximum=maximum), path)
                            hashed.assert_called_once_with(path)
                        else:
                            with self.assertRaisesRegex(ValueError, "file size"):
                                finalize.verified_file(self.validated, path.name, record, maximum=maximum)
                            hashed.assert_not_called()

    def test_signed_growth_beyond_cpu_bound_never_creates_candidate_manifest(self):
        def oversized(*args):
            report = self.fake_sign(*args)
            signed = args[-1] / args[0].name
            with signed.open("r+b") as stream:
                stream.truncate(1024**3 + 1)
            return report
        self.signer.side_effect = oversized
        with self.assertRaisesRegex(ValueError, "signed package.*size bound"):
            self.build_bundle()
        self.assertFalse((self.output / "candidate.json").exists())


class VariantCommandLineTests(unittest.TestCase):
    def test_cli_defaults_cpu_and_forwards_explicit_cuda_through_every_stage(self):
        commands = {
            "sign": ("sign_candidate", ["--validated", "/validated", "--expected-receipt", "1" * 64,
                                         "--output", "/signed"]),
            "verify": ("verify_bundle", ["--bundle", "/signed", "--expected-manifest", "2" * 64]),
            "complete": ("complete", ["--bundle", "/signed", "--expected-manifest", "2" * 64,
                                      "--deb-evidence", "/deb", "--rpm-evidence", "/rpm",
                                      "--output", "/qualified"]),
        }
        for command, (function, arguments) in commands.items():
            for variant, option in (("cpu", []), ("cuda", ["--variant", "cuda"])):
                with self.subTest(command=command, variant=variant):
                    with mock.patch.object(sys, "argv", ["finalize", command, *arguments, *option]), \
                            mock.patch.object(finalize, function, return_value={}) as target, \
                            mock.patch("builtins.print"):
                        finalize.main()
                    self.assertEqual(target.call_args.kwargs, {"variant": variant})


if __name__ == "__main__":
    unittest.main()

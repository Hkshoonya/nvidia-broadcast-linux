"""Adversarial OCI transport and authentication boundaries; no Docker or secrets."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("builder_delivery", ROOT / "packaging/native-runtime/builder_delivery.py")
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


class BuilderDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def fixture(self, fault=None):
        entries = {}

        def blob(value, media):
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            digest = hashlib.sha256(data).hexdigest()
            entries["blobs/sha256/" + digest] = data
            return {"mediaType": media, "size": len(data), "digest": "sha256:" + digest}

        roots, legacy, pins = [], [], {}
        for role in ("bindings", "deb", "rpm"):
            config = blob({"architecture": "amd64", "os": "linux", "role": role}, delivery.CONFIG)
            layer = blob(role.encode(), "application/vnd.oci.image.layer.v1.tar")
            if fault == "external" and role == "rpm":
                layer["urls"] = ["https://example.invalid/layer"]
            image = blob({"schemaVersion": 2, "mediaType": delivery.MANIFEST,
                          "config": config, "layers": [layer]}, delivery.MANIFEST)
            image["platform"] = {"architecture": "amd64", "os": "linux"}
            manifests = [image]
            if fault in {"inline-empty", "inline-replacement"}:
                empty = blob(b"{}", "application/vnd.oci.empty.v1+json")
                empty["data"] = "e30=" if fault == "inline-empty" else "e10="
                statement = blob({"role": role}, "application/vnd.in-toto+json")
                attestation = blob({"schemaVersion": 2, "mediaType": delivery.MANIFEST,
                                    "config": empty, "layers": [statement]}, delivery.MANIFEST)
                attestation["platform"] = {"architecture": "unknown", "os": "unknown"}
                manifests.append(attestation)
            index = blob({"schemaVersion": 2, "mediaType": delivery.INDEX, "manifests": manifests}, delivery.INDEX)
            roots.append(index)
            pins[role] = {"image": index["digest"], "packages": [role + " 1 amd64"]}
            legacy.append({"Config": "blobs/sha256/" + config["digest"][7:], "RepoTags": None,
                           "Layers": ["blobs/sha256/" + layer["digest"][7:]]})
        if fault == "descriptor-size":
            roots[0]["size"] += 1
        if fault == "wrong-root":
            roots[0]["digest"] = "sha256:" + "0" * 64
        if fault == "docker-mapping":
            legacy[0]["Layers"] = legacy[1]["Layers"]
        if fault == "docker-tag":
            legacy[0]["RepoTags"] = ["unexpected:latest"]
        if fault == "blob-hash":
            entries[next(iter(entries))] += b"tampered"
        if fault == "unused-blob":
            blob(b"not referenced", "application/vnd.in-toto+json")
        entries["index.json"] = json.dumps({"schemaVersion": 2, "mediaType": delivery.INDEX,
                                             "manifests": roots}).encode()
        entries["manifest.json"] = json.dumps(legacy).encode()
        entries["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
        if fault == "duplicate-json":
            entries["oci-layout"] = b'{"imageLayoutVersion":"1.0.0","imageLayoutVersion":"1.0.0"}'
        if fault == "escape":
            entries["../unexpected"] = b"unsafe"
        path = self.directory / "builders.tar"
        with tarfile.open(path, "w") as archive:
            for name, data in entries.items():
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
            if fault == "duplicate-member":
                archive.addfile(item, io.BytesIO(data))
            if fault == "symlink":
                link = tarfile.TarInfo("blobs/sha256/" + "1" * 64)
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                archive.addfile(link)
        lock = {"schema": 1, "status": "candidate", "attestation": None,
                "source": {"repository": delivery.REPOSITORY, "asset_id": 42},
                "archive": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}}
        return path, pins, lock

    def test_valid_candidate_only_claims_integrity(self):
        path, pins, lock = self.fixture()
        result = delivery.process_archive("inspect", path, lock, pins)
        self.assertEqual(result["status"], "integrity_verified_not_authenticated")
        self.assertFalse(result["original_build_provenance_authenticated"])
        self.assertEqual(result["blob_count"], 12)

    def test_rehashed_hostile_archives_are_rejected(self):
        for fault in ("external", "descriptor-size", "wrong-root", "docker-mapping", "docker-tag",
                      "blob-hash", "unused-blob", "duplicate-json", "escape", "duplicate-member", "symlink", "inline-replacement"):
            with self.subTest(fault=fault):
                path, pins, lock = self.fixture(fault)
                with self.assertRaises(ValueError):
                    delivery.process_archive("inspect", path, lock, pins)

    def test_buildkit_empty_config_is_checked_without_trusting_unsigned_provenance(self):
        path, pins, lock = self.fixture("inline-empty")
        result = delivery.process_archive("inspect", path, lock, pins)
        self.assertFalse(result["original_build_provenance_authenticated"])
        self.assertEqual(result["blob_count"], 19)

    def test_transport_hash_size_and_symlink_are_rejected(self):
        path, pins, lock = self.fixture()
        lock["archive"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "archive hash differs"):
            delivery.process_archive("inspect", path, lock, pins)
        lock["archive"]["bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            delivery.process_archive("inspect", path, lock, pins)
        link = self.directory / "redirect"
        link.symlink_to(path)
        with self.assertRaises(OSError):
            delivery.process_archive("inspect", link, lock, pins)
        fifo = self.directory / "fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular input"):
            delivery.process_archive("inspect", fifo, lock, pins)

    def test_candidate_cannot_be_loaded_as_authenticated(self):
        path, pins, lock = self.fixture()
        with mock.patch.object(delivery.subprocess, "run") as run, mock.patch.object(delivery, "qualify_local_images") as qualify:
            with self.assertRaisesRegex(ValueError, "not been authenticated/promoted"):
                delivery.process_archive("verify", path, lock, pins, load=True)
            run.assert_not_called()
            qualify.assert_not_called()

    def promoted(self, lock):
        lock["status"] = "promoted"
        lock["attestation"] = {"workflow": delivery.SIGNER, "predicate_type": delivery.PREDICATE,
                               "source_commit": "a" * 40}
        return lock

    def test_crypto_failure_prevents_docker_load_and_inventory_execution(self):
        path, pins, lock = self.fixture()
        with mock.patch.object(delivery.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "gh")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                delivery.process_archive("verify", path, self.promoted(lock), pins, load=True)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["gh", "attestation", "verify"])
            for flag in ("--source-digest", "--signer-digest", "--source-ref", "--signer-workflow",
                         "--predicate-type", "--deny-self-hosted-runners", "--hostname"):
                self.assertIn(flag, command)

    def test_unsupported_attestation_authority_is_refused(self):
        path, pins, lock = self.fixture()
        self.promoted(lock)["attestation"]["workflow"] = "attacker/repo/.github/workflows/build.yml"
        with mock.patch.object(delivery.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "unreviewed attestation authority"):
                delivery.process_archive("verify", path, lock, pins, load=True)
            run.assert_not_called()

    def test_load_consumes_private_verified_bytes_if_original_path_is_replaced(self):
        path, pins, lock = self.fixture()
        original = path.read_bytes()
        consumed = []

        def run(command, **kwargs):
            private = Path(command[3] if command[0] == "gh" else command[-1])
            self.assertNotEqual(private, path)
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            self.assertEqual(private.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(private.read_bytes(), original)
            consumed.append(command[0])
            path.write_bytes(b"replaced after hashing")

        with mock.patch.object(delivery.subprocess, "run", side_effect=run), mock.patch.object(delivery, "qualify_local_images", return_value={}):
            delivery.process_archive("verify", path, self.promoted(lock), pins, load=True)
        self.assertEqual(consumed, ["gh", "docker"])

    def test_classic_docker_store_is_rejected_before_inspect_or_execution(self):
        _, pins, _ = self.fixture()
        with mock.patch.object(delivery.subprocess, "check_output", return_value='[["Backing Filesystem","extfs"]]') as query:
            with self.assertRaisesRegex(ValueError, "containerd image store"):
                delivery.qualify_local_images(pins)
            self.assertEqual(query.call_count, 1)

    def test_inventory_drift_is_rejected_with_network_and_privileges_disabled(self):
        _, pins, _ = self.fixture()
        role = pins["bindings"]
        responses = [json.dumps([["driver-type", "io.containerd.snapshotter.v1"]]),
                     json.dumps([{"Id": role["image"], "Os": "linux", "Architecture": "amd64"}]), "unreviewed 2 amd64\n", ""]
        with mock.patch.object(delivery.subprocess, "check_output", side_effect=responses) as query, mock.patch.object(delivery.subprocess, "run"):
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                delivery.qualify_local_images(pins)
            command = query.call_args_list[2].args[0]
            for boundary in ("--network=none", "--pull=never", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges"):
                self.assertIn(boundary, command)

    def test_timed_out_inventory_removes_only_its_named_container(self):
        _, pins, _ = self.fixture()
        role = pins["bindings"]
        responses = [json.dumps([["driver-type", "io.containerd.snapshotter.v1"]]),
                     json.dumps([{"Id": role["image"], "Os": "linux", "Architecture": "amd64"}]),
                     subprocess.TimeoutExpired("docker", 120), ""]
        with mock.patch.object(delivery.subprocess, "check_output", side_effect=responses) as query, \
                mock.patch.object(delivery.subprocess, "run") as cleanup:
            with self.assertRaises(subprocess.TimeoutExpired):
                delivery.qualify_local_images(pins)
        command = query.call_args_list[2].args[0]
        name = command[command.index("--name") + 1]
        self.assertTrue(name.startswith("nvbroadcast-builder-inventory-"))
        self.assertEqual(cleanup.call_args.args[0], ["docker", "rm", "--force", name])
        self.assertEqual(query.call_args.args[0][-1], "name=^/" + name + "$")

    def test_wrong_release_asset_is_rejected_before_download(self):
        lock = json.loads((ROOT / "packaging/native-runtime/builder-delivery.json").read_text())
        lock["source"]["asset_id"] = 99
        release = {"tag_name": lock["source"]["tag"], "prerelease": True, "draft": False,
                   "assets": [{"name": lock["source"]["asset"], "id": 12, "size": lock["archive"]["bytes"],
                               "state": "uploaded", "digest": "sha256:" + lock["archive"]["sha256"]}]}
        with mock.patch.object(delivery.subprocess, "check_output", return_value=json.dumps(release)), mock.patch.object(delivery.subprocess, "run") as download:
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                delivery.fetch_candidate(self.directory / "download.tar", lock, 99)
            download.assert_not_called()

    def qualification(self, pins, report):
        run = {"id": 7, "status": "completed", "conclusion": "success", "event": "workflow_dispatch",
               "head_branch": "main", "head_sha": "a" * 40, "run_attempt": 2,
               "repository": {"full_name": delivery.REPOSITORY}, "head_repository": {"full_name": delivery.REPOSITORY},
               "path": ".github/workflows/native-builders-qualify.yml"}
        inventories = {role: {"image": item["image"], "package_count": len(item["packages"])} for role, item in pins.items()}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("archive-integrity.json", json.dumps(report))
            archive.writestr("qualification.json", json.dumps(dict(report, installed_inventory=inventories)))
        archive = buffer.getvalue()
        artifacts = {"total_count": 1, "artifacts": [{"id": 8, "name": "historical-native-builder-qualification-7-2",
                     "size_in_bytes": len(archive), "digest": "sha256:" + hashlib.sha256(archive).hexdigest(),
                     "expired": False, "workflow_run": {"id": 7, "head_sha": "a" * 40}}]}
        return run, artifacts, archive

    def test_import_claim_requires_same_source_success_and_preserves_historical_scope(self):
        path, pins, lock = self.fixture()
        report = delivery.process_archive("inspect", path, lock, pins)
        run, artifacts, archive = self.qualification(pins, report)
        with mock.patch.object(delivery.subprocess, "check_output", side_effect=[json.dumps(run), json.dumps(artifacts)]), \
                mock.patch.object(delivery.subprocess, "run", side_effect=lambda command, **kwargs: kwargs["stdout"].write(archive)) as fetch:
            predicate = delivery.import_predicate(lock, pins, report, 7, "a" * 40)
        self.assertEqual(predicate["claim"], "historical-builder-import")
        self.assertFalse(predicate["original_build_provenance_authenticated"])
        self.assertEqual(predicate["qualification"]["run_attempt"], 2)
        self.assertEqual(predicate["qualification"]["artifact_id"], 8)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0][0], "gh")

    def test_failed_other_branch_fork_or_different_source_qualification_cannot_promote(self):
        path, pins, lock = self.fixture()
        report = delivery.process_archive("inspect", path, lock, pins)
        for key, value in (("conclusion", "failure"), ("head_branch", "topic"), ("head_sha", "b" * 40),
                           ("event", "pull_request"), ("head_repository", {"full_name": "fork/repo"}),
                           ("path", ".github/workflows/unreviewed.yml")):
            with self.subTest(key=key):
                run, _, _ = self.qualification(pins, report)
                run[key] = value
                with mock.patch.object(delivery.subprocess, "check_output", return_value=json.dumps(run)) as api, \
                        mock.patch.object(delivery.subprocess, "run") as fetch:
                    with self.assertRaisesRegex(ValueError, "successful approved-main"):
                        delivery.import_predicate(lock, pins, report, 7, "a" * 40)
                    self.assertEqual(api.call_count, 1)
                    fetch.assert_not_called()

    def test_stale_attempt_or_tampered_evidence_cannot_promote(self):
        path, pins, lock = self.fixture()
        report = delivery.process_archive("inspect", path, lock, pins)
        run, artifacts, archive = self.qualification(pins, report)
        artifacts["artifacts"][0]["name"] = "historical-native-builder-qualification-7-1"
        with mock.patch.object(delivery.subprocess, "check_output", side_effect=[json.dumps(run), json.dumps(artifacts)]), \
                mock.patch.object(delivery.subprocess, "run") as fetch:
            with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                delivery.import_predicate(lock, pins, report, 7, "a" * 40)
            fetch.assert_not_called()
        run, artifacts, archive = self.qualification(pins, report)
        with mock.patch.object(delivery.subprocess, "check_output", side_effect=[json.dumps(run), json.dumps(artifacts)]), \
                mock.patch.object(delivery.subprocess, "run", side_effect=lambda command, **kwargs: kwargs["stdout"].write(archive[:-1])):
            with self.assertRaisesRegex(ValueError, "transport digest differs"):
                delivery.import_predicate(lock, pins, report, 7, "a" * 40)

    def test_different_inventory_report_is_rejected_even_with_valid_transport_hash(self):
        path, pins, lock = self.fixture()
        report = delivery.process_archive("inspect", path, lock, pins)
        wrong = {role: dict(entry, packages=entry["packages"] + ["unexpected 1 amd64"]) for role, entry in pins.items()}
        run, artifacts, archive = self.qualification(wrong, report)
        with mock.patch.object(delivery.subprocess, "check_output", side_effect=[json.dumps(run), json.dumps(artifacts)]), \
                mock.patch.object(delivery.subprocess, "run", side_effect=lambda command, **kwargs: kwargs["stdout"].write(archive)):
            with self.assertRaisesRegex(ValueError, "qualification inventories differ"):
                delivery.import_predicate(lock, pins, report, 7, "a" * 40)


if __name__ == "__main__":
    unittest.main()

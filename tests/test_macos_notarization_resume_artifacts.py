"""Verify resume-artifact provenance without network, Apple or package execution."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


_SPEC = importlib.util.spec_from_file_location(
    "prepare_macos_notarization_resume",
    Path(__file__).parents[1] / "scripts/prepare_macos_notarization_resume.py",
)
preparation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preparation)

REPO = "ExampleOwner/nvidia-broadcast-linux"
RUN_ID = 123456
ATTEMPT = 4
REPOSITORY_ID = 789
SOURCE_SHA = "a" * 40
UNSIGNED = b"reviewed original unsigned installer"
SIGNED = UNSIGNED + b".signed-upload"
SOURCE_HASH = hashlib.sha256(UNSIGNED).hexdigest()
SIGNED_HASH = hashlib.sha256(SIGNED).hexdigest()
SUBMISSION_ID = "7c8aaf20-6e8a-4bbf-aaf1-4a6af19c5633"


def archive(entries):
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name, data, mode in entries:
                info = zipfile.ZipInfo(name)
                # ZipInfo normally truncates a NUL before writing; retain it
                # here to model an archive carrying an actual ambiguous name.
                info.filename = name
                info.create_system = 3
                info.external_attr = mode << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(info, data)
    return output.getvalue()


class _GitHub:
    def __init__(self):
        repository = {"id": REPOSITORY_ID, "full_name": REPO}
        self.run = {
            "id": RUN_ID, "run_attempt": ATTEMPT, "repository": repository,
            "head_repository": dict(repository), "head_sha": SOURCE_SHA,
            "path": preparation.WORKFLOW_PATH, "event": "workflow_dispatch",
        }
        self.manifest = {
            "schema_version": 1, "state": "pending", "submission_id": SUBMISSION_ID,
            "notarization_status": "In Progress", "source_sha256": SOURCE_HASH,
            "source_bytes": len(UNSIGNED), "signed_upload_sha256": SIGNED_HASH,
            "signed_upload_bytes": len(SIGNED), "expected_team_id": "ABC123DE45",
            "signer": "Developer ID Installer: Example Developer (ABC123DE45)",
            "identity": "0" * 40,
        }
        self.archives = {}
        self.artifacts = [self.artifact(101, "macos-packages"), self.artifact(
            102, f"macos-notarization-checkpoint-attempt-{ATTEMPT}")]
        self.set_unsigned()
        self.set_checkpoint()
        self.calls = []
        self.fail_metadata = False

    def artifact(self, identifier, name):
        return {
            "id": identifier, "name": name, "expired": False,
            "workflow_run": {"id": RUN_ID, "repository_id": REPOSITORY_ID,
                             "head_repository_id": REPOSITORY_ID, "head_sha": SOURCE_SHA},
        }

    def set_archive(self, identifier, body):
        self.archives[identifier] = body
        metadata = next(item for item in self.artifacts if item["id"] == identifier)
        metadata.update(size_in_bytes=len(body), digest="sha256:" + hashlib.sha256(body).hexdigest())

    def set_unsigned(self, entries=None):
        self.set_archive(101, archive(entries if entries is not None else [
            ("NVBroadcast-1.5.3-1.pkg", UNSIGNED, stat.S_IFREG | 0o644)]))

    def set_checkpoint(self, entries=None):
        self.set_archive(102, archive(entries if entries is not None else [
            ("checkpoint.json", json.dumps(self.manifest).encode(), stat.S_IFREG | 0o600),
            ("signed-upload.pkg", SIGNED, stat.S_IFREG | 0o600)]))

    def __call__(self, arguments, **kwargs):
        self.calls.append(list(arguments))
        if arguments[:4] != ["gh", "api", "--method", "GET"]:
            raise AssertionError("Only read-only gh API calls are allowed")
        endpoint = arguments[4]
        if "stdout" in kwargs:
            identifier = int(endpoint.rsplit("/", 2)[1])
            kwargs["stdout"].write(self.archives[identifier])
            return subprocess.CompletedProcess(arguments, 0, None, b"")
        if self.fail_metadata:
            return subprocess.CompletedProcess(arguments, 1, "", "secret-never-output")
        if endpoint.endswith(f"/runs/{RUN_ID}/attempts/{ATTEMPT}"):
            body = self.run
        elif f"/runs/{RUN_ID}/artifacts?per_page=100&page=" in endpoint:
            page = int(endpoint.rsplit("=", 1)[1])
            body = {"total_count": len(self.artifacts), "artifacts": self.artifacts[(page - 1) * 100:page * 100]}
        else:
            raise AssertionError(f"Unexpected API endpoint: {endpoint}")
        return subprocess.CompletedProcess(arguments, 0, json.dumps(body), "")


class MacOSNotarizationResumeArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="macOS resume artifact tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "prepared"
        self.github = _GitHub()
        patch = mock.patch.object(preparation.subprocess, "run", side_effect=self.github)
        patch.start()
        self.addCleanup(patch.stop)

    def prepare(self, **overrides):
        arguments = dict(repo=REPO, run_id=RUN_ID, attempt=ATTEMPT,
                         expected_source_sha=SOURCE_SHA, source_sha256=SOURCE_HASH,
                         signed_sha256=SIGNED_HASH, output_dir=self.output)
        arguments.update(overrides)
        return preparation.prepare_resume(**arguments)

    def assert_failure(self, pattern=None, **overrides):
        context = (self.assertRaisesRegex((preparation.PreparationError, OSError), pattern)
                   if pattern else self.assertRaises((preparation.PreparationError, OSError)))
        with context:
            self.prepare(**overrides)
        self.assertFalse((self.output / "provenance.json").exists())

    def reset_output(self):
        self.output = self.root / f"prepared-{len(list(self.root.iterdir()))}"

    def test_success_keeps_original_unsigned_and_signed_upload_with_verified_receipts(self):
        result = self.prepare()
        self.assertEqual(Path(result["source"]).read_bytes(), UNSIGNED)
        checkpoint = Path(result["checkpoint_directory"])
        self.assertEqual((checkpoint / "signed-upload.pkg").read_bytes(), SIGNED)
        self.assertEqual(result["source_sha"], SOURCE_SHA)
        self.assertEqual(result["source_run_id"], RUN_ID)
        self.assertEqual(result["source_attempt"], ATTEMPT)
        self.assertEqual(result["submission_id"], SUBMISSION_ID)
        self.assertEqual(result["unsigned_artifact"]["id"], 101)
        self.assertEqual(result["checkpoint_artifact"]["id"], 102)
        self.assertEqual(json.loads((self.output / "provenance.json").read_text()), result)
        for label, identifier in (("unsigned", 101), ("checkpoint", 102)):
            self.assertEqual((self.output / f"{label}-artifact.zip").read_bytes(), self.github.archives[identifier])
            self.assertEqual(json.loads((self.output / f"{label}-artifact.json").read_text())["id"], identifier)
        self.assertEqual(json.loads((self.output / "source-run.json").read_text()), self.github.run)
        for path in [self.output, *self.output.rglob("*")]:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)
        self.assertEqual(len(self.github.calls), 4)

    def test_push_run_can_prepare_an_accepted_checkpoint(self):
        self.github.run["event"] = "push"
        self.github.manifest.update(state="accepted", notarization_status="Accepted")
        self.github.set_checkpoint()
        self.assertEqual(self.prepare()["submission_id"], SUBMISSION_ID)

    def test_run_repository_workflow_event_attempt_and_exact_commit_must_match(self):
        original = copy.deepcopy(self.github.run)
        changes = [
            {"id": RUN_ID + 1}, {"run_attempt": ATTEMPT + 1},
            {"path": ".github/workflows/other.yml"}, {"event": "pull_request"},
            {"head_sha": "b" * 40}, {"head_sha": "not-a-commit"},
            {"repository": {"id": REPOSITORY_ID, "full_name": "OtherOwner/other"}},
            {"head_repository": {"id": REPOSITORY_ID + 1, "full_name": REPO}},
            {"head_repository": {"id": REPOSITORY_ID, "full_name": "OtherOwner/other"}},
            {"repository": None}, {"head_repository": None},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.github.run = {**copy.deepcopy(original), **changed}
                self.github.calls.clear()
                self.assert_failure()
                self.assertEqual(len(self.github.calls), 1)
                self.reset_output()

    def test_required_artifacts_must_have_unique_names_and_exact_run_bindings(self):
        original = copy.deepcopy(self.github.artifacts)
        changes = [
            {"workflow_run": {**original[1]["workflow_run"], "id": RUN_ID + 1}},
            {"workflow_run": {**original[1]["workflow_run"], "repository_id": REPOSITORY_ID + 1}},
            {"workflow_run": {**original[1]["workflow_run"], "head_repository_id": REPOSITORY_ID + 1}},
            {"workflow_run": {**original[1]["workflow_run"], "head_sha": "b" * 40}},
            {"workflow_run": None}, {"expired": True}, {"expired": None},
            {"digest": None}, {"digest": "sha256:invalid"}, {"size_in_bytes": 0},
            {"size_in_bytes": preparation.MAX_ARCHIVE_BYTES + 1},
            {"name": "macos-notarization-checkpoint-attempt-3"},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.github.artifacts = [copy.deepcopy(original[0]), {**copy.deepcopy(original[1]), **changed}]
                self.github.calls.clear()
                self.assert_failure()
                self.assertFalse(any(call[-1].endswith("/zip") for call in self.github.calls))
                self.reset_output()
        self.github.artifacts = copy.deepcopy(original) + [{**copy.deepcopy(original[1]), "id": 103}]
        self.assert_failure("exactly one retained artifact")

    def test_downloaded_zip_digest_and_size_must_match_metadata_before_extraction(self):
        for label in ("digest", "size_in_bytes"):
            with self.subTest(label=label):
                original = self.github.artifacts[0][label]
                self.github.artifacts[0][label] = "sha256:" + "b" * 64 if label == "digest" else original + 1
                self.assert_failure("ZIP SHA-256" if label == "digest" else "ZIP size")
                self.assertFalse((self.output / "unsigned").exists())
                self.github.artifacts[0][label] = original
                self.reset_output()

    def test_original_unsigned_and_signed_pins_are_independent_of_manifest_values(self):
        for field in ("source_sha256", "signed_sha256"):
            with self.subTest(field=field):
                self.assert_failure("independent pin", **{field: "b" * 64})
                self.reset_output()
        for field in ("source_sha256", "signed_upload_sha256", "source_bytes", "signed_upload_bytes"):
            with self.subTest(field=field):
                original = self.github.manifest[field]
                self.github.manifest[field] = "b" * 64 if field.endswith("sha256") else original + 1
                self.github.set_checkpoint()
                self.assert_failure()
                self.github.manifest[field] = original
                self.reset_output()

    def test_github_integrity_does_not_replace_the_unsigned_or_signed_file_pin(self):
        self.github.set_unsigned([("NVBroadcast-1.5.3-1.pkg", b"different unsigned bytes", stat.S_IFREG | 0o644)])
        self.assert_failure("Original unsigned package SHA-256")
        self.reset_output()
        self.github.set_unsigned()
        self.github.set_checkpoint([
            ("checkpoint.json", json.dumps(self.github.manifest).encode(), stat.S_IFREG | 0o600),
            ("signed-upload.pkg", b"different signed bytes", stat.S_IFREG | 0o600),
        ])
        self.assert_failure("Signed upload SHA-256")

    def test_invalid_zip_and_bad_member_crc_fail_without_a_success_receipt(self):
        self.github.set_archive(101, b"not an archive")
        self.assert_failure("valid supported ZIP")
        self.reset_output()
        self.github.set_unsigned()
        damaged = bytearray(self.github.archives[101])
        central = damaged.index(b"PK\x01\x02")
        # Change the central directory's CRC while retaining a valid outer ZIP
        # hash, so extraction rather than the transport check must reject it.
        damaged[central + 16] ^= 1
        self.github.set_archive(101, bytes(damaged))
        self.assert_failure("valid supported ZIP")

    def test_checkpoint_must_have_supported_schema_and_a_resumable_submission(self):
        for field, value in [("schema_version", 2), ("schema_version", True),
                             ("state", "prepared"), ("state", "rejected"),
                             ("submission_id", None), ("submission_id", "not-a-uuid"),
                             ("submission_id", "0" * 32), ("submission_id", SUBMISSION_ID.upper())]:
            with self.subTest(field=field, value=value):
                original = self.github.manifest[field]
                self.github.manifest[field] = value
                self.github.set_checkpoint()
                self.assert_failure()
                self.github.manifest[field] = original
                self.reset_output()

    def test_unsigned_zip_rejects_traversal_absolute_duplicate_link_and_special_files(self):
        regular = stat.S_IFREG | 0o644
        cases = [
            [("../escape.pkg", UNSIGNED, regular)], [("/escape.pkg", UNSIGNED, regular)],
            [("subdir/installer.pkg", UNSIGNED, regular)], [("subdir\\installer.pkg", UNSIGNED, regular)],
            [("installer.pkg", b"../../escape", stat.S_IFLNK | 0o777)],
            [("installer.pkg", UNSIGNED, stat.S_IFIFO | 0o644)],
            [("installer.pkg", UNSIGNED, regular), ("installer.pkg", UNSIGNED, regular)],
            [("installer.pkg", UNSIGNED, regular), ("extra.pkg", UNSIGNED, regular)],
            [("installer.pkg", UNSIGNED, regular), ("directory/", b"", stat.S_IFDIR | 0o755)],
            [("installer.pkg\x00ignored", UNSIGNED, regular)],
        ]
        for entries in cases:
            with self.subTest(entries=[entry[0] for entry in entries]):
                self.github.set_unsigned(entries)
                self.assert_failure()
                self.assertFalse((self.output / "unsigned").exists())
                self.assertFalse((self.root / "escape.pkg").exists())
                self.reset_output()

    def test_checkpoint_zip_uses_only_fixed_names_and_refuses_link_or_extra_content(self):
        manifest = ("checkpoint.json", json.dumps(self.github.manifest).encode(), stat.S_IFREG | 0o600)
        for entries in [
            [manifest], [manifest, ("other.pkg", SIGNED, stat.S_IFREG | 0o600)],
            [manifest, ("signed-upload.pkg", b"../../escape", stat.S_IFLNK | 0o777)],
            [manifest, ("signed-upload.pkg", SIGNED, stat.S_IFREG | 0o600), ("extra", b"x", stat.S_IFREG | 0o600)],
        ]:
            with self.subTest(entries=[entry[0] for entry in entries]):
                self.github.set_checkpoint(entries)
                self.assert_failure()
                self.assertFalse((self.output / "checkpoint").exists())
                self.reset_output()

    def test_expanded_archive_and_manifest_sizes_are_bounded_before_writing_members(self):
        with mock.patch.object(preparation, "MAX_EXTRACTED_BYTES", len(UNSIGNED) - 1):
            self.assert_failure("expanded size")
        self.assertFalse((self.output / "unsigned").exists())
        self.reset_output()
        with mock.patch.object(preparation, "MAX_MANIFEST_BYTES", 8):
            self.assert_failure("member size")
        self.assertFalse((self.output / "checkpoint").exists())

    def test_pagination_selects_checkpoint_from_second_page_and_rejects_duplicates(self):
        checkpoint = self.github.artifacts[1]
        extras = [self.github.artifact(1000 + index, f"unrelated-{index}") for index in range(99)]
        self.github.artifacts = [self.github.artifacts[0], *extras, checkpoint]
        self.assertEqual(self.prepare()["checkpoint_artifact"]["id"], 102)
        self.assertTrue(any(call[-1].endswith("page=2") for call in self.github.calls))
        self.reset_output()
        self.github.artifacts.append({**checkpoint, "id": 103})
        self.assert_failure("exactly one retained artifact")

    def test_existing_output_and_symlinked_parents_are_refused_before_github_calls(self):
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_bytes(b"existing data")
        self.assert_failure()
        self.assertEqual(sentinel.read_bytes(), b"existing data")
        self.assertEqual(self.github.calls, [])
        self.output = self.root / "parent-link" / "new"
        self.output.parent.symlink_to(self.root, target_is_directory=True)
        self.assert_failure("without symlinks")
        self.assertFalse((self.root / "new").exists())
        self.assertEqual(self.github.calls, [])

    def test_invalid_arguments_fail_before_creating_output_or_invoking_gh(self):
        for changed in [{"repo": "../other"}, {"repo": "owner/repo?token=value"},
                        {"run_id": 0}, {"run_id": True}, {"attempt": -1},
                        {"expected_source_sha": "a" * 39}, {"source_sha256": "bad"},
                        {"signed_sha256": "b" * 63}, {"output_dir": self.root / "parent" / ".." / "new"}]:
            with self.subTest(changed=changed):
                self.assert_failure(**changed)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.github.calls, [])

    def test_cli_outputs_paths_on_success_and_never_echoes_failed_gh_stderr(self):
        arguments = ["prepare", "--repo", REPO, "--run-id", str(RUN_ID),
                     "--attempt", str(ATTEMPT), "--expected-source-sha", SOURCE_SHA,
                     "--source-sha256", SOURCE_HASH, "--signed-sha256", SIGNED_HASH,
                     "--output-dir", str(self.output)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(preparation.sys, "argv", arguments), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(preparation.main(), 0)
        self.assertEqual(json.loads(stdout.getvalue())["source"], str(self.output / "unsigned/NVBroadcast-1.5.3-1.pkg"))
        self.assertEqual(stderr.getvalue(), "")
        self.reset_output()
        arguments[-1] = str(self.output)
        self.github.fail_metadata = True
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(preparation.sys, "argv", arguments), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(preparation.main(), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("secret-never-output", stderr.getvalue())
        self.assertIn("metadata request failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

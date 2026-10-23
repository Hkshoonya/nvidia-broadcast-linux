"""Candidate transport, provenance and decoder boundaries; no network or Docker."""

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_candidate_validator', ROOT / 'packaging/native-runtime/validate_candidate.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
VERSION = '1.5.3-4.cpu'
NAMES = [
 f'dist/native-cpu/deb/nvbroadcast_{VERSION}_amd64.deb',
 f'dist/native-cpu/rpm/build/RPMS/x86_64/nvbroadcast-{VERSION}.x86_64.rpm',
 *['dist/native-cpu/' + n for n in ('packages.json','build-provenance.json','payload/manifest.json','deb/content.json','rpm/content.json','assembly/pylock.toml')],
 *['dist/builder-delivery/' + n for n in ('authenticated-builders.json','capacity.json','host-tools.json')],
 'packaging/native-runtime/builder-delivery.json','packaging/native-runtime/builders.json']

class Tests(unittest.TestCase):

    def zip_check(self, change=None, metadata_change=None):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            z = root / 'candidate.zip'
            records = [(n, b'content', None) for n in NAMES]
            if change:
                records = change(records)
            with zipfile.ZipFile(z, 'w') as archive:
                for name, data, mode in records:
                    entry = zipfile.ZipInfo(name)
                    if mode is not None:
                        entry.external_attr = mode << 16
                    archive.writestr(entry, data)
            metadata = {'size_in_bytes': z.stat().st_size, 'digest': 'sha256:' + v.file_digest(z)}
            if metadata_change:
                metadata_change(metadata)
            output = root / 'output'
            output.mkdir()
            return v.unpack_zip(z, metadata, output, VERSION)

    def test_exact_layout(self):
        self.assertEqual(self.zip_check()['files'], 13)

    def test_extra_file(self):
        with self.assertRaisesRegex(ValueError, 'file set'):
            self.zip_check(lambda x: x + [('extra', b'x', None)])

    def test_duplicate_name(self):
        with self.assertRaisesRegex(ValueError, 'file set'):
            self.zip_check(lambda x: x[:-1] + [x[0]])

    def test_traversal(self):
        with self.assertRaisesRegex(ValueError, 'file set'):
            self.zip_check(lambda x: x[:-1] + [('../escape', b'x', None)])

    def test_symlink(self):
        with self.assertRaisesRegex(ValueError, 'special'):
            self.zip_check(lambda x: [(x[0][0], b'target', stat.S_IFLNK | 511)] + x[1:])

    def test_zip_digest(self):
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.zip_check(metadata_change=lambda m: m.update(digest='sha256:' + '0' * 64))

    def test_zip_size(self):
        with self.assertRaisesRegex(ValueError, 'size'):
            self.zip_check(metadata_change=lambda m: m.update(size_in_bytes=1))

    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            v.parse_json(b'{"sha":1,"sha":2}')

    def test_command_stream_cap(self):
        with self.assertRaisesRegex(ValueError, 'exceeds evidence'):
            v.bounded([sys.executable, '-I', '-c', 'import sys,time;sys.stdout.write("x"*100000);sys.stdout.flush();time.sleep(10)'], limit=32, timeout=1)

    def test_command_timeout(self):
        with self.assertRaisesRegex(ValueError, 'timed out'):
            v.bounded([sys.executable, '-I', '-c', 'import time;time.sleep(10)'], timeout=0.1)

    def test_tar_extension_allocation_cap(self):
        for kind in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
            header = tarfile.TarInfo('pax')
            header.type = kind
            header.size = v.MIB + 1
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'extension exceeds'):
                v.BoundedTarStream(io.BytesIO(header.tobuf())).read(512)

    def test_tar_regular_cap(self):
        header = tarfile.TarInfo('huge')
        header.size = v.GIB + 1
        with self.assertRaisesRegex(ValueError, 'member exceeds'):
            v.BoundedTarStream(io.BytesIO(header.tobuf())).read(512)

    def test_tar_truncated_block(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            v.BoundedTarStream(io.BytesIO(b'x')).read(512)

    def test_tar_valid_stream(self):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w', format=tarfile.PAX_FORMAT) as archive:
            header = tarfile.TarInfo('file')
            header.size = 5
            archive.addfile(header, io.BytesIO(b'hello'))
        raw.seek(0)
        with tarfile.open(fileobj=v.BoundedTarStream(raw), mode='r|') as archive:
            member = next(iter(archive))
            self.assertEqual(archive.extractfile(member).read(), b'hello')

    def test_tar_hidden_second_archive(self):
        header = tarfile.TarInfo('hidden').tobuf()
        raw = io.BytesIO(bytes(512) + header + bytes(10240))
        with self.assertRaisesRegex(ValueError, 'terminal padding'):
            with tarfile.open(fileobj=v.BoundedTarStream(raw), mode='r|') as archive:
                list(archive)

    def test_paths(self):
        for path in ('../a', '/a', 'a/../b', 'a//b', '././a', 'a\\b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                v.path_name(path)
        self.assertEqual(v.path_name('./usr/bin/a', absolute=True), '/usr/bin/a')

    def test_live_source_mismatch(self):
        import argparse
        args = argparse.Namespace(run_id=1, attempt=2, source='a' * 40, artifact_id=3, gh='gh', expected_digest='sha256:' + 'b' * 64)
        run = {'id': 1, 'run_attempt': 2, 'head_sha': 'b' * 40, 'head_branch': 'main', 'repository': {'full_name': v.REPO, 'id': v.REPOSITORY_ID}, 'head_repository': {'full_name': v.REPO, 'id': v.REPOSITORY_ID}, 'event': 'workflow_dispatch', 'path': v.WORKFLOW, 'status': 'completed', 'conclusion': 'success'}
        with patch.object(v, 'bounded', return_value=json.dumps(run).encode()), self.assertRaisesRegex(ValueError, 'run identity'):
            v.live_metadata(args)

    def test_live_attempt_and_artifact(self):
        import argparse
        args = argparse.Namespace(run_id=1, attempt=2, source='a' * 40, artifact_id=3, gh='gh', expected_digest='sha256:' + 'b' * 64)
        run = {'id': 1, 'run_attempt': 2, 'head_sha': args.source, 'head_branch': 'main', 'repository': {'full_name': v.REPO, 'id': v.REPOSITORY_ID}, 'head_repository': {'full_name': v.REPO, 'id': v.REPOSITORY_ID}, 'event': 'workflow_dispatch', 'path': v.WORKFLOW, 'status': 'completed', 'conclusion': 'success'}
        artifact = {'id': 3, 'expired': False, 'name': f'native-cpu-candidate-{args.source}-1-2', 'workflow_run': {'id': 1, 'head_sha': args.source, 'head_branch': 'main', 'repository_id': v.REPOSITORY_ID, 'head_repository_id': v.REPOSITORY_ID}, 'size_in_bytes': 50, 'digest': 'sha256:' + 'b' * 64}
        with patch.object(v, 'bounded', side_effect=[json.dumps(run).encode(), json.dumps(artifact).encode()]):
            self.assertEqual(v.live_metadata(args), (run, artifact))
        artifact['name'] = f'native-cpu-candidate-{args.source}-1-1'
        with patch.object(v, 'bounded', side_effect=[json.dumps(run).encode(), json.dumps(artifact).encode()]), self.assertRaisesRegex(ValueError, 'artifact identity'):
            v.live_metadata(args)


class CandidateAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(run_id=7, attempt=2, source="a" * 40, artifact_id=9,
            expected_digest="sha256:" + "b" * 64, gh="gh")
        self.run = {"id": 7, "run_attempt": 2, "head_sha": self.args.source,
            "head_branch": "main", "repository": {"full_name": v.REPO, "id": v.REPOSITORY_ID},
            "head_repository": {"full_name": v.REPO, "id": v.REPOSITORY_ID},
            "event": "workflow_dispatch", "path": v.WORKFLOW,
            "status": "completed", "conclusion": "success"}
        self.artifact = {"id": 9, "expired": False,
            "name": f"native-cpu-candidate-{self.args.source}-7-2",
            "workflow_run": {"id": 7, "head_sha": self.args.source, "head_branch": "main",
                "repository_id": v.REPOSITORY_ID, "head_repository_id": v.REPOSITORY_ID},
            "size_in_bytes": 123, "digest": self.args.expected_digest}

    def check(self, run=None, artifact=None):
        with patch.object(v, "bounded", side_effect=[json.dumps(run or self.run).encode(),
                json.dumps(artifact or self.artifact).encode()]) as api:
            result = v.live_metadata(self.args)
        for call in api.call_args_list:
            self.assertEqual(call.args[0][:4], ["gh", "api", "--hostname", "github.com"])
        return result

    def test_exact_approved_run_and_artifact(self):
        self.assertEqual(self.check(), (self.run, self.artifact))

    def test_run_authority_attempt_source_or_result_mismatch(self):
        changes = [("id", 8), ("id", 7.0), ("run_attempt", 1), ("head_sha", "c" * 40),
            ("head_branch", "feature"), ("event", "pull_request"),
            ("path", ".github/workflows/unreviewed.yml"), ("status", "in_progress"),
            ("conclusion", "failure"),
            ("repository", {"full_name": v.REPO, "id": 42}),
            ("head_repository", {"full_name": v.REPO, "id": 42}),
            ("head_repository", {"full_name": "fork/project", "id": v.REPOSITORY_ID})]
        for key, value in changes:
            run = copy.deepcopy(self.run)
            run[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(run=run)

    def test_artifact_identity_and_transport_mismatch(self):
        changes = [("id", 10), ("id", 9.0), ("expired", True), ("expired", None), ("name", "other-attempt"),
            ("size_in_bytes", True), ("size_in_bytes", 0), ("size_in_bytes", 2*v.GIB + 1),
            ("digest", "b" * 64), ("digest", "sha256:" + "c" * 64)]
        for key, value in changes:
            artifact = copy.deepcopy(self.artifact)
            artifact[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(artifact=artifact)
        for key, value in (("id", 8), ("head_sha", "c" * 40), ("head_branch", "feature"),
                           ("repository_id", 42), ("head_repository_id", 42)):
            artifact = copy.deepcopy(self.artifact)
            artifact["workflow_run"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(artifact=artifact)

    def test_approval_mismatch_stops_before_download_and_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.args.repo = Path(temporary)
            self.args.output = Path(temporary) / "result"
            self.args.package_revision = "17"
            self.args.zip = None
            self.args.expected_digest = "sha256:" + "c" * 64
            with patch.object(v, "source_identity", return_value={}), \
                    patch.object(v, "git_bytes", return_value=b'[project]\nversion="2.3.4"\n'), \
                    patch.object(v, "bounded", side_effect=[json.dumps(self.run).encode(), json.dumps(self.artifact).encode()]), \
                    patch.object(v, "download_artifact") as download:
                with self.assertRaisesRegex(ValueError, "explicit approval"):
                    v.validate(self.args)
                download.assert_not_called()
            self.assertFalse(self.args.output.exists())

    def test_cli_requires_exact_approval_and_positive_identities(self):
        arguments = ["--run-id", "7", "--attempt", "2", "--artifact-id", "9",
            "--source", "a" * 40, "--expected-digest", "sha256:" + "b" * 64,
            "--package-revision", "17", "--repo", ".", "--output", "unused"]
        args = v.parse_arguments(arguments)
        self.assertEqual(args.package_revision, "17")
        self.assertIsNone(args.zip)
        for option, value in (("--run-id", "0"), ("--attempt", "-1"), ("--artifact-id", "0"),
                ("--source", "main"), ("--package-revision", "0"), ("--expected-digest", "b" * 64)):
            changed = list(arguments)
            changed[changed.index(option) + 1] = value
            with self.subTest(option=option), self.assertRaises(ValueError):
                v.parse_arguments(changed)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.target = self.directory / "candidate.zip"
        self.body = b"approved transport bytes"
        self.args = argparse.Namespace(gh="gh", artifact_id=9,
            expected_digest="sha256:" + hashlib.sha256(self.body).hexdigest())
        self.metadata = {"size_in_bytes": len(self.body), "digest": self.args.expected_digest}

    def fetch(self, program, *, timeout=2):
        original = subprocess.Popen

        def process(command, **kwargs):
            self.assertEqual(command, ["gh", "api", "--hostname", "github.com",
                f"repos/{v.REPO}/actions/artifacts/9/zip"])
            self.assertNotIn("GH_DEBUG", kwargs["env"])
            return original([sys.executable, "-I", "-c", program], **kwargs)

        with patch.object(v.subprocess, "Popen", side_effect=process), patch.dict(os.environ, {"GH_DEBUG": "api"}):
            v.download_artifact(self.args, self.metadata, self.target, timeout=timeout)

    def test_exact_streamed_bytes_are_private_and_read_only(self):
        self.fetch("import sys;sys.stdout.buffer.write(" + repr(self.body) + ")")
        self.assertEqual(self.target.read_bytes(), self.body)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o400)

    def test_oversize_truncated_changed_and_failed_downloads_are_removed(self):
        programs = ["import sys;sys.stdout.buffer.write(" + repr(body) + ")"
                    for body in (self.body + b"x", self.body[:-1], b"x" * len(self.body))]
        programs += ["raise SystemExit(1)"]
        for program in programs:
            with self.subTest(program=program), self.assertRaises(ValueError):
                self.fetch(program)
            self.assertFalse(self.target.exists())

    def test_timeout_removes_partial_download(self):
        with self.assertRaisesRegex(ValueError, "timed out"):
            self.fetch("import time;time.sleep(10)", timeout=0.1)
        self.assertFalse(self.target.exists())

    def test_existing_file_or_symlink_is_never_overwritten(self):
        self.target.write_bytes(b"existing")
        with patch.object(v.subprocess, "Popen") as process:
            with self.assertRaises(FileExistsError):
                v.download_artifact(self.args, self.metadata, self.target)
            process.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"existing")
        self.target.unlink()
        outside = self.directory / "outside"
        outside.write_bytes(b"untouched")
        self.target.symlink_to(outside)
        with self.assertRaises(FileExistsError):
            v.download_artifact(self.args, self.metadata, self.target)
        self.assertEqual(outside.read_bytes(), b"untouched")

    def test_unapproved_digest_is_rejected_before_process_or_file(self):
        self.metadata["digest"] = "sha256:" + "f" * 64
        with patch.object(v.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "not approved"):
                v.download_artifact(self.args, self.metadata, self.target)
            process.assert_not_called()
        self.assertFalse(self.target.exists())


class ExtractionBoundaryTests(unittest.TestCase):
    def test_nested_directory_symlink_does_not_redirect_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, outside = root / "output", root / "outside"
            output.mkdir()
            outside.mkdir()
            (output / "dist").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(OSError), v.create_member(output, "dist/candidate.deb"):
                self.fail("symlink parent accepted")
            self.assertEqual(list(outside.iterdir()), [])

    def test_existing_member_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "candidate").write_bytes(b"original")
            with self.assertRaises(FileExistsError), v.create_member(root, "candidate"):
                self.fail("existing member accepted")
            self.assertEqual((root / "candidate").read_bytes(), b"original")

    def test_member_traversal_is_rejected_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError), v.create_member(Path(temporary), "../outside"):
                self.fail("traversal accepted")

    def test_local_zip_symlink_and_fifo_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.write_bytes(b"x")
            metadata = {"size_in_bytes": 1, "digest": "sha256:" + hashlib.sha256(b"x").hexdigest()}
            link = root / "link"
            link.symlink_to(archive)
            with self.assertRaises(OSError):
                v.unpack_zip(link, metadata, root, VERSION)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "type/size"):
                v.unpack_zip(fifo, metadata, root, VERSION)

    def test_other_compression_is_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as saved:
                for name in NAMES:
                    saved.writestr(name, b"x", compress_type=zipfile.ZIP_LZMA)
            metadata = {"size_in_bytes": archive.stat().st_size, "digest": "sha256:" + v.file_digest(archive)}
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "unsupported ZIP compression"):
                v.unpack_zip(archive, metadata, output, VERSION)
            self.assertEqual(list(output.iterdir()), [])


class SourceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.env = dict(os.environ, GIT_AUTHOR_NAME="Fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
            GIT_COMMITTER_NAME="Fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid")
        self.git("-c", "init.templateDir=", "init", "-q")
        workflow = self.repo / v.WORKFLOW
        workflow.parent.mkdir(parents=True)
        workflow.write_bytes(b"trusted workflow\n")
        self.git("add", ".")
        tree = self.git("write-tree")
        self.first = self.git("commit-tree", tree, "-m", "ancestor")
        self.head = self.git("commit-tree", tree, "-p", self.first, "-m", "trusted checkout")
        self.other = self.git("commit-tree", tree, "-m", "unrelated")
        self.git("update-ref", "HEAD", self.head)

    def git(self, *arguments):
        return subprocess.check_output(["git", *arguments], cwd=self.repo, env=self.env, text=True).strip()

    def test_ancestor_and_workflow_are_bound_without_executing_source(self):
        record = v.source_identity(self.repo, self.first)
        self.assertEqual(record["trusted_checkout"], self.head)
        self.assertEqual(record["source_tree"], self.git("rev-parse", self.first + "^{tree}"))
        self.assertEqual(record["source_workflow_sha256"], hashlib.sha256(b"trusted workflow\n").hexdigest())

    def test_unrelated_source_is_refused_before_api_or_download(self):
        args = argparse.Namespace(repo=self.repo, source=self.other)
        with patch.object(v, "live_metadata") as metadata, patch.object(v, "download_artifact") as download:
            with self.assertRaises(ValueError):
                v.validate(args)
            metadata.assert_not_called()
            download.assert_not_called()

    def test_git_replacement_cannot_turn_unrelated_source_into_ancestor(self):
        self.git("replace", self.other, self.first)
        with self.assertRaises(ValueError):
            v.source_identity(self.repo, self.other)



if __name__ == "__main__":
    unittest.main()

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
from unittest.mock import Mock, patch
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



class VariantBoundaryTests(unittest.TestCase):
    def test_policy_allowlist_and_cpu_compatibility(self):
        cpu, cuda = (v.policy.policy_for(name) for name in ("cpu", "cuda"))
        self.assertEqual(tuple(cpu)[1:], (v.GIB, 2*v.GIB, 4*v.GIB, 5*v.GIB, 12*v.GIB))
        self.assertEqual(cuda.package_max_bytes, 2*v.GIB-1)
        self.assertGreaterEqual(cuda.transport_max_bytes, 2*cuda.package_max_bytes+32*v.MIB)
        self.assertLessEqual(cuda.transport_max_bytes, 4*v.GIB+64*v.MIB)
        self.assertGreater(cuda.payload_max_bytes, 5003906349)
        self.assertLessEqual(cuda.payload_max_bytes, 6*v.GIB)
        self.assertEqual(cuda.tar_max_bytes, 7*v.GIB)
        self.assertEqual(cuda.capacity_bytes, 40*v.GIB)
        for variant in (None, "CUDA", "cuda/../cpu", "", 1):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                v.policy.policy_for(variant)
        v.policy.check_variant({}, "cpu")
        for record, variant in (({}, "cuda"), ({"variant": "cuda"}, "cpu"),
                                ({"variant": "cpu"}, "cuda"), ({"variant": None}, "cpu")):
            with self.subTest(record=record, variant=variant), self.assertRaises(ValueError):
                v.policy.check_variant(record, variant)

    def test_raw_tar_total_bound_is_selected_without_relaxing_member_bounds(self):
        for variant in ("cpu", "cuda"):
            limit = v.policy.policy_for(variant).tar_max_bytes
            stream = v.BoundedTarStream(io.BytesIO(bytes(1024)), variant)
            stream.total = limit - 512
            self.assertEqual(len(stream.read(512)), 512)
            with self.assertRaisesRegex(ValueError, "total byte bound"):
                stream.read(512)
            header = tarfile.TarInfo("huge")
            header.size = v.GIB+1
            with self.assertRaisesRegex(ValueError, "member exceeds"):
                v.BoundedTarStream(io.BytesIO(header.tobuf()), variant).read(512)

    def test_decoded_payload_budget_counts_regular_bytes_for_each_variant(self):
        # Small real tar members exercise cumulative expansion without allocating
        # gigabytes or starting a container. Only process/resource boundaries vary.
        for variant in ("cpu", "cuda"):
            selected = v.policy.policy_for(variant)._replace(payload_max_bytes=32)
            for second_size in (16, 17):
                raw = io.BytesIO()
                with tarfile.open(fileobj=raw, mode="w") as archive:
                    for name, size in (("usr/a", 16), ("usr/b", second_size)):
                        member = tarfile.TarInfo(name)
                        member.size, member.mode = size, 0o644
                        archive.addfile(member, io.BytesIO(b"x"*size))
                raw.seek(0)
                process = Mock(stdout=raw, returncode=0)
                process.wait.return_value = process.poll.return_value = 0
                with self.subTest(variant=variant, second_size=second_size), \
                        patch.object(v.subprocess, "Popen", return_value=process), \
                        patch.object(v, "bounded", return_value=b""), \
                        patch.object(v.policy, "policy_for", return_value=selected) as choose:
                    if second_size == 16:
                        self.assertEqual(v.decoded_inventory(Path("fixture.deb"), "deb", variant=variant)[3], 32)
                    else:
                        with self.assertRaisesRegex(ValueError, "expands beyond bound"):
                            v.decoded_inventory(Path("fixture.deb"), "deb", variant=variant)
                    self.assertTrue(all(call.args == (variant,) for call in choose.call_args_list))

    def test_cross_variant_version_rejected_before_file_open(self):
        for variant, version in (("cpu", "1.5.3-7.cuda"), ("cuda", VERSION)):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "explicit variant"):
                v.unpack_zip(Path("does-not-exist"), {}, Path("unused"), version, variant)

    def test_cuda_archive_layout_and_individual_package_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "candidate.zip"
            names = [name.replace("native-cpu", "native-cuda").replace(".cpu", ".cuda") for name in NAMES]
            with zipfile.ZipFile(archive, "w") as saved:
                for name in names:
                    saved.writestr(name, b"content")
            metadata = {"size_in_bytes": archive.stat().st_size, "digest": "sha256:" + v.file_digest(archive)}
            output = root / "output"
            output.mkdir()
            self.assertEqual(v.unpack_zip(archive, metadata, output, VERSION.replace(".cpu", ".cuda"), "cuda")["files"], 13)
            with self.assertRaisesRegex(ValueError, "file set"):
                v.unpack_zip(archive, metadata, output, VERSION, "cpu")
            original = zipfile.ZipFile.infolist
            for size in (2*v.GIB-1, 2*v.GIB):
                def changed(saved):
                    members = original(saved)
                    members[0].file_size = size
                    return members
                with self.subTest(size=size), patch.object(zipfile.ZipFile, "infolist", changed), \
                        patch.object(zipfile.ZipFile, "open", side_effect=RuntimeError("reached extraction")):
                    expected = RuntimeError if size < 2*v.GIB else ValueError
                    message = "reached extraction" if size < 2*v.GIB else "entry exceeds bound"
                    with self.assertRaisesRegex(expected, message):
                        v.unpack_zip(archive, metadata, output, VERSION.replace(".cpu", ".cuda"), "cuda")

    def test_runtime_probe_requires_executed_cpu_and_failed_cuda(self):
        good = {"status": "pass", "variant": "cuda", "cuda_unavailable_expected": True,
                "cpu_execution": {"provider": "cpu", "success": True, "executed_providers": ["CPUExecutionProvider"]},
                "cuda_execution": {"provider": "cuda", "success": False, "executed_providers": []}}
        v.policy.validate_runtime_probe(good, "cuda")
        for key, value in (("variant", "cpu"), ("cuda_unavailable_expected", False),
                ("cpu_execution", {"provider": "cpu", "success": True, "executed_providers": []}),
                ("cuda_execution", {"provider": "cuda", "success": True, "executed_providers": ["CUDAExecutionProvider"]}),
                ("cuda_execution", {"provider": "cuda", "success": False, "executed_providers": ["CUDAExecutionProvider"]}),
                ("cupy_execution", {"output": [1]})):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                v.policy.validate_runtime_probe(dict(good, **{key: value}), "cuda")
        v.policy.validate_runtime_probe({"status": "pass"}, "cpu")

    def test_hook_variant_comes_from_explicit_selection(self):
        git = lambda path: (ROOT / path).read_bytes()
        for variant in ("cpu", "cuda"):
            version = "1.5.3-7." + variant
            script = v.expected_hook(git, "finish", version, "deb", b"", variant)
            self.assertIn(variant, script)
            self.assertIn(version, script)
            self.assertNotIn("@VARIANT@", script)


class CudaAuthorityTests(CandidateAuthorityTests):
    def setUp(self):
        super().setUp()
        self.args.variant = "cuda"
        self.artifact["name"] = self.artifact["name"].replace("native-cpu", "native-cuda")

    def test_artifact_identity_and_transport_mismatch(self):
        for size in (2*v.GIB+1, v.policy.policy_for("cuda").transport_max_bytes):
            self.artifact["size_in_bytes"] = size
            self.assertEqual(self.check()[1]["size_in_bytes"], size)
        for key, value in (("size_in_bytes", v.policy.policy_for("cuda").transport_max_bytes+1),
                           ("size_in_bytes", True), ("name", self.artifact["name"].replace("native-cuda", "native-cpu"))):
            changed = dict(self.artifact, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "artifact identity"):
                self.check(artifact=changed)



class VariantPayloadIdentityTests(unittest.TestCase):
    """Exercise real validation order with a small inert decoded payload fixture."""
    def validate_fixture(self, variant, change=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = {"revision": "a"*40, "version": "1.5.3", "source_date_epoch": 1}
            version = "1.5.3-7." + variant
            git = lambda path: (ROOT / path).read_bytes()
            adapter = {"revision": source["revision"], "tree": "b"*40,
                       "files": {name: v.digest(git(name)) for name in v.ADAPTER_PATHS}}
            base_lock = git(f"packaging/native-runtime/pylock.linux-x86_64-cp313-{variant}.toml")
            app_lock = ('\n[[packages]]\nname="nvbroadcast"\nversion="1.5.3"\n'
                        '[packages.archive]\npath="nvbroadcast-1.5.3-py3-none-any.whl"\n'
                        '[packages.archive.hashes]\nsha256="' + 'c'*64 + '"\n').encode()
            lock = base_lock + app_lock
            pins = json.loads(git("packaging/native-runtime/builders.json"))
            delivery = json.loads(git("packaging/native-runtime/builder-delivery.json"))
            packaging = next(p for p in v.tomllib.loads(git("packaging/runtime-prototype/pylock.build.toml").decode())["packages"] if p["name"] == "packaging")
            base = f"dist/native-{variant}/"
            files = {base+"build-provenance.json": {"application": source, "adapter": adapter},
                     base+"payload/manifest.json": {}, base+"assembly/pylock.toml": lock,
                     "packaging/native-runtime/builders.json": git("packaging/native-runtime/builders.json"),
                     "packaging/native-runtime/builder-delivery.json": git("packaging/native-runtime/builder-delivery.json"),
                     "dist/builder-delivery/capacity.json": {"variant": variant,
                         "required_free_bytes": v.policy.policy_for(variant).capacity_bytes,
                         "observed_free_bytes": {name: 64*v.GIB for name in ("GITHUB_WORKSPACE", "RUNNER_TEMP", "HOME")}},
                     "dist/builder-delivery/host-tools.json": {"interpreter": "/usr/bin/python3", "isolated": True,
                         "site_disabled": True, "packaging_version": packaging["version"],
                         "wheel_sha256": packaging["wheels"][0]["hashes"]["sha256"]},
                     "dist/builder-delivery/authenticated-builders.json": {"status": "authenticated_import_delivery",
                         "archive_sha256": delivery["archive"]["sha256"], "archive_bytes": delivery["archive"]["bytes"],
                         "blob_count": 23, "builder_images": {role: pin["image"] for role, pin in pins.items()},
                         "origin": "maintainer-qualified-historical-snapshot", "original_build_provenance_authenticated": False,
                         "installed_inventory": {role: {"image": pin["image"], "package_count": len(pin["packages"])} for role, pin in pins.items()}}}
            packages = []
            for family, name in (("deb", f"deb/nvbroadcast_{version}_amd64.deb"),
                                 ("rpm", f"rpm/build/RPMS/x86_64/nvbroadcast-{version}.x86_64.rpm")):
                files[base+name] = b"inert package"
                files[base+family+"/content.json"] = {}
                packages.append({"family": family, "artifact": name, "content": family+"/content.json",
                    "version": version, "variant": variant, "source": dict(source), "adapter": adapter,
                    "builder_image": pins[family]["image"], "runtime_manifest_sha256": v.digest(b"{}"),
                    "content_sha256": v.digest(b"{}"), "bytes": 13, "sha256": v.digest(b"inert package")})
            files[base+"packages.json"] = packages
            inputs = json.loads(git("packaging/runtime-prototype/inputs.json"))
            inputs["application"] = source
            kept = {"/usr/lib/nvbroadcast/runtime-manifest.json": b"{}",
                    v.PREFIX+"share/nvbroadcast-runtime-provenance/pylock.toml": lock,
                    v.PREFIX+"share/nvbroadcast-runtime-provenance/builders.json": git("packaging/native-runtime/builders.json"),
                    "/usr/lib/nvbroadcast/validate-install.py": git("packaging/native-runtime/validate_install.py"),
                    v.PREFIX+"share/nvbroadcast-runtime-provenance/inputs.json": json.dumps(inputs).encode(),
                    v.PREFIX+"share/nvbroadcast-runtime-provenance/selection.json": json.dumps({
                        "target": "linux-x86_64-cp313-"+variant, "lock_sha256": v.digest(lock), "source": source}).encode(),
                    "/usr/lib/nvbroadcast/package.json": json.dumps({"schema_version": 1, "variant": variant,
                        "version": version, "source": source, "adapter": adapter, "runtime_manifest_sha256": v.digest(b"{}")}).encode()}
            if change:
                change(files, kept, base)
            args = argparse.Namespace(variant=variant, repo=ROOT, source=source["revision"], package_revision="7",
                run_id=1, attempt=1, artifact_id=1, expected_digest="sha256:"+"d"*64,
                output=root/"output", zip=root/"unused.zip")
            def extract(*_):
                for name, value in files.items():
                    path = args.output / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
                return {"files": len(files)}
            def command(argv, **_):
                if argv[0] == "docker":
                    return (argv[-1]+"\n").encode()
                return b"" if "ls-tree" in argv else b"1"
            controls = {name: b"" for name in ("/control", "/preinst", "/prerm", "/postinst", "/postrm")}
            def decode(*_, control=False, **kwargs):
                self.assertEqual(kwargs["variant"], variant)
                return (controls, {}, controls, 0) if control else ({}, {}, kept, 0)
            with patch.object(v, "source_identity", return_value={"source_tree": "b"*40}), \
                    patch.object(v, "git_bytes", side_effect=lambda repo, revision, name: git(name)), \
                    patch.object(v, "live_metadata", return_value=({}, {"name": "fixture", "size_in_bytes": 1, "digest": args.expected_digest})), \
                    patch.object(v, "unpack_zip", side_effect=extract), patch.object(v, "bounded", side_effect=command), \
                    patch.object(v, "decoded_inventory", side_effect=decode), \
                    patch.object(v, "expected_hook", side_effect=RuntimeError("identity checked before script comparison")):
                v.validate(args)

    def test_selected_variant_identity_reaches_script_comparison(self):
        for variant in ("cpu", "cuda"):
            with self.subTest(variant=variant), self.assertRaisesRegex(RuntimeError, "identity checked"):
                self.validate_fixture(variant)

    def test_opposite_lock_and_provenance_cannot_cross_variant_boundary(self):
        for variant, other in (("cpu", "cuda"), ("cuda", "cpu")):
            def replace_lock(files, kept, base):
                files[base+"assembly/pylock.toml"] = (ROOT/f"packaging/native-runtime/pylock.linux-x86_64-cp313-{other}.toml").read_bytes()
            cases = [("lock prefix", replace_lock),
                ("provenance", lambda files, kept, base: files[base+"packages.json"][0].update(variant=other)),
                ("provenance", lambda files, kept, base: files[base+"packages.json"][0].update(version="1.5.3-7."+other)),
                ("provenance", lambda files, kept, base: files[base+"packages.json"][0]["source"].update(revision="f"*40)),
                ("variant differs", lambda files, kept, base: files["dist/builder-delivery/capacity.json"].update(variant=other)),
                ("input selection", lambda files, kept, base: kept.update({v.PREFIX+"share/nvbroadcast-runtime-provenance/selection.json": json.dumps({"target": "linux-x86_64-cp313-"+other}).encode()})),
                ("package identity", lambda files, kept, base: kept.update({"/usr/lib/nvbroadcast/package.json": json.dumps({"variant": other}).encode()}))]
            for message, change in cases:
                with self.subTest(variant=variant, message=message), self.assertRaisesRegex(ValueError, message):
                    self.validate_fixture(variant, change)


if __name__ == "__main__":
    unittest.main()

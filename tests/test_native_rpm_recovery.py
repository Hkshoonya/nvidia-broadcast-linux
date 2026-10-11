"""Authenticated native recovery preserves evidence and rejects unsafe plans."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_rpm_recovery", ROOT / "scripts/recover_native_rpm.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.umask, os.umask(0o022))
        temporary = tempfile.TemporaryDirectory(prefix="native recovery ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fs = recovery.Filesystem(self.root, os.getuid())
        self.version = "1.5.3-2.cpu"
        self.tid = int(time.time()) - 1
        self.members = {
            recovery.BASE: None, recovery.BASE + "/runtime": None,
            recovery.BASE + "/runtime/lib": None,
            recovery.BASE + "/runtime/lib/example.so": b"known payload contents",
            recovery.BASE + "/runtime-manifest.json": b"{}",
            recovery.BASE + "/validate-install.py": b"validator",
            recovery.BASE + "/package.json": b"{}",
            recovery.BASE + "/check-packages": b"check",
            "/usr/bin/nvbroadcast": b"launcher",
        }
        for name, data in self.members.items():
            path = self.fs.path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if data is None:
                path.mkdir(exist_ok=True)
            else:
                path.write_bytes(data)
        self.fs.path(recovery.MARKER).write_text(self.version + "\n")
        self.rpm = self.root / "authenticated.rpm"
        self.rpm.write_bytes(b"authenticated snapshot fixture")
        self.target = {"path": self.rpm, "sha256": recovery.digest(self.rpm), "version": self.version}
        self.key = ROOT / "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc"
        self.target["public_key"] = self.key

    def archive(self, members=None):
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode="w") as archive:
            for name, data in (members or self.members).items():
                item = tarfile.TarInfo(name.removeprefix("/"))
                item.mode = 0o755 if data is None else 0o644
                if data is None:
                    item.type = tarfile.DIRTYPE
                    archive.addfile(item)
                elif isinstance(data, tuple):
                    item.type = tarfile.SYMTYPE
                    item.mode = 0o777
                    item.linkname = data[0]
                    archive.addfile(item)
                else:
                    item.size = len(data)
                    archive.addfile(item, io.BytesIO(data))
        content.seek(0)
        return tarfile.open(fileobj=content, mode="r|")

    def inventory(self, package, filesystem, marker=None):
        with self.archive() as archive:
            return recovery.archive_inventory(archive, filesystem, marker)

    def plan(self):
        with mock.patch.object(recovery, "package_inventory", side_effect=self.inventory):
            return recovery.plan(self.fs, self.target)

    def fragment(self, name=None, data=b"known", tid=None):
        name = name or recovery.BASE + "/runtime/lib/example.so"
        path = self.fs.path(name + f";{self.tid if tid is None else tid:08x}")
        path.write_bytes(data)
        return path

    def test_inspection_and_preservation_use_exact_prefix_and_keep_marker(self):
        fragment = self.fragment()
        entries = self.plan()
        self.assertEqual(fragment.read_bytes(), b"known")
        report = recovery.preserve(self.fs, self.target, entries)
        self.assertEqual(report["preserved"], [entries[0]["original"]])
        destination = Path(report["package"]).parent
        self.assertEqual((destination / "files" / entries[0]["original"].lstrip("/")).read_bytes(), b"known")
        self.assertEqual(Path(report["package"]).read_bytes(), self.rpm.read_bytes())
        self.assertTrue(self.fs.path(recovery.MARKER).is_file())
        self.assertFalse(fragment.exists())
        self.assertIn("--replacepkgs", report["replay_command"])
        self.assertNotIn("installed successfully", report["status"])

    def test_unknown_private_file_aborts_complete_plan_without_moving_known_fragment(self):
        fragment = self.fragment()
        unknown = self.fs.path(recovery.BASE + "/administrator-note")
        unknown.write_text("leave alone")
        with self.assertRaisesRegex(recovery.RecoveryError, "unrecognized"):
            self.plan()
        self.assertEqual(fragment.read_bytes(), b"known")
        self.assertEqual(unknown.read_text(), "leave alone")
        self.assertFalse(self.fs.path(recovery.RECOVERY).exists())

    def test_wrong_prefix_oversize_and_full_matching_fragment(self):
        for value in (b"wrong", b"known payload contents plus extra"):
            with self.subTest(value=value):
                path = self.fragment(data=value)
                with self.assertRaises(recovery.RecoveryError):
                    self.plan()
                self.assertEqual(path.read_bytes(), value)
                path.unlink()
        path = self.fragment(data=b"known payload contents")
        self.assertEqual(self.plan()[0]["size"], path.stat().st_size)

    def test_stale_future_and_mixed_transaction_suffixes_refused(self):
        for tid in (self.tid - 600, self.tid + 30):
            with self.subTest(tid=tid):
                path = self.fragment(tid=tid)
                with self.assertRaisesRegex(recovery.RecoveryError, "outside this transaction"):
                    self.plan()
                path.unlink()
        first = self.fragment()
        second = self.fragment(tid=self.tid - 1)
        with self.assertRaisesRegex(recovery.RecoveryError, "more than one"):
            self.plan()
        self.assertTrue(first.exists() and second.exists())

    def test_unrecognized_external_neighbor_is_not_touched(self):
        known = self.fragment("/usr/bin/nvbroadcast", b"launch")
        unrelated = self.fs.path(f"/usr/bin/administrator-tool;{self.tid:08x}")
        unrelated.write_text("administrator content")
        entries = self.plan()
        recovery.preserve(self.fs, self.target, entries)
        self.assertFalse(known.exists())
        self.assertEqual(unrelated.read_text(), "administrator content")

    def test_symlink_fragment_is_compared_without_following_it(self):
        name = recovery.BASE + "/runtime/lib/python-link"
        self.members[name] = ("example.so",)
        fragment = self.fs.path(name + f";{self.tid:08x}")
        fragment.symlink_to("example.so")
        entries = self.plan()
        report = recovery.preserve(self.fs, self.target, entries)
        saved = Path(report["package"]).parent / "files" / entries[0]["original"].lstrip("/")
        self.assertTrue(saved.is_symlink())
        self.assertEqual(os.readlink(saved), "example.so")

    def test_wrong_symlink_destination_refused(self):
        name = recovery.BASE + "/runtime/lib/python-link"
        self.members[name] = ("example.so",)
        fragment = self.fs.path(name + f";{self.tid:08x}")
        fragment.symlink_to("/etc/passwd")
        with self.assertRaisesRegex(recovery.RecoveryError, "symlink differs"):
            self.plan()

    def test_hardlink_writable_and_wrong_owner_fragments_refused(self):
        fragment = self.fragment()
        alias = self.root / "administrator-hardlink"
        os.link(fragment, alias)
        with self.assertRaisesRegex(recovery.RecoveryError, "hardlinks"):
            self.plan()
        alias.unlink()
        fragment.chmod(0o666)
        with self.assertRaisesRegex(recovery.RecoveryError, "permissions"):
            self.plan()
        fragment.chmod(0o600)
        with mock.patch.object(self.fs, "owner", os.getuid() + 1):
            with self.assertRaises(recovery.RecoveryError):
                self.plan()

    def test_redirected_parent_and_writable_directory_refused(self):
        directory = self.fs.path(recovery.BASE + "/runtime/lib")
        moved = directory.with_name("elsewhere")
        directory.rename(moved)
        directory.symlink_to(moved)
        with self.assertRaises(recovery.RecoveryError):
            self.plan()
        directory.unlink()
        moved.rename(directory)
        directory.chmod(0o777)
        with self.assertRaises(recovery.RecoveryError):
            self.plan()

    def test_marker_must_be_safe_and_match_authenticated_target(self):
        marker = self.fs.path(recovery.MARKER)
        for text in ("1.5.3-3.cpu\n", "", self.version + "\nextra\n"):
            marker.write_text(text)
            with self.assertRaisesRegex(recovery.RecoveryError, "target version"):
                self.plan()
        marker.unlink()
        marker.symlink_to(self.rpm)
        with self.assertRaises(recovery.RecoveryError):
            self.plan()

    def test_path_traversal_and_unknown_payload_paths_are_rejected(self):
        for name in ("../escape", "/absolute", "usr/../etc/passwd", "usr//lib/nvbroadcast/file",
                     "usr/lib/nvbroadcast/./file", "usr/lib/nvbroadcast/a\nb", "etc/passwd"):
            with self.subTest(name=name), self.assertRaises(recovery.RecoveryError):
                recovery.member_path(name)
        self.assertEqual(recovery.member_path("./usr/lib/nvbroadcast/runtime/lib/a"), recovery.BASE + "/runtime/lib/a")

    def test_duplicate_members_hardlinks_and_escaping_symlinks_are_rejected(self):
        for kind in ("duplicate", "hardlink", "symlink", "nonroot"):
            content = io.BytesIO()
            with tarfile.open(fileobj=content, mode="w") as archive:
                item = tarfile.TarInfo("usr/lib/nvbroadcast/runtime/lib/a")
                item.mode = 0o644
                if kind == "hardlink":
                    item.type, item.linkname = tarfile.LNKTYPE, "etc/passwd"
                elif kind == "symlink":
                    item.type, item.linkname = tarfile.SYMTYPE, "/etc/passwd"
                elif kind == "nonroot":
                    item.uid = 1000
                archive.addfile(item)
                if kind == "duplicate":
                    archive.addfile(item)
            content.seek(0)
            with self.subTest(kind=kind), tarfile.open(fileobj=content, mode="r|") as archive:
                with self.assertRaises(recovery.RecoveryError):
                    recovery.archive_inventory(archive, self.fs, None)

    def test_changed_fragment_or_package_before_preservation_refused(self):
        fragment = self.fragment()
        entries = self.plan()
        fragment.write_bytes(b"other")
        with self.assertRaises(recovery.RecoveryError):
            recovery.preserve(self.fs, self.target, entries)
        self.assertEqual(fragment.read_bytes(), b"other")
        fragment.write_bytes(b"known")
        entries = self.plan()
        self.rpm.write_bytes(b"changed signed package")
        with self.assertRaisesRegex(recovery.RecoveryError, "snapshot changed"):
            recovery.preserve(self.fs, self.target, entries)
        self.assertTrue(fragment.exists())

    def test_interrupted_moves_retain_durable_intentions_and_completed_evidence(self):
        first = self.fragment()
        second = self.fragment("/usr/bin/nvbroadcast", b"launch")
        entries = self.plan()
        rename = recovery.os.rename
        calls = 0

        def interrupted(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated interruption")
            return rename(source, destination)

        with mock.patch.object(recovery.os, "rename", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interruption"):
                recovery.preserve(self.fs, self.target, entries)
        reports = list(self.fs.path(recovery.RECOVERY).glob("*/recovery.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertEqual(len(report["entries"]), 2)
        self.assertEqual(len(report["preserved"]), 1)
        self.assertEqual(sum(p.exists() for p in (first, second)), 1)
        self.assertTrue(Path(report["package"]).is_file())
        self.assertTrue(self.fs.path(recovery.MARKER).is_file())

    def test_forged_key_rejected_before_any_rpm_command(self):
        key = self.root / "forged.asc"
        key.write_text("forged key and adjacent checksum")
        with mock.patch.object(recovery, "tool") as tool:
            with self.assertRaisesRegex(recovery.RecoveryError, "pinned project key"):
                recovery.authenticate(self.rpm, key, self.root / "verification")
            tool.assert_not_called()

    def test_unsigned_digest_only_success_rejected(self):
        with mock.patch.object(recovery, "tool", side_effect=["", "package.rpm: digests OK"]):
            with self.assertRaisesRegex(recovery.RecoveryError, "no verified"):
                recovery.authenticate(self.rpm, self.key, self.root / "verification")

    def test_failed_signature_never_reads_payload_or_package_metadata(self):
        with mock.patch.object(recovery, "tool", side_effect=["", recovery.RecoveryError("bad signature")]) as tool:
            with self.assertRaisesRegex(recovery.RecoveryError, "bad signature"):
                recovery.authenticate(self.rpm, self.key, self.root / "verification")
            self.assertEqual(tool.call_count, 2)

    def test_private_snapshot_prevents_filename_output_injection_and_original_mutation(self):
        unsafe = self.root / "evil\nSignature: OK\n.rpm"
        unsafe.write_bytes(b"package")
        calls = []

        def native_tool(command, work):
            calls.append(command)
            if "--checksig" in command:
                self.assertEqual(Path(command[-1]).name, "package.rpm")
                unsafe.write_bytes(b"changed concurrently")
                return "Header OpenPGP V4 RSA/SHA256 signature: OK\n"
            if "-qp" in command:
                self.assertEqual(Path(command[-1]).read_bytes(), b"package")
                return "nvbroadcast\n1.5.3-2.cpu\nx86_64\n"
            return ""

        with mock.patch.object(recovery, "tool", side_effect=native_tool):
            result = recovery.authenticate(unsafe, self.key, self.root / "verification")
        self.assertEqual(result["sha256"], hashlib.sha256(b"package").hexdigest())
        self.assertEqual(result["path"].read_bytes(), b"package")

    def test_legacy_or_wrong_package_type_rejected(self):
        for metadata in ("nvbroadcast\n1.5.3-1\nnoarch\n", "other\n1.5.3-2.cpu\nx86_64\n",
                         "nvbroadcast\n1.5.3-2.cpu\nnoarch\n"):
            with self.subTest(metadata=metadata):
                directory = self.root / str(len(list(self.root.iterdir())))
                with mock.patch.object(recovery, "tool", side_effect=["", "Signature: OK\n", metadata]):
                    with self.assertRaisesRegex(recovery.RecoveryError, "native-runtime"):
                        recovery.authenticate(self.rpm, self.key, directory)

    def test_lock_contention_uses_separate_process_for_each_lock_type(self):
        for absolute in ("/var/lib/rpm/.rpm.lock", recovery.LOCK):
            path = self.fs.path(absolute)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        script = ("import fcntl,sys; f=open(sys.argv[1],'r+'); "
                  "getattr(fcntl,sys.argv[2])(f,fcntl.LOCK_EX|fcntl.LOCK_NB)")
        for absolute, function in (("/var/lib/rpm/.rpm.lock", "lockf"), (recovery.LOCK, "flock")):
            with self.subTest(function=function), self.fs.locked("/var/lib/rpm/.rpm.lock"):
                result = subprocess.run(["/usr/bin/python3", "-I", "-B", "-c", script,
                                         str(self.fs.path(absolute)), function], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"BlockingIOError", result.stderr)
        self.assertEqual(self.fs.path(recovery.LOCK).stat().st_size, 0)

    def test_tampered_lock_and_redirected_recovery_parent_refused(self):
        for absolute in ("/var/lib/rpm/.rpm.lock", recovery.LOCK):
            path = self.fs.path(absolute)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.fs.path(recovery.LOCK).write_text("not an empty lock")
        with self.assertRaisesRegex(recovery.RecoveryError, "not empty"), self.fs.locked("/var/lib/rpm/.rpm.lock"):
            self.fail("unsafe lock acquired")
        self.fragment()
        self.fs.path(recovery.RECOVERY).symlink_to(self.root)
        with self.assertRaises(recovery.RecoveryError):
            recovery.preserve(self.fs, self.target, self.plan())


    def test_signed_previous_package_can_recognize_removed_paths_without_staging(self):
        self.fragment()
        removed = self.fs.path(recovery.BASE + "/runtime/removed.py")
        removed.write_text("previous signed payload")
        with self.assertRaisesRegex(recovery.RecoveryError, "unrecognized"):
            self.plan()
        def inventory(package, filesystem, marker=None):
            if package is self.target:
                return self.inventory(package, filesystem, marker)
            return {recovery.BASE + "/runtime/removed.py": {"kind": "file"}}, []
        with mock.patch.object(recovery, "package_inventory", side_effect=inventory):
            entries = recovery.plan(self.fs, self.target, {"version": "1.5.3-1.cpu"})
        self.assertEqual(len(entries), 1)
        self.assertEqual(removed.read_text(), "previous signed payload")

    def test_hung_archive_producer_is_terminated_while_parser_waits(self):
        popen = subprocess.Popen
        def blocked(*args, **kwargs):
            return popen(["/usr/bin/python3", "-I", "-B", "-c", "import time; time.sleep(30)"], **kwargs)
        started = time.monotonic()
        with mock.patch.object(recovery, "STREAM_TIMEOUT", 0.1), mock.patch.object(recovery.subprocess, "Popen", side_effect=blocked):
            with self.assertRaises(tarfile.ReadError):
                recovery.package_inventory(self.target, self.fs)
        self.assertLess(time.monotonic() - started, 5)

    def test_embedded_public_key_uses_same_hash_pin(self):
        with mock.patch.object(recovery, "PUBLIC_ARMOR", self.key.read_bytes()):
            with mock.patch.object(recovery, "tool", side_effect=["", "Signature: OK\n", "nvbroadcast\n1.5.3-2.cpu\nx86_64\n"]):
                result = recovery.authenticate(self.rpm, None, self.root / "embedded")
        self.assertEqual(result["fingerprint"], recovery.PRIMARY)
        with mock.patch.object(recovery, "PUBLIC_ARMOR", b"forged key"), mock.patch.object(recovery, "tool") as tool:
            with self.assertRaisesRegex(recovery.RecoveryError, "pinned project key"):
                recovery.authenticate(self.rpm, None, self.root / "forged")
            tool.assert_not_called()


@unittest.skipUnless(shutil.which("rpm") and shutil.which("rpmbuild"), "RPM native tools required")
class NativeRPMLockTests(unittest.TestCase):
    def test_real_rpm_database_transaction_obeys_fcntl_lock(self):
        self.addCleanup(os.umask, os.umask(0o022))
        with tempfile.TemporaryDirectory(prefix="nvb-rpm-lock-") as temporary:
            root = Path(temporary)
            spec = root / "fixture.spec"
            spec.write_text("""Name: nvb-recovery-lock-fixture
Version: 1
Release: 1
Summary: Disposable RPM lock regression fixture
License: MIT
BuildArch: noarch
%description
This fixture is registered only in a disposable database, never installed.
%install
mkdir -p %{buildroot}/usr/share/nvb-recovery-lock-fixture
printf fixture > %{buildroot}/usr/share/nvb-recovery-lock-fixture/file
%files
/usr/share/nvb-recovery-lock-fixture/file
""")
            built = subprocess.run(["rpmbuild", "--define", f"_topdir {root / 'build'}",
                                    "--define", f"_tmppath {root}", "-bb", str(spec)],
                                   capture_output=True)
            self.assertEqual(built.returncode, 0, built.stderr.decode())
            package = next((root / "build/RPMS").rglob("*.rpm"))
            database = root / "rpmdb"
            subprocess.run(["rpm", "--dbpath", str(database), "--initdb"], check=True, capture_output=True)
            fs = recovery.Filesystem(root, os.getuid())
            runtime_lock = fs.path(recovery.LOCK)
            runtime_lock.parent.mkdir(parents=True)
            runtime_lock.touch()
            # No files or scriptlets are installed, even after releasing the lock.
            command = ["rpm", "--dbpath", str(database), "--upgrade", "--justdb",
                       "--nodeps", "--noscripts", "--nosignature", "--ignoresize", str(package)]
            with fs.locked("/rpmdb/.rpm.lock"):
                busy = subprocess.run(command, capture_output=True, env=dict(os.environ, LC_ALL="C"))
                self.assertNotEqual(busy.returncode, 0)
                self.assertIn(b"lock", busy.stderr.lower())
            success = subprocess.run(command, capture_output=True)
            self.assertEqual(success.returncode, 0, success.stderr.decode())
            self.assertFalse(Path("/usr/share/nvb-recovery-lock-fixture").exists())


if __name__ == "__main__":
    unittest.main()


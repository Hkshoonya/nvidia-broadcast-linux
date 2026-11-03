"""The signed RPM makes recovery available before staging the first payload."""
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rpm_bootstrap", ROOT / "packaging/native-runtime/recovery_bootstrap.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.umask, os.umask(0o022))
        temporary = tempfile.TemporaryDirectory(prefix="rpm-bootstrap-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "var/lib/nvbroadcast"
        self.state.mkdir(parents=True)
        self.directory = self.state / "recovery-tools"
        self.payload = b"#!/usr/bin/python3 -IB\nprint('authenticated recovery')\n"
        self.target = self.directory / ("recover-" + hashlib.sha256(self.payload).hexdigest() + ".py")
        self.owner = os.getuid()

    def run_bootstrap(self, action="prepare"):
        return module.bootstrap(action, self.payload, root=self.root, owner=self.owner)

    def install_helper(self, payload=None):
        installed = self.root / "usr/bin/nvbroadcast-recover-rpm"
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(self.payload if payload is None else payload)
        installed.chmod(0o755)
        return installed

    def test_prepare_is_durable_content_addressed_and_idempotent(self):
        with mock.patch.object(module, "sync", wraps=module.sync) as sync:
            self.assertEqual(self.run_bootstrap(), self.target)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
        sync.assert_any_call(self.directory)
        sync.assert_any_call(self.state)
        inode = self.target.stat().st_ino
        self.run_bootstrap()
        self.assertEqual(self.target.stat().st_ino, inode)
        self.assertEqual(list(self.directory.iterdir()), [self.target])
        self.assertFalse((self.root / "usr/lib/nvbroadcast/.transaction").exists())

    def test_changed_existing_copy_is_not_overwritten(self):
        self.run_bootstrap()
        self.target.write_bytes(b"different")
        with self.assertRaises(ValueError):
            self.run_bootstrap()
        self.assertEqual(self.target.read_bytes(), b"different")

    def test_symlink_hardlink_fifo_and_unsafe_file_mode_are_refused(self):
        self.directory.mkdir(mode=0o700)
        other = self.root / "unrelated"
        other.write_bytes(self.payload)
        other.chmod(0o600)
        operations = (
            lambda: self.target.symlink_to(other),
            lambda: os.link(other, self.target),
            lambda: os.mkfifo(self.target, 0o600),
            lambda: (self.target.write_bytes(self.payload), self.target.chmod(0o644)),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                operation()
                with self.assertRaises((ValueError, OSError)):
                    self.run_bootstrap()
                self.assertEqual(other.read_bytes(), self.payload)
                self.target.unlink()

    def test_redirected_or_writable_parent_is_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.directory.symlink_to(outside)
        with self.assertRaises((ValueError, OSError)):
            self.run_bootstrap()
        self.assertEqual(list(outside.iterdir()), [])
        self.directory.unlink()
        for directory, mode in ((self.state, 0o777), (self.directory, 0o755)):
            with self.subTest(directory=directory):
                directory.mkdir(exist_ok=True)
                directory.chmod(mode)
                with self.assertRaises(ValueError):
                    self.run_bootstrap()
                self.assertFalse(self.target.exists())
                directory.chmod(0o700)

    def test_wrong_directory_and_file_ownership_are_refused(self):
        with self.assertRaises(ValueError):
            module.bootstrap("prepare", self.payload, root=self.root, owner=self.owner + 1)
        self.run_bootstrap()
        with mock.patch.object(module.os, "fstat", wraps=os.fstat) as fstat:
            actual = self.target.stat()
            changed = list(actual)
            changed[4] = self.owner + 1
            fstat.return_value = os.stat_result(changed)
            with self.assertRaises(ValueError):
                self.run_bootstrap()

    def test_missing_or_changed_installed_helper_preserves_bootstrap(self):
        self.run_bootstrap()
        with self.assertRaises(FileNotFoundError):
            self.run_bootstrap("finish")
        installed = self.install_helper(b"wrong executable")
        with self.assertRaises(ValueError):
            self.run_bootstrap("finish")
        installed.write_bytes(self.payload)
        installed.chmod(0o777)
        with self.assertRaises(ValueError):
            self.run_bootstrap("finish")
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_finish_removes_only_current_validated_copy(self):
        self.run_bootstrap()
        self.install_helper()
        older = self.directory / "recover-older-failed-transaction.py"
        older.write_bytes(b"keep failed transaction evidence")
        self.run_bootstrap("finish")
        self.assertFalse(self.target.exists())
        self.assertEqual(older.read_bytes(), b"keep failed transaction evidence")
        older.unlink()
        self.run_bootstrap()
        self.run_bootstrap("finish")
        self.assertFalse(self.directory.exists())
        self.assertTrue(self.state.is_dir())

    def test_old_python_and_invalid_operations_make_no_bootstrap(self):
        with mock.patch.object(module.sys, "version_info", (3, 10, 9)):
            with self.assertRaisesRegex(ValueError, "Python 3.11"):
                self.run_bootstrap()
        for action, payload in (("remove", self.payload), ("prepare", b""), ("prepare", b"x" * 131073)):
            with self.subTest(action=action, size=len(payload)):
                with self.assertRaises(ValueError):
                    module.bootstrap(action, payload, root=self.root, owner=self.owner)
        self.assertFalse(self.directory.exists())

    def test_write_failure_leaves_no_partial_executable(self):
        with mock.patch.object(module.os, "fsync", side_effect=OSError("interrupted write")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.run_bootstrap()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_rendered_prepare_has_no_payload_execution_or_expansion(self):
        # Load the adapter in a subprocess, because it imports prototype helpers.
        script = """
import importlib.util, pathlib, subprocess
root = pathlib.Path.cwd()
spec = importlib.util.spec_from_file_location('adapter', root / 'packaging/native-runtime/build.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
for family in ('rpm', 'deb'):
    for action in ('prepare', 'prepare-remove', 'finish', 'remove'):
        hook = adapter.hook(action, '1.5.3-3.cpu', 'cpu', family)
        assert '@RPM_' not in hook
        subprocess.run(['sh', '-n'], input=hook, text=True, check=True)
        if family == 'rpm' and action in ('prepare', 'finish'):
            assert '/usr/bin/python3 -IB - "$1"' in hook
            begin = hook.index("<<'NVB_BOOTSTRAP_PY'\\n") + len("<<'NVB_BOOTSTRAP_PY'\\n")
            end = hook.index('\\nNVB_BOOTSTRAP_PY', begin)
            compile(hook[begin:end], '<signed bootstrap>', 'exec')
        else:
            assert 'recovery_bootstrap()' not in hook
            assert 'recovery_bootstrap prepare' not in hook
hook = adapter.hook('prepare', '1.5.3-3.cpu', 'cpu', 'rpm')
assert hook.index('recovery_bootstrap prepare') < hook.index('marker=$(mktemp')
hook = adapter.hook('finish', '1.5.3-3.cpu', 'cpu', 'rpm')
assert hook.index('validate-install.py" --variant') < hook.index('recovery_bootstrap finish')
assert hook.index('recovery_bootstrap finish') < hook.index('rm -f "$base/.transaction"')
"""
        subprocess.run([os.sys.executable, "-IB", "-c", script], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()

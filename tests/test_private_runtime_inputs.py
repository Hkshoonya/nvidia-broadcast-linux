"""Integrity and offline boundary tests for the non-release runtime prototype."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock


PROTOTYPE = Path(__file__).resolve().parents[1] / "packaging/runtime-prototype"
SPEC = importlib.util.spec_from_file_location("private_runtime_prepare", PROTOTYPE / "prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrivateRuntimeInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b"verified input"
        self.pin = {"url": "https://example.invalid/input.whl",
                    "sha256": hashlib.sha256(self.data).hexdigest()}

    def test_cached_input_is_checked_without_network(self):
        path = self.root / "input.whl"
        path.write_bytes(self.data)
        with mock.patch.object(prepare.urllib.request, "urlopen") as network:
            self.assertEqual(prepare.fetch(self.root, self.pin, offline=True), path)
            network.assert_not_called()

    def test_tampered_cache_fails_without_redownloading(self):
        (self.root / "input.whl").write_bytes(b"tampered")
        with mock.patch.object(prepare.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare.fetch(self.root, self.pin)
            network.assert_not_called()

    def test_offline_missing_input_never_uses_network(self):
        with mock.patch.object(prepare.urllib.request, "urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                prepare.fetch(self.root, self.pin, offline=True)
            network.assert_not_called()

    def test_bad_download_is_not_published_and_preserves_other_partial(self):
        other = self.root / "input.whl.someone-else.partial"
        other.write_bytes(b"in progress")
        with mock.patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare.fetch(self.root, self.pin)
        self.assertEqual(list(self.root.iterdir()), [other])
        self.assertEqual(other.read_bytes(), b"in progress")

    def test_https_and_simple_filename_required(self):
        for url in ("http://example.invalid/a.whl", "https://example.invalid/%2e%2e",
                    "https://example.invalid/%2foutside", "https://example.invalid/"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                prepare.filename({"url": url})

    def test_tar_rejects_traversal_absolute_device_and_escaping_links(self):
        cases = [("../escape", tarfile.REGTYPE, ""), ("/escape", tarfile.REGTYPE, ""),
                 ("device", tarfile.CHRTYPE, ""), ("link", tarfile.SYMTYPE, "../../escape"),
                 ("link", tarfile.LNKTYPE, "/escape")]
        for name, kind, target in cases:
            with self.subTest(name=name, kind=kind):
                member = tarfile.TarInfo(name)
                member.type, member.linkname = kind, target
                with self.assertRaises((ValueError, tarfile.FilterError)):
                    prepare.safe_member(member, str(self.root))

    def test_local_wheel_hash_and_marker_survive_lock_materialization(self):
        local = self.root / "built"
        local.mkdir()
        (local / "input.whl").write_bytes(self.data)
        lock = self.root / "pylock.toml"
        lock.write_text('lock-version = "1.0"\n[[packages]]\nname = "demo"\nversion = "1.0"\n'
                        'marker = "sys_platform == \'linux\'"\n'
                        'archive = {path = "unavailable/input.whl", hashes = {sha256 = "'
                        + self.pin["sha256"] + '"}}\n')
        with mock.patch.object(prepare.urllib.request, "urlopen") as network:
            requirements = prepare.fetch_lock(lock, self.root / "cache", local, offline=True)
            network.assert_not_called()
        self.assertEqual((self.root / "cache/wheels/input.whl").read_bytes(), self.data)
        self.assertEqual(requirements.read_text(),
                         "demo==1.0 ; sys_platform == 'linux' --hash=sha256:" + self.pin["sha256"] + "\n")
        (local / "input.whl").write_bytes(b"tampered rebuild")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            prepare.fetch_lock(lock, self.root / "another-cache", local, offline=True)
        self.assertFalse((self.root / "another-cache/requirements.txt").exists())

    def test_sdist_only_lock_cannot_trigger_build(self):
        lock = self.root / "pylock.toml"
        lock.write_text('lock-version = "1.0"\n[[packages]]\nname = "demo"\nversion = "1.0"\n'
                        'sdist = {url = "https://example.invalid/demo.tar.gz"}\n')
        with mock.patch.object(prepare.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "no wheel"):
                prepare.fetch_lock(lock, self.root / "cache")
            network.assert_not_called()

    def _python_archive(self, version="3.13.16", license_path="licenses/LICENSE.cpython.txt"):
        metadata = {"python_version": version, "target_triple": "x86_64-unknown-linux-gnu",
                    "python_tag": "cp313", "license_path": license_path}
        contents = {"python/PYTHON.json": json.dumps(metadata).encode(),
                    "python/licenses/LICENSE.cpython.txt": b"upstream license record",
                    "python/licenses/LICENSE.dependency.txt": b"dependency notice",
                    "python/install/bin/python3.13": b"test interpreter payload",
                    "python/build/not-shipped": b"discarded"}
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, data in contents.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        path = self.root / "python.tar.zst"
        path.write_bytes(buffer.getvalue())
        item = {"sha256": prepare.digest(path), "version": "3.13.16"}
        process = mock.Mock(stdout=io.BytesIO(buffer.getvalue()))
        process.wait.return_value = 0
        return path, item, process

    def test_unpack_retains_all_license_records_and_metadata(self):
        path, item, process = self._python_archive()
        with mock.patch.object(prepare.subprocess, "Popen", return_value=process):
            record = prepare.unpack_python(path, item, self.root / "unpacked")
        runtime = self.root / "unpacked/runtime"
        self.assertEqual(set(record["license_files"]),
                         {"LICENSE.cpython.txt", "LICENSE.dependency.txt"})
        self.assertEqual((runtime / "bin/python").read_bytes(), b"test interpreter payload")
        self.assertTrue((runtime / "share/nvbroadcast-runtime-provenance/PYTHON.json").is_file())
        self.assertFalse((self.root / "unpacked/python").exists())

    def test_unpack_rejects_mismatched_target_and_license_outside_notices(self):
        for options in ({"version": "3.12.0"}, {"license_path": "install/bin/python3.13"}):
            with self.subTest(options=options):
                path, item, process = self._python_archive(**options)
                with mock.patch.object(prepare.subprocess, "Popen", return_value=process):
                    with self.assertRaises(ValueError):
                        prepare.unpack_python(path, item, self.root / str(len(options)) / next(iter(options)))

    def test_bad_archive_hash_is_rejected_before_decompression(self):
        path = self.root / "input.whl"
        path.write_bytes(b"tampered")
        with mock.patch.object(prepare.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare.unpack_python(path, self.pin, self.root / "unpacked")
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()

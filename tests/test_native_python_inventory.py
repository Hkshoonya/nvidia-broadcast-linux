"""Metadata projection bounds and payload bindings, without package execution."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import tarfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_python_inventory", ROOT / "packaging/native-runtime/python_inventory.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)
validation_spec = importlib.util.spec_from_file_location("native_validator", ROOT / "packaging/native-runtime/validate_candidate.py")
validator = importlib.util.module_from_spec(validation_spec)
validation_spec.loader.exec_module(validator)
PATH = inventory.PREFIX + "lib/python3.13/site-packages/example-1.0.dist-info/METADATA"
DATA = b"Metadata-Version: 2.4\nName: example\nVersion: 1.0\n"


class Tests(unittest.TestCase):
    def decoded(self, data, collector):
        process = mock.Mock(stdout=io.BytesIO(data), returncode=0)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with mock.patch.object(validator.subprocess, "Popen", return_value=process), \
                mock.patch.object(validator, "bounded", return_value=b""):
            return validator.decoded_inventory(Path("/fixture.deb"), "deb", metadata_collector=collector)

    def payload_tar(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            parents = [str(p).lstrip("/") for p in PurePosixPath(PATH).parents if str(p) != "/"]
            for parent in sorted(parents, key=len):
                member = tarfile.TarInfo(parent)
                member.type, member.mode = tarfile.DIRTYPE, 0o755
                archive.addfile(member)
            member = tarfile.TarInfo(PATH.lstrip("/"))
            member.size, member.mode = len(DATA), 0o644
            archive.addfile(member, io.BytesIO(DATA))
        return data.getvalue()

    def test_existing_decoder_only_buffers_metadata_when_opted_in(self):
        data = self.payload_tar()
        original, _, kept, _ = self.decoded(data, None)
        self.assertNotIn(PATH, kept)
        collector = inventory.MetadataCollector()
        actual, _, kept, _ = self.decoded(data, collector)
        self.assertEqual(original, actual)
        self.assertNotIn(PATH, kept)  # Separate projection must not enter signing transfer.
        self.assertEqual(collector.data, {PATH: DATA})
        manifest = {PATH.removeprefix(inventory.PREFIX): dict(actual[PATH], mode=0o644)}
        self.assertEqual(len(collector.verified_records(actual, manifest)), 1)

    def test_decoder_rejects_oversized_metadata_before_buffering(self):
        member = tarfile.TarInfo(PATH.lstrip("/"))
        member.size, member.mode = inventory.MAX_METADATA_BYTES + 1, 0o644
        collector = inventory.MetadataCollector()
        with self.assertRaisesRegex(ValueError, "METADATA size"):
            self.decoded(member.tobuf() + b"\0" * 1024, collector)
        self.assertEqual(collector.data, {})

    def inputs(self):
        collector = inventory.MetadataCollector()
        payload = {str(parent): {"directory": True} for parent in PurePosixPath(PATH).parents
                   if str(parent) == inventory.PREFIX.rstrip("/") or str(parent).startswith(inventory.PREFIX)}
        for path in sorted(payload, key=len):
            collector.select(path, "directory")
        self.assertTrue(collector.select(PATH, "file", len(DATA)))
        collector.accept(PATH, DATA)
        record = {"sha256": hashlib.sha256(DATA).hexdigest(), "size": len(DATA)}
        payload[PATH] = record
        return collector, payload, {PATH.removeprefix(inventory.PREFIX): dict(record, mode=0o644)}

    def test_exact_projection_and_manifest_bytes(self):
        collector, payload, manifest = self.inputs()
        self.assertEqual(collector.verified_records(payload, manifest), {
            PATH.removeprefix(inventory.PREFIX): {"bytes": len(DATA), "sha256": hashlib.sha256(DATA).hexdigest()}})

    def test_missing_or_extra_metadata_cannot_be_hidden(self):
        collector, payload, manifest = self.inputs()
        del collector.data[PATH]
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            collector.verified_records(payload, manifest)
        collector, payload, manifest = self.inputs()
        payload[PATH.replace("example-1.0", "hidden-2.0")] = dict(payload[PATH])
        with self.assertRaisesRegex(ValueError, "complete metadata"):
            collector.verified_records(payload, manifest)
        collector, payload, manifest = self.inputs()
        manifest[PATH.removeprefix(inventory.PREFIX).replace("example-1.0", "hidden-2.0")] = dict(payload[PATH])
        with self.assertRaisesRegex(ValueError, "complete runtime manifest"):
            collector.verified_records(payload, manifest)

    def test_changed_manifest_or_decoded_bytes_fail(self):
        for target in ("manifest", "payload", "data"):
            collector, payload, manifest = self.inputs()
            if target == "manifest":
                manifest[PATH.removeprefix(inventory.PREFIX)]["sha256"] = "0" * 64
            elif target == "payload":
                payload[PATH]["size"] += 1
            else:
                collector.data[PATH] = b"X" * len(DATA)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "authenticated runtime"):
                collector.verified_records(payload, manifest)

    def test_generator_discoverable_bare_python_root_cannot_disappear(self):
        collector, payload, manifest = self.inputs()
        hidden = PATH.replace("python3.13/", "python/")
        payload[hidden] = dict(payload[PATH])
        with self.assertRaisesRegex(ValueError, "complete metadata"):
            collector.verified_records(payload, manifest)
        self.assertTrue(inventory.MetadataCollector().select(hidden, "file", len(DATA)))

    def test_source_parent_links_are_not_erased_by_projection(self):
        collector, payload, manifest = self.inputs()
        parent = str(PurePosixPath(PATH).parent)
        payload[parent] = {"symlink": "elsewhere"}
        with self.assertRaisesRegex(ValueError, "parent is missing or redirected"):
            collector.verified_records(payload, manifest)

    def test_missing_metadata_and_egg_entries_fail(self):
        collector, payload, manifest = self.inputs()
        collector.select(PATH.replace("example-1.0", "missing-2.0").removesuffix("/METADATA"), "directory")
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            collector.verified_records(payload, manifest)
        for suffix in ("bad.egg-info", "bad.EGG", "bad.EGG-INFO/file"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "egg"):
                collector.select(inventory.PREFIX + "lib/python3.13/site-packages/" + suffix, "file", 1)

    def test_bounds_are_reserved_before_reading_metadata(self):
        for size in (0, inventory.MAX_METADATA_BYTES + 1, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "size"):
                inventory.MetadataCollector().select(PATH, "file", size)
        collector = inventory.MetadataCollector()
        collector.total_bytes = inventory.MAX_TOTAL_METADATA_BYTES
        with self.assertRaisesRegex(ValueError, "aggregate"):
            collector.select(PATH, "file", 1)
        collector = inventory.MetadataCollector()
        collector.distributions = {str(number) for number in range(inventory.MAX_DISTRIBUTIONS)}
        with self.assertRaisesRegex(ValueError, "too many"):
            collector.select(PATH, "file", 1)

    def test_duplicate_or_changed_decode_is_rejected(self):
        collector = inventory.MetadataCollector()
        collector.select(PATH, "file", len(DATA))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collector.select(PATH, "file", len(DATA))
        with self.assertRaisesRegex(ValueError, "size changed"):
            collector.accept(PATH, DATA + b"x")
        collector.accept(PATH, DATA)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collector.accept(PATH, DATA)

    def test_paths_and_metadata_entry_types_fail_closed(self):
        for path in (PATH.replace("/example-1.0", "/../example-1.0"), PATH.replace("/METADATA", "\\METADATA"), PATH + "\n"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "noncanonical"):
                inventory.MetadataCollector().select(path, "file", 1)
        with self.assertRaisesRegex(ValueError, "regular file"):
            inventory.MetadataCollector().select(PATH, "symlink", 1)

    def projection(self, directory):
        collector, payload, manifest = self.inputs()
        return inventory.write_projection(directory, collector, payload, manifest,
            lock_bytes=b'lock-version = "1.0"\n[[packages]]\nname = "example"\nversion = "1.0"\n',
            source_commit="a" * 40, source_date_epoch=1700000000,
            archive_identity={"name": "nvbroadcast_1.5.4-9.cpu_amd64.deb", "bytes": 123, "sha256": "b" * 64},
            runtime_manifest_sha256="c" * 64)

    def test_projection_files_are_fresh_and_relocatable_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            self.assertEqual(self.projection(first), self.projection(second))
            self.assertEqual((first / "projection.json").read_bytes(), (second / "projection.json").read_bytes())
            self.assertNotIn(str(root), (first / "projection.json").read_text())
            record = json.loads((first / "projection.json").read_text())
            for relative, identity in record["files"].items():
                data = (first / relative).read_bytes()
                self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (identity["bytes"], identity["sha256"]))
            with self.assertRaisesRegex(ValueError, "fresh"):
                self.projection(first)

    def test_projection_refuses_redirected_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").mkdir()
            (root / "link").symlink_to(root / "target", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "fresh nonredirected"):
                self.projection(root / "link/output")


if __name__ == "__main__":
    unittest.main()

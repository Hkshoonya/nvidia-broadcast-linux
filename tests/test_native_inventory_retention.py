"""Final archive-bound Python evidence and projection trust boundaries."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from jsonschema import Draft7Validator, FormatChecker
from referencing import Registry

from scripts import generate_python_sbom as sbom
from tests.test_native_candidate_finalization import CandidateFixture, SOURCE, RUNTIME, finalize as fixture_finalizer
from tests.test_python_sbom import SCHEMA, SCHEMA_SHA256, no_remote_reference


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inventory_retention", ROOT / "packaging/native-runtime/retain_python_inventory.py")
retain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retain)


class HostToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nvb-schema-host-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "wheels").mkdir()

    def wheel(self, members):
        path = self.root / "wheels/example.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in members:
                archive.writestr(name, value)
        return [{"filename": path.name, "name": "example", "version": "1.0", "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]

    def test_verified_extraction_rechecks_original_wheel_and_every_member(self):
        pins = self.wheel([("example/__init__.py", "VALUE = 1\n")])
        site = self.root / "site"
        (site / "example").mkdir(parents=True)
        (site / "example/__init__.py").write_text("VALUE = 1\n")
        with mock.patch.object(retain.host_tools, "inputs", return_value=pins):
            self.assertEqual(retain.host_tools.verified_site(self.root)[0], site)
            (site / "example/__init__.py").write_text("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                retain.host_tools.verified_site(self.root)

    def test_unlisted_code_and_original_wheel_tamper_are_rejected(self):
        pins = self.wheel([("example.py", "VALUE = 1\n")])
        site = self.root / "site"
        site.mkdir()
        (site / "example.py").write_text("VALUE = 1\n")
        with mock.patch.object(retain.host_tools, "inputs", return_value=pins):
            (site / "injected.py").write_text("raise RuntimeError('not imported')\n")
            with self.assertRaises(ValueError):
                retain.host_tools.verified_site(self.root)
            (site / "injected.py").unlink()
            with (self.root / "wheels/example.whl").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "source pin"):
                retain.host_tools.verified_site(self.root)

    def test_traversal_pth_duplicate_and_scheme_members_are_refused(self):
        for members in (([("../escape.py", "x")]), ([("execute.pth", "x")]),
                        ([("wheel.data/purelib/a.py", "x")]), ([("a.py", "x"), ("a.py", "y")])):
            pins = self.wheel(members)
            with self.subTest(members=members), self.assertRaises(ValueError):
                retain.host_tools.wheel_files(self.root / "wheels", pins)

    def test_expansion_and_member_count_are_reserved_before_read(self):
        pins = self.wheel([("first.py", "abc"), ("second.py", "def")])
        for key, value in (("MAX_EXPANDED", 5), ("MAX_MEMBERS", 1)):
            with self.subTest(bound=key), mock.patch.object(retain.host_tools, key, value), self.assertRaises(ValueError):
                retain.host_tools.wheel_files(self.root / "wheels", pins)

    def test_redirected_extracted_file_is_never_imported(self):
        pins = self.wheel([("example.py", "VALUE = 1\n")])
        site = self.root / "site"
        site.mkdir()
        (site / "example.py").symlink_to(self.root / "wheels/example.whl")
        with mock.patch.object(retain.host_tools, "inputs", return_value=pins), self.assertRaisesRegex(ValueError, "redirected"):
            retain.host_tools.verified_site(self.root)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nvb-inventory-evidence-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.projection = self.root / "projection"
        self.path = retain.projection_tool.PREFIX + "lib/python3.13/site-packages/example-1.0.dist-info/METADATA"
        data = b"Metadata-Version: 2.4\nName: example\nVersion: 1.0\n"
        collector = retain.projection_tool.MetadataCollector()
        payload = {str(parent): {"directory": True} for parent in PurePosixPath(self.path).parents
                   if str(parent) == retain.projection_tool.PREFIX.rstrip("/") or str(parent).startswith(retain.projection_tool.PREFIX)}
        for path in sorted(payload, key=len):
            collector.select(path, "directory")
        collector.select(self.path, "file", len(data))
        collector.accept(self.path, data)
        payload[self.path] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        source_manifest = {self.path.removeprefix(retain.projection_tool.PREFIX): dict(payload[self.path], mode=0o644)}
        self.manifest = {"producer": {"source": "a" * 40}, "runtime_manifest_sha256": "c" * 64,
                         "packages": {"deb": {"path": "deb/nvbroadcast_1.5.4-9.cpu_amd64.deb", "bytes": 123, "sha256": "b" * 64}}}
        self.receipt = retain.projection_tool.write_projection(
            self.projection, collector, payload, source_manifest,
            lock_bytes=b'lock-version = "1.0"\n[[packages]]\nname = "example"\nversion = "1.0"\n',
            source_commit="a" * 40, source_date_epoch=1700000000,
            archive_identity={"name": "nvbroadcast_1.5.4-9.cpu_amd64.deb", "bytes": 123, "sha256": "b" * 64},
            runtime_manifest_sha256="c" * 64)
        self.binding = {"receipt_sha256": retain.finalize.sha(self.projection / "projection.json"),
                        "receipt_bytes": (self.projection / "projection.json").stat().st_size,
                        "metadata_count": self.receipt["metadata_count"], "metadata_bytes": self.receipt["metadata_bytes"],
                        "driver_sha256": retain.finalize.sha(ROOT / "packaging/native-runtime/python_inventory.py")}

    def check(self):
        return retain.checked_projection(self.projection, self.binding, self.manifest, 1700000000)

    def rewrite_receipt(self):
        path = self.projection / "projection.json"
        path.write_text(json.dumps(self.receipt, sort_keys=True, indent=2) + "\n")
        self.binding.update(receipt_sha256=retain.finalize.sha(path), receipt_bytes=path.stat().st_size)

    def test_bound_projection_has_only_verified_metadata_and_assembled_lock(self):
        receipt, runtime, lock = self.check()
        self.assertEqual(receipt, self.receipt)
        self.assertEqual(runtime, self.projection / "runtime")
        self.assertEqual(lock, self.projection / "context/assembled-pylock.toml")

    def test_receipt_tamper_fails_external_validation_anchor(self):
        with (self.projection / "projection.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "authenticated validation"):
            self.check()

    def test_self_consistent_wrong_source_epoch_archive_or_runtime_is_rejected(self):
        original = copy.deepcopy(self.receipt)
        for field, value in (("source_commit", "d" * 40), ("source_date_epoch", 1700000001),
                             ("runtime_manifest_sha256", "d" * 64), ("archive", {"name": "wrong.deb"})):
            self.receipt = copy.deepcopy(original)
            self.receipt[field] = value
            self.rewrite_receipt()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "context differs"):
                self.check()

    def test_changed_metadata_or_lock_fails_before_generation(self):
        for relative in self.receipt["files"]:
            path = self.projection / relative
            previous = path.read_bytes()
            path.write_bytes(previous + b"\n")
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "reviewed identity"):
                self.check()
            path.write_bytes(previous)

    def test_unlisted_code_or_empty_directory_cannot_be_transferred(self):
        extra = self.projection / "runtime/lib/python3.13/site-packages/execute.pth"
        extra.write_text("raise RuntimeError('must never run')\n")
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            self.check()
        extra.unlink()
        (self.projection / "extra-directory").mkdir()
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            self.check()

    def test_redirected_metadata_directory_is_rejected(self):
        path = self.projection / "runtime/lib/python3.13/site-packages/example-1.0.dist-info"
        relocated = self.root / "redirected"
        shutil.move(path, relocated)
        path.symlink_to(relocated, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "link or special"):
            self.check()

    def test_metadata_count_and_aggregate_cannot_be_rebound(self):
        for field in ("metadata_count", "metadata_bytes"):
            original = self.receipt[field]
            self.receipt[field] += 1
            self.binding[field] += 1
            self.rewrite_receipt()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check()
            self.receipt[field] = original
            self.binding[field] = original

    def test_stale_projection_driver_is_rejected(self):
        self.binding["driver_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "authenticated validation"):
            self.check()


class InventoryRetentionTests(CandidateFixture):
    def setUp(self):
        super().setUp()
        self.projection_fixture = ProjectionTests("test_bound_projection_has_only_verified_metadata_and_assembled_lock")
        self.projection_fixture.setUp()
        self.addCleanup(self.projection_fixture.doCleanups)
        self.projection = self.projection_fixture.projection
        projected = self.projection_fixture.receipt
        deb = self.report["packages"]["deb"]
        projected.update(source_commit=SOURCE, runtime_manifest_sha256=RUNTIME,
                         archive={"name": PurePosixPath(deb["relative_path"]).name,
                                  "bytes": deb["bytes"], "sha256": deb["sha256"]})
        self.projection_fixture.rewrite_receipt()
        self.report["python_inventory_projection"] = self.projection_fixture.binding
        self.save_receipt()
        self.build_bundle()
        deb_result, _ = self.execution("deb")
        rpm_result, _ = self.execution("rpm")
        self.execution("rpm", recovery=True)
        self.qualified = self.root / "qualified"
        fixture_finalizer.complete(self.output, self.manifest_hash, deb_result.parent.parent,
                                   rpm_result.parent.parent, self.qualified)
        patcher = mock.patch.object(retain, "finalize", fixture_finalizer)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(retain, "source_epoch", return_value=1700000000)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.validator = Draft7Validator(json.loads(SCHEMA.read_text()), registry=Registry(retrieve=no_remote_reference))
        patcher = mock.patch.object(retain, "load_generator", return_value=(
            sbom, {"fixture": "real generator; isolated host bootstrap tested separately"}, self.validator))
        self.generator_loader = patcher.start()
        self.addCleanup(patcher.stop)

    def retain(self, expected=None):
        return retain.retain_inventory(self.qualified, self.manifest_hash if expected is None else expected,
                                       self.projection, Path("/unused-fixture-host-wheel"), Path("/unused-fixture-schema-tools"))

    def test_final_deb_and_signed_rpm_have_schema_valid_archive_bound_sidecars(self):
        original = {path.relative_to(self.qualified / "packages").as_posix(): path.read_bytes()
                    for path in (self.qualified / "packages").rglob("*") if path.is_file()}
        result = self.retain()
        self.assertEqual(result["status"], "partial-python-inventory-evidence")
        self.assertEqual(hashlib.sha256(SCHEMA.read_bytes()).hexdigest(), SCHEMA_SHA256)
        validator = Draft7Validator(json.loads(SCHEMA.read_text()), registry=Registry(retrieve=no_remote_reference),
                                    format_checker=FormatChecker())
        for family in ("deb", "rpm"):
            path = self.qualified / "python-inventory" / (family + ".python-inventory.spdx.json")
            document = json.loads(path.read_text())
            validator.validate(document)
            self.assertEqual(document["packages"][0]["checksums"][0]["checksumValue"], self.manifest["packages"][family]["sha256"])
            self.assertTrue(all(package["filesAnalyzed"] is False for package in document["packages"]))
            self.assertEqual(sum(edge["relationshipType"] == "METAFILE_OF" for edge in document["relationships"]), 1)
        self.assertNotEqual(self.manifest["packages"]["rpm"]["sha256"], self.report["packages"]["rpm"]["sha256"])
        self.assertEqual(original, {path.relative_to(self.qualified / "packages").as_posix(): path.read_bytes()
                                    for path in (self.qualified / "packages").rglob("*") if path.is_file()})

    def test_external_signing_job_manifest_is_required_before_generator_import(self):
        with self.assertRaisesRegex(ValueError, "candidate manifest differs"):
            self.retain("0" * 64)
        self.generator_loader.assert_not_called()
        self.assertFalse((self.qualified / "python-inventory").exists())

    def test_self_asserted_qualification_cannot_replace_checked_execution(self):
        path = self.qualified / "qualification.json"
        report = json.loads(path.read_text())
        report["qualification"]["deb"]["sha256"] = "0" * 64
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "qualification evidence changed"):
            self.retain()
        self.generator_loader.assert_not_called()

    def test_repeated_generation_is_byte_identical_and_fresh_output_is_required(self):
        self.retain()
        output = self.qualified / "python-inventory"
        first = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.retain()
        shutil.rmtree(output)  # Only this test's own fresh evidence directory.
        self.retain()
        self.assertEqual(first, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_projection_change_during_generation_prevents_success_receipt(self):
        count = 0
        def generate(*args):
            nonlocal count
            document = sbom.build_document(*args)
            count += 1
            if count == 2:
                metadata = self.projection / "runtime/lib/python3.13/site-packages/example-1.0.dist-info/METADATA"
                metadata.write_bytes(metadata.read_bytes() + b"\n")
            return document
        self.generator_loader.return_value = (SimpleNamespace(build_document=generate,
            installed_metadata=sbom.installed_metadata, SCOPE=sbom.SCOPE), {}, self.validator)
        with self.assertRaisesRegex(ValueError, "reviewed identity|installed metadata differs"):
            self.retain()
        self.assertFalse((self.qualified / "python-inventory/receipt.json").exists())

    def test_each_family_is_generated_twice_before_success(self):
        with mock.patch.object(sbom, "build_document", wraps=sbom.build_document) as build:
            result = self.retain()
        self.assertEqual(build.call_count, 4)
        self.assertTrue(all(item["repeat_identical"] and item["official_schema_valid"]
                            and item["metadata_semantics_valid"] for item in result["inventories"].values()))

    def altered_generator(self, change):
        def generate(*args):
            document = sbom.build_document(*args)
            change(document)
            return document
        self.generator_loader.return_value = (SimpleNamespace(build_document=generate,
            installed_metadata=sbom.installed_metadata, SCOPE=sbom.SCOPE), {}, self.validator)

    def test_nondeterminism_refuses_success_receipt(self):
        count = 0
        def change(document):
            nonlocal count
            count += 1
            document["name"] += str(count)
        self.altered_generator(change)
        with self.assertRaisesRegex(ValueError, "repeated inventory"):
            self.retain()
        self.assertFalse((self.qualified / "python-inventory/receipt.json").exists())

    def test_schema_invalid_output_is_rejected_before_retention(self):
        self.altered_generator(lambda document: document.pop("dataLicense"))
        with self.assertRaisesRegex(ValueError, "official schema"):
            self.retain()
        self.assertFalse((self.qualified / "python-inventory/receipt.json").exists())

    def test_schema_valid_wrong_relationship_or_license_scope_is_rejected(self):
        for change in (lambda doc: doc["relationships"][-1].update(relationshipType="CONTAINS"),
                       lambda doc: doc["packages"][1].update(filesAnalyzed=True),
                       lambda doc: doc["files"][0]["checksums"][1].update(checksumValue="0" * 64)):
            self.altered_generator(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.retain()
            shutil.rmtree(self.qualified / "python-inventory")


if __name__ == "__main__":
    unittest.main()

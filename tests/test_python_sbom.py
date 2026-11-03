"""Read-only Python inventories, validated with the pinned upstream schema."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from jsonschema import Draft7Validator, FormatChecker
from referencing import Registry

from scripts import generate_python_sbom as sbom


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "tests/data/spdx-2.3/schema.json"
SCHEMA_SHA256 = "239208b7ac287b3cf5d9a9af23f9d69863971102a5e1587a27a398b43490b89b"


def no_remote_reference(uri):
    raise AssertionError(f"Schema validation attempted an external reference: {uri}")


class PythonSbomTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.root = self.directory / "runtime"
        self.site = self.root / "lib/python3.13/site-packages"
        self.site.mkdir(parents=True)
        self.artifact = self.directory / "candidate.deb"
        self.artifact.write_bytes(b"fixture archive identity, not an actual package")
        self.lock = self.directory / "pylock.fixture.toml"
        self.lock.write_text('lock-version="1.0"\ncreated-by="fixture"\n'
                             '[[packages]]\nname="example"\nversion="1.0"\n')
        self.metadata("example", "1.0")

    def metadata(self, name, version, *, site=None, extra=""):
        directory = (site or self.site) / f"{name.replace('-', '_')}-{version}.dist-info"
        directory.mkdir(parents=True)
        path = directory / "METADATA"
        path.write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n{extra}")
        return path

    def document(self, **kwargs):
        args = dict(runtime_root=self.root, architecture="amd64", artifact=self.artifact,
                    source_commit="a" * 40, source_date_epoch=1700000000, locks=(self.lock,))
        args.update(kwargs)
        return sbom.build_document(**args)

    def validate(self, document):
        data = SCHEMA.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), SCHEMA_SHA256)
        schema = json.loads(data)
        Draft7Validator.check_schema(schema)
        validator = Draft7Validator(schema, registry=Registry(retrieve=no_remote_reference),
                                    format_checker=FormatChecker())
        validator.validate(document)

    def test_inventory_passes_official_spdx_schema_and_hashes_real_metadata(self):
        self.metadata("Pillow", "13.0.0", extra="License-Expression: an-unverified-license\n")
        document = self.document()
        self.validate(document)
        self.assertEqual([p["name"] for p in document["packages"]],
                         ["candidate.deb", "example", "pillow"])
        self.assertTrue(all(p["filesAnalyzed"] is False for p in document["packages"]))
        self.assertTrue(all(p["licenseDeclared"] == "NOASSERTION" for p in document["packages"]))
        self.assertEqual(document["packages"][0]["checksums"][0]["checksumValue"],
                         hashlib.sha256(self.artifact.read_bytes()).hexdigest())
        for record in document["files"]:
            path = self.root / record["fileName"].removeprefix("./")
            self.assertIn({"algorithm": "SHA256", "checksumValue": hashlib.sha256(path.read_bytes()).hexdigest()},
                          record["checksums"])
        self.assertIn("not authenticated or verified", document["comment"])
        self.assertIn(hashlib.sha256(self.lock.read_bytes()).hexdigest(), document["packages"][0]["sourceInfo"])

    def test_unanalyzed_packages_have_metadata_edges_without_contained_files(self):
        document = self.document()
        file_ids = {record["SPDXID"] for record in document["files"]}
        unanalyzed = {package["SPDXID"] for package in document["packages"] if not package["filesAnalyzed"]}
        for relationship in document["relationships"]:
            self.assertFalse(relationship["spdxElementId"] in unanalyzed
                             and relationship["relationshipType"] == "CONTAINS"
                             and relationship["relatedSpdxElement"] in file_ids)
        metadata_edges = [edge for edge in document["relationships"] if edge["relationshipType"] == "METAFILE_OF"]
        self.assertEqual(len(metadata_edges), len(file_ids))
        self.assertTrue(all(edge["spdxElementId"] in file_ids and edge["relatedSpdxElement"] in unanalyzed
                            for edge in metadata_edges))

    def test_same_inputs_are_deterministic_and_host_directory_is_not_identity(self):
        document = self.document()
        self.assertEqual(document, self.document())
        renamed = self.directory / "relocated"
        self.root.rename(renamed)
        self.assertEqual(document, self.document(runtime_root=renamed))
        self.assertNotIn(str(self.directory), json.dumps(document))

    def test_source_artifact_lock_metadata_and_timestamp_change_document_identity(self):
        original = self.document()["documentNamespace"]
        self.assertNotEqual(original, self.document(source_commit="b" * 40)["documentNamespace"])
        self.assertNotEqual(original, self.document(source_date_epoch=1700000001)["documentNamespace"])
        for path in (self.artifact, self.lock, self.site / "example-1.0.dist-info/METADATA"):
            previous = path.read_bytes()
            path.write_bytes(previous + b"\n")
            with self.subTest(path=path.name):
                self.assertNotEqual(original, self.document()["documentNamespace"])
            path.write_bytes(previous)

    def test_no_artifact_code_pth_target_interpreter_or_network_is_run(self):
        sentinel = self.directory / "executed"
        (self.site / "example.py").write_text(f"raise RuntimeError({str(sentinel)!r})\n")
        (self.site / "execute.pth").write_text(f"import pathlib; pathlib.Path({str(sentinel)!r}).touch()\n")
        (self.root / "bin").mkdir()
        (self.root / "bin/python").write_text("never run the target interpreter")
        with mock.patch("subprocess.run", side_effect=AssertionError("subprocess")), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            self.validate(self.document())
        self.assertFalse(sentinel.exists())

    def test_duplicate_canonical_names_across_roots_are_rejected(self):
        other = self.root / "usr/lib/python3.13/site-packages"
        self.metadata("Example", "1.0", site=other)
        with self.assertRaisesRegex(ValueError, "Duplicate installed distribution"):
            self.document()

    def test_outside_root_and_redirected_metadata_are_rejected(self):
        path = self.site / "example-1.0.dist-info/METADATA"
        external = self.directory / "external"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "regular input"):
            self.document()
        path.unlink()
        path.write_bytes(external.read_bytes())
        self.site.rename(self.directory / "external-site")
        self.site.symlink_to(self.directory / "external-site", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "contained app-owned"):
            self.document()

    def test_redirected_distribution_directory_and_root_are_rejected(self):
        directory = self.site / "example-1.0.dist-info"
        directory.rename(self.directory / "external-info")
        directory.symlink_to(self.directory / "external-info", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Redirected"):
            self.document()
        alias = self.directory / "runtime-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "not a symlink"):
            self.document(runtime_root=alias)

    def test_missing_or_ambiguous_metadata_is_rejected(self):
        path = self.site / "example-1.0.dist-info/METADATA"
        for content in ("Name: example\n", "Name: example\nName: other\nVersion: 1.0\n",
                        "Name: ../example\nVersion: 1.0\n", "Name: example\nVersion: ?\n",
                        "Name: example-\nVersion: 1.0\n", "Name: example.\nVersion: 1.0\n",
                        "Name: example_\nVersion: 1.0\n"):
            path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.document()
        path.write_bytes(b"Name: \xff\nVersion: 1.0\n")
        with self.assertRaisesRegex(ValueError, "Invalid installed distribution name"):
            self.document()

    def test_unsupported_egg_inventory_is_not_silently_omitted(self):
        (self.site / "other.egg-info").mkdir()
        (self.site / "other.egg-info/PKG-INFO").write_text("Name: other\nVersion: 2.0\n")
        with mock.patch.object(sbom.ArtifactEnvironment, "inspect", side_effect=AssertionError("unbounded discovery")):
            with self.assertRaisesRegex(ValueError, "Unsupported egg"):
                self.document()

    def test_unsupported_egg_file_and_symlink_are_rejected_before_reading(self):
        for kind in ("regular", "symlink"):
            entry = self.site / "other.egg-info"
            if kind == "regular":
                entry.write_text("Name: other\nVersion: 2.0\n")
            else:
                entry.symlink_to(self.directory / "missing-external-metadata")
            with self.subTest(kind=kind), \
                    mock.patch.object(sbom.ArtifactEnvironment, "inspect", side_effect=AssertionError("discovery")):
                with self.assertRaisesRegex(ValueError, "Unsupported egg"):
                    self.document()
            entry.unlink()

    def test_case_insensitive_distribution_suffix_is_bounded(self):
        original = self.site / "example-1.0.dist-info"
        original.rename(self.site / "example-1.0.DIST-INFO")
        self.validate(self.document())

    def test_bounds_and_nonregular_input_fail_closed(self):
        with mock.patch.object(sbom, "MAX_METADATA_BYTES", 5):
            with self.assertRaisesRegex(ValueError, "Oversized"):
                self.document()
        with mock.patch.object(sbom, "MAX_DISTRIBUTIONS", 0):
            with mock.patch.object(sbom, "regular_file", side_effect=AssertionError("metadata read")), \
                    self.assertRaisesRegex(ValueError, "exceeds its bound"):
                self.document()
        with mock.patch.object(sbom, "MAX_TOTAL_METADATA_BYTES", 5):
            with self.assertRaisesRegex(ValueError, "Aggregate"):
                self.document()
        self.artifact.unlink()
        self.artifact.symlink_to(self.lock)
        with self.assertRaisesRegex(ValueError, "regular artifact"):
            self.document()

    def test_invalid_context_and_duplicate_lock_labels_are_rejected(self):
        for kwargs in ({"source_commit": "main"}, {"source_date_epoch": -1}, {"architecture": "ppc64le"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.document(**kwargs)
        other = self.directory / "other" / self.lock.name
        other.parent.mkdir()
        other.write_bytes(self.lock.read_bytes())
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            self.document(locks=(self.lock, other))
        self.lock.write_text('lock-version="2.0"\npackages=[]\n')
        with self.assertRaisesRegex(ValueError, "PEP 751"):
            self.document()
        with self.assertRaisesRegex(ValueError, "input bound"):
            self.document(locks=(self.lock,) * 17)

    def test_cli_refuses_overwrite_and_runtime_output(self):
        command = [sys.executable, "-m", "scripts.generate_python_sbom", "--runtime-root", str(self.root),
                   "--architecture", "amd64", "--artifact", str(self.artifact),
                   "--source-commit", "a" * 40, "--source-date-epoch", "1700000000", "--lock", str(self.lock)]
        output = self.directory / "inventory.spdx.json"
        first = subprocess.run(command + ["--output", str(output)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.validate(json.loads(output.read_text()))
        before = output.read_bytes()
        repeated = subprocess.run(command + ["--output", str(output)], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(before, output.read_bytes())
        inside = subprocess.run(command + ["--output", str(self.root / "forbidden.json")], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertNotEqual(inside.returncode, 0)
        self.assertFalse((self.root / "forbidden.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Exact-pin audit coverage and fail-closed API behavior, without network calls."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("locked_dependency_audit", ROOT / "scripts/audit_locked_dependencies.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def safe_release(name, version, vulnerabilities=()):
    return {"info": {"name": name, "version": version}, "vulnerabilities": list(vulnerabilities)}


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload, url="https://pypi.org/pypi/pillow/12.3.0/json"):
        super().__init__(json.dumps(payload).encode())
        self.url = url

    def geturl(self):
        return self.url


class LockedDependencyAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name in audit.REQUIRED_LOCKS:
            self.write(name, 'lock-version = "1.0"\n[[packages]]\nname = "Pillow"\nversion = "12.3.0"\n'
                       'wheels = [{url="https://supplier.example/pillow-12.3.0-py3-none-any.whl"}]\n')
        # The two prototype locks include the locally rebuilt app, which must
        # not be mistaken for an unrelated public PyPI project of the same name.
        self.append(audit.REQUIRED_LOCKS[-1], '\n[[packages]]\nname="nvbroadcast"\nversion="1.5.3"\n'
                    'archive={path="build/nvbroadcast-1.5.3-py3-none-any.whl"}\n')
        for name in audit.REQUIRED_INPUTS:
            self.write(name, json.dumps({"bootstrap": [{"name": "pip", "version": "26.2.2"}],
                "bindings": [{"name": "PyGObject", "version": "3.48.2"}],
                "uv": {"version": "0.13.4"}, "python": {"version": "3.13.18"}}))
        for name in audit.REQUIRED_FLATPAKS:
            self.write(name, 'modules:\n  - sources:\n      - &wheel\n        type: file\n'
                       '        url: "https://files.pythonhosted.org/packages/numpy-2.5.2-py3-none-any.whl"\n'
                       '      - *wheel\n')
        self.advisory = {"id": "GHSA-fixture", "aliases": ["CVE-2026-12345"],
                         "fixed_in": ["13.0.0"], "withdrawn": None,
                         "link": "https://osv.dev/vulnerability/GHSA-fixture"}

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def append(self, name, text):
        path = self.root / name
        path.write_text(path.read_text() + text)

    def test_all_active_pin_sources_are_captured_and_duplicates_coalesce(self):
        self.write("packaging/native-runtime/qualification/old-inputs.json", '{"name":"old","version":"1.0"}')
        self.write("packaging/native-runtime/qualification/pylock.old.toml", 'invalid history')
        inventory = audit.collect_pins(self.root)
        self.assertEqual({(p["name"], p["version"]) for p in inventory["pins"]}, {
            ("pillow", "12.3.0"), ("pip", "26.2.2"), ("pygobject", "3.48.2"), ("uv", "0.13.4"), ("numpy", "2.5.2")})
        self.assertEqual(len(inventory["input_sha256"]), 12)
        self.assertEqual(len(next(p for p in inventory["pins"] if p["name"] == "pillow")["sources"]), 6)
        self.assertEqual(inventory["skipped_local_application"][0]["name"], "nvbroadcast")
        self.assertEqual(inventory["outside_pypi_scope"][0]["component"], "CPython")
        fetch = mock.Mock(side_effect=safe_release)
        report = audit.audit_inventory(inventory, fetch=fetch)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(fetch.call_count, 5)

    def test_new_active_lock_is_discovered_and_local_gi_is_audited(self):
        self.write("packaging/new-runtime/pylock.new.toml", 'lock-version="1.0"\n[[packages]]\n'
                   'name="pycairo"\nversion="1.29.0"\narchive={path="locally-built.whl"}\n')
        pins = audit.collect_pins(self.root)["pins"]
        self.assertIn(("pycairo", "1.29.0"), [(p["name"], p["version"]) for p in pins])

    def test_flatpak_module_includes_keep_exact_sources_and_fail_on_missing_or_external_files(self):
        main = self.root / audit.REQUIRED_FLATPAKS[0]
        main.write_text('modules:\n  - python3-private-runtime.yml\n  - python3-flatpak-requirements.yaml\n')
        inventory = audit.collect_pins(self.root)
        self.assertIn(("numpy", "2.5.2"), [(p["name"], p["version"]) for p in inventory["pins"]])
        for module in ('missing.yml', '../external.yml'):
            with self.subTest(module=module):
                main.write_text(f'modules:\n  - {module}\n')
                with self.assertRaisesRegex(ValueError, 'module|included'):
                    audit.collect_pins(self.root)

    def test_exact_pinned_version_is_queried_even_when_latest_would_be_safe(self):
        requests = []
        def fetch(name, version):
            requests.append((name, version))
            return safe_release(name, version, [self.advisory] if name == "pillow" and version == "12.3.0" else [])
        report = audit.audit_inventory(audit.collect_pins(self.root), fetch=fetch)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["counts"]["vulnerable"], 1)
        self.assertIn(("pillow", "12.3.0"), requests)
        self.assertNotIn(("pillow", "13.0.0"), requests)
        result = next(r for r in report["results"] if r["name"] == "pillow")
        self.assertEqual(result["active_advisories"][0]["id"], "GHSA-fixture")

    def test_withdrawn_only_advisories_are_retained_without_blocking(self):
        withdrawn = {**self.advisory, "withdrawn": "2026-10-26T12:00:00Z"}
        report = audit.audit_inventory(audit.collect_pins(self.root),
            fetch=lambda name, version: safe_release(name, version, [withdrawn]))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["results"][0]["active_advisories"], [])
        self.assertEqual(report["results"][0]["withdrawn_advisories"][0]["id"], "GHSA-fixture")
        mixed = audit.check_release({"name": "pillow", "version": "12.3.0"},
            safe_release("pillow", "12.3.0", [withdrawn, self.advisory]))
        self.assertEqual(mixed["status"], "vulnerable")

    def test_unknown_version_and_network_failure_are_audit_failures(self):
        for error in (urllib.error.HTTPError("https://pypi.org/", 404, "Not Found", {}, None),
                      urllib.error.URLError("offline"), TimeoutError("deadline")):
            with self.subTest(error=type(error).__name__):
                report = audit.audit_inventory(audit.collect_pins(self.root), fetch=mock.Mock(side_effect=error))
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["counts"]["error"], 5)

    def test_overall_deadline_reports_unqueried_pins_as_errors(self):
        fetch = mock.Mock(side_effect=safe_release)
        with mock.patch.object(audit, "MAX_AUDIT_SECONDS", 0):
            report = audit.audit_inventory(audit.collect_pins(self.root), fetch=fetch)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["counts"]["error"], 5)
        fetch.assert_not_called()

    def test_identity_mismatch_and_missing_advisory_inventory_fail_closed(self):
        invalid = [safe_release("different-project", "12.3.0"), safe_release("pillow", "13.0.0"),
                   {"info": {"name": "pillow", "version": "12.3.0"}},
                   safe_release("pillow", "12.3.0", [{**self.advisory, "withdrawn": False}])]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises((ValueError, TypeError)):
                    audit.check_release({"name": "pillow", "version": "12.3.0"}, payload)

    def test_missing_input_unversioned_pin_and_empty_manifest_fail_closed(self):
        path = self.root / audit.REQUIRED_LOCKS[0]
        saved = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            audit.collect_pins(self.root)
        path.write_text(saved.replace('version = "12.3.0"', 'version = ">=12.3.0"'))
        with self.assertRaises(ValueError):
            audit.collect_pins(self.root)
        path.write_text(saved)
        self.write(audit.REQUIRED_FLATPAKS[0], 'modules: []\n')
        with self.assertRaisesRegex(ValueError, "no exact Python sources"):
            audit.collect_pins(self.root)

    def test_flatpak_sdist_version_is_audited_and_malformed_source_rejected(self):
        path = self.write(audit.REQUIRED_FLATPAKS[0], 'sources:\n  - url: https://files.pythonhosted.org/PyYAML-6.0.3.tar.gz\n')
        self.assertIn(("pyyaml", "6.0.3"), [(p["name"], p["version"]) for p in audit.collect_pins(self.root)["pins"]])
        path.write_text('sources:\n  - url: https://files.pythonhosted.org/not-a-package.bin\n')
        with self.assertRaises(ValueError):
            audit.collect_pins(self.root)

    def test_advisory_details_are_not_copied_to_public_evidence(self):
        entry = {**self.advisory, "details": "large untrusted narrative" * 1000}
        result = audit.check_release({"name": "pillow", "version": "12.3.0"}, safe_release("pillow", "12.3.0", [entry]))
        self.assertNotIn("large untrusted", json.dumps(result))
        self.assertLess(len(json.dumps(result)), 2000)

    def test_fetch_uses_exact_version_and_bounded_retry(self):
        with mock.patch.object(audit.urllib.request, "urlopen", side_effect=[TimeoutError("first"),
                Response(safe_release("pillow", "12.3.0"))]) as fetch, mock.patch.object(audit.time, "sleep"):
            result = audit.fetch_release("Pillow", "12.3.0")
        self.assertEqual(result["info"]["version"], "12.3.0")
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args.args[0].full_url, "https://pypi.org/pypi/pillow/12.3.0/json")
        self.assertLessEqual(fetch.call_args.kwargs["timeout"], 15)
        error = urllib.error.HTTPError("https://pypi.org/", 404, "missing", {}, None)
        with mock.patch.object(audit.urllib.request, "urlopen", side_effect=error) as fetch:
            with self.assertRaises(urllib.error.HTTPError):
                audit.fetch_release("pillow", "0.0.0")
        self.assertEqual(fetch.call_count, 1)

    def test_fetch_rejects_wrong_origin_and_oversized_response(self):
        with mock.patch.object(audit.urllib.request, "urlopen", return_value=Response({}, "https://other.example/")):
            with self.assertRaisesRegex(ValueError, "origin"):
                audit.fetch_release("pillow", "12.3.0")
        with mock.patch.object(audit, "MAX_BYTES", 100), mock.patch.object(audit.urllib.request, "urlopen",
                return_value=Response({"oversized": "x" * 1000})):
            with self.assertRaisesRegex(ValueError, "size bound"):
                audit.fetch_release("pillow", "12.3.0")


if __name__ == "__main__":
    unittest.main()

"""Guard distributed license text and the existing platform license scopes."""

import ast
import hashlib
import re
import tomllib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_LICENSE = "GPL-3.0-or-later"
COMPLETE_GPL_MARKER = b"COMPLETE GNU GENERAL PUBLIC LICENSE, VERSION 3\n\n"
# Unmodified https://www.gnu.org/licenses/gpl-3.0.txt, reviewed 5 October 2026.
OFFICIAL_GPL3_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"


class LicenseMetadataTests(unittest.TestCase):
    def test_license_contains_the_complete_unmodified_gpl3_text(self):
        license_bytes = (ROOT / "LICENSE").read_bytes()
        self.assertEqual(license_bytes.count(COMPLETE_GPL_MARKER), 1)
        _, _, complete_gpl = license_bytes.partition(COMPLETE_GPL_MARKER)
        self.assertEqual(hashlib.sha256(complete_gpl).hexdigest(), OFFICIAL_GPL3_SHA256)

    def test_project_grant_and_existing_attribution_terms_are_retained(self):
        project_terms = (ROOT / "LICENSE").read_bytes().partition(COMPLETE_GPL_MARKER)[0].decode()
        normalized = " ".join(project_terms.split())
        self.assertIn(
            "either version 3 of the License, or (at your option) any later version.",
            normalized,
        )
        self.assertIn("ATTRIBUTION REQUIREMENT", project_terms)
        for requirement in (
            "or derivative work MUST retain the following:",
            "1. This LICENSE file in its entirety",
            "2. The copyright headers in all source files",
            '3. The "by doczeus" attribution in the application UI',
            '4. The __author__ = "doczeus" metadata in __init__.py',
            "5. The original project URL: https://github.com/Hkshoonya/nvidia-broadcast-linux",
            "Removing or obscuring the original author attribution is a violation of this license.",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, normalized)

    def test_application_and_package_metadata_agree_with_the_or_later_grant(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        package = ast.parse((ROOT / "src/nvbroadcast/__init__.py").read_text())
        declarations = [
            ast.literal_eval(node.value)
            for node in package.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__license__" for target in node.targets)
        ]
        self.assertEqual(len(declarations), 1)
        snap = (ROOT / "snap/snapcraft.yaml").read_text()
        rpm = (ROOT / "packaging/rpm/nvbroadcast.spec").read_text()
        debian = (ROOT / "packaging/debian/copyright").read_text()
        default_stanzas = [
            stanza for stanza in debian.split("\n\n")
            if re.search(r"(?m)^Files:\s*\*\s*$", stanza)
        ]
        self.assertEqual(len(default_stanzas), 1)
        metainfo = ET.parse(ROOT / "data/com.doczeus.NVBroadcast.metainfo.xml").getroot()
        values = (
            ("pyproject", project["license"]["text"]),
            ("package __license__", declarations[0]),
            ("Snap", re.findall(r"(?m)^license:\s*(\S+)\s*$", snap)),
            ("RPM", re.findall(r"(?m)^License:\s*(\S+)\s*$", rpm)),
            ("Debian default Files stanza", re.findall(r"(?m)^License:\s*(\S+)\s*$", default_stanzas[0])),
            ("AppStream project", metainfo.findtext("project_license")),
        )
        for name, value in values:
            with self.subTest(metadata=name):
                self.assertEqual(value, [PROJECT_LICENSE] if isinstance(value, list) else PROJECT_LICENSE)
        self.assertIn(
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            project["classifiers"],
        )
        self.assertIn(f"License: {PROJECT_LICENSE}\n This program is free software:", debian)
        self.assertIn("(at your option) any later version.", debian)
        # AppStream's metadata has its own grant, independent of the app code.
        self.assertEqual(metainfo.findtext("metadata_license"), "CC0-1.0")

    def test_macos_camera_extension_keeps_its_separate_proprietary_terms(self):
        extension_terms = (ROOT / "macos/LICENSE").read_text()
        self.assertIn("PROPRIETARY LICENSE", extension_terms)
        self.assertIn("directory (macos/) are the proprietary work of Doczeus.", extension_terms)
        self.assertIn("licensed, not sold.", extension_terms)
        readme = (ROOT / "README.md").read_text()
        self.assertIn("**Python app & Linux code:** GPL-3.0-or-later", readme)
        self.assertIn("**macOS Camera Extension** (`macos/`): Proprietary", readme)
        self.assertIn("[macos/LICENSE](macos/LICENSE)", readme)


if __name__ == "__main__":
    unittest.main()

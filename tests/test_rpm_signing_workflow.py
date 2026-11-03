"""Exercise the signed-RPM to native-upgrader release boundary."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/build-packages.yml").read_text()


def step_body(name):
    step = WORKFLOW.split(f"      - name: {name}\n", 1)[1]
    body = step.split("        run: |\n", 1)[1]
    return textwrap.dedent(body.split("\n      - ", 1)[0].split("\n  #", 1)[0])


class RPMSigningWorkflowTests(unittest.TestCase):
    def test_upgrader_binds_the_signed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scripts", "bin", "dist/linux-unsigned/deb", "dist/linux-unsigned/rpm", "dist/linux-signed/rpm"):
                (root / name).mkdir(parents=True, exist_ok=True)
            for name in ("render_native_upgrade_helper.py", "native_package_upgrade.sh.in"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
            deb_name = "nvbroadcast_1.5.3-1_all.deb"
            rpm_name = "nvbroadcast-1.5.3-1.noarch.rpm"
            (root / "dist/linux-unsigned/deb" / deb_name).write_bytes(b"deb-payload")
            (root / "dist/linux-unsigned/rpm" / rpm_name).write_bytes(b"unsigned-rpm")
            (root / "dist/linux-signed/rpm" / rpm_name).write_bytes(b"final-signed-rpm")
            metadata = root / "bin/dpkg-deb"
            metadata.write_text("#!/bin/sh\nprintf '1.5.3-1\\n'\n")
            metadata.chmod(0o755)
            environment = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"])
            result = subprocess.run(["bash", "-c", step_body("Bind native upgrader to the signed package")],
                                    cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            helper = (root / "dist/linux-signed/nvbroadcast-native-upgrade").read_text()
            self.assertIn(hashlib.sha256(b"final-signed-rpm").hexdigest(), helper)
            self.assertNotIn(hashlib.sha256(b"unsigned-rpm").hexdigest(), helper)
            self.assertEqual((root / "dist/linux-signed/deb" / deb_name).read_bytes(), b"deb-payload")

    def test_ambiguous_package_set_cannot_generate_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("dist/linux-unsigned/deb", "dist/linux-signed/rpm"):
                (root / name).mkdir(parents=True)
            for name in ("one.deb", "two.deb"):
                (root / "dist/linux-unsigned/deb" / name).touch()
            (root / "dist/linux-signed/rpm/one.rpm").touch()
            result = subprocess.run(["bash", "-c", step_body("Bind native upgrader to the signed package")],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "dist/linux-signed/nvbroadcast-native-upgrade").exists())

    def test_unsigned_builds_are_excluded_from_release_and_attestation(self):
        for job in ("attest-release", "release"):
            body = WORKFLOW.split(f"\n  {job}:\n", 1)[1].split("\n  #", 1)[0]
            self.assertNotIn("artifacts/linux-packages/", body)
            self.assertIn("artifacts/linux-signed-packages/rpm/*.rpm", body)
        gate = WORKFLOW.split("\n  attest-release:\n", 1)[1].split("\n    steps:", 1)[0]
        self.assertIn("sign-linux", gate)


if __name__ == "__main__":
    unittest.main()

"""Opt-in real RPM6 signer -> RPM6/RPM4 reader regression with disposable keys.

Set NVB_RPM_SIGN_TEST_RPM6_IMAGE and NVB_RPM_SIGN_TEST_RPM4_IMAGE to existing
immutable local fixture image IDs. Images are never pulled and need Python,
GnuPG and RPM tools; the RPM6 fixture also needs rpmbuild/rpmsign.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
RPM6_IMAGE = os.environ.get('NVB_RPM_SIGN_TEST_RPM6_IMAGE', '')
RPM4_IMAGE = os.environ.get('NVB_RPM_SIGN_TEST_RPM4_IMAGE', '')


@unittest.skipUnless(RPM6_IMAGE and RPM4_IMAGE and shutil.which('docker'),
                     'explicit local RPM6/RPM4 signing fixture images required')
class RealRPMReaderInteropTests(unittest.TestCase):
    def test_format6_signature_and_payload_verify_with_both_readers(self):
        for image in (RPM6_IMAGE, RPM4_IMAGE):
            self.assertRegex(image, r'^sha256:[0-9a-f]{64}$')
            actual = subprocess.check_output(
                ['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                text=True, timeout=30).strip()
            self.assertEqual(actual, image)
        with tempfile.TemporaryDirectory(prefix='nvb-rpm-interop-') as temporary:
            output = Path(temporary)
            self._run(RPM6_IMAGE, output, 'produce')
            producer = json.loads((output / 'producer.json').read_text())
            self.assertEqual(producer['rpm_format'], 6)
            self.assertEqual(producer['signer_report']['status'], 'verified')
            self.assertNotEqual(producer['unsigned_sha256'],
                                producer['signer_report']['signed_sha256'])
            for major, image in ((6, RPM6_IMAGE), (4, RPM4_IMAGE)):
                with self.subTest(reader=major):
                    self._run(image, output, str(major))
                    result = json.loads((output / f'rpm{major}.json').read_text())
                    self.assertEqual(result['native_returncode'], 0, result)
                    self.assertTrue(result['actual_signature_ok'], result)
                    self.assertTrue(result['strict_verifier_accepts'], result)
                    self.assertEqual(result['payload_returncode'], 0, result)
                    self.assertTrue(result['payload_files_match'], result)

    def _run(self, image, output, phase):
        name = 'nvb-rpm-interop-' + uuid.uuid4().hex[:12]
        command = [
            'docker', 'run', '--runtime=runc', '--name', name, '--rm', '--init',
            '--log-driver=none', '--network=none', '--pull=never', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64',
            '--memory=1g', '--cpus=2', '--user', f'{os.getuid()}:{os.getgid()}',
            '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777',
            '--mount', f'type=bind,source={ROOT / "tests/fixtures/rpm_signing_interop.py"},target=/fixture.py,readonly',
            '--mount', f'type=bind,source={ROOT / "scripts/sign_rpm_package.py"},target=/fixture/signer.py,readonly',
            '--mount', f'type=bind,source={output},target=/out',
            '--entrypoint', 'python3', image, '-B', '-I', '-S', '/fixture.py', phase,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)


if __name__ == '__main__':
    unittest.main()

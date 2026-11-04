"""Run only inside explicitly selected disposable RPM signing test images."""

from pathlib import Path
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


OUTPUT = Path('/out')
PRIVATE_NAMES = ('NVB_RPM_PRIVATE_KEY', 'NVB_RPM_KEY_PASSPHRASE',
                 'RPM_SIGNING_PRIVATE_KEY', 'RPM_SIGNING_PASSPHRASE')


def call(arguments, *, environment=None, data=None):
    result = subprocess.run(arguments, env=environment, input=data,
                            capture_output=True, timeout=120)
    if result.returncode:
        # This build runs before any disposable key exists. Preserve build
        # diagnostics while keeping all GnuPG import/export failures private.
        details = result.stderr.decode(errors='replace')[-8192:] if arguments[0] == 'rpmbuild' else ''
        raise RuntimeError(f'{Path(arguments[0]).name} failed (exit {result.returncode}): {details}')
    return result.stdout


def signer_module():
    spec = importlib.util.spec_from_file_location('trusted_signer', '/fixture/signer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def produce():
    assert not any(os.environ.get(name) for name in PRIVATE_NAMES)
    signing = signer_module()
    assert call(['rpm', '--version']).decode().startswith('RPM version 6.')
    with tempfile.TemporaryDirectory(prefix='nvb-interop-', dir='/tmp') as temporary:
        root = Path(temporary)
        home = root / 'keys'
        home.mkdir(mode=0o700)
        environment = dict(os.environ, GNUPGHOME=str(home), LC_ALL='C')
        password = b'nvb-disposable-interoperability-test'
        try:
            spec = root / 'fixture.spec'
            spec.write_text('''Name: nvb-signature-interop
Version: 1.0
Release: 1
Summary: Disposable signature interoperability fixture
License: MIT
BuildArch: noarch
%description
No installer hooks or host configuration changes.
%install
mkdir -p %{buildroot}/usr/share/nvb-signature-interop
printf 'signature-test-payload\\n' > %{buildroot}/usr/share/nvb-signature-interop/example
%files
/usr/share/nvb-signature-interop/example
''')
            call(['rpmbuild', '--define', '_rpmformat 6', '--define', '_tmppath /tmp', '--define',
                  f'_topdir {root / "build"}', '-bb', str(spec)])
            package, = (root / 'build/RPMS').rglob('*.rpm')
            assert call(['rpm', '-qp', '--qf', '%{RPMFORMAT}', str(package)]) == b'6'
            call(['gpg', '--batch', '--pinentry-mode', 'loopback', '--passphrase-fd', '0',
                  '--quick-generate-key', 'NV Broadcast Disposable Signature Test',
                  'rsa2048', 'cert', '1d'], environment=environment, data=password)
            listing = call(['gpg', '--batch', '--with-colons', '--fingerprint',
                            '--list-secret-keys'], environment=environment).decode()
            fingerprint, = signing.primary_fingerprints(listing)
            call(['gpg', '--batch', '--pinentry-mode', 'loopback', '--passphrase-fd', '0',
                  '--quick-add-key', fingerprint, 'rsa2048', 'sign', '1d'],
                 environment=environment, data=password)
            private = call(['gpg', '--batch', '--pinentry-mode', 'loopback',
                            '--passphrase-fd', '0', '--armor', '--export-secret-subkeys',
                            fingerprint], environment=environment, data=password)
            public = root / 'public.asc'
            public.write_bytes(call(['gpg', '--batch', '--armor', '--export', fingerprint],
                                    environment=environment))
            os.environ['NVB_RPM_PRIVATE_KEY'] = private.decode()
            os.environ['NVB_RPM_KEY_PASSPHRASE'] = password.decode()
            before = hashlib.sha256(package.read_bytes()).hexdigest()
            report = signing.sign(package, public, fingerprint, root / 'signed')
            assert hashlib.sha256(package.read_bytes()).hexdigest() == before
            shutil.copyfile(root / 'signed' / package.name, OUTPUT / 'candidate.rpm')
            shutil.copyfile(public, OUTPUT / 'public.asc')
            (OUTPUT / 'producer.json').write_text(json.dumps({
                'rpm_format': 6, 'fingerprint': fingerprint, 'unsigned_sha256': before,
                'signer_report': report,
            }, indent=2) + '\n')
        finally:
            for name in PRIVATE_NAMES:
                os.environ.pop(name, None)
            subprocess.run(['gpgconf', '--homedir', str(home), '--kill', 'gpg-agent'],
                           capture_output=True, timeout=30)


def verify(major):
    assert not any(os.environ.get(name) for name in PRIVATE_NAMES)
    signing = signer_module()
    version = call(['rpm', '--version']).decode().strip()
    assert version.startswith(f'RPM version {major}.')
    producer = json.loads((OUTPUT / 'producer.json').read_text())
    package, public = OUTPUT / 'candidate.rpm', OUTPUT / 'public.asc'
    assert hashlib.sha256(package.read_bytes()).hexdigest() == producer['signer_report']['signed_sha256']
    with tempfile.TemporaryDirectory(prefix='nvb-interop-public-', dir='/tmp') as temporary:
        database = Path(temporary) / 'rpmdb'
        database.mkdir(mode=0o700)
        call(['rpmkeys', '--dbpath', str(database), '--import', str(public)])
        native = subprocess.run(['rpmkeys', '--dbpath', str(database), '--checksig',
                                 '--verbose', str(package)], capture_output=True,
                                text=True, timeout=60)
    # RPM4's rpm2cpio cannot emit the format6 64-bit-size CPIO stream. Its
    # supported rpm2archive reader verifies the same actual file contents.
    payload = subprocess.run(['rpm2archive', '--nocompression', str(package)],
                             capture_output=True, timeout=60)
    files = {}
    if payload.returncode == 0:
        assert len(payload.stdout) <= 4 * 1024**2
        with tarfile.open(fileobj=io.BytesIO(payload.stdout), mode='r:') as archive:
            members = archive.getmembers()
            assert len(members) <= 32
            for member in members:
                assert member.isdir() or member.isfile()
                if member.isfile():
                    assert 0 <= member.size <= 128
                    name = member.name.removeprefix('./')
                    assert name not in files
                    files[name] = archive.extractfile(member).read(129).decode()
    report = {
        'rpm_version': version, 'native_returncode': native.returncode,
        'native_output': native.stdout,
        'actual_signature_ok': bool(re.search(r'\bSignature\b[^\n]*:\s*OK\s*$',
                                             native.stdout, re.MULTILINE | re.IGNORECASE)),
        'payload_returncode': payload.returncode,
        'payload_error': payload.stderr.decode(errors='replace'),
        'payload_decoder': 'rpm2archive',
        'payload_files_match': files == {'usr/share/nvb-signature-interop/example': 'signature-test-payload\n'},
    }
    try:
        signing.verify_rpm(package, public, producer['fingerprint'], dict(os.environ))
        report['strict_verifier_accepts'] = True
    except signing.SigningError as error:
        report['strict_verifier_accepts'] = False
        report['verification_error'] = str(error)
    (OUTPUT / f'rpm{major}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'produce':
        produce()
    else:
        verify(int(sys.argv[1]))

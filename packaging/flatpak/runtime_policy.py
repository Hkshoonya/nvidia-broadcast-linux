#!/usr/bin/env python3
"""Prepare one private builder with authenticated, exact GNOME runtime inputs."""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import uuid


INSTALLATION = Path('/var/lib/nvbroadcast-flatpak')
BASE_REPO = Path('/var/lib/flatpak/repo')
CA_PATH = 'files/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem'
REFS = {'runtime/org.gnome.Platform/x86_64/50', 'runtime/org.gnome.Sdk/x86_64/50'}
HEX = re.compile(r'[0-9a-f]{64}')
IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}')
MINIMUM_FREE = 25 * 1024 ** 3
HTTP_USER_AGENT = 'NVBroadcast-runtime-preparation (https://github.com/Hkshoonya/nvidia-broadcast-linux)'
# Only unused toolchains on a disposable GitHub-hosted Linux runner are eligible.
# Never remove Docker data, the workspace, the runner, or the shared tool cache.
HOSTED_TOOLCHAINS = (Path('/usr/share/dotnet'), Path('/usr/local/lib/android'),
                     Path('/usr/share/swift'), Path('/opt/ghc'))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_regular(path, maximum=4 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f'Missing, redirected or oversized runtime input: {path}')
    return path.read_bytes()


def load_lock(path):
    data = read_regular(path, 65536)
    lock = json.loads(data)
    if not isinstance(lock, dict) or set(lock) != {'schema_version', 'base_image', 'remote_url', 'public_key_sha256',
                     'rejected_certificate_sha256', 'runtimes'} or lock['schema_version'] != 1:
        raise ValueError('Unsupported runtime lock')
    if not isinstance(lock['base_image'], str) or not re.fullmatch(r'ghcr\.io/flathub-infra/flatpak-github-actions@sha256:[0-9a-f]{64}', lock['base_image']):
        raise ValueError('Runtime base image must be an official immutable digest')
    if (lock['remote_url'] != 'https://dl.flathub.org/repo/' or
            not isinstance(lock['public_key_sha256'], str) or not HEX.fullmatch(lock['public_key_sha256'])):
        raise ValueError('Unexpected runtime signing source')
    rejected = lock['rejected_certificate_sha256']
    if not isinstance(rejected, list) or not rejected or len(rejected) > 20 or not all(isinstance(x, str) and HEX.fullmatch(x) for x in rejected):
        raise ValueError('Runtime trust rejection policy is missing or malformed')
    entries = lock['runtimes']
    if (not isinstance(entries, list) or len(entries) != 2 or
            not all(isinstance(x, dict) and isinstance(x.get('ref'), str) for x in entries) or
            {x.get('ref') for x in entries} != REFS):
        raise ValueError('Runtime lock must contain exactly the x86_64 GNOME50 Platform and SDK')
    for entry in entries:
        if set(entry) != {'ref', 'commit', 'ca_sha256'} or not all(isinstance(entry[k], str) and HEX.fullmatch(entry[k]) for k in ('commit', 'ca_sha256')):
            raise ValueError('Runtime commits and CA bundles require exact hashes')
    return lock


def command(*args, env=None, timeout=120):
    try:
        result = subprocess.run(args, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, timeout=timeout)
    except subprocess.CalledProcessError as error:
        error.add_note((error.stdout or b'')[-65536:].decode('utf-8', errors='replace'))
        raise
    if len(result.stdout) > 4 * 1024 * 1024:
        raise ValueError('Runtime command output exceeds its bound')
    return result.stdout.decode('utf-8')


def runtime_env():
    return {**os.environ, 'FLATPAK_SYSTEM_DIR': str(INSTALLATION), 'LC_ALL': 'C'}


def prepare_hosted_runner_storage(output):
    expected = {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
                'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64'}
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise ValueError('Storage cleanup is restricted to GitHub-hosted Linux x64 runners')
    if output.exists():
        raise ValueError('Use a fresh runner-storage output directory')
    output.mkdir(parents=True)
    before = shutil.disk_usage(output).free
    removed = []
    if before < MINIMUM_FREE:
        for path in HOSTED_TOOLCHAINS:
            if path.is_symlink() or path.resolve() != path:
                raise ValueError(f'Unused runner toolchain is redirected: {path}')
            if path.exists():
                if not path.is_dir():
                    raise ValueError(f'Unused runner toolchain is not a directory: {path}')
                command('sudo', '--non-interactive', '/usr/bin/rm', '-rf', '--', str(path), timeout=300)
                removed.append(str(path))
                if shutil.disk_usage(output).free >= MINIMUM_FREE:
                    break
    after = shutil.disk_usage(output).free
    receipt = {'schema_version': 1, 'status': 'pass' if after >= MINIMUM_FREE else 'fail',
               'runner': expected, 'free_bytes_before': before, 'free_bytes_after': after,
               'required_free_bytes': MINIMUM_FREE, 'removed_toolchains': removed}
    (output / 'storage-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if after < MINIMUM_FREE:
        raise ValueError(f'Builder preparation requires at least 25 GiB free; available: {after} bytes')
    return receipt


def remove_container(name):
    try:
        command('docker', 'rm', '--force', name)
    except subprocess.CalledProcessError as error:
        message = (error.stdout or b'').decode('utf-8', errors='replace')
        if 'No such container: ' + name not in message:
            raise


def validate_prepared_image(record, base_image, prepared_image):
    if (not isinstance(prepared_image, str) or not IMAGE_ID.fullmatch(prepared_image) or
            not isinstance(record, dict) or record.get('base_image') != base_image or
            record.get('prepared_image_id') != prepared_image or
            not isinstance(record.get('base_image_id'), str) or
            not IMAGE_ID.fullmatch(record.get('base_image_id', ''))):
        raise ValueError('Prepared image identity differs from its requested base or image ID')
    base = record.get('base_rootfs_layers', [])
    prepared = record.get('prepared_rootfs_layers', [])
    if (not isinstance(base, list) or not base or not isinstance(prepared, list) or
            not all(isinstance(x, str) and IMAGE_ID.fullmatch(x) for x in base + prepared) or
            len(prepared) <= len(base) or prepared[:len(base)] != base):
        raise ValueError('Prepared image does not extend the exact base filesystem layers')
    if record.get('effective_environment') != {
            'FLATPAK_SYSTEM_DIR': str(INSTALLATION), 'NVIDIA_VISIBLE_DEVICES': 'void'}:
        raise ValueError('Prepared image effective runtime environment differs from policy')


def inspect_prepared_image(base_image, prepared_image):
    base = json.loads(command('docker', 'image', 'inspect', base_image))
    prepared = json.loads(command('docker', 'image', 'inspect', prepared_image))
    if not isinstance(base, list) or len(base) != 1 or not isinstance(prepared, list) or len(prepared) != 1:
        raise ValueError('Expected one base and one committed image inspection')
    base, prepared = base[0], prepared[0]
    if (base_image not in base.get('RepoDigests', []) or
            any(item.get('Architecture') != 'amd64' or item.get('Os') != 'linux'
                or item.get('RootFS', {}).get('Type') != 'layers' for item in (base, prepared))):
        raise ValueError('Prepared image base digest or native platform differs from policy')
    environment = {}
    for value in prepared.get('Config', {}).get('Env', []):
        key, separator, value = value.partition('=')
        if not separator or key in environment:
            raise ValueError('Prepared image has malformed or duplicate environment entries')
        environment[key] = value
    record = {'base_image': base_image, 'base_image_id': base.get('Id'),
              'prepared_image_id': prepared.get('Id'),
              'base_rootfs_layers': base['RootFS'].get('Layers'),
              'prepared_rootfs_layers': prepared['RootFS'].get('Layers'),
              'effective_environment': {key: environment.get(key) for key in
                                        ('FLATPAK_SYSTEM_DIR', 'NVIDIA_VISIBLE_DEVICES')}}
    validate_prepared_image(record, base_image, prepared_image)
    return record


def inspect_committed_runtime(lock_path, image):
    name = 'nvbroadcast-flatpak-inspect-' + uuid.uuid4().hex[:12]
    try:
        # Inherit and verify the committed image's environment; do not repair it
        # with docker -e overrides. The mounted policy remains reviewed source.
        return json.loads(command('docker', 'run', '--rm', '--pull=never', '--name', name,
                '--runtime=runc', '--network=none', '--log-driver=none',
                '--pids-limit=256', '--memory=4g', '--cpus=2', '--cap-drop=ALL',
                '--security-opt=no-new-privileges',
                '--mount', f'type=bind,source={lock_path.parent},target=/runtime-policy,readonly',
                image, 'python3', '/runtime-policy/runtime_policy.py', '--inspect',
                '--lock', '/runtime-policy/' + lock_path.name,
                '--output', '/tmp/committed-runtime-receipt.json', timeout=600))
    finally:
        remove_container(name)


def verify_preparation_proof(path, base_image, prepared_image, runtime_inputs):
    data = read_regular(path)
    proof = json.loads(data)
    validate_prepared_image(proof, base_image, prepared_image)
    if (proof.get('schema_version') != 1 or proof.get('status') != 'pass' or
            proof.get('runtime_inputs') != runtime_inputs or
            proof.get('lock_sha256') != runtime_inputs.get('lock_sha256')):
        raise ValueError('Committed image inspection does not bind the active runtime inputs')
    return proof, sha256(data)


def verify_remote(lock):
    config = configparser.ConfigParser(interpolation=None)
    config.read(INSTALLATION / 'repo/config')
    remote = config['remote "flathub"']
    if (remote.get('url') != lock['remote_url'] or
            not remote.getboolean('gpg-verify', fallback=False) or
            not remote.getboolean('gpg-verify-summary', fallback=False) or
            remote.getboolean('tls-permissive', fallback=False) or
            remote.getboolean('http2', fallback=True)):
        raise ValueError('Official HTTPS remote, both GPG settings and HTTP/1.1 are required')
    key = read_regular(INSTALLATION / 'repo/flathub.trustedkeys.gpg', 65536)
    if sha256(key) != lock['public_key_sha256']:
        raise ValueError('Runtime trusted public key differs from the reviewed key')


def verify_signature(output, commit):
    if (not re.search(r'^commit ' + re.escape(commit) + r'$', output, re.M) or
            not re.search(r'^Found [1-9][0-9]* signature', output, re.M) or
            'Good signature from "Flathub Repo Signing Key <flathub@flathub.org>"' not in output or
            re.search(r'BAD signature|expired|revoked|Cannot verify|Can.t check', output, re.I)):
        raise ValueError('Exact runtime commit lacks a valid Flathub signature')


def inspect_ca(data, expected_sha256, rejected):
    certificates = re.findall(b'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', data, re.S)
    if not certificates or len(certificates) != data.count(b'-----BEGIN CERTIFICATE-----'):
        raise ValueError('Runtime TLS CA bundle is malformed or empty')
    digests = {sha256(ssl.PEM_cert_to_DER_cert(cert.decode('ascii'))) for cert in certificates}
    if digests.intersection(rejected):
        raise ValueError('Runtime TLS CA bundle contains a rejected certificate')
    if sha256(data) != expected_sha256:
        raise ValueError('Runtime TLS CA bundle differs from the reviewed commit input')
    return {'sha256': sha256(data), 'certificate_count': len(certificates),
            'rejected_certificates_present': False}


def signed_commit(entry):
    repo = f'--repo={INSTALLATION / "repo"}'
    signature = command('ostree', repo, 'show', '--gpg-verify-remote=flathub',
                        entry['commit'], env=runtime_env())
    verify_signature(signature, entry['commit'])
    binding = command('ostree', repo, 'show', '--print-metadata-key=ostree.ref-binding',
                      entry['commit'], env=runtime_env()).strip()
    if binding != repr([entry['ref']]):
        raise ValueError('Signed runtime commit does not bind the expected ref')
    return signature


def inspect_runtime(lock_path):
    lock = load_lock(lock_path)
    if os.environ.get('FLATPAK_SYSTEM_DIR') != str(INSTALLATION):
        raise ValueError('Prepared runtime installation must be selected explicitly')
    verify_remote(lock)
    records = []
    for entry in lock['runtimes']:
        actual = command('flatpak', 'info', '--system', '--show-commit', entry['ref'], env=runtime_env()).strip()
        if actual != entry['commit']:
            raise ValueError('Active runtime commit differs from the reviewed lock')
        signature = signed_commit(entry)
        location = Path(command('flatpak', 'info', '--system', '--show-location', entry['ref'], env=runtime_env()).strip())
        expected = INSTALLATION / entry['ref'] / entry['commit']
        if location != expected:
            raise ValueError('Runtime deployment is outside the prepared installation')
        ca = inspect_ca(read_regular(location / CA_PATH), entry['ca_sha256'],
                        lock['rejected_certificate_sha256'])
        records.append({**entry, 'signature': signature, 'tls_ca_bundle': ca})
    return {'schema_version': 1, 'status': 'pass', 'base_image': lock['base_image'],
            'lock_sha256': sha256(read_regular(lock_path)), 'installation': str(INSTALLATION),
            'official_remote_url': lock['remote_url'], 'gpg_verify': True,
            'gpg_verify_summary': True, 'tls_permissive': False, 'http2': False,
            'public_key_sha256': lock['public_key_sha256'],
            'rejected_certificate_sha256': lock['rejected_certificate_sha256'],
            'runtimes': records,
            'limits': ['Base-container trust and native/system package audits remain separate',
                       'This verifies exact SDK/Platform signatures and CA inputs, not all OS components']}


def prepare_installation(lock_path):
    lock = load_lock(lock_path)
    if INSTALLATION.exists():
        raise ValueError('Runtime preparation requires a fresh private installation')
    key = BASE_REPO / 'flathub.trustedkeys.gpg'
    if sha256(read_regular(key, 65536)) != lock['public_key_sha256']:
        raise ValueError('Pinned base public key differs from the reviewed key')
    command('flatpak', 'remote-add', '--system', f'--gpg-import={key}',
            'flathub', lock['remote_url'], env=runtime_env())
    repo = f'--repo={INSTALLATION / "repo"}'
    # libcurl's HTTP/2 bulk-delta transfer failed with a framing error. This
    # private remote uses OSTree's HTTP/1.1 option; TLS and GPG stay enabled.
    command('ostree', repo, 'config', 'set', 'remote "flathub".http2', 'false',
            env=runtime_env())
    verify_remote(lock)
    for entry in lock['runtimes']:
        # Native HTTP pulls authenticate signed commits. Cached objects remain
        # checksum-verified; no runtime from the image is silently reused.
        # Identify this application; the CDN rejects the bare library identifier.
        # Fetch the same verified objects directly: bulk delta transfers failed
        # with partial-file errors even over HTTP/1.1.
        command('ostree', repo, 'pull', '--untrusted', '--disable-static-deltas',
                f'--localcache-repo={BASE_REPO}',
                '--http-header=User-Agent=' + HTTP_USER_AGENT,
                'flathub', entry['commit'], env=runtime_env(), timeout=1800)
        signed_commit(entry)
        command('ostree', repo, 'refs', f'--create=flathub:{entry["ref"]}',
                entry['commit'], env=runtime_env())
    # Install only the authenticated local refs. A moving remote cannot select
    # a different revision or add unreviewed related extensions here.
    command('flatpak', 'install', '--system', '--noninteractive', '--no-pull',
            '--no-related', '--no-deps', 'flathub',
            *(entry['ref'] for entry in lock['runtimes']), env=runtime_env(), timeout=900)
    return inspect_runtime(lock_path)


def prepare_builder(lock_path, output):
    lock_path = lock_path.resolve()
    lock = load_lock(lock_path)
    if output.exists():
        raise ValueError('Use a fresh builder preparation output directory')
    output.mkdir(parents=True)
    output = output.resolve()
    if any(',' in str(path) for path in (lock_path.parent, output)):
        raise ValueError('Docker mount paths cannot contain commas')
    space = {'before_pull_free_bytes': shutil.disk_usage(output).free}
    (output / 'space.json').write_text(json.dumps(space, indent=2) + '\n')
    if space['before_pull_free_bytes'] < MINIMUM_FREE:
        raise ValueError('Builder preparation requires at least 25 GiB free')
    command('docker', 'pull', '--platform=linux/amd64', lock['base_image'], timeout=1200)
    space['after_pull_free_bytes'] = shutil.disk_usage(output).free
    (output / 'space.json').write_text(json.dumps(space, indent=2) + '\n')
    name = 'nvbroadcast-flatpak-runtime-' + uuid.uuid4().hex[:12]
    try:
        command('docker', 'create', '--pull=never', '--platform=linux/amd64',
                '--name', name, '--runtime=runc', '--log-driver=none',
                '--pids-limit=256', '--memory=4g', '--cpus=2',
                '--cap-drop=ALL', '--security-opt=no-new-privileges',
                '-e', 'NVIDIA_VISIBLE_DEVICES=void',
                '-e', f'FLATPAK_SYSTEM_DIR={INSTALLATION}',
                '--mount', f'type=bind,source={lock_path.parent},target=/runtime-policy,readonly',
                lock['base_image'], 'sleep', 'infinity')
        command('docker', 'start', name)
        result = command('docker', 'exec', name, 'python3', '/runtime-policy/runtime_policy.py',
                '--inside-container', '--lock', '/runtime-policy/' + lock_path.name,
                '--output', '/tmp/runtime-receipt.json', timeout=4500)
        receipt = json.loads(result)
        if receipt.get('status') != 'pass' or receipt.get('lock_sha256') != sha256(read_regular(lock_path)):
            raise ValueError('Prepared runtime receipt does not match the requested lock')
        (output / 'runtime-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        writable = command('docker', 'inspect', '--size', '--format', '{{.SizeRw}}', name).strip()
        space.update(before_commit_free_bytes=shutil.disk_usage(output).free,
                     writable_bytes=int(writable) if writable.isdigit() else None,
                     commit_free_margin_bytes=10 * 1024 ** 3)
        (output / 'space.json').write_text(json.dumps(space, indent=2) + '\n')
        if not writable.isdigit() or space['before_commit_free_bytes'] < int(writable) + 10 * 1024 ** 3:
            raise ValueError('Insufficient space to commit builder while retaining 10 GiB free')
        image = command('docker', 'commit', '--change', 'CMD ["/bin/sh"]', name, timeout=900).strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Docker did not return an immutable prepared image ID')
    finally:
        remove_container(name)
    try:
        proof = inspect_prepared_image(lock['base_image'], image)
        committed = inspect_committed_runtime(lock_path, image)
        if committed != receipt:
            raise ValueError('Fresh committed image runtime inspection differs from preparation')
        proof.update(schema_version=1, status='pass', lock_sha256=receipt['lock_sha256'],
                     runtime_inputs=committed)
        (output / 'prepared-image.json').write_text(json.dumps(proof, indent=2) + '\n')
        space['after_commit_free_bytes'] = shutil.disk_usage(output).free
        (output / 'space.json').write_text(json.dumps(space, indent=2) + '\n')
        (output / 'image-id').write_text(image + '\n')
    except BaseException as error:
        # This untagged image was just created for this preparation attempt.
        # Do not retain a failed builder; report its exact ID if cleanup fails.
        try:
            command('docker', 'image', 'rm', '--no-prune', image)
        except subprocess.SubprocessError as cleanup_error:
            error.add_note(f'Failed to remove newly prepared image {image}: {cleanup_error}')
        raise
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path(__file__).with_name('runtime-lock.json'))
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inside-container', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--prepare-hosted-runner-storage', action='store_true')
    args = parser.parse_args()
    if args.prepare_hosted_runner_storage:
        print(json.dumps(prepare_hosted_runner_storage(args.output)))
    elif args.inside_container or args.inspect:
        result = inspect_runtime(args.lock) if args.inspect else prepare_installation(args.lock)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    else:
        print(prepare_builder(args.lock, args.output))


if __name__ == '__main__':
    main()

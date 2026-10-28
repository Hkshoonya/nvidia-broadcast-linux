import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import ssl
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'packaging/flatpak'
SPEC = importlib.util.spec_from_file_location('runtime_policy', DIRECTORY / 'runtime_policy.py')
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)
LOCK = json.loads((DIRECTORY / 'runtime-lock.json').read_text())
SMOKE_SPEC = importlib.util.spec_from_file_location('runtime_smoke', DIRECTORY / 'runtime_smoke.py')
smoke = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(smoke)


# Public CA fixtures: a clean GNOME50 root and the rejected emSign ECC G3 root.
CLEAN_CA = """-----BEGIN CERTIFICATE-----
MIIBtjCCAVugAwIBAgITBmyf1XSXNmY/Owua2eiedgPySjAKBggqhkjOPQQDAjA5
MQswCQYDVQQGEwJVUzEPMA0GA1UEChMGQW1hem9uMRkwFwYDVQQDExBBbWF6b24g
Um9vdCBDQSAzMB4XDTE1MDUyNjAwMDAwMFoXDTQwMDUyNjAwMDAwMFowOTELMAkG
A1UEBhMCVVMxDzANBgNVBAoTBkFtYXpvbjEZMBcGA1UEAxMQQW1hem9uIFJvb3Qg
Q0EgMzBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABCmXp8ZBf8ANm+gBG1bG8lKl
ui2yEujSLtf6ycXYqm0fc4E7O5hrOXwzpcVOho6AF2hiRVd9RFgdszflZwjrZt6j
QjBAMA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0PAQH/BAQDAgGGMB0GA1UdDgQWBBSr
ttvXBp43rDCGB5Fwx5zEGbF4wDAKBggqhkjOPQQDAgNJADBGAiEA4IWSoxe3jfkr
BqWTrBqYaGFy+uGh0PsceGCmQ5nFuMQCIQCcAu/xlJyzlvnrxir4tiz+OpAUFteM
YyRIHN8wfdVoOw==
-----END CERTIFICATE-----
"""
CLEAN_CA_SUBJECT_HASH = '8cb5ee0f'
CLEAN_CA_SHA256 = '18ce6cfe7bf14e60b2e347b8dfe868cb31d02ebb3ada271569f50343b46db3a4'
REJECTED_CA = """-----BEGIN CERTIFICATE-----
MIICTjCCAdOgAwIBAgIKPPYHqWhwDtqLhDAKBggqhkjOPQQDAzBrMQswCQYDVQQG
EwJJTjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNo
bm9sb2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0g
RzMwHhcNMTgwMjE4MTgzMDAwWhcNNDMwMjE4MTgzMDAwWjBrMQswCQYDVQQGEwJJ
TjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNobm9s
b2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0gRzMw
djAQBgcqhkjOPQIBBgUrgQQAIgNiAAQjpQy4LRL1KPOxst3iAhKAnjlfSU2fySU0
WXTsuwYc58Byr+iuL+FBVIcUqEqy6HyC5ltqtdyzdc6LBtCGI79G1Y4PPwT01xyS
fvalY8L1X44uT6EYGQIrMgqCZH0Wk9GjQjBAMB0GA1UdDgQWBBR8XQKEE9TMipuB
zhccLikenEhjQjAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zAKBggq
hkjOPQQDAwNpADBmAjEAvvNhzwIQHWSVB7gYboiFBS+DCBeQyh+KTOgNG3qxrdWB
CUfvO6wIBHxcmbHtRwfSAjEAnbpV/KlK6O3t5nYBQnvI+GDZjVGLVTv7jHvrZQnD
+JbNR6iC8hZVdyR+EhCVBCyj
-----END CERTIFICATE-----
"""
REJECTED_CA_SUBJECT_HASH = '14bc7599'
REJECTED_CA_SHA256 = '86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b'


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lock = self.root / 'runtime-lock.json'
        self.lock.write_text(json.dumps(LOCK))

    def test_moving_image_missing_rejection_and_wrong_refs_fail_closed(self):
        for field, value in (('base_image', 'example/image:latest'),
                             ('rejected_certificate_sha256', []),
                             ('runtimes', [LOCK['runtimes'][0]])):
            with self.subTest(field=field):
                self.lock.write_text(json.dumps({**LOCK, field: value}))
                with self.assertRaises(ValueError):
                    policy.load_lock(self.lock)

    def test_redirected_lock_is_rejected(self):
        redirect = self.root / 'redirect.json'
        redirect.symlink_to(self.lock)
        with self.assertRaisesRegex(ValueError, 'redirected'):
            policy.load_lock(redirect)

    def test_remote_key_signatures_https_and_http1_are_required(self):
        repo = self.root / 'repo'
        repo.mkdir()
        key = b'reviewed public key'
        (repo / 'flathub.trustedkeys.gpg').write_bytes(key)
        lock = {**LOCK, 'public_key_sha256': hashlib.sha256(key).hexdigest()}
        good = '[remote "flathub"]\nurl=https://dl.flathub.org/repo/\ngpg-verify=true\ngpg-verify-summary=true\nhttp2=false\n'
        with patch.object(policy, 'INSTALLATION', self.root):
            for content in (good.replace('gpg-verify=true', 'gpg-verify=false'),
                            good.replace('gpg-verify-summary=true', 'gpg-verify-summary=false'),
                            good.replace('dl.flathub.org', 'example.invalid'),
                            good.replace('https://', 'http://'),
                            good + 'tls-permissive=true\n',
                            good.replace('http2=false', 'http2=true'),
                            good.replace('http2=false\n', '')):
                (repo / 'config').write_text(content)
                with self.assertRaises(ValueError):
                    policy.verify_remote(lock)
            (repo / 'config').write_text(good)
            policy.verify_remote(lock)
            (repo / 'flathub.trustedkeys.gpg').write_bytes(b'different key')
            with self.assertRaisesRegex(ValueError, 'public key'):
                policy.verify_remote(lock)

    def test_commit_signature_must_be_positive_and_bind_exact_commit(self):
        commit = 'a' * 64
        good = f'commit {commit}\nFound 1 signature:\nGood signature from "Flathub Repo Signing Key <flathub@flathub.org>"\n'
        policy.verify_signature(good, commit)
        for output in (good.replace(commit, 'b' * 64), good.replace('Found 1', 'Found 0'),
                       good + 'Signature expired\n', good.replace('Good signature', 'BAD signature')):
            with self.subTest(output=output), self.assertRaises(ValueError):
                policy.verify_signature(output, commit)

    def test_signed_commit_for_another_ref_is_rejected(self):
        entry = LOCK['runtimes'][0]
        with patch.object(policy, 'command', side_effect=['signed', "['runtime/org.example.Other/x86_64/50']"]), \
                patch.object(policy, 'verify_signature'):
            with self.assertRaisesRegex(ValueError, 'expected ref'):
                policy.signed_commit(entry)

    def test_rejected_ca_and_unreviewed_ca_bytes_fail(self):
        # PEM conversion is enough here; real ASN.1/default SSL contexts are
        # exercised by the finished-runtime smoke, not simulated by this test.
        pem = b'-----BEGIN CERTIFICATE-----\nYWJj\n-----END CERTIFICATE-----\n'
        digest = hashlib.sha256(pem).hexdigest()
        blocked = hashlib.sha256(b'abc').hexdigest()
        with self.assertRaisesRegex(ValueError, 'rejected certificate'):
            policy.inspect_ca(pem, digest, [blocked])
        with self.assertRaisesRegex(ValueError, 'reviewed commit'):
            policy.inspect_ca(pem, '0' * 64, [])
        self.assertEqual(policy.inspect_ca(pem, digest, [])['certificate_count'], 1)
        with self.assertRaisesRegex(ValueError, 'malformed'):
            policy.inspect_ca(b'', hashlib.sha256(b'').hexdigest(), [])

    def test_default_installation_or_stale_active_commit_cannot_pass(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'selected explicitly'):
                policy.inspect_runtime(self.lock)
        with patch.dict(os.environ, {'FLATPAK_SYSTEM_DIR': str(policy.INSTALLATION)}), \
                patch.object(policy, 'verify_remote'), \
                patch.object(policy, 'command', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'Active runtime commit'):
                policy.inspect_runtime(self.lock)

    def test_prepare_failure_never_publishes_image_and_always_cleans_container(self):
        for failing in ('create', 'start', 'exec', 'commit'):
            with self.subTest(failing=failing):
                calls = []
                output = self.root / failing
                def execute(*args, **kwargs):
                    calls.append(args)
                    if args[1] == failing:
                        raise subprocess.TimeoutExpired(args, 1)
                    if args[1] == 'exec':
                        return json.dumps({'status': 'pass', 'lock_sha256': policy.sha256(self.lock.read_bytes())})
                    if args[1] == 'inspect':
                        return '100'
                    return ''
                with patch.object(policy, 'command', side_effect=execute), \
                        patch.object(policy.shutil, 'disk_usage', return_value=SimpleNamespace(free=80 * 1024**3)):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        policy.prepare_builder(self.lock, output)
                self.assertEqual(calls[-1][:3], ('docker', 'rm', '--force'))
                self.assertFalse((output / 'image-id').exists())
                self.assertIn('--pull=never', next(args for args in calls if args[1] == 'create'))

    def test_capacity_guard_precedes_docker_pull(self):
        with patch.object(policy.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024)), \
                patch.object(policy, 'command') as command:
            with self.assertRaisesRegex(ValueError, '25 GiB'):
                policy.prepare_builder(self.lock, self.root / 'output')
            command.assert_not_called()

    @staticmethod
    def image_inspections():
        base = {'Id': 'sha256:' + 'b' * 64, 'RepoDigests': [LOCK['base_image']],
                'Architecture': 'amd64', 'Os': 'linux',
                'RootFS': {'Type': 'layers', 'Layers': ['sha256:' + 'c' * 64]}}
        prepared = {'Id': 'sha256:' + 'e' * 64, 'Architecture': 'amd64', 'Os': 'linux',
                    'RootFS': {'Type': 'layers', 'Layers': ['sha256:' + 'c' * 64, 'sha256:' + 'd' * 64]},
                    'Config': {'Env': ['PATH=/usr/bin', 'FLATPAK_SYSTEM_DIR=' + str(policy.INSTALLATION),
                                       'NVIDIA_VISIBLE_DEVICES=void']}}
        return base, prepared

    def test_committed_image_requires_actual_base_layers_environment_and_id(self):
        for changed in ('layers', 'environment', 'image-id', 'base-digest'):
            base, prepared = self.image_inspections()
            if changed == 'layers':
                prepared['RootFS']['Layers'][0] = 'sha256:' + 'f' * 64
            elif changed == 'environment':
                prepared['Config']['Env'][1] = 'FLATPAK_SYSTEM_DIR=/var/lib/flatpak'
            elif changed == 'image-id':
                prepared['Id'] = 'sha256:' + 'f' * 64
            else:
                base['RepoDigests'] = ['example.invalid/base@sha256:' + 'f' * 64]
            with self.subTest(changed=changed), \
                    patch.object(policy, 'command', side_effect=[json.dumps([base]), json.dumps([prepared])]), \
                    self.assertRaises(ValueError):
                policy.inspect_prepared_image(LOCK['base_image'], 'sha256:' + 'e' * 64)
        base, prepared = self.image_inspections()
        with patch.object(policy, 'command', side_effect=[json.dumps([base]), json.dumps([prepared])]):
            proof = policy.inspect_prepared_image(LOCK['base_image'], prepared['Id'])
        self.assertEqual(proof['prepared_image_id'], prepared['Id'])
        self.assertEqual(proof['prepared_rootfs_layers'][:-1], proof['base_rootfs_layers'])

    def test_fresh_committed_inspection_is_offline_and_inherits_environment(self):
        with patch.object(policy, 'command', return_value='{"status":"pass"}') as command, \
                patch.object(policy, 'remove_container') as cleanup:
            result = policy.inspect_committed_runtime(self.lock, 'sha256:' + 'e' * 64)
        args = command.call_args.args
        self.assertEqual(result['status'], 'pass')
        self.assertIn('--network=none', args)
        self.assertIn('--log-driver=none', args)
        self.assertIn('--inspect', args)
        self.assertNotIn('-e', args)  # A missing committed environment must not be repaired.
        self.assertIn('sha256:' + 'e' * 64, args)
        cleanup.assert_called_once()

    def test_failed_fresh_inspection_never_publishes_and_removes_new_image(self):
        calls = []
        output = self.root / 'failed-inspect'
        receipt = {'status': 'pass', 'lock_sha256': policy.sha256(self.lock.read_bytes())}
        def execute(*args, **kwargs):
            calls.append(args)
            if args[1] == 'exec':
                return json.dumps(receipt)
            if args[1] == 'inspect':
                return '100'
            if args[1] == 'commit':
                return 'sha256:' + 'e' * 64
            return ''
        with patch.object(policy, 'command', side_effect=execute), \
                patch.object(policy.shutil, 'disk_usage', return_value=SimpleNamespace(free=80 * 1024**3)), \
                patch.object(policy, 'inspect_prepared_image', return_value={}), \
                patch.object(policy, 'inspect_committed_runtime', side_effect=ValueError('fresh inspection failed')):
            with self.assertRaisesRegex(ValueError, 'fresh inspection failed'):
                policy.prepare_builder(self.lock, output)
        self.assertFalse((output / 'image-id').exists())
        self.assertFalse((output / 'prepared-image.json').exists())
        self.assertIn(('docker', 'image', 'rm', '--no-prune', 'sha256:' + 'e' * 64), calls)

    def test_preparation_publishes_only_matching_fresh_image_runtime_proof(self):
        base, prepared = self.image_inspections()
        receipt = {'status': 'pass', 'lock_sha256': policy.sha256(self.lock.read_bytes())}
        for matches in (True, False):
            output = self.root / f'fresh-{matches}'
            calls = []
            def execute(*args, **kwargs):
                calls.append(args)
                if args[1] == 'exec':
                    return json.dumps(receipt)
                if args[1] == 'run':
                    return json.dumps(receipt if matches else {**receipt, 'lock_sha256': 'f' * 64})
                if args[1] == 'inspect':
                    return '100'
                if args[1] == 'commit':
                    return prepared['Id']
                if args[1:3] == ('image', 'inspect'):
                    return json.dumps([base if args[-1] == LOCK['base_image'] else prepared])
                return ''
            with self.subTest(matches=matches), patch.object(policy, 'command', side_effect=execute), \
                    patch.object(policy.shutil, 'disk_usage', return_value=SimpleNamespace(free=80 * 1024**3)):
                if matches:
                    image = policy.prepare_builder(self.lock, output)
                    self.assertEqual(image, prepared['Id'])
                    self.assertEqual((output / 'image-id').read_text().strip(), image)
                    proof, digest = policy.verify_preparation_proof(
                        output / 'prepared-image.json', LOCK['base_image'], image, receipt)
                    self.assertEqual(proof['runtime_inputs'], receipt)
                    self.assertEqual(digest, policy.sha256((output / 'prepared-image.json').read_bytes()))
                else:
                    with self.assertRaisesRegex(ValueError, 'runtime inspection differs'):
                        policy.prepare_builder(self.lock, output)
                    self.assertFalse((output / 'image-id').exists())
                    self.assertFalse((output / 'prepared-image.json').exists())
                    self.assertIn(('docker', 'image', 'rm', '--no-prune', prepared['Id']), calls)

    def test_runner_cleanup_refuses_local_self_hosted_or_other_platform_before_mutation(self):
        hosted = {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
                  'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64'}
        for environment in ({}, {**hosted, 'RUNNER_ENVIRONMENT': 'self-hosted'},
                            {**hosted, 'RUNNER_OS': 'macOS'}, {**hosted, 'RUNNER_ARCH': 'ARM64'}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch.object(policy, 'command') as command:
                with self.assertRaisesRegex(ValueError, 'restricted to GitHub-hosted'):
                    policy.prepare_hosted_runner_storage(self.root / 'forbidden')
                command.assert_not_called()
                self.assertFalse((self.root / 'forbidden').exists())

    def test_runner_cleanup_uses_only_fixed_toolchains_and_retains_space_guard(self):
        toolchain = self.root / 'unused-toolchain'
        toolchain.mkdir()
        hosted = {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
                  'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64'}
        for available, succeeds in ((30, True), (20, False)):
            output = self.root / f'storage-{available}'
            with self.subTest(available=available), patch.dict(os.environ, hosted, clear=True), \
                    patch.object(policy, 'HOSTED_TOOLCHAINS', (toolchain, self.root / 'absent')), \
                    patch.object(policy, 'command') as command, \
                    patch.object(policy.shutil, 'disk_usage', side_effect=[
                        SimpleNamespace(free=14 * 1024**3), SimpleNamespace(free=available * 1024**3),
                        SimpleNamespace(free=available * 1024**3)]):
                if succeeds:
                    self.assertEqual(policy.prepare_hosted_runner_storage(output)['status'], 'pass')
                else:
                    with self.assertRaisesRegex(ValueError, '25 GiB'):
                        policy.prepare_hosted_runner_storage(output)
                command.assert_called_once_with('sudo', '--non-interactive', '/usr/bin/rm', '-rf', '--',
                                                str(toolchain), timeout=300)
            receipt = json.loads((output / 'storage-receipt.json').read_text())
            self.assertEqual(receipt['free_bytes_after'], available * 1024**3)
            self.assertEqual(receipt['status'], 'pass' if succeeds else 'fail')

    def test_runner_cleanup_rejects_redirected_toolchain(self):
        link = self.root / 'redirect'
        link.symlink_to(self.root, target_is_directory=True)
        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
                'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64'}, clear=True), \
                patch.object(policy, 'HOSTED_TOOLCHAINS', (link,)), \
                patch.object(policy, 'command') as command, \
                patch.object(policy.shutil, 'disk_usage', return_value=SimpleNamespace(free=14 * 1024**3)):
            with self.assertRaisesRegex(ValueError, 'redirected'):
                policy.prepare_hosted_runner_storage(self.root / 'redirect-output')
            command.assert_not_called()

    def test_runner_with_sufficient_space_does_not_remove_toolchains(self):
        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
                'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64'}, clear=True), \
                patch.object(policy, 'command') as command, \
                patch.object(policy.shutil, 'disk_usage', return_value=SimpleNamespace(free=80 * 1024**3)):
            receipt = policy.prepare_hosted_runner_storage(self.root / 'already-sufficient')
            self.assertEqual(receipt['status'], 'pass')
            self.assertEqual(receipt['removed_toolchains'], [])
            command.assert_not_called()

    def test_failed_native_fetch_does_not_install_or_return_success(self):
        key = b'key'
        self.lock.write_text(json.dumps({**LOCK, 'public_key_sha256': policy.sha256(key)}))
        base = self.root / 'base'
        base.mkdir()
        (base / 'flathub.trustedkeys.gpg').write_bytes(key)
        calls = []
        def execute(*args, **kwargs):
            calls.append(args)
            if 'pull' in args:
                raise subprocess.CalledProcessError(1, args)
            return ''
        with patch.object(policy, 'BASE_REPO', base), \
                patch.object(policy, 'INSTALLATION', self.root / 'private'), \
                patch.object(policy, 'verify_remote'), \
                patch.object(policy, 'command', side_effect=execute):
            with self.assertRaises(subprocess.CalledProcessError):
                policy.prepare_installation(self.lock)
        self.assertFalse(any('install' in args for args in calls))
        configured = next(args for args in calls if 'config' in args)
        self.assertEqual(configured, ('ostree', '--repo=' + str(self.root / 'private/repo'),
                         'config', 'set', 'remote "flathub".http2', 'false'))
        pull = next(args for args in calls if 'pull' in args)
        self.assertLess(calls.index(configured), calls.index(pull))
        self.assertIn('--untrusted', pull)
        self.assertIn('--disable-static-deltas', pull)
        self.assertEqual([arg for arg in pull if arg.startswith('--http-header=')], [
            '--http-header=User-Agent=NVBroadcast-runtime-preparation '
            '(https://github.com/Hkshoonya/nvidia-broadcast-linux)'])
        self.assertNotIn('--http-trusted', pull)
        self.assertNotIn('--disable-verify-bindings', pull)

    def test_application_context_rejects_bad_root_and_disabled_verification(self):
        context = SimpleNamespace(verify_mode=ssl.CERT_REQUIRED, check_hostname=True,
                                  get_ca_certs=lambda **kwargs: [b'root'])
        smoke.check_context(context, set())
        with self.assertRaisesRegex(ValueError, 'rejected root'):
            smoke.check_context(context, {hashlib.sha256(b'root').hexdigest()})
        context.verify_mode = ssl.CERT_NONE
        with self.assertRaisesRegex(ValueError, 'certificate/hostname'):
            smoke.check_context(context, set())
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False
        with self.assertRaisesRegex(ValueError, 'certificate/hostname'):
            smoke.check_context(context, set())
        context.get_ca_certs = lambda **kwargs: []
        with self.assertRaisesRegex(ValueError, 'empty'):
            smoke.check_context(context, set())

    def test_shared_setuptools_origin_cannot_claim_app_override(self):
        module = SimpleNamespace(__name__='setuptools', __file__='/usr/lib/python3.13/site-packages/setuptools/__init__.py')
        with self.assertRaisesRegex(ValueError, 'shared runtime'):
            smoke.app_origin(module)


    def make_ca_directory(self, name, pem, subject_hash):
        directory = self.root / name
        directory.mkdir()
        (directory / (subject_hash + '.0')).write_text(pem)
        return directory

    def test_capath_rejection_covers_root_omitted_by_lazy_context(self):
        cafile = self.root / 'clean-ca.pem'
        cafile.write_text(CLEAN_CA)
        directory = self.make_ca_directory('rejected-capath', REJECTED_CA,
                                           REJECTED_CA_SUBJECT_HASH)
        context = ssl.create_default_context(cafile=str(cafile), capath=str(directory))
        rejected = {REJECTED_CA_SHA256}
        # This demonstrates the old false negative with actual OpenSSL: its
        # unused hashed directory root is absent from the loaded-root list.
        self.assertEqual(smoke.check_context(context, rejected)['root_count'], 1)
        self.assertEqual({hashlib.sha256(x).hexdigest()
                          for x in context.get_ca_certs(binary_form=True)},
                         {CLEAN_CA_SHA256})
        with patch.dict(os.environ, {'SSL_CERT_DIR': str(directory)}):
            defaults = ssl.get_default_verify_paths()
            with self.assertRaisesRegex(ValueError, 'rejected root'):
                smoke.check_capaths(defaults, rejected)

    def test_clean_capath_and_public_certificate_symlink_pass_offline(self):
        directory = self.make_ca_directory('clean-capath', CLEAN_CA,
                                           CLEAN_CA_SUBJECT_HASH)
        certificate = directory / (CLEAN_CA_SUBJECT_HASH + '.0')
        canonical = self.root / 'public-ca.pem'
        certificate.rename(canonical)
        certificate.symlink_to(canonical)
        (directory / 'unrelated.txt').write_text('not a CA')
        with patch.dict(os.environ, {'SSL_CERT_DIR': str(directory)}):
            records = smoke.check_capaths(ssl.get_default_verify_paths(),
                                          {REJECTED_CA_SHA256})
        self.assertEqual(records[0]['hashed_files'], 1)
        self.assertEqual(records[0]['root_count'], 1)
        self.assertFalse(records[0]['rejected_certificates_present'])

    def test_raw_capath_list_cannot_hide_rejected_root(self):
        clean = self.make_ca_directory('clean', CLEAN_CA, CLEAN_CA_SUBJECT_HASH)
        rejected = self.make_ca_directory('rejected', REJECTED_CA,
                                          REJECTED_CA_SUBJECT_HASH)
        paths = os.pathsep.join((str(clean), str(rejected)))
        with patch.dict(os.environ, {'SSL_CERT_DIR': paths}):
            defaults = ssl.get_default_verify_paths()
            self.assertIsNone(defaults.capath)
            with self.assertRaisesRegex(ValueError, 'rejected root'):
                smoke.check_capaths(defaults, {REJECTED_CA_SHA256})


if __name__ == '__main__':
    unittest.main()

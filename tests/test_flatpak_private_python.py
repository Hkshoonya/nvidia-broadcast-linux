import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
import unittest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'packaging/flatpak'
SPEC = importlib.util.spec_from_file_location('private_python', HERE / 'private_python.py')
private = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(private)


class PrivatePythonTests(unittest.TestCase):
    def make_prefix_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repo, prefix = root / 'repo', root / 'prefix'
        sources = {'private-inputs.json': 'packaging/flatpak/inputs.json',
                   'python-inputs.json': private.PYTHON_INPUTS, 'pylock.build.toml': private.BUILD_LOCK,
                   'prepare.py': 'packaging/runtime-prototype/prepare.py'}
        for name in sources.values():
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        input_path = repo / 'packaging/flatpak/inputs.json'
        inputs = json.loads(input_path.read_text())
        inputs['ssl_fixture']['sha256'] = hashlib.sha256(b'fixture bytes').hexdigest()
        input_path.write_text(json.dumps(inputs))
        _, python, tools = private.load_inputs(input_path, repo / private.PYTHON_INPUTS, repo / private.BUILD_LOCK)
        (prefix / 'bin').mkdir(parents=True)
        (prefix / 'bin/python3.13').write_bytes(b'fixture executable, never run')
        fixture = prefix / inputs['ssl_fixture']['path']
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b'fixture bytes')
        provenance = prefix / 'share/nvbroadcast-runtime-provenance'
        licenses = provenance / 'python-licenses'
        licenses.mkdir(parents=True)
        (licenses / 'LICENSE.cpython.txt').write_text('fixture notice')
        metadata = provenance / 'PYTHON.json'
        metadata.write_text(json.dumps({'python_version': python['version'], 'python_tag': 'cp313',
                                        'target_triple': 'x86_64-unknown-linux-gnu'}))
        for name, source in (('flatpak-inputs.json', input_path), ('python-inputs.json', repo / private.PYTHON_INPUTS),
                             ('pylock.build.toml', repo / private.BUILD_LOCK)):
            shutil.copyfile(source, provenance / name)
        record = {'schema_version': 1, 'python': python, 'tools': tools, 'ssl_fixture': inputs['ssl_fixture'],
                  'input_sha256': {name: private.digest(repo / source) for name, source in sources.items()},
                  'interpreter': '/app/bin/python3.13', 'pythonpath': private.PYTHON_PATH,
                  'shared_python_path': private.SHARED_SITE, 'python_metadata_sha256': private.digest(metadata),
                  'python_binary_sha256': private.digest(prefix / 'bin/python3.13'),
                  'license_files': {'LICENSE.cpython.txt': private.digest(licenses / 'LICENSE.cpython.txt')}}
        path = provenance / 'flatpak-private-python.json'
        path.write_text(json.dumps(record))
        return repo, prefix, path, record

    def test_generated_module_matches_authoritative_python_and_offline_tool_pins(self):
        module = json.loads((HERE / 'python3-private-runtime.yml').read_text())
        self.assertEqual(module, private.generated_module(ROOT))
        canonical = json.loads((ROOT / private.PYTHON_INPUTS).read_text())['python']
        self.assertEqual(module['sources'][0]['url'], canonical['url'])
        self.assertEqual(module['sources'][0]['sha256'], canonical['sha256'])
        self.assertEqual(module['build-options']['build-args'], ['--unshare=network'])
        _, python, tools = private.load_inputs(HERE / 'inputs.json', ROOT / private.PYTHON_INPUTS,
                                              ROOT / private.BUILD_LOCK)
        self.assertEqual(python['version'], '3.13.18')
        self.assertEqual({x['name']: x['version'] for x in tools},
                         {'pip': '26.2.2', 'packaging': '26.0', 'setuptools': '84.1.0', 'wheel': '0.48.0'})
        self.assertEqual(len([x for x in module['sources'] if 'url' in x]), 5)

    def test_wrong_python_floor_and_unhashed_or_wrong_wheel_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / 'python.json'
            canonical = json.loads((ROOT / private.PYTHON_INPUTS).read_text())
            for version in ('3.13.16', '3.12.16', '3.14.10'):
                with self.subTest(version=version):
                    python.write_text(json.dumps({**canonical, 'python': {**canonical['python'], 'version': version}}))
                    with self.assertRaisesRegex(ValueError, 'patched CPython'):
                        private.load_inputs(HERE / 'inputs.json', python, ROOT / private.BUILD_LOCK)
        _, _, tools = private.load_inputs(HERE / 'inputs.json', ROOT / private.PYTHON_INPUTS,
                                          ROOT / private.BUILD_LOCK)
        pip = tools[0]
        for change in ({'sha256': ''}, {'url': pip['url'].replace('files.pythonhosted.org', 'example.invalid')},
                       {'version': '26.2.1'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                private.wheel_name({**pip, **change})

    def test_install_cannot_modify_an_external_interpreter_prefix(self):
        for prefix in ('/usr', '/usr/local', '/var/lib/nvbroadcast-flatpak'):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, '/app prefix'):
                private.install(Path(prefix))

    def test_prefix_copy_preserves_links_and_notices_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, prefix = root / 'runtime', root / 'prefix'
            (runtime / 'bin').mkdir(parents=True)
            (runtime / 'bin/python3.13').write_bytes(b'fixture executable')
            (runtime / 'bin/python3').symlink_to('python3.13')
            (runtime / 'share/python-licenses').mkdir(parents=True)
            (runtime / 'share/python-licenses/LICENSE.cpython.txt').write_text('fixture license')
            (prefix / 'bin').mkdir(parents=True)
            private.copy_prefix(runtime, prefix)
            self.assertTrue((prefix / 'bin/python3').is_symlink())
            self.assertEqual((prefix / 'share/python-licenses/LICENSE.cpython.txt').read_text(), 'fixture license')
            with self.assertRaisesRegex(ValueError, 'not fresh'):
                private.copy_prefix(runtime, prefix)

    def test_export_provenance_binds_canonical_pins_executable_fixture_and_native_notices(self):
        repo, prefix, path, record = self.make_prefix_fixture()
        actual, digest = private.verify_prefix_provenance(prefix, repo)
        self.assertEqual(actual, record)
        self.assertEqual(digest, private.digest(path))
        for field, value in (('python', {**record['python'], 'version': '3.13.16'}),
                             ('tools', []), ('python_binary_sha256', '0' * 64),
                             ('license_files', {'../PYTHON.json': '0' * 64})):
            with self.subTest(field=field):
                path.write_text(json.dumps({**record, field: value}))
                with self.assertRaises(ValueError):
                    private.verify_prefix_provenance(prefix, repo)
        path.write_text(json.dumps(record))
        (prefix / 'bin/python3.13').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'metadata/binary/SSL'):
            private.verify_prefix_provenance(prefix, repo)

    def test_app_path_and_default_ca_are_explicit_in_build_and_installed_environment(self):
        manifest = (HERE / 'com.nvbroadcast.NVBroadcast.yml').read_text()
        self.assertIn('  - --env=PYTHONPATH=' + private.PYTHON_PATH + '\n', manifest)
        self.assertIn('    PYTHONPATH: ' + private.PYTHON_PATH + '\n', manifest)
        self.assertIn('  - --env=SSL_CERT_FILE=' + private.CA_FILE + '\n', manifest)
        self.assertIn('  - --env=SSL_CERT_DIR=' + private.CA_DIRECTORY + '\n', manifest)


if __name__ == '__main__':
    unittest.main()

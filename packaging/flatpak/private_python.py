#!/usr/bin/env python3
"""Generate and install the Flatpak's verified, offline private Python prefix."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import urllib.parse


HERE = Path(__file__).resolve().parent
PYTHON_INPUTS = 'packaging/runtime-prototype/inputs.json'
BUILD_LOCK = 'packaging/runtime-prototype/pylock.build.toml'
APP_SITE = '/app/lib/python3.13/site-packages'
SHARED_SITE = '/usr/lib/python3.13/site-packages'
PYTHON_PATH = APP_SITE + ':' + SHARED_SITE
CA_FILE = '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem'
CA_DIRECTORY = '/etc/pki/tls/certs'


def bootstrap_environment():
    # Shared GI/cairo belongs to the finished app, not pip's installation view:
    # outside a venv, force-reinstall may uninstall any visible SDK distribution.
    return {**os.environ, 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': APP_SITE,
            'SSL_CERT_FILE': CA_FILE, 'SSL_CERT_DIR': CA_DIRECTORY}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_input(path, maximum=4 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f'Missing, redirected or oversized private Python input: {path}')
    return path.read_bytes()


def wheel_name(item):
    url = urllib.parse.urlsplit(item['url'])
    name = urllib.parse.unquote(url.path.rsplit('/', 1)[-1])
    expected = item['name'].replace('-', '_') + '-' + item['version'] + '-'
    if (url.scheme != 'https' or url.hostname != 'files.pythonhosted.org' or
            not name.startswith(expected) or not name.endswith('.whl') or
            not re.fullmatch(r'[A-Za-z0-9_.+-]+', name) or
            not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
        raise ValueError('Private Python build tools require exact hashed PyPI wheels')
    return name


def load_inputs(inputs_path, python_path, lock_path):
    inputs = json.loads(read_input(inputs_path))
    if (set(inputs) != {'schema_version', 'target', 'python_inputs', 'build_lock', 'build_tools', 'bootstrap', 'ssl_fixture'} or
            inputs['schema_version'] != 1 or inputs['target'] != 'linux-x86_64-cp313-cpu' or
            inputs['python_inputs'] != PYTHON_INPUTS or inputs['build_lock'] != BUILD_LOCK or
            inputs['build_tools'] != ['packaging', 'setuptools', 'wheel'] or
            len(inputs['bootstrap']) != 1 or inputs['bootstrap'][0]['name'] != 'pip' or
            inputs['bootstrap'][0]['version'] != '26.2.2'):
        raise ValueError('Private Python input identity or offline build-tool policy differs')
    fixture = inputs['ssl_fixture']
    if (set(fixture) != {'path', 'sha256', 'source_url'} or
            fixture['path'] != 'lib/python3.13/test/certdata/distrusted.zip' or
            not re.fullmatch(r'[0-9a-f]{64}', fixture['sha256']) or
            not re.fullmatch(r'https://raw\.githubusercontent\.com/python/cpython/[0-9a-f]{40}/Lib/test/certdata/distrusted\.zip', fixture['source_url'])):
        raise ValueError('Private Python SSL regression fixture must have a reviewed immutable identity')
    python = json.loads(read_input(python_path))['python']
    version = re.fullmatch(r'3\.13\.([0-9]+)', python['version'])
    if (not version or int(version[1]) < 18 or
            not re.fullmatch(r'[0-9a-f]{64}', python['sha256']) or
            not python['url'].startswith('https://github.com/astral-sh/python-build-standalone/releases/download/')):
        raise ValueError('The app requires a reviewed patched CPython 3.13 input')
    lock = tomllib.loads(read_input(lock_path).decode())
    if lock['lock-version'] != '1.0':
        raise ValueError('Unsupported private Python build lock')
    tools = list(inputs['bootstrap'])
    for name in inputs['build_tools']:
        package, = (item for item in lock['packages'] if item['name'] == name)
        wheel, = package['wheels']
        tools.append({'name': name, 'version': package['version'],
                      'url': wheel['url'], 'sha256': wheel['hashes']['sha256']})
    for tool in tools:
        wheel_name(tool)
    return inputs, python, tools


def generated_module(repo):
    inputs, python, tools = load_inputs(repo / 'packaging/flatpak/inputs.json',
                                       repo / PYTHON_INPUTS, repo / BUILD_LOCK)
    # Preserve the archive-authenticated executable bytes and install-time hash.
    # These options apply only to this module, not subsequent app dependencies.
    return {'name': 'python3-private-runtime', 'buildsystem': 'simple',
            'build-options': {'build-args': ['--unshare=network'],
                              'strip': False, 'no-debuginfo': True},
            'build-commands': ['test "$(uname -m)" = "x86_64"',
                               'python3 private_python.py install --prefix=${FLATPAK_DEST}'],
            'sources': [
                {'type': 'file', 'url': python['url'], 'sha256': python['sha256'],
                 'dest-filename': 'python-input.tar.zst', 'only-arches': ['x86_64']},
                *({'type': 'file', 'url': tool['url'], 'sha256': tool['sha256'],
                   'dest-filename': wheel_name(tool)} for tool in tools),
                {'type': 'file', 'path': 'private_python.py'},
                {'type': 'file', 'path': 'inputs.json', 'dest-filename': 'private-inputs.json'},
                {'type': 'file', 'path': '../runtime-prototype/inputs.json', 'dest-filename': 'python-inputs.json'},
                {'type': 'file', 'path': '../runtime-prototype/pylock.build.toml', 'dest-filename': 'pylock.build.toml'},
                {'type': 'file', 'path': '../runtime-prototype/prepare.py', 'dest-filename': 'prepare.py'},
            ],
            'cleanup': ['/bin/wheel', '/lib/python3.13/site-packages/wheel',
                        '/lib/python3.13/site-packages/wheel-*.dist-info']}


def verify_prefix_provenance(prefix, repo):
    inputs, python, tools = load_inputs(repo / 'packaging/flatpak/inputs.json',
                                       repo / PYTHON_INPUTS, repo / BUILD_LOCK)
    provenance = prefix / 'share/nvbroadcast-runtime-provenance'
    path = provenance / 'flatpak-private-python.json'
    record = json.loads(read_input(path))
    source_files = {'private-inputs.json': repo / 'packaging/flatpak/inputs.json',
                    'python-inputs.json': repo / PYTHON_INPUTS, 'pylock.build.toml': repo / BUILD_LOCK,
                    'prepare.py': repo / 'packaging/runtime-prototype/prepare.py'}
    if (record.get('schema_version') != 1 or record.get('python') != python or
            record.get('tools') != tools or record.get('ssl_fixture') != inputs['ssl_fixture'] or
            record.get('input_sha256') != {name: digest(source) for name, source in source_files.items()} or
            record.get('interpreter') != '/app/bin/python3.13' or record.get('pythonpath') != PYTHON_PATH or
            record.get('tool_locations') != {tool['name']: APP_SITE for tool in tools} or
            record.get('shared_python_path') != SHARED_SITE):
        raise ValueError('Private Python prefix provenance differs from the named source inputs')
    for name, source in (('flatpak-inputs.json', source_files['private-inputs.json']),
                         ('python-inputs.json', source_files['python-inputs.json']),
                         ('pylock.build.toml', source_files['pylock.build.toml'])):
        if read_input(provenance / name) != read_input(source):
            raise ValueError('Packaged private Python source inputs differ')
    metadata = json.loads(read_input(provenance / 'PYTHON.json'))
    if (digest(provenance / 'PYTHON.json') != record.get('python_metadata_sha256') or
            metadata.get('python_version') != python['version'] or metadata.get('python_tag') != 'cp313' or
            metadata.get('target_triple') != 'x86_64-unknown-linux-gnu' or
            digest(prefix / 'bin/python3.13') != record.get('python_binary_sha256') or
            digest(prefix / inputs['ssl_fixture']['path']) != inputs['ssl_fixture']['sha256']):
        raise ValueError('Packaged private interpreter metadata/binary/SSL fixture differs')
    licenses = provenance / 'python-licenses'
    records = record.get('license_files', {})
    if not records or len(records) > 256 or not (licenses / 'LICENSE.cpython.txt').is_file():
        raise ValueError('Private interpreter/native notices are missing')
    for name, expected in records.items():
        path = (licenses / name).resolve()
        if (not path.is_relative_to(licenses.resolve()) or not path.is_file() or
                path.stat().st_size > 1024 * 1024 or digest(path) != expected):
            raise ValueError('Private interpreter/native notice differs')
    return record, digest(provenance / 'flatpak-private-python.json')


def copy_prefix(runtime, prefix):
    # Flatpak may have created empty prefix directories already. Merge those,
    # but never overwrite an existing file, link, or incompatible directory.
    for source in runtime.rglob('*'):
        target = prefix / source.relative_to(runtime)
        if target.exists() or target.is_symlink():
            if not (source.is_dir() and not source.is_symlink() and
                    target.is_dir() and not target.is_symlink()):
                raise ValueError(f'Private Python prefix is not fresh: {target}')
    shutil.copytree(runtime, prefix, dirs_exist_ok=True, symlinks=True)


def install(prefix):
    if prefix != Path('/app'):
        raise ValueError('Private Flatpak Python must use the /app prefix')
    inputs_path, python_path, lock_path = map(Path, ('private-inputs.json', 'python-inputs.json', 'pylock.build.toml'))
    inputs, python, tools = load_inputs(inputs_path, python_path, lock_path)
    unpacker_path = Path('prepare.py')
    read_input(unpacker_path)
    spec = importlib.util.spec_from_file_location('reviewed_python_unpacker', unpacker_path)
    unpacker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unpacker)
    stage = Path('private-python-stage')
    record = unpacker.unpack_python(Path('python-input.tar.zst'), python, stage)
    runtime = stage / 'runtime'
    if digest(runtime / inputs['ssl_fixture']['path']) != inputs['ssl_fixture']['sha256']:
        raise ValueError('The authenticated Python archive lacks the reviewed SSL regression fixture')
    copy_prefix(runtime, prefix)
    shutil.rmtree(stage)
    for tool in tools:
        path = Path(wheel_name(tool))
        read_input(path, 16 * 1024 * 1024)
        if digest(path) != tool['sha256']:
            raise ValueError('Offline private Python wheel differs from its hash')
    requirements = Path('private-build-requirements.txt')
    requirements.write_text(''.join(f'{tool["name"]}=={tool["version"]} --hash=sha256:{tool["sha256"]}\n'
                                    for tool in tools))
    interpreter = prefix / 'bin/python3.13'
    environment = bootstrap_environment()
    subprocess.run([str(interpreter), '-m', 'pip', '--isolated', 'install', '--no-index',
                    '--no-deps', '--no-cache-dir', '--force-reinstall', '--require-hashes',
                    '--only-binary=:all:', '--find-links=.', '-r', str(requirements)],
                   env=environment, check=True, timeout=300)
    expected_versions = {tool['name']: tool['version'] for tool in tools}
    check = ('import sys,json,importlib.metadata as m; from pathlib import Path; '
             'assert sys.version_info[:2] == (3,13) and sys.version_info[2] >= 18; '
             'assert sys.executable == "/app/bin/python3.13"; '
             'print(json.dumps({name:{"version":m.version(name), '
             '"location":str(Path(m.distribution(name).locate_file("")).resolve())} '
             'for name in ' + repr(list(expected_versions)) + '}))')
    result = subprocess.run([str(interpreter), '-c', check], env=environment,
                            check=True, capture_output=True, text=True, timeout=30)
    actual_tools = json.loads(result.stdout)
    if (set(actual_tools) != set(expected_versions) or
            any(item['version'] != expected_versions[name] or item['location'] != APP_SITE
                for name, item in actual_tools.items())):
        raise ValueError('Private Python build-tool versions or private-prefix origins differ after offline installation')
    provenance = prefix / 'share/nvbroadcast-runtime-provenance'
    shutil.copyfile(inputs_path, provenance / 'flatpak-inputs.json')
    shutil.copyfile(python_path, provenance / 'python-inputs.json')
    shutil.copyfile(lock_path, provenance / 'pylock.build.toml')
    (provenance / 'flatpak-private-python.json').write_text(json.dumps({
        'schema_version': 1, 'python': python, 'tools': tools,
        'tool_locations': {name: item['location'] for name, item in actual_tools.items()},
        'input_sha256': {str(path): digest(path) for path in (inputs_path, python_path, lock_path, unpacker_path)},
        'python_metadata_sha256': record['python_metadata_sha256'],
        'python_binary_sha256': digest(interpreter),
        'license_files': record['license_files'], 'interpreter': str(interpreter),
        'shared_python_path': SHARED_SITE, 'pythonpath': PYTHON_PATH,
        'ssl_fixture': inputs['ssl_fixture'],
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'check', 'install'))
    parser.add_argument('--prefix', type=Path, default=Path('/app'))
    args = parser.parse_args()
    if args.command == 'install':
        install(args.prefix)
        return
    path = HERE / 'python3-private-runtime.yml'
    expected = json.dumps(generated_module(HERE.parents[1]), indent=2) + '\n'
    if args.command == 'generate':
        path.write_text(expected)
    elif path.read_text() != expected:
        raise ValueError('Regenerate private runtime sources from their authoritative pins')


if __name__ == '__main__':
    main()

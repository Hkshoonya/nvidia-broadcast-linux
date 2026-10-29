#!/usr/bin/env python3
"""Check the finished application's actual imports and TLS contexts, offline."""

import sys

startup_shim = sys.modules.get('_distutils_hack')

import hashlib
import http.client
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import ssl
import sysconfig
import tempfile
import zipfile


def app_origin(module):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to('/app/'):
        raise ValueError(f'{module.__name__} comes from the shared runtime: {path}')
    return str(path)


def private_ssl_origin(module, interpreter):
    # The reviewed standalone CPython links _ssl into its executable. Bind that
    # implementation to the private binary instead of expecting a shared object.
    if (getattr(module, '__file__', None) is None and module.__name__ == '_ssl' and
            module.__spec__.origin == 'built-in' and '_ssl' in sys.builtin_module_names):
        return {'kind': 'built-in', 'interpreter': str(interpreter),
                'interpreter_sha256': hashlib.sha256(interpreter.read_bytes()).hexdigest()}
    if getattr(module, '__file__', None) is None:
        raise ValueError('The private SSL implementation has no authenticated origin')
    return {'kind': 'extension', 'path': app_origin(module)}


def check_context(context, rejected):
    roots = context.get_ca_certs(binary_form=True)
    fingerprints = {hashlib.sha256(cert).hexdigest() for cert in roots}
    if not roots or fingerprints.intersection(rejected):
        raise ValueError('An application-facing TLS context is empty or trusts a rejected root')
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise ValueError('Application TLS context does not enforce certificate/hostname checks')
    return {'root_count': len(roots), 'rejected_certificates_present': False}


def check_capaths(defaults, rejected):
    # OpenSSL loads capath certificates lazily. Force each hashed CA file into
    # an offline context so get_ca_certs() cannot omit an unused rejected root.
    # The raw setting also covers SSL_CERT_DIR lists whose resolved capath is None.
    configured = os.environ.get(defaults.openssl_capath_env, defaults.openssl_capath) or ''
    records = []
    for name in configured.split(os.pathsep):
        if not name:
            continue
        directory = Path(name)
        if not directory.is_dir():
            records.append({'path': name, 'exists': False, 'hashed_files': 0})
            continue
        files = sorted(path for path in directory.iterdir()
                       if re.fullmatch(r'[0-9a-fA-F]{8}\.[0-9]+', path.name))
        if len(files) > 4096:
            raise ValueError('Application CA directory contains too many hashed files')
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        total = 0
        for path in files:
            if not path.is_file() or path.stat().st_size > 1024 * 1024:
                raise ValueError(f'Missing or oversized hashed CA file: {path}')
            total += path.stat().st_size
            if total > 16 * 1024 * 1024:
                raise ValueError('Application hashed CA files exceed their byte bound')
            context.load_verify_locations(cafile=str(path))
        result = check_context(context, rejected) if files else {'root_count': 0}
        records.append({'path': name, 'exists': True, 'hashed_files': len(files), **result})
    return records


def check_spki_distrust(fixture, expected_hash):
    """Use CPython's otherwise-valid compromised-intermediate fixture offline."""
    if (fixture.is_symlink() or not fixture.is_file() or fixture.stat().st_size > 65536 or
            hashlib.sha256(fixture.read_bytes()).hexdigest() != expected_hash):
        raise ValueError('Private Python SSL fixture differs from its reviewed hash')
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    names = {'distrusted/', 'distrusted/distrusted_keycert.pem',
             'distrusted/distrusted_intermediate.pem', 'distrusted/distrusted_root.pem'}
    with tempfile.TemporaryDirectory() as temporary, zipfile.ZipFile(fixture) as archive:
        items = archive.infolist()
        if len(items) != 4 or {item.filename for item in items} != names or sum(item.file_size for item in items) > 16384:
            raise ValueError('Private Python SSL fixture has unexpected members or sizes')
        root = Path(temporary) / 'root.pem'
        keycert = Path(temporary) / 'keycert.pem'
        # Fixed public upstream test-fixture password, not a credential.
        password = b'Brave Sir Robin'
        root.write_bytes(archive.read('distrusted/distrusted_root.pem', pwd=password))
        keycert.write_bytes(archive.read('distrusted/distrusted_keycert.pem', pwd=password))
        client_context.load_verify_locations(cafile=str(root))
        server_context.load_cert_chain(str(keycert))
    client_in, client_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    server_in, server_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    client = client_context.wrap_bio(client_in, client_out, server_hostname='localhost')
    server = server_context.wrap_bio(server_in, server_out, server_side=True)
    completed = set()
    for _ in range(32):
        for name, endpoint in (('client', client), ('server', server)):
            try:
                endpoint.do_handshake()
                completed.add(name)
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                pass
            except ssl.SSLCertVerificationError as error:
                if name != 'client' or error.verify_code != 28:
                    raise ValueError('SSL fixture failed for a reason other than distrusted SPKI') from error
                return {'status': 'pass', 'verify_code': error.verify_code, 'network_used': False}
        if completed == {'client', 'server'}:
            raise ValueError('Private Python accepted a distrusted intermediate key')
        for output, incoming in ((client_out, server_in), (server_out, client_in)):
            data = output.read()
            if data:
                incoming.write(data)
    raise ValueError('Private Python SSL regression did not complete within its bound')


def check_private_python(python_pin, inputs, rejected):
    expected = tuple(int(value) for value in python_pin['version'].split('.'))
    if (expected[:2] != (3, 13) or expected < (3, 13, 18) or sys.version_info[:3] != expected or
            Path(sys.prefix).resolve() != Path('/app') or
            Path(sys.executable).resolve() != Path('/app/bin/python3.13').resolve()):
        raise ValueError('The application is not using its exact patched private CPython')
    import _ssl
    import pip
    import pip._vendor.certifi
    origins = {'ssl': app_origin(ssl), '_ssl': private_ssl_origin(_ssl, Path(sys.executable)),
               'pip': app_origin(pip)}
    pip_pin, = inputs['bootstrap']
    if pip_pin['name'] != 'pip' or importlib.metadata.version('pip') != pip_pin['version']:
        raise ValueError('The application private pip differs from the reviewed bootstrap')
    launcher = Path('/app/bin/nvbroadcast').read_text().splitlines()[0]
    if launcher != '#!/app/bin/python3.13':
        raise ValueError('The installed application launcher does not select private Python')
    provenance = Path('/app/share/nvbroadcast-runtime-provenance')
    record = json.loads((provenance / 'flatpak-private-python.json').read_text())
    metadata = provenance / 'PYTHON.json'
    if record['python'] != python_pin or hashlib.sha256(metadata.read_bytes()).hexdigest() != record['python_metadata_sha256']:
        raise ValueError('Private Python provenance differs from its canonical input')
    if hashlib.sha256(Path('/app/bin/python3.13').read_bytes()).hexdigest() != record['python_binary_sha256']:
        raise ValueError('Private Python executable differs from its provenance')
    licenses = provenance / 'python-licenses'
    if not record['license_files'] or len(record['license_files']) > 256:
        raise ValueError('Private Python native license inventory is missing or oversized')
    for name, expected_hash in record['license_files'].items():
        path = (licenses / name).resolve()
        if (not path.is_relative_to(licenses) or not path.is_file() or path.stat().st_size > 1024 * 1024 or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash):
            raise ValueError('A retained interpreter/native license differs or is missing')
    if not (licenses / 'LICENSE.cpython.txt').is_file():
        raise ValueError('The primary CPython license is missing')
    fixture = inputs['ssl_fixture']
    if record['ssl_fixture'] != fixture:
        raise ValueError('Private Python SSL fixture provenance differs')
    denylist = check_spki_distrust(Path('/app') / fixture['path'], fixture['sha256'])
    pip_context = check_context(ssl.create_default_context(cafile=pip._vendor.certifi.where()), rejected)
    return {'interpreter': sys.executable, 'prefix': sys.prefix, 'version': sys.version,
            'soabi': sysconfig.get_config_var('SOABI'), 'module_origins': origins,
            'launcher_shebang': launcher, 'openssl': ssl.OPENSSL_VERSION,
            'native_license_count': len(record['license_files']),
            'distrusted_intermediate': denylist, 'pip_vendor_context': pip_context}


def check_shared_bindings():
    import gi
    gi.require_foreign('cairo')
    origins = {}
    for name in ('gi', 'gi._gi', 'gi._gi_cairo', 'cairo', 'cairo._cairo'):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to('/usr/lib/python3.13/site-packages'):
            raise ValueError('Shared CP313 GI/cairo binding origin differs from policy')
        origins[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for namespace, version in (('Gtk', '4.0'), ('Adw', '1'), ('Gst', '1.0'), ('GstApp', '1.0'), ('GstVideo', '1.0')):
        gi.require_version(namespace, version)
    from gi.repository import Adw, Gst, GstApp, GstVideo, Gtk
    Gst.init([])
    return {'origins': origins, 'gtk_major': Gtk.get_major_version(),
            'adwaita_major': Adw.get_major_version(), 'gstreamer': Gst.version_string(),
            'gst_app_sink': GstApp.AppSink.__name__, 'gst_video_info': GstVideo.VideoInfo.__name__}


def main():
    policy = json.loads(sys.argv[1])
    rejected = set(policy['rejected_certificate_sha256'])
    private = check_private_python(json.loads(sys.argv[2])['python'], json.loads(sys.argv[3]), rejected)
    bindings = check_shared_bindings()
    if startup_shim is None:
        raise ValueError('The application setuptools startup shim was not loaded')
    origins = {'_distutils_hack': app_origin(startup_shim)}
    # Import the application modules without opening a window or a physical device.
    for name in ('setuptools', 'certifi', 'requests', 'httpx', 'cv2', 'mediapipe',
                 'onnxruntime', 'pyrnnoise', 'faster_whisper', 'nvbroadcast.app',
                 'nvbroadcast.audio.pipeline', 'nvbroadcast.video.pipeline'):
        origins[name] = app_origin(importlib.import_module(name))
    if importlib.metadata.version('setuptools') != '84.1.0':
        raise ValueError('The reviewed application setuptools override is missing')
    if importlib.metadata.version('certifi') != '2026.10.19':
        raise ValueError('The reviewed application certifi bundle is missing')
    import distutils
    origins['distutils'] = app_origin(distutils)
    if any(name == 'mako' or name.startswith('mako.') for name in sys.modules):
        raise ValueError('Application imports unexpectedly loaded the unaudited runtime Mako')

    import certifi
    import requests.adapters
    import requests.certs
    from httpx._config import create_ssl_context

    bundle = Path(certifi.where()).resolve()
    if (not bundle.is_relative_to('/app/') or
            Path(requests.certs.where()).resolve() != bundle or
            Path(requests.adapters.DEFAULT_CA_BUNDLE_PATH).resolve() != bundle):
        raise ValueError('Requests does not use the application certifi bundle')
    defaults = ssl.get_default_verify_paths()
    capaths = check_capaths(defaults, rejected)
    if defaults.cafile is None:
        raise ValueError('The application default CA file is missing')
    default_ca = Path(defaults.cafile)
    platform = next(item for item in policy['runtimes'] if '/org.gnome.Platform/' in item['ref'])
    if hashlib.sha256(default_ca.read_bytes()).hexdigest() != platform['ca_sha256']:
        raise ValueError('The active application CA file differs from the pinned Platform')
    contexts = {
        'stdlib_ssl': ssl.create_default_context(),
        'stdlib_https': http.client.HTTPSConnection('localhost')._context,
        'httpx': create_ssl_context(),
        'requests_certifi': ssl.create_default_context(cafile=str(bundle)),
    }
    result = {'status': 'pass', 'python': sys.version, 'module_origins': origins,
              'private_python': private, 'shared_cp313_bindings': bindings,
              'default_verify_paths': defaults._asdict(),
              'default_capath_checks': capaths,
              'default_ca_sha256': platform['ca_sha256'],
              'certifi_ca_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
              'contexts': {name: check_context(context, rejected)
                           for name, context in contexts.items()},
              'mako_loaded_by_checked_imports': False,
              'limits': ['No network connection or physical device used',
                         'Shared runtime copies and unimported components are not a full OS audit']}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

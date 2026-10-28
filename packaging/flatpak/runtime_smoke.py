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


def app_origin(module):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to('/app/'):
        raise ValueError(f'{module.__name__} comes from the shared runtime: {path}')
    return str(path)


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


def main():
    policy = json.loads(sys.argv[1])
    rejected = set(policy['rejected_certificate_sha256'])
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

"""Qualify the finished Nix CPU runtime with generated data and no devices/network."""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile
import zipfile


def check_spki_distrust(context: ssl.SSLContext, fixture: Path, expected_hash: str) -> dict:
    """Reuse CPython's public compromised-intermediate MemoryBIO regression case."""
    if fixture.is_symlink() or fixture.stat().st_size > 65536:
        raise ValueError("Invalid TLS regression fixture")
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("TLS fixture hash mismatch")
    expected = {"distrusted/", "distrusted/distrusted_keycert.pem",
                "distrusted/distrusted_intermediate.pem", "distrusted/distrusted_root.pem"}
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(fixture) as archive:
        members = archive.infolist()
        if len(members) != 4 or {m.filename for m in members} != expected or sum(m.file_size for m in members) > 16384:
            raise ValueError("Unexpected TLS regression fixture members")
        root, keycert = Path(directory) / "root.pem", Path(directory) / "keycert.pem"
        # Fixed public upstream test-fixture password, never a user credential.
        password = b"Brave Sir Robin"
        root.write_bytes(archive.read("distrusted/distrusted_root.pem", pwd=password))
        keycert.write_bytes(archive.read("distrusted/distrusted_keycert.pem", pwd=password))
        context.load_verify_locations(cafile=str(root))
        server_context.load_cert_chain(str(keycert))
    client_in, client_out, server_in, server_out = (ssl.MemoryBIO() for _ in range(4))
    client = context.wrap_bio(client_in, client_out, server_hostname="localhost")
    server = server_context.wrap_bio(server_in, server_out, server_side=True)
    complete = set()
    for _ in range(32):
        for name, endpoint in (("client", client), ("server", server)):
            try:
                endpoint.do_handshake()
                complete.add(name)
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                pass
            except ssl.SSLCertVerificationError as error:
                if name != "client" or error.verify_code != 28:
                    raise ValueError("TLS fixture failed for a reason other than rejected SPKI") from error
                return {"verify_code": error.verify_code, "network_used": False}
        if complete == {"client", "server"}:
            raise ValueError("Runtime accepted the compromised intermediate")
        for output, incoming in ((client_out, server_in), (server_out, client_in)):
            data = output.read()
            if data:
                incoming.write(data)
    raise ValueError("TLS fixture exceeded its finite handshake bound")


def check_context(context: ssl.SSLContext, expected_roots: set[str], rejected_roots: list[str]) -> int:
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise ValueError("Runtime TLS context must require certificates and hostnames")
    observed = {hashlib.sha256(item).hexdigest() for item in context.get_ca_certs(binary_form=True)}
    if observed != expected_roots or observed.intersection(rejected_roots):
        raise ValueError("Unexpected or rejected runtime trust roots")
    return len(observed)


def requests_tls_context() -> ssl.SSLContext:
    """Capture Requests' configured context immediately before socket wrapping."""
    import requests
    from unittest.mock import patch
    from urllib3.util import ssl_ as urllib3_ssl

    class ContextCaptured(Exception):
        pass

    class NoNetworkTransport:
        def close(self):
            pass

    adapter = requests.adapters.HTTPAdapter()
    request = requests.Request("GET", "https://localhost/").prepare()
    pool = adapter.get_connection_with_tls_context(request, verify=True, proxies={})
    adapter.cert_verify(pool, request.url, verify=True, cert=None)
    connection = pool._new_conn()
    transport = NoNetworkTransport()
    captured = []

    def capture(sock, context, tls_in_tls, server_hostname):
        if sock is not transport or tls_in_tls or server_hostname != "localhost":
            raise ValueError("Unexpected Requests TLS transport path")
        captured.append(context)
        raise ContextCaptured

    try:
        # Preserve Requests' CA selection and urllib3's real context creation,
        # trust loading and TLS settings. Replace only TCP creation and the
        # final socket-wrapping boundary, before either can perform socket I/O.
        with patch.object(type(connection), "_new_conn", return_value=transport), \
                patch.object(urllib3_ssl, "_ssl_wrap_socket_impl", side_effect=capture):
            try:
                connection.connect()
            except ContextCaptured:
                pass
        if len(captured) != 1 or not isinstance(captured[0], ssl.SSLContext):
            raise ValueError("Cannot establish the actual Requests adapter TLS context")
        return captured[0]
    finally:
        connection.close()
        adapter.close()


def qualify(package: Path, source: Path, pins: dict, fixture: Path, model: Path, wheels: dict) -> dict:
    import av
    import av.option  # noqa: F401 - audiolab's required interface
    import certifi
    import cv2
    import numpy as np
    import nvbroadcast
    from packaging.version import Version
    from pyrnnoise import rnnoise
    from nvbroadcast.core import dependency_installer, resources
    from nvbroadcast.runtime.artifact import ArtifactEnvironment
    from nvbroadcast.runtime.probe import ProbeProvider, probe_execution_provider
    from nvbroadcast.runtime.variants import RuntimeVariant, validate_current_runtime
    from nvbroadcast.video.face_landmarks import _probe_mediapipe_runtime

    if sys.version_info[:3] != (3, 13, 18):
        raise ValueError("Finished runtime must use CPython 3.13.18")
    if not Path(nvbroadcast.__file__).resolve().is_relative_to(package):
        raise ValueError("Application was imported outside the finished package")
    environment = ArtifactEnvironment.current()
    problems = environment.dependency_closure_problems(
        roots=("nvbroadcast",), root_extras={"nvbroadcast": ("cpu",)}
    ) + validate_current_runtime(RuntimeVariant.CPU)
    if problems:
        raise ValueError("Finished CPU dependency closure failed: " + "; ".join(problems))
    if any(environment.installed.get(name) for name in ("pytest", "jsonschema")):
        raise ValueError("Development check tools leaked into the finished runtime")
    installed = {}
    imported = {}
    module_names = {"pillow": "PIL", "certifi": "certifi", "filelock": "filelock",
                    "urllib3": "urllib3", "setuptools": "setuptools"}
    for name, pin in pins["packages"].items():
        versions = environment.installed.get(name, ())
        if len(versions) != 1 or Version(versions[0]) != Version(pin["version"]):
            raise ValueError(f"Expected one exact runtime owner for {name}: {versions}")
        installed[name] = versions[0]
        module = importlib.import_module(module_names[name])
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path("/nix/store")):
            raise ValueError("Runtime module came from outside the store: " + name)
        expected_module = metadata.distribution(name).locate_file(module_names[name] + "/__init__.py")
        if path != Path(expected_module).resolve():
            raise ValueError("Imported module differs from its selected distribution: " + name)
        imported[name] = {"version": str(module.__version__), "path": str(path)}
        if Version(imported[name]["version"]) != Version(pin["version"]):
            raise ValueError("Imported module version differs from metadata: " + name)
    if not hasattr(cv2, "face") or not metadata.version("opencv-contrib-python"):
        raise ValueError("The declared OpenCV contrib provider is absent")
    cv2.cvtColor(np.zeros((4, 4, 3), dtype=np.uint8), cv2.COLOR_BGR2RGB)
    if av.Codec("pcm_s16le", "w").canonical_name != "pcm_s16le":
        raise ValueError("AV canonical codec interface failed")
    state = rnnoise.create()
    if not state:
        raise ValueError("RNNoise did not initialize native state")
    try:
        frame, probability = rnnoise.process_mono_frame(state, np.zeros(rnnoise.FRAME_SIZE, dtype=np.int16))
        if frame.shape != (480,) or not np.isfinite(frame).all() or not np.isfinite(probability):
            raise ValueError("RNNoise failed native frame processing")
    finally:
        rnnoise.destroy(state)
    cpu = probe_execution_provider(ProbeProvider.CPU, use_cache=False)
    if not cpu.success or cpu.executed_providers != ("CPUExecutionProvider",) or cpu.output != (1.0, 4.0, 9.0, 16.0):
        raise ValueError("Actual isolated CPU execution failed: " + str(cpu.to_payload()))
    isolated_environment = {k: v for k, v in os.environ.items()
                            if k not in {"PYTHONPATH", "PYTHONHOME", "NIX_PYTHONPATH"}}
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import json, mediapipe; print(json.dumps({'version':mediapipe.__version__, 'path':mediapipe.__file__}))"],
        env=isolated_environment, capture_output=True, text=True, timeout=20, check=True,
    )
    isolated = json.loads(result.stdout)
    import mediapipe
    if (Version(isolated["version"]) != Version("1.0.1") or
            isolated["version"] != mediapipe.__version__ or
            Path(isolated["path"]).resolve() != Path(mediapipe.__file__).resolve()):
        raise ValueError("Isolated MediaPipe differs from the selected parent runtime")
    if hashlib.sha256(model.read_bytes()).hexdigest() != pins["faceModel"]["sha256"]:
        raise ValueError("MediaPipe model hash mismatch")
    ok, detail = _probe_mediapipe_runtime(model)
    if not ok:
        raise ValueError("Actual application's isolated face-model probe failed: " + detail)
    if dependency_installer._runtime_install_block_reason() is None:
        raise ValueError("The Nix runtime unexpectedly permits dependency mutation")
    for resource in (resources.find_app_icon(), resources.find_app_icon_png(), resources.find_backgrounds_dir()):
        if not resource.exists():
            raise ValueError("Installed application resource is missing")
    source_hashes = {}
    app_root = Path(nvbroadcast.__file__).resolve().parent
    for path in sorted((source / "src/nvbroadcast").rglob("*.py")):
        relative = path.relative_to(source / "src/nvbroadcast")
        expected = path.read_bytes()
        if relative.as_posix() == "core/resources.py":
            expected = expected.replace(b'Path(sys.prefix) / "share"', f'Path("{package}") / "share"'.encode())
        actual = (app_root / relative).read_bytes()
        if actual != expected:
            raise ValueError("Installed application source mismatch: " + relative.as_posix())
        source_hashes[relative.as_posix()] = hashlib.sha256(actual).hexdigest()
    notices = {}
    for name in ("LICENSE", "NOTICE", "CONTRIBUTORS.md"):
        actual = (package / "share/doc/nvbroadcast" / name).read_bytes()
        if actual != (source / name).read_bytes():
            raise ValueError("Installed application notice mismatch: " + name)
        notices[name] = hashlib.sha256(actual).hexdigest()
    supplier_notices = {}
    for name, wheel in wheels.items():
        entries = []
        distribution = metadata.distribution(name)
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            if len(members) > 20000:
                raise ValueError("Supplier wheel member bound exceeded")
            for member in members:
                path = Path(member.filename)
                if member.is_dir() or not ("licenses" in path.parts or
                        re.match(r"(?:license|copying|notice)(?:[._-].*)?$", path.name, re.I)):
                    continue
                if path.is_absolute() or ".." in path.parts or member.file_size > 1024**2:
                    raise ValueError("Unexpected supplier notice member")
                expected = archive.read(member)
                installed_notice = Path(distribution.locate_file(member.filename)).resolve()
                if not installed_notice.is_relative_to(Path("/nix/store")) or installed_notice.read_bytes() != expected:
                    raise ValueError("Supplier notice lost/changed during fixup: " + name + ":" + member.filename)
                entries.append({"path": member.filename, "sha256": hashlib.sha256(expected).hexdigest()})
        supplier_notices[name] = entries

    bundle = Path(certifi.where())
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != pins["certifiBundleSha256"]:
        raise ValueError("Default certifi bundle differs from the exact upstream bundle")
    expected_roots = {hashlib.sha256(ssl.PEM_cert_to_DER_cert(x.decode())).hexdigest()
                      for x in re.findall(rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", bundle.read_bytes(), re.S)}
    default_context = ssl.create_default_context()
    requests_context = requests_tls_context()
    contexts = {"stdlib-default": default_context, "requests-adapter": requests_context}
    root_counts = {}
    for name, context in contexts.items():
        root_counts[name] = check_context(context, expected_roots, pins["rejectedRootDerSha256"])
    contexts["explicit-fixture-root"] = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    tls = {name: check_spki_distrust(context, fixture, pins["tlsFixture"]["sha256"])
           for name, context in contexts.items()}
    return {
        "scope": "Finished Nix CPU interpreter and application dependency paths; generated native/inference/TLS checks only, no physical devices or public release qualification",
        "python": sys.version, "interpreter": sys.executable, "openssl": ssl.OPENSSL_VERSION,
        "package": str(package), "installed": installed, "installed_environment": environment.installed,
        "imported": imported, "isolated_mediapipe": isolated,
        "cpu": cpu.to_payload(), "av": av.__version__, "opencv": cv2.__version__,
        "application_source_sha256": source_hashes, "certifi_bundle_sha256": pins["certifiBundleSha256"],
        "default_verify_paths": ssl.get_default_verify_paths()._asdict(),
        "trust_environment": {name: os.environ.get(name) for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "NIX_SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")},
        "trust_roots": root_counts, "tls": tls, "application_notices": notices,
        "supplier_notices": supplier_notices,
    }


if __name__ == "__main__":
    package, source, inputs, fixture, model, output = map(Path, sys.argv[1:7])
    report = qualify(package.resolve(), source, json.loads(inputs.read_text()), fixture, model, json.loads(sys.argv[7]))
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")

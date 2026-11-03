#!/usr/bin/env python3
"""Export and verify an unsigned Flatpak test bundle without installing it."""

import argparse
import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib


APP_ID = "com.nvbroadcast.NVBroadcast"
APP_REF = f"app/{APP_ID}/x86_64/master"
BUNDLE_NAME = "nvbroadcast-development-cpu-x86_64.flatpak"
FLATPAK_METADATA = {
    "share/applications/com.nvbroadcast.NVBroadcast.desktop":
        "packaging/flatpak/com.nvbroadcast.NVBroadcast.desktop",
    "share/metainfo/com.nvbroadcast.NVBroadcast.metainfo.xml":
        "packaging/flatpak/com.nvbroadcast.NVBroadcast.metainfo.xml",
    "share/icons/hicolor/scalable/apps/com.nvbroadcast.NVBroadcast.svg":
        "packaging/flatpak/com.nvbroadcast.NVBroadcast.svg",
    "share/icons/hicolor/128x128/apps/com.nvbroadcast.NVBroadcast.png":
        "packaging/flatpak/com.nvbroadcast.NVBroadcast.png",
}
BUILD_INPUTS = (
    "pyproject.toml", "packaging/flatpak/com.nvbroadcast.NVBroadcast.yml",
    "packaging/flatpak/python3-flatpak-requirements.yaml",
    "packaging/flatpak/requirements.txt",
    "packaging/flatpak/runtime-lock.json",
    "packaging/flatpak/runtime_policy.py",
    "packaging/flatpak/runtime_smoke.py",
    "packaging/flatpak/inputs.json",
    "packaging/flatpak/private_python.py",
    "packaging/flatpak/python3-private-runtime.yml",
    "packaging/runtime-prototype/inputs.json",
    "packaging/runtime-prototype/prepare.py",
    "packaging/runtime-prototype/pylock.build.toml",
    "packaging/flatpak/export_development_bundle.py",
    ".github/workflows/flatpak.yml",
)

_SPEC = importlib.util.spec_from_file_location(
    "flatpak_runtime_policy", Path(__file__).with_name("runtime_policy.py")
)
runtime_policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runtime_policy)
_PRIVATE_SPEC = importlib.util.spec_from_file_location(
    "flatpak_private_python", Path(__file__).with_name("private_python.py")
)
private_python = importlib.util.module_from_spec(_PRIVATE_SPEC)
_PRIVATE_SPEC.loader.exec_module(private_python)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE).stdout


def commit(repo):
    value = command("ostree", f"--repo={repo}", "rev-parse", APP_REF)
    value = value.decode().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid application OSTree commit")
    return value


def source_payloads(source, build):
    """Map project-owned installed files to the checkout that produced them."""
    project = tomllib.loads((source / "pyproject.toml").read_text())
    packages = list((build / "files/lib").glob("python*/site-packages"))
    if len(packages) != 1:
        raise ValueError("Expected one installed Python site-packages directory")
    site = packages[0].relative_to(build / "files")
    pairs = {}
    for path in sorted((source / "src/nvbroadcast").rglob("*.py")):
        pairs[(site / path.relative_to(source / "src")).as_posix()] = path
    for package, patterns in project["tool"]["setuptools"]["package-data"].items():
        directory = source / "src" / package.replace(".", "/")
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if not matches:
                raise ValueError(f"Missing package data: {package}/{pattern}")
            for path in matches:
                pairs[(site / path.relative_to(source / "src")).as_posix()] = path
    for directory, paths in project["tool"]["setuptools"]["data-files"].items():
        for name in paths:
            path = source / name
            if path.name.startswith("com.doczeus.NVBroadcast."):
                continue  # Native identity is replaced by Flatpak-specific metadata.
            pairs[(Path(directory) / path.name).as_posix()] = path
    for target, name in FLATPAK_METADATA.items():
        pairs[target] = source / name
    licenses = list(packages[0].glob("nvbroadcast-*.dist-info/**/LICENSE"))
    if len(licenses) != 1:
        raise ValueError("Expected the complete project LICENSE in the package")
    pairs[licenses[0].relative_to(build / "files").as_posix()] = source / "LICENSE"
    if not any(name.endswith("nvbroadcast/__main__.py") for name in pairs):
        raise ValueError("Application source is missing")
    return pairs


def verify_payloads(repo, application_commit, pairs):
    records = []
    for name, source in sorted(pairs.items()):
        received = command(
            "ostree", f"--repo={repo}", "cat", application_commit, f"files/{name}"
        )
        if received != source.read_bytes():
            raise ValueError(f"Imported bundle does not match source: {name}")
        records.append({"path": name, "sha256": sha256(source)})
    return records


def private_python_payloads(prefix, record):
    provenance = Path('share/nvbroadcast-runtime-provenance')
    names = [Path('bin/python3.13'), Path('lib/python3.13/ssl.py'),
             Path(record['ssl_fixture']['path']),
             *(provenance / name for name in ('flatpak-private-python.json', 'PYTHON.json',
                                               'flatpak-inputs.json', 'python-inputs.json', 'pylock.build.toml')),
             *(provenance / 'python-licenses' / name for name in record['license_files'])]
    return {name.as_posix(): prefix / name for name in names}


def verify_git_sources(source, source_revision, paths):
    """Reject staged, unstaged, or untracked shipping inputs at the named SHA."""
    source = source.resolve()
    for path in sorted(set(paths)):
        path = path.absolute()
        name = path.relative_to(source).as_posix()
        if any(
            parent.is_symlink() for parent in (path, *path.parents)
            if parent == source or source in parent.parents
        ):
            raise ValueError(f"Symlinks are not supported in shipping inputs: {name}")
        committed = subprocess.run(
            ("git", "-c", f"safe.directory={source}", "-C", str(source),
             "show", f"{source_revision}:{name}"),
            check=True, stdout=subprocess.PIPE,
        ).stdout
        if path.read_bytes() != committed:
            raise ValueError(f"Source differs from named Git revision: {name}")


def export_bundle(source, build, repo, output, source_revision, builder_image,
                  prepared_builder_image):
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Expected the full checked-out Git source revision")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", builder_image):
        raise ValueError("Expected a digest-pinned builder image")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", prepared_builder_image):
        raise ValueError("Expected an immutable prepared builder image ID")
    if output.exists():
        raise ValueError("Output directory already exists; use a fresh destination")
    metadata = configparser.ConfigParser()
    metadata.optionxform = str
    metadata.read(build / "metadata")
    if metadata["Application"]["name"] != APP_ID:
        raise ValueError("Unexpected application identity")
    runtime_ref = metadata["Application"]["runtime"]
    if runtime_ref != "org.gnome.Platform/x86_64/50":
        raise ValueError("Unexpected development runtime")
    if metadata.get("Environment", "PYTHONPATH", fallback="") != "/app/lib/python3.13/site-packages:/usr/lib/python3.13/site-packages":
        raise ValueError("Application metadata must select its Python packages before startup")
    for key, value in {'SSL_CERT_FILE': private_python.CA_FILE,
                       'SSL_CERT_DIR': private_python.CA_DIRECTORY, 'PYTHONNOUSERSITE': '1'}.items():
        if metadata.get('Environment', key, fallback='') != value:
            raise ValueError('Application metadata must retain its explicit private Python trust environment')
    application_commit = commit(repo)
    pairs = source_payloads(source, build)
    verify_git_sources(
        source, source_revision,
        [*pairs.values(), *(source / name for name in BUILD_INPUTS)],
    )
    # Re-read active deployments, signatures and CA bytes inside this image.
    runtime_inputs = runtime_policy.inspect_runtime(
        source / "packaging/flatpak/runtime-lock.json"
    )
    if runtime_inputs["base_image"] != builder_image:
        raise ValueError("Builder base differs from the authenticated runtime lock")
    preparation, preparation_sha256 = runtime_policy.verify_preparation_proof(
        source / "dist/flatpak-runtime/prepared-image.json", builder_image,
        prepared_builder_image, runtime_inputs,
    )
    private_runtime, private_runtime_sha256 = private_python.verify_prefix_provenance(build / "files", source)
    private_payloads = private_python_payloads(build / 'files', private_runtime)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed import or payload comparison must not leave a usable artifact.
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        artifacts = temporary / "artifacts"
        artifacts.mkdir()
        bundle = artifacts / BUNDLE_NAME
        command(
            "flatpak", "build-bundle", "--arch=x86_64", str(repo), str(bundle),
            APP_ID, "master",
        )
        imported = temporary / "imported-repo"
        command("ostree", f"--repo={imported}", "init", "--mode=archive-z2")
        command(
            "flatpak", "build-import-bundle", "--no-update-summary",
            str(imported), str(bundle),
        )
        if commit(imported) != application_commit:
            raise ValueError("Imported application commit differs from the build")
        command("ostree", f"--repo={imported}", "fsck")
        records = verify_payloads(imported, application_commit, pairs)
        private_records = verify_payloads(imported, application_commit, private_payloads)
        received_metadata = command(
            "ostree", f"--repo={imported}", "cat", application_commit, "metadata"
        )
        if received_metadata != (build / "metadata").read_bytes():
            raise ValueError("Imported sandbox metadata differs from the build")
        evidence = {
            "schema_version": 1,
            "source_revision": source_revision,
            "builder_image": builder_image,
            "prepared_builder_image": prepared_builder_image,
            "builder_preparation": preparation,
            "builder_preparation_sha256": preparation_sha256,
            "runtime_inputs": runtime_inputs,
            "private_python": private_runtime,
            "private_python_sha256": private_runtime_sha256,
            "application_ref": APP_REF,
            "application_commit": application_commit,
            "runtime_ref": runtime_ref,
            "bundle": {"name": BUNDLE_NAME, "bytes": bundle.stat().st_size,
                       "sha256": sha256(bundle)},
            "signature": "unsigned development bundle",
            "public_distribution_qualified": False,
            "host_installation_performed": False,
            "verification": {"imported_commit_matches": True,
                             "ostree_fsck": "passed",
                             "sandbox_metadata_matches": True,
                             "source_payloads": records,
                             "private_python_payloads": private_records},
            "build_inputs": {
                name: sha256(source / name) for name in BUILD_INPUTS
            },
        }
        provenance = artifacts / "bundle-provenance.json"
        provenance.write_text(json.dumps(evidence, indent=2) + "\n")
        (artifacts / "SHA256SUMS").write_text(
            f"{sha256(bundle)}  {BUNDLE_NAME}\n"
            f"{sha256(provenance)}  {provenance.name}\n"
        )
        artifacts.rename(output)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--build", type=Path, default=Path("flatpak-build"))
    parser.add_argument("--repo", type=Path, default=Path("flatpak-repo"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--builder-image", required=True)
    parser.add_argument("--prepared-builder-image", required=True)
    args = parser.parse_args()
    evidence = export_bundle(
        args.source, args.build, args.repo, args.output,
        args.source_revision, args.builder_image, args.prepared_builder_image,
    )
    print(json.dumps(evidence["bundle"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export and verify an unsigned Flatpak test bundle without installing it."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib


APP_ID = "com.doczeus.NVBroadcast"
APP_REF = f"app/{APP_ID}/x86_64/master"
BUNDLE_NAME = "nvbroadcast-development-cpu-x86_64.flatpak"
BUILD_INPUTS = (
    "pyproject.toml", "packaging/flatpak/com.doczeus.NVBroadcast.yml",
    "packaging/flatpak/python3-flatpak-requirements.yaml",
    "packaging/flatpak/requirements.txt",
)


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
            pairs[(Path(directory) / path.name).as_posix()] = path
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


def export_bundle(source, build, repo, output, source_revision, builder_image):
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Expected the full checked-out Git source revision")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", builder_image):
        raise ValueError("Expected a digest-pinned builder image")
    if output.exists():
        raise ValueError("Output directory already exists; use a fresh destination")
    metadata = configparser.ConfigParser()
    metadata.read(build / "metadata")
    if metadata["Application"]["name"] != APP_ID:
        raise ValueError("Unexpected application identity")
    runtime_ref = metadata["Application"]["runtime"]
    if runtime_ref != "org.gnome.Platform/x86_64/50":
        raise ValueError("Unexpected development runtime")
    application_commit = commit(repo)
    pairs = source_payloads(source, build)
    verify_git_sources(
        source, source_revision,
        [*pairs.values(), *(source / name for name in BUILD_INPUTS)],
    )
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
        received_metadata = command(
            "ostree", f"--repo={imported}", "cat", application_commit, "metadata"
        )
        if received_metadata != (build / "metadata").read_bytes():
            raise ValueError("Imported sandbox metadata differs from the build")
        evidence = {
            "schema_version": 1,
            "source_revision": source_revision,
            "builder_image": builder_image,
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
                             "source_payloads": records},
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
    args = parser.parse_args()
    evidence = export_bundle(
        args.source, args.build, args.repo, args.output,
        args.source_revision, args.builder_image,
    )
    print(json.dumps(evidence["bundle"]))


if __name__ == "__main__":
    main()

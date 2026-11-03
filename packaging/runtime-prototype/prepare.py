#!/usr/bin/env python3
"""Fetch verified Stage 0 inputs and unpack a private interpreter for probing.

Requires Python 3.12+ and zstd. The committed pins are the trust input; this is
not a signed runtime downloader or an application installer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filename(item: dict) -> str:
    url = urllib.parse.urlsplit(item["url"])
    name = urllib.parse.unquote(url.path.rsplit("/", 1)[-1])
    if url.scheme != "https" or not name or name in {".", ".."} or "/" in name:
        raise ValueError("input must have an HTTPS URL and a simple filename")
    return name


def verify(path: Path, item: dict) -> None:
    expected = item["sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("invalid SHA-256 pin")
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def fetch(directory: Path, item: dict, *, offline: bool = False) -> Path:
    path = directory / filename(item)
    if path.exists():
        verify(path, item)
        return path
    if offline:
        raise FileNotFoundError(f"missing verified offline input: {path}")
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                shutil.copyfileobj(response, stream)
        verify(temporary, item)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def safe_member(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {member.name}")
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise ValueError(f"unsupported archive entry: {member.name}")
    # data_filter also verifies link targets against the extraction root.
    return tarfile.data_filter(member, destination)


def unpack_python(archive: Path, item: dict, output: Path) -> dict:
    verify(archive, item)
    output.mkdir(parents=True, exist_ok=False)
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                if (member.name == "python/PYTHON.json"
                        or member.name.startswith("python/install/")
                        or member.name.startswith("python/licenses/")):
                    stream.extract(member, path=output, filter=safe_member)
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError("zstd could not decode the verified Python archive")
    except BaseException:
        process.kill()
        process.wait()
        raise

    metadata_path = output / "python/PYTHON.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata["python_version"] != item["version"]
            or metadata["target_triple"] != "x86_64-unknown-linux-gnu"
            or metadata["python_tag"] != "cp313"):
        raise ValueError("Python archive metadata does not match the target")

    # Keep all upstream license files, including notices for libraries whose
    # optional modules will not be exercised by this limited prototype.
    licenses = output / "python/licenses"
    primary_license = (output / "python" / metadata["license_path"]).resolve()
    if (not licenses.is_dir() or not primary_license.is_relative_to(licenses.resolve())
            or not primary_license.is_file()):
        raise ValueError("Python archive is missing its license records")
    license_records = {
        str(p.relative_to(licenses)): digest(p)
        for p in sorted(licenses.rglob("*")) if p.is_file()
    }
    if not license_records:
        raise ValueError("Python archive contains no license files")
    runtime = output / "runtime"
    (output / "python/install").rename(runtime)
    provenance = runtime / "share/nvbroadcast-runtime-provenance"
    provenance.mkdir(parents=True)
    metadata_path.rename(provenance / "PYTHON.json")
    licenses.rename(provenance / "python-licenses")
    (output / "python").rmdir()
    interpreter = runtime / "bin/python"
    if not interpreter.exists():
        interpreter.symlink_to("python3.13")
    record = {"archive": item, "python_metadata_sha256": digest(provenance / "PYTHON.json"),
              "license_files": license_records}
    (provenance / "input.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def unpack_uv(archive: Path, item: dict, output: Path) -> None:
    verify(archive, item)
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(path=output, filter=safe_member)


def fetch_lock(lock_path: Path, directory: Path, local_wheels: Path | None = None,
               *, offline: bool = False) -> Path:
    """Materialize only hashed wheels; never resolve dependencies or build sdists."""
    lock = tomllib.loads(lock_path.read_text())
    if lock["lock-version"] != "1.0":
        raise ValueError("unsupported lock format")
    wheels = directory / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    requirements = []
    for package in lock["packages"]:
        candidates = package.get("wheels", [])
        if package.get("archive"):
            archive = package["archive"]
            if not archive.get("path", "").endswith(".whl"):
                raise ValueError("only local wheel archives are accepted")
            candidates = [archive]
        if not candidates:
            raise ValueError(f"no wheel for {package['name']}")
        hashes = []
        for wheel in candidates:
            expected = wheel["hashes"]["sha256"]
            if "url" in wheel:
                item = {"url": wheel["url"], "sha256": expected}
                if not filename(item).endswith(".whl"):
                    raise ValueError("only wheel URLs are accepted")
                fetch(wheels, item, offline=offline)
            elif "path" in wheel:
                local = ((local_wheels / Path(wheel["path"]).name) if local_wheels
                         else (lock_path.parent / wheel["path"]))
                local = local.resolve(strict=True)
                verify(local, {"sha256": expected})
                target = wheels / local.name
                if not target.exists():
                    shutil.copyfile(local, target)
                verify(target, {"sha256": expected})
            else:
                raise ValueError("wheel has no URL or local path")
            hashes.append(f"--hash=sha256:{expected}")
        requirement = f"{package['name']}=={package['version']}"
        if package.get("marker"):
            requirement += f" ; {package['marker']}"
        requirements.append(requirement + " " + " ".join(hashes))
    destination = directory / "requirements.txt"
    destination.write_text("\n".join(requirements) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "fetch-lock", "unpack"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--local-wheels", type=Path,
                        help="directory of rebuilt local wheels (hash pins still enforced)")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    os.umask(0o022)
    pins = json.loads((HERE / "inputs.json").read_text())
    inputs = args.directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch":
        for item in [pins["python"], pins["uv"], *pins["bindings"]]:
            print(fetch(inputs, item, offline=args.offline))
    elif args.command == "fetch-lock":
        if args.lock is None:
            parser.error("fetch-lock requires --lock")
        print(fetch_lock(args.lock, args.directory, args.local_wheels, offline=args.offline))
    else:
        record = unpack_python(inputs / filename(pins["python"]), pins["python"], args.directory / "python")
        unpack_uv(inputs / filename(pins["uv"]), pins["uv"], args.directory / "tools")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

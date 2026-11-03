#!/usr/bin/env python3
"""Rebuild faster-whisper model pins from authoritative, bounded upstream reads.

By default, preserve revisions already recorded in the manifest. Pass
--refresh-revisions to select the current upstream commits for review. Model
weights are never downloaded: their SHA-256 values come from Hugging Face LFS
metadata. Small files are fetched at the recorded commit and hashed locally.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


UPSTREAM = "faster-whisper==1.2.1"
UPSTREAM_COMMIT = "65882eee9f5cdbeeb2d877f1131d48cf241b327d"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/"
    f"{UPSTREAM_COMMIT}/faster_whisper/utils.py"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src/nvbroadcast/ai/whisper-models.json"
)
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_SMALL_FILE_BYTES = 8 * 1024 * 1024
REQUIRED_FILES = {"config.json", "model.bin", "tokenizer.json"}


def read_url(url: str, limit: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "nvbroadcast-model-manifest/1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Response exceeded {limit} bytes: {url}")
    return data


def upstream_aliases() -> dict[str, str]:
    source = ast.parse(read_url(UPSTREAM_URL, MAX_METADATA_BYTES))
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_MODELS"
            for target in node.targets
        ):
            aliases = ast.literal_eval(node.value)
            if (
                not isinstance(aliases, dict)
                or len(aliases) != 19
                or len(set(aliases.values())) != 17
                or not all(
                    isinstance(alias, str) and isinstance(repo, str)
                    for alias, repo in aliases.items()
                )
            ):
                raise ValueError("Unexpected faster-whisper 1.2.1 alias mapping")
            return aliases
    raise ValueError("Upstream _MODELS mapping was not found")


def model_manifest(
    repo_id: str, aliases: list[str], revision: str | None
) -> tuple[str, dict]:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    selected_revision = revision or "main"
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Invalid existing revision for {repo_id}")
    metadata_url = (
        f"https://huggingface.co/api/models/{encoded_repo}"
        f"/revision/{selected_revision}?blobs=true"
    )
    metadata = json.loads(read_url(metadata_url, MAX_METADATA_BYTES))
    commit = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Invalid upstream revision for {repo_id}")
    if revision is not None and commit != revision:
        raise ValueError(f"Revision mismatch for {repo_id}")

    files = {}
    for entry in sorted(metadata["siblings"], key=lambda item: item["rfilename"]):
        filename = entry["rfilename"]
        if filename not in REQUIRED_FILES | {"preprocessor_config.json"} and not (
            filename.startswith("vocabulary.") and "/" not in filename
        ):
            continue
        size = entry["size"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid file size for {repo_id}/{filename}")
        lfs = entry.get("lfs")
        if filename == "model.bin":
            if not lfs or not re.fullmatch(r"[0-9a-f]{64}", lfs["sha256"]):
                raise ValueError(f"Missing LFS SHA-256 for {repo_id}/{filename}")
            digest = lfs["sha256"]
        else:
            if size > MAX_SMALL_FILE_BYTES:
                raise ValueError(f"Small-file limit exceeded: {repo_id}/{filename}")
            url = (
                f"https://huggingface.co/{encoded_repo}/resolve/{commit}/"
                f"{urllib.parse.quote(filename, safe='')}"
            )
            data = read_url(url, size)
            if len(data) != size:
                raise ValueError(f"Size mismatch for {repo_id}/{filename}")
            digest = hashlib.sha256(data).hexdigest()
            if lfs:
                if digest != lfs["sha256"]:
                    raise ValueError(f"LFS hash mismatch for {repo_id}/{filename}")
            else:
                git_blob = f"blob {size}\0".encode() + data
                blob_id = hashlib.sha1(git_blob, usedforsecurity=False).hexdigest()
                if blob_id != entry["blobId"]:
                    raise ValueError(f"Git blob mismatch for {repo_id}/{filename}")
        if lfs and lfs["size"] != size:
            raise ValueError(f"LFS size mismatch for {repo_id}/{filename}")
        files[filename] = {"size": size, "sha256": digest}

    missing = REQUIRED_FILES - files.keys()
    if missing or not any(name.startswith("vocabulary.") for name in files):
        raise ValueError(f"Missing required model files for {repo_id}: {missing}")
    return repo_id, {
        "revision": commit,
        "aliases": sorted(aliases),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-revisions", action="store_true")
    parser.add_argument("--check", action="store_true", help="Compare without writing")
    args = parser.parse_args()
    existing = {}
    if args.output.exists() and not args.refresh_revisions:
        existing = json.loads(args.output.read_text(encoding="utf-8"))["models"]

    repositories: dict[str, list[str]] = {}
    for alias, repo_id in upstream_aliases().items():
        repositories.setdefault(repo_id, []).append(alias)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                model_manifest,
                repo_id,
                aliases,
                existing.get(repo_id, {}).get("revision"),
            )
            for repo_id, aliases in sorted(repositories.items())
        ]
        models = dict(future.result() for future in futures)

    manifest = {"schema_version": 1, "upstream": UPSTREAM, "models": models}
    text = json.dumps(manifest, indent=2) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != text:
            raise SystemExit("Model manifest differs from upstream evidence")
    else:
        args.output.write_text(text, encoding="utf-8")
    total_files = sum(len(model["files"]) for model in models.values())
    print(f"Verified {len(models)} repositories, 19 aliases, and {total_files} files")
    print(f"Alias source: {UPSTREAM_URL}")


if __name__ == "__main__":
    main()

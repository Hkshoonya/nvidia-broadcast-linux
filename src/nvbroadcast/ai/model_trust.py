"""Pin and verify faster-whisper snapshots before passing them to CTranslate2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Callable


_MANIFEST = Path(__file__).with_name("whisper-models.json")
_REQUIRED_FILES = {"config.json", "model.bin", "tokenizer.json"}
_MODEL_FILES = _REQUIRED_FILES | {
    "preprocessor_config.json",
    "vocabulary.json",
    "vocabulary.txt",
}


class ModelTrustError(RuntimeError):
    """The requested model is not pinned or its cached files are untrusted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_entry(model_name: str, installed_version: str) -> tuple[str, dict]:
    try:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelTrustError("faster-whisper model manifest is unavailable or invalid") from exc

    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("upstream") != f"faster-whisper=={installed_version}"
        or not isinstance(manifest.get("models"), dict)
    ):
        raise ModelTrustError("faster-whisper version does not match the model manifest")

    selected = None
    for repo_id, entry in manifest["models"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("aliases"), list):
            raise ModelTrustError("faster-whisper model manifest has invalid aliases")
        if model_name == repo_id or model_name in entry["aliases"]:
            if selected is not None:
                raise ModelTrustError(f"Ambiguous faster-whisper model: {model_name}")
            selected = (repo_id, entry)

    if selected is None:
        raise ModelTrustError(
            f"Unpinned faster-whisper model {model_name!r}; choose a bundled model alias"
        )

    repo_id, entry = selected
    revision = entry.get("revision")
    files = entry.get("files")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ModelTrustError(f"Invalid faster-whisper revision for {repo_id}")
    if not isinstance(files, dict) or not _REQUIRED_FILES <= files.keys() or not any(
        name.startswith("vocabulary.") for name in files
    ):
        raise ModelTrustError(f"Incomplete faster-whisper model manifest for {repo_id}")
    for name, expected in files.items():
        if (
            name not in _MODEL_FILES
            or not isinstance(expected, dict)
            or type(expected.get("size")) is not int
            or expected["size"] <= 0
            or not isinstance(expected.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
        ):
            raise ModelTrustError(f"Invalid faster-whisper file pin for {repo_id}/{name}")
    return repo_id, entry


def verified_model_path(
    model_name: str,
    installed_version: str,
    download_model: Callable[..., str],
) -> Path:
    """Verify a pinned Hub snapshot, or use an explicitly named local directory."""
    # An absolute path or a ./../ path is an explicit local model selection.
    # This preserves user-provided CTranslate2 directories without treating an
    # unknown Hub ID as permission to download arbitrary remote weights.
    if Path(model_name).is_absolute() or model_name.startswith(("./", "../")):
        local_path = Path(model_name)
        if not local_path.is_dir():
            raise ModelTrustError(f"Local faster-whisper model directory not found: {model_name}")
        return local_path

    repo_id, entry = _model_entry(model_name, installed_version)
    try:
        snapshot = Path(
            download_model(repo_id, revision=entry["revision"], use_auth_token=False)
        )
    except Exception as exc:
        raise ModelTrustError(
            f"Cannot resolve pinned faster-whisper revision for {repo_id}"
        ) from exc
    if not snapshot.is_dir():
        raise ModelTrustError(f"Missing faster-whisper snapshot for {repo_id}")

    files = entry["files"]
    try:
        for name, expected in files.items():
            path = snapshot / name
            if not path.is_file() or path.stat().st_size != expected["size"]:
                raise ModelTrustError(
                    f"Missing or wrong-size faster-whisper file: {repo_id}/{name}"
                )
            if _sha256_file(path) != expected["sha256"]:
                raise ModelTrustError(f"SHA-256 mismatch for faster-whisper file: {repo_id}/{name}")

        # The upstream downloader also permits optional preprocessor files and
        # vocabulary.*. Reject any model input absent from the manifest.
        for path in snapshot.iterdir():
            if (
                path.name in _MODEL_FILES or path.name.startswith("vocabulary.")
            ) and path.name not in files:
                raise ModelTrustError(f"Unpinned faster-whisper file: {repo_id}/{path.name}")
    except OSError as exc:
        raise ModelTrustError(f"Cannot verify faster-whisper snapshot for {repo_id}") from exc
    return snapshot

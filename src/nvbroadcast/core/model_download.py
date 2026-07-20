"""Verified model downloads into a writable per-user cache."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

from nvbroadcast.core.resources import PROJECT_ROOT


DOWNLOAD_TIMEOUT_S = 30
DOWNLOAD_DEADLINE_S = 600
MAX_MODEL_BYTES = 512 * 1024 * 1024


def model_cache_dir() -> Path:
    override = os.getenv("NVBROADCAST_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser()

    snap_user_common = os.getenv("SNAP_USER_COMMON", "").strip()
    if snap_user_common:
        return Path(snap_user_common) / "models"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "nvbroadcast" / "models"

    cache_home = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "nvbroadcast" / "models"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_model(
    filename: str,
    url: str,
    expected_sha256: str,
    *,
    bundled_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Return a verified bundled/cached model, downloading atomically if needed."""
    if not filename or Path(filename).name != filename:
        raise ValueError("Model filename must not contain a directory path")
    if urlparse(url).scheme != "https":
        raise ValueError(f"Model URL must use HTTPS: {url}")
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"Invalid SHA-256 for {filename}")

    bundled_path = (bundled_dir or PROJECT_ROOT / "models") / filename
    target = (cache_dir or model_cache_dir()) / filename
    candidates = [bundled_path]
    if target != bundled_path:
        candidates.append(target)

    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected:
            return candidate

    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        target.parent.chmod(0o700)
    except OSError:
        pass

    print(f"[NV Broadcast] Downloading {filename}...", flush=True)
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    downloaded = 0
    digest = hashlib.sha256()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{filename}.",
            suffix=".part",
            dir=target.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            # URL scheme is restricted above and every payload is digest-pinned.
            with urllib.request.urlopen(  # nosec B310
                url, timeout=DOWNLOAD_TIMEOUT_S
            ) as response:
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"model download exceeded {DOWNLOAD_DEADLINE_S}s"
                        )
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_MODEL_BYTES:
                        raise RuntimeError(
                            f"model download exceeded {MAX_MODEL_BYTES} bytes"
                        )
                    digest.update(chunk)
                    temp_file.write(chunk)

        if digest.hexdigest() != expected:
            raise RuntimeError(f"SHA-256 mismatch for {filename}")

        temp_path.replace(target)
        temp_path = None
        try:
            target.chmod(0o600)
        except OSError:
            pass
        print(f"[NV Broadcast] Downloaded and verified {filename}", flush=True)
        return target
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

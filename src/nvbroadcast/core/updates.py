# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""Release update checks against GitHub Releases."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen


LATEST_RELEASE_URL = "https://api.github.com/repos/Hkshoonya/nvidia-broadcast-linux/releases/latest"
DEFAULT_CHECK_INTERVAL_SECONDS = 12 * 60 * 60


@dataclass
class ReleaseInfo:
    tag_name: str
    version: str
    html_url: str
    published_at: str = ""


def _version_key(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts) if parts else (0,)


def is_newer_version(latest: str, current: str) -> bool:
    return _version_key(latest) > _version_key(current)


def should_check_for_updates(config, now: int | None = None,
                             interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS) -> bool:
    if not getattr(config, "check_for_updates", True):
        return False
    if now is None:
        now = int(time.time())
    last_check = int(getattr(config, "last_update_check", 0) or 0)
    return (now - last_check) >= interval_seconds


def release_info_from_payload(payload: dict) -> ReleaseInfo:
    tag_name = str(payload.get("tag_name", "")).strip()
    version = tag_name[1:] if tag_name.startswith("v") else tag_name
    return ReleaseInfo(
        tag_name=tag_name,
        version=version,
        html_url=str(payload.get("html_url", "")).strip(),
        published_at=str(payload.get("published_at", "")).strip(),
    )


def fetch_latest_release(timeout: int = 5) -> ReleaseInfo | None:
    request = Request(
        LATEST_RELEASE_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "nvbroadcast-update-checker",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    return release_info_from_payload(payload)

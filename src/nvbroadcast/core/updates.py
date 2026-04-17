# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""Release update checks against GitHub Releases."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import Request, urlopen


LATEST_RELEASE_URL = "https://api.github.com/repos/Hkshoonya/nvidia-broadcast-linux/releases/latest"
DEFAULT_CHECK_INTERVAL_SECONDS = 6 * 60 * 60
SNAP_STORE_URL = "https://snapcraft.io/nvbroadcast"


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    download_url: str


@dataclass
class ReleaseInfo:
    tag_name: str
    version: str
    html_url: str
    published_at: str = ""
    assets: list[ReleaseAsset] = field(default_factory=list)


@dataclass(frozen=True)
class UpdateTarget:
    button_label: str
    tooltip: str
    url: str


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


def find_release_asset(release: ReleaseInfo, suffix: str) -> ReleaseAsset | None:
    suffix = suffix.lower()
    for asset in release.assets:
        if asset.name.lower().endswith(suffix):
            return asset
    return None


def resolve_update_target(release: ReleaseInfo) -> UpdateTarget:
    """Choose the most useful user-facing update target for this install."""
    if os.environ.get("SNAP"):
        return UpdateTarget(
            button_label="Open Snap Update",
            tooltip="Open the Snap Store page for the latest stable refresh",
            url=SNAP_STORE_URL,
        )

    if sys.platform == "darwin":
        pkg_asset = find_release_asset(release, ".pkg")
        if pkg_asset is not None:
            return UpdateTarget(
                button_label="Download macOS Update",
                tooltip=f"Download the macOS package for v{release.version}",
                url=pkg_asset.download_url,
            )

    if release.html_url:
        return UpdateTarget(
            button_label="Open Release Update",
            tooltip=f"Open the release downloads for v{release.version}",
            url=release.html_url,
        )

    return UpdateTarget(
        button_label="Update Available",
        tooltip=f"Open the latest release information for v{release.version}",
        url=LATEST_RELEASE_URL,
    )


def release_info_from_payload(payload: dict) -> ReleaseInfo:
    tag_name = str(payload.get("tag_name", "")).strip()
    version = tag_name[1:] if tag_name.startswith("v") else tag_name
    assets: list[ReleaseAsset] = []
    raw_assets = payload.get("assets", [])
    if isinstance(raw_assets, list):
        for asset in raw_assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", "")).strip()
            download_url = str(asset.get("browser_download_url", "")).strip()
            if name and download_url:
                assets.append(ReleaseAsset(name=name, download_url=download_url))
    return ReleaseInfo(
        tag_name=tag_name,
        version=version,
        html_url=str(payload.get("html_url", "")).strip(),
        published_at=str(payload.get("published_at", "")).strip(),
        assets=assets,
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

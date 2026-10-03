"""Keep the Nix package source list aligned with the Git-tracked tree."""

from pathlib import Path
import subprocess

import pytest


def test_nix_source_files_match_tracked_files():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("Git index is unavailable in the Nix build source")

    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode().split("\0")
    package_files = sorted(
        path for path in tracked
        if path and path not in {"flake.nix", "flake.lock"}
        and not path.startswith("nix/")
    )
    manifest = (root / "nix/source-files.txt").read_text().splitlines()
    assert manifest == package_files

#!/usr/bin/env python3
"""Require external human pull-request authors in cumulative project credits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nvbroadcast.contributors import is_registered_contributor  # noqa: E402


MAINTAINER_LOGINS = frozenset({"hkshoonya"})


def requires_credit(github_login: str, account_type: str) -> bool:
    """Return whether this pull-request author needs a registry entry."""
    normalized_login = github_login.casefold()
    if account_type.casefold() == "bot" or normalized_login.endswith("[bot]"):
        return False
    return normalized_login not in MAINTAINER_LOGINS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--login", required=True, help="Pull-request author login")
    parser.add_argument(
        "--account-type",
        default="User",
        help="GitHub account type from the pull-request event",
    )
    args = parser.parse_args()

    login = args.login.strip()
    account_type = args.account_type.strip()
    if not login:
        parser.error("--login must not be empty")

    if not requires_credit(login, account_type):
        print(f"Contributor credit not required for @{login} ({account_type}).")
        return 0

    if is_registered_contributor(login):
        print(f"Contributor credit verified for @{login}.")
        return 0

    print(
        f"Contributor credit missing for @{login}. Add a reviewed Contributor "
        "entry to src/nvbroadcast/contributors.py and document the accepted work "
        "in CONTRIBUTORS.md before merging this pull request.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

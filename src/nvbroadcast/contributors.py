# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Canonical credits for accepted external contributions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Contributor:
    """A human contributor credited in every packaged app version."""

    name: str
    github_login: str

    @property
    def github_url(self) -> str:
        return f"https://github.com/{self.github_login}"

    @property
    def about_credit(self) -> str:
        return f"{self.name} (@{self.github_login}) {self.github_url}"


# Keep this list cumulative. The pull-request credit gate requires every
# accepted external human contributor to be registered before their PR merges.
CONTRIBUTORS: tuple[Contributor, ...] = (
    Contributor("John Maingi", "JohnMaingi-IXP"),
    Contributor("Jon Fuller", "perfectra1n"),
    Contributor("Cédric Prezelin", "Tenshock"),
    Contributor("Cenkay Çoban", "pastor0711"),
    Contributor("KadotyGamer", "KadotyGamer"),
)

CONTRIBUTOR_GITHUB_LOGINS = frozenset(
    contributor.github_login.casefold() for contributor in CONTRIBUTORS
)


def is_registered_contributor(github_login: str) -> bool:
    """Return whether a GitHub login is present in the cumulative registry."""
    return github_login.casefold() in CONTRIBUTOR_GITHUB_LOGINS


def app_contributor_credits() -> list[str]:
    """Return GTK-ready credit lines without exposing mutable registry state."""
    return [contributor.about_credit for contributor in CONTRIBUTORS]

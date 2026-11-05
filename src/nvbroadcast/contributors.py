# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0-or-later - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Canonical credits for accepted external contributions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContributionTier(str, Enum):
    """Display recognition for accepted work; never repository permissions."""

    CORE_MAINTAINER = "core_maintainer"
    MAJOR_FEATURE = "major_feature"
    CONTRIBUTOR = "contributor"


@dataclass(frozen=True, slots=True)
class Contributor:
    """A person and the reviewed scope behind their cumulative credit."""

    name: str
    github_login: str
    tier: ContributionTier = ContributionTier.CONTRIBUTOR
    areas: tuple[str, ...] = ()
    summary: str = ""
    evidence: tuple[str, ...] = ()
    delivery_note: str = ""

    @property
    def github_url(self) -> str:
        return f"https://github.com/{self.github_login}"

    @property
    def about_credit(self) -> str:
        return f"{self.name} (@{self.github_login}) {self.github_url}"

    @property
    def recognition_credit(self) -> str:
        """Keep the profile URL last for AboutWindow's person-link parser."""
        scope = self.summary
        if self.delivery_note:
            scope = f"{scope} {self.delivery_note}".strip()
        detail = f" — {scope}" if scope else ""
        return f"{self.name} (@{self.github_login}){detail} {self.github_url}"


_PROJECT_URL = "https://github.com/Hkshoonya/nvidia-broadcast-linux"


def _pull_requests(*numbers: int) -> tuple[str, ...]:
    return tuple(f"{_PROJECT_URL}/pull/{number}" for number in numbers)


CORE_MAINTAINERS: tuple[Contributor, ...] = (
    Contributor(
        "DocZeus", "Hkshoonya", tier=ContributionTier.CORE_MAINTAINER,
        areas=("Project creation", "Maintenance"),
        summary="Original project creator and maintainer.",
        evidence=(_PROJECT_URL,),
    ),
)


# Keep this list cumulative. The pull-request credit gate requires every
# accepted external human contributor to be registered before their PR merges.
CONTRIBUTORS: tuple[Contributor, ...] = (
    Contributor(
        "John Maingi", "JohnMaingi-IXP", tier=ContributionTier.CONTRIBUTOR,
        areas=("Installer diagnosis",),
        summary="Validated Whisper installer dependency diagnosis.",
        evidence=_pull_requests(10),
    ),
    Contributor(
        "Jon Fuller", "perfectra1n", tier=ContributionTier.MAJOR_FEATURE,
        areas=("GPU video", "Audio processing", "Runtime reliability"),
        summary="GPU frame transport, audio processing, and runtime reliability.",
        evidence=_pull_requests(23, 24, 25, 26, 27, 28, 29),
    ),
    Contributor(
        "Cédric Prezelin", "Tenshock", tier=ContributionTier.MAJOR_FEATURE,
        areas=("Camera pipelines", "Runtime ownership", "Dependency security"),
        summary="Camera recovery, runtime ownership, and dependency security.",
        evidence=_pull_requests(19, 40, 41, 42, 43, 51, 61, 62, 70, 76),
    ),
    Contributor(
        "Cenkay Çoban", "pastor0711", tier=ContributionTier.CONTRIBUTOR,
        areas=("Profiles", "Startup recovery"),
        summary="Explicit profile auto-start and safe failed-start recovery.",
        evidence=_pull_requests(52),
    ),
    Contributor(
        "KadotyGamer", "KadotyGamer", tier=ContributionTier.CONTRIBUTOR,
        areas=("Source installation", "Python compatibility"),
        summary="Python 3.14 source installation and desktop binding checks.",
        evidence=_pull_requests(103),
    ),
    Contributor(
        "Mauricio Zaffari", "mauriciozaffari", tier=ContributionTier.CONTRIBUTOR,
        areas=("CUDA compatibility",),
        summary="CUDA detection for system-provided onnxruntime packages.",
        evidence=_pull_requests(156),
    ),
    Contributor(
        "Mradul Kumar", "Nirusaki-Malaal", tier=ContributionTier.MAJOR_FEATURE,
        areas=("Python compatibility", "Native-model recovery"),
        summary="Experimental Python 3.15 CPU tooling and native-model checks.",
        evidence=_pull_requests(180, 185),
        delivery_note="Not included in v1.5.4.",
    ),
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


def app_credit_sections() -> tuple[tuple[str, list[str]], ...]:
    """Return ordered, independent AboutWindow sections with reviewed scopes."""
    people = CORE_MAINTAINERS + CONTRIBUTORS
    return tuple(
        (title, [person.recognition_credit for person in people if person.tier == tier])
        for tier, title in (
            (ContributionTier.CORE_MAINTAINER, "Core Maintainers"),
            (ContributionTier.MAJOR_FEATURE, "Major Feature Contributors"),
            (ContributionTier.CONTRIBUTOR, "Contributors"),
        )
    )

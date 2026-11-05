import re
import subprocess
import sys
import unittest
from pathlib import Path

from nvbroadcast.contributors import (
    CORE_MAINTAINERS,
    CONTRIBUTORS,
    ContributionTier,
    Contributor,
    app_credit_sections,
    app_contributor_credits,
    is_registered_contributor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CREDIT_CHECK = REPO_ROOT / "scripts" / "check_contributor_credit.py"


class ContributorCreditTests(unittest.TestCase):
    def test_public_contributor_record_matches_cumulative_registry(self):
        contributor_record = (REPO_ROOT / "CONTRIBUTORS.md").read_text()
        documented_logins = {
            match.casefold()
            for match in re.findall(
                r"^### .+ \(\[@([A-Za-z0-9-]+)\]"
                r"\(https://github\.com/\1\)\)$",
                contributor_record,
                flags=re.MULTILINE,
            )
        }
        people = CORE_MAINTAINERS + CONTRIBUTORS
        registry_logins = {person.github_login.casefold() for person in people}
        self.assertEqual(documented_logins, registry_logins)

        readme = (REPO_ROOT / "README.md").read_text()
        for contributor in people:
            with self.subTest(github_login=contributor.github_login):
                heading = (
                    f"### {contributor.name} "
                    f"([@{contributor.github_login}]({contributor.github_url}))"
                )
                start = contributor_record.index(heading) + len(heading)
                ends = [
                    end for boundary in ("\n### ", "\n## ")
                    if (end := contributor_record.find(boundary, start)) >= 0
                ]
                section = contributor_record[start : min(ends) if ends else None]
                self.assertRegex(section, re.compile(r"\n\n- \S"))
                self.assertIn(contributor.summary, section)
                for area in contributor.areas:
                    self.assertIn(area, section)
                for evidence in contributor.evidence:
                    self.assertIn(evidence, section)
                if contributor.delivery_note:
                    self.assertIn(contributor.delivery_note, section)
                self.assertIn(contributor.github_url, readme)

        contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text()
        self.assertIn("describe the accepted work in `CONTRIBUTORS.md`", contributing)

    def test_registry_contains_all_accepted_external_contributors(self):
        required_logins = {
            "johnmaingi-ixp",
            "perfectra1n",
            "tenshock",
            "pastor0711",
            "kadotygamer",
            "mauriciozaffari",
            "nirusaki-malaal",
        }
        actual_logins = {
            contributor.github_login.casefold() for contributor in CONTRIBUTORS
        }
        self.assertTrue(required_logins.issubset(actual_logins))

    def test_registry_entries_are_unique_and_well_formed(self):
        people = CORE_MAINTAINERS + CONTRIBUTORS
        logins = [person.github_login.casefold() for person in people]
        self.assertEqual(len(logins), len(set(logins)))
        for contributor in people:
            self.assertTrue(contributor.name.strip())
            self.assertRegex(
                contributor.github_login,
                re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"),
            )
            self.assertEqual(
                contributor.github_url,
                f"https://github.com/{contributor.github_login}",
            )
            self.assertIsInstance(contributor.tier, ContributionTier)
            self.assertTrue(contributor.areas)
            self.assertEqual(len(contributor.areas), len(set(contributor.areas)))
            self.assertTrue(all(area.strip() for area in contributor.areas))
            self.assertTrue(contributor.summary.strip())
            self.assertTrue(contributor.evidence)
            self.assertTrue(all(
                url.startswith("https://github.com/Hkshoonya/nvidia-broadcast-linux")
                for url in contributor.evidence
            ))

    def test_three_ordered_sections_preserve_all_people_and_reviewed_scopes(self):
        sections = app_credit_sections()
        self.assertEqual([title for title, _ in sections], [
            "Core Maintainers", "Major Feature Contributors", "Contributors",
        ])
        expected_logins = (
            {"Hkshoonya"},
            {"perfectra1n", "Tenshock", "Nirusaki-Malaal"},
            {"JohnMaingi-IXP", "pastor0711", "KadotyGamer", "mauriciozaffari"},
        )
        all_credits = []
        for (_, credits), expected in zip(sections, expected_logins):
            self.assertEqual({re.search(r"\(@([^)]+)\)", line)[1] for line in credits}, expected)
            all_credits.extend(credits)
        for person in CORE_MAINTAINERS + CONTRIBUTORS:
            self.assertEqual(all_credits.count(person.recognition_credit), 1)
            self.assertIn(person.summary, person.recognition_credit)
            self.assertTrue(person.recognition_credit.endswith(person.github_url))
        # Mutating a returned section must never change the immutable registry.
        sections[0][1].clear()
        self.assertEqual(len(app_credit_sections()[0][1]), 1)

    def test_display_groups_do_not_replace_acceptance_or_release_facts(self):
        people = {person.github_login: person for person in CONTRIBUTORS}
        finding = people["JohnMaingi-IXP"]
        self.assertIn("Validated", finding.summary)
        self.assertIn("/pull/10", finding.evidence[0])
        contributor_record = (REPO_ROOT / "CONTRIBUTORS.md").read_text()
        self.assertIn("closed as superseded rather than merged verbatim", contributor_record)
        advanced = people["Nirusaki-Malaal"]
        self.assertEqual(advanced.tier, ContributionTier.MAJOR_FEATURE)
        self.assertEqual(advanced.delivery_note, "Not included in v1.5.4.")
        self.assertIn("Experimental", advanced.summary)
        self.assertEqual({url.rsplit("/", 1)[-1] for url in advanced.evidence}, {"180", "185"})
        self.assertIn("unpublished v1.5.5 preparation", contributor_record)
        self.assertNotIn("release pending", advanced.recognition_credit)
        self.assertFalse(is_registered_contributor("Hkshoonya"))

    def test_new_credit_defaults_to_contributor_without_count_based_promotion(self):
        entry = Contributor(
            "Verified Fixer", "verified-fixer",
            areas=("Verified fixes",), summary="A verified focused fix.",
            evidence=tuple(f"https://example.invalid/pr/{index}" for index in range(50)),
        )
        self.assertEqual(entry.tier, ContributionTier.CONTRIBUTOR)

    def test_about_credits_are_cumulative_and_include_tenshock(self):
        credits = app_contributor_credits()
        self.assertEqual(len(credits), len(CONTRIBUTORS))
        self.assertIn(
            "Cédric Prezelin (@Tenshock) https://github.com/Tenshock",
            credits,
        )
        credits.clear()
        self.assertEqual(len(app_contributor_credits()), len(CONTRIBUTORS))

    def test_registry_lookup_is_case_insensitive(self):
        self.assertTrue(is_registered_contributor("TENSHOCK"))
        self.assertFalse(is_registered_contributor("not-a-registered-contributor"))

    def test_credit_gate_accepts_registered_contributor(self):
        completed = self._run_credit_check("Tenshock")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("verified for @Tenshock", completed.stdout)

    def test_credit_gate_exempts_maintainer_and_bots(self):
        maintainer = self._run_credit_check("Hkshoonya")
        bot = self._run_credit_check("dependabot[bot]", "Bot")
        self.assertEqual(maintainer.returncode, 0, maintainer.stderr)
        self.assertEqual(bot.returncode, 0, bot.stderr)

    def test_credit_gate_rejects_unregistered_human(self):
        completed = self._run_credit_check("unregistered-human")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("Contributor credit missing", completed.stderr)
        self.assertIn("src/nvbroadcast/contributors.py", completed.stderr)
        self.assertIn("CONTRIBUTORS.md", completed.stderr)

    def test_pull_request_workflow_runs_dedicated_credit_gate(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-checks.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("contributor-credit:", workflow)
        self.assertIn("name: Contributor credit", workflow)
        self.assertIn("python3 scripts/check_contributor_credit.py", workflow)
        self.assertIn("github.event.pull_request.user.login", workflow)
        self.assertIn("github.event.pull_request.user.type", workflow)
        self.assertIn("persist-credentials: false", workflow)

    def _run_credit_check(
        self,
        login: str,
        account_type: str = "User",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(CREDIT_CHECK),
                "--login",
                login,
                "--account-type",
                account_type,
            ),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()

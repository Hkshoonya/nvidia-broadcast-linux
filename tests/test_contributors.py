import re
import subprocess
import sys
import unittest
from pathlib import Path

from nvbroadcast.contributors import (
    CONTRIBUTORS,
    app_contributor_credits,
    is_registered_contributor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CREDIT_CHECK = REPO_ROOT / "scripts" / "check_contributor_credit.py"


class ContributorCreditTests(unittest.TestCase):
    def test_registry_contains_all_accepted_external_contributors(self):
        required_logins = {
            "johnmaingi-ixp",
            "perfectra1n",
            "tenshock",
            "pastor0711",
        }
        actual_logins = {
            contributor.github_login.casefold() for contributor in CONTRIBUTORS
        }
        self.assertTrue(required_logins.issubset(actual_logins))

    def test_registry_entries_are_unique_and_well_formed(self):
        logins = [contributor.github_login.casefold() for contributor in CONTRIBUTORS]
        self.assertEqual(len(logins), len(set(logins)))
        for contributor in CONTRIBUTORS:
            self.assertTrue(contributor.name.strip())
            self.assertRegex(
                contributor.github_login,
                re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"),
            )
            self.assertEqual(
                contributor.github_url,
                f"https://github.com/{contributor.github_login}",
            )

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

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ACCOUNT = "https://github.com/Hkshoonya"
CANONICAL_REPOSITORY = f"{CANONICAL_ACCOUNT}/nvidia-broadcast-linux"
WRONG_ACCOUNT = "https://github.com/doczeus"


class ProjectIdentityTests(unittest.TestCase):
    def test_canonical_identity_is_consistent(self):
        expected_references = {
            "LICENSE": CANONICAL_REPOSITORY,
            "NOTICE": CANONICAL_REPOSITORY,
            "README.md": CANONICAL_REPOSITORY,
            "CONTRIBUTORS.md": CANONICAL_REPOSITORY,
            "CONTRIBUTING.md": CANONICAL_ACCOUNT,
            "docs/CANONICAL_PROJECT.md": CANONICAL_REPOSITORY,
            "pyproject.toml": CANONICAL_REPOSITORY,
            "snap/snapcraft.yaml": CANONICAL_REPOSITORY,
        }

        for relative_path, expected in expected_references.items():
            with self.subTest(path=relative_path):
                content = (REPO_ROOT / relative_path).read_text()
                self.assertIn(expected, content)
                self.assertNotIn(WRONG_ACCOUNT, content)

    def test_notice_names_original_creator_and_official_releases(self):
        notice = (REPO_ROOT / "NOTICE").read_text()

        self.assertIn("Original creator and maintainer: DocZeus (@Hkshoonya)", notice)
        self.assertIn(f"{CANONICAL_REPOSITORY}/releases", notice)
        self.assertIn("does not replace the LICENSE file", notice)
        self.assertIn("does not remove, replace, or diminish authorship credit", notice)


if __name__ == "__main__":
    unittest.main()

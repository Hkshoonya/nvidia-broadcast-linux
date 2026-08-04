import tempfile
import unittest
from pathlib import Path

from scripts.validate_snap_runtime import dependency_problems, discover_package_roots


class SnapRuntimeValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snap_root = Path(self.temp_dir.name)
        self.site_packages = self.snap_root / "lib/python3.12/site-packages"
        self.site_packages.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _add_distribution(
        self,
        name: str,
        version: str,
        requirements: tuple[str, ...] = (),
    ) -> None:
        metadata_dir = self.site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
        metadata_dir.mkdir()
        lines = [
            "Metadata-Version: 2.4",
            f"Name: {name}",
            f"Version: {version}",
        ]
        lines.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
        (metadata_dir / "METADATA").write_text("\n".join(lines) + "\n")

    def _add_valid_runtime(self) -> None:
        self._add_distribution("packaging", "26.3")
        self._add_distribution("setuptools", "83.0.0")
        self._add_distribution("opencv-contrib-python", "4.14.0.94")
        self._add_distribution(
            "nvbroadcast",
            "1.4.0",
            (
                "packaging>=26.0",
                "setuptools>=83.0.0",
                "opencv-contrib-python>=4.8.1.78,<5",
                'ignored-extra; extra == "meeting"',
                'ignored-platform; sys_platform == "darwin"',
            ),
        )

    def test_discovers_snap_python_roots(self):
        self.assertEqual(discover_package_roots(self.snap_root), [self.site_packages])

    def test_accepts_closed_runtime_dependency_set(self):
        self._add_valid_runtime()

        count, problems = dependency_problems(self.snap_root, "arm64")

        self.assertEqual(count, 4)
        self.assertEqual(problems, [])

    def test_rejects_missing_dependency_and_opencv_major_upgrade(self):
        self._add_distribution("setuptools", "83.0.0")
        self._add_distribution("opencv-contrib-python", "5.0.0.93")
        self._add_distribution("nvbroadcast", "1.4.0", ("packaging>=26.0",))

        _, problems = dependency_problems(self.snap_root, "amd64")

        self.assertTrue(any("packaging" in problem for problem in problems))
        self.assertTrue(any("do not satisfy <5,>=4.8" in problem for problem in problems))

    def test_rejects_multiple_opencv_owners(self):
        self._add_valid_runtime()
        self._add_distribution("opencv-python-headless", "4.14.0.94")

        _, problems = dependency_problems(self.snap_root, "arm64")

        self.assertTrue(
            any("exactly one OpenCV owner" in problem for problem in problems)
        )


if __name__ == "__main__":
    unittest.main()

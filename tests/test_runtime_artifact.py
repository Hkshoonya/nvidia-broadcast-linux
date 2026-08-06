import tempfile
import unittest
from pathlib import Path

from nvbroadcast.runtime.artifact import ArtifactEnvironment


class ArtifactDependencySubstitutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifact_root = Path(self.temp_dir.name)
        self.site_packages = self.artifact_root / "lib/python3.12/site-packages"
        self.site_packages.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _add_distribution(
        self,
        name: str,
        version: str,
        requirements: tuple[str, ...] = (),
    ) -> None:
        metadata_dir = self.site_packages / (
            f"{name.replace('-', '_')}-{version}.dist-info"
        )
        metadata_dir.mkdir()
        lines = [
            "Metadata-Version: 2.4",
            f"Name: {name}",
            f"Version: {version}",
        ]
        lines.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
        (metadata_dir / "METADATA").write_text("\n".join(lines) + "\n")

    def _problems(self) -> list[str]:
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        return environment.dependency_closure_problems(
            {"onnxruntime": "onnxruntime-gpu"}
        )

    def test_substitute_distribution_satisfies_dependency(self):
        self._add_distribution(
            "faster-whisper", "1.2.1", ("onnxruntime>=1.14,<2",)
        )
        self._add_distribution("onnxruntime-gpu", "1.24.4")

        self.assertEqual(self._problems(), [])

    def test_missing_substitute_distribution_remains_unsatisfied(self):
        self._add_distribution(
            "faster-whisper", "1.2.1", ("onnxruntime>=1.14,<2",)
        )

        problems = self._problems()

        self.assertTrue(any("requires missing package onnxruntime" in item for item in problems))

    def test_substitute_distribution_must_satisfy_version_constraint(self):
        self._add_distribution(
            "faster-whisper", "1.2.1", ("onnxruntime>=1.14,<2",)
        )
        self._add_distribution("onnxruntime-gpu", "2.0.0")

        problems = self._problems()

        self.assertTrue(any("found 2.0.0" in item for item in problems))


if __name__ == "__main__":
    unittest.main()

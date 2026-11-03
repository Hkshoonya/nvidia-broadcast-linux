import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def _problems(self, roots: tuple[str, ...] | None = None) -> list[str]:
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        return environment.dependency_closure_problems(
            {"onnxruntime": "onnxruntime-gpu"}, roots=roots
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

    def test_rooted_validation_ignores_unrelated_broken_distribution(self):
        self._add_distribution(
            "faster-whisper", "1.2.1", ("ctranslate2>=4.0",)
        )
        self._add_distribution("ctranslate2", "4.6.0")
        self._add_distribution(
            "unrelated-package", "1.0", ("missing-development-package",)
        )

        self.assertEqual(self._problems(("faster-whisper",)), [])

    def test_rooted_validation_rejects_missing_transitive_dependency(self):
        self._add_distribution(
            "faster-whisper", "1.2.1", ("ctranslate2>=4.0",)
        )
        self._add_distribution(
            "ctranslate2", "4.6.0", ("future-backend-dependency>=1",)
        )

        problems = self._problems(("faster-whisper",))

        self.assertTrue(
            any(
                "ctranslate2 requires missing package future-backend-dependency"
                in item
                for item in problems
            )
        )

    def test_current_deduplicates_symlinked_python_paths(self):
        self._add_distribution("faster-whisper", "1.2.1")
        alias = self.artifact_root / "lib64"
        alias.symlink_to(self.artifact_root / "lib", target_is_directory=True)
        aliased_site_packages = alias / "python3.12/site-packages"

        with mock.patch.object(
            sys,
            "path",
            [str(self.site_packages), str(aliased_site_packages)],
        ):
            environment = ArtifactEnvironment.current()

        self.assertEqual(len(environment.distributions), 1)
        self.assertEqual(environment.installed, {"faster-whisper": ("1.2.1",)})

    def test_transitive_requested_extra_requires_its_missing_dependency(self):
        self._add_distribution("app", "1.0", ("transport[tls]>=1",))
        self._add_distribution(
            "transport", "1.0", ('tls-backend>=1; extra == "tls"',)
        )
        for roots in (None, ("app",)):
            with self.subTest(roots=roots):
                self.assertTrue(any("missing package tls-backend" in p
                                    for p in self._problems(roots)))

    def test_explicit_root_extra_requires_its_missing_dependency(self):
        self._add_distribution(
            "app", "1.0", ('cuda-dependency>=2; extra == "cuda"',)
        )
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        self.assertEqual(environment.dependency_closure_problems(roots=("app",)), [])
        problems = environment.dependency_closure_problems(
            roots=("app",), root_extras={"app": {"cuda"}}
        )
        self.assertTrue(any("missing package cuda-dependency" in p for p in problems))

    def test_multiple_selected_extras_detect_conflicting_version_requirements(self):
        self._add_distribution(
            "app", "1.0", ('backend<2; extra == "old"',
                           'backend>=2; extra == "new"')
        )
        self._add_distribution("backend", "2.0")
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        problems = environment.dependency_closure_problems(
            roots=("app",), root_extras={"app": {"old", "new"}}
        )
        self.assertTrue(any("app requires backend<2" in p for p in problems))
        self.assertEqual(environment.dependency_closure_problems(
            roots=("app",), root_extras={"app": {"new"}}
        ), [])

    def test_extra_context_does_not_hide_base_marker_requirements(self):
        self._add_distribution(
            "app", "1.0", ('base-backend; extra != "cuda"',
                           'cuda-backend; extra == "cuda"')
        )
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        problems = environment.dependency_closure_problems(
            roots=("app",), root_extras={"app": {"cuda"}}
        )
        self.assertTrue(any("missing package base-backend" in p for p in problems))
        self.assertTrue(any("missing package cuda-backend" in p for p in problems))

    def test_extra_context_preserves_platform_and_python_markers(self):
        self._add_distribution("app", "1.0", (
            'linux-backend; extra == "feature" and sys_platform == "linux"',
            'mac-backend; extra == "feature" and sys_platform == "darwin"',
            'future-backend; extra == "feature" and python_version >= "3.14"',
        ))
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        problems = environment.dependency_closure_problems(
            root_extras={"app": {"feature"}}
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("missing package linux-backend", problems[0])

    def test_unselected_extras_do_not_leak_to_dependency_packages(self):
        self._add_distribution("app", "1.0", ('backend; extra == "feature"',))
        self._add_distribution("backend", "1.0", ('unselected; extra == "feature"',))
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        self.assertEqual(environment.dependency_closure_problems(
            root_extras={"app": {"feature"}}
        ), [])

    def test_shared_dependencies_union_extras_regardless_of_visit_order(self):
        self._add_distribution("app", "1.0", ("backend[first]", "bridge"))
        self._add_distribution("bridge", "1.0", ("backend[second]",))
        self._add_distribution("backend", "1.0", (
            'first-leaf; extra == "first"', 'second-leaf; extra == "second"',
        ))
        for roots in (None, ("app",)):
            with self.subTest(roots=roots):
                problems = self._problems(roots)
                self.assertTrue(any("missing package first-leaf" in p for p in problems))
                self.assertTrue(any("missing package second-leaf" in p for p in problems))

    def test_cycle_revisits_a_package_when_an_extra_is_requested_later(self):
        self._add_distribution("app", "1.0", ("bridge", 'leaf; extra == "late"'))
        self._add_distribution("bridge", "1.0", ("app[late]",))
        problems = self._problems(("app",))
        self.assertEqual(len(problems), 1)
        self.assertIn("missing package leaf", problems[0])
        self._add_distribution("leaf", "1.0")
        self.assertEqual(self._problems(("app",)), [])

    def test_extra_names_and_distribution_names_are_normalized(self):
        self._add_distribution("app-package", "1.0", (
            'backend[fast_path]; extra == "Fast-Feature"',
        ))
        self._add_distribution("backend", "1.0", (
            'leaf; extra == "fast-path"',
        ))
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        problems = environment.dependency_closure_problems(
            roots=("APP_package",), root_extras={"App.Package": {"fast_feature"}}
        )
        self.assertTrue(any("missing package leaf" in p for p in problems))

    def test_runtime_substitution_retains_dependency_extra_requirements(self):
        self._add_distribution("app", "1.0", ("onnxruntime[feature]>=1",))
        self._add_distribution("onnxruntime-gpu", "1.24.4", (
            'runtime-leaf; extra == "feature"',
        ))
        problems = self._problems(("app",))
        self.assertTrue(any("missing package runtime-leaf" in p for p in problems))

    def test_requested_extra_root_must_exist(self):
        environment = ArtifactEnvironment.inspect(self.artifact_root, "amd64")
        self.assertEqual(environment.dependency_closure_problems(
            root_extras={"missing-app": {"feature"}}
        ), ["required dependency root is missing: missing-app"])


if __name__ == "__main__":
    unittest.main()

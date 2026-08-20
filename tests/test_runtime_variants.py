import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import install_runtime_variant
from nvbroadcast.runtime.variants import (
    FASTER_WHISPER_REQUIREMENT,
    FASTER_WHISPER_VERSION,
    RuntimeVariant,
    current_distribution_inventory,
    detect_runtime_variant,
    runtime_ownership_problems,
)


class RuntimeVariantTests(unittest.TestCase):
    def test_supported_meeting_backend_version_is_pinned(self):
        self.assertEqual(
            FASTER_WHISPER_REQUIREMENT,
            "faster-whisper==1.2.1",
        )

    def test_user_site_runtime_is_hidden_from_system_site_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / "venv"
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", venv],
                check=True,
            )
            python = venv / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            env = os.environ.copy()
            env.pop("PYTHONNOUSERSITE", None)
            env["PYTHONUSERBASE"] = str(root / "user-base")
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

            paths = subprocess.run(
                [
                    python,
                    "-c",
                    (
                        "import json, site, sysconfig; "
                        "print(json.dumps([sysconfig.get_path('purelib'), "
                        "site.getusersitepackages()]))"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            venv_site, user_site = map(Path, json.loads(paths.stdout))

            for site_packages, version in (
                (venv_site, "98.0"),
                (user_site, "99.0"),
            ):
                dist_info = site_packages / f"onnxruntime-{version}.dist-info"
                dist_info.mkdir(parents=True)
                (dist_info / "METADATA").write_text(
                    "Metadata-Version: 2.1\n"
                    "Name: onnxruntime\n"
                    f"Version: {version}\n"
                )

            inventory_command = [
                python,
                "-c",
                (
                    "import json; "
                    "from nvbroadcast.runtime.variants import "
                    "current_distribution_inventory; "
                    "print(json.dumps(current_distribution_inventory()))"
                ),
            ]
            contaminated = subprocess.run(
                inventory_command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            contaminated_versions = json.loads(contaminated.stdout)["onnxruntime"]
            self.assertIn("98.0", contaminated_versions)
            self.assertIn("99.0", contaminated_versions)

            env["PYTHONNOUSERSITE"] = "1"
            isolated = subprocess.run(
                inventory_command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            isolated_versions = json.loads(isolated.stdout)["onnxruntime"]
            self.assertIn("98.0", isolated_versions)
            self.assertNotIn("99.0", isolated_versions)

    def test_inventory_deduplicates_symlinked_python_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site_packages = root / "lib/python3.13/site-packages"
            dist_info = site_packages / "onnxruntime-1.24.4.dist-info"
            dist_info.mkdir(parents=True)
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\n"
                "Name: onnxruntime\n"
                "Version: 1.24.4\n"
            )
            alias = root / "lib64"
            alias.symlink_to(root / "lib", target_is_directory=True)

            with mock.patch.object(
                sys,
                "path",
                [str(site_packages), str(alias / "python3.13/site-packages")],
            ):
                inventory = current_distribution_inventory()

        self.assertEqual(inventory, {"onnxruntime": ("1.24.4",)})

    def test_cpu_contract_accepts_single_cpu_owner(self):
        self.assertEqual(
            runtime_ownership_problems(
                RuntimeVariant.CPU,
                {"onnxruntime": ("1.24.4",)},
                ["CPUExecutionProvider"],
            ),
            [],
        )

    def test_cuda_contract_accepts_single_gpu_owner_with_cpu_fallback(self):
        self.assertEqual(
            runtime_ownership_problems(
                RuntimeVariant.CUDA,
                {"onnxruntime-gpu": ("1.24.4",)},
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            [],
        )

    def test_mixed_owners_are_rejected(self):
        problems = runtime_ownership_problems(
            RuntimeVariant.CUDA,
            {"onnxruntime": ("1.24.4",), "onnxruntime-gpu": ("1.24.4",)},
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertTrue(any("unexpected runtime distribution" in item for item in problems))

    def test_duplicate_owner_is_rejected(self):
        problems = runtime_ownership_problems(
            RuntimeVariant.CPU,
            {"onnxruntime": ("1.24.4", "1.24.4")},
            ["CPUExecutionProvider"],
        )
        self.assertTrue(any("found 2" in item for item in problems))

    def test_provider_contract_is_enforced(self):
        problems = runtime_ownership_problems(
            RuntimeVariant.CPU,
            {"onnxruntime": ("1.24.4",)},
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
        )
        self.assertTrue(any("forbidden execution provider" in item for item in problems))

    def test_detect_requires_exactly_one_owner(self):
        self.assertEqual(
            detect_runtime_variant({"onnxruntime-gpu": ("1.24.4",)}),
            RuntimeVariant.CUDA,
        )
        self.assertIsNone(
            detect_runtime_variant(
                {"onnxruntime": ("1.24.4",), "onnxruntime-gpu": ("1.24.4",)}
            )
        )

    def test_installer_uses_support_extra_before_no_deps_backend(self):
        with (
            mock.patch.object(
                install_runtime_variant,
                "runtime_owner_inventory",
                side_effect=[{}, {"onnxruntime-gpu": ("1.24.4",)}],
            ),
            mock.patch.object(
                install_runtime_variant, "run_pip"
            ) as run_pip,
            mock.patch.object(
                install_runtime_variant, "validate_meeting_dependencies"
            ) as validate_meeting_dependencies,
            mock.patch.object(
                install_runtime_variant.subprocess, "run"
            ) as run,
        ):
            install_runtime_variant.install(Path("/project"), "cuda", "faster")

        self.assertEqual(
            run_pip.call_args_list,
            [
                mock.call(
                    "install", "--upgrade", "/project[cuda,meeting-support]"
                ),
                mock.call(
                    "install",
                    "--no-deps",
                    FASTER_WHISPER_REQUIREMENT,
                ),
            ],
        )
        validate_meeting_dependencies.assert_called_once_with("cuda", "faster")
        run.assert_called_once_with(
            [
                install_runtime_variant.sys.executable,
                "-m",
                "nvbroadcast.runtime",
                "--variant",
                "cuda",
            ],
            check=True,
        )

    def test_installer_all_policy_preserves_both_meeting_backends(self):
        with (
            mock.patch.object(
                install_runtime_variant,
                "runtime_owner_inventory",
                side_effect=[{}, {"onnxruntime": ("1.24.4",)}],
            ),
            mock.patch.object(
                install_runtime_variant, "run_pip"
            ) as run_pip,
            mock.patch.object(
                install_runtime_variant, "validate_meeting_dependencies"
            ) as validate_meeting_dependencies,
            mock.patch.object(install_runtime_variant.subprocess, "run"),
        ):
            install_runtime_variant.install(Path("/project"), "cpu", "all")

        self.assertEqual(
            run_pip.call_args_list,
            [
                mock.call(
                    "install",
                    "--upgrade",
                    "/project[cpu,meeting-support,meeting]",
                ),
                mock.call(
                    "install",
                    "--no-deps",
                    FASTER_WHISPER_REQUIREMENT,
                ),
            ],
        )
        validate_meeting_dependencies.assert_called_once_with("cpu", "all")

    def test_refuses_cpu_to_cuda_before_mutation(self):
        self._assert_owner_transition_is_refused(
            "cuda", {"onnxruntime": ("1.24.4",)}, "onnxruntime-gpu"
        )

    def test_refuses_cuda_to_cpu_before_mutation(self):
        self._assert_owner_transition_is_refused(
            "cpu", {"onnxruntime-gpu": ("1.24.4",)}, "onnxruntime"
        )

    def test_refuses_mixed_owners_before_mutation(self):
        self._assert_owner_transition_is_refused(
            "cpu",
            {
                "onnxruntime": ("1.24.4",),
                "onnxruntime-gpu": ("1.24.4",),
            },
            "onnxruntime",
        )

    def test_refuses_duplicate_owner_before_mutation(self):
        self._assert_owner_transition_is_refused(
            "cpu",
            {"onnxruntime": ("1.24.4", "1.24.4")},
            "onnxruntime",
        )

    def _assert_owner_transition_is_refused(
        self,
        variant: str,
        inventory: dict[str, tuple[str, ...]],
        selected_owner: str,
    ) -> None:
        with (
            mock.patch.object(
                install_runtime_variant,
                "runtime_owner_inventory",
                return_value=inventory,
            ),
            mock.patch.object(
                install_runtime_variant, "run_pip"
            ) as run_pip,
            mock.patch.object(
                install_runtime_variant, "guard_source_environment"
            ) as guard,
            mock.patch.object(
                install_runtime_variant.subprocess, "run"
            ) as run,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                rf"Refusing runtime-owner transition.*{selected_owner}.*remove /project/.venv",
            ):
                install_runtime_variant.install(
                    Path("/project"),
                    variant,
                    "none",
                    source_venv=Path("/project/.venv"),
                )

        run_pip.assert_not_called()
        guard.assert_not_called()
        run.assert_not_called()

    def test_source_guard_runs_before_runtime_installation(self):
        events = []

        def guard(*_args):
            events.append("guard")

        def run_pip(*_args):
            events.append("pip")

        with (
            mock.patch.object(
                install_runtime_variant,
                "runtime_owner_inventory",
                side_effect=[
                    {"onnxruntime": ("1.24.4",)},
                    {"onnxruntime": ("1.24.4",)},
                ],
            ),
            mock.patch.object(
                install_runtime_variant,
                "guard_source_environment",
                side_effect=guard,
            ),
            mock.patch.object(
                install_runtime_variant, "run_pip", side_effect=run_pip
            ),
            mock.patch.object(install_runtime_variant.subprocess, "run"),
        ):
            install_runtime_variant.install(
                Path("/project"),
                "cpu",
                "none",
                editable=True,
                source_venv=Path("/project/.venv"),
            )

        self.assertEqual(events, ["guard", "pip"])

    def test_preflight_only_guards_source_environment(self):
        project = Path("/project")
        source_venv = project / ".venv"
        with (
            mock.patch.object(
                install_runtime_variant,
                "preflight_runtime_owner",
            ) as preflight_runtime_owner,
            mock.patch.object(
                install_runtime_variant,
                "guard_source_environment",
            ) as guard_source_environment,
            mock.patch.object(
                install_runtime_variant.sys,
                "argv",
                [
                    "install_runtime_variant.py",
                    "--project",
                    str(project),
                    "--variant",
                    "cpu",
                    "--source-venv",
                    str(source_venv),
                    "--preflight-only",
                ],
            ),
        ):
            self.assertEqual(install_runtime_variant.main(), 0)

        preflight_runtime_owner.assert_called_once_with(
            project, "cpu", source_venv
        )
        guard_source_environment.assert_called_once_with(project, source_venv)

    def test_installer_rejects_mixed_owners_after_installation(self):
        mixed_inventory = {
            "onnxruntime": ("1.24.4",),
            "onnxruntime-gpu": ("1.24.4",),
        }
        with (
            mock.patch.object(
                install_runtime_variant,
                "runtime_owner_inventory",
                side_effect=[{}, mixed_inventory],
            ),
            mock.patch.object(
                install_runtime_variant, "run_pip"
            ) as run_pip,
            mock.patch.object(
                install_runtime_variant.subprocess, "run"
            ) as run,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "expected exactly one onnxruntime-gpu distribution",
            ):
                install_runtime_variant.install(
                    Path("/project"), "cuda", "none"
                )

        run_pip.assert_called_once_with(
            "install", "--upgrade", "/project[cuda]"
        )
        run.assert_not_called()

    def test_cuda_meeting_closure_substitutes_gpu_runtime(self):
        environment = mock.Mock(
            installed={
                "faster-whisper": (
                    FASTER_WHISPER_VERSION,
                )
            }
        )
        environment.dependency_closure_problems.return_value = []

        with mock.patch(
            "nvbroadcast.runtime.artifact.ArtifactEnvironment.current",
            return_value=environment,
        ):
            install_runtime_variant.validate_meeting_dependencies(
                "cuda", "faster"
            )

        environment.dependency_closure_problems.assert_called_once_with(
            {"onnxruntime": "onnxruntime-gpu"},
            roots={"nvbroadcast", "faster-whisper"},
        )

    def test_cpu_meeting_closure_uses_standard_runtime_requirement(self):
        environment = mock.Mock(
            installed={
                "faster-whisper": (
                    FASTER_WHISPER_VERSION,
                )
            }
        )
        environment.dependency_closure_problems.return_value = []

        with mock.patch(
            "nvbroadcast.runtime.artifact.ArtifactEnvironment.current",
            return_value=environment,
        ):
            install_runtime_variant.validate_meeting_dependencies(
                "cpu", "faster"
            )

        environment.dependency_closure_problems.assert_called_once_with(
            None, roots={"nvbroadcast", "faster-whisper"}
        )

    def test_all_meeting_closure_includes_supported_openai_whisper_root(self):
        environment = mock.Mock(
            installed={
                "faster-whisper": (
                    FASTER_WHISPER_VERSION,
                )
            }
        )
        environment.dependency_closure_problems.return_value = []

        with mock.patch(
            "nvbroadcast.runtime.artifact.ArtifactEnvironment.current",
            return_value=environment,
        ), mock.patch.object(
            install_runtime_variant.sys, "version_info", (3, 13)
        ):
            install_runtime_variant.validate_meeting_dependencies("cpu", "all")

        environment.dependency_closure_problems.assert_called_once_with(
            None,
            roots={"nvbroadcast", "faster-whisper", "openai-whisper"},
        )

    def test_meeting_closure_rejects_unresolved_backend_dependency(self):
        environment = mock.Mock(
            installed={
                "faster-whisper": (
                    FASTER_WHISPER_VERSION,
                )
            }
        )
        environment.dependency_closure_problems.return_value = [
            "faster-whisper requires missing package future-dependency"
        ]

        with mock.patch(
            "nvbroadcast.runtime.artifact.ArtifactEnvironment.current",
            return_value=environment,
        ), self.assertRaisesRegex(RuntimeError, "future-dependency"):
            install_runtime_variant.validate_meeting_dependencies(
                "cuda", "faster"
            )

    def test_meeting_closure_rejects_unsupported_backend_version(self):
        environment = mock.Mock(installed={"faster-whisper": ("9.9.9",)})
        environment.dependency_closure_problems.return_value = []

        with mock.patch(
            "nvbroadcast.runtime.artifact.ArtifactEnvironment.current",
            return_value=environment,
        ), self.assertRaisesRegex(RuntimeError, "must be 1.2.1, found 9.9.9"):
            install_runtime_variant.validate_meeting_dependencies(
                "cpu", "faster"
            )


if __name__ == "__main__":
    unittest.main()

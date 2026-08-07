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
    RuntimeVariant,
    detect_runtime_variant,
    runtime_ownership_problems,
)


class RuntimeVariantTests(unittest.TestCase):
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
        with mock.patch.object(install_runtime_variant, "run_pip") as run_pip, \
             mock.patch.object(install_runtime_variant.subprocess, "run") as run:
            install_runtime_variant.install(Path("/project"), "cuda", "faster")

        self.assertEqual(
            run_pip.call_args_list,
            [
                mock.call(
                    "install", "--upgrade", "/project[cuda,meeting-support]"
                ),
                mock.call("install", "--no-deps", "faster-whisper"),
            ],
        )
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
        with mock.patch.object(install_runtime_variant, "run_pip") as run_pip, \
             mock.patch.object(install_runtime_variant.subprocess, "run"):
            install_runtime_variant.install(Path("/project"), "cpu", "all")

        self.assertEqual(
            run_pip.call_args_list,
            [
                mock.call(
                    "install",
                    "--upgrade",
                    "/project[cpu,meeting-support,meeting]",
                ),
                mock.call("install", "--no-deps", "faster-whisper"),
            ],
        )


if __name__ == "__main__":
    unittest.main()

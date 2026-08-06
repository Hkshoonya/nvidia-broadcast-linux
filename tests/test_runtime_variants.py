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

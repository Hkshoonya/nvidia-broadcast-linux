import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = REPO_ROOT / "scripts" / "select_python_interpreter.sh"
INSTALLER = REPO_ROOT / "install.sh"


class PythonInterpreterSelectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.bin_dir = Path(self.temporary_directory.name) / "bin"
        self.bin_dir.mkdir()

    def _write_interpreter(
        self,
        name: str,
        version: tuple[int, int],
        *,
        implementation: str = "CPython",
        has_venv: bool = True,
        has_desktop_bindings: bool = True,
    ) -> Path:
        path = self.bin_dir / name
        major, minor = version
        venv_status = 0 if has_venv else 1
        desktop_status = 0 if has_desktop_bindings else 1
        path.write_text(
            f"""#!/bin/sh
if [ "$1" = "-I" ] && [ "$2" = "-c" ] && echo "$3" | grep -q platform.python_implementation; then
    printf '{implementation}\\t{major}\\t{minor}\\n'
    exit 0
fi
if [ "$1" = "-I" ] && [ "$2" = "-c" ] && echo "$3" | grep -q ensurepip; then
    exit {venv_status}
fi
if [ "$1" = "-I" ] && [ "$2" = "-c" ] && echo "$3" | grep -q gi.require_version; then
    exit {desktop_status}
fi
exit 1
"""
        )
        path.chmod(0o755)
        return path

    def _run_selector(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PATH"] = f"{self.bin_dir}:{environment['PATH']}"
        return subprocess.run(
            ["bash", str(SELECTOR), "--package-manager", "unknown", *arguments],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def _shadow_all_candidates(self, version: tuple[int, int] = (3, 14)) -> None:
        for name in ("python3.13", "python3.12", "python3.11", "python3"):
            self._write_interpreter(name, version)

    def test_prefers_313_then_312_then_311(self):
        self._write_interpreter("python3.13", (3, 13))
        self._write_interpreter("python3.12", (3, 12))
        self._write_interpreter("python3.11", (3, 11))
        self._write_interpreter("python3", (3, 11))

        result = self._run_selector()

        self.assertEqual(result.returncode, 0, result.stderr)
        executable, version, major, minor = result.stdout.strip().split("\t")
        self.assertEqual(Path(executable), self.bin_dir / "python3.13")
        self.assertEqual((version, major, minor), ("3.13", "3", "13"))

    def test_skips_candidate_without_venv_support(self):
        self._write_interpreter("python3.13", (3, 13), has_venv=False)
        self._write_interpreter("python3.12", (3, 12))
        self._write_interpreter("python3.11", (3, 11))
        self._write_interpreter("python3", (3, 11))

        result = self._run_selector()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(result.stdout.split("\t", 1)[0]), self.bin_dir / "python3.12"
        )

    def test_uses_compatible_generic_python3_as_last_fallback(self):
        for name in ("python3.13", "python3.12", "python3.11"):
            self._write_interpreter(name, (3, 14))
        self._write_interpreter("python3", (3, 11))

        result = self._run_selector()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(result.stdout.split("\t", 1)[0]), self.bin_dir / "python3"
        )

    def test_desktop_check_falls_back_to_compatible_interpreter(self):
        self._write_interpreter("python3.13", (3, 13), has_desktop_bindings=False)
        self._write_interpreter("python3.12", (3, 12))
        self._write_interpreter("python3.11", (3, 11))
        self._write_interpreter("python3", (3, 12))

        result = self._run_selector("--require-desktop-bindings")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(result.stdout.split("\t", 1)[0]), self.bin_dir / "python3.12"
        )

    def test_explicit_path_overrides_automatic_preference(self):
        self._write_interpreter("python3.13", (3, 13))
        requested_dir = Path(self.temporary_directory.name) / "python bin"
        requested_dir.mkdir()
        requested = self._write_interpreter("python3.12", (3, 12))
        requested = requested.rename(requested_dir / "custom python")

        result = self._run_selector("--python", str(requested))

        self.assertEqual(result.returncode, 0, result.stderr)
        executable, version, *_ = result.stdout.strip().split("\t")
        self.assertEqual(Path(executable), requested)
        self.assertEqual(version, "3.12")

    def test_rejects_explicit_python_314(self):
        requested = self._write_interpreter("python3.14", (3, 14))

        result = self._run_selector("--python", str(requested))

        self.assertEqual(result.returncode, 1)
        self.assertIn("must select CPython 3.11-3.13; found 3.14", result.stderr)
        self.assertIn("will not replace the system Python", result.stderr)

    def test_rejects_non_cpython_interpreter(self):
        requested = self._write_interpreter("pypy3", (3, 11), implementation="PyPy")

        result = self._run_selector("--python", str(requested))

        self.assertEqual(result.returncode, 1)
        self.assertIn("must select CPython; found PyPy", result.stderr)

    def test_rejects_explicit_interpreter_without_desktop_bindings(self):
        requested = self._write_interpreter(
            "python3.13", (3, 13), has_desktop_bindings=False
        )

        result = self._run_selector(
            "--python", str(requested), "--require-desktop-bindings"
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot import the required GTK4", result.stderr)
        self.assertIn("will not add a third-party repository", result.stderr)

    def test_reports_guidance_when_no_compatible_interpreter_exists(self):
        self._shadow_all_candidates()

        result = self._run_selector()

        self.assertEqual(result.returncode, 1)
        self.assertIn("No fully supported CPython interpreter", result.stderr)
        self.assertIn("official repositories", result.stderr)
        self.assertIn("--python /path/to/python3", result.stderr)

    def test_selector_rejects_empty_explicit_path(self):
        result = self._run_selector("--python", "")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--python requires an interpreter path", result.stderr)

    def test_installer_rejects_empty_python_option_before_preflight(self):
        result = subprocess.run(
            ["bash", str(INSTALLER), "--python="],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--python requires an interpreter path", result.stdout)


if __name__ == "__main__":
    unittest.main()

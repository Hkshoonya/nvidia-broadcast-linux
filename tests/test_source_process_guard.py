import os
from pathlib import Path
import tempfile
import unittest

from scripts.check_source_venv_processes import (
    ProcessInspectionError,
    SourceProcess,
    find_source_processes,
)


class SourceProcessGuardTests(unittest.TestCase):
    def _write_process(
        self,
        proc_root: Path,
        pid: int,
        arguments: list[str],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        process_dir = proc_root / str(pid)
        process_dir.mkdir()
        (process_dir / "cmdline").write_bytes(
            b"\0".join(os.fsencode(item) for item in arguments) + b"\0"
        )
        if cwd is not None:
            (process_dir / "cwd").symlink_to(cwd, target_is_directory=True)
        if environment is not None:
            (process_dir / "environ").write_bytes(
                b"\0".join(
                    os.fsencode(f"{name}={value}")
                    for name, value in environment.items()
                )
                + b"\0"
            )

    def test_detects_main_app_and_service_for_supported_python_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc_root = root / "proc"
            proc_root.mkdir()
            project = root / "project"
            venv = project / ".venv"
            (venv / "bin").mkdir(parents=True)

            cases = (
                (101, [str(venv / "bin/python"), "-m", "nvbroadcast"], None, None),
                (
                    102,
                    [str(venv / "bin/python3"), "-m", "nvbroadcast.vcam_service"],
                    None,
                    None,
                ),
                (
                    103,
                    [".venv/bin/python3", "-m", "nvbroadcast"],
                    project,
                    None,
                ),
                (
                    104,
                    [".venv/bin/python3", "-m", "nvbroadcast.vcam_service"],
                    project,
                    None,
                ),
                (
                    105,
                    ["python3.14t", "-m", "nvbroadcast"],
                    None,
                    {"VIRTUAL_ENV": str(venv)},
                ),
                (
                    106,
                    ["python", "-m", "nvbroadcast.vcam_service", "--format", "i420"],
                    None,
                    {"VIRTUAL_ENV": str(venv)},
                ),
            )
            for pid, arguments, cwd, environment in cases:
                self._write_process(
                    proc_root,
                    pid,
                    arguments,
                    cwd=cwd,
                    environment=environment,
                )

            self.assertEqual(
                find_source_processes(venv, proc_root=proc_root),
                [
                    SourceProcess(101, "nvbroadcast"),
                    SourceProcess(102, "nvbroadcast.vcam_service"),
                    SourceProcess(103, "nvbroadcast"),
                    SourceProcess(104, "nvbroadcast.vcam_service"),
                    SourceProcess(105, "nvbroadcast"),
                    SourceProcess(106, "nvbroadcast.vcam_service"),
                ],
            )

    def test_detects_exact_venv_console_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc_root = root / "proc"
            proc_root.mkdir()
            project = root / "project"
            venv = project / ".venv"
            (venv / "bin").mkdir(parents=True)

            self._write_process(
                proc_root,
                107,
                [str(venv / "bin/python"), str(venv / "bin/nvbroadcast")],
            )
            self._write_process(
                proc_root,
                108,
                [
                    str(venv / "bin/python"),
                    ".venv/bin/nvbroadcast-vcam",
                    "--format",
                    "i420",
                ],
                cwd=project,
            )

            self.assertEqual(
                find_source_processes(venv, proc_root=proc_root),
                [
                    SourceProcess(107, "nvbroadcast"),
                    SourceProcess(108, "nvbroadcast.vcam_service"),
                ],
            )

    def test_detects_modules_after_python_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc_root = root / "proc"
            proc_root.mkdir()
            venv = root / "project/.venv"
            (venv / "bin").mkdir(parents=True)

            cases = (
                (109, [str(venv / "bin/python"), "-s", "-m", "nvbroadcast"]),
                (
                    110,
                    [
                        str(venv / "bin/python"),
                        "-EsOO",
                        "-W",
                        "ignore",
                        "-Xdev",
                        "--check-hash-based-pycs",
                        "always",
                        "-m",
                        "nvbroadcast.vcam_service",
                    ],
                ),
                (
                    111,
                    ["python3.14t", "-mnvbroadcast"],
                    {"VIRTUAL_ENV": str(venv)},
                ),
            )
            for case in cases:
                pid, arguments, *environment = case
                self._write_process(
                    proc_root,
                    pid,
                    arguments,
                    environment=environment[0] if environment else None,
                )

            self.assertEqual(
                find_source_processes(venv, proc_root=proc_root),
                [
                    SourceProcess(109, "nvbroadcast"),
                    SourceProcess(110, "nvbroadcast.vcam_service"),
                    SourceProcess(111, "nvbroadcast"),
                ],
            )

    def test_ignores_foreign_environments_unrelated_modules_and_lookalikes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc_root = root / "proc"
            proc_root.mkdir()
            venv = root / "project/.venv"
            foreign_venv = root / "other/.venv"
            (venv / "bin").mkdir(parents=True)
            (foreign_venv / "bin").mkdir(parents=True)

            cases = (
                (201, [str(foreign_venv / "bin/python"), "-m", "nvbroadcast"]),
                (202, [str(venv / "bin/python"), "-m", "nvbroadcast.audio.service"]),
                (203, [str(venv / "bin/python"), "-m", "nvbroadcast.vcam_service.extra"]),
                (204, [str(venv / "bin/python"), "-c", "import nvbroadcast"]),
                (205, [str(venv / "bin/not-python"), "-m", "nvbroadcast"]),
                (
                    207,
                    [
                        str(foreign_venv / "bin/python"),
                        str(venv / "bin/nvbroadcast"),
                    ],
                ),
                (
                    208,
                    [str(venv / "bin/python"), str(root / "bin/nvbroadcast")],
                ),
                (
                    209,
                    [
                        str(venv / "bin/python"),
                        str(venv / "bin/nvbroadcast-extra"),
                    ],
                ),
                (
                    210,
                    [str(venv / "bin/python"), "-c", "pass", "-m", "nvbroadcast"],
                ),
                (
                    211,
                    [str(venv / "bin/python"), "script.py", "-m", "nvbroadcast"],
                ),
                (
                    212,
                    [str(venv / "bin/python"), "--unknown", "-m", "nvbroadcast"],
                ),
                (213, [str(venv / "bin/python"), "-W"]),
                (
                    214,
                    [
                        str(venv / "bin/python"),
                        "--check-hash-based-pycs",
                        "sometimes",
                        "-m",
                        "nvbroadcast",
                    ],
                ),
            )
            for pid, arguments in cases:
                self._write_process(proc_root, pid, arguments)
            self._write_process(
                proc_root,
                206,
                ["python", "-m", "nvbroadcast"],
                environment={"VIRTUAL_ENV": str(foreign_venv)},
            )

            self.assertEqual(find_source_processes(venv, proc_root=proc_root), [])

    def test_filters_processes_owned_by_another_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc_root = root / "proc"
            proc_root.mkdir()
            venv = root / ".venv"
            (venv / "bin").mkdir(parents=True)
            self._write_process(
                proc_root,
                301,
                [str(venv / "bin/python"), "-m", "nvbroadcast"],
            )

            self.assertEqual(
                find_source_processes(
                    venv,
                    proc_root=proc_root,
                    uid=os.getuid() + 1,
                ),
                [],
            )

    def test_fails_closed_when_relevant_process_data_is_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp) / "proc"
            process_dir = proc_root / "401"
            (process_dir / "cmdline").mkdir(parents=True)

            with self.assertRaisesRegex(
                ProcessInspectionError,
                "cannot inspect process command line",
            ):
                find_source_processes(Path(tmp) / ".venv", proc_root=proc_root)

    def test_fails_closed_without_proc_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ProcessInspectionError,
                "process filesystem is unavailable",
            ):
                find_source_processes(
                    Path(tmp) / ".venv",
                    proc_root=Path(tmp) / "missing-proc",
                )


if __name__ == "__main__":
    unittest.main()

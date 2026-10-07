"""Exercise macOS installer scripts in temporary directories with mocked tools.

No Homebrew, package installation, GUI, camera or microphone is used here.
"""

from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "1.5.3-1"
RUNTIME_ID = "a" * 64


def package_script(delimiter: str) -> str:
    builder = (ROOT / "build-packages.sh").read_text().split("build_pkg() {", 1)[1]
    return re.split(r"<<\s*'" + re.escape(delimiter) + "'", builder, maxsplit=1)[1].split(
        f"\n{delimiter}", 1
    )[0].lstrip()


class MacPackageInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.app = self.root / "opt" / "nvbroadcast"
        self.app.mkdir(parents=True)
        self.home = self.root / "home with spaces"
        self.home.mkdir()
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.brew = self.root / "homebrew"
        self.log = self.root / "commands.log"
        self.environment = dict(
            os.environ,
            HOME=str(self.home),
            PATH=f"{self.tools}:/usr/bin:/bin",
            MACOS_TEST_LOG=str(self.log),
            MACOS_TEST_INTERPRETER=sys.executable,
            MACOS_TEST_BREW=str(self.brew),
            MACOS_TEST_FAIL="",
            MACOS_TEST_BAD_PATH="",
            MACOS_TEST_ACL_PATH="",
            MACOS_TEST_VERSION="14.6.1",
            MACOS_TEST_ARCH="arm64",
        )
        (self.app / "macos-package-version").write_text(PACKAGE_ID + "\n")
        (self.app / "macos-runtime-id").write_text(RUNTIME_ID + "\n")
        for directory in ("src", "data", "scripts"):
            (self.app / directory).mkdir()
        (self.app / "src" / "source.txt").write_text("package source\n")
        for filename in (
            "pyproject.toml", "LICENSE", "NOTICE", "README.md", "CONTRIBUTORS.md"
        ):
            (self.app / filename).write_text("package fixture\n")
        self.write_executable(
            self.app / "scripts" / "install_runtime_variant.py", "exit 0"
        )
        self.write_executable(
            self.app / "scripts" / "setup_macos_runtime.sh", "exit 0"
        )
        self.local_bin = self.root / "usr" / "local" / "bin"
        self.local_bin.mkdir(parents=True)
        self.write_executable(self.local_bin / "nvbroadcast", "exit 0")
        self.write_executable(
            self.tools / "uname",
            'if [[ "$1" == "-m" ]]; then echo "$MACOS_TEST_ARCH"; '
            'else echo Darwin; fi',
        )
        self.write_executable(
            self.tools / "sw_vers", 'echo "$MACOS_TEST_VERSION"'
        )
        # Simulated stat owner isolates privilege policy from the test host's UID.
        self.write_executable(
            self.tools / "ls",
            'echo "drwxr-xr-x root wheel fixture"\n'
            'if [[ "$2" == "$MACOS_TEST_ACL_PATH" ]]; then '
            'echo " 0: user:someone allow write,add_file"; fi',
        )
        self.write_executable(
            self.tools / "stat",
            r'''
if [[ "$3" == "$MACOS_TEST_BAD_PATH" ]]; then
    echo "1000 755"
else
    mode=$("$MACOS_TEST_INTERPRETER" -c \
        'import os, stat, sys; print(format(stat.S_IMODE(os.stat(sys.argv[1]).st_mode), "o"))' \
        "$3")
    printf '0 %s\n' "$mode"
fi
''',
        )

        for path in self.root.rglob("*"):
            path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)

    @staticmethod
    def write_executable(path: Path, body: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\nset -e\n" + body + "\n")
        path.chmod(0o755)

    @property
    def runtime(self) -> Path:
        return (
            self.home / "Library" / "Application Support" / "NVBroadcast"
            / f"{PACKAGE_ID}-{RUNTIME_ID}"
        )

    def commands(self) -> str:
        return self.log.read_text() if self.log.exists() else ""

    def run_script(self, script: str, *arguments: str):
        path = self.root / "run-script.sh"
        path.write_text(script)
        return subprocess.run(
            ["/bin/bash", str(path), *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def root_script(self, delimiter: str) -> str:
        script = package_script(delimiter)
        script = script.replace(
            "export PATH=/usr/bin:/bin:/usr/sbin:/sbin",
            f'export PATH="{self.tools}:/usr/bin:/bin"',
        )
        for tool in ("uname", "sw_vers", "stat"):
            script = script.replace(f"/usr/bin/{tool}", str(self.tools / tool))
        script = script.replace("/bin/ls", str(self.tools / "ls"))
        script = script.replace("/opt/nvbroadcast", str(self.app))
        script = script.replace("/usr/local/bin", str(self.local_bin))
        # Replace the remaining ancestor checks only; subprocesses stay isolated.
        script = script.replace("/usr/local ", str(self.local_bin.parent) + " ")
        script = script.replace("/usr ", str(self.local_bin.parent.parent) + " ")
        script = script.replace("/opt ", str(self.app.parent) + " ")
        return script

    def setup_script(self, *, simulated_root=False) -> str:
        script = (ROOT / "scripts" / "setup_macos_runtime.sh").read_text()
        script = script.replace(
            'INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
            f'INSTALL_DIR="{self.app}"',
        )
        script = script.replace("/opt/homebrew", str(self.brew))
        script = script.replace("/usr/bin/uname", str(self.tools / "uname"))
        if simulated_root:
            # EUID is readonly: exercise the same branch with a literal test UID.
            script = script.replace("(( EUID == 0 ))", "(( 0 == 0 ))")
        return script

    def provide_brew(self):
        self.write_executable(
            self.brew / "bin" / "brew",
            'echo "brew $*" >> "$MACOS_TEST_LOG"\n'
            '[[ "$1" == "--prefix" ]]\n'
            'echo "$MACOS_TEST_BREW"',
        )

    def provide_python(self, minor=13, *, probe_ok=True):
        path = self.brew / "opt" / f"python@3.{minor}" / "bin" / f"python3.{minor}"
        body = r'''
echo "python $0 $*" >> "$MACOS_TEST_LOG"
if [[ "$1" == "-" ]]; then
    /bin/cat > /dev/null
    if [[ "$0" != *"/.venv/"* && PROBE_STATUS != 0 ]]; then exit 1; fi
    if [[ "$MACOS_TEST_FAIL" == "gi" && "$0" == *"/.venv/"* ]]; then exit 19; fi
elif [[ "$1 $2" == "-m venv" ]]; then
    if [[ "$MACOS_TEST_FAIL" == "venv" ]]; then exit 17; fi
    mkdir -p "$3/bin"
    cp "$0" "$3/bin/python"
    chmod 755 "$3/bin/python"
elif [[ "$1 $2 $3" == "-m pip install" ]]; then
    if [[ "$MACOS_TEST_FAIL" == "pip" ]]; then exit 18; fi
elif [[ "$1 $2 $3" == "-m pip check" ]]; then
    if [[ "$MACOS_TEST_FAIL" == "closure" ]]; then exit 21; fi
elif [[ "$1" == *"install_runtime_variant.py" ]]; then
    [[ "$2" == "--project" && -f "$3/src/source.txt" && -w "$3" ]]
    if [[ "$MACOS_TEST_FAIL" == "runtime" ]]; then exit 20; fi
fi
'''.replace("PROBE_STATUS", "0" if probe_ok else "1")
        self.write_executable(path, body)
        return path

    def make_ready_runtime(self):
        self.write_executable(
            self.runtime / ".venv" / "bin" / "python",
            'printf "runtime" >> "$MACOS_TEST_LOG"\n'
            'printf " <%s>" "$@" >> "$MACOS_TEST_LOG"',
        )
        (self.runtime / "runtime-ready").write_text(f"{PACKAGE_ID}:{RUNTIME_ID}\n")

    def launcher(self):
        return package_script("LAUNCHER").replace(
            'INSTALL_DIR="/opt/nvbroadcast"', f'INSTALL_DIR="{self.app}"'
        )

    def test_root_installer_scripts_do_not_execute_hostile_brew_or_python(self):
        for name in ("brew", "python", "python3", "pip", "python3.13"):
            self.write_executable(
                self.tools / name,
                'echo "HOSTILE TOOL EXECUTED" >> "$MACOS_TEST_LOG"; exit 99',
            )
        for delimiter in ("PREINST", "POSTINST"):
            with self.subTest(script=delimiter):
                result = self.run_script(self.root_script(delimiter), "", "", "/")
                self.assertEqual(result.returncode, 0, result.stderr)
                if delimiter == "POSTINST":
                    self.assertIn("Runtime setup is still required", result.stdout)
                self.assertEqual(self.commands(), "")

    def test_postinstall_reports_incomplete_payload_as_failure(self):
        (self.app / "scripts" / "setup_macos_runtime.sh").unlink()
        result = self.run_script(self.root_script("POSTINST"), "", "", "/")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("payload is incomplete", result.stderr)

    def test_scripts_refuse_alternate_installer_target(self):
        for delimiter in ("PREINST", "POSTINST"):
            with self.subTest(script=delimiter):
                result = self.run_script(self.root_script(delimiter), "", "", "/Volumes/Other")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("running system volume", result.stderr)

    def test_preinstall_accepts_safe_destination(self):
        result = self.run_script(self.root_script("PREINST"), "", "", "/")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preinstall_rejects_non_admin_destination(self):
        self.environment["MACOS_TEST_BAD_PATH"] = str(self.local_bin)
        result = self.run_script(self.root_script("PREINST"), "", "", "/")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("admin-owned", result.stderr)

    def test_preinstall_rejects_writable_destination(self):
        self.local_bin.chmod(0o777)
        result = self.run_script(self.root_script("PREINST"), "", "", "/")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not group/other writable", result.stderr)

    def test_preinstall_rejects_destination_acl_that_posix_mode_hides(self):
        self.environment["MACOS_TEST_ACL_PATH"] = str(self.local_bin)
        result = self.run_script(self.root_script("PREINST"), "", "", "/")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ACL requires administrator review", result.stderr)

    def test_preinstall_rejects_redirected_launcher_and_nested_payload(self):
        for path in (self.local_bin / "nvbroadcast", self.app / "src" / "redirect.py"):
            with self.subTest(path=path):
                if path.exists():
                    path.unlink()
                path.symlink_to(self.root / "outside")
                result = self.run_script(self.root_script("PREINST"), "", "", "/")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("symlinked package destination", result.stderr)
                path.unlink()

    def test_preinstall_rejects_unsupported_architecture_and_os(self):
        for key, value in (("MACOS_TEST_ARCH", "x86_64"), ("MACOS_TEST_VERSION", "12.7")):
            with self.subTest(key=key):
                original = self.environment[key]
                self.environment[key] = value
                result = self.run_script(self.root_script("PREINST"), "", "", "/")
                self.assertNotEqual(result.returncode, 0)
                self.environment[key] = original

    def test_setup_refuses_root_before_invoking_user_tools(self):
        self.provide_brew()
        self.provide_python()
        result = self.run_script(self.setup_script(simulated_root=True))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without sudo", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_setup_missing_homebrew_is_actionable_and_does_not_download(self):
        result = self.run_script(self.setup_script())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("https://brew.sh", result.stderr)
        self.assertEqual(self.commands(), "")
        self.assertFalse(self.runtime.exists())

    def test_setup_rejects_python_with_unusable_gi(self):
        self.provide_brew()
        self.provide_python(probe_ok=False)
        result = self.run_script(self.setup_script())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("working GTK/Adw/GStreamer", result.stderr)
        self.assertIn("brew install python@3.13", result.stderr)
        self.assertFalse(self.runtime.exists())

    def test_setup_resolves_versioned_homebrew_without_python_on_path(self):
        self.provide_brew()
        python = self.provide_python()
        result = self.run_script(self.setup_script())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(python), self.commands())
        self.assertIn("--variant cpu --meeting-backends faster", self.commands())
        self.assertEqual(
            (self.runtime / "runtime-ready").read_text(), f"{PACKAGE_ID}:{RUNTIME_ID}\n"
        )
        self.assertEqual((self.runtime / "runtime-ready").stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.app / ".venv").exists())
        self.assertEqual((self.app / "src" / "source.txt").read_text(), "package source\n")
        self.assertNotIn("brew install", self.commands())

    def test_setup_selects_next_supported_python_when_first_gi_abi_fails(self):
        self.provide_brew()
        self.provide_python(13, probe_ok=False)
        second = self.provide_python(12)
        result = self.run_script(self.setup_script())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{second} -m venv", self.commands())

    def test_setup_failures_leave_no_ready_marker_and_preserve_user_files(self):
        self.provide_brew()
        self.provide_python()
        sentinel = self.home / "keep.txt"
        sentinel.write_text("keep me")
        for index, (failure, exit_code) in enumerate(
            (("venv", 17), ("pip", 18), ("runtime", 20), ("closure", 21), ("gi", 19))
        ):
            with self.subTest(failure=failure):
                self.environment["MACOS_TEST_FAIL"] = failure
                # Each test uses a different immutable generation, without deleting.
                identity = "b" * 63 + format(index, "x")
                (self.app / "macos-runtime-id").write_text(identity + "\n")
                result = self.run_script(self.setup_script())
                self.assertEqual(result.returncode, exit_code, result.stderr)
                runtime = self.runtime.parent / f"{PACKAGE_ID}-{identity}"
                self.assertFalse((runtime / "runtime-ready").exists())
                self.assertEqual(sentinel.read_text(), "keep me")

    def test_setup_does_not_replace_existing_incomplete_or_redirected_runtime(self):
        self.provide_brew()
        self.provide_python()
        self.runtime.mkdir(parents=True)
        sentinel = self.runtime / "keep.txt"
        sentinel.write_text("preserve")
        result = self.run_script(self.setup_script())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Existing runtime is incomplete", result.stderr)
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_setup_refuses_symlinked_parent(self):
        self.provide_brew()
        self.provide_python()
        (self.home / "Library").symlink_to(self.root)
        result = self.run_script(self.setup_script())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlinked runtime parent", result.stderr)

    def test_setup_refuses_symlinked_generation_without_modifying_target(self):
        self.provide_brew()
        self.provide_python()
        self.runtime.parent.mkdir(parents=True)
        self.runtime.symlink_to(self.app)
        result = self.run_script(self.setup_script())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete or redirected", result.stderr)
        self.assertEqual((self.app / "src" / "source.txt").read_text(), "package source\n")

    def test_existing_ready_runtime_is_checked_without_rebuilding(self):
        self.provide_brew()
        self.provide_python()
        self.make_ready_runtime()
        result = self.run_script(self.setup_script())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already ready", result.stdout)
        self.assertIn("nvbroadcast.runtime", self.commands())
        self.assertIn("pip", self.commands())
        self.assertNotIn("-m venv", self.commands())

    def test_rebuilt_package_gets_new_runtime_and_preserves_previous_generation(self):
        self.provide_brew()
        self.provide_python()
        self.make_ready_runtime()
        (self.app / "macos-runtime-id").write_text("c" * 64 + "\n")
        (self.app / "src" / "source.txt").write_text("rebuilt package source\n")
        result = self.run_script(self.setup_script())
        self.assertEqual(result.returncode, 0, result.stderr)
        rebuilt = self.runtime.parent / f"{PACKAGE_ID}-{'c' * 64}"
        self.assertEqual((rebuilt / "runtime-ready").read_text(), f"{PACKAGE_ID}:{'c' * 64}\n")
        self.assertEqual(
            (rebuilt / "install-source" / "src" / "source.txt").read_text(),
            "rebuilt package source\n",
        )
        self.assertEqual((self.runtime / "runtime-ready").read_text(), f"{PACKAGE_ID}:{RUNTIME_ID}\n")

    def test_launcher_cannot_fall_back_to_system_python(self):
        self.write_executable(
            self.tools / "python3", 'echo "WRONG PYTHON" >> "$MACOS_TEST_LOG"'
        )
        result = self.run_script(self.launcher())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("setup_macos_runtime.sh", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_launcher_requires_exact_ready_marker_and_preserves_arguments(self):
        self.make_ready_runtime()
        (self.runtime / "runtime-ready").write_text("other package\n")
        result = self.run_script(self.launcher(), "--argument", "with spaces")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.commands(), "")
        (self.runtime / "runtime-ready").write_text(f"{PACKAGE_ID}:{RUNTIME_ID}\n")
        result = self.run_script(self.launcher(), "--argument", "with spaces")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), "runtime <-m> <nvbroadcast> <--argument> <with spaces>")

    def test_same_version_new_source_identity_cannot_reuse_old_runtime(self):
        self.make_ready_runtime()
        (self.app / "macos-runtime-id").write_text("c" * 64 + "\n")
        result = self.run_script(self.launcher())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.commands(), "")
        self.assertTrue((self.runtime / "runtime-ready").exists())

    def test_launcher_refuses_root_before_running_user_interpreter(self):
        self.make_ready_runtime()
        result = self.run_script(self.launcher().replace("(( EUID == 0 ))", "(( 0 == 0 ))"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without sudo", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_package_source_identity_changes_with_payload_and_mode(self):
        identity_program = package_script("IDENTITY")

        def identity():
            result = subprocess.run(
                [sys.executable, "-", str(self.app)],
                input=identity_program,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return (self.app / "macos-runtime-id").read_text()

        first = identity()
        self.assertEqual(identity(), first)  # The identity file excludes itself.
        source = self.app / "src" / "source.txt"
        source.write_text("different package source\n")
        changed = identity()
        self.assertNotEqual(changed, first)
        source.chmod(0o755)
        self.assertNotEqual(identity(), changed)

    def test_source_installer_refuses_root_before_homebrew_download(self):
        script = (ROOT / "install_macos.sh").read_text().replace(
            "(( EUID == 0 ))", "(( 0 == 0 ))"
        )
        for name in ("brew", "curl"):
            self.write_executable(
                self.tools / name, 'echo "UNSAFE" >> "$MACOS_TEST_LOG"; exit 99'
            )
        result = self.run_script(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without sudo", result.stderr)
        self.assertEqual(self.commands(), "")


if __name__ == "__main__":
    unittest.main()

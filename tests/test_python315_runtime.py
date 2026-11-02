import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import install_runtime_variant as installer
from scripts import python315_runtime as runtime


class Python315RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def wheel(
        self, name, version, tag="cp315-cp315-linux_x86_64", *, metadata_version=None
    ):
        path = self.directory / f"{name}-{version}-{tag}.whl"
        prefix = f"{name}-{version}.dist-info/"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                prefix + "METADATA",
                f"Name: {name}\nVersion: {metadata_version or version}\n",
            )
            archive.writestr(prefix + "WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
        return path

    def wheels(self):
        return [
            self.wheel(name, version)
            for name, version in runtime.SOURCE_VERSIONS.items()
        ]

    def test_matching_source_wheels_are_accepted(self):
        wheels = self.wheels()
        self.assertEqual(runtime.check_wheelhouse(self.directory), wheels)

    def test_stable_abi_av_wheel_is_compatible_with_regular_python315(self):
        av = self.wheel("av", "16.1.0", "cp311-abi3-linux_x86_64")
        ort = self.wheel("onnxruntime", "1.24.4")
        self.assertEqual(runtime.check_wheelhouse(self.directory), [av, ort])

    def test_wrong_abi_platform_version_or_metadata_is_rejected(self):
        for tag, version, metadata in (
            ("cp314-cp314-linux_x86_64", "16.1.0", None),
            ("cp316-abi3-linux_x86_64", "16.1.0", None),
            ("cp315-cp315t-linux_x86_64", "16.1.0", None),
            ("cp315-cp315-linux_aarch64", "16.1.0", None),
            ("cp315-cp315-win_amd64", "16.1.0", None),
            ("cp315-cp315-linux_x86_64", "17.0.0", None),
            ("cp315-cp315-linux_x86_64", "16.1.0", "17.0.0"),
        ):
            with self.subTest(tag=tag, version=version, metadata=metadata):
                wheel = self.wheel("av", version, tag, metadata_version=metadata)
                with self.assertRaisesRegex(RuntimeError, "av==16.1.0"):
                    runtime.check_wheelhouse(self.directory)
                wheel.unlink()

    def test_missing_onnx_runtime_is_rejected(self):
        self.wheel("av", "16.1.0")
        with self.assertRaisesRegex(RuntimeError, "onnxruntime==1.24.4"):
            runtime.check_wheelhouse(self.directory)

    def test_ambiguous_builds_are_rejected(self):
        self.wheels()
        self.wheel("av", "16.1.0", "cp315-cp315-manylinux_2_28_x86_64")
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            runtime.check_wheelhouse(self.directory)

    def test_corrupt_wheel_is_rejected(self):
        (self.directory / "av-16.1.0-cp315-cp315-linux_x86_64.whl").write_bytes(
            b"invalid zip"
        )
        with self.assertRaisesRegex(RuntimeError, "Invalid runtime wheel"):
            runtime.check_wheelhouse(self.directory)

    def test_free_threaded_runtime_is_rejected(self):
        with (
            mock.patch.object(runtime.sys, "version_info", (3, 15)),
            mock.patch.object(runtime.sysconfig, "get_config_var", return_value=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "not 3.15t"):
                runtime.require_python315()

    def test_ffmpeg_library_paths_with_spaces_are_retained_and_deduplicated(self):
        directory = self.directory / "private FFmpeg" / "lib"
        directory.mkdir(parents=True)
        other = self.directory / "other lib"
        other.mkdir()
        paths = [directory] * 6 + [other]
        with (
            mock.patch.object(runtime.subprocess, "check_output",
                              side_effect=[f"{path}\n" for path in paths]) as pkgconfig,
            mock.patch.dict(os.environ, {"LDFLAGS": "-Wl,--as-needed"}),
        ):
            environment = runtime.av_build_environment()
        self.assertEqual(shlex.split(environment["LDFLAGS"]), [
            "-Wl,--as-needed", "-Wl,--disable-new-dtags",
            f"-Wl,-rpath,{directory}", f"-Wl,-rpath,{other}",
        ])
        self.assertEqual(pkgconfig.call_args_list, [
            mock.call(["pkg-config", "--variable=libdir", package], text=True)
            for package in runtime.FFMPEG_PACKAGES
        ])

    def test_relative_missing_and_loader_token_paths_fail_before_build(self):
        values = ["relative/lib", str(self.directory / "missing")]
        for character in (":", ",", "$", "\t", "\n", "\r"):
            directory = self.directory / f"invalid{character}lib"
            directory.mkdir()
            values.append(str(directory))
        for value in values:
            with (
                self.subTest(value=value),
                mock.patch.object(runtime.subprocess, "check_output", return_value=value + "\n"),
                self.assertRaisesRegex(RuntimeError, "Unsafe or missing"),
            ):
                runtime.av_build_environment()
        link = self.directory / "safe-link"
        link.symlink_to(self.directory / "invalid$lib", target_is_directory=True)
        with mock.patch.object(runtime.subprocess, "check_output", return_value=str(link) + "\n"):
            with self.assertRaisesRegex(RuntimeError, "Unsupported resolved"):
                runtime.av_build_environment()

    def test_clean_native_probe_ignores_exported_build_and_python_paths(self):
        ambient = {
            "PATH": "/usr/bin", "LD_LIBRARY_PATH": "/private/lib",
            "LD_PRELOAD": "/preload.so", "LD_AUDIT": "/audit.so",
            "LIBRARY_PATH": "/build/lib", "PYTHONPATH": "/build/python",
            "PYTHONHOME": "/other/python", "DISPLAY": ":1",
        }
        with (
            mock.patch.dict(os.environ, ambient, clear=True),
            mock.patch.object(runtime.subprocess, "run") as probe,
        ):
            runtime.verify_native_runtime("/candidate/bin/python", "faster")
        options = probe.call_args.kwargs
        self.assertEqual(options["env"], {
            "PATH": "/usr/bin", "DISPLAY": ":1", "PYTHONNOUSERSITE": "1",
        })
        self.assertEqual(probe.call_args.args[0][:3], ["/candidate/bin/python", "-I", "-c"])
        self.assertEqual(probe.call_args.args[0][-1], "faster")
        self.assertTrue(options["check"])

    def test_native_probe_failure_propagates_to_caller(self):
        with mock.patch.object(runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["probe"])):
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.verify_native_runtime("/candidate/bin/python")

    def test_av_rebuild_cannot_reuse_existing_output_or_pip_cache(self):
        build = self.directory / "build"
        (build / "venv").mkdir(parents=True)
        wheelhouse = self.directory / "wheels"
        wheelhouse.mkdir()
        old = wheelhouse / "av-16.1.0-cp311-abi3-linux_x86_64.whl"
        old.write_bytes(b"old wheel without rpath")
        staging_paths = []

        def run(*arguments, **kwargs):
            if arguments[1:4] == ("-m", "pip", "wheel"):
                self.assertIn("--no-cache-dir", arguments)
                staging = Path(arguments[arguments.index("--wheel-dir") + 1])
                self.assertNotEqual(staging, wheelhouse)
                self.assertEqual(kwargs["env"], {"LDFLAGS": "new rpath"})
                (staging / old.name).write_bytes(b"fresh wheel with rpath")
                staging_paths.append(staging)

        with (
            mock.patch.object(runtime.shutil, "which", return_value="/usr/bin/tool"),
            mock.patch.object(runtime, "av_build_environment", return_value={"LDFLAGS": "new rpath"}),
            mock.patch.object(runtime.subprocess, "check_output", side_effect=["62.1.0\n", RuntimeError("stop before ORT build")]),
            mock.patch.object(runtime, "run", side_effect=run),
            self.assertRaisesRegex(RuntimeError, "stop before ORT"),
        ):
            runtime.build(build, wheelhouse, 2)
        self.assertEqual(old.read_bytes(), b"fresh wheel with rpath")
        self.assertEqual(len(staging_paths), 1)
        self.assertFalse(staging_paths[0].exists())

    def test_installer_uses_exact_local_wheels_and_keeps_inference_validation(self):
        wheels = self.wheels()
        with (
            mock.patch.object(installer.sys, "version_info", (3, 15)),
            mock.patch.object(runtime, "require_python315"),
            mock.patch.object(installer, "preflight_runtime_install"),
            mock.patch.object(installer, "prepare_virtualenv_namespaces"),
            mock.patch.object(installer, "validate_selected_owner") as owner,
            mock.patch.object(installer, "validate_meeting_dependencies") as closure,
            mock.patch.object(installer, "run_pip") as pip,
            mock.patch.object(runtime, "verify_native_runtime") as native,
            mock.patch.object(installer.subprocess, "run") as probe,
        ):
            installer.install(
                Path("/project"), "cpu", "none", python315_wheelhouse=self.directory
            )
        arguments = pip.call_args.args
        self.assertIn("--only-binary=av,onnxruntime", arguments)
        for wheel in wheels:
            self.assertIn(str(wheel), arguments)
        owner.assert_called_once_with(Path("/project"), "cpu")
        closure.assert_called_once_with("cpu", "none", development=False)
        native.assert_called_once_with(installer.sys.executable, "none")
        self.assertIn("nvbroadcast.runtime", probe.call_args.args[0])
        self.assertIn("cpu", probe.call_args.args[0])
        self.assertNotIn("LD_LIBRARY_PATH", probe.call_args.kwargs["env"])

    def test_installer_stops_before_model_probe_when_native_av_or_audio_fails(self):
        self.wheels()
        with (
            mock.patch.object(installer.sys, "version_info", (3, 15)),
            mock.patch.object(runtime, "require_python315"),
            mock.patch.object(installer, "preflight_runtime_install"),
            mock.patch.object(installer, "prepare_virtualenv_namespaces"),
            mock.patch.object(installer, "validate_selected_owner"),
            mock.patch.object(installer, "validate_meeting_dependencies"),
            mock.patch.object(installer, "run_pip"),
            mock.patch.object(runtime, "verify_native_runtime", side_effect=subprocess.CalledProcessError(1, ["native probe"])),
            mock.patch.object(installer.subprocess, "run") as model,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                installer.install(Path("/project"), "cpu", "none", python315_wheelhouse=self.directory)
        model.assert_not_called()

    def test_missing_wheelhouse_or_cuda_fails_before_pip_and_mutations(self):
        with (
            mock.patch.object(installer.sys, "version_info", (3, 15)),
            mock.patch.object(runtime, "require_python315"),
            mock.patch.object(installer, "preflight_runtime_install") as preflight,
            mock.patch.object(installer, "run_pip") as pip,
        ):
            for variant, wheelhouse in (("cpu", None), ("cuda", self.directory)):
                with self.assertRaisesRegex(
                    RuntimeError, "experimental CPU wheelhouse"
                ):
                    installer.install(
                        Path("/project"),
                        variant,
                        "none",
                        python315_wheelhouse=wheelhouse,
                    )
        preflight.assert_not_called()
        pip.assert_not_called()

    def test_shell_installer_stops_before_system_changes_without_wheelhouse(self):
        # Exercise argument plumbing and early preflight with a fake 3.15 executable.
        fake = self.directory / "python3.15"
        fake.write_text("""#!/bin/sh
case "$3" in
  *platform.python_implementation*) printf 'CPython\\t3\\t15\\n' ;;
  *ensurepip*) exit 0 ;;
  *) exit 99 ;;
esac
""")
        fake.chmod(0o755)
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                "bash",
                str(root / "install.sh"),
                "--experimental-python315",
                "--python",
                str(fake),
            ],
            cwd=root,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("requires --wheelhouse DIR", result.stdout)
        self.assertNotIn("Installing system", result.stdout)


if __name__ == "__main__":
    unittest.main()

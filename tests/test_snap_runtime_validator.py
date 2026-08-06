import tempfile
import unittest
from pathlib import Path

from scripts.validate_snap_runtime import (
    desktop_launcher_problems,
    dependency_problems,
    discover_package_roots,
    platform_shadow_problems,
    python_runtime_problems,
)


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

    def _add_valid_runtime(self, variant: str = "cpu") -> None:
        self._add_distribution("numpy", "2.5.1")
        self._add_distribution("packaging", "26.3")
        self._add_distribution("protobuf", "7.35.1")
        self._add_distribution("setuptools", "83.0.0")
        self._add_distribution("opencv-contrib-python", "4.14.0.94")
        self._add_distribution(
            "nvbroadcast",
            "1.4.0",
            (
                "packaging>=26.0",
                "numpy>=1.26",
                "protobuf>=5.29.6",
                "setuptools>=83.0.0",
                "opencv-contrib-python>=4.8.1.78,<5",
                'ignored-extra; extra == "meeting"',
                'ignored-platform; sys_platform == "darwin"',
            ),
        )
        runtime_owner = "onnxruntime-gpu" if variant == "cuda" else "onnxruntime"
        self._add_distribution(runtime_owner, "1.24.4")

    def _add_valid_python_runtime(self) -> None:
        (self.snap_root / "pyvenv.cfg").write_text(
            "home = /usr/bin\n"
            "include-system-site-packages = false\n"
            "version = 3.12.3\n"
        )
        bin_dir = self.snap_root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").symlink_to("/usr/bin/python3.12")

    def _add_valid_desktop_launcher(self) -> None:
        gui_dir = self.snap_root / "meta/gui"
        gui_dir.mkdir(parents=True)
        (gui_dir / "icon.svg").write_text("<svg/>")
        (gui_dir / "nvbroadcast.desktop").write_text(
            "[Desktop Entry]\n"
            "Name=NV Broadcast\n"
            "Exec=nvbroadcast\n"
            "Icon=${SNAP}/meta/gui/icon.svg\n"
            "Type=Application\n"
        )

    def test_discovers_snap_python_roots(self):
        self.assertEqual(discover_package_roots(self.snap_root), [self.site_packages])

    def test_accepts_closed_runtime_dependency_set(self):
        self._add_valid_runtime()

        count, problems = dependency_problems(
            self.snap_root, "arm64", ("CPUExecutionProvider",)
        )

        self.assertEqual(count, 7)
        self.assertEqual(problems, [])

    def test_accepts_core24_python_runtime_layout(self):
        self._add_valid_python_runtime()

        self.assertEqual(python_runtime_problems(self.snap_root), [])

    def test_rejects_build_sdk_python_runtime_layout(self):
        (self.snap_root / "pyvenv.cfg").write_text(
            "home = /snap/gnome-46-2404-sdk/current/usr/bin\n"
            "include-system-site-packages = false\n"
            "version = 3.12.3\n"
            "executable = /snap/gnome-46-2404-sdk/187/usr/bin/python3.12\n"
            "command = /snap/gnome-46-2404-sdk/current/usr/bin/python3 "
            "-m venv /build/nvbroadcast/parts/nvbroadcast/install\n"
        )
        bin_dir = self.snap_root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").symlink_to("/usr/bin/python3.12")

        problems = python_runtime_problems(self.snap_root)

        self.assertTrue(any("build-only path" in problem for problem in problems))
        self.assertTrue(any("core24 Python home" in problem for problem in problems))

    def test_rejects_python_runtime_outside_core24(self):
        self._add_valid_python_runtime()
        (self.snap_root / "bin/python3").unlink()
        (self.snap_root / "bin/python3").symlink_to(
            "/snap/gnome-46-2404-sdk/current/usr/bin/python3.12"
        )

        problems = python_runtime_problems(self.snap_root)

        self.assertTrue(
            any("must target core24 Python" in problem for problem in problems)
        )

    def test_rejects_libraries_owned_by_gnome_platform(self):
        shadow = self.snap_root / "usr/lib/x86_64-linux-gnu/libgtk-4.so.1"
        shadow.parent.mkdir(parents=True)
        shadow.touch()

        self.assertEqual(
            platform_shadow_problems(self.snap_root),
            [
                "Snap shadows a GNOME platform library: "
                "usr/lib/x86_64-linux-gnu/libgtk-4.so.1"
            ],
        )

    def test_accepts_packaged_desktop_launcher(self):
        self._add_valid_desktop_launcher()

        self.assertEqual(desktop_launcher_problems(self.snap_root), [])

    def test_rejects_missing_desktop_launcher(self):
        self.assertEqual(
            desktop_launcher_problems(self.snap_root),
            ["Snap is missing meta/gui/nvbroadcast.desktop"],
        )

    def test_rejects_desktop_launcher_outside_snap_payload(self):
        self._add_valid_desktop_launcher()
        launcher = self.snap_root / "meta/gui/nvbroadcast.desktop"
        launcher.write_text(
            "[Desktop Entry]\n"
            "Name=NV Broadcast\n"
            "Exec=/opt/nvbroadcast/bin/nvbroadcast\n"
            "Icon=com.doczeus.NVBroadcast\n"
            "Type=Application\n"
        )

        problems = desktop_launcher_problems(self.snap_root)

        self.assertTrue(any("Exec must start" in problem for problem in problems))
        self.assertTrue(any("Icon must be ${SNAP}" in problem for problem in problems))

    def test_rejects_missing_desktop_launcher_icon(self):
        self._add_valid_desktop_launcher()
        (self.snap_root / "meta/gui/icon.svg").unlink()

        problems = desktop_launcher_problems(self.snap_root)

        self.assertEqual(
            problems,
            [
                "Snap desktop launcher icon does not exist: "
                "${SNAP}/meta/gui/icon.svg"
            ],
        )

    def test_rejects_missing_dependency_and_opencv_major_upgrade(self):
        self._add_distribution("setuptools", "83.0.0")
        self._add_distribution("opencv-contrib-python", "5.0.0.93")
        self._add_distribution("nvbroadcast", "1.4.0", ("packaging>=26.0",))

        _, problems = dependency_problems(
            self.snap_root,
            "amd64",
            ("CPUExecutionProvider", "CUDAExecutionProvider"),
        )

        self.assertTrue(any("packaging" in problem for problem in problems))
        self.assertTrue(any("do not satisfy <5,>=4.8" in problem for problem in problems))

    def test_rejects_multiple_opencv_owners(self):
        self._add_valid_runtime()
        self._add_distribution("opencv-python-headless", "4.14.0.94")

        _, problems = dependency_problems(
            self.snap_root, "arm64", ("CPUExecutionProvider",)
        )

        self.assertTrue(
            any("exactly one OpenCV owner" in problem for problem in problems)
        )

    def test_rejects_duplicate_required_runtime_owner(self):
        self._add_valid_runtime()
        self.site_packages = self.snap_root / "usr/lib/python3/dist-packages"
        self.site_packages.mkdir(parents=True)
        self._add_distribution("packaging", "26.3")
        self._add_distribution("protobuf", "7.35.1")

        _, problems = dependency_problems(
            self.snap_root,
            "amd64",
            ("CPUExecutionProvider", "CUDAExecutionProvider"),
        )

        self.assertTrue(
            any("packaging (26.3, 26.3)" in problem for problem in problems)
        )
        self.assertTrue(
            any("protobuf (7.35.1, 7.35.1)" in problem for problem in problems)
        )

    def test_cuda_owner_satisfies_faster_whisper_cpu_distribution_metadata(self):
        self._add_valid_runtime("cuda")
        self._add_distribution(
            "faster-whisper", "1.2.1", ("onnxruntime>=1.14,<2",)
        )

        _, problems = dependency_problems(
            self.snap_root,
            "amd64",
            ("CPUExecutionProvider", "CUDAExecutionProvider"),
        )

        self.assertEqual(problems, [])

    def test_rejects_wrong_runtime_owner_for_architecture(self):
        self._add_valid_runtime("cpu")

        _, problems = dependency_problems(
            self.snap_root,
            "amd64",
            ("CPUExecutionProvider", "CUDAExecutionProvider"),
        )

        self.assertTrue(any("onnxruntime-gpu" in problem for problem in problems))
        self.assertTrue(any("unexpected runtime distribution" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()

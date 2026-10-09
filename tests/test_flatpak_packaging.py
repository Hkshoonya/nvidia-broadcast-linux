import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLATPAK_DIR = ROOT / "packaging" / "flatpak"
MANIFEST = FLATPAK_DIR / "com.nvbroadcast.NVBroadcast.yml"
GENERATED = FLATPAK_DIR / "python3-flatpak-requirements.yaml"
REQUIREMENTS = FLATPAK_DIR / "requirements.txt"
README = FLATPAK_DIR / "README.md"
WORKFLOW = ROOT / ".github" / "workflows" / "flatpak.yml"


class FlatpakPackagingTests(unittest.TestCase):
    def test_manifest_uses_pinned_gnome_runtime_and_expected_identity(self):
        manifest = MANIFEST.read_text(encoding="utf-8")

        self.assertIn("id: com.nvbroadcast.NVBroadcast", manifest)
        self.assertIn("runtime: org.gnome.Platform", manifest)
        self.assertIn('runtime-version: "50"', manifest)
        self.assertIn("sdk: org.gnome.Sdk", manifest)
        self.assertIn("command: nvbroadcast", manifest)
        self.assertIn('test "$(uname -m)" = "x86_64"', manifest)
        self.assertIn(
            "ln -s /usr/lib/x86_64-linux-gnu/libsndfile.so.1", manifest
        )

    def test_manifest_keeps_sandbox_permissions_scoped(self):
        manifest = MANIFEST.read_text(encoding="utf-8")

        for required in (
            "--share=network",
            "--socket=wayland",
            "--socket=fallback-x11",
            "--socket=pulseaudio",
            "--filesystem=xdg-videos:create",
            "--device=all",
            "--talk-name=org.kde.StatusNotifierWatcher",
        ):
            self.assertIn(required, manifest)

        for forbidden in (
            "--filesystem=host",
            "--filesystem=home",
            "--socket=session-bus",
            "--socket=system-bus",
            "--talk-name=org.freedesktop.Flatpak",
        ):
            self.assertNotIn(forbidden, manifest)

    def test_every_generated_remote_source_has_a_sha256(self):
        generated = GENERATED.read_text(encoding="utf-8")
        urls = re.findall(r"^\s+url: (https://\S+)$", generated, re.MULTILINE)
        hashes = re.findall(r"^\s+sha256: ([0-9a-f]{64})$", generated, re.MULTILINE)

        self.assertGreater(len(urls), 40)
        self.assertEqual(len(urls), len(hashes))
        self.assertTrue(all(url.startswith("https://") for url in urls))

    def test_dependency_cleanup_cannot_remove_application_launcher(self):
        generated = GENERATED.read_text(encoding="utf-8")

        self.assertNotIn("--cleanup scripts", generated.splitlines()[0])
        self.assertNotRegex(generated, r"(?m)^\s+- /bin\s*$")

    def test_flatpak_ci_is_pinned_read_only_and_does_not_publish(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertRegex(
            workflow,
            r"ghcr\.io/flathub-infra/flatpak-github-actions@sha256:[0-9a-f]{64}",
        )
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("persist-credentials: false", workflow)
        for packaged_input in (
            "'CONTRIBUTORS.md'",
            "'LICENSE'",
            "'NOTICE'",
            "'README.md'",
            "'src/**'",
        ):
            self.assertIn(packaged_input, workflow)
        self.assertIn("flatpak-builder-lint manifest", workflow)
        self.assertIn("python3 -m pip check", workflow)
        self.assertIn('_model_entry("base", faster_whisper.__version__)', workflow)
        self.assertIn("/app/share/doc/nvbroadcast/NOTICE", workflow)
        self.assertIn("/app/share/doc/nvbroadcast/CONTRIBUTORS.md", workflow)
        self.assertIn(
            "python3 -m nvbroadcast.video.recording_smoke", workflow
        )
        self.assertIn("export_development_bundle.py", workflow)
        self.assertRegex(workflow, r"actions/upload-artifact@[0-9a-f]{40}")
        self.assertIn("path: dist/flatpak-development/", workflow)
        self.assertIn("if-no-files-found: error", workflow)
        self.assertIn("retention-days: 14", workflow)
        self.assertNotIn("gh release", workflow)
        self.assertNotIn("flatpak build-update-repo", workflow)
        self.assertNotRegex(workflow, r"(?m)^\s+push:\s*$")

    def test_dependency_inputs_include_cpu_and_meeting_runtime_only(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8")

        self.assertIn("onnxruntime>=1.24.4,<1.25", requirements)
        self.assertIn("faster-whisper==1.2.1", requirements)
        self.assertNotIn("onnxruntime-gpu", requirements)
        self.assertNotIn("tensorrt", requirements)
        self.assertNotIn("cupy", requirements)

    def test_flatpak_metadata_has_its_own_identity_and_cpu_scope(self):
        import configparser
        import xml.etree.ElementTree as ET

        metadata = ET.parse(FLATPAK_DIR / "com.nvbroadcast.NVBroadcast.metainfo.xml").getroot()
        desktop = configparser.ConfigParser(interpolation=None)
        desktop.read(FLATPAK_DIR / "com.nvbroadcast.NVBroadcast.desktop")
        self.assertEqual(metadata.findtext("id"), "com.nvbroadcast.NVBroadcast")
        self.assertEqual(metadata.findtext("name"), desktop["Desktop Entry"]["Name"])
        self.assertEqual(metadata.findtext("launchable"), "com.nvbroadcast.NVBroadcast.desktop")
        description = " ".join(" ".join(metadata.find("description").itertext()).split())
        self.assertIn("CPU processing on x86_64", description)
        self.assertIn("not affiliated with", description)
        self.assertNotIn("NVENC", description)
        self.assertNotIn("NVIDIA eye", (FLATPAK_DIR / "com.nvbroadcast.NVBroadcast.svg").read_text())
        # Screenshot dimensions in metadata must match actual PNG headers.
        import struct
        for picture in metadata.findall("screenshots/screenshot/image"):
            name = picture.text.rsplit("/", 1)[-1]
            image = (ROOT / "docs/screenshots" / name).read_bytes()
            self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", image[16:24]),
                             (int(picture.attrib["width"]), int(picture.attrib["height"])))

    def test_public_distribution_blockers_remain_explicit(self):
        readme = README.read_text(encoding="utf-8")

        for blocker in (
            "application ID",
            "license metadata",
            "metainfo-missing-screenshots",
            "trademark",
            "faster-whisper",
            "CUDA and TensorRT",
            "aarch64",
            "1.2 GB",
            "Flathub",
        ):
            self.assertIn(blocker, readme)
        self.assertIn("first-use faster-whisper download", readme)
        self.assertNotIn("faster-whisper model path is not hash-pinned", readme)


if __name__ == "__main__":
    unittest.main()

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLATPAK_DIR = ROOT / "packaging" / "flatpak"
MANIFEST = FLATPAK_DIR / "com.doczeus.NVBroadcast.yml"
GENERATED = FLATPAK_DIR / "python3-flatpak-requirements.yaml"
REQUIREMENTS = FLATPAK_DIR / "requirements.txt"
README = FLATPAK_DIR / "README.md"
WORKFLOW = ROOT / ".github" / "workflows" / "flatpak.yml"


class FlatpakPackagingTests(unittest.TestCase):
    def test_manifest_uses_pinned_gnome_runtime_and_expected_identity(self):
        manifest = MANIFEST.read_text(encoding="utf-8")

        self.assertIn("id: com.doczeus.NVBroadcast", manifest)
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
        self.assertIn("flatpak-builder-lint manifest", workflow)
        self.assertIn("python3 -m pip check", workflow)
        self.assertIn("/app/share/doc/nvbroadcast/NOTICE", workflow)
        self.assertIn("/app/share/doc/nvbroadcast/CONTRIBUTORS.md", workflow)
        self.assertNotIn("upload-artifact", workflow)
        self.assertNotRegex(workflow, r"(?m)^\s+push:\s*$")

    def test_dependency_inputs_include_cpu_and_meeting_runtime_only(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8")

        self.assertIn("onnxruntime>=1.24.4,<1.25", requirements)
        self.assertIn("faster-whisper==1.2.1", requirements)
        self.assertNotIn("onnxruntime-gpu", requirements)
        self.assertNotIn("tensorrt", requirements)
        self.assertNotIn("cupy", requirements)

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


if __name__ == "__main__":
    unittest.main()

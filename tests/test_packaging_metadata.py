import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    def test_install_script_uses_supported_tensorrt_command(self):
        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn("pip install tensorrt-cu12", install_script)
        self.assertNotIn("tensorrt-cu12-bindings", install_script)
        self.assertNotIn("tensorrt-cu12-libs", install_script)
        self.assertIn("requires Python 3.8-3.13", install_script)

    def test_debian_postinst_installs_meeting_runtime(self):
        postinst = (REPO_ROOT / "packaging" / "debian" / "postinst").read_text()
        self.assertIn("openai-whisper", postinst)

    def test_rpm_postinst_installs_meeting_runtime(self):
        spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        self.assertIn("pip install openai-whisper", spec)

    def test_snap_package_bundles_lighter_meeting_runtime(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        self.assertIn("- faster-whisper", snapcraft)
        self.assertIn("- ctranslate2", snapcraft)
        self.assertIn("- httpx", snapcraft)
        self.assertNotIn("- openai-whisper", snapcraft)

    def test_packaged_backgrounds_include_bundled_default(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("data/backgrounds/studio_bg.png", pyproject)


if __name__ == "__main__":
    unittest.main()

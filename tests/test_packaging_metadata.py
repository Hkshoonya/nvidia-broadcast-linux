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
        self.assertIn("Python runtime notice", install_script)
        self.assertIn("some premium paths use safer defaults", install_script)
        self.assertIn('rc=$?; echo ""; echo "ERROR: Installation failed at line $LINENO (exit code $rc)"', install_script)
        self.assertIn('if CUPY_TEST=$("$VENV_DIR/bin/python" -c "import cupy; a=cupy.ones(10); print(\'OK\')" 2>&1); then', install_script)
        self.assertIn("CuPy installed but verification failed.", install_script)

    def test_debian_postinst_installs_meeting_runtime(self):
        postinst = (REPO_ROOT / "packaging" / "debian" / "postinst").read_text()
        self.assertIn("pip\" install --no-deps faster-whisper", postinst)
        self.assertIn("pip\" install ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", postinst)
        self.assertNotIn("openai-whisper", postinst)
        self.assertNotIn("install --no-deps faster-whisper ctranslate2", postinst)

    def test_rpm_postinst_installs_meeting_runtime(self):
        spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        self.assertIn("pip install --no-deps faster-whisper", spec)
        self.assertIn("pip install ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", spec)
        self.assertNotIn("openai-whisper", spec)
        self.assertNotIn("install --no-deps faster-whisper ctranslate2", spec)

    def test_macos_postinstall_installs_meeting_runtime_in_two_steps(self):
        script = (REPO_ROOT / "build-packages.sh").read_text()
        self.assertIn("pip install -q --no-deps faster-whisper", script)
        self.assertIn("pip install -q ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", script)
        self.assertNotIn("install -q --no-deps faster-whisper ctranslate2", script)

    def test_snap_package_bundles_lighter_meeting_runtime(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        self.assertIn("- faster-whisper", snapcraft)
        self.assertIn("- ctranslate2", snapcraft)
        self.assertIn("- httpx", snapcraft)
        self.assertNotIn("- openai-whisper", snapcraft)

    def test_packaged_backgrounds_include_bundled_default(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("data/backgrounds/studio_bg.png", pyproject)

    def test_python_meeting_extra_uses_faster_whisper_stack(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("faster-whisper", pyproject)
        self.assertIn("ctranslate2", pyproject)
        self.assertIn("httpx", pyproject)
        self.assertIn('openai-whisper>=20231117; python_version < "3.14"', pyproject)

    def test_sponsor_walls_keep_action_markers_balanced(self):
        for relative in ("README.md", "SPONSORS.md"):
            content = (REPO_ROOT / relative).read_text()
            self.assertEqual(content.count("<!-- featured -->"), 2, relative)
            self.assertEqual(content.count("<!-- sponsors -->"), 2, relative)
            self.assertIn("https://github.com/Mattsky", content)

    def test_sponsors_workflow_noops_without_token(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "sponsors.yml").read_text()
        self.assertIn("id: sponsor-token", workflow)
        self.assertIn("SPONSORS_TOKEN is not configured yet", workflow)
        self.assertEqual(
            workflow.count("if: steps.sponsor-token.outputs.available == 'true'"),
            5,
        )


if __name__ == "__main__":
    unittest.main()

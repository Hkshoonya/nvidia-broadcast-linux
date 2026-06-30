import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    def test_release_version_metadata_is_current(self):
        current = "1.1.11"
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        package_init = (REPO_ROOT / "src" / "nvbroadcast" / "__init__.py").read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        metainfo = (REPO_ROOT / "data" / "com.doczeus.NVBroadcast.metainfo.xml").read_text()
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        docs_index = (REPO_ROOT / "docs" / "index.html").read_text()
        snap_workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()

        self.assertIn(f'version = "{current}"', pyproject)
        self.assertIn(f'__version__ = "{current}"', package_init)
        self.assertIn(f"version: '{current}'", snapcraft)
        self.assertIn(f"Version:        {current}", rpm_spec)
        self.assertIn(f'<release version="{current}" date="2026-06-23">', metainfo)
        self.assertIn(f"### v{current}", readme)
        self.assertIn(f"nvbroadcast_{current}-1_all.deb", docs_index)
        self.assertIn(f"nvbroadcast-{current}-1.noarch.rpm", docs_index)
        self.assertIn(f"NVBroadcast-{current}-1.pkg", docs_index)
        self.assertIn(f"such as v{current}", snap_workflow)

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

    def test_source_installer_installs_cuda_extra_before_gpu_verification(self):
        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn('"$VENV_DIR/bin/pip" install --upgrade "$SCRIPT_DIR[cuda]"', install_script)
        self.assertLess(
            install_script.index('"$VENV_DIR/bin/pip" install --upgrade "$SCRIPT_DIR[cuda]"'),
            install_script.index("Verifying GPU acceleration"),
        )
        self.assertIn("CUDA_ACCEL_AVAILABLE=true", install_script)
        self.assertIn("CUDA runtime: $VENV_DIR/bin/pip install --upgrade", install_script)
        self.assertIn("unavailable until CuPy installs", install_script)
        self.assertIn("CUDA modes still need GPU inference runtime", install_script)

    def test_source_installer_does_not_auto_enable_headless_vcam_service(self):
        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn("NVBROADCAST_ENABLE_HEADLESS_SERVICE", install_script)
        self.assertIn("installed but disabled by default", install_script)
        self.assertIn("disable nvbroadcast-vcam.service", install_script)
        self.assertIn("stop nvbroadcast-vcam.service", install_script)

    def test_source_setup_safely_migrates_live_v4l2loopback_label(self):
        install_script = (REPO_ROOT / "install.sh").read_text()
        setup_script = (REPO_ROOT / "scripts" / "setup_v4l2loopback.sh").read_text()

        for script in (install_script, setup_script):
            self.assertIn("LIVE", script)
            self.assertIn("LOOPBACK_COUNT", script)
            self.assertIn("fuser -s", script)
            self.assertIn("modprobe -r v4l2loopback", script)
            self.assertIn("Skipping live", script)
            self.assertIn('card_label="${', script)

    def test_core_runtime_includes_audio_denoiser_import_dependencies(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        requirements = (REPO_ROOT / "requirements.txt").read_text()
        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn('"pyrnnoise>=0.4"', pyproject)
        self.assertIn('"av>=12"', pyproject)
        self.assertIn("pyrnnoise>=0.4", requirements)
        self.assertIn("av>=12", requirements)
        self.assertIn("- pyrnnoise", snapcraft)
        self.assertIn("- av", snapcraft)
        self.assertIn("import av; import av.option", install_script)
        self.assertIn("av ... OK", install_script)

    def test_readme_documents_cuda_extra_for_source_gpu_installs(self):
        readme = (REPO_ROOT / "README.md").read_text()
        self.assertIn('pip install -e ".[cuda]"', readme)
        self.assertIn('.venv/bin/pip install --upgrade ".[cuda]"', readme)
        self.assertIn("CUDAExecutionProvider", readme)

    def test_cuda_extra_contains_onnxruntime_gpu_provider(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("cuda = [", pyproject)
        self.assertIn('"onnxruntime-gpu==1.24.4"', pyproject)
        self.assertIn('"onnxruntime>=1.24.4,<1.25"', pyproject)
        self.assertNotIn('"pycuda>=2024.1"', pyproject)
        self.assertNotIn('"nvidia-cusparse-cu12"', pyproject)
        self.assertNotIn('"nvidia-cusolver-cu12"', pyproject)

    def test_linux_package_postinstalls_install_cuda_extra(self):
        deb_postinst = (REPO_ROOT / "packaging" / "debian" / "postinst").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        rpm_postinst = rpm_spec.split("%post", 1)[1].split("%preun", 1)[0]
        self.assertIn('pip" install --upgrade "$INSTALL_DIR[cuda]"', deb_postinst)
        self.assertIn('pip install --upgrade "/opt/nvbroadcast[cuda]"', rpm_postinst)
        self.assertNotIn("pip\" install cupy-cuda12x nvidia-cuda-nvrtc-cu12", deb_postinst)

    def test_virtual_camera_label_is_nvbroadcast_everywhere(self):
        constants = (REPO_ROOT / "src" / "nvbroadcast" / "core" / "constants.py").read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        install_script = (REPO_ROOT / "install.sh").read_text()
        config_template = (REPO_ROOT / "configs" / "v4l2loopback" / "nvbroadcast.conf").read_text()
        build_packages = (REPO_ROOT / "build-packages.sh").read_text()
        setup_script = (REPO_ROOT / "scripts" / "setup_v4l2loopback.sh").read_text()
        deb_postinst = (REPO_ROOT / "packaging" / "debian" / "postinst").read_text()
        deb_rules = (REPO_ROOT / "packaging" / "debian" / "rules").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        macos_constants = (REPO_ROOT / "macos" / "Shared" / "Constants.swift").read_text()

        self.assertIn('VIRTUAL_CAM_LABEL = "NVbroadcast"', constants)
        self.assertIn('else VIRTUAL_CAM_LABEL', constants)
        self.assertIn('card_label="NVbroadcast"', readme)
        self.assertIn('select **"NVbroadcast"** as your camera', readme)
        self.assertIn('V4L2_LABEL="NVbroadcast"', install_script)
        self.assertIn('card_label=\\"${V4L2_LABEL}\\"', install_script)
        self.assertIn('card_label="${V4L2_LABEL}"', install_script)
        self.assertIn("Description=NVbroadcast Virtual Camera Service", install_script)
        self.assertIn('card_label="NVbroadcast"', config_template)
        self.assertIn("Description=NVbroadcast Virtual Camera Service", build_packages)
        self.assertIn('LABEL="NVbroadcast"', setup_script)
        self.assertIn('card_label="NVbroadcast"', deb_postinst)
        self.assertIn("Description=NVbroadcast Virtual Camera Service", deb_rules)
        self.assertIn('card_label="NVbroadcast"', rpm_spec)
        self.assertIn("Description=NVbroadcast Virtual Camera Service", rpm_spec)
        self.assertIn('static let deviceName = "NVbroadcast"', macos_constants)
        self.assertIn('static let deviceModel = "NVbroadcast"', macos_constants)
        self.assertIn("NVBROADCAST_ALLOW_OBS_VCAM_FALLBACK", (REPO_ROOT / "src" / "nvbroadcast" / "video" / "pipeline.py").read_text())

        generated_content = "\n".join(
            line
            for line in (
                install_script
                + config_template
                + build_packages
                + setup_script
                + deb_postinst
                + deb_rules
                + rpm_spec
                + macos_constants
            ).splitlines()
            if "grep -Eq" not in line
        )
        self.assertNotIn('card_label="NVIDIA Broadcast"', generated_content)
        self.assertNotIn('card_label="NVIDIA Broadcast Virtual Camera"', generated_content)
        self.assertNotIn('card_label="NV Broadcast"', generated_content)
        self.assertNotIn("Description=NVIDIA Broadcast Virtual Camera Service", generated_content)
        self.assertNotIn("Description=NV Broadcast Virtual Camera Service", generated_content)
        self.assertNotIn('deviceName = "NV Broadcast"', generated_content)
        self.assertNotIn('deviceModel = "NV Broadcast Virtual Camera"', generated_content)

    def test_release_copy_preserves_proprietary_mode_names(self):
        readme = (REPO_ROOT / "README.md").read_text()
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        metainfo = (REPO_ROOT / "data" / "com.doczeus.NVBroadcast.metainfo.xml").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        website = (REPO_ROOT / "docs" / "index.html").read_text()
        ui_window = (REPO_ROOT / "src" / "nvbroadcast" / "ui" / "window.py").read_text()

        for name in ("DocZeus", "Zeus", "Killer"):
            self.assertIn(name, readme)
            self.assertIn(name, snapcraft)
            self.assertIn(name, metainfo)
            self.assertIn(name, rpm_spec)
            self.assertIn(name, website)
            self.assertIn(name, ui_window)

    def test_debian_postinst_installs_meeting_runtime(self):
        postinst = (REPO_ROOT / "packaging" / "debian" / "postinst").read_text()
        self.assertIn("pip\" install --no-deps faster-whisper", postinst)
        self.assertIn("pip\" install ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", postinst)
        self.assertNotIn("openai-whisper", postinst)
        self.assertNotIn("install --no-deps faster-whisper ctranslate2", postinst)

    def test_rpm_postinst_installs_meeting_runtime(self):
        spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        postinst = spec.split("%post", 1)[1].split("%preun", 1)[0]
        self.assertIn("pip install --no-deps faster-whisper", postinst)
        self.assertIn("pip install ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", postinst)
        self.assertNotIn("openai-whisper", postinst)
        self.assertNotIn("install --no-deps faster-whisper ctranslate2", postinst)

    def test_macos_postinstall_installs_meeting_runtime_in_two_steps(self):
        script = (REPO_ROOT / "build-packages.sh").read_text()
        self.assertIn("pip install -q --no-deps faster-whisper", script)
        self.assertIn("pip install -q ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", script)
        self.assertNotIn("install -q --no-deps faster-whisper ctranslate2", script)

    def test_macos_source_installer_guards_openai_whisper(self):
        script = (REPO_ROOT / "install_macos.sh").read_text()
        self.assertIn("pip install -q --no-deps faster-whisper", script)
        self.assertIn("pip install -q ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm", script)
        self.assertIn("sys.version_info < (3, 14)", script)
        self.assertIn('"openai-whisper>=20231117"', script)
        self.assertNotIn("pip install -q openai-whisper\n", script)

    def test_snap_package_bundles_lighter_meeting_runtime(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        self.assertIn("- faster-whisper", snapcraft)
        self.assertIn("- ctranslate2", snapcraft)
        self.assertIn("- httpx", snapcraft)
        self.assertNotIn("- openai-whisper", snapcraft)
        self.assertIn("onnxruntime==1.24.4", snapcraft)
        self.assertIn("onnxruntime-gpu==1.24.4", snapcraft)
        self.assertIn("Installing amd64 CUDA mode runtime into Snap", snapcraft)
        self.assertIn("Skipping CUDA mode runtime", snapcraft)
        self.assertIn("arm64 Snap build stays portable and CPU-safe", snapcraft)

    def test_packaged_backgrounds_include_bundled_default(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("data/backgrounds/studio_bg.png", pyproject)

    def test_python_meeting_extra_uses_faster_whisper_stack(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("faster-whisper", pyproject)
        self.assertIn("ctranslate2", pyproject)
        self.assertIn("httpx", pyproject)
        self.assertIn('openai-whisper>=20231117; python_version < "3.14"', pyproject)

    def test_requirements_keep_meeting_runtime_python314_safe(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text()
        self.assertIn("faster-whisper", requirements)
        self.assertIn("ctranslate2", requirements)
        self.assertIn("httpx", requirements)
        self.assertIn('openai-whisper>=20231117; python_version < "3.14"', requirements)
        self.assertIn("onnxruntime-gpu==1.24.4", requirements)
        self.assertIn("onnxruntime>=1.24.4,<1.25", requirements)
        self.assertNotIn("onnxruntime-gpu>=1.16", requirements)
        self.assertNotIn("\nopenai-whisper>=20231117\n", requirements)

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

    def test_snap_workflow_uses_current_snapcraft_revisions_command(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        self.assertIn('snapcraft revisions "$SNAP_NAME"', workflow)
        self.assertNotIn("snapcraft list-revisions", workflow)

    def test_snap_workflow_supports_manual_release_recovery(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        self.assertIn("publish:", workflow)
        self.assertIn("release_tag:", workflow)
        self.assertIn("id: release-target", workflow)
        self.assertIn("release_tag is required when publishing from workflow_dispatch", workflow)
        self.assertIn("tag_name: ${{ steps.release-target.outputs.tag }}", workflow)

    def test_about_window_lists_public_backers_by_tier(self):
        window = (REPO_ROOT / "src" / "nvbroadcast" / "ui" / "window.py").read_text()
        self.assertIn('add_credit_section("Backers & Supporters"', window)
        self.assertIn("Mattsky https://github.com/Mattsky", window)
        self.assertNotIn('add_credit_section("Featured Sponsors"', window)


if __name__ == "__main__":
    unittest.main()

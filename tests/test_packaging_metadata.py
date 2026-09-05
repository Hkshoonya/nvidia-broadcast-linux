import json
import re
import stat
import struct
import tomllib
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    def _snap_description(self, snapcraft: str) -> str:
        marker = "description: |\n"
        start = snapcraft.index(marker) + len(marker)
        end = snapcraft.index("\ngrade:", start)
        lines = snapcraft[start:end].splitlines()
        return "\n".join(line[2:] if line.startswith("  ") else line for line in lines)

    def test_release_version_metadata_is_current(self):
        current = "1.5.2"
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        package_init = (REPO_ROOT / "src" / "nvbroadcast" / "__init__.py").read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
        metainfo = (REPO_ROOT / "data" / "com.doczeus.NVBroadcast.metainfo.xml").read_text()
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        docs_index = (REPO_ROOT / "docs" / "index.html").read_text()
        snap_workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        release_notes = (REPO_ROOT / "docs" / f"RELEASE_NOTES_{current}.md").read_text()

        self.assertIn(f'version = "{current}"', pyproject)
        self.assertIn(f'__version__ = "{current}"', package_init)
        self.assertIn(f"version: '{current}'", snapcraft)
        self.assertIn("title: NV Broadcast", snapcraft)
        self.assertIn(f"Version:        {current}", rpm_spec)
        self.assertIn(f'<release version="{current}" date="2026-09-04">', metainfo)
        self.assertIn(f"## v{current}", changelog)
        self.assertIn("See [CHANGELOG.md](./CHANGELOG.md)", readme)
        # Direct website downloads follow the latest published release only
        # after its artifacts and Store revisions are public.
        published = "1.5.2"
        self.assertIn(f"nvbroadcast_{published}-1_all.deb", docs_index)
        self.assertIn(f"nvbroadcast-{published}-1.noarch.rpm", docs_index)
        self.assertIn(f"NVBroadcast-{published}-1.pkg", docs_index)
        self.assertIn(f"such as v{current}", snap_workflow)
        self.assertIn(f"# NV Broadcast v{current}", release_notes)

    def test_published_native_downloads_protect_legacy_upgrades(self):
        website = (REPO_ROOT / "docs" / "index.html").read_text()
        commands = website.split("const commands = {", 1)[1].split("};", 1)[0]

        self.assertNotIn("sudo dpkg -i", commands)
        self.assertIn("nvbroadcast-native-upgrade", commands)
        self.assertIn("SHA256SUMS.packages | sha256sum -c -", commands)
        self.assertEqual(commands.count("set -euo pipefail"), 2)
        self.assertEqual(commands.count("WORKDIR=$(mktemp -d)"), 2)
        self.assertEqual(commands.count("wget --https-only"), 2)
        self.assertIn("if dpkg-query", commands)
        self.assertIn("if rpm -q nvbroadcast", commands)
        self.assertIn("sudo apt-get install --yes --no-remove", commands)
        self.assertIn("sudo dnf install --assumeyes", commands)

    def test_public_homepage_and_search_metadata_are_canonical(self):
        canonical = "https://nvbroadcast.com"
        retired = "nvbroadcast.domjarvis.com"
        homepage_files = (
            "README.md",
            "pyproject.toml",
            "build-packages.sh",
            "packaging/debian/control",
            "packaging/rpm/nvbroadcast.spec",
            "snap/snapcraft.yaml",
            "data/com.doczeus.NVBroadcast.metainfo.xml",
            "src/nvbroadcast/__init__.py",
            "src/nvbroadcast/ui/window.py",
            "docs/index.html",
        )

        for relative in homepage_files:
            content = (REPO_ROOT / relative).read_text()
            self.assertIn(canonical, content, relative)
            self.assertNotIn(retired, content, relative)

        self.assertEqual((REPO_ROOT / "docs" / "CNAME").read_text().strip(), "nvbroadcast.com")

        website = (REPO_ROOT / "docs" / "index.html").read_text()
        self.assertIn('<link rel="canonical" href="https://nvbroadcast.com/">', website)
        self.assertIn('<meta property="og:url" content="https://nvbroadcast.com/">', website)
        self.assertIn(
            '<meta property="og:image" content="https://nvbroadcast.com/og-image.png">',
            website,
        )
        self.assertIn('<meta name="twitter:card" content="summary_large_image">', website)
        self.assertNotIn('<meta name="keywords"', website)

        structured = website.split('<script type="application/ld+json">', 1)[1].split(
            "</script>", 1
        )[0]
        graph = json.loads(structured)["@graph"]
        nodes = {node["@type"]: node for node in graph}
        self.assertEqual(nodes["WebSite"]["url"], f"{canonical}/")
        application = nodes["SoftwareApplication"]
        self.assertEqual(application["softwareVersion"], "1.5.2")
        self.assertEqual(application["applicationCategory"], "MultimediaApplication")
        self.assertEqual(application["offers"]["price"], "0")

        robots = (REPO_ROOT / "docs" / "robots.txt").read_text()
        self.assertIn("User-agent: *\nAllow: /", robots)
        self.assertIn(f"Sitemap: {canonical}/sitemap.xml", robots)

        sitemap = ET.parse(REPO_ROOT / "docs" / "sitemap.xml").getroot()
        namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        self.assertEqual(
            sitemap.findtext("sitemap:url/sitemap:loc", namespaces=namespace),
            f"{canonical}/",
        )

        image = (REPO_ROOT / "docs" / "og-image.png").read_bytes()
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", image[16:24]), (1200, 630))

    def test_snap_description_stays_within_store_limit(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        description = self._snap_description(snapcraft)
        self.assertLessEqual(len(description), 4096)

    def test_snap_summary_stays_descriptive_between_releases(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        summary = re.search(r"^summary: (.+)$", snapcraft, flags=re.MULTILINE).group(1)

        self.assertLessEqual(len(summary), 79)
        self.assertIn("virtual camera", summary.lower())
        self.assertIn("background", summary.lower())
        self.assertNotRegex(summary, r"\bv?\d+\.\d+")

    def test_install_script_uses_supported_tensorrt_command(self):
        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn("pip install tensorrt-cu12", install_script)
        self.assertNotIn("tensorrt-cu12-bindings", install_script)
        self.assertNotIn("tensorrt-cu12-libs", install_script)
        self.assertIn("requires Python 3.8-3.13", install_script)
        self.assertIn("Python runtime notice", install_script)
        self.assertIn("some premium paths use safer defaults", install_script)
        self.assertIn('rc=$?; echo ""; echo "ERROR: Installation failed at line $LINENO (exit code $rc)"', install_script)
        self.assertIn("cupy-cuda12x>=14.1.1,<15", install_script)
        self.assertIn("preload_nvidia_runtime_libs; preload_nvidia_runtime_libs(); import cupy", install_script)
        self.assertIn("CuPy installed but verification failed.", install_script)

    def test_source_installer_selects_and_validates_one_runtime_variant(self):
        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn("--runtime auto|cpu|cuda] [--with-meeting", install_script)
        self.assertIn("[--python /path/to/python]", install_script)
        self.assertIn("scripts/select_python_interpreter.sh", install_script)
        self.assertIn("--require-desktop-bindings", install_script)
        self.assertIn("Python desktop bindings ... OK", install_script)
        self.assertIn('"$PYTHON_BIN" -m venv "$VENV_DIR"', install_script)
        self.assertIn("SELECTED_PYTHON_BASE", install_script)
        self.assertIn('--variant "$1"', install_script)
        self.assertIn('--meeting-backends "$meeting_backends"', install_script)
        self.assertLess(
            install_script.index('install_runtime_variant "$SELECTED_RUNTIME_VARIANT"'),
            install_script.index("Verifying GPU acceleration"),
        )
        self.assertIn('rm -rf -- "$VENV_DIR"', install_script)
        self.assertIn("CUDA_ACCEL_AVAILABLE=true", install_script)
        self.assertIn("CUDA execution probe ... OK", install_script)
        self.assertIn("--variant cuda --provider tensorrt", install_script)
        self.assertIn("TensorRT execution probe ... OK", install_script)
        self.assertIn('if [ "$TRT_INSTALLED" = true ]; then', install_script)
        self.assertNotIn("get_available_providers", install_script)
        self.assertIn("Runtime switch: stop NVBroadcast", install_script)
        self.assertIn("unavailable until CuPy installs", install_script)
        self.assertIn("CUDA modes still need GPU inference runtime", install_script)

    def test_runtime_readiness_requires_fresh_process_model_execution(self):
        runtime_cli = (
            REPO_ROOT / "src" / "nvbroadcast" / "runtime" / "__main__.py"
        ).read_text()
        probe = (
            REPO_ROOT / "src" / "nvbroadcast" / "runtime" / "probe.py"
        ).read_text()
        readme = (REPO_ROOT / "README.md").read_text()

        self.assertIn("probe_execution_provider(provider, use_cache=False)", runtime_cli)
        self.assertIn('"session.disable_cpu_ep_fallback", "1"', probe)
        self.assertIn("session.disable_fallback()", probe)
        self.assertIn("_profile_execution_providers", probe)
        self.assertIn("PROBE_MODEL_SHA256", probe)
        self.assertIn(
            ".venv/bin/python -m nvbroadcast.runtime --variant cuda",
            readme,
        )
        self.assertNotIn(
            'import onnxruntime as ort; print(ort.get_available_providers())',
            readme,
        )

    def test_source_setup_targets_select_one_runtime_owner(self):
        setup_script = (REPO_ROOT / "setup_deps.sh").read_text()
        makefile = (REPO_ROOT / "Makefile").read_text()

        preflight = (
            ".venv/bin/python scripts/install_runtime_variant.py \\\n"
            "            --project . --variant cpu --meeting-backends none \\\n"
            "            --source-venv .venv --preflight-only"
        )
        runtime_install = (
            ".venv/bin/python scripts/install_runtime_variant.py \\\n"
            "    --project . --variant cpu --meeting-backends none \\\n"
            "    --source-venv .venv --editable"
        )
        self.assertIn(preflight, setup_script)
        self.assertIn(runtime_install, setup_script)
        preflight_calls = [
            match.start()
            for match in re.finditer(
                r"^preflight_python_environment$", setup_script, flags=re.MULTILINE
            )
        ]
        self.assertEqual(len(preflight_calls), 2)
        self.assertLess(
            preflight_calls[0],
            setup_script.index("sudo apt install"),
        )
        self.assertLess(
            setup_script.index("sudo apt install"),
            preflight_calls[1],
        )
        self.assertLess(
            preflight_calls[1],
            setup_script.index("python3 -m venv .venv"),
        )
        self.assertLess(
            preflight_calls[1],
            setup_script.index(".venv/bin/pip install --upgrade"),
        )

        setup_targets = {
            "install": "$(RUNTIME_INSTALLER) --variant cpu",
            "install-gpu": "$(RUNTIME_INSTALLER) --variant cuda",
            "dev": "$(RUNTIME_INSTALLER) --variant cpu --development",
            "dev-gpu": "$(RUNTIME_INSTALLER) --variant cuda --development",
        }
        for target, command in setup_targets.items():
            recipe = f"{target}: $(VENV)\n\t{command}"
            self.assertIn(recipe, makefile)

        self.assertNotIn("\t$(PIP) install -e .\n", makefile)
        self.assertNotIn('\t$(PIP) install -e ".[dev]"', makefile)
        self.assertIn("--source-venv $(VENV) --editable", makefile)

    def test_source_installer_guards_live_environment_before_mutations(self):
        install_script = (REPO_ROOT / "install.sh").read_text()
        guard_calls = [
            match.start()
            for match in re.finditer(
                r"^guard_source_environment$", install_script, flags=re.MULTILINE
            )
        ]

        self.assertEqual(len(guard_calls), 2)
        self.assertLess(guard_calls[0], install_script.index("# ─── Step 1:"))

        environment_step = install_script.index("# ─── Step 3:")
        first_environment_mutation = min(
            install_script.index('rm -rf -- "$VENV_DIR"', environment_step),
            install_script.index(
                '"$VENV_DIR/bin/pip" install --upgrade', environment_step
            ),
        )
        self.assertGreater(guard_calls[1], environment_step)
        self.assertLess(guard_calls[1], first_environment_mutation)
        self.assertIn("check_source_venv_processes.py", install_script)
        self.assertIn(
            "Stop NVBroadcast and the virtual-camera service", install_script
        )
        self.assertIn(
            "systemctl --user stop nvbroadcast-vcam.service", install_script
        )

    def test_managed_python_environments_disable_user_site_packages(self):
        system_site_files = (
            "README.md",
            "Makefile",
            "setup_deps.sh",
            "install.sh",
            "install_macos.sh",
            "build-packages.sh",
            "packaging/debian/postinst",
            "packaging/rpm/nvbroadcast.spec",
            ".github/workflows/pr-checks.yml",
            ".github/workflows/build-packages.yml",
        )
        for relative in system_site_files:
            content = (REPO_ROOT / relative).read_text()
            self.assertIn("--system-site-packages", content, relative)
            self.assertIn("PYTHONNOUSERSITE", content, relative)

        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertGreaterEqual(
            install_script.count("export PYTHONNOUSERSITE=1"),
            3,
        )
        self.assertIn("Environment=PYTHONNOUSERSITE=1", install_script)

        build_script = (REPO_ROOT / "build-packages.sh").read_text()
        self.assertGreaterEqual(
            build_script.count("export PYTHONNOUSERSITE=1"),
            4,
        )
        self.assertIn("Environment=PYTHONNOUSERSITE=1", build_script)

        deb_rules = (REPO_ROOT / "packaging" / "debian" / "rules").read_text()
        self.assertEqual(deb_rules.count("export PYTHONNOUSERSITE=1"), 2)
        self.assertIn("Environment=PYTHONNOUSERSITE=1", deb_rules)

        rpm_spec = (
            REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec"
        ).read_text()
        self.assertEqual(rpm_spec.count("export PYTHONNOUSERSITE=1"), 3)
        self.assertIn("Environment=PYTHONNOUSERSITE=1", rpm_spec)

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
            self.assertIn("NVBROADCAST_VCAM_DEVICE_NUM", script)
            self.assertIn("NVBROADCAST_VCAM_DEVICE", script)
            self.assertIn("is not a v4l2loopback virtual camera", script)
            self.assertIn("NV Broadcast|NVbroadcast", script)
            self.assertIn('card_label="${', script)

    def test_core_runtime_includes_audio_denoiser_import_dependencies(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn(
            '"pyrnnoise>=0.4; sys_platform == \'linux\'"', pyproject
        )
        # av 17 removed av.option (needed by pyrnnoise->audiolab); av 14
        # added Codec.canonical_name. Only 16.x satisfies both.
        self.assertIn('"av>=16,<17"', pyproject)
        self.assertIn("- pyrnnoise", snapcraft)
        self.assertIn("- av>=16,<17", snapcraft)
        self.assertIn("- gnome-settings-daemon-common", snapcraft)
        self.assertIn("interface: dbus", snapcraft)
        self.assertIn("bus: session", snapcraft)
        self.assertIn("name: com.doczeus.NVBroadcast", snapcraft)
        self.assertIn("- nvbroadcast-dbus", snapcraft)
        self.assertIn("import av; import av.option", install_script)
        self.assertIn("av ... OK", install_script)

    def test_runtime_dependency_floors_include_security_fixes(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        build_workflow = (REPO_ROOT / ".github" / "workflows" / "build-packages.yml").read_text()

        for content in (pyproject, snapcraft, build_workflow):
            self.assertIn("onnx>=1.22.0", content)
            self.assertIn("click>=8.3.3", content)
            self.assertIn("protobuf>=6.33.5,<7", content)

        for content in (pyproject, snapcraft, build_workflow):
            self.assertIn("opencv-contrib-python>=4.8.1.78,<5", content)
            self.assertNotIn("opencv-python-headless", content)
            self.assertIn("Pillow>=12.3.0", content)

        self.assertIn("pyvirtualcam>=0.14", pyproject)
        self.assertIn('"packaging>=26.0"', pyproject)
        self.assertIn('dev = ["pytest>=9.0.3"]', pyproject)
        self.assertIn('"mediapipe>=1.0.0"', pyproject)
        self.assertNotIn("\n      - mediapipe\n", snapcraft)
        self.assertIn(
            '"onnxruntime>=1.23.2,<1.24; sys_platform == \'darwin\'"',
            pyproject,
        )
        self.assertFalse((REPO_ROOT / "requirements.txt").exists())

        installers = (
            REPO_ROOT / "install.sh",
            REPO_ROOT / "install_macos.sh",
            REPO_ROOT / "setup_deps.sh",
            REPO_ROOT / "packaging" / "debian" / "postinst",
            REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec",
            REPO_ROOT / "build-packages.sh",
        )
        for installer in installers:
            self.assertIn("pip>=26.2", installer.read_text(), str(installer))
            self.assertIn(
                "setuptools>=83.0.0", installer.read_text(), str(installer)
            )

        install_script = (REPO_ROOT / "install.sh").read_text()
        self.assertIn(
            "CORE_PY_MODULES=(numpy cv2 onnxruntime PIL psutil onnx mediapipe)",
            install_script,
        )
        self.assertIn('requires = ["setuptools>=83.0.0", "wheel"]', pyproject)
        self.assertIn("pip>=26.2", build_workflow)
        self.assertIn("setuptools>=83.0.0", build_workflow)

    def test_native_package_payloads_use_safe_ownership_and_permissions(self):
        build_script = (REPO_ROOT / "build-packages.sh").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()

        self.assertIn("mktemp -d", build_script)
        self.assertIn("dpkg-deb -Zxz --root-owner-group --build", build_script)
        self.assertIn("--ownership recommended", build_script)
        self.assertGreaterEqual(build_script.count('-type f -exec chmod 644 {} +'), 2)
        self.assertEqual(build_script.count('-o -name "*.egg-info"'), 3)
        self.assertIn("packaging/debian/copyright", build_script)
        self.assertIn("packaging/debian/changelog", build_script)
        self.assertIn("/^#DEBHELPER#$/d", build_script)
        self.assertIn("%{buildroot}/opt/nvbroadcast -type f -exec chmod 644 {} +", rpm_spec)
        self.assertIn(
            "chmod 644 LICENSE NOTICE README.md CONTRIBUTORS.md",
            rpm_spec,
        )

        for relative in ("LICENSE", "NOTICE", "README.md", "CONTRIBUTORS.md"):
            self.assertEqual((REPO_ROOT / relative).stat().st_mode & stat.S_IXUSR, 0, relative)

    def test_canonical_notice_and_contributors_ship_in_package_payloads(self):
        records = ("NOTICE", "CONTRIBUTORS.md")
        manifest = (REPO_ROOT / "MANIFEST.in").read_text()
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        build_script = (REPO_ROOT / "build-packages.sh").read_text()
        debian_rules = (REPO_ROOT / "packaging" / "debian" / "rules").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        macos_installer = (REPO_ROOT / "install_macos.sh").read_text()
        build_workflow = (
            REPO_ROOT / ".github" / "workflows" / "build-packages.yml"
        ).read_text()

        self.assertIn('"share/doc/nvbroadcast"', pyproject)
        for record in records:
            with self.subTest(record=record):
                self.assertIn(f"include {record}", manifest)
                self.assertIn(f'"{record}"', pyproject)
                self.assertIn(record, build_script)
                self.assertIn(record, debian_rules)
                self.assertIn(record, rpm_spec)
                self.assertIn(record, macos_installer)
                self.assertIn(f"/opt/nvbroadcast/{record}", build_workflow)

    def test_advertised_shell_entrypoints_are_executable(self):
        for relative in ("build-packages.sh", "install.sh", "install_macos.sh"):
            self.assertNotEqual(
                (REPO_ROOT / relative).stat().st_mode & stat.S_IXUSR,
                0,
                relative,
            )

    def test_native_package_upgrader_is_attested_with_exact_packages(self):
        build_script = (REPO_ROOT / "build-packages.sh").read_text()
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "build-packages.yml"
        ).read_text()
        renderer = (
            REPO_ROOT / "scripts" / "render_native_upgrade_helper.py"
        ).read_text()
        template = (
            REPO_ROOT / "scripts" / "native_package_upgrade.sh.in"
        ).read_text()

        self.assertIn("build_upgrade_helper()", build_script)
        self.assertIn("render_native_upgrade_helper.py", build_script)
        self.assertGreaterEqual(
            workflow.count("artifacts/linux-packages/nvbroadcast-native-upgrade"),
            3,
        )
        self.assertIn("dist/nvbroadcast-native-upgrade", workflow)
        self.assertIn("O_NOFOLLOW", renderer)
        self.assertIn("@DEB_SHA256@", template)
        self.assertIn("@RPM_SHA256@", template)
        self.assertIn("dpkg-deb --field", template)
        self.assertIn("rpm -qp --qf", template)

    def test_native_package_final_removal_cleans_generated_payloads(self):
        deb_postinst = (REPO_ROOT / "packaging" / "debian" / "postinst").read_text()
        deb_postrm = (REPO_ROOT / "packaging" / "debian" / "postrm").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        rpm_postun = rpm_spec.split("%postun", 1)[1].split("%files", 1)[0]

        self.assertIn('"$INSTALL_DIR/build"', deb_postinst)
        self.assertIn('"$INSTALL_DIR/src/nvbroadcast.egg-info"', deb_postinst)
        self.assertEqual(deb_postrm.count("rm -rf -- /opt/nvbroadcast"), 2)
        self.assertIn('if [ "$1" -eq 0 ]', rpm_postun)
        self.assertIn("rm -rf -- /opt/nvbroadcast", rpm_postun)
        self.assertIn("/opt/nvbroadcast/build", rpm_spec)
        self.assertIn("/opt/nvbroadcast/src/nvbroadcast.egg-info", rpm_spec)

    def test_snap_review_dispatch_cannot_publish_or_attach_a_release(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        attach_job = workflow.split("  attach-release:", 1)[1]

        self.assertIn(
            "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
            attach_job,
        )
        self.assertIn(
            "github.event_name == 'workflow_dispatch' && inputs.publish == true",
            attach_job,
        )
        self.assertNotIn("draft: ${{", attach_job)
        self.assertIn("draft: true", attach_job)

    def test_rpm_changelog_weekdays_match_dates(self):
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        for line in rpm_spec.splitlines():
            match = re.match(r"\* (\w{3}) (\w{3} \d{2} \d{4})", line)
            if not match:
                continue
            stated_weekday, date_text = match.groups()
            actual_weekday = datetime.strptime(date_text, "%b %d %Y").strftime("%a")
            self.assertEqual(stated_weekday, actual_weekday, line)

    def test_runtime_model_downloads_are_verified_and_user_writable(self):
        helper = (REPO_ROOT / "src" / "nvbroadcast" / "core" / "model_download.py").read_text()
        self.assertIn("SNAP_USER_COMMON", helper)
        self.assertIn("XDG_CACHE_HOME", helper)
        self.assertIn("Library\" / \"Caches", helper)
        self.assertIn("NamedTemporaryFile", helper)
        self.assertIn("SHA-256 mismatch", helper)

        for relative in (
            "src/nvbroadcast/audio/deepfilter.py",
            "src/nvbroadcast/video/autoframe.py",
            "src/nvbroadcast/video/effects.py",
            "src/nvbroadcast/video/face_landmarks.py",
        ):
            module = (REPO_ROOT / relative).read_text()
            self.assertIn("download_verified_model", module, relative)
            self.assertNotIn("urlretrieve", module, relative)

    def test_release_workflow_actions_are_commit_pinned(self):
        for relative in (
            ".github/workflows/build-packages.yml",
            ".github/workflows/pr-checks.yml",
            ".github/workflows/snap-promote.yml",
            ".github/workflows/snap.yml",
        ):
            workflow = (REPO_ROOT / relative).read_text()
            refs = re.findall(r"uses:\s+[^@\s]+@([^\s]+)", workflow)
            self.assertTrue(refs, relative)
            for ref in refs:
                self.assertRegex(ref, r"^[0-9a-f]{40}$", f"{relative}: {ref}")

    def test_release_builders_issue_pinned_provenance_attestations(self):
        build_workflow = (
            REPO_ROOT / ".github" / "workflows" / "build-packages.yml"
        ).read_text()
        snap_workflow = (
            REPO_ROOT / ".github" / "workflows" / "snap.yml"
        ).read_text()
        attest_action = (
            "actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d"
        )

        attestation_job = build_workflow.split("  attest-release:", 1)[1].split(
            "  release:", 1
        )[0]
        snap_builder = snap_workflow.split("  build-snap:", 1)[1].split(
            "  attach-release:", 1
        )[0]

        for builder in (attestation_job, snap_builder):
            self.assertIn("attestations: write", builder)
            self.assertIn("contents: read", builder)
            self.assertIn("id-token: write", builder)
            self.assertNotIn("contents: write", builder)
            self.assertIn(attest_action, builder)

        self.assertIn(
            "needs: [build-linux, build-macos, test-macos, test-linux, test-python]",
            attestation_job,
        )
        self.assertIn("artifacts/linux-packages/deb/*.deb", attestation_job)
        self.assertIn("artifacts/linux-packages/rpm/*.rpm", attestation_job)
        self.assertIn("artifacts/macos-packages/*.pkg", attestation_job)
        self.assertIn("artifacts/SHA256SUMS.packages", attestation_job)
        self.assertIn("steps.snapcraft.outputs.snap", snap_builder)
        self.assertEqual(build_workflow.count(attest_action), 1)
        self.assertEqual(snap_workflow.count(attest_action), 1)

    def test_release_workflows_publish_deterministic_checksum_manifests(self):
        build_workflow = (
            REPO_ROOT / ".github" / "workflows" / "build-packages.yml"
        ).read_text()
        snap_workflow = (
            REPO_ROOT / ".github" / "workflows" / "snap.yml"
        ).read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        verification = (
            REPO_ROOT / "docs" / "RELEASE_VERIFICATION.md"
        ).read_text()

        for workflow in (build_workflow, snap_workflow):
            self.assertIn("scripts/generate_release_checksums.py", workflow)

        native_attestation = build_workflow.split("  attest-release:", 1)[1].split(
            "  release:", 1
        )[0]
        native_release = build_workflow.split("  release:", 1)[1]
        self.assertIn("--output artifacts/SHA256SUMS.packages", native_attestation)
        self.assertIn("name: release-metadata", native_attestation)
        self.assertIn(
            "artifacts/release-metadata/SHA256SUMS.packages",
            native_release,
        )
        self.assertIn("needs: attest-release", native_release)

        snap_release = snap_workflow.split("  attach-release:", 1)[1]
        self.assertIn("--output release-assets/SHA256SUMS.snap", snap_release)
        self.assertIn("release-assets/SHA256SUMS.snap", snap_release)
        self.assertIn("Duplicate Snap release asset name", snap_release)
        snap_attachment = snap_release.split(
            "- name: Attach snaps to GitHub Release", 1
        )[1]
        self.assertIn("release-assets/nvbroadcast_*.snap", snap_attachment)
        self.assertNotIn("release-assets/*.snap", snap_attachment)
        self.assertEqual(
            snap_attachment.count("release-assets/SHA256SUMS.snap"),
            1,
        )

        self.assertIn("docs/RELEASE_VERIFICATION.md", readme)
        self.assertIn("gh attestation verify", verification)
        self.assertIn("--signer-workflow", verification)
        self.assertIn('--source-ref "refs/tags/$TAG"', verification)
        self.assertIn('--source-digest "$SOURCE_COMMIT"', verification)
        self.assertIn("--deny-self-hosted-runners", verification)
        self.assertIn("not yet hermetic or independently reproducible", verification)

    def test_snap_store_actions_are_pinned_to_the_release_tag(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        build_job = workflow.split("  build-snap:", 1)[1].split(
            "  attach-release:", 1
        )[0]
        attach_job = workflow.split("  attach-release:", 1)[1]
        release_target = build_job.split(
            "- name: Resolve release target", 1
        )[1].split("- name: Build snap", 1)[0]

        self.assertIn("ref: ${{ github.ref }}", build_job)
        self.assertIn("steps.store-target.outputs.action != ''", release_target)
        self.assertIn('RELEASE_TAG=""', release_target)
        self.assertIn('if [ -n "$DISPATCH_RELEASE_TAG" ]', release_target)
        self.assertIn('[[ "$GITHUB_REF" == refs/tags/v* ]]', release_target)
        self.assertIn(
            "release_tag is required for a Store action unless workflow_dispatch runs from the release tag",
            release_target,
        )
        self.assertIn('EXPECTED_REF="refs/tags/$RELEASE_TAG"', build_job)
        self.assertIn('if [ "$GITHUB_REF" != "$EXPECTED_REF" ]', build_job)
        self.assertIn("Snap Store actions must run from $EXPECTED_REF", build_job)
        self.assertIn('TAG_COMMIT="$(git rev-parse "$RELEASE_TAG^{commit}")"', build_job)
        self.assertIn('CHECKED_OUT_COMMIT="$(git rev-parse "HEAD^{commit}")"', build_job)
        self.assertIn('"$CHECKED_OUT_COMMIT" != "$GITHUB_SHA"', build_job)
        self.assertLess(
            build_job.index("- name: Resolve release target"),
            build_job.index("- name: Build snap"),
        )
        self.assertIn("- name: Checkout release source", attach_job)
        self.assertIn("Release attachment must run from $EXPECTED_REF", attach_job)
        self.assertIn("ref: ${{ github.sha }}", attach_job)

    def test_pull_request_checks_are_read_only_and_hardware_independent(self):
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "pr-checks.yml"
        ).read_text()

        self.assertIn("pull_request:", workflow)
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("pip_audit . --skip-editable", workflow)
        self.assertIn("bandit -q -r src -ll", workflow)
        self.assertIn("--ignore=tests/test_integration.py", workflow)
        self.assertIn("scripts/release_smoke.py", workflow)
        self.assertIn('- "3.11"', workflow)
        self.assertIn('- "3.14"', workflow)
        self.assertNotIn("secrets.", workflow)

    def test_snap_build_does_not_receive_release_write_permission(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        build_job = workflow.split("  build-snap:", 1)[1].split(
            "  attach-release:", 1
        )[0]
        attach_job = workflow.split("  attach-release:", 1)[1]

        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertNotIn("action-gh-release", build_job)
        self.assertIn("snapcraft upload-metadata", build_job)
        self.assertIn("matrix.arch == 'arm64'", build_job)
        self.assertIn("inputs.candidate", build_job)
        self.assertIn("inputs.review", build_job)
        self.assertIn("steps.store-target.outputs.action == 'upload'", build_job)
        self.assertIn('snapcraft upload "${{ steps.snapcraft.outputs.snap }}"', build_job)
        self.assertIn("release: ${{ steps.store-target.outputs.channel }}", build_job)
        self.assertIn("steps.store-target.outputs.channel == 'stable'", build_job)
        self.assertIn("secrets.SNAP_TOKEN", build_job)
        self.assertIn("secrets.SNAP_CANDIDATE_TOKEN", build_job)
        review_step = build_job.split(
            "- name: Upload to Snap Store for review", 1
        )[1].split("- name: Publish to Snap Store", 1)[0]
        publish_step = build_job.split(
            "- name: Publish to Snap Store", 1
        )[1].split("- name: Update Snap Store metadata", 1)[0]
        self.assertIn("secrets.SNAP_CANDIDATE_TOKEN", review_step)
        self.assertNotIn("secrets.SNAP_TOKEN", review_step)
        self.assertIn("timeout-minutes: 60", review_step)
        self.assertIn("timeout-minutes: 60", publish_step)
        self.assertIn("permissions:\n      contents: write", attach_job)
        self.assertIn("action-gh-release", attach_job)
        self.assertNotIn("inputs.candidate", attach_job)
        self.assertNotIn("inputs.review", attach_job)

    def test_snap_tag_build_does_not_auto_publish_to_store(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        website = (REPO_ROOT / "docs" / "index.html").read_text()
        store_target = workflow.split("- name: Resolve Snap Store target", 1)[1].split(
            "- name: Resolve release target", 1
        )[0]

        self.assertIn('if [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]', store_target)
        self.assertNotIn("refs/tags", store_target)
        self.assertIn("startsWith(github.ref, 'refs/tags/v')", workflow)
        self.assertNotIn("published from Git tags", website)
        self.assertNotIn("Tag pushes publish", website)
        self.assertIn("released separately after artifact inspection", website)

    def test_release_workflows_reject_version_mismatched_tags(self):
        build_workflow = (REPO_ROOT / ".github" / "workflows" / "build-packages.yml").read_text()
        snap_workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()

        self.assertIn("Tag $GITHUB_REF_NAME does not match package version", build_workflow)
        self.assertIn("Release tag $RELEASE_TAG does not match Snap version", snap_workflow)

    def test_tag_artifacts_remain_draft_until_inspected(self):
        build_workflow = (REPO_ROOT / ".github" / "workflows" / "build-packages.yml").read_text()
        snap_workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        package_release = build_workflow.split("- name: Create GitHub Release", 1)[1]
        snap_release = snap_workflow.split("- name: Attach snaps to GitHub Release", 1)[1]

        self.assertIn("draft: true", package_release)
        self.assertIn("draft: true", snap_release)
        self.assertNotIn("draft: ${{", snap_release)

    def test_release_workflow_requires_rpm_and_installs_linux_dependencies(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "build-packages.yml").read_text()
        rpm_step = workflow.split("- name: Build .rpm package", 1)[1].split(
            "- name: Upload Linux artifacts", 1
        )[0]
        linux_test_job = workflow.split("  test-linux:", 1)[1].split(
            "  test-python:", 1
        )[0]

        self.assertNotIn("continue-on-error", rpm_step)
        self.assertIn('python -m pip install ".[dev,cpu]"', workflow)
        self.assertIn("python -m pip check", workflow)
        self.assertIn('"pip-audit>=2.9" "bandit>=1.8"', linux_test_job)
        self.assertIn(
            "python -m pip_audit . --skip-editable --progress-spinner off",
            linux_test_job,
        )
        self.assertNotIn("python -m pip_audit --skip-editable", linux_test_job)
        self.assertIn("python -m bandit -q -r src -ll", linux_test_job)
        self.assertIn("Install Linux desktop test dependencies", linux_test_job)
        self.assertIn("gir1.2-gstreamer-1.0", linux_test_job)
        self.assertIn(
            "/usr/bin/python3 -m venv --system-site-packages .venv",
            linux_test_job,
        )

    def test_release_gates_run_video_regressions(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "build-packages.yml").read_text()
        release_smoke = (REPO_ROOT / "scripts" / "release_smoke.py").read_text()

        self.assertIn("tests/test_tensorrt_rvm.py", workflow)
        for module in (
            "tests.test_tensorrt_rvm",
            "tests.test_gpu_frame_path",
            "tests.test_video_pipeline",
            "tests.test_blur_controls",
            "tests.test_app_gpu_frame_policy",
            "tests.test_app_vcam_policy",
            "tests.test_vcam_monitor",
            "tests.test_macos_camera",
        ):
            self.assertIn(f'"{module}"', release_smoke)

    def test_release_workflow_requires_every_built_package(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "build-packages.yml").read_text()
        release_job = workflow.split("  release:", 1)[1]

        self.assertIn("artifacts/linux-packages/deb/*.deb", release_job)
        self.assertIn("artifacts/linux-packages/rpm/*.rpm", release_job)
        self.assertIn("artifacts/macos-packages/*.pkg", release_job)
        self.assertNotIn("artifacts/macos-packages/pkg/*.pkg", release_job)
        self.assertIn("fail_on_unmatched_files: true", release_job)

    def test_macos_ci_installs_the_actual_project_dependency_set(self):
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "build-packages.yml"
        ).read_text()
        macos_job = workflow.split("  test-macos:", 1)[1].split(
            "  test-linux:", 1
        )[0]

        self.assertIn('python3 -m pip install ".[dev,cpu]"', macos_job)
        self.assertIn("python3 -m pip check", macos_job)
        self.assertIn("python3 -m pip_audit --skip-editable", macos_job)
        self.assertNotIn("--dry-run", macos_job)
        self.assertIn("macos-15", macos_job)
        self.assertNotIn("macos-15-intel", macos_job)
        for version in ("'3.11'", "'3.12'", "'3.13'"):
            self.assertIn(version, macos_job)

        build_job = workflow.split("  build-macos:", 1)[1].split(
            "  test-macos:", 1
        )[0]
        self.assertNotIn("NVBroadcastExtension.systemextension/**", build_job)
        self.assertIn('hostArchitectures="arm64"', build_job)

    def test_readme_documents_cuda_extra_for_source_gpu_installs(self):
        readme = (REPO_ROOT / "README.md").read_text()
        self.assertIn('pip install -e ".[cuda]"', readme)
        self.assertIn('./install.sh --runtime cuda', readme)
        self.assertIn('pip install -e ".[cpu]"', readme)
        self.assertIn("Never overlay `.[cuda]`", readme)
        self.assertIn(
            'pip install "cupy-cuda12x>=14.1.1,<15" '
            "nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12",
            readme,
        )
        self.assertIn("CUDAExecutionProvider", readme)

    def test_cuda_extra_contains_onnxruntime_gpu_provider(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        self.assertIn("cuda = [", pyproject)
        base_dependencies = pyproject.split("dependencies = [", 1)[1].split(
            "[project.urls]", 1
        )[0]
        self.assertNotIn("onnxruntime", base_dependencies)
        self.assertIn("cpu = [", pyproject)
        self.assertIn('"cupy-cuda12x>=14.1.1,<15"', pyproject)
        self.assertIn('"cupy-cuda12x>=14.1.1,<15"', snapcraft)
        self.assertIn('export CUDA_PATH="$CUDA_RUNTIME"', snapcraft)
        self.assertIn('"onnxruntime-gpu==1.24.4"', pyproject)
        self.assertIn('"nvidia-nvimgcodec-cu12"', pyproject)
        self.assertIn('"nvidia-nvjpeg-cu12"', pyproject)
        self.assertIn(
            '"onnxruntime>=1.24.4,<1.25; sys_platform != \'darwin\'"',
            pyproject,
        )
        self.assertNotIn('"pycuda>=2024.1"', pyproject)
        self.assertNotIn('"nvidia-cusparse-cu12"', pyproject)
        self.assertNotIn('"nvidia-cusolver-cu12"', pyproject)
        self.assertNotIn("nvidia-nvimgcodec-cu12", snapcraft)
        self.assertNotIn("nvidia-nvjpeg-cu12", snapcraft)
        self.assertIn("intentionally uses GStreamer's CPU MJPEG decoder", readme)

    def test_linux_package_postinstalls_choose_one_runtime_before_resolution(self):
        deb_postinst = (REPO_ROOT / "packaging" / "debian" / "postinst").read_text()
        rpm_spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        rpm_postinst = rpm_spec.split("%post", 1)[1].split("%preun", 1)[0]
        for postinst in (deb_postinst, rpm_postinst):
            self.assertIn('RUNTIME_VARIANT="cpu"', postinst)
            self.assertIn('RUNTIME_VARIANT="cuda"', postinst)
            self.assertIn('rm -rf --', postinst)
            self.assertIn('install_runtime_variant.py', postinst)
            self.assertIn('--meeting-backends faster', postinst)
            self.assertIn('recreating clean CPU environment', postinst)
            self.assertNotIn('[cuda]"', postinst)
        self.assertIn("pkill -f", rpm_spec.split("%pre", 1)[1].split("%post", 1)[0])
        self.assertNotIn('pkill -f "nvbroadcast"', rpm_spec)

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
        self.assertIn('MACOS_VIRTUAL_CAM_LABEL = "OBS Virtual Camera"', constants)
        self.assertIn('else MACOS_VIRTUAL_CAM_LABEL', constants)
        self.assertIn('card_label="NVbroadcast"', readme)
        self.assertIn(
            'select **"NVbroadcast"** on Linux or **"OBS Virtual Camera"** on macOS',
            readme,
        )
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
        pipeline = (REPO_ROOT / "src" / "nvbroadcast" / "video" / "pipeline.py").read_text()
        self.assertIn("pyvirtualcam.PixelFormat.BGR", pipeline)
        self.assertNotIn("NVBROADCAST_ALLOW_OBS_VCAM_FALLBACK", pipeline)

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
        self.assertIn("install_runtime_variant.py", postinst)
        self.assertIn("--meeting-backends faster", postinst)
        self.assertNotIn("openai-whisper", postinst)

    def test_rpm_postinst_installs_meeting_runtime(self):
        spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        postinst = spec.split("%post", 1)[1].split("%preun", 1)[0]
        self.assertIn("install_runtime_variant.py", postinst)
        self.assertIn("--meeting-backends faster", postinst)
        self.assertNotIn("openai-whisper", postinst)

    def test_rpm_uses_fedora_cairo_package_name(self):
        spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        self.assertIn("Requires:       python3-cairo", spec)
        self.assertNotIn("python3-gobject-cairo", spec)

    def test_native_package_builders_preserve_runtime_installer_mode(self):
        script = (REPO_ROOT / "build-packages.sh").read_text()
        deb_builder = script.split("build_deb() {", 1)[1].split(
            "build_rpm() {", 1
        )[0]
        self.assertIn(
            '"$PKG_DIR/opt/nvbroadcast/scripts/install_runtime_variant.py"',
            deb_builder.split("# Normalize the payload", 1)[1],
        )

        spec = (REPO_ROOT / "packaging" / "rpm" / "nvbroadcast.spec").read_text()
        self.assertIn(
            "chmod 755 "
            "%{buildroot}/opt/nvbroadcast/scripts/install_runtime_variant.py",
            spec,
        )

    def test_macos_postinstall_installs_meeting_runtime_in_two_steps(self):
        script = (REPO_ROOT / "build-packages.sh").read_text()
        pkg_builder = script.split("build_pkg() {", 1)[1]
        self.assertIn("install_runtime_variant.py", script)
        self.assertIn("--variant cpu --meeting-backends faster", script)
        self.assertIn('rm -rf -- "$INSTALL_DIR/.venv"', script)
        self.assertIn('pkill -f "^${INSTALL_DIR}/.venv/bin/python -m nvbroadcast', script)
        self.assertIn(
            'mkdir -p "$INSTALL_ROOT/opt/nvbroadcast/scripts"', pkg_builder
        )
        self.assertIn(
            "install -m 755 scripts/install_runtime_variant.py", pkg_builder
        )
        self.assertNotIn(
            "install -Dm 755 scripts/install_runtime_variant.py", pkg_builder
        )

        workflow = (
            REPO_ROOT / ".github" / "workflows" / "build-packages.yml"
        ).read_text()
        macos_builder = workflow.split("  build-macos:", 1)[1].split(
            "  test-macos:", 1
        )[0]
        self.assertIn("run: bash build-packages.sh pkg", macos_builder)
        self.assertIn(
            "opt/nvbroadcast/scripts/install_runtime_variant.py", macos_builder
        )
        self.assertIn('test -x "$runtime_installer"', macos_builder)

    def test_macos_source_installer_guards_openai_whisper(self):
        script = (REPO_ROOT / "install_macos.sh").read_text()
        self.assertIn("install_runtime_variant.py", script)
        self.assertIn("--variant cpu --meeting-backends faster", script)
        self.assertIn('rm -rf -- "$INSTALL_DIR/venv"', script)
        self.assertIn('pkill -f "^${INSTALL_DIR}/venv/bin/python -m nvbroadcast', script)
        self.assertIn("sys.version_info < (3, 14)", script)
        self.assertIn('pip install -q "openai-whisper>=20231117"', script)

    def test_snap_package_bundles_lighter_meeting_runtime(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        build_workflow = (REPO_ROOT / ".github" / "workflows" / "build-packages.yml").read_text()
        self.assertIn("        faster-whisper==1.2.1", snapcraft)
        self.assertIn("- ctranslate2", snapcraft)
        self.assertIn("- httpx", snapcraft)
        self.assertIn("- av", snapcraft)
        self.assertIn("- sympy", snapcraft)
        self.assertNotIn("- openai-whisper", snapcraft)
        self.assertIn("onnxruntime==1.24.4", snapcraft)
        self.assertIn("onnxruntime-gpu==1.24.4", snapcraft)
        python_packages = snapcraft.split("python-packages:", 1)[1].split(
            "stage-packages:", 1
        )[0]
        self.assertNotIn("onnxruntime", python_packages)
        self.assertNotIn("faster-whisper", python_packages)
        self.assertIn("Installing sole amd64 CUDA runtime owner", snapcraft)
        self.assertIn("Installing sole arm64 CPU runtime owner", snapcraft)
        self.assertIn("arm64 Snap build stays portable and CPU-safe", snapcraft)
        cuda_install = snapcraft.split(
            "Installing sole amd64 CUDA runtime owner into Snap", 1
        )[1].split("# nvImageCodec", 1)[0]
        self.assertIn("- protobuf>=6.33.5,<7", snapcraft)
        self.assertNotIn('"protobuf>=6.33.5,<7"', cuda_install)
        self.assertIn("onnxruntime==1.24.4", build_workflow)
        arm64_wheel_check = build_workflow.split(
            "- name: Validate arm64 Python wheel availability", 1
        )[1].split("- name: Install Linux project dependencies", 1)[0]
        self.assertIn("protobuf>=6.33.5,<7", arm64_wheel_check)
        self.assertIn("opencv-contrib-python>=4.8.1.78,<5", arm64_wheel_check)
        for package in (
            "pyrnnoise",
            "av>=16,<17",
            "faster-whisper==1.2.1",
            "ctranslate2",
            "huggingface-hub",
            "httpx",
            "tokenizers",
            "soundfile",
            "tqdm",
        ):
            self.assertIn(package, build_workflow)

    def test_snap_transcriber_uses_private_shared_memory(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        self.assertIn(
            "plugs:\n"
            "  shared-memory:\n"
            "    interface: shared-memory\n"
            "    private: true\n",
            snapcraft,
        )
        app = snapcraft.split("apps:\n", 1)[1].split("\nparts:", 1)[0]
        self.assertIn("      - shared-memory\n", app)

    def test_snap_excludes_runtime_pip_installer(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()

        self.assertNotRegex(snapcraft, r"(?m)^\s+- pip(?:[<>=].*)?$")
        self.assertIn('"$CRAFT_PART_INSTALL/bin/pip"', snapcraft)
        self.assertIn("-name 'pip-*.dist-info'", snapcraft)
        self.assertIn("Verify Snap excludes runtime pip", workflow)
        self.assertIn("must not contain a runtime pip installer", workflow)

    def test_snap_validates_runtime_dependency_closure(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        workflow = (REPO_ROOT / ".github" / "workflows" / "snap.yml").read_text()
        cuda_install = snapcraft.split(
            'echo "Installing sole amd64 CUDA runtime owner into Snap..."', 1
        )[1].split("# nvImageCodec", 1)[0]

        self.assertIn("- packaging>=26.0", snapcraft)
        self.assertIn("- setuptools>=83.0.0", snapcraft)
        self.assertIn("--no-deps", cuda_install)
        self.assertIn('"cuda-pathfinder>=1.3.4,<2"', cuda_install)
        self.assertNotRegex(
            cuda_install,
            r'(?m)^\s+"?(?:numpy|packaging|protobuf)(?:[<>=][^"]*)?"?\s+\\$',
        )
        self.assertIn("Verify Snap runtime dependency closure", workflow)
        self.assertIn("scripts/validate_snap_runtime.py", workflow)
        self.assertIn('IMPORT_PROBES = ("packaging", "setuptools", "onnxruntime")', (REPO_ROOT / "scripts" / "validate_snap_runtime.py").read_text())

    def test_snap_prioritizes_app_owned_python_packages(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        pythonpath = re.search(
            r"(?m)^\s+PYTHONPATH:\s*(\S+)$",
            snapcraft,
        )

        self.assertIsNotNone(pythonpath)
        self.assertEqual(
            pythonpath.group(1).split(":", 1)[0],
            "$SNAP/lib/python3.12/site-packages",
        )

    def test_snap_relocates_python_venv_for_strict_runtime(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        validator = (
            REPO_ROOT / "scripts" / "validate_snap_runtime.py"
        ).read_text()

        self.assertIn('PYVENV_CFG="$CRAFT_PART_INSTALL/pyvenv.cfg"', snapcraft)
        self.assertIn("'s|^home = .*|home = /usr/bin|'", snapcraft)
        self.assertIn("'/^executable = /d'", snapcraft)
        self.assertIn("'/^command = /d'", snapcraft)
        self.assertIn("python_runtime_problems", validator)
        self.assertIn("pyvenv.cfg contains build-only path", validator)

    def test_snap_uses_gnome_content_runtime_without_shadowing_it(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        validator = (
            REPO_ROOT / "scripts" / "validate_snap_runtime.py"
        ).read_text()

        for package in (
            "python3-gi",
            "gir1.2-gtk-4.0",
            "gir1.2-adw-1",
            "gstreamer1.0-plugins-base",
            "libayatana-appindicator3-1",
        ):
            self.assertNotIn(f"      - {package}\n", snapcraft)
        self.assertIn(
            'PLATFORM_LIB="$SNAP/gnome-platform/usr/lib/$TRIPLET"', snapcraft
        )
        self.assertIn("platform_shadow_problems", validator)
        self.assertIn("--timeout 120", snapcraft)
        self.assertIn("--retries 5", snapcraft)

    def test_snap_packages_a_registered_desktop_launcher(self):
        snapcraft = (REPO_ROOT / "snap" / "snapcraft.yaml").read_text()
        validator = (
            REPO_ROOT / "scripts" / "validate_snap_runtime.py"
        ).read_text()

        self.assertIn(
            "desktop: share/applications/com.doczeus.NVBroadcast.desktop",
            snapcraft,
        )
        self.assertIn("desktop_launcher_problems", validator)
        self.assertIn("meta/gui/nvbroadcast.desktop", validator)

    def test_packaged_backgrounds_include_bundled_default(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("data/backgrounds/studio_bg.png", pyproject)

    def test_python_meeting_support_extra_does_not_resolve_runtime_owner(self):
        metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        installer = (REPO_ROOT / "scripts" / "install_runtime_variant.py").read_text()
        support = metadata["project"]["optional-dependencies"]["meeting-support"]
        self.assertIn("ctranslate2", support)
        self.assertIn("httpx", support)
        self.assertNotIn("faster-whisper", support)
        self.assertFalse(any(item.startswith("openai-whisper") for item in support))
        self.assertIn('extras.append("meeting-support")', installer)
        self.assertIn('extras.append("meeting")', installer)
        self.assertNotIn("MEETING_SUPPORT =", installer)

    def test_meeting_extra_preserves_openai_whisper_compatibility(self):
        metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        readme = (REPO_ROOT / "README.md").read_text()
        extras = metadata["project"]["optional-dependencies"]
        requirement = 'openai-whisper>=20231117; python_version < "3.14"'

        self.assertIn(requirement, extras["meeting"])
        for extra in ("cpu", "cuda"):
            self.assertFalse(
                any(item.startswith("openai-whisper") for item in extras[extra])
            )
        self.assertFalse(
            any(item.startswith("faster-whisper") for item in extras["meeting"])
        )
        self.assertIn("`.[cpu,meeting]`", readme)
        self.assertIn("`.[cuda,meeting]`", readme)
        self.assertIn("./install.sh --runtime auto --with-meeting", readme)

    def test_macos_packages_require_the_runtime_wheel_baseline(self):
        installer = (REPO_ROOT / "install_macos.sh").read_text()
        build_script = (REPO_ROOT / "build-packages.sh").read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        website = (REPO_ROOT / "docs" / "index.html").read_text()

        self.assertIn('[[ "$MACOS_VER" -lt 13 ]]', installer)
        self.assertIn("macOS 13 (Ventura) or newer", installer)
        self.assertIn('[[ "$MACOS_ARCH" != "arm64" ]]', installer)
        self.assertIn("supports Apple Silicon Macs only", installer)
        self.assertIn(
            "for p in python3.13 python3.12 python3.11 python3; do", installer
        )
        self.assertIn('"$minor" -le 13', installer)
        self.assertIn(
            "for p in python3.13 python3.12 python3.11 python3; do",
            build_script,
        )
        self.assertIn('[ "$minor" -le 13 ]', build_script)
        self.assertIn('hostArchitectures="arm64"', build_script)
        self.assertIn('<os-version min="13.0"/>', build_script)
        self.assertIn("Apple Silicon Mac with macOS 13+", readme)
        self.assertIn("Python 3.11-3.13", readme)
        self.assertIn("macOS 13 Ventura or newer", website)
        self.assertIn("Apple Silicon (M1+) required", website)

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
        self.assertIn("candidate:", workflow)
        self.assertIn("review:", workflow)
        self.assertIn("release_tag:", workflow)
        self.assertIn("id: store-target", workflow)
        self.assertIn("id: release-target", workflow)
        self.assertIn('CHANNEL="candidate"', workflow)
        self.assertIn('ACTION="upload"', workflow)
        self.assertIn("publish, candidate, and review are mutually exclusive", workflow)
        self.assertIn(
            "release_tag is required for a Store action unless workflow_dispatch runs from the release tag",
            workflow,
        )
        self.assertIn("tag_name: ${{ steps.release-target.outputs.tag }}", workflow)

    def test_snap_promotion_workflow_validates_exact_revisions_and_rolls_back(self):
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "snap-promote.yml"
        ).read_text()

        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)
        permissions = workflow.split("permissions:", 1)[1].split("concurrency:", 1)[0]
        self.assertIn("attestations: read", permissions)
        self.assertIn("contents: read", permissions)
        self.assertIn("group: snap-store-promotion", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("amd64_revision:", workflow)
        self.assertIn("arm64_revision:", workflow)
        self.assertIn("          - edge", workflow)
        self.assertIn("validate_revision amd64", workflow)
        self.assertIn("validate_revision arm64", workflow)
        self.assertIn("rollback_candidate", workflow)
        self.assertIn('snapcraft release "$SNAP_NAME" "$ARM64_REVISION" candidate', workflow)
        self.assertIn('snapcraft release "$SNAP_NAME" "$AMD64_REVISION" candidate', workflow)
        self.assertIn("secrets.SNAP_CANDIDATE_TOKEN", workflow)
        self.assertIn("secrets.SNAP_TOKEN", workflow)
        self.assertIn("--from-channel=candidate", workflow)
        self.assertIn('--to-channel="$TARGET_CHANNEL"', workflow)
        self.assertIn("inputs.operation == 'edge' || inputs.operation == 'stable'", workflow)
        self.assertIn(
            "(inputs.operation == 'edge' && secrets.SNAP_EDGE_TOKEN) || "
            "(inputs.operation == 'stable' && secrets.SNAP_TOKEN)",
            workflow,
        )
        self.assertNotIn(
            "secrets.SNAP_EDGE_TOKEN || secrets.SNAP_TOKEN",
            workflow,
        )
        self.assertIn("A channel-scoped Snap Store credential is required", workflow)
        self.assertIn('current_revision "$AMD64_TABLE" "$TARGET_CHANNEL"', workflow)
        self.assertIn('current_revision "$ARM64_TABLE" "$TARGET_CHANNEL"', workflow)
        self.assertIn("channel verification does not match", workflow)
        self.assertIn('snap download "$SNAP_NAME"', workflow)
        self.assertIn("download_and_verify_candidate amd64", workflow)
        self.assertIn("download_and_verify_candidate arm64", workflow)
        self.assertIn("gh attestation verify", workflow)
        self.assertIn("--signer-workflow", workflow)
        self.assertIn('--source-ref "refs/tags/$RELEASE_TAG"', workflow)
        self.assertIn('--source-digest "$RELEASE_COMMIT"', workflow)
        self.assertIn("--deny-self-hosted-runners", workflow)
        verification_function = workflow.split(
            "download_and_verify_candidate()", 1
        )[1].split("download_and_verify_candidate amd64", 1)[0]
        self.assertIn('if ! snap download "$SNAP_NAME"', verification_function)
        self.assertIn("if ! gh attestation verify", verification_function)
        self.assertIn("Attestation verification failed", verification_function)
        self.assertLess(
            workflow.index("gh attestation verify"),
            workflow.index('snapcraft promote "$SNAP_NAME"'),
        )
        self.assertIn(
            'snapcraft upload-metadata "$CANDIDATE_ARM64_SNAP" --force',
            workflow,
        )
        self.assertIn('if [ "$TARGET_CHANNEL" = "stable" ]; then', workflow)
        self.assertNotIn('if [ -n "${{ inputs.', workflow)

    def test_about_window_separates_authorship_sponsors_and_contributors(self):
        window = (REPO_ROOT / "src" / "nvbroadcast" / "ui" / "window.py").read_text()
        self.assertIn('developers=["Code by doczeus https://github.com/Hkshoonya"]', window)
        self.assertIn('add_credit_section("Project Sponsors", _APP_SPONSORS)', window)
        self.assertIn("Mattsky — GitHub Sponsor https://github.com/Mattsky", window)
        self.assertIn("from nvbroadcast.contributors import app_contributor_credits", window)
        self.assertIn('"Contributions to App",\n                app_contributor_credits(),', window)
        self.assertNotIn("_APP_CONTRIBUTORS", window)
        self.assertNotIn('add_credit_section("Backers & Supporters"', window)


if __name__ == "__main__":
    unittest.main()

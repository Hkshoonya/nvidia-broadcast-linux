import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPO_ROOT / "scripts" / "render_native_upgrade_helper.py"
TEMPLATE_PATH = REPO_ROOT / "scripts" / "native_package_upgrade.sh.in"
SPEC = importlib.util.spec_from_file_location(
    "render_native_upgrade_helper", RENDERER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


LEGACY_DEBIAN_PRERMS = (
    """#!/bin/bash
# NV Broadcast pre-removal script
set -e

# Stop any running instances
pkill -f "nvbroadcast" 2>/dev/null || true

# Disable systemd service
if systemctl --user is-active nvbroadcast-vcam.service &>/dev/null; then
    systemctl --user stop nvbroadcast-vcam.service 2>/dev/null || true
    systemctl --user disable nvbroadcast-vcam.service 2>/dev/null || true
fi

#DEBHELPER#
""",
    """#!/bin/bash
# NV Broadcast pre-removal script
set -e

case "$1" in
    upgrade|deconfigure)
        # On package upgrade, stop the running app but preserve service state.
        pkill -f "nvbroadcast" 2>/dev/null || true
        if systemctl --user is-active nvbroadcast-vcam.service &>/dev/null; then
            systemctl --user stop nvbroadcast-vcam.service 2>/dev/null || true
        fi
        ;;
    remove|purge)
        pkill -f "nvbroadcast" 2>/dev/null || true
        if systemctl --user is-active nvbroadcast-vcam.service &>/dev/null; then
            systemctl --user stop nvbroadcast-vcam.service 2>/dev/null || true
        fi
        if systemctl --user is-enabled nvbroadcast-vcam.service &>/dev/null; then
            systemctl --user disable nvbroadcast-vcam.service 2>/dev/null || true
        fi
        ;;
esac

#DEBHELPER#
""",
    """#!/bin/bash
# NV Broadcast pre-removal script
set -e

case "$1" in
    upgrade|deconfigure)
        # On package upgrade, stop the running app but preserve service state.
        pkill -f "nvbroadcast" 2>/dev/null || true
        if systemctl --user is-active nvbroadcast-vcam.service &>/dev/null; then
            systemctl --user stop nvbroadcast-vcam.service 2>/dev/null || true
        fi
        ;;
    remove|purge)
        pkill -f "nvbroadcast" 2>/dev/null || true
        if systemctl --user is-active nvbroadcast-vcam.service &>/dev/null; then
            systemctl --user stop nvbroadcast-vcam.service 2>/dev/null || true
        fi
        if systemctl --user is-enabled nvbroadcast-vcam.service &>/dev/null; then
            systemctl --user disable nvbroadcast-vcam.service 2>/dev/null || true
        fi
        ;;
esac

""",
)

LEGACY_DEBIAN_HASHES = (
    "a20da1ed2a66e80ca46d21738c8329b046642d0cf6d42dc8c25b7ee8ed996c36",
    "c4ebf6d1ccc960870f8f69c62a5bee480f31b89e891c1fc45b462fc1eccee51c",
    "ef1e8ed9cde08f74ab6ed5f1db2af09137a69b305e2b455727f7658fa0e171c4",
)


class NativePackageUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.deb = self.root / "nvbroadcast_9.8.7-1_all.deb"
        self.rpm = self.root / "nvbroadcast-9.8.7-1.noarch.rpm"
        self.output = self.root / "nvbroadcast-native-upgrade"
        self.deb.write_bytes(b"exact deb payload")
        self.rpm.write_bytes(b"exact rpm payload")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def render(self) -> str:
        return RENDERER.render_helper(
            TEMPLATE_PATH,
            self.deb,
            self.rpm,
            "9.8.7",
            "1",
            self.output,
        )

    def run_patch(self, prerm: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                "bash",
                "-c",
                'source "$1"; patch_legacy_debian_prerm "$2"',
                "bash",
                str(self.output),
                str(prerm),
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_renderer_binds_exact_artifact_hashes_and_mode(self):
        content = self.render()

        self.assertEqual(content, self.output.read_text(encoding="ascii"))
        self.assertIn("readonly TARGET_VERSION='9.8.7'", content)
        self.assertIn(hashlib.sha256(self.deb.read_bytes()).hexdigest(), content)
        self.assertIn(hashlib.sha256(self.rpm.read_bytes()).hexdigest(), content)
        self.assertNotRegex(content, r"@(TARGET_|DEB_SHA256|RPM_SHA256)")
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o755)
        syntax = subprocess.run(
            ("bash", "-n", str(self.output)),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_renderer_rejects_symlink_artifact_without_replacing_output(self):
        target = self.root / "target.deb"
        target.write_bytes(b"payload")
        self.deb.unlink()
        self.deb.symlink_to(target)
        self.output.write_text("existing\n", encoding="ascii")

        with self.assertRaisesRegex(RENDERER.RenderError, "cannot open"):
            self.render()

        self.assertEqual(self.output.read_text(encoding="ascii"), "existing\n")

    def test_renderer_rejects_one_path_for_both_package_formats(self):
        self.rpm.unlink()
        self.rpm = self.deb

        with self.assertRaisesRegex(
            RENDERER.RenderError, "unexpected release artifact"
        ):
            self.render()

    def test_all_public_legacy_debian_scripts_are_patched_exactly(self):
        self.render()
        for index, legacy in enumerate(LEGACY_DEBIAN_PRERMS):
            with self.subTest(index=index):
                self.assertEqual(
                    hashlib.sha256(legacy.encode("ascii")).hexdigest(),
                    LEGACY_DEBIAN_HASHES[index],
                )
                prerm = self.root / f"prerm-{index}"
                prerm.write_text(legacy, encoding="ascii")
                prerm.chmod(0o751)

                completed = self.run_patch(prerm)

                self.assertEqual(completed.returncode, 0, completed.stderr)
                patched = prerm.read_text(encoding="ascii")
                self.assertNotIn('pkill -f "nvbroadcast"', patched)
                self.assertIn(
                    "pkill -f '^/opt/nvbroadcast/\\.venv/bin/python -m "
                    "nvbroadcast(\\.vcam_service)?( |$)'",
                    patched,
                )
                self.assertEqual(prerm.stat().st_mode & 0o777, 0o751)

                repeated = self.run_patch(prerm)
                self.assertEqual(repeated.returncode, 0, repeated.stderr)
                self.assertEqual(prerm.read_text(encoding="ascii"), patched)

    def test_modified_legacy_debian_script_fails_closed(self):
        self.render()
        prerm = self.root / "modified-prerm"
        modified = LEGACY_DEBIAN_PRERMS[-1].replace(
            "# NV Broadcast pre-removal script",
            "# Locally modified pre-removal script",
        )
        prerm.write_text(modified, encoding="ascii")

        completed = self.run_patch(prerm)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unfamiliar; refusing to rewrite", completed.stderr)
        self.assertEqual(prerm.read_text(encoding="ascii"), modified)

    def test_requoted_legacy_debian_kill_fails_closed(self):
        self.render()
        prerm = self.root / "requote-prerm"
        modified = LEGACY_DEBIAN_PRERMS[-1].replace(
            'pkill -f "nvbroadcast"',
            "pkill -f 'nvbroadcast'",
        )
        prerm.write_text(modified, encoding="ascii")

        completed = self.run_patch(prerm)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unfamiliar; refusing to rewrite", completed.stderr)
        self.assertEqual(prerm.read_text(encoding="ascii"), modified)

    def test_locally_commented_safe_debian_kill_is_left_unchanged(self):
        self.render()
        prerm = self.root / "safe-prerm"
        safe = (
            LEGACY_DEBIAN_PRERMS[-1]
            .replace(
                'pkill -f "nvbroadcast" 2>/dev/null || true',
                "pkill -f '^/opt/nvbroadcast/\\.venv/bin/python -m "
                "nvbroadcast(\\.vcam_service)?( |$)' 2>/dev/null || true",
            )
            .replace(
                "# NV Broadcast pre-removal script",
                "# Locally documented NV Broadcast pre-removal script",
            )
        )
        prerm.write_text(safe, encoding="ascii")

        completed = self.run_patch(prerm)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(prerm.read_text(encoding="ascii"), safe)

    def test_rpm_migration_skips_only_the_exact_legacy_preun(self):
        content = self.render()
        self.assertIn("installed_preun=\"$(rpm -q --qf '%{PREUN}'", content)
        self.assertIn('[[ "$installed_preun" == "$LEGACY_RPM_PREUN" ]]', content)
        self.assertIn('requirements="$(rpm -qp --requires', content)
        self.assertIn('dnf --assumeyes install "${target_requirements[@]}"', content)
        self.assertIn('rpm --upgrade --test --nopreun "$STAGED_PACKAGE"', content)
        self.assertIn('rpm --upgrade --nopreun "$STAGED_PACKAGE"', content)
        self.assertNotIn("tsflags=nopreun", content)
        self.assertIn(
            'has_unfamiliar_nvbroadcast_kill <<< "$installed_preun"',
            content,
        )
        self.assertIn("installed RPM pre-uninstall script is unfamiliar", content)

    def test_interrupted_debian_install_has_a_bounded_recovery_path(self):
        content = self.render()
        self.assertIn('[[ "$status" == i* ]]', content)
        self.assertIn('if [[ "$status" == ii* ]]', content)
        self.assertIn('dpkg --unpack "$STAGED_PACKAGE"', content)
        self.assertIn("apt-get --fix-broken install --yes --no-remove", content)


if __name__ == "__main__":
    unittest.main()

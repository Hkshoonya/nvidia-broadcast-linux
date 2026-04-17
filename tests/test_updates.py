import os
import unittest
from unittest import mock

from nvbroadcast.core.config import AppConfig
from nvbroadcast.core.updates import (
    find_release_asset,
    is_newer_version,
    release_info_from_payload,
    resolve_update_target,
    should_check_for_updates,
)


class UpdateTests(unittest.TestCase):
    def test_version_comparison_handles_v_prefix(self):
        self.assertTrue(is_newer_version("1.0.3", "1.0.2"))
        self.assertFalse(is_newer_version("1.0.2", "1.0.2"))
        self.assertFalse(is_newer_version("1.0.1", "1.0.2"))

    def test_should_check_respects_interval(self):
        config = AppConfig()
        config.last_update_check = 1_000
        self.assertFalse(should_check_for_updates(config, now=1_100, interval_seconds=200))
        self.assertTrue(should_check_for_updates(config, now=1_300, interval_seconds=200))

    def test_should_check_respects_disable_flag(self):
        config = AppConfig(check_for_updates=False, last_update_check=0)
        self.assertFalse(should_check_for_updates(config, now=10_000))

    def test_release_payload_parsing(self):
        release = release_info_from_payload({
            "tag_name": "v1.0.2",
            "html_url": "https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v1.0.2",
            "published_at": "2026-03-27T00:00:00Z",
            "assets": [
                {
                    "name": "NVBroadcast-1.0.2-1.pkg",
                    "browser_download_url": "https://example.invalid/NVBroadcast-1.0.2-1.pkg",
                }
            ],
        })
        self.assertEqual(release.version, "1.0.2")
        self.assertEqual(release.tag_name, "v1.0.2")
        self.assertIn("/releases/tag/v1.0.2", release.html_url)
        self.assertEqual(find_release_asset(release, ".pkg").name, "NVBroadcast-1.0.2-1.pkg")

    def test_resolve_update_target_prefers_snap_store_inside_snap(self):
        release = release_info_from_payload({
            "tag_name": "v1.1.3",
            "html_url": "https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v1.1.3",
        })
        with mock.patch.dict(os.environ, {"SNAP": "/snap/nvbroadcast/current"}, clear=False):
            target = resolve_update_target(release)
        self.assertEqual(target.button_label, "Open Snap Update")
        self.assertIn("snapcraft.io/nvbroadcast", target.url)

    @mock.patch("sys.platform", "darwin")
    def test_resolve_update_target_prefers_pkg_on_macos(self):
        release = release_info_from_payload({
            "tag_name": "v1.1.3",
            "html_url": "https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v1.1.3",
            "assets": [
                {
                    "name": "NVBroadcast-1.1.3-1.pkg",
                    "browser_download_url": "https://example.invalid/NVBroadcast-1.1.3-1.pkg",
                }
            ],
        })
        with mock.patch.dict(os.environ, {}, clear=True):
            target = resolve_update_target(release)
        self.assertEqual(target.button_label, "Download macOS Update")
        self.assertTrue(target.url.endswith(".pkg"))


if __name__ == "__main__":
    unittest.main()

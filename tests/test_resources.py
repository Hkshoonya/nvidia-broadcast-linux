import unittest
from pathlib import Path
from unittest import mock

from nvbroadcast.core.resources import (
    DEFAULT_BACKGROUND,
    find_app_icon,
    find_app_icon_png,
    find_backgrounds_dir,
    find_bundled_backgrounds,
)


class ResourceLookupTests(unittest.TestCase):
    def test_bundled_backgrounds_include_default_first(self):
        backgrounds = find_bundled_backgrounds()

        self.assertTrue(backgrounds)
        self.assertEqual(backgrounds[0].name, DEFAULT_BACKGROUND)

    def test_flatpak_share_directory_is_checked_for_installed_assets(self):
        with mock.patch(
            "nvbroadcast.core.resources._existing", return_value=None
        ) as existing:
            find_app_icon()
            svg_candidates = existing.call_args.args[0]
            find_app_icon_png()
            png_candidates = existing.call_args.args[0]
            find_backgrounds_dir()
            background_candidates = existing.call_args.args[0]

        self.assertIn(
            Path("/app/share/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg"),
            svg_candidates,
        )
        self.assertIn(
            Path("/app/share/icons/hicolor/128x128/apps/com.doczeus.NVBroadcast.png"),
            png_candidates,
        )
        self.assertIn(
            Path("/app/share/nvbroadcast/backgrounds"), background_candidates
        )


if __name__ == "__main__":
    unittest.main()

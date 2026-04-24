import unittest

from nvbroadcast.core.resources import DEFAULT_BACKGROUND, find_bundled_backgrounds


class ResourceLookupTests(unittest.TestCase):
    def test_bundled_backgrounds_include_default_first(self):
        backgrounds = find_bundled_backgrounds()

        self.assertTrue(backgrounds)
        self.assertEqual(backgrounds[0].name, DEFAULT_BACKGROUND)


if __name__ == "__main__":
    unittest.main()

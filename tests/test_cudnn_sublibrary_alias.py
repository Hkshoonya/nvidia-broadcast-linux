"""cuDNN sub-library aliases keep versioned dlopen inside the wheel."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.core import platform


class CudnnSublibraryAliasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lib = Path(self.tmp.name)
        for name in ("libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_ops.so.9",
                     "libcudnn_engines_precompiled.so.9"):
            (self.lib / name).write_bytes(b"")

    def _alias(self, version=92801):
        library = SimpleNamespace(cudnnGetVersion=lambda: version)
        with mock.patch.object(platform.ctypes, "CDLL", return_value=library):
            return platform.alias_cudnn_sublibraries(self.lib)

    def test_versioned_names_point_at_the_wheel_files(self):
        created = self._alias()
        for stem in ("libcudnn", "libcudnn_graph", "libcudnn_ops",
                     "libcudnn_engines_precompiled"):
            for suffix in ("9.28.1", "9.28"):
                alias = self.lib / f"{stem}.so.{suffix}"
                self.assertTrue(alias.is_symlink(), alias)
                self.assertEqual(alias.readlink(), Path(f"{stem}.so.9"))
        self.assertEqual(len(created), 8)

    def test_repeat_and_existing_files_are_left_alone(self):
        (self.lib / "libcudnn_ops.so.9.28.1").write_bytes(b"real")
        self._alias()
        self.assertFalse((self.lib / "libcudnn_ops.so.9.28.1").is_symlink())
        self.assertEqual(self._alias(), [])

    def test_other_major_versions_are_ignored(self):
        self.assertEqual(self._alias(version=8907), [])
        self.assertEqual(sorted(p.name for p in self.lib.iterdir() if p.is_symlink()), [])

    def test_missing_library_is_a_no_op(self):
        with mock.patch.object(platform.ctypes, "CDLL", side_effect=OSError("missing")):
            self.assertEqual(platform.alias_cudnn_sublibraries(self.lib), [])

    def test_read_only_directory_is_tolerated(self):
        with mock.patch.object(Path, "symlink_to", side_effect=PermissionError("ro")):
            self.assertEqual(self._alias(), [])


if __name__ == "__main__":
    unittest.main()

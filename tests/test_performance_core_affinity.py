"""Hybrid-CPU affinity preference without touching the real scheduler."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nvbroadcast.core import platform


class PerformanceCoreAffinityTests(unittest.TestCase):
    def _sysfs(self, core="0-15\n", atom="16-31\n"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        if core is not None:
            (root / "cpu_core").mkdir()
            (root / "cpu_core" / "cpus").write_text(core)
        if atom is not None:
            (root / "cpu_atom").mkdir()
            (root / "cpu_atom" / "cpus").write_text(atom)
        return root

    def _run(self, root, allowed=frozenset(range(32)), env=None):
        with mock.patch.object(platform, "IS_LINUX", True), \
             mock.patch.dict(platform.os.environ, env or {}, clear=False), \
             mock.patch.object(platform.os, "sched_getaffinity", return_value=set(allowed)), \
             mock.patch.object(platform.os, "sched_setaffinity") as setter:
            if env is None:
                platform.os.environ.pop("NVBROADCAST_CPU_AFFINITY", None)
            return platform.prefer_performance_cores(root), setter

    def test_hybrid_cpu_pins_to_performance_cores(self):
        result, setter = self._run(self._sysfs())
        self.assertEqual(result, set(range(16)))
        setter.assert_called_once_with(0, set(range(16)))

    def test_respects_an_existing_narrower_mask(self):
        result, setter = self._run(self._sysfs(), allowed={2, 3, 20})
        self.assertEqual(result, {2, 3})
        setter.assert_called_once_with(0, {2, 3})

    def test_non_hybrid_or_disabled_is_left_alone(self):
        for root, env in (
            (self._sysfs(atom=None), None),
            (self._sysfs(core=None, atom=None), None),
            (self._sysfs(), {"NVBROADCAST_CPU_AFFINITY": "all"}),
        ):
            result, setter = self._run(root, env=env)
            self.assertIsNone(result)
            setter.assert_not_called()

    def test_mask_without_performance_cores_is_not_changed(self):
        result, setter = self._run(self._sysfs(), allowed={20, 21})
        self.assertIsNone(result)
        setter.assert_not_called()

    def test_cpu_list_parsing(self):
        self.assertEqual(platform._parse_cpu_list("0-3,8,10-11\n"), {0, 1, 2, 3, 8, 10, 11})


if __name__ == "__main__":
    unittest.main()

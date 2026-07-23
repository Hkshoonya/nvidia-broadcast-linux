import os
import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp


def _fuser_result(returncode, stdout):
    return SimpleNamespace(returncode=returncode, stdout=stdout)


class ProbeVcamConsumersTests(unittest.TestCase):
    """_probe_vcam_consumers: monitor first, fuser fallback with the
    userns liveness guard (blind fuser must yield None, never 0)."""

    @staticmethod
    def _fake_app(*, monitor=None, pipeline="holds"):
        if pipeline == "holds":
            video_pipeline = SimpleNamespace(_vcam_failed=False)
        elif pipeline == "failed":
            video_pipeline = SimpleNamespace(_vcam_failed=True)
        else:
            video_pipeline = None
        return SimpleNamespace(
            _vcam_monitor=monitor,
            _vcam_device="/dev/video10",
            _vcam_available=True,
            _video_pipeline=video_pipeline,
            # Upstream resolves the device through this accessor so a
            # configured override wins; the probe calls it, not the field.
            _active_vcam_device=lambda: "/dev/video10",
        )

    def _probe(self, fake):
        return NVBroadcastApp._probe_vcam_consumers(fake)

    def test_monitor_value_wins_without_subprocess(self):
        monitor = SimpleNamespace(running=True, consumers=lambda: 2)
        fake = self._fake_app(monitor=monitor)
        with mock.patch("subprocess.run") as run:
            self.assertEqual(self._probe(fake), 2)
        run.assert_not_called()

    def test_dead_monitor_falls_back_to_fuser(self):
        monitor = SimpleNamespace(running=False, consumers=lambda: 5)
        fake = self._fake_app(monitor=monitor, pipeline=None)
        with mock.patch("subprocess.run",
                        return_value=_fuser_result(1, "")):
            self.assertEqual(self._probe(fake), 0)

    def test_blind_fuser_returns_none_when_pipeline_holds_device(self):
        # Empty fuser output while our own v4l2sink holds the device:
        # fuser is namespace-blind, so its zero is not trustworthy.
        fake = self._fake_app(pipeline="holds")
        with mock.patch("subprocess.run",
                        return_value=_fuser_result(1, "")):
            self.assertIsNone(self._probe(fake))

    def test_empty_fuser_is_zero_when_we_hold_nothing(self):
        fake = self._fake_app(pipeline=None)
        with mock.patch("subprocess.run",
                        return_value=_fuser_result(1, "")):
            self.assertEqual(self._probe(fake), 0)

    def test_fuser_counts_others_and_excludes_own_pid(self):
        own = str(os.getpid())
        fake = self._fake_app(pipeline="holds")
        with mock.patch("subprocess.run",
                        return_value=_fuser_result(0, f" {own} 4242")):
            self.assertEqual(self._probe(fake), 1)

    def test_fuser_error_returns_none(self):
        fake = self._fake_app(pipeline=None)
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(self._probe(fake))


class WakeFromMonitorTests(unittest.TestCase):
    def test_wake_noop_when_not_idle(self):
        fake = SimpleNamespace(_idle_active=False, _exit_idle=mock.Mock())
        self.assertFalse(NVBroadcastApp._wake_from_vcam_monitor(fake))
        fake._exit_idle.assert_not_called()

    def test_wake_exits_idle_when_idle(self):
        fake = SimpleNamespace(_idle_active=True, _exit_idle=mock.Mock())
        self.assertFalse(NVBroadcastApp._wake_from_vcam_monitor(fake))
        fake._exit_idle.assert_called_once_with(
            "consumer detected (v4l2 event)")


if __name__ == "__main__":
    unittest.main()

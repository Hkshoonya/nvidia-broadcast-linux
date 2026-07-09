import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nvbroadcast.video.vcam_monitor import (
    DISTRUST_SECONDS,
    IN_CLOSE_NOWRITE,
    IN_CLOSE_WRITE,
    IN_IGNORED,
    IN_OPEN,
    IN_Q_OVERFLOW,
    WAKE_DEBOUNCE_SECONDS,
    VcamConsumerMonitor,
    _parse_events,
)

DEVICE = "/dev/video10"


def _event_bytes(wd, mask, name=b""):
    return struct.pack("iIII", wd, mask, 0, len(name)) + name


class ParseEventsTests(unittest.TestCase):
    def test_single_event(self):
        self.assertEqual(_parse_events(_event_bytes(1, IN_OPEN)),
                         [(1, IN_OPEN)])

    def test_batch(self):
        buf = _event_bytes(1, IN_OPEN) + _event_bytes(1, IN_CLOSE_NOWRITE)
        self.assertEqual(_parse_events(buf),
                         [(1, IN_OPEN), (1, IN_CLOSE_NOWRITE)])

    def test_name_payload_is_skipped(self):
        buf = (_event_bytes(1, IN_OPEN, b"name\x00\x00\x00\x00")
               + _event_bytes(1, IN_CLOSE_WRITE))
        self.assertEqual(_parse_events(buf),
                         [(1, IN_OPEN), (1, IN_CLOSE_WRITE)])

    def test_truncated_header_ignored(self):
        self.assertEqual(_parse_events(b"\x01\x02\x03"), [])


class _FakeInotify:
    def __init__(self):
        self.batches = []
        self.add_watch_calls = 0
        self.add_watch_error = None

    def read_events(self):
        return self.batches.pop(0) if self.batches else []

    def add_watch(self, path):
        self.add_watch_calls += 1
        if self.add_watch_error is not None:
            raise self.add_watch_error
        return 1

    def fileno(self):
        return -1

    def close(self):
        pass


class _Harness:
    """Monitor with injected inotify, own-fd count, and clock — drives
    _tick directly, no thread."""

    def __init__(self, baseline_own=0):
        self.clock = [0.0]
        self.own = [baseline_own]
        self.wakes = 0
        self.inotify = _FakeInotify()
        self.monitor = VcamConsumerMonitor(
            DEVICE,
            wake_callback=self._wake,
            own_fd_counter=lambda: self.own[0],
            time_fn=lambda: self.clock[0],
        )
        self.monitor._inotify = self.inotify
        self.monitor._baseline_own = baseline_own
        self.monitor._published = 0
        # consumers() requires a live thread; fake one for unit tests.
        self.monitor._thread = SimpleNamespace(is_alive=lambda: True)

    def _wake(self):
        self.wakes += 1

    def tick(self, events=None, at=None, own=None):
        if at is not None:
            self.clock[0] = at
        if own is not None:
            self.own[0] = own
        if events:
            self.inotify.batches.append(events)
        self.monitor._tick()
        return self.monitor.consumers()


class VcamConsumerMonitorTests(unittest.TestCase):
    def test_probe_pair_does_not_wake(self):
        h = _Harness()
        # wireplumber-style open/close within one drain
        self.assertEqual(
            h.tick([(1, IN_OPEN), (1, IN_CLOSE_NOWRITE)], at=0.0), 0)
        self.assertEqual(h.tick(at=5.0), 0)
        self.assertEqual(h.wakes, 0)

    def test_sustained_open_wakes_once_and_rearms(self):
        h = _Harness()
        self.assertEqual(h.tick([(1, IN_OPEN)], at=0.0), 1)
        self.assertEqual(h.wakes, 0)  # debounce pending
        h.tick(at=WAKE_DEBOUNCE_SECONDS + 0.1)
        self.assertEqual(h.wakes, 1)
        h.tick(at=5.0)
        self.assertEqual(h.wakes, 1)  # no refire while episode lasts
        self.assertEqual(h.tick([(1, IN_CLOSE_NOWRITE)], at=6.0), 0)
        # second episode fires again
        h.tick([(1, IN_OPEN)], at=10.0)
        h.tick(at=10.0 + WAKE_DEBOUNCE_SECONDS + 0.1)
        self.assertEqual(h.wakes, 2)

    def test_own_fd_churn_is_compensated(self):
        h = _Harness(baseline_own=0)
        # our v4l2sink opens the device: event + own fd move together
        self.assertEqual(h.tick([(1, IN_OPEN)], at=0.0, own=1), 0)
        # pipeline rebuild: close + reopen
        self.assertEqual(
            h.tick([(1, IN_CLOSE_WRITE), (1, IN_OPEN)], at=1.0, own=1), 0)
        self.assertEqual(h.tick(at=5.0), 0)
        self.assertEqual(h.wakes, 0)

    def test_pre_baseline_close_clamps_at_zero(self):
        h = _Harness()
        # a consumer attached before start() closes: unmatched CLOSE
        self.assertEqual(h.tick([(1, IN_CLOSE_NOWRITE)], at=0.0), 0)
        # and a fresh real consumer is still detected afterwards
        self.assertEqual(h.tick([(1, IN_OPEN)], at=1.0), 0)  # net back to 0
        self.assertEqual(h.tick([(1, IN_OPEN)], at=2.0), 1)

    def test_overflow_distrusts_then_recovers(self):
        h = _Harness()
        self.assertIsNone(h.tick([(-1, IN_Q_OVERFLOW)], at=10.0))
        self.assertIsNone(h.tick(at=10.0 + DISTRUST_SECONDS - 1))
        self.assertEqual(h.tick(at=10.0 + DISTRUST_SECONDS + 1), 0)
        self.assertEqual(h.wakes, 0)

    def test_distrusted_estimate_never_wakes(self):
        h = _Harness()
        h.tick([(-1, IN_Q_OVERFLOW), (1, IN_OPEN)], at=0.0)
        h.tick(at=WAKE_DEBOUNCE_SECONDS + 1)
        self.assertEqual(h.wakes, 0)

    def test_ignored_rewatches_and_resyncs(self):
        h = _Harness()
        self.assertIsNone(h.tick([(1, IN_IGNORED)], at=0.0))
        # same tick already re-added the watch (device recreated fast)
        self.assertEqual(h.inotify.add_watch_calls, 1)
        self.assertFalse(h.monitor._watch_lost)
        # resync distrust window, then trusted again
        self.assertIsNone(h.tick(at=1.0))
        self.assertEqual(h.tick(at=DISTRUST_SECONDS + 2), 0)

    def test_ignored_with_missing_device_retries(self):
        h = _Harness()
        h.inotify.add_watch_error = OSError(2, "no such file")
        self.assertIsNone(h.tick([(1, IN_IGNORED)], at=0.0))
        self.assertTrue(h.monitor._watch_lost)
        h.inotify.add_watch_error = None
        self.assertIsNone(h.tick(at=0.5))  # before retry interval
        self.assertIsNone(h.tick(at=2.5))  # retry succeeds -> resync/distrust
        self.assertFalse(h.monitor._watch_lost)

    def test_consumers_none_without_thread(self):
        monitor = VcamConsumerMonitor(DEVICE)
        self.assertIsNone(monitor.consumers())
        self.assertFalse(monitor.running)

    def test_start_false_when_inotify_unavailable(self):
        with patch("nvbroadcast.video.vcam_monitor._Inotify",
                   side_effect=OSError(38, "not supported")):
            monitor = VcamConsumerMonitor(DEVICE)
            self.assertFalse(monitor.start())
            self.assertIsNone(monitor.consumers())

    def test_start_false_when_device_missing(self):
        monitor = VcamConsumerMonitor("/nonexistent/device")
        self.assertFalse(monitor.start())

    def test_real_start_stop_roundtrip(self):
        # Real inotify on a file that always exists; exercises the ctypes
        # wrapper and thread lifecycle end to end.
        monitor = VcamConsumerMonitor("/dev/null")
        self.assertTrue(monitor.start())
        try:
            self.assertTrue(monitor.running)
            self.assertEqual(monitor.consumers(), 0)
        finally:
            monitor.stop()
        self.assertFalse(monitor.running)
        self.assertIsNone(monitor.consumers())


if __name__ == "__main__":
    unittest.main()

import os
import time
import unittest

from nvbroadcast.video.vcam_monitor import (
    VcamConsumerMonitor,
    _V4l2EventSource,
)

DEVICE = "/dev/video10"


class _FakeSource:
    """Scripted event source: each drain() pops one batch of counts.
    A batch may instead be an OSError to raise (device gone)."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.closed = False

    def wait(self, timeout):
        # Real wait blocks up to `timeout`; keep the test loop fast but
        # let empty scripts idle briefly so the thread doesn't spin.
        if not self.batches:
            time.sleep(0.01)

    def drain(self):
        if not self.batches:
            return []
        batch = self.batches.pop(0)
        if isinstance(batch, OSError):
            raise batch
        return batch

    def close(self):
        self.closed = True


def _wait_for(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class VcamConsumerMonitorTests(unittest.TestCase):
    def _monitor(self, batches, factories=None):
        """Monitor whose source_factory yields _FakeSource(batches);
        `factories` overrides subsequent factory calls (for resubscribe)."""
        sources = [_FakeSource(batches)] + (factories or [])
        calls = []

        def factory(device):
            calls.append(device)
            if not sources:
                raise OSError(19, "no such device")
            src = sources.pop(0)
            if isinstance(src, OSError):
                raise src
            return src

        wakes = []
        monitor = VcamConsumerMonitor(
            DEVICE, wake_callback=lambda: wakes.append(time.time()),
            source_factory=factory)
        return monitor, wakes, calls

    def test_initial_event_publishes_count(self):
        monitor, wakes, _ = self._monitor([[0]])
        self.assertTrue(monitor.start())
        try:
            self.assertTrue(_wait_for(lambda: monitor.consumers() == 0))
            self.assertEqual(wakes, [])
        finally:
            monitor.stop()

    def test_unknown_until_initial_event(self):
        monitor, _, _ = self._monitor([])  # no events ever
        self.assertTrue(monitor.start())
        try:
            time.sleep(0.05)
            self.assertIsNone(monitor.consumers())
        finally:
            monitor.stop()

    def test_consumer_appearing_fires_wake_once(self):
        monitor, wakes, _ = self._monitor([[0], [1], [1]])
        self.assertTrue(monitor.start())
        try:
            self.assertTrue(_wait_for(lambda: monitor.consumers() == 1))
            time.sleep(0.05)
            self.assertEqual(len(wakes), 1)  # no refire while positive
        finally:
            monitor.stop()

    def test_wake_rearms_after_count_returns_to_zero(self):
        monitor, wakes, _ = self._monitor([[0], [1], [0], [1]])
        self.assertTrue(monitor.start())
        try:
            self.assertTrue(_wait_for(lambda: len(wakes) == 2))
        finally:
            monitor.stop()

    def test_initial_event_with_consumer_fires_wake(self):
        # Consumer already streaming when we subscribe (SEND_INITIAL).
        monitor, wakes, _ = self._monitor([[1]])
        self.assertTrue(monitor.start())
        try:
            self.assertTrue(_wait_for(lambda: len(wakes) == 1))
            self.assertEqual(monitor.consumers(), 1)
        finally:
            monitor.stop()

    def test_device_loss_publishes_none_then_resubscribes(self):
        lost = OSError(19, "no such device")
        recovered = _FakeSource([[0]])
        monitor, _, calls = self._monitor([[1], lost], factories=[recovered])
        self.assertTrue(monitor.start())
        try:
            self.assertTrue(_wait_for(lambda: monitor.consumers() == 0))
            self.assertGreaterEqual(len(calls), 2)  # resubscribed
        finally:
            monitor.stop()

    def test_wake_callback_exception_does_not_kill_thread(self):
        monitor, _, _ = self._monitor([[0], [1], [0]])
        monitor._wake_callback = lambda: (_ for _ in ()).throw(RuntimeError())
        self.assertTrue(monitor.start())
        try:
            self.assertTrue(_wait_for(lambda: monitor.consumers() == 0
                                      and not monitor._was_positive))
            self.assertTrue(monitor.running)
        finally:
            monitor.stop()

    def test_consumers_none_without_thread(self):
        monitor = VcamConsumerMonitor(DEVICE)
        self.assertIsNone(monitor.consumers())
        self.assertFalse(monitor.running)

    def test_start_false_when_source_unavailable(self):
        monitor = VcamConsumerMonitor(
            DEVICE, source_factory=lambda d: (_ for _ in ()).throw(
                OSError(2, "no such file")))
        self.assertFalse(monitor.start())
        self.assertIsNone(monitor.consumers())

    def test_stop_closes_source(self):
        src = _FakeSource([[0]])
        monitor = VcamConsumerMonitor(DEVICE, source_factory=lambda d: src)
        self.assertTrue(monitor.start())
        monitor.stop()
        self.assertTrue(src.closed)
        self.assertIsNone(monitor.consumers())


@unittest.skipUnless(os.path.exists(DEVICE), "no v4l2loopback device")
class RealDeviceTests(unittest.TestCase):
    def test_subscribe_and_initial_count(self):
        try:
            source = _V4l2EventSource(DEVICE)
        except OSError as e:
            self.skipTest(f"client-usage event unsupported: {e}")
        try:
            source.wait(1.0)
            counts = source.drain()
            # SEND_INITIAL guarantees at least one event promptly.
            self.assertTrue(counts, "no initial client-usage event")
            self.assertTrue(all(isinstance(c, int) for c in counts))
        finally:
            source.close()

    def test_monitor_roundtrip(self):
        monitor = VcamConsumerMonitor(DEVICE)
        if not monitor.start():
            self.skipTest("client-usage event unsupported")
        try:
            self.assertTrue(_wait_for(
                lambda: monitor.consumers() is not None))
        finally:
            monitor.stop()
        self.assertFalse(monitor.running)


if __name__ == "__main__":
    unittest.main()

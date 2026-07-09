# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Virtual-camera consumer detection via inotify.

`fuser`-style /proc scanning cannot see other processes' fds when the app
runs inside a sandbox user namespace (bubblewrap): the /proc/PID/fd magic
links of tasks outside the namespace fail the kernel's ptrace-mode check,
so fuser reports a confident "no consumers" instead of an error. inotify
open/close events on the device node are delivered kernel-side on the
inode and are immune to namespace boundaries, so they work everywhere.

inotify events carry no PID, so the app's own opens (GStreamer v4l2sink,
including its close/reopen churn during pipeline rebuilds) are
indistinguishable from consumers. /proc/self/fd IS always readable, so the
estimate compensates with the app's own fd count:

    consumers = max(0, baseline_own + net_open_events - own_fds_now)

Known limitation: consumers already holding the device before start() are
invisible until they cycle their fd (external fds cannot be enumerated
from the namespace at all — the very reason this module exists). Their
reopen attempts still generate events.
"""

import ctypes
import os
import select
import struct
import threading
import time
from typing import Callable, Optional

POLL_TIMEOUT_SECONDS = 0.5
# Something system-side (wireplumber/tray probing) opens and closes the
# device every few seconds; only a fd held longer than this is a consumer.
WAKE_DEBOUNCE_SECONDS = 0.7
DISTRUST_SECONDS = 30.0
REWATCH_INTERVAL_SECONDS = 2.0

IN_OPEN = 0x20
IN_CLOSE_WRITE = 0x8
IN_CLOSE_NOWRITE = 0x10
IN_Q_OVERFLOW = 0x4000
IN_IGNORED = 0x8000
IN_NONBLOCK = 0x800
IN_CLOEXEC = 0x80000

_WATCH_MASK = IN_OPEN | IN_CLOSE_WRITE | IN_CLOSE_NOWRITE
_CLOSE_MASK = IN_CLOSE_WRITE | IN_CLOSE_NOWRITE
_EVENT_HEADER = struct.Struct("iIII")  # wd, mask, cookie, name-len


def _parse_events(buf: bytes) -> list[tuple[int, int]]:
    """Return (wd, mask) pairs from a raw inotify read.

    Device watches never carry a name payload, but the header's len field
    is honoured anyway so a surprise payload cannot desync the stream.
    """
    events = []
    offset = 0
    while offset + _EVENT_HEADER.size <= len(buf):
        wd, mask, _cookie, name_len = _EVENT_HEADER.unpack_from(buf, offset)
        events.append((wd, mask))
        offset += _EVENT_HEADER.size + name_len
    return events


class _Inotify:
    """Minimal ctypes inotify wrapper; no pyinotify/watchdog dependency."""

    def __init__(self):
        self._libc = ctypes.CDLL(None, use_errno=True)
        fd = self._libc.inotify_init1(IN_NONBLOCK | IN_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        self._fd = fd

    def add_watch(self, path: str) -> int:
        wd = self._libc.inotify_add_watch(
            self._fd, path.encode(), _WATCH_MASK)
        if wd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"inotify_add_watch({path}): {os.strerror(err)}")
        return wd

    def read_events(self) -> list[tuple[int, int]]:
        events = []
        while True:
            try:
                data = os.read(self._fd, 4096)
            except BlockingIOError:
                return events
            except OSError:
                return events
            if not data:
                return events
            events.extend(_parse_events(data))

    def fileno(self) -> int:
        return self._fd

    def close(self):
        try:
            os.close(self._fd)
        except OSError:
            pass


class VcamConsumerMonitor:
    """Watches a v4l2loopback node and estimates external consumer count.

    consumers() returns None whenever the answer is not trustworthy (not
    running, event-queue overflow, device node gone) — callers must treat
    None as "in use" so a detection failure can never freeze a camera.
    """

    def __init__(self, device: str,
                 wake_callback: Optional[Callable[[], None]] = None,
                 own_fd_counter: Optional[Callable[[], int]] = None,
                 time_fn: Callable[[], float] = time.monotonic):
        self._device = device
        self._wake_callback = wake_callback
        self._own_fd_counter = own_fd_counter or self._count_own_fds
        self._time = time_fn
        self._inotify: Optional[_Inotify] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._baseline_own = 0
        self._net = 0
        self._published: Optional[int] = None
        self._distrust_until = 0.0
        self._watch_lost = False
        self._next_rewatch = 0.0
        self._above_zero_since: Optional[float] = None
        self._wake_fired = False

    # ─── Public API ─────────────────────────────────────────────────────

    def start(self) -> bool:
        """Arm the watch and start the monitor thread.

        Ordering is load-bearing: add_watch BEFORE the own-fd baseline
        read, from a moment when the app cannot be opening the device —
        otherwise an own open lands in the baseline but not the event
        stream (or vice versa) and skews the estimate permanently.
        """
        try:
            inotify = _Inotify()
        except OSError as e:
            print(f"[NV Broadcast] vcam monitor unavailable: {e}", flush=True)
            return False
        try:
            inotify.add_watch(self._device)
        except OSError as e:
            print(f"[NV Broadcast] vcam monitor unavailable: {e}", flush=True)
            inotify.close()
            return False
        self._inotify = inotify
        self._baseline_own = self._own_fd_counter()
        self._net = 0
        self._published = 0
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="nvbroadcast-vcam-monitor")
        self._thread.start()
        return True

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * POLL_TIMEOUT_SECONDS + 0.5)
            self._thread = None

    def consumers(self) -> Optional[int]:
        """Latest external-consumer estimate; None = unknown (= in use)."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return None
        return self._published

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ─── Internals ──────────────────────────────────────────────────────

    def _count_own_fds(self) -> int:
        """Count this process's open fds on the device via /proc/self/fd.

        Always readable even inside a user namespace. Matches the
        "(deleted)" readlink suffix too: after a v4l2loopback reload our
        stale fd would otherwise vanish from the count while its close
        event still arrives later, leaving a phantom consumer.
        """
        count = 0
        deleted = self._device + " (deleted)"
        try:
            entries = os.listdir("/proc/self/fd")
        except OSError:
            return count
        for entry in entries:
            try:
                target = os.readlink(f"/proc/self/fd/{entry}")
            except OSError:
                continue  # fds churn while we scan
            if target == self._device or target == deleted:
                count += 1
        return count

    def _run(self):
        try:
            while not self._stop_evt.is_set():
                try:
                    select.select(
                        [self._inotify.fileno()], [], [], POLL_TIMEOUT_SECONDS)
                except OSError:
                    pass
                if self._stop_evt.is_set():
                    break
                self._tick()
        finally:
            if self._inotify is not None:
                self._inotify.close()

    def _tick(self):
        """One monitor pass: drain events, then count own fds.

        Draining first matters: an own open appears in /proc/self/fd at
        the same syscall that queues its IN_OPEN, so any own fd we count
        already had its event consumed (or both arrive together next
        pass). Residual races last one pass and cannot beat the wake
        debounce.
        """
        for _wd, mask in self._inotify.read_events():
            if mask & IN_Q_OVERFLOW:
                self._resync("event queue overflow")
                continue
            if mask & IN_IGNORED:
                self._watch_lost = True
                self._next_rewatch = self._time()
                continue
            if mask & IN_OPEN:
                self._net += 1
            if mask & _CLOSE_MASK:
                self._net -= 1

        now = self._time()
        if self._watch_lost and now >= self._next_rewatch:
            try:
                self._inotify.add_watch(self._device)
                self._watch_lost = False
                self._resync("device reappeared")
            except OSError:
                self._next_rewatch = now + REWATCH_INTERVAL_SECONDS

        if self._watch_lost or now < self._distrust_until:
            estimate: Optional[int] = None
        else:
            estimate = max(
                0, self._baseline_own + self._net - self._own_fd_counter())
        self._published = estimate

        if estimate is not None and estimate > 0:
            if self._above_zero_since is None:
                self._above_zero_since = now
            elif (not self._wake_fired
                  and now - self._above_zero_since >= WAKE_DEBOUNCE_SECONDS):
                self._wake_fired = True
                if self._wake_callback is not None:
                    try:
                        self._wake_callback()
                    except Exception as e:
                        print(f"[NV Broadcast] vcam wake callback failed: {e}",
                              flush=True)
        else:
            self._above_zero_since = None
            self._wake_fired = False

    def _resync(self, reason: str):
        """Reset counters after losing event continuity; distrust briefly."""
        self._net = 0
        self._baseline_own = self._own_fd_counter()
        self._distrust_until = self._time() + DISTRUST_SECONDS
        self._above_zero_since = None
        self._wake_fired = False
        print(f"[NV Broadcast] vcam monitor resync: {reason}", flush=True)

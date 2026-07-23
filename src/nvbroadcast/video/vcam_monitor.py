# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Virtual-camera consumer detection via v4l2loopback client-usage events.

`fuser`-style /proc scanning cannot see other processes' fds when the app
runs inside a sandbox user namespace (bubblewrap): the /proc/PID/fd magic
links of tasks outside the namespace fail the kernel's ptrace-mode check,
so fuser reports a confident "no consumers" instead of an error.

inotify open/close counting was tried and is unreliable: the kernel
coalesces adjacent identical events, so overlapping opens (ffmpeg and
browser camera stacks probe-open the device while their capture fd is
already open) silently lose an IN_OPEN and the net count drifts low —
which pauses the camera mid-call.

v4l2loopback (>= 0.13) solves this properly: subscribing to its private
V4L2_EVENT_PRI_CLIENT_USAGE event on any fd of the device yields an
absolute "are capture clients streaming" count, updated on every capture
STREAMON/STREAMOFF and once immediately at subscribe time
(V4L2_EVENT_SUB_FL_SEND_INITIAL). It is kernel truth delivered through
the device fd itself, so no namespace can hide consumers, and non-
streaming opens (wireplumber's periodic device probes) are ignored.
Our own producer is an OUTPUT client and never takes a capture token,
so the count needs no self-compensation.
"""

import errno
import fcntl
import os
import select
import struct
import threading
import time
from typing import Callable, Optional

POLL_TIMEOUT_SECONDS = 0.5
RESUBSCRIBE_INTERVAL_SECONDS = 2.0

# v4l2loopback.h: V4L2_EVENT_PRIVATE_START + V4L2LOOPBACK_EVENT_OFFSET + 1
V4L2_EVENT_PRI_CLIENT_USAGE = 0x08000000 + 0x08E00000 + 1
V4L2_EVENT_SUB_FL_SEND_INITIAL = 0x1

# struct v4l2_event_subscription: u32 type, id, flags, reserved[5] = 32 bytes
_VIDIOC_SUBSCRIBE_EVENT = (1 << 30) | (32 << 16) | (ord("V") << 8) | 90
# struct v4l2_event on 64-bit: u32 type; pad; union u (8-aligned, 64 bytes);
# u32 pending; u32 sequence; timespec; u32 id; u32 reserved[8] -> 136 bytes.
# The client-usage count is the first u32 of the union, at offset 8.
_V4L2_EVENT_SIZE = 136
_VIDIOC_DQEVENT = (2 << 30) | (_V4L2_EVENT_SIZE << 16) | (ord("V") << 8) | 89
_EVENT_TYPE_OFFSET = 0
_EVENT_COUNT_OFFSET = 8
_EVENT_PENDING_OFFSET = 72


class _V4l2EventSource:
    """One subscribed fd on the loopback device; raises OSError when the
    device is unavailable or the module has no client-usage event."""

    def __init__(self, device: str):
        self._fd = os.open(device, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        sub = struct.pack(
            "III5I", V4L2_EVENT_PRI_CLIENT_USAGE, 0,
            V4L2_EVENT_SUB_FL_SEND_INITIAL, 0, 0, 0, 0, 0)
        try:
            fcntl.ioctl(self._fd, _VIDIOC_SUBSCRIBE_EVENT, sub)
        except OSError:
            os.close(self._fd)
            raise

    def wait(self, timeout: float):
        """Block until an event is pending (POLLPRI) or timeout elapses."""
        select.select([], [], [self._fd], timeout)

    def drain(self) -> list[int]:
        """Dequeue all pending events; return their counts in order.

        Raises OSError (e.g. ENODEV) when the device went away — the
        caller must drop this source and resubscribe.
        """
        counts = []
        while True:
            buf = bytearray(_V4L2_EVENT_SIZE)
            try:
                fcntl.ioctl(self._fd, _VIDIOC_DQEVENT, buf)
            except OSError as e:
                if e.errno == errno.ENOENT:  # queue empty
                    return counts
                raise
            etype, = struct.unpack_from("I", buf, _EVENT_TYPE_OFFSET)
            if etype == V4L2_EVENT_PRI_CLIENT_USAGE:
                count, = struct.unpack_from("I", buf, _EVENT_COUNT_OFFSET)
                counts.append(count)
            pending, = struct.unpack_from("I", buf, _EVENT_PENDING_OFFSET)
            if pending == 0:
                return counts

    def close(self):
        try:
            os.close(self._fd)
        except OSError:
            pass


class VcamConsumerMonitor:
    """Publishes the number of external capture clients on a v4l2loopback
    device, using the module's client-usage event.

    consumers() returns None whenever the answer is not trustworthy (not
    running, device gone, resubscribing) — callers must treat None as
    "in use" so a detection failure can never freeze someone's camera.
    """

    def __init__(self, device: str,
                 wake_callback: Optional[Callable[[], None]] = None,
                 source_factory: Optional[Callable[[str], object]] = None,
                 time_fn: Callable[[], float] = time.monotonic):
        self._device = device
        self._wake_callback = wake_callback
        self._source_factory = source_factory or _V4l2EventSource
        self._time = time_fn
        self._source = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._published: Optional[int] = None
        self._was_positive = False

    # ─── Public API ─────────────────────────────────────────────────────

    def start(self) -> bool:
        """Subscribe and start the monitor thread.

        Returns False when the device cannot be opened or the loopback
        module predates the client-usage event (< 0.13) — callers fall
        back to fuser.
        """
        try:
            self._source = self._source_factory(self._device)
        except OSError as e:
            print(f"[NV Broadcast] vcam monitor unavailable: {e}", flush=True)
            return False
        self._published = None  # trust nothing until the initial event
        self._was_positive = False
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
        """Latest streaming-client count; None = unknown (= in use)."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return None
        return self._published

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ─── Internals ──────────────────────────────────────────────────────

    def _run(self):
        try:
            while not self._stop_evt.is_set():
                if self._source is None:
                    if not self._resubscribe():
                        self._stop_evt.wait(RESUBSCRIBE_INTERVAL_SECONDS)
                        continue
                try:
                    self._source.wait(POLL_TIMEOUT_SECONDS)
                    if self._stop_evt.is_set():
                        break
                    counts = self._source.drain()
                except OSError as e:
                    print(f"[NV Broadcast] vcam monitor lost device: {e}",
                          flush=True)
                    self._drop_source()
                    continue
                for count in counts:
                    self._publish(count)
        finally:
            self._drop_source()

    def _publish(self, count: int):
        self._published = count
        if count > 0:
            if not self._was_positive and self._wake_callback is not None:
                try:
                    self._wake_callback()
                except Exception as e:
                    print(f"[NV Broadcast] vcam wake callback failed: {e}",
                          flush=True)
            self._was_positive = True
        else:
            self._was_positive = False

    def _drop_source(self):
        if self._source is not None:
            self._source.close()
            self._source = None
        self._published = None  # unknown until resubscribed

    def _resubscribe(self) -> bool:
        try:
            self._source = self._source_factory(self._device)
        except OSError:
            return False
        self._was_positive = False
        print("[NV Broadcast] vcam monitor resubscribed", flush=True)
        return True

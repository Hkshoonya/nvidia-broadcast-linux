# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0-or-later - see LICENSE file
#
"""Overlap matte inference for later frames with compositing earlier ones.

Inline inference (max_quality/balanced and fused CUDA Fast replace) pairs
every composite with the matte inferred for that same camera frame, which
keeps replaced-background edges from trailing motion. Done serially, one
frame costs inference + matte refinement + compositing, and the capture
thread drops every camera frame that arrives meanwhile.

This module runs that per-frame work as a chain of stages, one worker
thread each (for example model inference -> alpha refinement -> replacement
matte), while the capture thread composites the oldest finished frame.
Every stage still processes frames strictly in order and keeps its own
history, and each composite uses its own frame's matte, so rendered frames
match the serial path. Output trails capture by one frame per stage in
exchange for throughput bounded by the slowest stage instead of their sum.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, NamedTuple, Sequence


class StagedFrame(NamedTuple):
    """A frame whose stages have all finished."""

    frame: Any
    prepared: Any
    width: int
    height: int


class _Pending(NamedTuple):
    frame: Any
    width: int
    height: int
    future: Future
    submitted: float


class StageStats:
    """Opt-in per-stage timing log (NVBROADCAST_STAGE_STATS=1), every 5 s."""

    INTERVAL_S = 5.0

    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = os.getenv("NVBROADCAST_STAGE_STATS") == "1"
        self.enabled = enabled
        self._lock = threading.Lock()
        self._totals: dict[str, list[float]] = {}
        self._since = time.monotonic()

    def record(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            entry = self._totals.setdefault(name, [0.0, 0, 0.0])
            entry[0] += seconds
            entry[1] += 1
            entry[2] = max(entry[2], seconds)
            now = time.monotonic()
            if now - self._since < self.INTERVAL_S:
                return
            parts = [
                f"{key}={total / count * 1000:.1f}ms(max {peak * 1000:.0f})"
                for key, (total, count, peak) in self._totals.items() if count
            ]
            frames = max((count for _t, count, _p in self._totals.values()), default=0)
            fps = frames / (now - self._since)
            self._totals = {}
            self._since = now
        print(f"[NV Broadcast] stage stats {fps:.1f}/s " + " ".join(parts), flush=True)


class PipelinedInferenceStage:
    """Chain of single-worker stages with one frame in flight per stage."""

    # A pending frame older than this belongs to a stalled or restarted
    # capture; never surface it as an earlier frame.
    MAX_PENDING_AGE_S = 0.5

    def __init__(self, first: Callable[[Any, int, int], Any],
                 *rest: Callable[[Any], Any]):
        self._stages: Sequence[Callable] = (first, *rest)
        self.stats = StageStats()
        self._lock = threading.Lock()
        self._executors: list[ThreadPoolExecutor] | None = None
        self._pending: deque[_Pending] = deque()

    @property
    def depth(self) -> int:
        """Frames in flight once the pipeline is full."""
        return len(self._stages)

    def _ensure_executors(self) -> list[ThreadPoolExecutor]:
        if self._executors is None:
            self._executors = [
                ThreadPoolExecutor(max_workers=1,
                                   thread_name_prefix=f"nvb-infer-stage{index}")
                for index in range(len(self._stages))
            ]
        return self._executors

    def _timed(self, index: int, stage: Callable, *args):
        if not self.stats.enabled:
            return stage(*args)
        start = time.perf_counter()
        try:
            return stage(*args)
        finally:
            self.stats.record(f"stage{index}", time.perf_counter() - start)

    def _submit(self, frame, width: int, height: int) -> Future:
        executors = self._ensure_executors()
        future = executors[0].submit(self._timed, 0, self._stages[0], frame, width, height)
        for index, (executor, stage) in enumerate(
                zip(executors[1:], self._stages[1:]), start=1):
            # Each worker waits on its predecessor's result for this frame;
            # single-worker executors keep every stage in frame order.
            future = executor.submit(
                lambda previous=future, stage=stage, index=index:
                    self._timed(index, stage, previous.result()))
        return future

    def advance(self, frame, width: int, height: int) -> StagedFrame | None:
        """Queue ``frame`` and return the oldest frame once ``depth`` are queued.

        Returns None while the pipeline (re)fills: on the first frames, after
        a size change, or after a stale pending frame; the caller decides how
        to fill that slot. Exceptions from a stage propagate to the caller for
        the frame that raised them, matching the serial path.
        """
        with self._lock:
            newest = self._pending[-1] if self._pending else None
            if newest is not None and (
                (newest.width, newest.height) != (width, height)
                or time.monotonic() - newest.submitted > self.MAX_PENDING_AGE_S
            ):
                # Stages stay ordered: let stale jobs finish before the next
                # frame may touch the per-stage matte history.
                self._drain_locked()
            future = self._submit(frame, width, height)
            self._pending.append(_Pending(frame, width, height, future, time.monotonic()))
            oldest = (self._pending.popleft()
                      if len(self._pending) > self.depth else None)
        if oldest is None:
            return None
        return StagedFrame(oldest.frame, oldest.future.result(),
                           oldest.width, oldest.height)

    def take_oldest(self) -> StagedFrame | None:
        """Wait for the oldest queued frame and remove it from the pipeline."""
        with self._lock:
            pending = self._pending.popleft() if self._pending else None
        if pending is None:
            return None
        return StagedFrame(pending.frame, pending.future.result(),
                           pending.width, pending.height)

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def drain(self) -> None:
        """Finish and discard every in-flight frame."""
        with self._lock:
            self._drain_locked()

    def _drain_locked(self) -> None:
        while self._pending:
            self._wait_quietly(self._pending.popleft().future)

    def shutdown(self) -> None:
        self.drain()
        with self._lock:
            executors, self._executors = self._executors, None
        for executor in executors or ():
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _wait_quietly(future: Future) -> None:
        try:
            future.result()
        except Exception:
            pass

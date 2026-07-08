# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""Startup timing marks.

Always cheap (one monotonic read + compare); per-mark lines print only
with NVBROADCAST_STARTUP_TRACE=1 so a slow start can be diagnosed from a
single relaunch instead of guesswork.
"""

import os
import time

_T0 = time.monotonic()
_VERBOSE = os.getenv("NVBROADCAST_STARTUP_TRACE", "") == "1"
_last = _T0


def mark(label: str) -> None:
    global _last
    now = time.monotonic()
    if _VERBOSE:
        print(f"[NV Broadcast][startup +{now - _T0:6.2f}s "
              f"(+{now - _last:5.2f}s)] {label}", flush=True)
    _last = now


def elapsed() -> float:
    return time.monotonic() - _T0

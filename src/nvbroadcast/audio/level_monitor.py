# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Real-time audio level monitoring for VU meter display."""

import threading
import time
import numpy as np


class AudioLevelMonitor:
    """Monitors audio input level for VU meter UI."""

    def __init__(self):
        self._level_db = -60.0  # dB, -60 = silence
        self._peak_db = -60.0
        self._peak_hold_time = 0
        self._lock = threading.Lock()

    def update(self, audio_chunk: np.ndarray):
        """Update level from an audio chunk (float32, -1.0 to 1.0)."""
        if len(audio_chunk) == 0:
            return

        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -60.0

        db = max(-60.0, min(0.0, db))

        with self._lock:
            self._level_db = db
            now = time.monotonic()
            if db > self._peak_db or now - self._peak_hold_time > 1.5:
                self._peak_db = db
                self._peak_hold_time = now

    @property
    def level_db(self) -> float:
        with self._lock:
            return self._level_db

    @property
    def peak_db(self) -> float:
        with self._lock:
            return self._peak_db

    @property
    def level_normalized(self) -> float:
        """Level as 0.0-1.0 (for progress bar)."""
        db = self.level_db
        return max(0.0, (db + 60.0) / 60.0)

    @property
    def peak_normalized(self) -> float:
        db = self.peak_db
        return max(0.0, (db + 60.0) / 60.0)

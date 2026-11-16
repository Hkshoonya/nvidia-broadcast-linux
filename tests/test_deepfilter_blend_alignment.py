import unittest

import numpy as np

from nvbroadcast.audio import deepfilter
from nvbroadcast.audio.deepfilter import HOP_SIZE, MODEL_DELAY_HOPS, DeepFilterDenoiser


class _DelayingSession:
    """Stand-in for the streaming model: returns input hop k-MODEL_DELAY_HOPS."""

    def __init__(self):
        self._hist = [np.zeros(HOP_SIZE, dtype=np.float32)] * MODEL_DELAY_HOPS

    def run(self, _outputs, feeds):
        self._hist.append(np.array(feeds["input_frame"], dtype=np.float32))
        return self._hist.pop(0).copy(), feeds["states"], np.zeros(1, np.float32)


def _denoise(signal: np.ndarray, intensity: float, block: int = 960) -> np.ndarray:
    d = DeepFilterDenoiser()
    d.session = _DelayingSession()
    d.reset()
    d._initialized = True
    return np.concatenate([
        d.process_chunk(signal[i:i + block].copy(), deepfilter.SAMPLE_RATE, intensity)
        for i in range(0, len(signal), block)
    ])


class BlendAlignmentTests(unittest.TestCase):
    def test_partial_strength_does_not_duplicate_the_voice(self):
        # One click in; a mis-aligned dry blend emits two clicks 20 ms apart
        # (the "echo" users heard at Strength < 1).
        signal = np.zeros(HOP_SIZE * 40, dtype=np.float32)
        signal[HOP_SIZE * 10 + 7] = 1.0
        for intensity in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = _denoise(signal, intensity)
            hits = np.flatnonzero(np.abs(out) > 1e-6)
            self.assertEqual(len(hits), 1, f"intensity={intensity}: clicks at {hits}")
            self.assertAlmostEqual(float(out[hits[0]]), 1.0, places=5)

    def test_dry_and_enhanced_paths_share_the_same_latency(self):
        signal = np.zeros(HOP_SIZE * 40, dtype=np.float32)
        signal[HOP_SIZE * 10 + 7] = 1.0
        wet = np.flatnonzero(_denoise(signal, 1.0))
        dry = np.flatnonzero(_denoise(signal, 0.0))
        self.assertEqual(wet.tolist(), dry.tolist())


if __name__ == "__main__":
    unittest.main()

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Voice effects — bass boost, treble, compression, warmth, EQ.

Real-time audio processing for professional microphone quality.

One effect chain serves both GPU and CPU modes so they sound identical.
The stateless warmth (tanh saturation) stage runs on CUDA via CuPy when
GPU mode is on; the shelving filters and compressor are stateful/sequential
and run vectorized on CPU in both modes. Either way the whole chain costs
well under a millisecond per ~21ms block at 48kHz mono.
"""

import numpy as np
from dataclasses import dataclass

# Try CuPy for GPU audio processing
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


@dataclass
class VoiceFXSettings:
    """Voice processing settings."""
    bass_boost: float = 0.0       # -1.0 to 1.0 (negative = cut, positive = boost)
    treble: float = 0.0           # -1.0 to 1.0
    warmth: float = 0.0           # 0.0 to 1.0 (adds harmonic saturation)
    compression: float = 0.0      # 0.0 to 1.0 (dynamic range compression)
    gate_threshold: float = 0.0   # 0.0 to 1.0 (noise gate — silence below threshold)
    gain: float = 0.0             # -1.0 to 1.0 (output volume adjustment)


def clone_voice_fx_settings(settings: "VoiceFXSettings") -> "VoiceFXSettings":
    """Return a detached copy of a settings object."""
    return VoiceFXSettings(
        bass_boost=settings.bass_boost,
        treble=settings.treble,
        warmth=settings.warmth,
        compression=settings.compression,
        gate_threshold=settings.gate_threshold,
        gain=settings.gain,
    )


def normalize_voice_fx_preset_name(preset_name: str | None) -> str:
    """Normalize legacy preset names to the current UI-facing presets."""
    if not preset_name:
        return "Flat"
    if preset_name == "Natural":
        return "Flat"
    return preset_name


def is_flat_voice_fx_settings(settings: "VoiceFXSettings", tol: float = 1e-6) -> bool:
    """Return whether the settings are effectively a no-op."""
    values = (
        settings.bass_boost,
        settings.treble,
        settings.warmth,
        settings.compression,
        settings.gate_threshold,
        settings.gain,
    )
    return all(abs(value) <= tol for value in values)


class VoiceFX:
    """Real-time voice effects processor."""

    def __init__(self, use_gpu: bool = True):
        self.settings = VoiceFXSettings()
        self._enabled = False
        self._use_gpu = use_gpu and _HAS_CUPY
        # Filter state (for IIR continuity across chunks)
        self._bass_state = np.zeros(2)
        self._treble_state = np.zeros(2)
        self._comp_env = 0.0  # Compressor envelope follower

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value: bool):
        self._use_gpu = value and _HAS_CUPY

    @property
    def gpu_available(self) -> bool:
        return _HAS_CUPY

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def process_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        gate_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process an audio chunk with all enabled effects.

        Uses GPU (CuPy) when available for batch processing all effects
        in a single upload/download cycle. Falls back to CPU (numpy).

        Args:
            audio: float32 array, values in [-1.0, 1.0]
            sample_rate: sample rate in Hz

        Returns:
            Processed float32 array, same length
        """
        if not self._enabled:
            return audio

        if gate_reference is None:
            gate_reference = audio

        result = audio.copy()
        s = self.settings

        # A single effect chain for both GPU and CPU modes. A previous GPU
        # branch ran the effects in a different order (gain before
        # compression, an extra mid-chain clip), so toggling GPU audibly
        # changed the sound — an ordering bug, not a hardware difference.
        # The stateless warmth stage dispatches to CUDA when enabled; the
        # stateful IIR/compressor stages are inherently sequential and stay
        # on CPU in both modes.

        # Noise gate — silence audio below threshold
        if s.gate_threshold > 0:
            result = self._noise_gate(result, s.gate_threshold, gate_reference)

        # Bass boost/cut — low-shelf filter at ~200Hz
        if abs(s.bass_boost) > 0.01:
            result = self._bass_filter(result, s.bass_boost, sample_rate)

        # Treble boost/cut — high-shelf filter at ~4kHz
        if abs(s.treble) > 0.01:
            result = self._treble_filter(result, s.treble, sample_rate)

        # Warmth — subtle harmonic saturation (tape emulation)
        if s.warmth > 0.01:
            if self._use_gpu and _HAS_CUPY and len(result) > 512:
                try:
                    result = self._warmth_gpu(result, s.warmth)
                except Exception:
                    result = self._warmth(result, s.warmth)
            else:
                result = self._warmth(result, s.warmth)

        # Compression — reduce dynamic range
        if s.compression > 0.01:
            result = self._compress(result, s.compression, sample_rate)

        # Output gain
        if abs(s.gain) > 0.01:
            gain_linear = 10 ** (s.gain * 12 / 20)  # ±12dB range
            result = result * gain_linear

        # Clip to prevent distortion
        return np.clip(result, -1.0, 1.0)

    @staticmethod
    def _warmth_gpu(audio: np.ndarray, warmth: float) -> np.ndarray:
        """CUDA warmth saturation — same formula as the CPU _warmth."""
        d = cp.asarray(audio, dtype=cp.float32)
        drive = 1.0 + warmth * 3.0
        wet = cp.tanh(d * drive) / cp.tanh(cp.float32(drive))
        d = d * (1 - warmth * 0.5) + wet * (warmth * 0.5)
        return cp.asnumpy(d)

    @staticmethod
    def _gate_threshold_linear(threshold: float) -> float:
        """Map a 0..1 gate control to a conservative linear threshold."""
        thresh_db = -60 + threshold * 40
        return 10 ** (thresh_db / 20)

    @staticmethod
    def _reference_rms(reference_audio: np.ndarray) -> float:
        """Compute an RMS level used for gate decisions."""
        if reference_audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(reference_audio.astype(np.float64) ** 2)))

    @staticmethod
    def _gate_gain(rms: float, threshold_linear: float) -> float:
        """Return a soft gate gain based on the reference RMS."""
        if threshold_linear <= 0 or rms >= threshold_linear:
            return 1.0
        ratio = max(rms / threshold_linear, 0.0)
        return max(0.1, ratio * ratio)

    def _noise_gate(
        self,
        audio: np.ndarray,
        threshold: float,
        reference_audio: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simple noise gate — attenuate only when the reference level is quiet."""
        reference = audio if reference_audio is None else reference_audio
        thresh_linear = self._gate_threshold_linear(threshold)
        rms = self._reference_rms(reference)
        return audio * self._gate_gain(rms, thresh_linear)

    def _bass_filter(self, audio: np.ndarray, amount: float,
                     sample_rate: int) -> np.ndarray:
        """Low-shelf filter for bass boost/cut — vectorized."""
        from scipy.signal import lfilter
        fc = 200.0
        w0 = 2 * np.pi * fc / sample_rate
        alpha = w0 / (w0 + 1)
        gain = 1.0 + amount * 0.8

        # 1-pole low-pass: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
        b = [alpha]
        a = [1, -(1 - alpha)]
        lp, self._bass_state[:1] = lfilter(b, a, audio, zi=self._bass_state[:1])
        return audio + (gain - 1.0) * lp

    def _treble_filter(self, audio: np.ndarray, amount: float,
                       sample_rate: int) -> np.ndarray:
        """High-shelf filter for treble boost/cut — vectorized."""
        from scipy.signal import lfilter
        fc = 4000.0
        w0 = 2 * np.pi * fc / sample_rate
        alpha = w0 / (w0 + 1)
        gain = 1.0 + amount * 0.6

        b = [alpha]
        a = [1, -(1 - alpha)]
        lp, self._treble_state[:1] = lfilter(b, a, audio, zi=self._treble_state[:1])
        hp = audio - lp
        return audio + (gain - 1.0) * hp

    def _warmth(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Tape-style harmonic saturation for warmth."""
        # Soft clipping via tanh — adds even harmonics
        drive = 1.0 + amount * 3.0
        wet = np.tanh(audio * drive) / np.tanh(drive)
        return audio * (1 - amount * 0.5) + wet * (amount * 0.5)

    def _compress(self, audio: np.ndarray, amount: float,
                  sample_rate: int) -> np.ndarray:
        """RMS compressor — vectorized with block processing."""
        threshold_db = -20 + (1 - amount) * 10
        threshold = 10 ** (threshold_db / 20)
        ratio = 1.0 + amount * 5.0
        makeup = 1.0 + amount * 0.5

        # Block-based compression (process in 256-sample blocks for speed)
        block_size = 256
        result = audio.copy()
        for start in range(0, len(audio), block_size):
            end = min(start + block_size, len(audio))
            block = audio[start:end]
            rms = np.sqrt(np.mean(block ** 2))
            if rms > threshold:
                gain_reduction = (threshold / rms) ** (1 - 1 / ratio)
                result[start:end] = block * gain_reduction

        return result * makeup


# Presets for common use cases
VOICE_PRESETS = {
    "Flat": VoiceFXSettings(),
    "Radio": VoiceFXSettings(
        bass_boost=0.3, treble=0.2, warmth=0.3,
        compression=0.5, gate_threshold=0.1, gain=0.1
    ),
    "Podcast": VoiceFXSettings(
        bass_boost=0.2, treble=0.1, warmth=0.2,
        compression=0.6, gate_threshold=0.12, gain=0.0
    ),
    "Deep Voice": VoiceFXSettings(
        bass_boost=0.6, treble=-0.2, warmth=0.4,
        compression=0.3, gate_threshold=0.08, gain=0.1
    ),
    "Bright": VoiceFXSettings(
        bass_boost=-0.1, treble=0.5, warmth=0.0,
        compression=0.2, gate_threshold=0.08, gain=0.0
    ),
    "Studio": VoiceFXSettings(
        bass_boost=0.15, treble=0.15, warmth=0.25,
        compression=0.7, gate_threshold=0.0, gain=0.05
    ),
}

DEFAULT_VOICE_FX_PRESET = "Studio"


def get_voice_fx_preset(preset_name: str | None) -> VoiceFXSettings | None:
    """Return a copied preset definition by name, handling legacy aliases."""
    normalized = normalize_voice_fx_preset_name(preset_name)
    preset = VOICE_PRESETS.get(normalized)
    if preset is None:
        return None
    return clone_voice_fx_settings(preset)

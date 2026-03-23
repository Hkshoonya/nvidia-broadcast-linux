# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Audio noise removal using RNNoise.

Real-time AI-powered noise suppression for microphone input.
Uses Mozilla's RNNoise via pyrnnoise (low-level C API).
"""

import numpy as np

from nvbroadcast.core.constants import MAXINE_AFX_PATH


class AudioEffects:
    """Real-time audio noise removal.

    Uses RNNoise algorithm: processes 480-sample frames (10ms at 48kHz)
    of int16 PCM audio.
    """

    def __init__(self, gpu_index: int = 1):
        self._gpu_index = gpu_index
        self._initialized = False
        self._state = None  # ctypes pointer to RNNoise state
        self._enabled = False
        self._intensity = 1.0

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if value and not self._initialized:
            self.initialize()

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, min(1.0, value))

    def initialize(self) -> bool:
        """Initialize the noise removal engine."""
        if self._initialized:
            return True

        try:
            from pyrnnoise import rnnoise
            self._rnnoise = rnnoise
            self._state = rnnoise.create()
            self._frame_size = rnnoise.FRAME_SIZE  # 480 samples
            self._initialized = True
            print("[NVIDIA Broadcast] Audio denoiser initialized (RNNoise)")
            return True
        except Exception as e:
            print(f"[NVIDIA Broadcast] Failed to initialize audio denoiser: {e}")
            return False

    def process_chunk(self, audio_data: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Process an audio chunk through the denoiser.

        Args:
            audio_data: Float32 mono audio samples
            sample_rate: Sample rate (48000 required for RNNoise)

        Returns:
            Denoised float32 audio samples
        """
        if not self._enabled or not self._initialized:
            return audio_data

        try:
            output = np.zeros_like(audio_data)
            total_samples = len(audio_data)
            fs = self._frame_size  # 480

            for i in range(0, total_samples - fs + 1, fs):
                frame = audio_data[i:i + fs]

                # Convert float32 -> int16 for RNNoise
                frame_int16 = (frame * 32767).clip(-32768, 32767).astype(np.int16)

                # Process through RNNoise
                denoised_int16, _vad_prob = self._rnnoise.process_mono_frame(
                    self._state, frame_int16
                )

                # Convert back to float32
                denoised = denoised_int16.astype(np.float32) / 32767.0

                # Blend based on intensity (1.0 = full denoise, 0.0 = no effect)
                if self._intensity < 1.0:
                    denoised = self._intensity * denoised + (1 - self._intensity) * frame

                output[i:i + fs] = denoised

            # Pass through remaining samples
            remainder = total_samples % fs
            if remainder > 0:
                output[total_samples - remainder:] = audio_data[total_samples - remainder:]

            return output

        except Exception as e:
            print(f"[NVIDIA Broadcast] Audio processing error: {e}")
            return audio_data

    def cleanup(self):
        """Release resources."""
        if self._state is not None:
            try:
                self._rnnoise.destroy(self._state)
            except Exception:
                pass
            self._state = None
        self._initialized = False

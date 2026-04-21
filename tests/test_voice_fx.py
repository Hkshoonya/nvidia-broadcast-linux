import unittest

import numpy as np

from nvbroadcast.audio.voice_fx import (
    DEFAULT_VOICE_FX_PRESET,
    VoiceFX,
    VoiceFXSettings,
    get_voice_fx_preset,
    normalize_voice_fx_preset_name,
)


class VoiceFXTests(unittest.TestCase):
    def test_legacy_preset_name_normalizes_to_flat(self):
        self.assertEqual(normalize_voice_fx_preset_name("Natural"), "Flat")

    def test_get_voice_fx_preset_returns_detached_copy(self):
        preset = get_voice_fx_preset(DEFAULT_VOICE_FX_PRESET)
        self.assertIsNotNone(preset)
        preset.bass_boost = 0.99
        fresh = get_voice_fx_preset(DEFAULT_VOICE_FX_PRESET)
        self.assertNotEqual(preset.bass_boost, fresh.bass_boost)

    def test_noise_gate_uses_reference_signal_when_provided(self):
        voice_fx = VoiceFX(use_gpu=False)
        voice_fx.enabled = True
        voice_fx.settings = VoiceFXSettings(gate_threshold=0.25)

        processed = np.full(4800, 1e-4, dtype=np.float32)
        loud_reference = np.full(4800, 0.01, dtype=np.float32)

        gated_without_reference = voice_fx.process_chunk(processed, gate_reference=processed)
        gated_with_reference = voice_fx.process_chunk(processed, gate_reference=loud_reference)

        self.assertLess(np.max(np.abs(gated_without_reference)), 2e-5)
        self.assertGreater(np.max(np.abs(gated_with_reference)), 5e-5)


if __name__ == "__main__":
    unittest.main()

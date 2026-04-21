import unittest

from nvbroadcast.core.config import (
    AppConfig,
    apply_builtin_profile,
    build_default_config,
    _config_to_toml,
    _load_from_toml,
)
from nvbroadcast.audio.voice_fx import DEFAULT_VOICE_FX_PRESET, get_voice_fx_preset


class ConfigPersistenceTests(unittest.TestCase):
    def test_roundtrip_persists_speaker_and_profile(self):
        config = AppConfig()
        config.current_profile = "Meeting"
        config.auto_mode = True
        config.mode_key = "cpu_light"
        config.ui_card_expanded = {
            "background": True,
            "voice_effects": False,
        }
        config.video.width = 800
        config.video.height = 600
        config.video.fps = 30
        config.video.output_format = "I420"
        config.audio.mic_device = "mic0"
        config.audio.speaker_device = "speaker0"
        config.audio.voice_fx_enabled = True
        config.audio.voice_fx_use_gpu = False
        config.audio.voice_fx_preset = "Podcast"
        config.audio.voice_fx_warmth = 0.4

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(_config_to_toml(config))
            loaded = _load_from_toml(path)

        self.assertEqual(loaded.current_profile, "Meeting")
        self.assertTrue(loaded.auto_mode)
        self.assertEqual(loaded.mode_key, "cpu_light")
        self.assertEqual(loaded.ui_card_expanded, {
            "background": True,
            "voice_effects": False,
        })
        self.assertEqual((loaded.video.width, loaded.video.height, loaded.video.fps), (800, 600, 30))
        self.assertEqual(loaded.video.output_format, "I420")
        self.assertEqual(loaded.audio.mic_device, "mic0")
        self.assertEqual(loaded.audio.speaker_device, "speaker0")
        self.assertTrue(loaded.audio.voice_fx_enabled)
        self.assertFalse(loaded.audio.voice_fx_use_gpu)
        self.assertEqual(loaded.audio.voice_fx_preset, "Podcast")
        self.assertEqual(loaded.audio.voice_fx_warmth, 0.4)

    def test_build_default_config_preserves_runtime_flags(self):
        existing = AppConfig()
        existing.first_run = False
        existing.auto_start = False
        existing.minimize_on_close = False
        existing.check_for_updates = False
        existing.last_update_check = 123
        existing.last_notified_version = "1.1.1"
        existing.compute_gpu = 2
        existing.auto_mode = True
        existing.current_profile = "Custom"
        existing.ui_card_expanded = {"background": True}
        existing.audio.speaker_device = "speaker0"
        existing.video.background_removal = True

        reset = build_default_config(existing)

        self.assertFalse(reset.first_run)
        self.assertFalse(reset.auto_start)
        self.assertFalse(reset.minimize_on_close)
        self.assertFalse(reset.check_for_updates)
        self.assertEqual(reset.last_update_check, 123)
        self.assertEqual(reset.last_notified_version, "1.1.1")
        self.assertEqual(reset.compute_gpu, 2)
        self.assertTrue(reset.auto_mode)
        self.assertEqual(reset.ui_card_expanded, {"background": True})
        self.assertEqual(reset.current_profile, "Default")
        self.assertEqual(reset.audio.speaker_device, "")
        self.assertFalse(reset.video.background_removal)

    def test_builtin_profiles_do_not_overwrite_manual_mode_or_capture_settings(self):
        config = AppConfig()
        config.auto_mode = False
        config.mode_key = "cpu_light"
        config.performance_profile = "performance"
        config.compositing = "cpu"
        config.video.width = 640
        config.video.height = 360
        config.video.fps = 30
        config.video.output_format = "I420"

        changed = apply_builtin_profile(config, "Meeting")

        self.assertTrue(changed)
        self.assertFalse(config.auto_mode)
        self.assertEqual(config.mode_key, "cpu_light")
        self.assertEqual(config.performance_profile, "performance")
        self.assertEqual(config.compositing, "cpu")
        self.assertEqual((config.video.width, config.video.height, config.video.fps), (640, 360, 30))
        self.assertEqual(config.video.output_format, "I420")

    def test_legacy_natural_voice_fx_defaults_migrate_to_audible_preset(self):
        legacy = """
[audio]
voice_fx_preset = "Natural"
voice_fx_bass_boost = 0.0
voice_fx_treble = 0.0
voice_fx_warmth = 0.0
voice_fx_compression = 0.0
voice_fx_gate_threshold = 0.0
voice_fx_gain = 0.0
"""

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.toml"
            path.write_text(legacy)
            loaded = _load_from_toml(path)

        expected = get_voice_fx_preset(DEFAULT_VOICE_FX_PRESET)
        self.assertIsNotNone(expected)
        self.assertEqual(loaded.audio.voice_fx_preset, DEFAULT_VOICE_FX_PRESET)
        self.assertEqual(loaded.audio.voice_fx_bass_boost, expected.bass_boost)
        self.assertEqual(loaded.audio.voice_fx_treble, expected.treble)
        self.assertEqual(loaded.audio.voice_fx_warmth, expected.warmth)
        self.assertEqual(loaded.audio.voice_fx_compression, expected.compression)
        self.assertEqual(loaded.audio.voice_fx_gate_threshold, expected.gate_threshold)
        self.assertEqual(loaded.audio.voice_fx_gain, expected.gain)

    def test_legacy_studio_gate_migrates_to_safer_default(self):
        legacy = """
[audio]
voice_fx_preset = "Studio"
voice_fx_bass_boost = 0.15
voice_fx_treble = 0.15
voice_fx_warmth = 0.25
voice_fx_compression = 0.7
voice_fx_gate_threshold = 0.25
voice_fx_gain = 0.05
"""

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_studio.toml"
            path.write_text(legacy)
            loaded = _load_from_toml(path)

        expected = get_voice_fx_preset("Studio")
        self.assertIsNotNone(expected)
        self.assertEqual(loaded.audio.voice_fx_preset, "Studio")
        self.assertEqual(loaded.audio.voice_fx_gate_threshold, expected.gate_threshold)


if __name__ == "__main__":
    unittest.main()

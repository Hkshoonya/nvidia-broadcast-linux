import unittest

from nvbroadcast.core.config import AppConfig, build_default_config, _config_to_toml, _load_from_toml


class ConfigPersistenceTests(unittest.TestCase):
    def test_roundtrip_persists_speaker_and_profile(self):
        config = AppConfig()
        config.current_profile = "Meeting"
        config.audio.mic_device = "mic0"
        config.audio.speaker_device = "speaker0"

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(_config_to_toml(config))
            loaded = _load_from_toml(path)

        self.assertEqual(loaded.current_profile, "Meeting")
        self.assertEqual(loaded.audio.mic_device, "mic0")
        self.assertEqual(loaded.audio.speaker_device, "speaker0")

    def test_build_default_config_preserves_runtime_flags(self):
        existing = AppConfig()
        existing.first_run = False
        existing.auto_start = False
        existing.minimize_on_close = False
        existing.check_for_updates = False
        existing.last_update_check = 123
        existing.last_notified_version = "1.1.1"
        existing.compute_gpu = 2
        existing.current_profile = "Custom"
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
        self.assertEqual(reset.current_profile, "Default")
        self.assertEqual(reset.audio.speaker_device, "")
        self.assertFalse(reset.video.background_removal)


if __name__ == "__main__":
    unittest.main()

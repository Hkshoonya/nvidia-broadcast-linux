import unittest
from unittest import mock

from nvbroadcast.core.config import (
    AppConfig,
    apply_builtin_profile,
    build_default_config,
    load_config,
    save_config,
    _config_to_toml,
    _load_from_toml,
)
from nvbroadcast.audio.voice_fx import DEFAULT_VOICE_FX_PRESET, get_voice_fx_preset


class ConfigPersistenceTests(unittest.TestCase):
    def test_roundtrip_persists_speaker_and_profile(self):
        config = AppConfig()
        config.current_profile = "Meeting"
        config.last_python_runtime_notice = "python-runtime-3.14"
        config.compute_focus = "gpu"
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
        config.video.vcam_device = "/dev/video11"
        config.video.auto_frame_mode = "stable"
        config.video.eye_contact_mode = "gaze_lock"
        config.video.blur_intensity = 0.9
        config.video.blur_dim = 0.4
        config.video.blur_desaturate = 0.75
        config.audio.mic_device = "mic0"
        config.audio.speaker_device = "speaker0"
        config.audio.voice_fx_enabled = True
        config.audio.voice_fx_use_gpu = False
        config.audio.voice_fx_preset = "Podcast"
        config.audio.voice_fx_warmth = 0.4
        config.hotkeys.enabled = True
        config.hotkeys.toggle_background = "<Control><Alt>b"
        config.hotkeys.toggle_auto_frame = "<Shift><Super>F12"

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(_config_to_toml(config))
            loaded = _load_from_toml(path)

        self.assertEqual(loaded.current_profile, "Meeting")
        self.assertEqual(loaded.last_python_runtime_notice, "python-runtime-3.14")
        self.assertEqual(loaded.compute_focus, "gpu")
        self.assertTrue(loaded.auto_mode)
        self.assertEqual(loaded.mode_key, "cpu_light")
        self.assertEqual(loaded.ui_card_expanded, {
            "background": True,
            "voice_effects": False,
        })
        self.assertEqual((loaded.video.width, loaded.video.height, loaded.video.fps), (800, 600, 30))
        self.assertEqual(loaded.video.output_format, "I420")
        self.assertEqual(loaded.video.vcam_device, "/dev/video11")
        self.assertEqual(loaded.video.auto_frame_mode, "stable")
        self.assertEqual(loaded.video.eye_contact_mode, "gaze_lock")
        self.assertEqual(loaded.video.blur_intensity, 0.9)
        self.assertEqual(loaded.video.blur_dim, 0.4)
        self.assertEqual(loaded.video.blur_desaturate, 0.75)
        self.assertEqual(loaded.audio.mic_device, "mic0")
        self.assertEqual(loaded.audio.speaker_device, "speaker0")
        self.assertTrue(loaded.audio.voice_fx_enabled)
        self.assertFalse(loaded.audio.voice_fx_use_gpu)
        self.assertEqual(loaded.audio.voice_fx_preset, "Podcast")
        self.assertEqual(loaded.audio.voice_fx_warmth, 0.4)
        self.assertTrue(loaded.hotkeys.enabled)
        self.assertEqual(loaded.hotkeys.toggle_background, "<Control><Alt>b")
        self.assertEqual(loaded.hotkeys.toggle_auto_frame, "<Shift><Super>F12")

    def test_build_default_config_preserves_runtime_flags(self):
        existing = AppConfig()
        existing.first_run = False
        existing.auto_start = False
        existing.minimize_on_close = False
        existing.check_for_updates = False
        existing.last_update_check = 123
        existing.last_notified_version = "1.1.1"
        existing.last_python_runtime_notice = "python-runtime-3.14"
        existing.compute_gpu = 2
        existing.compute_focus = "cpu"
        existing.auto_mode = True
        existing.current_profile = "Custom"
        existing.ui_card_expanded = {"background": True}
        existing.audio.speaker_device = "speaker0"
        existing.video.vcam_device = "/dev/video11"
        existing.video.background_removal = True
        existing.hotkeys.enabled = True
        existing.hotkeys.toggle_background = "<Control><Alt>b"

        reset = build_default_config(existing)

        self.assertFalse(reset.first_run)
        self.assertFalse(reset.auto_start)
        self.assertFalse(reset.minimize_on_close)
        self.assertFalse(reset.check_for_updates)
        self.assertEqual(reset.last_update_check, 123)
        self.assertEqual(reset.last_notified_version, "1.1.1")
        self.assertEqual(reset.last_python_runtime_notice, "python-runtime-3.14")
        self.assertEqual(reset.compute_gpu, 2)
        self.assertEqual(reset.compute_focus, "cpu")
        self.assertTrue(reset.auto_mode)
        self.assertEqual(reset.ui_card_expanded, {"background": True})
        self.assertEqual(reset.current_profile, "Default")
        self.assertEqual(reset.audio.speaker_device, "")
        self.assertEqual(reset.video.vcam_device, "/dev/video11")
        self.assertFalse(reset.video.background_removal)
        self.assertTrue(reset.hotkeys.enabled)
        self.assertEqual(reset.hotkeys.toggle_background, "<Control><Alt>b")

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

    def test_invalid_compute_focus_loads_as_auto(self):
        raw = 'compute_focus = "broken"\n'

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(raw)
            loaded = _load_from_toml(path)

        self.assertEqual(loaded.compute_focus, "auto")

    def test_invalid_autoframe_mode_loads_as_center(self):
        raw = '[video]\nauto_frame_mode = "broken"\n'

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(raw)
            loaded = _load_from_toml(path)

        self.assertEqual(loaded.video.auto_frame_mode, "center")

    def test_invalid_eye_contact_mode_loads_as_natural(self):
        raw = '[video]\neye_contact_mode = "broken"\n'

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(raw)
            loaded = _load_from_toml(path)

        self.assertEqual(loaded.video.eye_contact_mode, "natural")

    def test_invalid_hotkey_types_and_control_characters_are_ignored(self):
        raw = """
[hotkeys]
enabled = "yes"
toggle_background = 42
toggle_auto_frame = "<Control><Alt>a\\n"
toggle_eye_contact = "<Control><Alt>e"
"""

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(raw)
            loaded = _load_from_toml(path)

        self.assertFalse(loaded.hotkeys.enabled)
        self.assertEqual(loaded.hotkeys.toggle_background, "")
        self.assertEqual(loaded.hotkeys.toggle_auto_frame, "")
        self.assertEqual(loaded.hotkeys.toggle_eye_contact, "<Control><Alt>e")

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


class ConfigCorruptionRecoveryTests(unittest.TestCase):
    """save_config must never leave a state that load_config cannot recover.

    A truncated write used to leave invalid TOML, and load_config silently
    returned defaults, so the next save persisted a wiped config."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_dir = Path(self._tmp.name)
        self.config_file = self.config_dir / "config.toml"
        patches = {
            "CONFIG_DIR": self.config_dir,
            "CONFIG_FILE": self.config_file,
        }
        for name, value in patches.items():
            patcher = mock.patch(f"nvbroadcast.core.config.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _save(self, profile: str):
        config = AppConfig()
        config.current_profile = profile
        save_config(config)

    def test_save_then_load_roundtrip(self):
        self._save("Meeting")
        self.assertEqual(load_config().current_profile, "Meeting")

    def test_corrupt_config_falls_back_to_backup(self):
        self._save("Meeting")
        self._save("Streaming")  # Previous save becomes config.toml.bak
        self.config_file.write_text("current_profile = \"trunc")  # Crash mid-write
        self.assertEqual(load_config().current_profile, "Meeting")

    def test_corrupt_config_without_backup_returns_defaults(self):
        self.config_file.write_text("not toml [[[")
        loaded = load_config()
        self.assertEqual(loaded.current_profile, AppConfig().current_profile)

    def test_save_leaves_no_temp_file(self):
        self._save("Meeting")
        self.assertEqual(
            sorted(p.name for p in self.config_dir.iterdir()),
            ["config.toml"],
        )
        self._save("Streaming")
        self.assertEqual(
            sorted(p.name for p in self.config_dir.iterdir()),
            ["config.toml", "config.toml.bak"],
        )


if __name__ == "__main__":
    unittest.main()

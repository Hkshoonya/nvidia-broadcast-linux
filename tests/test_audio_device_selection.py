"""Regression coverage for microphone selection and persistence."""

from types import SimpleNamespace
import unittest
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig
from nvbroadcast.ui.window import NVBroadcastWindow


class MicrophoneSelectionTests(unittest.TestCase):
    @staticmethod
    def _window(mics, saved=""):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        window._mic_selector = mock.Mock()
        window._app = SimpleNamespace(
            config=SimpleNamespace(audio=SimpleNamespace(mic_device=saved)),
            list_microphones=mock.Mock(return_value=mics),
            set_microphone=mock.Mock(),
        )
        return window

    def test_first_visible_microphone_is_persisted_on_first_launch(self):
        mics = [{"name": "Desk Mic", "device": "alsa_input.desk"}]
        window = self._window(mics)

        window._populate_mics()

        window._mic_selector.set_devices.assert_called_once_with(mics)
        window._mic_selector.set_selected_index.assert_called_once_with(0)
        window._app.set_microphone.assert_called_once_with("alsa_input.desk")

    def test_saved_microphone_remains_selected_without_rewriting_config(self):
        mics = [
            {"name": "Desk Mic", "device": "alsa_input.desk"},
            {"name": "Headset", "device": "alsa_input.headset"},
        ]
        window = self._window(mics, saved="alsa_input.headset")

        window._populate_mics()

        window._mic_selector.set_selected_index.assert_called_once_with(1)
        window._app.set_microphone.assert_not_called()

    def test_missing_saved_microphone_falls_back_to_visible_device(self):
        mics = [{"name": "Desk Mic", "device": "alsa_input.desk"}]
        window = self._window(mics, saved="alsa_input.unplugged")

        window._populate_mics()

        window._mic_selector.set_selected_index.assert_called_once_with(0)
        window._app.set_microphone.assert_called_once_with("alsa_input.desk")

    def test_default_fallback_clears_a_missing_saved_microphone(self):
        mics = [{"name": "Default Microphone", "device": ""}]
        window = self._window(mics, saved="alsa_input.unplugged")

        window._populate_mics()

        window._mic_selector.set_selected_index.assert_called_once_with(0)
        window._app.set_microphone.assert_called_once_with("")

    def test_empty_device_list_does_not_change_configuration(self):
        window = self._window([])

        window._populate_mics()

        window._mic_selector.set_devices.assert_called_once_with([])
        window._mic_selector.set_selected_index.assert_not_called()
        window._app.set_microphone.assert_not_called()

    @mock.patch("nvbroadcast.app.save_config")
    def test_application_persists_selected_microphone(self, save_config):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app._audio_pipeline = None

        app.set_microphone("alsa_input.desk")

        self.assertEqual(app.config.audio.mic_device, "alsa_input.desk")
        save_config.assert_called_once_with(app.config)

    @mock.patch("nvbroadcast.app.save_config")
    def test_application_rebuilds_running_audio_for_new_microphone(self, save_config):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app._audio_pipeline = SimpleNamespace(_running=True)
        app._rebuild_audio_pipeline = mock.Mock()

        app.set_microphone("alsa_input.headset")

        app._rebuild_audio_pipeline.assert_called_once_with(restart=True)
        save_config.assert_called_once_with(app.config)


if __name__ == "__main__":
    unittest.main()

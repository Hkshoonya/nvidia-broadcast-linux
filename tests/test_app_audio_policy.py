import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp


class AppAudioPolicyTests(unittest.TestCase):
    @staticmethod
    def _fake_app(*, noise_removal=False, voice_fx_enabled=False):
        fake = SimpleNamespace(
            config=SimpleNamespace(
                audio=SimpleNamespace(
                    noise_removal=noise_removal,
                    voice_fx_enabled=voice_fx_enabled,
                )
            )
        )
        fake._audio_pipeline_should_publish = lambda: NVBroadcastApp._audio_pipeline_should_publish(fake)
        return fake

    @mock.patch("nvbroadcast.app.has_virtual_mic_backend", return_value=True)
    def test_audio_pipeline_runs_as_passthrough_when_virtual_mic_backend_exists(self, _backend):
        fake = self._fake_app(noise_removal=False, voice_fx_enabled=False)
        self.assertTrue(NVBroadcastApp._audio_pipeline_should_publish(fake))
        self.assertTrue(NVBroadcastApp._audio_pipeline_should_run(fake))

    @mock.patch("nvbroadcast.app.has_virtual_mic_backend", return_value=False)
    def test_audio_pipeline_does_not_run_without_backend_or_effects(self, _backend):
        fake = self._fake_app(noise_removal=False, voice_fx_enabled=False)
        self.assertFalse(NVBroadcastApp._audio_pipeline_should_publish(fake))
        self.assertFalse(NVBroadcastApp._audio_pipeline_should_run(fake))

    @mock.patch("nvbroadcast.app.has_virtual_mic_backend", return_value=False)
    def test_audio_pipeline_runs_without_backend_when_effects_enabled(self, _backend):
        fake = self._fake_app(noise_removal=True, voice_fx_enabled=False)
        self.assertTrue(NVBroadcastApp._audio_pipeline_should_run(fake))

    @mock.patch("nvbroadcast.app.save_config")
    def test_camera_power_save_toggle_does_not_restart_audio(self, save_config):
        fake = SimpleNamespace(
            config=SimpleNamespace(auto_idle=True),
            _idle_active=False,
            _idle_strikes=2,
            _audio_pipeline=mock.Mock(),
            _restart_audio_pipeline_for_live_settings=mock.Mock(),
        )

        NVBroadcastApp.set_auto_idle(fake, False)

        self.assertFalse(fake.config.auto_idle)
        self.assertEqual(fake._idle_strikes, 0)
        save_config.assert_called_once_with(fake.config)
        fake._restart_audio_pipeline_for_live_settings.assert_not_called()

    def test_transcriber_preload_waits_while_streaming(self):
        fake = SimpleNamespace(
            _meeting_active=False,
            _meeting_finalizing=False,
            _streaming=True,
            _preload_transcriber=mock.Mock(),
        )
        self.assertTrue(NVBroadcastApp._preload_transcriber_when_idle(fake))
        fake._preload_transcriber.assert_not_called()

    def test_transcriber_preload_runs_once_idle(self):
        fake = SimpleNamespace(
            _meeting_active=False,
            _meeting_finalizing=False,
            _streaming=False,
            _preload_transcriber=mock.Mock(),
        )
        self.assertFalse(NVBroadcastApp._preload_transcriber_when_idle(fake))
        fake._preload_transcriber.assert_called_once_with()

    @mock.patch("nvbroadcast.app.time.sleep")
    @mock.patch("nvbroadcast.app.subprocess.run")
    @mock.patch("nvbroadcast.app.IS_LINUX", True)
    def test_gui_startup_stops_active_headless_vcam_service(self, run, _sleep):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        run.side_effect = [
            mock.Mock(returncode=0),
            mock.Mock(returncode=0),
        ]

        self.assertTrue(NVBroadcastApp._stop_headless_vcam_service(app))

        self.assertEqual(run.call_args_list[0].args[0], [
            "systemctl", "--user", "is-active", "--quiet", "nvbroadcast-vcam.service",
        ])
        self.assertEqual(run.call_args_list[1].args[0], [
            "systemctl", "--user", "stop", "nvbroadcast-vcam.service",
        ])

    @mock.patch("nvbroadcast.app.subprocess.run", return_value=mock.Mock(returncode=3))
    @mock.patch("nvbroadcast.app.IS_LINUX", True)
    def test_gui_startup_leaves_inactive_headless_vcam_service_alone(self, run):
        app = NVBroadcastApp.__new__(NVBroadcastApp)

        self.assertFalse(NVBroadcastApp._stop_headless_vcam_service(app))
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()

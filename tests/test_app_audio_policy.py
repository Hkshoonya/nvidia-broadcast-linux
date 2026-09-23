import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.ui.window import NVBroadcastWindow


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

    @mock.patch("nvbroadcast.app.subprocess.run")
    @mock.patch("nvbroadcast.app.IS_LINUX", True)
    @mock.patch.dict("nvbroadcast.app.os.environ", {"SNAP": "/snap/nvbroadcast/1"})
    def test_snap_startup_skips_host_headless_service(self, run):
        app = NVBroadcastApp.__new__(NVBroadcastApp)

        self.assertFalse(NVBroadcastApp._stop_headless_vcam_service(app))
        run.assert_not_called()

    @mock.patch(
        "nvbroadcast.app.subprocess.run",
        side_effect=PermissionError("strict confinement"),
    )
    @mock.patch("nvbroadcast.app.IS_LINUX", True)
    def test_gui_startup_tolerates_denied_systemctl(self, run):
        app = NVBroadcastApp.__new__(NVBroadcastApp)

        self.assertFalse(NVBroadcastApp._stop_headless_vcam_service(app))
        run.assert_called_once()

    def test_meeting_cannot_replace_an_active_rec_file(self):
        pipeline = mock.Mock(is_recording=True)
        app = SimpleNamespace(_video_pipeline=pipeline, _meeting_finalizing=False)
        with mock.patch("nvbroadcast.app.create_session") as create_session:
            self.assertEqual(NVBroadcastApp.start_meeting(app), "")
        create_session.assert_not_called()
        pipeline.start_recording.assert_not_called()
        pipeline.stop_recording.assert_not_called()

    def test_rec_cannot_replace_an_active_meeting_file(self):
        pipeline = mock.Mock(is_recording=True)
        app = SimpleNamespace(_meeting_active=True, _meeting_finalizing=False,
                              _video_pipeline=pipeline)
        self.assertEqual(NVBroadcastApp.start_recording(app), "")
        NVBroadcastApp.stop_recording(app)
        pipeline.start_recording.assert_not_called()
        pipeline.stop_recording.assert_not_called()

    def test_record_button_cannot_stop_a_meeting(self):
        app = SimpleNamespace(meeting_active=True, is_recording=True,
                              stop_recording=mock.Mock())
        window = SimpleNamespace(_app=app, set_status=mock.Mock())
        NVBroadcastWindow._on_record_toggle(window, None)
        app.stop_recording.assert_not_called()
        window.set_status.assert_called_once_with("End the meeting before using Rec")

    def test_meeting_button_requires_stopping_rec_first(self):
        app = SimpleNamespace(meeting_finalizing=False, meeting_active=False,
                              is_recording=True, recording_finalizing=False,
                              start_meeting=mock.Mock())
        window = SimpleNamespace(_app=app, set_status=mock.Mock())
        NVBroadcastWindow._on_meeting_toggle(window, None)
        app.start_meeting.assert_not_called()
        window.set_status.assert_called_once_with(
            "Stop or finish Rec before starting a meeting"
        )

    def test_recording_error_clears_failed_meeting_video_path(self):
        window = SimpleNamespace(on_recording_error=mock.Mock())
        app = SimpleNamespace(_meeting_active=True, _meeting_video_path="meeting.mp4",
                              _last_recording_path="meeting.mp4", _window=window)
        NVBroadcastApp._on_recording_error(app, "audio source disconnected")
        self.assertEqual(app._meeting_video_path, "")
        self.assertEqual(app._last_recording_path, "")
        window.on_recording_error.assert_called_once_with("audio source disconnected")

    def test_recording_error_resets_rec_button(self):
        button = mock.Mock()
        window = SimpleNamespace(
            _app=SimpleNamespace(meeting_active=False), _record_btn=button,
            set_status=mock.Mock(),
        )
        NVBroadcastWindow.on_recording_error(window, "audio source disconnected")
        button.set_label.assert_called_once_with("Rec")
        window.set_status.assert_called_once_with(
            "Recording stopped after an audio/video error; "
            "restart the app if Rec remains unavailable"
        )

    def test_rec_explains_cleanup_block_after_recording_error(self):
        pipeline = SimpleNamespace(
            is_recording=False, recording_finalizing=True,
            recording_audio_error="Microphone disconnected",
        )
        app = SimpleNamespace(
            _meeting_active=False, _meeting_finalizing=False,
            _video_pipeline=pipeline,
        )
        self.assertEqual(NVBroadcastApp.start_recording(app), "")
        self.assertIn("restart the app", app._recording_start_error)

    def test_meeting_video_finalize_does_not_replace_transcription_status(self):
        window = SimpleNamespace(on_recording_finalized=mock.Mock())
        app = SimpleNamespace(_window=window, _meeting_active=False,
                              _meeting_finalizing=True, is_recording=False)
        self.assertFalse(NVBroadcastApp._on_recording_finalized(
            app, "meeting", True, ""
        ))
        window.on_recording_finalized.assert_not_called()

    def test_meeting_capture_error_warns_without_claiming_audio(self):
        capture = SimpleNamespace(running=False)
        window = SimpleNamespace(set_status=mock.Mock())
        app = SimpleNamespace(
            _meeting_capture=capture, _meeting_active=True, _window=window,
        )
        self.assertFalse(NVBroadcastApp.meeting_audio_capture_present.fget(app))
        NVBroadcastApp._on_meeting_capture_error(app, capture, "can't connect")
        window.set_status.assert_called_once_with(
            "Meeting transcription audio stopped; check audio devices"
        )

    def test_meeting_capture_start_failure_releases_failed_pipeline(self):
        import tempfile
        from pathlib import Path

        capture = mock.Mock()
        capture.start.side_effect = RuntimeError("source could not start")
        pipeline = mock.Mock(is_recording=False, recording_finalizing=False)
        transcriber = mock.Mock()
        transcriber.start.return_value = True
        app = SimpleNamespace(
            _video_pipeline=pipeline, _meeting_finalizing=False,
            _meeting_active=False, _window=None, _transcriber=transcriber,
            config=SimpleNamespace(audio=SimpleNamespace(
                mic_device="", speaker_device=""
            )),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "nvbroadcast.app.create_session",
            return_value=("session", Path(directory)),
        ), mock.patch(
            "nvbroadcast.app.MeetingAudioCapture", return_value=capture,
        ):
            self.assertTrue(NVBroadcastApp.start_meeting(app))

        capture.stop.assert_called_once_with()
        self.assertIsNone(app._meeting_capture)

    def test_missing_h264_encoder_is_reported_to_recording_ui(self):
        import tempfile
        from pathlib import Path

        pipeline = mock.Mock(is_recording=False, recording_finalizing=False)
        pipeline.start_recording.side_effect = RuntimeError(
            "No H.264 recording encoder is installed"
        )
        app = SimpleNamespace(
            _meeting_active=False, _meeting_finalizing=False,
            _video_pipeline=pipeline, _idle_active=False,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            Path, "home", return_value=Path(directory)
        ):
            self.assertEqual(NVBroadcastApp.start_recording(app), "")
        self.assertEqual(app._recording_start_error, "H.264 encoder missing")


if __name__ == "__main__":
    unittest.main()

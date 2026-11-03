"""CoreAudio routing tests; no device or native PLAYING state is used."""

import unittest
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.audio import devices, meeting_capture, mic_test, monitor, pipeline, source_probe
from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.ui.window import NVBroadcastWindow
from nvbroadcast.video.pipeline import VideoPipeline


class MacOSAudioRoutingTests(unittest.TestCase):
    @staticmethod
    def _make_coreaudio_device(name, device_id, device_class, unique_id=None):
        device = mock.Mock()
        device.get_display_name.return_value = name
        device.get_device_class.return_value = device_class
        device.get_property.return_value = device_id
        device.get_properties.return_value.get_string.return_value = unique_id
        return device

    def test_processed_microphone_capture_uses_coreaudio(self):
        capture = pipeline.AudioPipeline(use_helper_process=False)
        elements = {}

        def make(factory, name):
            elements[factory] = mock.Mock()
            return elements[factory]

        with mock.patch.object(pipeline, "IS_LINUX", False), \
             mock.patch.object(pipeline, "IS_MACOS", True, create=True), \
             mock.patch.object(pipeline.Gst.Pipeline, "new"), \
             mock.patch.object(pipeline.Gst.ElementFactory, "make", side_effect=make):
            capture._build_capture_pipeline()
        self.assertIn("osxaudiosrc", elements)
        self.assertNotIn("pipewiresrc", elements)
        elements["osxaudiosrc"].set_property.assert_any_call("device", 0)

    def test_selected_processing_microphone_gets_integer_coreaudio_id(self):
        capture = pipeline.AudioPipeline(use_helper_process=False)
        capture.configure("coreaudio:uid:USB%20Mic")
        source = mock.Mock()
        with mock.patch.object(pipeline, "IS_MACOS", True), \
             mock.patch.object(pipeline.Gst.Pipeline, "new"), \
             mock.patch.object(pipeline.Gst.ElementFactory, "make", return_value=source), \
             mock.patch.object(pipeline, "resolve_coreaudio_device", return_value=42) as resolve:
            capture._build_capture_pipeline()
        resolve.assert_called_once_with("coreaudio:uid:USB%20Mic")
        source.set_property.assert_any_call("device", 42)

    def test_missing_coreaudio_capture_plugin_reports_installation_error(self):
        capture = pipeline.AudioPipeline(use_helper_process=False)
        with mock.patch.object(pipeline, "IS_MACOS", True), \
             mock.patch.object(pipeline.Gst.Pipeline, "new"), \
             mock.patch.object(pipeline.Gst.ElementFactory, "make", return_value=None), \
             self.assertRaisesRegex(RuntimeError, "CoreAudio microphone capture is not installed"):
            capture._build_capture_pipeline()

    def test_unconfigured_processed_output_cannot_play_into_mac_speakers(self):
        capture = pipeline.AudioPipeline(use_helper_process=False)
        factories = []

        def make(factory, name):
            factories.append(factory)
            return mock.Mock()

        with mock.patch.object(pipeline, "IS_LINUX", False), \
             mock.patch.object(pipeline, "IS_MACOS", True, create=True), \
             mock.patch.object(pipeline.Gst.Pipeline, "new"), \
             mock.patch.object(pipeline.Gst.ElementFactory, "make", side_effect=make):
            capture._build_output_pipeline()
        self.assertIn("fakesink", factories)
        self.assertNotIn("pipewiresink", factories)
        self.assertNotIn("osxaudiosink", factories)
        self.assertNotIn("autoaudiosink", factories)

    def test_recording_probe_uses_coreaudio_without_linux_fallback(self):
        with mock.patch.object(source_probe, "IS_MACOS", True, create=True), \
             mock.patch.object(source_probe.Gst.ElementFactory, "find", return_value=object()), \
             mock.patch.object(source_probe.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            source, error = source_probe.probe_audio_source()
        self.assertEqual((source, error), ("osxaudiosrc", ""))
        self.assertEqual(run.call_args.args[0][-1], "osxaudiosrc")
        self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_coreaudio_probe_transports_an_integer_as_a_separate_argument(self):
        selection = 'coreaudio:uid:USB%20%22Mic%22%20%21'
        with mock.patch.object(source_probe, "IS_MACOS", True), \
             mock.patch.object(source_probe.Gst.ElementFactory, "find", return_value=object()), \
             mock.patch.object(devices, "resolve_coreaudio_device", return_value=88), \
             mock.patch.object(source_probe.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(source_probe.probe_audio_source(selection), ("osxaudiosrc", ""))
        args = run.call_args.args[0]
        self.assertEqual(args[-3:], ["osxaudiosrc", "device", "88"])
        self.assertNotIn(selection, args[2])
        self.assertIn("int(sys.argv[3])", args[2])

    def test_coreaudio_probe_child_sets_a_typed_device_property(self):
        from gi.repository import Gst
        with mock.patch.object(source_probe, "IS_MACOS", True), \
             mock.patch.object(source_probe.Gst.ElementFactory, "find", return_value=object()), \
             mock.patch.object(devices, "resolve_coreaudio_device", return_value=42), \
             mock.patch.object(source_probe.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            source_probe.probe_audio_source("coreaudio:uid:USB")
        recording = mock.Mock()
        recording.set_state.return_value = Gst.StateChangeReturn.SUCCESS
        recording.get_bus.return_value.timed_pop_filtered.return_value.type = Gst.MessageType.EOS
        with mock.patch.object(sys, "argv", ["probe", "osxaudiosrc", "device", "42"]), \
             mock.patch.object(Gst, "parse_launch", return_value=recording):
            exec(compile(run.call_args.args[0][2], "audio_probe_child", "exec"), {})
        recording.get_by_name("probe_source").set_property.assert_called_once_with("device", 42)
        recording.set_state.assert_any_call(Gst.State.NULL)

    def test_linux_capture_transport_and_device_properties_are_preserved(self):
        for backend, factory, prop in (("pulse", "pulsesrc", "device"), ("pw-loopback", "pipewiresrc", "target-object")):
            with self.subTest(backend=backend):
                capture = pipeline.AudioPipeline(use_helper_process=False)
                capture.configure("alsa_input.selected")
                capture._virtual_mic_backend = backend
                elements = {}

                def make(factory, name):
                    elements[name] = mock.Mock()
                    return elements[name]

                with mock.patch.object(pipeline, "IS_MACOS", False), \
                     mock.patch.object(pipeline, "IS_LINUX", True), \
                     mock.patch.object(pipeline.Gst.Pipeline, "new"), \
                     mock.patch.object(pipeline.Gst.ElementFactory, "make", side_effect=make) as factories:
                    capture._build_capture_pipeline()
                factories.assert_any_call(factory, "mic-source")
                elements["mic-source"].set_property.assert_any_call(prop, "alsa_input.selected")

    def test_failed_coreaudio_probe_does_not_fall_back_to_linux(self):
        with mock.patch.object(source_probe, "IS_MACOS", True), \
             mock.patch.object(source_probe.Gst.ElementFactory, "find", return_value=object()) as find, \
             mock.patch.object(source_probe.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="Microphone permission denied")) as run:
            source, error = source_probe.probe_audio_source()
        self.assertIsNone(source)
        self.assertIn("Microphone permission denied", error)
        find.assert_called_once_with("osxaudiosrc")
        run.assert_called_once()

    def test_stale_coreaudio_microphone_never_probes_the_default(self):
        with mock.patch.object(source_probe, "IS_MACOS", True), \
             mock.patch.object(source_probe.Gst.ElementFactory, "find", return_value=object()), \
             mock.patch.object(devices, "_coreaudio_devices", return_value=[]), \
             mock.patch.object(source_probe.subprocess, "run") as run:
            source, error = source_probe.probe_audio_source("coreaudio:uid:missing")
        self.assertIsNone(source)
        self.assertIn("Selected microphone is unavailable", error)
        run.assert_not_called()

    def test_coreaudio_provider_filters_inputs_and_preserves_persistent_ids(self):
        from gi.repository import Gst
        provider = mock.Mock()
        provider.get_devices.return_value = [
            self._make_coreaudio_device("USB Mic", 42, "Audio/Source", 'USB "Mic" /1'),
            self._make_coreaudio_device("Speakers", 43, "Audio/Sink", "Speakers"),
            self._make_coreaudio_device("Legacy Mic", 44, "Audio/Source"),
        ]
        with mock.patch.object(Gst.DeviceProviderFactory, "get_by_name", return_value=provider) as get:
            entries = devices._coreaudio_devices("Audio/Source")
        self.assertEqual(entries, [
            {"name": "USB Mic", "device": "coreaudio:uid:USB%20%22Mic%22%20%2F1", "device_id": 42},
            {"name": "Legacy Mic", "device": "coreaudio:id:44", "device_id": 44},
        ])
        get.assert_called_once_with("osxaudiodeviceprovider")
        provider.start.assert_not_called()
        provider.stop.assert_not_called()

    def test_coreaudio_enumeration_is_deduped_and_avoids_linux_commands(self):
        entry = {"name": "USB Mic", "device": "coreaudio:uid:USB", "device_id": 42}
        with mock.patch.object(devices, "IS_MACOS", True), \
             mock.patch.object(devices, "_coreaudio_devices", return_value=[entry, entry]), \
             mock.patch.object(devices.subprocess, "run") as run:
            self.assertEqual(devices.list_microphones(), [{"name": "USB Mic", "device": "coreaudio:uid:USB"}])
            self.assertEqual(devices.list_speakers(), [{"name": "USB Mic", "device": "coreaudio:uid:USB"}])
            self.assertEqual(devices.default_speaker_device(), "")
        run.assert_not_called()

    def test_coreaudio_default_remains_available_when_provider_is_missing(self):
        from gi.repository import Gst
        with mock.patch.object(devices, "IS_MACOS", True), \
             mock.patch.object(Gst.DeviceProviderFactory, "get_by_name", return_value=None):
            self.assertEqual(devices.list_microphones(), [{"name": "Default Microphone", "device": ""}])
            self.assertEqual(devices.list_speakers(), [{"name": "Default Speaker", "device": ""}])
        self.assertEqual(devices.resolve_coreaudio_device(""), 0)

    def test_saved_unique_id_follows_current_device_id_after_reconnect(self):
        with mock.patch.object(devices, "_coreaudio_devices", return_value=[
            {"name": "USB Mic", "device": "coreaudio:uid:USB%20Mic", "device_id": 92},
        ]):
            self.assertEqual(devices.resolve_coreaudio_device("coreaudio:uid:USB%20Mic"), 92)

    def test_legacy_numeric_coreaudio_id_must_still_exist(self):
        with mock.patch.object(devices, "_coreaudio_devices", return_value=[
            {"name": "Legacy", "device": "coreaudio:id:44", "device_id": 44},
        ]):
            self.assertEqual(devices.resolve_coreaudio_device("44"), 44)
            self.assertEqual(devices.resolve_coreaudio_device("coreaudio:id:44"), 44)
            with self.assertRaisesRegex(ValueError, "Selected microphone is unavailable"):
                devices.resolve_coreaudio_device("45")

    def test_selected_speaker_is_resolved_only_among_outputs(self):
        with mock.patch.object(devices, "_coreaudio_devices", return_value=[]) as get:
            with self.assertRaisesRegex(ValueError, "Selected speaker is unavailable"):
                devices.resolve_coreaudio_device("coreaudio:uid:USB", "Audio/Sink")
        get.assert_called_once_with("Audio/Sink")

    def test_mic_test_records_the_selected_coreaudio_microphone(self):
        test = mic_test.MicTest()
        recording = mock.Mock()
        selection = "coreaudio:uid:USB%20Mic"
        with mock.patch.object(mic_test, "IS_MACOS", True), \
             mock.patch.object(mic_test, "resolve_coreaudio_device", return_value=42) as resolve, \
             mock.patch.object(mic_test.Gst, "parse_launch", return_value=recording) as parse, \
             mock.patch.object(mic_test.threading, "Thread"):
            test.start_recording(selection, duration=15)
        self.assertTrue(test.is_recording)
        self.assertIn("osxaudiosrc name=test_microphone", parse.call_args.args[0])
        self.assertNotIn(selection, parse.call_args.args[0])
        resolve.assert_called_once_with(selection)
        recording.get_by_name("test_microphone").set_property.assert_called_once_with("device", 42)

    def test_stale_mic_test_selection_does_not_start_capture(self):
        test = mic_test.MicTest()
        with mock.patch.object(mic_test, "IS_MACOS", True), \
             mock.patch.object(mic_test, "resolve_coreaudio_device", side_effect=ValueError("Selected microphone is unavailable")), \
             mock.patch.object(mic_test.Gst, "parse_launch") as parse:
            test.start_recording("coreaudio:uid:missing")
        self.assertFalse(test.is_recording)
        parse.assert_not_called()

    def test_mic_test_plays_only_on_selected_coreaudio_output(self):
        test = mic_test.MicTest()
        playback = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            test._test_file = str(Path(directory) / "test.wav")
            Path(test._test_file).write_bytes(b"RIFFfake")
            with mock.patch.object(mic_test, "IS_MACOS", True), \
                 mock.patch.object(mic_test, "resolve_coreaudio_device", return_value=43) as resolve, \
                 mock.patch.object(mic_test.Gst, "parse_launch", return_value=playback) as parse:
                test.play_recording("coreaudio:uid:Speakers")
        self.assertTrue(test.is_playing)
        self.assertIn("osxaudiosink name=test_speaker", parse.call_args.args[0])
        resolve.assert_called_once_with("coreaudio:uid:Speakers", "Audio/Sink")
        playback.get_by_name("test_speaker").set_property.assert_called_once_with("device", 43)

    def test_mic_test_capture_start_failure_resets_the_pipeline(self):
        from gi.repository import Gst
        test = mic_test.MicTest()
        recording = mock.Mock()
        recording.set_state.return_value = Gst.StateChangeReturn.FAILURE
        with mock.patch.object(mic_test, "IS_MACOS", True), \
             mock.patch.object(mic_test.Gst, "parse_launch", return_value=recording), \
             mock.patch.object(mic_test.threading, "Thread") as thread:
            test.start_recording()
        self.assertFalse(test.is_recording)
        self.assertIsNone(test._rec_pipeline)
        recording.set_state.assert_any_call(Gst.State.NULL)
        thread.assert_not_called()

    def test_mic_test_permission_error_skips_eos_and_never_claims_success(self):
        test = mic_test.MicTest()
        recording = mock.Mock()
        error = mock.Mock()
        error.parse_error.return_value = (SimpleNamespace(message="Microphone permission denied"), "")
        recording.get_bus.return_value.timed_pop_filtered.return_value = error
        test._rec_pipeline = recording
        test._recording = True
        test._stop_recording()
        self.assertFalse(test.is_recording)
        self.assertIsNone(test._rec_pipeline)
        self.assertEqual(test.last_error, "Microphone permission denied")
        recording.send_event.assert_not_called()

    def test_mic_test_finalize_timeout_never_claims_success(self):
        test = mic_test.MicTest()
        recording = mock.Mock()
        recording.get_bus.return_value.timed_pop_filtered.return_value = None
        test._rec_pipeline = recording
        test._recording = True
        test._stop_recording()
        self.assertEqual(test.last_error, "Microphone recording did not finish")
        self.assertFalse(test.is_recording)

    def test_failed_mic_test_is_visible_and_cannot_be_played(self):
        test = mock.Mock(is_recording=False, last_error="Microphone permission denied")
        test.start_recording.side_effect = lambda *args, **kwargs: kwargs["on_complete"]()
        window = SimpleNamespace(
            _mic_test=test, _mic_selector=mock.Mock(), _test_duration_selector=mock.Mock(),
            _test_source_selector=mock.Mock(), _test_status=mock.Mock(),
            _test_rec_btn=mock.Mock(), _test_play_btn=mock.Mock(),
        )
        window._test_duration_selector.get_selected_device.return_value = "15"
        window._test_source_selector.get_selected_device.return_value = "original"
        with mock.patch("nvbroadcast.ui.window.IS_MACOS", True):
            NVBroadcastWindow._on_test_record(window, mock.Mock())
        window._test_play_btn.set_sensitive.assert_called_with(False)
        self.assertIn("Recording failed: Microphone permission denied", window._test_status.set_text.call_args.args[0])

    def test_meeting_capture_uses_selected_mic_and_explains_no_system_audio(self):
        capture = meeting_capture.MeetingAudioCapture()
        elements = {}

        def make(factory, name):
            elements[name] = mock.Mock()
            return elements[name]

        with mock.patch.object(meeting_capture, "IS_MACOS", True), \
             mock.patch.object(meeting_capture, "probe_audio_source", return_value=("osxaudiosrc", "")) as probe, \
             mock.patch.object(meeting_capture, "resolve_coreaudio_device", return_value=42), \
             mock.patch.object(meeting_capture.Gst.Pipeline, "new"), \
             mock.patch.object(meeting_capture.Gst.ElementFactory, "make", side_effect=make):
            capture.build("coreaudio:uid:USB", "coreaudio:uid:Speakers", "/tmp/meeting.wav")
        probe.assert_called_once_with("coreaudio:uid:USB")
        elements["mic-src"].set_property.assert_any_call("device", 42)
        self.assertNotIn("speaker-src", elements)
        self.assertIn("microphone only", capture.route_warning)
        self.assertIn("system audio capture is unavailable", capture.route_warning)

    def test_mac_speaker_denoise_never_substitutes_a_microphone(self):
        speaker = monitor.SpeakerMonitor()
        with mock.patch.object(monitor, "IS_MACOS", True), \
             mock.patch.object(monitor.Gst.Pipeline, "new") as new, \
             mock.patch.object(monitor.Gst.ElementFactory, "find") as find, \
             self.assertRaisesRegex(RuntimeError, "system audio capture is not implemented"):
            speaker.build()
        new.assert_not_called()
        find.assert_not_called()

    def test_mp4_recording_sets_coreaudio_integer_device_outside_graph(self):
        from gi.repository import Gst
        video = VideoPipeline()
        video._recording_aac_candidates = [("avenc_aac", "F32LE")]
        recording = mock.Mock()
        recording.set_state.return_value = Gst.StateChangeReturn.SUCCESS
        selection = 'coreaudio:uid:USB%20%22Mic%22'
        with mock.patch.object(video, "_recording_audio_source", return_value=("osxaudiosrc", "")), \
             mock.patch.object(video, "_select_recording_encoder", return_value="videoconvert ! x264enc ! h264parse"), \
             mock.patch.object(devices, "resolve_coreaudio_device", return_value=42), \
             mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch", return_value=recording) as parse:
            video.start_recording("/tmp/recording.mp4", mic_device=selection)
        self.assertTrue(video.recording_has_audio)
        self.assertIn("osxaudiosrc name=recording_audio_source", parse.call_args.args[0])
        self.assertNotIn(selection, parse.call_args.args[0])
        recording.get_by_name("recording_audio_source").set_property.assert_any_call("device", 42)

    def test_app_preserves_coreaudio_selection_without_pipewire_resolution(self):
        app = SimpleNamespace(config=SimpleNamespace(audio=SimpleNamespace(mic_device="coreaudio:uid:USB")))
        with mock.patch("nvbroadcast.app.IS_MACOS", True), \
             mock.patch.object(devices, "resolve_pipewire_target") as resolve:
            self.assertEqual(NVBroadcastApp._resolved_audio_capture_device(app), "coreaudio:uid:USB")
        resolve.assert_not_called()

    def test_app_rejects_saved_speaker_denoise_on_mac_without_building(self):
        app = SimpleNamespace(
            config=SimpleNamespace(audio=SimpleNamespace(speaker_denoise=True)),
            _speaker_monitor=mock.Mock(), _window=mock.Mock(),
        )
        with mock.patch("nvbroadcast.app.IS_MACOS", True), \
             mock.patch("nvbroadcast.app.save_config"):
            NVBroadcastApp.set_speaker_denoise(app, True)
        self.assertFalse(app.config.audio.speaker_denoise)
        app._speaker_monitor.stop.assert_called_once_with()
        app._speaker_monitor.build.assert_not_called()
        app._window.set_status.assert_called_once_with("Speaker noise removal is unavailable on macOS.")

    def test_processed_mic_test_is_rejected_before_recording_on_mac(self):
        window = SimpleNamespace(
            _mic_test=mock.Mock(is_recording=False), _mic_selector=mock.Mock(),
            _test_duration_selector=mock.Mock(), _test_source_selector=mock.Mock(),
            _test_status=mock.Mock(),
        )
        window._test_duration_selector.get_selected_device.return_value = "15"
        window._test_source_selector.get_selected_device.return_value = "processed"
        with mock.patch("nvbroadcast.ui.window.IS_MACOS", True):
            NVBroadcastWindow._on_test_record(window, mock.Mock())
        window._mic_test.start_recording.assert_not_called()
        self.assertIn("unavailable on macOS", window._test_status.set_text.call_args.args[0])

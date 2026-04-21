import unittest
from unittest import mock

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import numpy as np

from nvbroadcast.audio.pipeline import AudioPipeline


class AudioPipelineLifecycleTests(unittest.TestCase):
    def test_start_uses_loopback_virtual_mic_before_playing(self):
        pipeline = AudioPipeline(use_helper_process=False)
        pipeline._pipeline = mock.Mock()
        pipeline._uses_loopback_virtual_mic = True
        pipeline._effects = mock.Mock()

        with mock.patch("nvbroadcast.audio.pipeline.create_virtual_mic", return_value=True) as create_virtual_mic:
            pipeline.start()

        create_virtual_mic.assert_called_once_with()
        pipeline._effects.initialize.assert_called_once_with()
        pipeline._pipeline.set_state.assert_called_once_with(Gst.State.PLAYING)
        self.assertTrue(pipeline._running)

    def test_start_aborts_when_loopback_virtual_mic_creation_fails(self):
        pipeline = AudioPipeline(use_helper_process=False)
        pipeline._pipeline = mock.Mock()
        pipeline._uses_loopback_virtual_mic = True
        pipeline._effects = mock.Mock()

        with mock.patch("nvbroadcast.audio.pipeline.create_virtual_mic", return_value=False):
            pipeline.start()

        pipeline._effects.initialize.assert_not_called()
        pipeline._pipeline.set_state.assert_not_called()
        self.assertFalse(pipeline._running)

    def test_stop_destroys_loopback_virtual_mic(self):
        pipeline = AudioPipeline(use_helper_process=False)
        legacy_pipeline = mock.Mock()
        pipeline._pipeline = legacy_pipeline
        pipeline._uses_loopback_virtual_mic = True
        pipeline._running = True

        with mock.patch("nvbroadcast.audio.pipeline.destroy_virtual_mic") as destroy_virtual_mic:
            pipeline.stop()

        legacy_pipeline.set_state.assert_called_once_with(Gst.State.NULL)
        destroy_virtual_mic.assert_called_once_with()
        self.assertFalse(pipeline._running)

    def test_processed_output_uses_monotonic_output_timestamps(self):
        pipeline = AudioPipeline(use_helper_process=False)
        pipeline._effects = mock.Mock()
        pipeline._voice_fx = mock.Mock(enabled=True)
        pipeline._appsrc = mock.Mock()

        audio = np.linspace(-0.25, 0.25, 1024, dtype=np.float32)
        input_buf = Gst.Buffer.new_wrapped(audio.tobytes())
        input_buf.pts = 123456789
        input_buf.dts = 123456789
        input_buf.duration = 21 * Gst.MSECOND

        sample = mock.Mock()
        sample.get_buffer.return_value = input_buf
        appsink = mock.Mock()
        appsink.emit.return_value = sample

        pipeline._effects.process_chunk.return_value = audio
        pipeline._voice_fx.process_chunk.return_value = audio

        result = pipeline._on_new_sample(appsink)

        self.assertEqual(result, Gst.FlowReturn.OK)
        pipeline._voice_fx.process_chunk.assert_called_once()
        gate_reference = pipeline._voice_fx.process_chunk.call_args.kwargs["gate_reference"]
        np.testing.assert_allclose(gate_reference, audio)

        pushed = pipeline._appsrc.emit.call_args.args[1]
        expected_duration = Gst.util_uint64_scale(len(audio), Gst.SECOND, pipeline._sample_rate)
        self.assertEqual(pushed.duration, expected_duration)
        self.assertEqual(pushed.pts, 0)
        self.assertEqual(pushed.dts, 0)

    def test_processed_output_advances_timestamps_across_buffers(self):
        pipeline = AudioPipeline(use_helper_process=False)
        pipeline._effects = mock.Mock()
        pipeline._voice_fx = mock.Mock(enabled=False)
        pipeline._appsrc = mock.Mock()

        audio = np.linspace(-0.1, 0.1, 1024, dtype=np.float32)
        sample = mock.Mock()
        sample.get_buffer.return_value = Gst.Buffer.new_wrapped(audio.tobytes())
        appsink = mock.Mock()
        appsink.emit.return_value = sample
        pipeline._effects.process_chunk.return_value = audio

        pipeline._on_new_sample(appsink)
        first = pipeline._appsrc.emit.call_args.args[1]
        pipeline._on_new_sample(appsink)
        second = pipeline._appsrc.emit.call_args.args[1]

        self.assertEqual(first.pts, 0)
        self.assertEqual(
            second.pts,
            Gst.util_uint64_scale(len(audio), Gst.SECOND, pipeline._sample_rate),
        )

    def test_start_uses_helper_process_when_enabled(self):
        pipeline = AudioPipeline()
        pipeline._uses_loopback_virtual_mic = True
        pipeline._effects = mock.Mock()

        with mock.patch("nvbroadcast.audio.pipeline.create_virtual_mic", return_value=True) as create_virtual_mic, \
             mock.patch.object(pipeline, "_start_helper_process", return_value=True) as start_helper:
            pipeline.start()

        create_virtual_mic.assert_called_once_with()
        start_helper.assert_called_once_with()
        pipeline._effects.initialize.assert_not_called()
        self.assertTrue(pipeline._running)

    def test_helper_state_captures_live_audio_settings(self):
        pipeline = AudioPipeline(use_helper_process=False)
        pipeline.configure(mic_device="blue-mic", sample_rate=44100)
        pipeline.effects.enabled = True
        pipeline.effects.intensity = 0.65
        pipeline.voice_fx.enabled = True
        pipeline.voice_fx.use_gpu = False
        pipeline.voice_fx.settings.bass_boost = 0.2
        pipeline.voice_fx.settings.treble = 0.1
        pipeline.voice_fx.settings.warmth = 0.3
        pipeline.voice_fx.settings.compression = 0.5
        pipeline.voice_fx.settings.gate_threshold = 0.0
        pipeline.voice_fx.settings.gain = 0.15

        state = pipeline._helper_state()

        self.assertEqual(state["mic_device"], "blue-mic")
        self.assertEqual(state["sample_rate"], 44100)
        self.assertTrue(state["noise_removal"])
        self.assertAlmostEqual(state["noise_intensity"], 0.65)
        self.assertTrue(state["voice_fx_enabled"])
        self.assertFalse(state["voice_fx_use_gpu"])
        self.assertAlmostEqual(state["voice_fx_settings"]["compression"], 0.5)
        self.assertAlmostEqual(state["voice_fx_settings"]["gain"], 0.15)


if __name__ == "__main__":
    unittest.main()

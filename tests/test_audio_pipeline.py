import unittest
import os
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

    def test_start_helper_process_passes_parent_pid(self):
        pipeline = AudioPipeline(use_helper_process=False)
        fake_proc = mock.Mock()
        fake_proc.poll.return_value = None

        with mock.patch.object(pipeline, "_stop_helper_process"), \
             mock.patch.object(pipeline, "_stop_stale_helper_processes"), \
             mock.patch("nvbroadcast.audio.pipeline.subprocess.Popen", return_value=fake_proc) as popen, \
             mock.patch("nvbroadcast.audio.pipeline.time.sleep"):
            started = pipeline._start_helper_process()

        self.assertTrue(started)
        cmd = popen.call_args.args[0]
        self.assertIn("--parent-pid", cmd)
        self.assertIn(str(os.getpid()), cmd)

    def test_stop_stale_helper_processes_terminates_orphaned_helpers(self):
        pipeline = AudioPipeline(use_helper_process=False)
        current_pid = os.getpid()
        helper_pid = 50001
        stale_pid = 50002
        healthy_other_pid = 50003

        with mock.patch.object(
            pipeline,
            "_iter_helper_pids",
            return_value=[helper_pid, stale_pid, healthy_other_pid],
        ), mock.patch.object(
            pipeline,
            "_read_process_ppid",
            side_effect=lambda pid: {
                helper_pid: current_pid,
                stale_pid: 90001,
                healthy_other_pid: 90002,
            }[pid],
        ), mock.patch.object(
            pipeline,
            "_read_process_cmdline",
            side_effect=lambda pid: {
                helper_pid: "python -m nvbroadcast.audio.service --parent-pid 123",
                stale_pid: "python -m nvbroadcast.audio.service --state-b64 abc",
                healthy_other_pid: "python -m nvbroadcast.audio.service --parent-pid 456",
                90001: "/usr/lib/systemd/systemd --user",
                90002: "/home/doczeus/Projects/Nvidia Wrappers/Broadcast/.venv/bin/python -m nvbroadcast",
            }.get(pid, ""),
        ), mock.patch.object(pipeline, "_terminate_process") as terminate:
            pipeline._stop_stale_helper_processes()

        terminate.assert_called_once_with(stale_pid)


class DeepFilterStreamingParityTests(unittest.TestCase):
    """The ring-buffer rewrite of process_chunk must be sample-exact with
    the original concatenate-based streaming logic."""

    @staticmethod
    def _reference(session, blocks, intensity, hop, state_size):
        """Original algorithm (np.concatenate streaming) as the oracle."""
        state = np.zeros(state_size, dtype=np.float32)
        in_buf = np.zeros(0, dtype=np.float32)
        out_buf = np.zeros(hop, dtype=np.float32)
        prev_dry = np.zeros(hop, dtype=np.float32)
        outputs = []
        for block in blocks:
            in_buf = np.concatenate((in_buf, block.astype(np.float32)))
            while len(in_buf) >= hop:
                frame = in_buf[:hop]
                in_buf = in_buf[hop:]
                enhanced, state, _ = session(frame, state)
                if intensity < 1.0:
                    enhanced = (intensity * enhanced
                                + (1.0 - intensity) * prev_dry)
                prev_dry = frame
                out_buf = np.concatenate((out_buf, enhanced))
            n = len(block)
            if len(out_buf) < n:
                pad = np.zeros(n - len(out_buf), dtype=np.float32)
                out_buf = np.concatenate((pad, out_buf))
            outputs.append(out_buf[:n])
            out_buf = out_buf[n:]
        return outputs

    def test_ring_buffer_matches_reference_across_block_sizes(self):
        from nvbroadcast.audio import deepfilter as df

        rng = np.random.default_rng(42)

        def fake_session_run(_names, feeds):
            frame = feeds["input_frame"]
            state = feeds["states"]
            # Deterministic fake "model": scaled input + state coupling.
            enhanced = (frame * 0.5 + state[:len(frame)] * 0.25).astype(
                np.float32)
            new_state = np.roll(state, 1).astype(np.float32)
            new_state[0] = float(frame.sum())
            return [enhanced, new_state, np.float32(0.0)]

        def oracle_session(frame, state):
            out = fake_session_run(None, {"input_frame": frame,
                                          "states": state,
                                          "atten_lim_db": None})
            return out[0], out[1], out[2]

        # Mixed block sizes incl. non-multiples of the hop and a huge block
        # that forces the ring to grow.
        sizes = [480, 128, 480, 960, 333, 480, 17, 4800, 480 * 33, 240]
        blocks = [rng.standard_normal(s).astype(np.float32) for s in sizes]

        denoiser = df.DeepFilterDenoiser.__new__(df.DeepFilterDenoiser)
        denoiser._initialized = True
        denoiser._atten = np.array([100.0], dtype=np.float32)
        denoiser.session = mock.Mock(run=mock.Mock(
            side_effect=fake_session_run))
        denoiser.reset()

        intensity = 0.76
        expected = self._reference(oracle_session, blocks, intensity,
                                   df.HOP_SIZE, df.STATE_SIZE)
        for block, want in zip(blocks, expected):
            got = denoiser.process_chunk(block, df.SAMPLE_RATE,
                                         intensity=intensity)
            self.assertEqual(len(got), len(block))
            self.assertTrue(got.flags.owndata)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

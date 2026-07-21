import unittest
import os
import threading
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

    def test_iter_helper_pids_ignores_commands_that_only_mention_module(self):
        pipeline = AudioPipeline(use_helper_process=False)
        result = mock.Mock(returncode=0, stdout="50001\n50002\n50003\n")
        argv = {
            50001: ["bash", "-lc", "pgrep -f nvbroadcast.audio.service"],
            50002: [
                "python",
                "-m",
                "nvbroadcast.audio.service",
                "--parent-pid",
                "123",
            ],
            50003: ["python", "-c", "print('nvbroadcast.audio.service')"],
        }

        with mock.patch(
            "nvbroadcast.audio.pipeline.subprocess.run", return_value=result
        ), mock.patch.object(
            pipeline, "_read_process_argv", side_effect=lambda pid: argv[pid]
        ):
            self.assertEqual(pipeline._iter_helper_pids(), [50002])

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


def _pactl_result(payload):
    result = mock.Mock()
    result.returncode = 0
    result.stdout = payload
    return result


class IdleMonitorRestartTests(unittest.TestCase):
    """Restarting the idle monitor while the previous run is blocked inside
    a consumer probe must retire the old thread: each run owns its stop
    event, so a stop can never be undone by the next start clearing it."""

    def test_restart_while_probe_blocked_retires_old_monitor(self):
        pipeline = AudioPipeline(use_helper_process=False)
        pipeline._uses_loopback_virtual_mic = True
        pipeline.auto_idle = True
        pipeline._running = True
        pipeline._idle_poll_interval_s = 0.01
        probe_entered = threading.Event()
        release_probe = threading.Event()

        def blocked_probe():
            probe_entered.set()
            release_probe.wait(5)
            return None  # unknown reads as in-use, never idles

        pipeline._count_virtual_mic_consumers = blocked_probe
        new_thread = None
        try:
            pipeline._start_idle_monitor()
            old_thread = pipeline._idle_monitor
            self.assertTrue(probe_entered.wait(2), "probe never entered")

            pipeline._stop_idle_monitor()
            pipeline._start_idle_monitor()
            new_thread = pipeline._idle_monitor
            self.assertIsNotNone(new_thread)
            self.assertIsNot(new_thread, old_thread)

            release_probe.set()
            old_thread.join(2)
            self.assertFalse(old_thread.is_alive(),
                             "superseded monitor survived the restart")
            self.assertTrue(new_thread.is_alive(),
                            "replacement monitor should be running")
        finally:
            release_probe.set()
            pipeline._stop_idle_monitor()
            if new_thread is not None:
                new_thread.join(2)


class VirtualMicConsumerCountTests(unittest.TestCase):
    """The counter must see only real recorders on nvbroadcast_mic — the
    remap module holds nvbroadcast_sink.monitor open forever, and counting
    that stream kept mic power save from ever engaging."""

    SOURCES = (
        '[{"index": 40, "name": "nvbroadcast_sink.monitor"},'
        ' {"index": 41, "name": "nvbroadcast_mic"},'
        ' {"index": 42, "name": "alsa_input.usb-mic"}]'
    )

    def _count(self, sources_json, outputs_json):
        pipeline = AudioPipeline(use_helper_process=False)

        def fake_run(cmd, **kwargs):
            return _pactl_result(
                sources_json if "sources" in cmd else outputs_json)

        with mock.patch("nvbroadcast.audio.pipeline.subprocess.run",
                        side_effect=fake_run):
            return pipeline._count_virtual_mic_consumers()

    def test_monitor_held_by_remap_module_counts_zero(self):
        outputs = '[{"index": 7, "source": 40}]'  # remap loop on the monitor
        self.assertEqual(self._count(self.SOURCES, outputs), 0)

    def test_real_recorder_on_virtual_mic_counts(self):
        outputs = '[{"index": 7, "source": 40}, {"index": 8, "source": 41}]'
        self.assertEqual(self._count(self.SOURCES, outputs), 1)

    def test_missing_virtual_mic_source_returns_none(self):
        sources = '[{"index": 40, "name": "nvbroadcast_sink.monitor"}]'
        self.assertIsNone(self._count(sources, "[]"))

    def test_pactl_failure_returns_none(self):
        pipeline = AudioPipeline(use_helper_process=False)
        failed = mock.Mock()
        failed.returncode = 1
        failed.stdout = ""
        with mock.patch("nvbroadcast.audio.pipeline.subprocess.run",
                        return_value=failed):
            self.assertIsNone(pipeline._count_virtual_mic_consumers())


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

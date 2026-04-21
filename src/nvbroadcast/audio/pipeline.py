# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Audio pipeline: mic capture -> denoise -> browser-safe virtual mic output."""

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np

from nvbroadcast.audio.effects import AudioEffects
from nvbroadcast.audio.virtual_mic import (
    create_virtual_mic,
    destroy_virtual_mic,
    has_virtual_mic_backend,
    virtual_mic_backend,
    virtual_mic_sink_name,
)
from nvbroadcast.core.platform import IS_LINUX


class AudioPipeline:
    """Real-time microphone denoise pipeline.

    The Linux processed-mic path runs inside a helper process for isolation.
    It still feeds the virtual sink through a dedicated playback client because
    browsers and meeting apps only need the exported source to be conventional
    and stable; the helper process keeps the main UI/video runtime from
    starving the audio transport.
    """

    def __init__(
        self,
        manage_virtual_mic: bool = True,
        use_helper_process: bool | None = None,
    ):
        Gst.init(None)
        self._pipeline: Gst.Pipeline | None = None
        self._capture_pipeline: Gst.Pipeline | None = None
        self._output_pipeline: Gst.Pipeline | None = None
        self._output_process: subprocess.Popen[bytes] | None = None
        self._helper_process: subprocess.Popen[bytes] | None = None
        self._appsrc = None
        self._effects = AudioEffects()
        self._sample_rate = 48000
        self._channels = 1
        self._running = False
        self._mic_device = ""
        self._level_monitor = None
        self._voice_fx = None
        self._transcriber_feed = None
        self._uses_loopback_virtual_mic = False
        self._virtual_mic_backend = ""
        self._manage_virtual_mic = manage_virtual_mic
        if use_helper_process is None:
            disable_helper = os.getenv("NVBROADCAST_AUDIO_DISABLE_HELPER", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            use_helper_process = IS_LINUX and has_virtual_mic_backend() and not disable_helper
        self._use_helper_process = bool(use_helper_process)
        self._output_frames_pushed = 0
        self._output_buffer_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
        self._output_worker: threading.Thread | None = None
        self._stop_output_worker = threading.Event()
        self._debug_audio = os.getenv("NVBROADCAST_AUDIO_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._debug_audio_buffers = 0
        self._debug_output_pushes = 0

    @property
    def effects(self) -> AudioEffects:
        return self._effects

    @property
    def level_monitor(self):
        if self._level_monitor is None:
            from nvbroadcast.audio.level_monitor import AudioLevelMonitor

            self._level_monitor = AudioLevelMonitor()
        return self._level_monitor

    @property
    def voice_fx(self):
        if self._voice_fx is None:
            from nvbroadcast.audio.voice_fx import VoiceFX

            self._voice_fx = VoiceFX()
        return self._voice_fx

    @property
    def uses_helper_process(self) -> bool:
        return self._use_helper_process and self._uses_loopback_virtual_mic

    def configure(self, mic_device: str = "", sample_rate: int = 48000):
        self._mic_device = mic_device
        self._sample_rate = sample_rate

    def build(self) -> None:
        """Build the audio capture pipeline and output transport."""
        self._uses_loopback_virtual_mic = IS_LINUX and has_virtual_mic_backend()
        self._virtual_mic_backend = virtual_mic_backend() if self._uses_loopback_virtual_mic else ""
        self.stop()

        if self.uses_helper_process:
            self._pipeline = None
            self._capture_pipeline = None
            self._output_pipeline = None
            return

        self._capture_pipeline = self._build_capture_pipeline()
        if not self._uses_loopback_virtual_mic:
            self._output_pipeline = self._build_output_pipeline()

        # Preserve the legacy attribute for callers/tests that still expect it.
        self._pipeline = self._capture_pipeline

    def _build_capture_pipeline(self) -> Gst.Pipeline:
        pipeline = Gst.Pipeline.new("nvbroadcast-audio-input")

        if IS_LINUX and self._virtual_mic_backend == "pulse":
            source = Gst.ElementFactory.make("pulsesrc", "mic-source")
            if self._mic_device:
                source.set_property("device", self._mic_device)
        else:
            source = Gst.ElementFactory.make("pipewiresrc", "mic-source")
            if self._mic_device:
                source.set_property("target-object", self._mic_device)
            source.set_property("do-timestamp", True)
            source.set_property("min-buffers", 2)
            source.set_property("max-buffers", 4)
            source.set_property(
                "stream-properties",
                Gst.Structure.new_from_string("properties,node.latency=1024/48000"),
            )

        convert_in = Gst.ElementFactory.make("audioconvert", "convert-in")
        resample_in = Gst.ElementFactory.make("audioresample", "resample-in")

        in_caps = Gst.ElementFactory.make("capsfilter", "in-caps")
        in_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )

        appsink = Gst.ElementFactory.make("appsink", "audio-sink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 8)
        appsink.set_property("drop", False)
        appsink.connect("new-sample", self._on_new_sample)

        for el in [source, convert_in, resample_in, in_caps, appsink]:
            pipeline.add(el)

        source.link(convert_in)
        convert_in.link(resample_in)
        resample_in.link(in_caps)
        in_caps.link(appsink)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)
        return pipeline

    def _build_output_pipeline(self) -> Gst.Pipeline:
        """Fallback output path used only when no managed virtual mic exists."""
        pipeline = Gst.Pipeline.new("nvbroadcast-audio-output")

        self._appsrc = Gst.ElementFactory.make("appsrc", "audio-src")
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("do-timestamp", False)
        self._appsrc.set_property("max-buffers", 8)
        self._appsrc.set_property("max-time", 200 * Gst.MSECOND)
        self._appsrc.set_property("block", True)
        self._appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )

        convert_out = Gst.ElementFactory.make("audioconvert", "convert-out")
        out_caps = Gst.ElementFactory.make("capsfilter", "out-caps")
        out_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,rate={self._sample_rate},"
                f"channels={self._channels},layout=interleaved"
            ),
        )

        sink = Gst.ElementFactory.make("pipewiresink", "mic-output")
        sink.set_property("mode", 2)  # provide
        sink.set_property("sync", False)
        sink.set_property("async", False)
        sink.set_property(
            "stream-properties",
            Gst.Structure.new_from_string(
                "properties,media.class=Audio/Source/Virtual,"
                "node.name=nvbroadcast_mic,"
                'node.description="nvbroadcast"'
            ),
        )

        for el in [self._appsrc, convert_out, out_caps, sink]:
            pipeline.add(el)

        self._appsrc.link(convert_out)
        convert_out.link(out_caps)
        out_caps.link(sink)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)
        return pipeline

    def _on_new_sample(self, appsink):
        """Process one captured chunk through denoise and output transport."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        audio = np.frombuffer(map_info.data, dtype=np.float32).copy()
        buf.unmap(map_info)

        if self._level_monitor:
            self._level_monitor.update(audio)

        if self._transcriber_feed:
            try:
                self._transcriber_feed.feed_audio(audio, self._sample_rate)
            except Exception:
                pass

        processed = self._effects.process_chunk(audio, self._sample_rate)
        if self._voice_fx and self._voice_fx.enabled:
            processed = self._voice_fx.process_chunk(
                processed,
                self._sample_rate,
                gate_reference=audio,
            )

        if self._uses_loopback_virtual_mic:
            output_state = self._enqueue_output_audio(processed)
        else:
            output_state = self._push_buffer_direct(processed)

        if self._debug_audio and self._debug_audio_buffers < 40:
            self._debug_audio_buffers += 1
            in_rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            out_rms = float(np.sqrt(np.mean(np.square(processed)))) if len(processed) else 0.0
            in_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            out_peak = float(np.max(np.abs(processed))) if len(processed) else 0.0
            print(
                "[NVIDIA Broadcast Audio Debug] "
                f"in_frames={len(audio)} out_frames={len(processed)} "
                f"in_rms={in_rms:.6f} out_rms={out_rms:.6f} "
                f"in_peak={in_peak:.6f} out_peak={out_peak:.6f} "
                f"buf_duration_ms="
                f"{(buf.duration / Gst.MSECOND) if buf.duration not in (None, Gst.CLOCK_TIME_NONE) else None} "
                f"output_state={output_state}",
                flush=True,
            )
        return Gst.FlowReturn.OK

    def _push_buffer_direct(self, processed: np.ndarray):
        """Push one processed chunk to the active output backend."""
        if self._output_process is not None:
            if self._output_process.poll() is not None:
                return f"process-exited={self._output_process.returncode}"
            if self._output_process.stdin is None:
                return "no-process-stdin"
            stereo = np.repeat(processed.astype(np.float32, copy=False), 2)
            payload = stereo.tobytes()
            try:
                self._output_process.stdin.write(payload)
                return f"wrote={len(payload)}"
            except (BrokenPipeError, OSError) as exc:
                return f"write-failed={exc.__class__.__name__}"

        if self._appsrc is None:
            return "no-output"

        payload = processed.astype(np.float32, copy=False).tobytes()
        new_buf = Gst.Buffer.new_allocate(None, len(payload), None)
        new_buf.fill(0, payload)
        frame_count = len(processed)
        duration = Gst.util_uint64_scale(frame_count, Gst.SECOND, self._sample_rate)
        new_buf.pts = Gst.util_uint64_scale(self._output_frames_pushed, Gst.SECOND, self._sample_rate)
        new_buf.dts = new_buf.pts
        new_buf.duration = duration
        self._output_frames_pushed += frame_count
        return self._appsrc.emit("push-buffer", new_buf)

    def _enqueue_output_audio(self, processed: np.ndarray) -> str:
        """Queue processed audio for the dedicated output worker."""
        chunk = np.array(processed, dtype=np.float32, copy=True)
        try:
            self._output_buffer_queue.put(chunk, timeout=0.25)
        except queue.Full:
            return "queue-full"
        return f"queued={self._output_buffer_queue.qsize()}"

    def _start_output_worker(self):
        if self._output_worker is not None and self._output_worker.is_alive():
            return
        self._stop_output_worker.clear()
        self._output_worker = threading.Thread(
            target=self._output_worker_main,
            name="nvbroadcast-audio-output",
            daemon=True,
        )
        self._output_worker.start()

    def _stop_output_worker_sync(self):
        self._stop_output_worker.set()
        if self._output_worker is not None:
            while True:
                try:
                    self._output_buffer_queue.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self._output_buffer_queue.get_nowait()
                    except queue.Empty:
                        break
            self._output_worker.join(timeout=2.0)
            self._output_worker = None
        self._clear_output_queue()

    def _clear_output_queue(self):
        while True:
            try:
                self._output_buffer_queue.get_nowait()
            except queue.Empty:
                break

    def _output_worker_main(self):
        while True:
            try:
                chunk = self._output_buffer_queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_output_worker.is_set():
                    break
                continue

            if chunk is None:
                break

            flow = self._push_buffer_direct(chunk)
            if self._debug_audio and self._debug_output_pushes < 40:
                self._debug_output_pushes += 1
                chunk_rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
                chunk_peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
                print(
                    "[NVIDIA Broadcast Audio Debug] "
                    f"worker_push_flow={flow} chunk_rms={chunk_rms:.6f} chunk_peak={chunk_peak:.6f}",
                    flush=True,
                )

    def _start_output_process(self) -> bool:
        if self._output_process is not None and self._output_process.poll() is None:
            return True

        self._stop_output_process()
        sink_name = virtual_mic_sink_name()
        if self._virtual_mic_backend == "pulse":
            cmd = [
                "pacat",
                "--playback",
                "--raw",
                f"--device={sink_name}",
                "--rate",
                str(self._sample_rate),
                "--format",
                "float32le",
                "--channels",
                "2",
                "--channel-map",
                "front-left,front-right",
                "--process-time-msec",
                "10",
                "--latency-msec",
                "40",
                "--client-name",
                "nvbroadcast",
                "--stream-name",
                "nvbroadcast mic output",
            ]
        elif self._virtual_mic_backend == "pw-loopback":
            cmd = [
                "pw-cat",
                "--playback",
                "--target",
                sink_name,
                "--rate",
                str(self._sample_rate),
                "--channels",
                "2",
                "--channel-map",
                "FL,FR",
                "--format",
                "f32",
                "--latency",
                "40ms",
                "-",
            ]
        else:
            return False

        try:
            self._output_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception:
            self._output_process = None
            return False

        time.sleep(0.1)
        if self._output_process.poll() is not None:
            self._output_process = None
            return False
        return True

    def _helper_state(self) -> dict:
        settings = self.voice_fx.settings
        return {
            "mic_device": self._mic_device,
            "sample_rate": self._sample_rate,
            "noise_removal": self._effects.enabled,
            "noise_intensity": self._effects.intensity,
            "voice_fx_enabled": self.voice_fx.enabled,
            "voice_fx_use_gpu": self.voice_fx.use_gpu,
            "voice_fx_settings": {
                "bass_boost": settings.bass_boost,
                "treble": settings.treble,
                "warmth": settings.warmth,
                "compression": settings.compression,
                "gate_threshold": settings.gate_threshold,
                "gain": settings.gain,
            },
        }

    def _start_helper_process(self) -> bool:
        if self._helper_process is not None and self._helper_process.poll() is None:
            return True

        self._stop_helper_process()
        state_json = json.dumps(self._helper_state(), separators=(",", ":")).encode("utf-8")
        state_b64 = base64.urlsafe_b64encode(state_json).decode("ascii")
        stdio = None if self._debug_audio else subprocess.DEVNULL
        cmd = [
            sys.executable,
            "-m",
            "nvbroadcast.audio.service",
            "--state-b64",
            state_b64,
        ]

        try:
            self._helper_process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=stdio,
                stderr=stdio,
                bufsize=0,
            )
        except Exception:
            self._helper_process = None
            return False

        time.sleep(0.2)
        if self._helper_process.poll() is not None:
            self._helper_process = None
            return False
        return True

    def _stop_helper_process(self):
        if self._helper_process is None:
            return
        try:
            self._helper_process.terminate()
            self._helper_process.wait(timeout=3)
        except Exception:
            try:
                self._helper_process.kill()
                self._helper_process.wait(timeout=1)
            except Exception:
                pass
        self._helper_process = None

    def _stop_output_process(self):
        if self._output_process is None:
            return
        try:
            if self._output_process.stdin is not None:
                self._output_process.stdin.close()
        except Exception:
            pass
        try:
            self._output_process.terminate()
            self._output_process.wait(timeout=2)
        except Exception:
            try:
                self._output_process.kill()
                self._output_process.wait(timeout=1)
            except Exception:
                pass
        self._output_process = None

    def start(self):
        if self.uses_helper_process:
            if self._manage_virtual_mic and not create_virtual_mic():
                return
            if not self._start_helper_process():
                if self._manage_virtual_mic:
                    destroy_virtual_mic()
                return
            self._running = True
            return

        if self._capture_pipeline:
            if self._uses_loopback_virtual_mic:
                if self._manage_virtual_mic and not create_virtual_mic():
                    return
                if not self._start_output_process():
                    if self._manage_virtual_mic:
                        destroy_virtual_mic()
                    return
                self._start_output_worker()
            elif self._output_pipeline:
                self._output_pipeline.set_state(Gst.State.PLAYING)

            self._output_frames_pushed = 0
            self._debug_audio_buffers = 0
            self._debug_output_pushes = 0
            self._clear_output_queue()
            self._effects.initialize()
            self._capture_pipeline.set_state(Gst.State.PLAYING)
            self._running = True
            return

        if self._pipeline:
            if self._uses_loopback_virtual_mic and not create_virtual_mic():
                return
            self._output_frames_pushed = 0
            self._debug_audio_buffers = 0
            self._debug_output_pushes = 0
            self._effects.initialize()
            self._pipeline.set_state(Gst.State.PLAYING)
            self._running = True

    def stop(self):
        capture_pipeline = self._capture_pipeline
        output_pipeline = self._output_pipeline
        legacy_pipeline = self._pipeline

        self._capture_pipeline = None
        self._output_pipeline = None
        self._pipeline = None

        self._stop_helper_process()

        if capture_pipeline is not None:
            capture_pipeline.set_state(Gst.State.NULL)
        elif legacy_pipeline is not None:
            legacy_pipeline.set_state(Gst.State.NULL)

        self._stop_output_worker_sync()
        self._stop_output_process()

        if output_pipeline is not None:
            if self._appsrc is not None:
                try:
                    self._appsrc.emit("end-of-stream")
                except Exception:
                    pass
            output_pipeline.set_state(Gst.State.NULL)

        self._appsrc = None
        self._running = False
        self._output_frames_pushed = 0
        self._debug_audio_buffers = 0
        self._debug_output_pushes = 0

        if self._manage_virtual_mic and self._uses_loopback_virtual_mic:
            destroy_virtual_mic()

    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"[NVIDIA Broadcast Audio] Error: {err.message}")
        if debug:
            print(f"[NVIDIA Broadcast Audio] Debug: {debug}")

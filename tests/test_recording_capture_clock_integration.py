"""Generated-media regression for processing latency in saved H.264/AAC.

The bounded child uses the production recorder, with no camera, microphone,
display or sound device. A legacy arrival-timestamp control establishes that
the decoded event comparison actually detects the injected processing delay.
"""

import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def _synthetic_capture_clock():
    # A hosted runner may be younger than the ten-minute camera origin. Give
    # this private clock a positive epoch without calibrating the global clock.
    clock = Gst.SystemClock(name="synthetic-recording-clock")
    internal = clock.get_internal_time()
    clock.set_calibration(internal, internal + 3600 * Gst.SECOND, 1, 1)
    return clock


def _encoded_case(path, video_encoder, aac_encoder, mapped):
    from nvbroadcast.video.pipeline import VideoPipeline

    Gst.init([])
    p = VideoPipeline()
    p._width, p._height, p._fps = 64, 48, 15
    capture = Gst.Pipeline.new("synthetic-old-camera")
    clock = _synthetic_capture_clock()
    capture.use_clock(clock)
    capture.set_start_time(Gst.CLOCK_TIME_NONE)
    capture.set_base_time(clock.get_time() - 600 * Gst.SECOND)
    assert capture.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
    assert capture.get_clock() == clock
    p._pipeline = capture
    p._capture_generation = 17
    caps = Gst.Caps.from_string(
        "video/x-raw,format=BGRA,width=64,height=48,framerate=15/1"
    )
    markers = (48, 128, 224)
    delays = (0, 100, 300)
    events = []

    def snapshot():
        buffer = Gst.Buffer.new()
        buffer.pts = clock.get_time() - capture.get_base_time()
        buffer.duration = Gst.SECOND // p._fps
        segment = Gst.Segment.new()
        segment.init(Gst.Format.TIME)
        sample = Gst.Sample.new(buffer, caps, segment, None)
        token = p._capture_recording_time(sample, capture, 17)
        assert token.absolute_time is not None
        assert buffer.pts >= 600 * Gst.SECOND
        return token

    def push(gray, token=None):
        if token is None:
            token = snapshot()
        result = p._push_recording_frame(
            bytes((gray, gray, gray, 255)) * (64 * 48),
            Gst.SECOND // p._fps,
            capture_time=token if mapped else None,
        )
        assert result == Gst.FlowReturn.OK, result

    try:
        with mock.patch.object(
            p, "_has_gst_element",
            side_effect=lambda name: name in {"h264parse", video_encoder, aac_encoder},
        ), mock.patch.object(
            p, "_recording_audio_source",
            return_value=(
                "audiotestsrc is-live=true wave=sine freq=1000 "
                "volume=0 samplesperbuffer=480", "",
            ),
        ):
            p.start_recording(path)
            assert p.recording_has_audio
            assert p._recording_pipeline.get_clock() == clock
            audio = p._recording_pipeline.get_by_name("recording_audio_source")
            # Supplying initial frames also exercises live appsrc preroll: Rec
            # has just started, while the camera's TIME origin is ten minutes old.
            push(0)
            time.sleep(0.07)
            push(0)
            time.sleep(0.07)
            for gray, delay in zip(markers, delays):
                token = snapshot()  # Capture event precedes simulated effects.
                events.append(token.absolute_time - p._recording_pipeline.get_base_time())
                audio.set_property("volume", 0.8)
                # Only the first 40 ms is audible, regardless of effect cost.
                # The remaining wait models delivery delayed by CPU work.
                time.sleep(min(delay / 1000, 0.04))
                if delay < 40:
                    push(gray, token)
                    time.sleep((40 - delay) / 1000)
                audio.set_property("volume", 0.0)
                if delay >= 40:
                    time.sleep((delay - 40) / 1000)
                    push(gray, token)
                time.sleep(0.07)
                push(0)
                time.sleep(0.07)
                push(0)
                time.sleep(0.07)
            assert p.stop_recording(), p.recording_audio_error
    finally:
        if p.is_recording:
            p.stop_recording()
        capture.set_state(Gst.State.NULL)

    video_pts = {}
    loud_times = []
    failures = []

    def video_handoff(_sink, buffer, _pad):
        ok, mapping = buffer.map(Gst.MapFlags.READ)
        if not ok:
            failures.append("Cannot map decoded generated video")
            return
        try:
            # Convert back to BGRA so limited-range H.264 luma is expanded
            # before classifying the generated grayscale code.
            pixels = mapping.data[0::4]
            mean = sum(pixels) / len(pixels)
            marker = min(markers, key=lambda gray: abs(mean - gray))
            if abs(mean - marker) < 12:
                video_pts.setdefault(marker, int(buffer.pts))
        finally:
            buffer.unmap(mapping)

    def audio_handoff(_sink, buffer, _pad):
        ok, mapping = buffer.map(Gst.MapFlags.READ)
        if not ok:
            failures.append("Cannot map decoded generated audio")
            return
        try:
            samples = array.array("f")
            samples.frombytes(mapping.data)
            for index, value in enumerate(samples):
                if abs(value) > 0.15:
                    loud_times.append(int(buffer.pts) + index * Gst.SECOND // 48000)
        finally:
            buffer.unmap(mapping)

    def decoded_sink(description, callback):
        sink = Gst.parse_bin_from_description(description, True)
        sink.get_by_name("decoded").connect("handoff", callback)
        return sink

    player = Gst.ElementFactory.make("playbin")
    player.set_property("video-sink", decoded_sink(
        "videoconvert ! video/x-raw,format=BGRA ! "
        "fakesink name=decoded sync=false signal-handoffs=true", video_handoff,
    ))
    player.set_property("audio-sink", decoded_sink(
        "audioconvert ! audio/x-raw,format=F32LE,layout=interleaved,"
        "rate=48000,channels=1 ! "
        "fakesink name=decoded sync=false signal-handoffs=true", audio_handoff,
    ))
    player.set_property("uri", Gst.filename_to_uri(path))
    try:
        assert player.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
        message = player.get_bus().timed_pop_filtered(
            5 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        assert message is not None, "Timed out decoding generated MP4"
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            raise AssertionError(f"Decode failed: {error.message}; {debug}")
    finally:
        player.set_state(Gst.State.NULL)
    assert not failures, failures
    assert set(video_pts) == set(markers), video_pts
    pulses = []
    previous = None
    for pts in loud_times:
        if previous is None or pts - previous > 80_000_000:
            pulses.append(pts)
        previous = pts
    assert len(pulses) == 3, pulses
    offsets = [video_pts[gray] - pulse for gray, pulse in zip(markers, pulses)]
    relative_offsets = [offset - offsets[0] for offset in offsets]
    result = {
        "mapped": bool(mapped), "capture_origin_seconds": 600,
        "processing_delay_ms": list(delays), "capture_event_ns": events,
        "video_event_ns": [video_pts[gray] for gray in markers],
        "audio_event_ns": pulses, "relative_av_offset_ns": relative_offsets,
    }
    print(json.dumps(result, sort_keys=True))


class RecordingCaptureClockIntegrationTests(unittest.TestCase):
    def test_private_clock_supports_old_camera_origin_on_a_young_runner(self):
        Gst.init([])
        global_clock = Gst.SystemClock.obtain()
        global_calibration = global_clock.get_calibration()
        young = Gst.SystemClock(name="young-synthetic-clock")
        young.set_calibration(young.get_internal_time(), Gst.SECOND, 1, 1)
        capture = Gst.Pipeline.new("young-camera-fixture")
        self.assertLess(young.get_time(), 600 * Gst.SECOND)
        with self.assertRaises(OverflowError):
            capture.set_base_time(young.get_time() - 600 * Gst.SECOND)
        # Inject a real clock with a young epoch, without mocking its time math.
        with mock.patch.object(Gst, "SystemClock", return_value=young):
            clock = _synthetic_capture_clock()
        self.assertIs(clock, young)
        self.assertNotEqual(clock, global_clock)
        capture.use_clock(clock)
        capture.set_base_time(clock.get_time() - 600 * Gst.SECOND)
        self.assertGreater(capture.get_base_time(), 0)
        self.assertGreaterEqual(clock.get_time() - capture.get_base_time(), 600 * Gst.SECOND)
        self.assertEqual(global_clock.get_calibration(), global_calibration)

    def test_encoded_av_preserves_capture_time_after_processing(self):
        Gst.init([])
        required = (
            "audiotestsrc", "aacparse", "h264parse", "mp4mux", "playbin",
            "videoconvert", "audioconvert",
        )
        missing = [name for name in required if Gst.ElementFactory.find(name) is None]
        if missing:
            self.skipTest(f"GStreamer recording plugins unavailable: {missing}")
        video = next((name for name in ("x264enc", "openh264enc")
                      if Gst.ElementFactory.find(name) is not None), None)
        audio = next((name for name in ("avenc_aac", "voaacenc")
                      if Gst.ElementFactory.find(name) is not None), None)
        if not video or not audio:
            self.skipTest("CPU H.264 and AAC encoders are required")
        invocation = (
            "import runpy,sys; "
            "m=runpy.run_path(sys.argv[1]); "
            "m['_encoded_case'](sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]=='1')"
        )
        results = {}
        with tempfile.TemporaryDirectory(prefix="NVB capture timing ") as directory:
            for mapped in (False, True):
                path = str(Path(directory) / f"generated-{int(mapped)}.mp4")
                child = subprocess.run(
                    [sys.executable, "-c", invocation, str(Path(__file__).resolve()),
                     path, video, audio, str(int(mapped))],
                    capture_output=True, text=True, timeout=20, check=False,
                )
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
                results[mapped] = json.loads(child.stdout.splitlines()[-1])
        # Codec priming and container-origin offsets cancel against event zero.
        # The legacy control must expose both injected delays, so a decoder that
        # erases or ignores PTS cannot accidentally make the fixed case pass.
        legacy = results[False]["relative_av_offset_ns"]
        self.assertGreater(legacy[1], 60_000_000, results)
        self.assertGreater(legacy[2], 240_000_000, results)
        fixed = results[True]["relative_av_offset_ns"]
        self.assertLess(abs(fixed[1]), 50_000_000, results)
        self.assertLess(abs(fixed[2]), 50_000_000, results)


if __name__ == "__main__":
    unittest.main()

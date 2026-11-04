"""Capture-event clock rebasing, lifecycle races and invalid-timing fallback."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from nvbroadcast.video.pipeline import Gst, VideoPipeline


class RecordingCaptureClockTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = VideoPipeline()
        clock = object()
        capture = SimpleNamespace(base=1000 * Gst.SECOND, get_clock=lambda: clock)
        capture.get_base_time = lambda: capture.base
        recorder = SimpleNamespace(base=1600 * Gst.SECOND, get_clock=lambda: clock)
        recorder.get_base_time = lambda: recorder.base
        pipeline._pipeline = capture
        pipeline._capture_generation = 7
        pipeline._recording_session = 3
        pipeline._recording = True
        pipeline._recording_pipeline = recorder
        pipeline._recording_capture_clock = clock
        pipeline._rec_appsrc = mock.Mock()
        pipeline._rec_appsrc.emit.return_value = Gst.FlowReturn.OK
        return pipeline, capture, recorder

    def sample(self, pts=600 * Gst.SECOND + 50_000_000, segment=None):
        buffer = Gst.Buffer.new_wrapped(bytes((10, 20, 30, 255)) * 4)
        buffer.pts, buffer.duration = pts, Gst.SECOND // 30
        if segment is None:
            segment = Gst.Segment.new()
            segment.init(Gst.Format.TIME)
        caps = Gst.Caps.from_string("video/x-raw,format=BGRA,width=2,height=2")
        return Gst.Sample.new(buffer, caps, segment, None)

    def test_old_camera_origin_and_processing_time_do_not_shift_event_pts(self):
        for processing_ns in (0, 100_000_000, 300_000_000):
            with self.subTest(processing_ns=processing_ns):
                pipeline, capture, recorder = self.make_pipeline()
                stamp = pipeline._capture_recording_time(self.sample())
                # Even a later capture-base change cannot alter this snapshot.
                capture.base += processing_ns
                pipeline._push_recording_frame(b"pixels", 1234, capture_time=stamp)
                buffer = pipeline._rec_appsrc.emit.call_args.args[1]
                self.assertEqual((buffer.pts, buffer.dts), (50_000_000, 50_000_000))
                self.assertEqual(buffer.duration, 1234)
                self.assertEqual(buffer.extract_dup(0, buffer.get_size()), b"pixels")
                with self.assertRaises(FrozenInstanceError):
                    stamp.absolute_time = 0

    def test_segment_running_time_is_converted_instead_of_copying_camera_pts(self):
        pipeline, _capture, _recorder = self.make_pipeline()
        segment = Gst.Segment.new()
        segment.init(Gst.Format.TIME)
        segment.start, segment.base, segment.time = 100 * Gst.SECOND, 20 * Gst.SECOND, 777 * Gst.SECOND
        sample = self.sample(680 * Gst.SECOND + 50_000_000, segment)
        stamp = pipeline._capture_recording_time(sample)
        pipeline._push_recording_frame(b"pixels", capture_time=stamp)
        buffer = pipeline._rec_appsrc.emit.call_args.args[1]
        self.assertEqual(buffer.pts, 50_000_000)

    def test_valid_clock_and_base_work_during_pending_playing_without_preroll_deadlock(self):
        pipeline, _capture, recorder = self.make_pipeline()
        recorder.get_state = mock.Mock(return_value=(
            Gst.StateChangeReturn.ASYNC, Gst.State.PAUSED, Gst.State.PLAYING))
        stamp = pipeline._capture_recording_time(self.sample())
        self.assertEqual(pipeline._push_recording_frame(b"pixels", capture_time=stamp), Gst.FlowReturn.OK)
        self.assertEqual(pipeline._rec_appsrc.emit.call_args.args[1].pts, 50_000_000)

    def test_pre_rec_event_and_actual_different_clock_are_dropped(self):
        for reason in ("pre_rec", "different_clock", "capture_clock_replaced", "invalid_recorder_base"):
            with self.subTest(reason=reason):
                pipeline, _capture, recorder = self.make_pipeline()
                stamp = pipeline._capture_recording_time(self.sample())
                if reason == "pre_rec":
                    recorder.base = stamp.absolute_time + 1
                elif reason == "different_clock":
                    recorder.get_clock = lambda: object()
                elif reason == "capture_clock_replaced":
                    pipeline._pipeline.get_clock = lambda: object()
                else:
                    recorder.base = Gst.CLOCK_TIME_NONE
                self.assertEqual(pipeline._push_recording_frame(b"pixels", capture_time=stamp), Gst.FlowReturn.OK)
                pipeline._rec_appsrc.emit.assert_not_called()

    def test_record_start_or_restart_during_processing_does_not_retarget_frame(self):
        for already_recording in (False, True):
            with self.subTest(already_recording=already_recording):
                pipeline, _capture, _recorder = self.make_pipeline()
                pipeline._recording = already_recording
                stamp = pipeline._capture_recording_time(self.sample())
                pipeline._recording = True
                pipeline._recording_session += 1
                self.assertEqual(pipeline._push_recording_frame(b"pixels", capture_time=stamp), Gst.FlowReturn.OK)
                pipeline._rec_appsrc.emit.assert_not_called()

    def test_old_capture_callback_cannot_combine_its_sample_with_rebuilt_base(self):
        pipeline, capture, _recorder = self.make_pipeline()
        generation = pipeline._capture_generation
        pipeline._pipeline = SimpleNamespace(get_clock=lambda: object(), get_base_time=lambda: 9999)
        pipeline._capture_generation += 1
        stamp = pipeline._capture_recording_time(self.sample(), capture, generation)
        self.assertEqual(stamp.absolute_time, 1600 * Gst.SECOND + 50_000_000)
        self.assertEqual(pipeline._push_recording_frame(b"pixels", capture_time=stamp), Gst.FlowReturn.OK)
        pipeline._rec_appsrc.emit.assert_not_called()

    def test_retired_or_detached_sink_is_not_treated_as_current_source(self):
        for detached in (False, True):
            with self.subTest(detached=detached):
                pipeline, old_capture, recorder = self.make_pipeline()
                pipeline._running, pipeline._vcam_enabled = True, False
                pipeline._pipeline = SimpleNamespace(
                    get_clock=recorder.get_clock, get_base_time=lambda: 9999 * Gst.SECOND)
                parent = None if detached else old_capture
                sample = self.sample()
                sink = SimpleNamespace(emit=lambda _name: sample, get_parent=lambda: parent)
                pipeline._on_preview_sample(sink, pipeline._capture_generation)
                pipeline._rec_appsrc.emit.assert_not_called()

    def test_stop_clearing_capture_without_new_generation_drops_retiring_sample(self):
        pipeline, old_capture, _recorder = self.make_pipeline()
        pipeline._pipeline = None
        pipeline._running, pipeline._vcam_enabled = True, False
        sample = self.sample()
        sink = SimpleNamespace(emit=lambda _name: sample, get_parent=lambda: old_capture)
        pipeline._on_preview_sample(sink, pipeline._capture_generation)
        pipeline._rec_appsrc.emit.assert_not_called()

    def test_capture_clear_during_clock_read_is_dropped_without_dereferencing_none(self):
        pipeline, capture, recorder = self.make_pipeline()
        stamp = pipeline._capture_recording_time(self.sample())
        clock = capture.get_clock()

        def clear_capture():
            pipeline._pipeline = None
            return clock

        recorder.get_clock = clear_capture
        self.assertEqual(pipeline._push_recording_frame(b"old", capture_time=stamp), Gst.FlowReturn.OK)
        pipeline._rec_appsrc.emit.assert_not_called()

    def test_session_started_without_capture_clock_retains_video_arrival_timing(self):
        pipeline, capture, _old_recorder = self.make_pipeline()
        pipeline._recording = False
        capture.get_clock = lambda: None
        recorder, recsrc, recfile = mock.Mock(), mock.Mock(), mock.Mock()
        recorder.get_by_name.side_effect = {"recsrc": recsrc, "recfile": recfile}.__getitem__
        recorder.set_state.return_value = Gst.StateChangeReturn.ASYNC
        recsrc.emit.return_value = Gst.FlowReturn.OK
        with mock.patch.object(pipeline, "_select_recording_encoder", return_value="identity"), \
             mock.patch.object(pipeline, "_recording_audio_source", return_value=(None, "no audio")), \
             mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch", return_value=recorder):
            pipeline.start_recording("/tmp/generated.mp4")
        recorder.use_clock.assert_not_called()
        self.assertIsNone(pipeline._recording_capture_clock)
        base = 1600 * Gst.SECOND
        recorder.get_base_time.return_value = base
        recorder.get_clock.return_value = SimpleNamespace(get_time=lambda: base + 200_000_000)
        capture.get_clock = lambda: object()
        # A later valid camera clock may differ from the recorder's audio clock.
        stamp = pipeline._capture_recording_time(self.sample())
        self.assertIsNotNone(stamp.absolute_time)
        pipeline._push_recording_frame(b"video", capture_time=stamp)
        self.assertEqual(pipeline._rec_appsrc.emit.call_args.args[1].pts, 200_000_000)

    def test_invalid_source_timing_and_legacy_callers_keep_arrival_fallback(self):
        for invalid in ("missing_pts", "missing_segment", "wrong_segment", "missing_clock", "invalid_base", "legacy"):
            with self.subTest(invalid=invalid):
                pipeline, capture, _recorder = self.make_pipeline()
                sample = self.sample()
                if invalid == "missing_pts":
                    sample = self.sample(Gst.CLOCK_TIME_NONE)
                elif invalid == "missing_segment":
                    sample = SimpleNamespace(get_segment=lambda: None, get_buffer=sample.get_buffer)
                elif invalid == "wrong_segment":
                    sample = SimpleNamespace(get_segment=lambda: SimpleNamespace(format=Gst.Format.BYTES),
                                             get_buffer=sample.get_buffer)
                elif invalid == "missing_clock":
                    capture.get_clock = lambda: None
                elif invalid == "invalid_base":
                    capture.base = Gst.CLOCK_TIME_NONE
                stamp = None if invalid == "legacy" else pipeline._capture_recording_time(sample)
                pipeline._push_recording_frame(b"pixels", capture_time=stamp)
                buffer = pipeline._rec_appsrc.emit.call_args.args[1]
                self.assertEqual((buffer.pts, buffer.dts), (Gst.CLOCK_TIME_NONE, Gst.CLOCK_TIME_NONE))
                self.assertEqual(buffer.duration, Gst.SECOND // 30)

    def test_session_change_during_payload_copy_cannot_enqueue_into_either_recorder(self):
        pipeline, _capture, _recorder = self.make_pipeline()
        old_appsrc = pipeline._rec_appsrc
        new_appsrc = mock.Mock()
        stamp = pipeline._capture_recording_time(self.sample())

        def replace_recorder(*_args):
            pipeline._recording_session += 1
            pipeline._rec_appsrc = new_appsrc

        buffer = SimpleNamespace(fill=replace_recorder)
        with mock.patch.object(Gst.Buffer, "new_allocate", return_value=buffer):
            self.assertEqual(pipeline._push_recording_frame(b"pixels", capture_time=stamp), Gst.FlowReturn.FLUSHING)
        old_appsrc.emit.assert_not_called()
        new_appsrc.emit.assert_not_called()

    def test_arrival_fallback_then_earlier_valid_event_drops_without_clamping(self):
        pipeline, capture, recorder = self.make_pipeline()
        clock = SimpleNamespace(get_time=lambda: recorder.base + 200_000_000)
        capture.get_clock = recorder.get_clock = lambda: clock
        pipeline._recording_capture_clock = clock
        missing = pipeline._capture_recording_time(self.sample(Gst.CLOCK_TIME_NONE))
        pipeline._push_recording_frame(b"missing", capture_time=missing)
        first = pipeline._rec_appsrc.emit.call_args.args[1]
        self.assertEqual((first.pts, first.dts), (200_000_000, 200_000_000))
        earlier = pipeline._capture_recording_time(self.sample())
        pipeline._push_recording_frame(b"older", capture_time=earlier)
        pipeline._push_recording_frame(b"same", capture_time=missing)
        self.assertEqual(pipeline._rec_appsrc.emit.call_count, 1)
        later = pipeline._capture_recording_time(self.sample(600 * Gst.SECOND + 250_000_000))
        pipeline._push_recording_frame(b"later", capture_time=later)
        self.assertEqual(pipeline._rec_appsrc.emit.call_count, 2)
        last = pipeline._rec_appsrc.emit.call_args.args[1]
        self.assertEqual(last.pts, 250_000_000)

    def test_failed_enqueue_does_not_advance_timestamp_guard(self):
        pipeline, _capture, _recorder = self.make_pipeline()
        pipeline._rec_appsrc.emit.return_value = Gst.FlowReturn.FLUSHING
        stamp = pipeline._capture_recording_time(self.sample())
        pipeline._push_recording_frame(b"failed", capture_time=stamp)
        self.assertIsNone(pipeline._recording_last_pts)
        pipeline._rec_appsrc.emit.return_value = Gst.FlowReturn.OK
        pipeline._push_recording_frame(b"retry", capture_time=stamp)
        self.assertEqual(pipeline._recording_last_pts, 50_000_000)

    def test_all_three_capture_routes_snapshot_before_processing(self):
        for route in ("preview", "cpu", "gpu"):
            with self.subTest(route=route):
                pipeline, capture, _recorder = self.make_pipeline()
                pipeline._running, pipeline._vcam_enabled, pipeline._preview_enabled = True, False, False
                pipeline._width = pipeline._height = 2
                pipeline._mark_capture_started = lambda: setattr(capture, "base", 9999 * Gst.SECOND)
                sample = self.sample()
                sink = SimpleNamespace(emit=lambda _name: sample, get_parent=lambda: capture)
                if route == "preview":
                    callback = pipeline._on_preview_sample
                elif route == "cpu":
                    pipeline._effect_callback = lambda frame, *_args: frame
                    callback = pipeline._on_effects_sample
                else:
                    pipeline._frame_processor = SimpleNamespace(
                        configure=lambda *_args: True, ingest=lambda *_args: None,
                        composite=lambda **_kwargs: None,
                        download_bgra=lambda: np.full((2, 2, 4), 100, np.uint8))
                    callback = pipeline._on_effects_sample_gpu
                self.assertEqual(callback(sink, pipeline._capture_generation), Gst.FlowReturn.OK)
                buffer = pipeline._rec_appsrc.emit.call_args.args[1]
                self.assertEqual((buffer.pts, buffer.dts), (50_000_000, 50_000_000))
                self.assertEqual(pipeline._callbacks_in_flight, 0)

    def test_start_pins_capture_clock_before_playing_and_publishes_new_session(self):
        pipeline, capture, _recorder = self.make_pipeline()
        pipeline._recording = False
        pipeline._recording_last_pts = 900 * Gst.SECOND
        recorder = mock.Mock()
        recsrc, recfile = mock.Mock(), mock.Mock()
        recorder.get_by_name.side_effect = lambda name: recsrc if name == "recsrc" else recfile

        def play(state):
            self.assertEqual(state, Gst.State.PLAYING)
            recorder.use_clock.assert_called_once_with(capture.get_clock())
            return Gst.StateChangeReturn.ASYNC

        recorder.set_state.side_effect = play
        with mock.patch.object(pipeline, "_select_recording_encoder", return_value="identity"), \
             mock.patch.object(pipeline, "_recording_audio_source", return_value=(None, "no audio")), \
             mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch", return_value=recorder):
            pipeline.start_recording("/tmp/generated.mp4")
        recsrc.set_property.assert_called_once_with("block", False)
        self.assertEqual(pipeline._recording_session, 4)
        self.assertIsNone(pipeline._recording_last_pts)
        self.assertIs(pipeline._recording_pipeline, recorder)


if __name__ == "__main__":
    unittest.main()

import unittest
import threading
import time
from unittest import mock

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from nvbroadcast.video.pipeline import VideoPipeline


class VideoPipelineRebuildTests(unittest.TestCase):
    def _fake_gst_pipeline(self):
        fake_pipeline = mock.Mock()
        fake_sink = mock.Mock()
        fake_bus = mock.Mock()
        fake_pipeline.get_by_name.return_value = fake_sink
        fake_pipeline.get_bus.return_value = fake_bus
        return fake_pipeline

    def test_effects_pipeline_uses_raw_source_without_jpeg_decode(self):
        pipeline = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.select_camera_capture_format",
            return_value="raw",
        ):
            pipeline.configure(
                "/dev/video1",
                "/dev/video10",
                width=640,
                height=480,
                fps=30,
            )
        pipeline._effects_active = True

        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch", return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=False)

        pipeline_str = parse_launch.call_args.args[0]
        self.assertIn("video/x-raw,width=640,height=480,framerate=30/1", pipeline_str)
        self.assertNotIn("image/jpeg", pipeline_str)
        self.assertNotIn("jpegdec", pipeline_str)

    def test_effects_pipeline_keeps_mjpeg_decode_when_supported(self):
        pipeline = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.select_camera_capture_format",
            return_value="mjpeg",
        ):
            pipeline.configure(
                "/dev/video1",
                "/dev/video10",
                width=1280,
                height=720,
                fps=30,
            )
        pipeline._effects_active = True

        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch", return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=False)

        pipeline_str = parse_launch.call_args.args[0]
        self.assertIn("image/jpeg,width=1280,height=720,framerate=30/1", pipeline_str)
        self.assertIn("jpegdec", pipeline_str)

    def _gpu_pipeline(self, output_format="YUY2", capture="mjpeg"):
        pipeline = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.select_camera_capture_format",
            return_value=capture,
        ):
            pipeline.configure(
                "/dev/video1", "/dev/video10",
                width=1280, height=720, fps=30,
                output_format=output_format,
            )
        pipeline._effects_active = True
        processor = mock.Mock()
        processor.configure.return_value = True
        pipeline.set_frame_processor(processor, lambda: (True, False, True))
        return pipeline

    def test_gpu_capture_leg_has_no_convert_element(self):
        pipeline = self._gpu_pipeline()
        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch",
                        return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=False)
        pipeline_str = parse_launch.call_args.args[0]
        self.assertIn("jpegdec", pipeline_str)
        self.assertNotIn("videoconvert", pipeline_str)
        self.assertNotIn("cudaconvert", pipeline_str)
        self.assertNotIn("BGRA", pipeline_str)
        # the appsink caps restrict to formats the GPU kernels support
        sink = fake_pipeline.get_by_name.return_value
        caps_calls = [c for c in sink.set_property.call_args_list
                      if c.args and c.args[0] == "caps"]
        self.assertEqual(len(caps_calls), 1)
        self.assertIn("I420", caps_calls[0].args[1].to_string())

    def test_gpu_vcam_leg_is_convert_free_yuy2(self):
        pipeline = self._gpu_pipeline()
        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch",
                        return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=True)
        vcam_str = parse_launch.call_args_list[-1].args[0]
        self.assertIn("format=YUY2", vcam_str)
        self.assertIn("v4l2sink", vcam_str)
        self.assertNotIn("videoconvert", vcam_str)
        self.assertNotIn("cudaconvert", vcam_str)

    def test_demoted_gpu_path_rebuilds_legacy_strings(self):
        pipeline = self._gpu_pipeline()
        pipeline._gpu_path_demoted = True
        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch",
                        return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=False)
        pipeline_str = parse_launch.call_args.args[0]
        self.assertIn("format=BGRA", pipeline_str)

    def test_non_yuy2_output_format_uses_legacy_path(self):
        pipeline = self._gpu_pipeline(output_format="NV12")
        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch",
                        return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=False)
        pipeline_str = parse_launch.call_args.args[0]
        self.assertIn("format=BGRA", pipeline_str)
        self.assertFalse(pipeline._gpu_capture_active)

    def test_set_effects_active_queues_only_one_rebuild(self):
        pipeline = VideoPipeline()
        pipeline._running = True

        with mock.patch("nvbroadcast.video.pipeline.GLib.timeout_add", return_value=41) as timeout_add:
            pipeline.set_effects_active(True)
            pipeline.set_effects_active(False)

        timeout_add.assert_called_once_with(
            10, pipeline._rebuild_pipeline, priority=mock.ANY
        )
        self.assertTrue(pipeline._rebuild_pending)
        self.assertEqual(pipeline._rebuild_source_id, 41)
        self.assertFalse(pipeline._effects_active)

    def test_rebuild_waits_for_teardown_before_restart(self):
        pipeline = VideoPipeline()
        pipeline._pipeline = object()
        pipeline._vcam_enabled = False
        pipeline._rebuild_pending = True
        pipeline._rebuild_source_id = 17

        def fake_stop(*, clear_rebuild_request=True):
            self = pipeline
            self._pipeline = None
            self._teardown_done = False

        pipeline.stop = mock.Mock(side_effect=fake_stop)
        pipeline.build = mock.Mock()
        pipeline.start = mock.Mock()

        first = pipeline._rebuild_pipeline()

        pipeline.stop.assert_called_once_with(clear_rebuild_request=False)
        pipeline.build.assert_not_called()
        pipeline.start.assert_not_called()
        self.assertTrue(first)

        pipeline._teardown_done = True
        second = pipeline._rebuild_pipeline()

        pipeline.build.assert_called_once_with(vcam_enabled=False)
        pipeline.start.assert_called_once_with()
        self.assertFalse(second)
        self.assertFalse(pipeline._rebuild_pending)
        self.assertEqual(pipeline._rebuild_source_id, 0)

    def test_stop_cancels_pending_rebuild(self):
        pipeline = VideoPipeline()
        pipeline._running = True
        pipeline._pipeline = mock.Mock()
        pipeline._rebuild_pending = True
        pipeline._rebuild_source_id = 123

        with mock.patch("nvbroadcast.video.pipeline.GLib.source_remove") as source_remove, \
             mock.patch("nvbroadcast.video.pipeline.GLib.timeout_add", return_value=456):
            pipeline.stop()

        source_remove.assert_called_once_with(123)
        self.assertFalse(pipeline._rebuild_pending)
        self.assertEqual(pipeline._rebuild_source_id, 0)
        self.assertEqual(pipeline._teardown_source_id, 456)

    def test_effects_sample_uses_stable_vcam_appsrc_reference(self):
        pipeline = VideoPipeline()
        pipeline._running = True
        pipeline._vcam_enabled = True
        pipeline._width = 2
        pipeline._height = 2
        pipeline._effect_callback = lambda frame, _w, _h: frame

        frame = bytes([0] * (pipeline._width * pipeline._height * 4))
        sample_buffer = Gst.Buffer.new_wrapped(frame)
        sample_buffer.pts = 123
        sample_buffer.duration = 456

        sample = mock.Mock()
        sample.get_buffer.return_value = sample_buffer
        appsink = mock.Mock()
        appsink.emit.return_value = sample

        class RaceAppSrc:
            def __init__(self, owner):
                self.owner = owner
                self.calls = 0

            def __bool__(self):
                self.owner._vcam_appsrc = None
                return True

            def emit(self, signal_name, _buffer):
                self.calls += 1
                return None

        appsrc = RaceAppSrc(pipeline)
        pipeline._vcam_appsrc = appsrc

        result = pipeline._on_effects_sample(appsink)

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertEqual(appsrc.calls, 1)

    def test_alpha_worker_reuses_single_thread_and_keeps_latest_frame(self):
        pipeline = VideoPipeline()
        seen_threads = []
        processed_markers = []
        first_started = threading.Event()
        second_done = threading.Event()

        def alpha_callback(frame_data, _width, _height):
            seen_threads.append(threading.get_ident())
            processed_markers.append(frame_data[0])
            if len(processed_markers) == 1:
                first_started.set()
                time.sleep(0.05)
            elif len(processed_markers) >= 2:
                second_done.set()

        pipeline.set_alpha_callback(alpha_callback)

        try:
            pipeline._submit_alpha_frame(bytes([1]) * 16, 2, 2)
            self.assertTrue(first_started.wait(1.0))
            pipeline._submit_alpha_frame(bytes([2]) * 16, 2, 2)
            pipeline._submit_alpha_frame(bytes([3]) * 16, 2, 2)
            self.assertTrue(second_done.wait(1.0))
        finally:
            pipeline._stop_alpha_worker()

        self.assertGreaterEqual(len(processed_markers), 2)
        self.assertEqual(processed_markers[0], 1)
        self.assertEqual(processed_markers[1], 3)
        self.assertEqual(len(set(seen_threads)), 1)


if __name__ == "__main__":
    unittest.main()

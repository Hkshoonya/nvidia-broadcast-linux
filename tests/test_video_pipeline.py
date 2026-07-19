import unittest
import threading
import time
from types import SimpleNamespace
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

    def test_macos_both_pipeline_modes_initialize_obs_backend(self):
        import nvbroadcast.core.platform as platform_mod

        for builder_name in (
            "_build_passthrough_pipeline",
            "_build_effects_pipeline",
        ):
            with self.subTest(builder=builder_name):
                pipeline = VideoPipeline()
                pipeline._source_device = "0"
                pipeline._capture_format = "raw"
                pipeline._width = 640
                pipeline._height = 480
                pipeline._fps = 30
                fake_pipeline = self._fake_gst_pipeline()

                with mock.patch.object(platform_mod, "IS_MACOS", True), \
                     mock.patch(
                         "nvbroadcast.video.pipeline.Gst.parse_launch",
                         return_value=fake_pipeline,
                     ), mock.patch.object(
                         pipeline, "_setup_macos_virtual_camera"
                     ) as setup_vcam:
                    getattr(pipeline, builder_name)(True)

                setup_vcam.assert_called_once_with()

    def test_macos_backend_uses_obs_with_bgr_pixel_format(self):
        pipeline = VideoPipeline()
        pipeline._width = 640
        pipeline._height = 480
        pipeline._fps = 30
        camera = SimpleNamespace(device="OBS Virtual Camera", close=mock.Mock())
        camera_factory = mock.Mock(return_value=camera)
        bgr_format = object()
        pyvirtualcam = SimpleNamespace(
            Camera=camera_factory,
            PixelFormat=SimpleNamespace(BGR=bgr_format),
        )

        with mock.patch.dict("sys.modules", {"pyvirtualcam": pyvirtualcam}):
            pipeline._setup_macos_virtual_camera()

        camera_factory.assert_called_once_with(
            width=640,
            height=480,
            fps=30,
            fmt=bgr_format,
            backend="obs",
        )
        self.assertIs(pipeline._pyvirtualcam, camera)
        self.assertFalse(pipeline._vcam_failed)
        self.assertTrue(pipeline.virtual_camera_active)

    def test_macos_frame_conversion_is_contiguous_bgr(self):
        pipeline = VideoPipeline()
        pipeline._width = 2
        pipeline._height = 1
        camera = SimpleNamespace(send=mock.Mock(), close=mock.Mock())
        pipeline._pyvirtualcam = camera

        pipeline._send_macos_virtual_camera_frame(
            bytes((10, 20, 30, 255, 40, 50, 60, 255))
        )

        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.tolist(), [[[10, 20, 30], [40, 50, 60]]])
        self.assertTrue(sent.flags.c_contiguous)

    def test_macos_send_failure_disables_only_virtual_camera(self):
        pipeline = VideoPipeline()
        pipeline._width = 1
        pipeline._height = 1
        pipeline._vcam_enabled = True
        camera = SimpleNamespace(
            send=mock.Mock(side_effect=RuntimeError("backend stopped")),
            close=mock.Mock(),
        )
        pipeline._pyvirtualcam = camera

        pipeline._send_macos_virtual_camera_frame(bytes((10, 20, 30, 255)))

        camera.close.assert_called_once_with()
        self.assertIsNone(pipeline._pyvirtualcam)
        self.assertFalse(pipeline._vcam_enabled)
        self.assertTrue(pipeline._vcam_failed)
        self.assertFalse(pipeline.virtual_camera_active)

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

    def test_macos_passthrough_sample_sends_frame_to_virtual_camera(self):
        pipeline = VideoPipeline()
        pipeline._running = True
        pipeline._vcam_enabled = True
        pipeline._width = 2
        pipeline._height = 1
        frame = bytes((10, 20, 30, 255, 40, 50, 60, 255))
        sample_buffer = Gst.Buffer.new_wrapped(frame)
        sample = mock.Mock()
        sample.get_buffer.return_value = sample_buffer
        appsink = mock.Mock()
        appsink.emit.return_value = sample

        with mock.patch.object(
            pipeline, "_send_macos_virtual_camera_frame"
        ) as send_frame:
            result = pipeline._on_preview_sample(appsink)

        self.assertEqual(result, Gst.FlowReturn.OK)
        send_frame.assert_called_once_with(frame)

    def test_macos_effects_sample_sends_processed_frame_to_virtual_camera(self):
        pipeline = VideoPipeline()
        pipeline._running = True
        pipeline._vcam_enabled = True
        pipeline._width = 2
        pipeline._height = 1
        input_frame = bytes((10, 20, 30, 255, 40, 50, 60, 255))
        output_frame = bytes((60, 50, 40, 255, 30, 20, 10, 255))
        pipeline._effect_callback = lambda _frame, _w, _h: output_frame
        sample_buffer = Gst.Buffer.new_wrapped(input_frame)
        sample_buffer.pts = 123
        sample_buffer.duration = 456
        sample = mock.Mock()
        sample.get_buffer.return_value = sample_buffer
        appsink = mock.Mock()
        appsink.emit.return_value = sample

        with mock.patch.object(
            pipeline, "_send_macos_virtual_camera_frame"
        ) as send_frame:
            result = pipeline._on_effects_sample(appsink)

        self.assertEqual(result, Gst.FlowReturn.OK)
        send_frame.assert_called_once_with(output_frame)

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

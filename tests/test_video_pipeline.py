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
    def setUp(self):
        with VideoPipeline._capture_success_cache_lock:
            VideoPipeline._capture_success_cache.clear()

    def tearDown(self):
        with VideoPipeline._capture_success_cache_lock:
            VideoPipeline._capture_success_cache.clear()

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
            "nvbroadcast.video.virtual_camera.camera_capture_candidates",
            return_value=[{
                "format": "raw", "width": 640, "height": 480, "fps": 30,
            }],
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
        self.assertIn("v4l2src name=camera_source", pipeline_str)
        self.assertIn("video/x-raw,width=640,height=480,framerate=30/1", pipeline_str)
        self.assertNotIn("image/jpeg", pipeline_str)
        self.assertNotIn("jpegdec", pipeline_str)

    def test_effects_pipeline_keeps_mjpeg_decode_when_supported(self):
        pipeline = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.camera_capture_candidates",
            return_value=[{
                "format": "mjpeg", "width": 1280, "height": 720, "fps": 30,
            }],
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

    @staticmethod
    def _capture_error(message, debug, source=None):
        msg = mock.Mock()
        msg.parse_error.return_value = (SimpleNamespace(message=message), debug)
        msg.src = source
        return msg

    @staticmethod
    def _set_camera_source(pipeline):
        camera_source = mock.Mock()
        pipeline._pipeline = mock.Mock()
        pipeline._pipeline.get_by_name.return_value = camera_source
        return camera_source

    def test_camera_source_startup_error_tries_next_advertised_format(self):
        pipeline = VideoPipeline()
        pipeline._capture_candidates = [
            {"format": "mjpeg", "width": 640, "height": 360, "fps": 30},
            {"format": "raw", "width": 640, "height": 360, "fps": 30},
        ]
        pipeline._capture_format = "mjpeg"
        pipeline._gpu_capture_active = True
        camera_source = self._set_camera_source(pipeline)

        with mock.patch(
            "nvbroadcast.video.pipeline.GLib.timeout_add", return_value=73
        ) as timeout_add:
            pipeline._on_error(
                None,
                self._capture_error(
                    "A future GStreamer error message.",
                    "No implementation-specific text required.",
                    camera_source,
                ),
            )

        self.assertEqual(pipeline._capture_format, "raw")
        self.assertEqual(pipeline._capture_candidate_index, 1)
        self.assertTrue(pipeline._capture_retry_pending)
        self.assertFalse(pipeline._gpu_path_demoted)
        timeout_add.assert_called_once_with(
            10, pipeline._rebuild_pipeline, priority=mock.ANY
        )
        self.assertTrue(pipeline._rebuild_pending)
        self.assertEqual(pipeline._rebuild_source_id, 73)

    def test_secondary_error_during_candidate_retry_is_ignored(self):
        pipeline = VideoPipeline()
        pipeline._capture_candidates = [
            {"format": "mjpeg", "width": 640, "height": 360, "fps": 30},
            {"format": "raw", "width": 640, "height": 360, "fps": 30},
        ]
        pipeline._capture_candidate_index = 1
        pipeline._capture_format = "raw"
        pipeline._capture_retry_pending = True
        pipeline._gpu_capture_active = True

        with mock.patch(
            "nvbroadcast.video.pipeline.GLib.idle_add"
        ) as idle_add:
            pipeline._on_error(
                None,
                self._capture_error(
                    "Internal data stream error.",
                    "streaming stopped, reason not-negotiated (-4)",
                ),
            )

        self.assertFalse(pipeline._gpu_path_demoted)
        idle_add.assert_not_called()

    def test_error_from_retired_pipeline_generation_is_ignored(self):
        pipeline = VideoPipeline()
        pipeline._capture_generation = 4
        message = self._capture_error(
            "Internal data stream error.", "retired pipeline"
        )

        with mock.patch(
            "nvbroadcast.video.pipeline.GLib.idle_add"
        ) as idle_add:
            pipeline._on_error(None, message, generation=3)

        message.parse_error.assert_not_called()
        idle_add.assert_not_called()

    def test_capture_bus_error_handler_records_pipeline_generation(self):
        pipeline = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.camera_capture_candidates",
            return_value=[{
                "format": "raw", "width": 640, "height": 480, "fps": 30,
            }],
        ):
            pipeline.configure(
                "/dev/video1", "/dev/video10",
                width=640, height=480, fps=30,
            )
        pipeline._effects_active = True
        fake_pipeline = self._fake_gst_pipeline()

        with mock.patch(
            "nvbroadcast.video.pipeline.Gst.parse_launch",
            return_value=fake_pipeline,
        ):
            pipeline.build(vcam_enabled=False)

        fake_pipeline.get_bus.return_value.connect.assert_any_call(
            "message::error", pipeline._on_error, 1
        )
        self.assertEqual(pipeline._capture_generation, 1)

    def test_successful_fallback_is_preferred_by_replacement_pipeline(self):
        candidates = [
            {"format": "mjpeg", "width": 640, "height": 360, "fps": 30},
            {"format": "raw", "width": 640, "height": 360, "fps": 30},
        ]
        first = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.camera_capture_candidates",
            return_value=candidates,
        ):
            first.configure(
                "/dev/video1", "/dev/video10",
                width=640, height=360, fps=30,
            )
        first._capture_candidate_index = 1
        first._capture_format = "raw"
        first._mark_capture_started()

        replacement = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.camera_capture_candidates",
            return_value=candidates,
        ):
            replacement.configure(
                "/dev/video1", "/dev/video10",
                width=640, height=360, fps=30,
            )

        self.assertEqual(replacement._capture_format, "raw")
        self.assertEqual(
            [candidate["format"] for candidate in replacement._capture_candidates],
            ["raw", "mjpeg"],
        )

    def test_failed_candidate_is_not_cached_before_a_valid_frame(self):
        pipeline = VideoPipeline()
        pipeline._capture_cache_key = ("/dev/video1", 640, 360, 30)
        pipeline._capture_candidates = [
            {"format": "mjpeg", "width": 640, "height": 360, "fps": 30},
            {"format": "raw", "width": 640, "height": 360, "fps": 30},
        ]
        pipeline._capture_format = "mjpeg"
        camera_source = self._set_camera_source(pipeline)

        with mock.patch(
            "nvbroadcast.video.pipeline.GLib.idle_add", return_value=73
        ):
            pipeline._on_error(
                None,
                self._capture_error("startup failed", None, camera_source),
            )

        self.assertNotIn(
            pipeline._capture_cache_key, VideoPipeline._capture_success_cache
        )

    def test_runtime_camera_error_does_not_change_capture_candidate(self):
        pipeline = VideoPipeline()
        pipeline._capture_candidates = [
            {"format": "mjpeg", "width": 1280, "height": 720, "fps": 30},
            {"format": "raw", "width": 1280, "height": 720, "fps": 30},
        ]
        pipeline._capture_format = "mjpeg"
        pipeline._capture_started = True
        pipeline._gpu_capture_active = True
        camera_source = self._set_camera_source(pipeline)

        with mock.patch(
            "nvbroadcast.video.pipeline.GLib.idle_add", return_value=74
        ) as idle_add:
            pipeline._on_error(
                None,
                self._capture_error(
                    "Internal data stream error.", "device disconnected",
                    camera_source,
                ),
            )

        self.assertEqual(pipeline._capture_format, "mjpeg")
        self.assertEqual(pipeline._capture_candidate_index, 0)
        self.assertTrue(pipeline._gpu_path_demoted)
        idle_add.assert_called_once_with(pipeline._queue_rebuild)

    def test_startup_error_reports_when_no_advertised_candidate_remains(self):
        pipeline = VideoPipeline()
        pipeline._capture_candidates = [
            {"format": "mjpeg", "width": 640, "height": 360, "fps": 30},
        ]
        pipeline._gpu_capture_active = False
        camera_source = self._set_camera_source(pipeline)

        with mock.patch("builtins.print") as print_mock, mock.patch(
            "nvbroadcast.video.pipeline.GLib.idle_add"
        ) as idle_add:
            pipeline._on_error(
                None,
                self._capture_error(
                    "Failed to allocate required memory.",
                    "Buffer pool activation failed",
                    camera_source,
                ),
            )

        rendered = "\n".join(str(call) for call in print_mock.call_args_list)
        self.assertIn("Failed to allocate required memory.", rendered)
        self.assertIn("Buffer pool activation failed", rendered)
        idle_add.assert_not_called()

    def _gpu_pipeline(self, output_format="YUY2", capture="mjpeg"):
        pipeline = VideoPipeline()
        with mock.patch(
            "nvbroadcast.video.virtual_camera.camera_capture_candidates",
            return_value=[{
                "format": capture, "width": 1280, "height": 720, "fps": 30,
            }],
        ):
            pipeline.configure(
                "/dev/video1", "/dev/video10",
                width=1280, height=720, fps=30,
                output_format=output_format,
            )
        pipeline._effects_active = True
        processor = mock.Mock()
        processor.configure.return_value = True
        processor.supports_jpeg = False
        pipeline.set_frame_processor(processor, lambda: (True, False, True))
        return pipeline

    def test_gpu_jpeg_capture_leg_skips_jpegdec(self):
        pipeline = self._gpu_pipeline()
        pipeline._frame_processor.supports_jpeg = True
        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch",
                        return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=False)
        pipeline_str = parse_launch.call_args.args[0]
        self.assertNotIn("jpegdec", pipeline_str)
        sink = fake_pipeline.get_by_name.return_value
        caps_calls = [c for c in sink.set_property.call_args_list
                      if c.args and c.args[0] == "caps"]
        self.assertIn("image/jpeg", caps_calls[0].args[1].to_string())
        self.assertTrue(pipeline._gpu_jpeg_active)

    def test_gpu_jpeg_demotion_falls_back_to_jpegdec_leg(self):
        pipeline = self._gpu_pipeline()
        pipeline._frame_processor.supports_jpeg = True
        pipeline._gpu_jpeg_demoted = True
        fake_pipeline = self._fake_gst_pipeline()
        with mock.patch("nvbroadcast.video.pipeline.Gst.parse_launch",
                        return_value=fake_pipeline) as parse_launch:
            pipeline.build(vcam_enabled=False)
        pipeline_str = parse_launch.call_args.args[0]
        self.assertIn("jpegdec", pipeline_str)
        self.assertFalse(pipeline._gpu_jpeg_active)
        self.assertTrue(pipeline._gpu_capture_active)

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
        self.assertIn("interlace-mode=progressive", vcam_str)
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

    def test_replacing_live_frame_processor_queues_rebuild_and_resets_demotion(self):
        pipeline = self._gpu_pipeline()
        pipeline._running = True
        pipeline._gpu_path_demoted = True
        replacement = mock.Mock()

        with mock.patch(
            "nvbroadcast.video.pipeline.GLib.timeout_add", return_value=71
        ) as timeout_add:
            pipeline.set_frame_processor(
                replacement, lambda: (True, False, True))

        self.assertIs(pipeline._frame_processor, replacement)
        self.assertFalse(pipeline._gpu_path_demoted)
        self.assertTrue(pipeline._rebuild_pending)
        timeout_add.assert_called_once()

    def test_detaching_frame_processor_waits_for_inflight_callback(self):
        pipeline = self._gpu_pipeline()
        pipeline._callbacks_in_flight = 1

        worker = threading.Thread(
            target=lambda: pipeline.set_frame_processor(
                None, None, wait_for_inflight=True)
        )
        worker.start()
        time.sleep(0.02)

        self.assertTrue(worker.is_alive())
        with pipeline._callback_lock:
            pipeline._callbacks_in_flight = 0
        worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertIsNone(pipeline._frame_processor)

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
        pipeline._capture_retry_pending = True

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
        self.assertFalse(pipeline._capture_retry_pending)
        self.assertEqual(pipeline._rebuild_source_id, 0)

    def test_stop_cancels_pending_rebuild(self):
        pipeline = VideoPipeline()
        pipeline._running = True
        pipeline._pipeline = mock.Mock()
        pipeline._rebuild_pending = True
        pipeline._rebuild_source_id = 123
        pipeline._capture_retry_pending = True

        with mock.patch("nvbroadcast.video.pipeline.GLib.source_remove") as source_remove, \
             mock.patch("nvbroadcast.video.pipeline.GLib.timeout_add", return_value=456):
            pipeline.stop()

        source_remove.assert_called_once_with(123)
        self.assertFalse(pipeline._rebuild_pending)
        self.assertEqual(pipeline._rebuild_source_id, 0)
        self.assertFalse(pipeline._capture_retry_pending)
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


class GpuPathFailureBridgingTests(unittest.TestCase):
    """A GPU frame path error must never leave the virtual camera without a
    frame: the last good payload is replayed while errors accumulate or the
    demotion rebuild is in flight."""

    def _make_pipeline(self, processor):
        pipeline = VideoPipeline()
        pipeline._running = True
        pipeline._vcam_enabled = True
        pipeline._vcam_appsrc = mock.Mock()
        pipeline._frame_processor = processor
        pipeline._frame_plan = None
        return pipeline

    @staticmethod
    def _make_appsink(fmt="I420", width=4, height=2):
        buf = mock.Mock()
        buf.pts = 0
        buf.duration = 0
        structure = mock.Mock()
        structure.get_name.return_value = "video/x-raw"
        structure.get_string.return_value = fmt
        structure.get_value.side_effect = lambda key: {
            "width": width, "height": height}[key]
        caps = mock.Mock()
        caps.get_structure.return_value = structure
        sample = mock.Mock()
        sample.get_buffer.return_value = buf
        sample.get_caps.return_value = caps
        appsink = mock.Mock()
        appsink.emit.return_value = sample
        return appsink

    def test_transient_error_replays_last_good_frame(self):
        processor = mock.Mock()
        processor.configure.return_value = True
        processor.ingest.side_effect = RuntimeError("transient CUDA hiccup")
        pipeline = self._make_pipeline(processor)
        payload = b"\x80" * (4 * 2 * 2)
        pipeline._last_good_yuy2 = (payload, (4, 2))
        pipeline._gpu_frame_size = (4, 2)

        result = pipeline._on_effects_sample_gpu(self._make_appsink())

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertEqual(pipeline._gpu_path_errors, 1)
        self.assertFalse(pipeline._gpu_path_demoted)
        pushes = [c for c in pipeline._vcam_appsrc.emit.call_args_list
                  if c.args[0] == "push-buffer"]
        self.assertEqual(len(pushes), 1, "vcam must still receive a frame")
        self.assertEqual(pushes[0].args[1].get_size(), len(payload))

    def test_stale_sized_frame_is_not_replayed(self):
        processor = mock.Mock()
        processor.configure.return_value = True
        processor.ingest.side_effect = RuntimeError("transient CUDA hiccup")
        pipeline = self._make_pipeline(processor)
        pipeline._last_good_yuy2 = (b"\x80" * 16, (2, 2))  # pre-rebuild size

        pipeline._on_effects_sample_gpu(self._make_appsink(width=4, height=2))

        pushes = [c for c in pipeline._vcam_appsrc.emit.call_args_list
                  if c.args[0] == "push-buffer"]
        self.assertEqual(pushes, [], "wrong-sized payload must not be pushed")

    def test_rejected_new_caps_never_replay_previous_sized_frame(self):
        processor = mock.Mock()
        processor.configure.return_value = False
        pipeline = self._make_pipeline(processor)
        pipeline._last_good_yuy2 = (b"\x80" * 8, (2, 2))
        pipeline._gpu_frame_size = (2, 2)

        pipeline._on_effects_sample_gpu(
            self._make_appsink(width=4, height=2))

        pushes = [c for c in pipeline._vcam_appsrc.emit.call_args_list
                  if c.args[0] == "push-buffer"]
        self.assertEqual(pushes, [], "old caps payload must not cross the change")
        self.assertTrue(pipeline._gpu_path_demoted)

    def test_unsupported_negotiation_demotes_immediately(self):
        processor = mock.Mock()
        processor.configure.return_value = False  # can never succeed on retry
        pipeline = self._make_pipeline(processor)

        result = pipeline._on_effects_sample_gpu(self._make_appsink())

        self.assertEqual(result, Gst.FlowReturn.OK)
        self.assertTrue(pipeline._gpu_path_demoted,
                        "persistent failure should not burn the 3-strike budget")


if __name__ == "__main__":
    unittest.main()

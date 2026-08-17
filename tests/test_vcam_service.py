import io
import sys
import unittest
from unittest import mock

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from nvbroadcast.vcam_service import (
    _strict_vcam_preference,
    build_pipeline,
    main,
    start_pipeline_with_fallback,
)


class VCamServicePipelineTests(unittest.TestCase):
    def test_main_stops_before_vcam_setup_when_no_camera_exists(self):
        config = mock.Mock()
        config.video.camera_device = "/dev/video0"
        config.video.width = 1280
        config.video.height = 720
        config.video.fps = 30
        config.video.vcam_device = "/dev/video10"

        with mock.patch.object(sys, "argv", ["nvbroadcast-vcam"]), mock.patch(
            "nvbroadcast.vcam_service.Gst.init"
        ), mock.patch(
            "nvbroadcast.vcam_service.load_config", return_value=config
        ), mock.patch(
            "nvbroadcast.vcam_service.resolve_camera_device",
            return_value=None,
        ) as resolve_camera_device, mock.patch(
            "nvbroadcast.vcam_service.ensure_virtual_camera"
        ) as ensure_virtual_camera, mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            with self.assertRaisesRegex(SystemExit, "1"):
                main()

        resolve_camera_device.assert_called_once_with("/dev/video0")
        ensure_virtual_camera.assert_not_called()
        self.assertIn("no usable physical camera found", stderr.getvalue())

    def test_strict_vcam_preference_only_for_explicit_or_non_default_device(self):
        self.assertIsNone(_strict_vcam_preference("/dev/video10"))
        self.assertEqual(_strict_vcam_preference("/dev/video11"), "/dev/video11")
        self.assertEqual(
            _strict_vcam_preference("/dev/video10", explicit=True),
            "/dev/video10",
        )

    def test_build_pipeline_uses_raw_source_without_jpeg_decode(self):
        with mock.patch(
            "nvbroadcast.vcam_service.select_camera_mode",
            return_value={"format": "raw", "width": 640, "height": 480, "fps": 30},
        ), mock.patch("nvbroadcast.vcam_service.Gst.parse_launch", return_value=mock.Mock()) as parse_launch:
            build_pipeline("/dev/video1", "/dev/video10", 640, 480, 30, "yuy2")

        pipeline_str = parse_launch.call_args.args[0]
        self.assertIn("video/x-raw,width=640,height=480,framerate=30/1", pipeline_str)
        self.assertNotIn("image/jpeg", pipeline_str)
        self.assertNotIn("jpegdec", pipeline_str)

    def test_start_pipeline_with_fallback_retries_next_camera_mode(self):
        first_pipeline = mock.Mock()
        second_pipeline = mock.Mock()
        candidates = [
            {"format": "mjpeg", "width": 1280, "height": 720, "fps": 60},
            {"format": "mjpeg", "width": 1280, "height": 720, "fps": 30},
        ]

        with mock.patch(
            "nvbroadcast.vcam_service.camera_mode_candidates",
            return_value=candidates,
        ), mock.patch(
            "nvbroadcast.vcam_service.build_pipeline",
            side_effect=[first_pipeline, second_pipeline],
        ) as build_pipeline_mock, mock.patch(
            "nvbroadcast.vcam_service._start_pipeline_once",
            side_effect=[False, True],
        ), mock.patch(
            "nvbroadcast.vcam_service._vcam_ready_for_writer",
            return_value=True,
        ):
            pipeline, mode = start_pipeline_with_fallback(
                "/dev/video0", "/dev/video10", 1280, 720, 60, "yuy2"
            )

        self.assertIs(pipeline, second_pipeline)
        self.assertEqual(mode, candidates[1])
        first_pipeline.set_state.assert_called_once_with(Gst.State.NULL)
        self.assertEqual(build_pipeline_mock.call_count, 2)
        self.assertEqual(build_pipeline_mock.call_args_list[0].kwargs["capture_format"], "mjpeg")
        self.assertEqual(build_pipeline_mock.call_args_list[1].args[4], 30)

    def test_start_pipeline_with_fallback_stops_when_vcam_is_busy(self):
        with mock.patch(
            "nvbroadcast.vcam_service._vcam_ready_for_writer",
            return_value=False,
        ), mock.patch(
            "nvbroadcast.vcam_service._describe_vcam_device",
            return_value="caps=capture, holders=python",
        ), mock.patch(
            "nvbroadcast.vcam_service.build_pipeline",
        ) as build_pipeline_mock:
            pipeline, mode = start_pipeline_with_fallback(
                "/dev/video0", "/dev/video10", 1280, 720, 30, "yuy2"
            )

        self.assertIsNone(pipeline)
        self.assertIsNone(mode)
        build_pipeline_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

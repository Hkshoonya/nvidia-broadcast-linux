import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig


class AppGpuFramePolicyTests(unittest.TestCase):
    @staticmethod
    def _make_app():
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.compute_focus = "gpu"
        app.config.compositing = "cupy"
        app.config.video.output_format = "YUY2"
        app._gpu_frame_path = None
        app._gpu_frame_path_failed = False
        app._video_pipeline = None
        app._video_effects = SimpleNamespace(
            _gpu_index=0,
            available=False,
            _cleanup_backend=mock.Mock(),
            initialize=mock.Mock(),
        )
        app._perf_monitor = SimpleNamespace(set_gpu_index=mock.Mock())
        app._window = None
        return app

    @mock.patch("nvbroadcast.app.IS_MACOS", False)
    def test_cpu_focus_disables_gpu_frame_transport(self):
        app = self._make_app()
        app.config.compute_focus = "cpu"

        self.assertFalse(app._gpu_frame_path_allowed())

    @mock.patch("nvbroadcast.app.IS_MACOS", False)
    def test_cpu_compositing_disables_gpu_frame_transport(self):
        app = self._make_app()
        app.config.compositing = "cpu"

        self.assertFalse(app._gpu_frame_path_allowed())

    @mock.patch("nvbroadcast.app.IS_MACOS", False)
    def test_gpu_policy_enables_yuy2_frame_transport(self):
        app = self._make_app()

        self.assertTrue(app._gpu_frame_path_allowed())
        self.assertFalse(app._gpu_frame_path_allowed("NV12"))

    @mock.patch("nvbroadcast.app.IS_MACOS", False)
    def test_sync_detaches_processor_for_cpu_policy(self):
        app = self._make_app()
        old_processor = object()
        app._gpu_frame_path = old_processor
        app._video_pipeline = mock.Mock()
        app.config.compute_focus = "cpu"

        app._sync_gpu_frame_path()

        self.assertIsNone(app._gpu_frame_path)
        app._video_pipeline.set_frame_processor.assert_called_once_with(
            None, None, wait_for_inflight=True)

    @mock.patch("nvbroadcast.app.save_config")
    @mock.patch("nvbroadcast.core.config.apply_performance_profile")
    @mock.patch("nvbroadcast.app.IS_MACOS", False)
    def test_live_cpu_mode_change_detaches_gpu_transport(
        self, _apply_profile, _save
    ):
        app = self._make_app()
        old_processor = object()
        app._gpu_frame_path = old_processor
        app._video_pipeline = mock.Mock()
        app._video_effects.set_compositing = mock.Mock()
        app._video_effects.set_engine_mode = mock.Mock()
        app._video_effects.set_profile_infer_height = mock.Mock()
        app._video_effects._apply_edge_config = mock.Mock()
        app._video_effects._backend = None
        app._beautifier = SimpleNamespace(set_compositing=mock.Mock())
        app._refresh_inference_policy = mock.Mock()
        app._inline_inference = False
        app._use_nvdec = False

        app.set_performance_profile(
            "max_quality",
            compositing="cpu",
            mode_key="cpu_quality",
        )

        self.assertEqual(app.config.compositing, "cpu")
        self.assertIsNone(app._gpu_frame_path)
        app._video_pipeline.set_frame_processor.assert_called_once_with(
            None, None, wait_for_inflight=True)

    @mock.patch("nvbroadcast.app.save_config")
    @mock.patch("nvbroadcast.core.gpu.detect_gpus", return_value=[])
    @mock.patch("nvbroadcast.video.gpu_frame_path.GpuFramePath.create")
    @mock.patch("nvbroadcast.app.IS_MACOS", False)
    def test_gpu_switch_recreates_and_rebinds_live_processor(
        self, create, _detect_gpus, _save
    ):
        app = self._make_app()
        old_processor = object()
        new_processor = object()
        app._gpu_frame_path = old_processor
        app._video_pipeline = mock.Mock()
        app._video_effects.available = True
        create.return_value = new_processor

        app.set_compute_gpu(1)

        self.assertEqual(app.config.compute_gpu, 1)
        self.assertEqual(app._video_effects._gpu_index, 1)
        app._video_effects._cleanup_backend.assert_called_once_with()
        app._video_effects.initialize.assert_called_once_with()
        create.assert_called_once_with(app._video_effects, gpu_index=1)
        self.assertEqual(
            app._video_pipeline.set_frame_processor.call_args_list[0],
            mock.call(None, None, wait_for_inflight=True),
        )
        rebound = app._video_pipeline.set_frame_processor.call_args_list[1]
        self.assertIs(rebound.args[0], new_processor)
        self.assertIs(rebound.args[1].__self__, app)
        self.assertFalse(rebound.kwargs["wait_for_inflight"])


if __name__ == "__main__":
    unittest.main()

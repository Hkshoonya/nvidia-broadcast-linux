import unittest
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig


class AppGpuFramePolicyTests(unittest.TestCase):
    @staticmethod
    def _mode_snapshot(gpu_index: int) -> dict[str, object]:
        return {
            "gpu_index": gpu_index,
            "has_cuda": True,
            "has_tensorrt": True,
            "modes": {},
        }

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
            set_gpu_index=mock.Mock(),
        )
        app._perf_monitor = SimpleNamespace(set_gpu_index=mock.Mock())
        app._dependency_installer = SimpleNamespace(set_compute_gpu=mock.Mock())
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
        app._dependency_installer.set_compute_gpu.assert_called_once_with(1)
        app._video_effects.set_gpu_index.assert_called_once_with(1)
        app._video_effects._cleanup_backend.assert_not_called()
        app._video_effects.initialize.assert_not_called()
        create.assert_called_once_with(app._video_effects, gpu_index=1)
        self.assertEqual(
            app._video_pipeline.set_frame_processor.call_args_list[0],
            mock.call(None, None, wait_for_inflight=True),
        )
        rebound = app._video_pipeline.set_frame_processor.call_args_list[1]
        self.assertIs(rebound.args[0], new_processor)
        self.assertIs(rebound.args[1].__self__, app)
        self.assertFalse(rebound.kwargs["wait_for_inflight"])

    def test_mode_availability_refresh_rebuilds_only_for_current_gpu(self):
        app = self._make_app()
        app.config.compute_gpu = 1
        app._mode_availability_generation = 3
        app._window = SimpleNamespace(
            rebuild_mode_selector=mock.Mock(),
            set_status=mock.Mock(),
        )

        self.assertFalse(app._finish_mode_availability_refresh(2, 1))
        app._window.rebuild_mode_selector.assert_not_called()

        snapshot = self._mode_snapshot(1)
        self.assertFalse(
            app._finish_mode_availability_refresh(
                3,
                1,
                snapshot,
                {"cpu_cores": 8, "has_nvidia": True, "gpu_vram_mb": 8192},
            )
        )
        self.assertTrue(app._window._mode_availability_ready)
        self.assertEqual(app._window._mode_availability_gpu, 1)
        app._window.rebuild_mode_selector.assert_called_once_with(
            app.config.compositing,
            app.config.performance_profile,
            availability_snapshot=snapshot,
        )

    @mock.patch("nvbroadcast.app.save_config")
    @mock.patch("nvbroadcast.core.gpu.detect_gpus", return_value=[])
    def test_gpu_switch_invalidates_old_mode_rows_before_async_refresh(
        self, _detect_gpus, _save_config
    ):
        app = self._make_app()
        app._sync_gpu_frame_path = mock.Mock()
        app._refresh_mode_availability_async = mock.Mock()
        app._window = SimpleNamespace(
            _mode_retry_generation=5,
            _mode_retry_in_flight=True,
            _mode_availability_ready=True,
            _mode_availability_gpu=0,
            _mode_availability_snapshot=self._mode_snapshot(0),
            _update_gpu_info=mock.Mock(),
            rebuild_mode_selector=mock.Mock(),
            set_status=mock.Mock(),
        )

        app.set_compute_gpu(1)

        self.assertEqual(app._window._mode_retry_generation, 6)
        self.assertFalse(app._window._mode_retry_in_flight)
        self.assertFalse(app._window._mode_availability_ready)
        self.assertEqual(app._window._mode_availability_gpu, 1)
        app._refresh_mode_availability_async.assert_called_once_with(1)

    def test_mode_availability_worker_uses_captured_gpu_checker(self):
        app = self._make_app()
        app.config.compute_gpu = 2
        app._dependency_installer = mock.Mock()
        snapshot = self._mode_snapshot(2)
        app._dependency_installer.mode_availability_snapshot.return_value = snapshot
        app._window = SimpleNamespace(
            rebuild_mode_selector=mock.Mock(),
            set_status=mock.Mock(),
        )

        class ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self._target = target

            def start(self):
                self._target()

        with mock.patch(
            "nvbroadcast.app.threading.Thread",
            ImmediateThread,
        ), mock.patch(
            "nvbroadcast.app.GLib.idle_add",
            side_effect=lambda callback, *args: callback(*args),
        ), mock.patch(
            "nvbroadcast.core.config.detect_system_capabilities",
            return_value={
                "cpu_cores": 8,
                "has_nvidia": True,
                "gpu_vram_mb": 8192,
            },
        ):
            app._refresh_mode_availability_async(2)

        app._dependency_installer.mode_availability_snapshot.assert_called_once_with(2)

    @mock.patch("nvbroadcast.app.save_config")
    def test_setup_auto_on_new_gpu_waits_for_provider_refresh(self, _save_config):
        app = self._make_app()
        app.config.compute_gpu = 0
        app.config.first_run = True
        app.set_auto_mode_enabled = mock.Mock()

        def commit_gpu(gpu_index):
            app.config.compute_gpu = gpu_index

        app.set_compute_gpu = mock.Mock(side_effect=commit_gpu)
        app._window = SimpleNamespace(
            rebuild_mode_selector=mock.Mock(),
            _gpu_selector=None,
            set_status=mock.Mock(),
        )

        app._on_setup_complete(None, "auto", 1, "cupy")

        app.set_compute_gpu.assert_called_once_with(1)
        app.set_auto_mode_enabled.assert_not_called()
        self.assertEqual(app._pending_auto_mode_gpu, 1)
        self.assertTrue(app.config.auto_mode)
        self.assertEqual(app.config.compute_focus, "auto")
        self.assertIn(
            "Checking selected GPU",
            app._window.set_status.call_args.args[0],
        )

    def test_selected_gpu_refresh_finishes_deferred_setup_auto_mode(self):
        app = self._make_app()
        app.config.compute_gpu = 1
        app._mode_availability_generation = 6
        app._pending_auto_mode_gpu = 1
        app.set_auto_mode_enabled = mock.Mock()
        app._window = SimpleNamespace(
            rebuild_mode_selector=mock.Mock(),
            set_status=mock.Mock(),
        )

        self.assertFalse(
            app._finish_mode_availability_refresh(
                6,
                1,
                self._mode_snapshot(1),
                {"cpu_cores": 8, "has_nvidia": True, "gpu_vram_mb": 8192},
            )
        )

        app.set_auto_mode_enabled.assert_called_once_with(True)
        self.assertIsNone(app._pending_auto_mode_gpu)
        app._window.set_status.assert_called_once_with("Auto mode enabled")


if __name__ == "__main__":
    unittest.main()

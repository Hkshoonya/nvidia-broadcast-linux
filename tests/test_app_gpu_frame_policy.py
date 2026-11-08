import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig
from nvbroadcast.runtime.process_scope import ProviderProbeScope


class AppGpuFramePolicyTests(unittest.TestCase):
    @contextmanager
    def _controlled_mode_workers(self):
        app = self._make_app()
        app._streaming = False
        app._vcam_available = True
        app.config.auto_mode = False
        app._dependency_installer = mock.Mock()
        app._dependency_installer.busy = False
        app._dependency_installer.restart_pending.return_value = False
        app._dependency_installer.mode_availability_snapshot.side_effect = self._mode_snapshot
        app._window = self._mode_window()
        app._maybe_warn_weak_device = mock.Mock(return_value=False)
        app._preferred_auto_mode = mock.Mock(return_value="doczeus")
        app._recommended_capture_mode = mock.Mock(return_value=None)
        app.apply_mode_key = mock.Mock(return_value=True)
        app.start_pipeline = mock.Mock(return_value=False)
        workers, callbacks = [], []

        class QueuedThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                workers.append(self.target)

        with (
            mock.patch("nvbroadcast.app.threading.Thread", QueuedThread),
            mock.patch("nvbroadcast.app.GLib.idle_add", side_effect=lambda cb, *args: (
                callbacks.append((cb, args)) or len(callbacks)
            )),
            mock.patch("nvbroadcast.app.save_config"),
            mock.patch("nvbroadcast.core.config.detect_system_capabilities", return_value={
                "cpu_cores": 8, "has_nvidia": True, "gpu_vram_mb": 8192,
            }),
        ):
            yield app, workers, callbacks
        app._mode_probe_scope.close()

    @staticmethod
    def _mode_window():
        return SimpleNamespace(
            _mode_retry_generation=0,
            _mode_retry_in_flight=False,
            _mode_retry_request=None,
            _mode_retry_workers={},
            _resume_mode_runtime_retry=mock.Mock(),
            _mode_availability_ready=False,
            _mode_availability_snapshot=None,
            _cancel_mode_runtime_retry=mock.Mock(),
            rebuild_mode_selector=mock.Mock(),
            set_status=mock.Mock(),
            set_visible=mock.Mock(),
            show_advisory=mock.Mock(),
            _update_gpu_info=mock.Mock(),
        )

    def test_same_gpu_refreshes_coalesce_without_duplicate_provider_work(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            for _ in range(20):
                app._refresh_mode_availability_async(0)
            self.assertEqual(len(workers), 1)
            app._dependency_installer.mode_availability_snapshot.assert_not_called()
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._dependency_installer.mode_availability_snapshot.assert_called_once_with(0)
            app._window.rebuild_mode_selector.assert_called_once()
            self.assertFalse(app._mode_availability_refresh_in_flight)

    def test_availability_waits_for_existing_same_gpu_retry_worker(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app._window._mode_retry_workers[0] = 9
            app._refresh_mode_availability_async(0)
            self.assertFalse(workers)
            self.assertTrue(app._mode_availability_refresh_in_flight)
            app._window._mode_retry_workers.clear()
            app._resume_mode_availability()
            self.assertEqual(len(workers), 1)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._window.rebuild_mode_selector.assert_called_once()

    def test_invalidation_during_same_gpu_probe_runs_fresh_followup(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app._refresh_mode_availability_async(0)
            app._invalidate_mode_availability(0)
            app._refresh_mode_availability_async(0)
            app._refresh_mode_availability_async(0)
            self.assertEqual(len(workers), 1)
            app._window.rebuild_mode_selector.reset_mock()
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._window.rebuild_mode_selector.assert_not_called()
            self.assertEqual(len(workers), 1)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            self.assertEqual(app._dependency_installer.mode_availability_snapshot.call_count, 2)
            self.assertTrue(app._window._mode_availability_ready)

    def test_gpu_round_trip_keeps_one_worker_per_device_and_ignores_old_results(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app._refresh_mode_availability_async(0)
            app.config.compute_gpu = 1
            app._refresh_mode_availability_async(1)
            app.config.compute_gpu = 0
            app._refresh_mode_availability_async(0)
            self.assertEqual(len(workers), 2)
            old_zero, old_one = workers[:]
            workers.clear()
            old_zero()
            callback, args = callbacks.pop(0)
            callback(*args)
            self.assertEqual(len(workers), 1)
            old_one()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._window.rebuild_mode_selector.assert_not_called()
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._window.rebuild_mode_selector.assert_called_once()
            self.assertEqual(app._mode_availability_snapshot["gpu_index"], 0)

    def test_window_replacement_rejects_old_completion_and_refreshes_new_window(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            old_window = app._window
            app._refresh_mode_availability_async(0)
            app._window = self._mode_window()
            app._refresh_mode_availability_async(0)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            old_window.rebuild_mode_selector.assert_not_called()
            app._window.rebuild_mode_selector.assert_not_called()
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._window.rebuild_mode_selector.assert_called_once()

    def test_first_run_setup_owns_checks_until_dismissed(self):
        with self._controlled_mode_workers() as (app, workers, _callbacks):
            app._setup_wizard_open = True
            app._refresh_mode_availability_async(0)
            self.assertFalse(workers)
            app._on_setup_closed()
            self.assertFalse(app._setup_wizard_open)
            self.assertEqual(len(workers), 1)

    def test_runtime_install_defers_provider_work_until_install_completion(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app._dependency_installer.busy = True
            app._refresh_mode_availability_async(0)
            self.assertFalse(workers)
            app._dependency_installer.busy = False
            app._refresh_mode_availability_async(0)
            self.assertEqual(len(workers), 1)

    def test_startup_auto_and_autostart_wait_for_one_completed_snapshot(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app.set_auto_mode_enabled(True)
            app._auto_start()
            self.assertEqual(len(workers), 1)
            app._preferred_auto_mode.assert_not_called()
            app.start_pipeline.assert_not_called()
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._preferred_auto_mode.assert_called_once()
            app.apply_mode_key.assert_called_once()
            app.start_pipeline.assert_called_once()
            self.assertIsNone(app._pending_auto_start_gpu)
            callback(*args)
            app.start_pipeline.assert_called_once()

    def test_disabling_pending_auto_preserves_autostart_but_cancels_mode_intent(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app.set_auto_mode_enabled(True)
            app._auto_start()
            app.set_auto_mode_enabled(False)
            self.assertIsNone(app._pending_auto_mode_gpu)
            self.assertEqual(app._pending_auto_start_gpu, 0)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            self.assertFalse(app.config.auto_mode)
            app._preferred_auto_mode.assert_not_called()
            app.apply_mode_key.assert_not_called()
            app.start_pipeline.assert_called_once()

    def test_capability_failure_keeps_auto_and_live_fps_advisory_off_inline_probes(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app.set_auto_mode_enabled(True)
            app._auto_start()
            with mock.patch("nvbroadcast.core.config.detect_system_capabilities",
                            side_effect=RuntimeError("capability discovery failed")) as detector:
                workers.pop(0)()
                callback, args = callbacks.pop(0)
                callback(*args)
                self.assertFalse(app._window._mode_availability_ready)
                self.assertFalse(app._mode_snapshot_ready())
                app._preferred_auto_mode.assert_not_called()
                app.start_pipeline.assert_not_called()
                app.config.auto_mode = False
                app._streaming = True
                app._perf_monitor.fps = 3.0
                app._lower_recommendation_mode = mock.Mock(side_effect=AssertionError("inline probe"))
                for _ in range(4):
                    self.assertTrue(app._auto_tune_tick())
                app._lower_recommendation_mode.assert_not_called()
                detector.assert_called_once()
            self.assertEqual(app._pending_auto_start_gpu, 0)

    def test_pending_startup_follows_gpu_switch_and_starts_once(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app.set_auto_mode_enabled(True)
            app._auto_start()
            app._sync_gpu_frame_path = mock.Mock()
            with mock.patch("nvbroadcast.core.gpu.detect_gpus", return_value=[]):
                app.set_compute_gpu(1)
            self.assertEqual(app._pending_auto_start_gpu, 1)
            self.assertEqual(len(workers), 2)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app.start_pipeline.assert_not_called()
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app.start_pipeline.assert_called_once()
            self.assertIsNone(app._pending_auto_start_gpu)

    def test_explicit_start_clears_pending_startup_without_provider_work(self):
        with self._controlled_mode_workers() as (app, _workers, _callbacks):
            app._pending_auto_start_gpu = 0
            app._pipeline_teardown = None
            app._clear_finished_teardown = mock.Mock()
            app._restart_after_stop = mock.Mock()
            self.assertFalse(NVBroadcastApp.start_pipeline(app, "/dev/test-camera"))
            self.assertIsNone(app._pending_auto_start_gpu)

    def test_weak_device_advisory_uses_completed_capabilities_without_probing(self):
        with self._controlled_mode_workers() as (app, _workers, _callbacks):
            app.config.first_run = False
            app._resolved_mode_key = mock.Mock(return_value="cpu_quality")
            app._lower_recommendation_mode = mock.Mock(return_value="cpu_low")
            app._recommendation_text = mock.Mock(return_value="Use CPU Low End")
            with mock.patch("nvbroadcast.core.config.detect_system_capabilities") as probe:
                self.assertFalse(NVBroadcastApp._maybe_warn_weak_device(app))
                app._window.show_advisory.assert_not_called()
                app._mode_capabilities_snapshot = (0, {"has_nvidia": False, "cpu_cores": 2})
                self.assertFalse(NVBroadcastApp._maybe_warn_weak_device(app))
                app._window.show_advisory.assert_called_once()
                probe.assert_not_called()

    def test_close_drops_queued_completions_and_rejects_new_refreshes(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app._refresh_mode_availability_async(0)
            workers.pop(0)()
            app.config.minimize_on_close = False
            app._tray = None
            self.assertFalse(app._on_close_request(app._window))
            callback, args = callbacks.pop(0)
            callback(*args)
            app._refresh_mode_availability_async(0)
            app._window.rebuild_mode_selector.assert_not_called()
            self.assertFalse(workers)
            self.assertTrue(app._mode_availability_closed)

    def test_minimize_to_tray_preserves_inflight_refresh(self):
        with self._controlled_mode_workers() as (app, workers, callbacks):
            app._refresh_mode_availability_async(0)
            app.config.minimize_on_close = True
            app._tray = SimpleNamespace(available=True, update_status=mock.Mock())
            self.assertTrue(app._on_close_request(app._window))
            self.assertFalse(app._mode_availability_closed)
            workers.pop(0)()
            callback, args = callbacks.pop(0)
            callback(*args)
            app._window.rebuild_mode_selector.assert_called_once()

    def test_thread_start_failure_reports_error_without_leaving_refresh_stuck(self):
        with self._controlled_mode_workers() as (app, workers, _callbacks):
            with mock.patch("nvbroadcast.app.threading.Thread", side_effect=RuntimeError("no threads")):
                app._refresh_mode_availability_async(0)
            self.assertFalse(app._mode_availability_refresh_in_flight)
            self.assertFalse(app._mode_availability_workers)
            self.assertIn("Could not start", app._window.set_status.call_args.args[0])

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
        app._dependency_installer = SimpleNamespace(set_compute_gpu=mock.Mock(), busy=False)
        app._window = None
        app._mode_availability_generation = 0
        app._mode_availability_request = None
        app._mode_availability_workers = {}
        app._mode_availability_refresh_in_flight = False
        app._mode_availability_closed = False
        app._setup_wizard_open = False
        app._mode_probe_scope = ProviderProbeScope()
        app._pending_auto_start_gpu = None
        app._startup_tray_source_id = 0
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
            _resume_mode_runtime_retry=mock.Mock(),
        )
        app._mode_availability_request = (3, 1, app._window)
        app._mode_availability_workers = {1: 3}

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
        app._dependency_installer.busy = False
        snapshot = self._mode_snapshot(2)
        app._dependency_installer.mode_availability_snapshot.return_value = snapshot
        app._window = SimpleNamespace(
            rebuild_mode_selector=mock.Mock(),
            set_status=mock.Mock(),
            _mode_retry_workers={},
            _resume_mode_runtime_retry=mock.Mock(),
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
            _resume_mode_runtime_retry=mock.Mock(),
        )
        app._mode_availability_request = (6, 1, app._window)
        app._mode_availability_workers = {1: 6}

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

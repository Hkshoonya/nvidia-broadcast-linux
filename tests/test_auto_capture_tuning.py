import unittest
from unittest import mock
from types import SimpleNamespace

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.core.config import AppConfig


class AutoCaptureTuningTests(unittest.TestCase):
    def test_next_lower_capture_mode_steps_down_one_supported_mode(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.camera_device = "/dev/video0"
        app.config.video.width = 1280
        app.config.video.height = 720
        app.config.video.fps = 30

        modes = [
            {"width": 640, "height": 360, "fps": [30]},
            {"width": 800, "height": 600, "fps": [30]},
            {"width": 1280, "height": 720, "fps": [30]},
        ]

        with mock.patch("nvbroadcast.video.virtual_camera.list_camera_modes", return_value=modes):
            self.assertEqual(app._next_lower_capture_mode(), (800, 600, 30))

    def test_get_valid_fps_uses_start_camera_device_override(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.camera_device = "/dev/video0"

        modes = [{"width": 1280, "height": 720, "fps": [24]}]
        with mock.patch("nvbroadcast.video.virtual_camera.list_camera_modes", return_value=modes) as list_modes:
            self.assertEqual(
                app._get_valid_fps(1280, 720, 30, camera_device="/dev/video2"),
                24,
            )

        list_modes.assert_called_once_with("/dev/video2")

    def test_quality_profile_keeps_inline_alpha_even_for_heavy_face_stack(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.performance_profile = "max_quality"
        app.config.use_tensorrt = False
        app._video_effects = SimpleNamespace(enabled=True, mode="replace")
        app._beautifier = SimpleNamespace(enabled=True)
        app._eye_contact = SimpleNamespace(enabled=True)
        app._relighter = SimpleNamespace(enabled=True)
        app._autoframe = SimpleNamespace(enabled=False)

        self.assertTrue(app._compute_inline_inference())

    def test_heavy_face_stack_reuses_landmarks_longer(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app._beautifier = SimpleNamespace(enabled=True)
        app._eye_contact = SimpleNamespace(enabled=True)
        app._relighter = SimpleNamespace(enabled=True)
        app._autoframe = SimpleNamespace(enabled=False)

        self.assertEqual(app._landmark_reuse_frames(), 4)

    def test_eye_contact_only_requests_fresh_landmarks(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app._beautifier = SimpleNamespace(enabled=False)
        app._eye_contact = SimpleNamespace(enabled=True)
        app._relighter = SimpleNamespace(enabled=False)
        app._autoframe = SimpleNamespace(enabled=False)

        self.assertEqual(app._landmark_reuse_frames(), 1)

    def test_performance_profile_uses_async_alpha(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.performance_profile = "performance"
        app.config.compositing = "cpu"
        app.config.use_tensorrt = False
        app.config.use_fused_kernel = False
        app._video_effects = SimpleNamespace(enabled=True, mode="replace")
        app._beautifier = SimpleNamespace(enabled=True)
        app._eye_contact = SimpleNamespace(enabled=True)
        app._relighter = SimpleNamespace(enabled=True)
        app._autoframe = SimpleNamespace(enabled=False)

        self.assertFalse(app._compute_inline_inference())

    def test_cuda_fast_replace_uses_inline_alpha_for_fresh_edges(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.performance_profile = "performance"
        app.config.compositing = "cupy"
        app.config.use_tensorrt = False
        app.config.use_fused_kernel = True
        app._video_effects = SimpleNamespace(enabled=True, mode="replace")

        self.assertTrue(app._compute_inline_inference())

    def test_cuda_fast_blur_keeps_async_alpha_for_throughput(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.performance_profile = "performance"
        app.config.compositing = "cupy"
        app.config.use_tensorrt = False
        app.config.use_fused_kernel = True
        app._video_effects = SimpleNamespace(enabled=True, mode="blur")

        self.assertFalse(app._compute_inline_inference())

    def test_cpu_focus_auto_modes_avoid_gpu_ladder(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.compute_focus = "cpu"
        app._dependency_installer = SimpleNamespace(
            unsupported_reason_for_mode=lambda _mode: None,
            missing_for_mode=lambda _mode: [],
        )

        caps = {"has_nvidia": True, "gpu_vram_mb": 24576, "cpu_cores": 16}
        with mock.patch("nvbroadcast.core.config.detect_system_capabilities", return_value=caps):
            self.assertEqual(
                app._available_auto_modes(),
                ["cpu_quality", "cpu_light", "cpu_low"],
            )
            self.assertEqual(app._preferred_auto_mode(), "cpu_quality")

    def test_gpu_focus_auto_modes_prefer_cuda_ladder(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.compute_focus = "gpu"
        app._dependency_installer = SimpleNamespace(
            unsupported_reason_for_mode=lambda _mode: None,
            missing_for_mode=lambda _mode: [],
        )

        caps = {"has_nvidia": True, "gpu_vram_mb": 4096, "cpu_cores": 16}
        with mock.patch("nvbroadcast.core.config.detect_system_capabilities", return_value=caps):
            self.assertEqual(
                app._available_auto_modes(),
                ["doczeus", "cuda_balanced", "cuda_perf", "cpu_quality", "cpu_light", "cpu_low"],
            )
            self.assertEqual(app._preferred_auto_mode(), "cuda_balanced")

    @mock.patch("nvbroadcast.app.save_config")
    def test_set_compute_focus_applies_preferred_manual_mode(self, _save):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.auto_mode = True
        app._preferred_auto_mode = mock.Mock(return_value="cpu_quality")
        app.apply_mode_key = mock.Mock(return_value=True)

        changed = NVBroadcastApp.set_compute_focus(app, "cpu")

        self.assertTrue(changed)
        self.assertEqual(app.config.compute_focus, "cpu")
        self.assertFalse(app.config.auto_mode)
        app.apply_mode_key.assert_called_once()
        self.assertEqual(app.apply_mode_key.call_args.args[0], "cpu_quality")

    def test_mode_compute_focus_maps_premium_modes_to_stable_ladder(self):
        self.assertEqual(NVBroadcastApp._mode_compute_focus("killer"), "gpu")
        self.assertEqual(NVBroadcastApp._mode_compute_focus("zeus"), "gpu")
        self.assertEqual(NVBroadcastApp._mode_compute_focus("cpu_light"), "cpu")

    @mock.patch("nvbroadcast.app.save_config")
    def test_set_eye_contact_mode_persists_and_normalizes(self, save_config):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app._eye_contact = SimpleNamespace(mode="natural")
        app._restoring = False

        NVBroadcastApp.set_eye_contact_mode(app, "gaze_lock")

        self.assertEqual(app._eye_contact.mode, "gaze_lock")
        self.assertEqual(app.config.video.eye_contact_mode, "gaze_lock")

        NVBroadcastApp.set_eye_contact_mode(app, "broken")

        self.assertEqual(app._eye_contact.mode, "natural")
        self.assertEqual(app.config.video.eye_contact_mode, "natural")
        self.assertEqual(save_config.call_count, 2)

    def test_profile_infer_height_uses_process_scale(self):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.height = 576

        self.assertEqual(app._profile_infer_height("max_quality"), 576)
        self.assertEqual(app._profile_infer_height("performance"), 288)
        self.assertEqual(
            app._profile_infer_height(
                "performance",
                use_tensorrt=False,
                use_fused_kernel=True,
            ),
            480,
        )
        app.config.video.height = 360
        self.assertEqual(
            app._profile_infer_height(
                "performance",
                use_tensorrt=False,
                use_fused_kernel=True,
            ),
            360,
        )

    @mock.patch("nvbroadcast.app.save_config")
    @mock.patch("nvbroadcast.core.config.apply_performance_profile")
    def test_set_performance_profile_applies_profile_infer_height(
        self,
        apply_performance_profile,
        _save,
    ):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.height = 576
        app.config.video.fps = 30
        app._video_effects = SimpleNamespace(
            set_compositing=mock.Mock(),
            set_engine_mode=mock.Mock(),
            set_profile_infer_height=mock.Mock(),
            _apply_edge_config=mock.Mock(),
            # Single-frame backends do not expose RVM's infer-height cap.
            _backend=SimpleNamespace(),
            _skip_interval=1,
        )
        app._beautifier = SimpleNamespace(set_compositing=mock.Mock())
        app._video_pipeline = None
        app._window = SimpleNamespace(set_status=mock.Mock())
        app._refresh_inference_policy = mock.Mock()
        app._sync_gpu_frame_path = mock.Mock()
        app._use_nvdec = False

        NVBroadcastApp.set_performance_profile(
            app,
            "performance",
            compositing="cupy",
            use_tensorrt=False,
            use_fused_kernel=False,
            use_nvdec=False,
            mode_key="cuda_perf",
        )

        apply_performance_profile.assert_called_once_with(app.config, "performance")
        app._video_effects.set_profile_infer_height.assert_not_called()
        app._video_effects.set_engine_mode.assert_called_once_with(
            False,
            False,
            quality=None,
            profile_infer_height=288,
        )
        app._sync_gpu_frame_path.assert_called_once_with()
        app._window.set_status.assert_called_once_with("Mode: Performance | ?p")

    @mock.patch("nvbroadcast.app.save_config")
    def test_apply_mode_key_syncs_named_mode_quality_preset(self, _save):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.quality_preset = "quality"
        app._video_effects = SimpleNamespace(quality="quality")
        app._window = None
        app.set_performance_profile = mock.Mock()

        changed = NVBroadcastApp.apply_mode_key(app, "killer")

        self.assertTrue(changed)
        app.set_performance_profile.assert_called_once()
        self.assertEqual(
            app.set_performance_profile.call_args.kwargs["quality_preset"],
            "performance",
        )
        self.assertEqual(app._video_effects.quality, "performance")
        self.assertEqual(app.config.video.quality_preset, "performance")

    @mock.patch("nvbroadcast.app.save_config")
    def test_cuda_fast_mode_uses_fused_compositor(self, _save):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.quality_preset = "quality"
        app._video_effects = SimpleNamespace(quality="quality")
        app._window = None
        app.set_performance_profile = mock.Mock()

        changed = NVBroadcastApp.apply_mode_key(app, "cuda_perf")

        self.assertTrue(changed)
        _, kwargs = app.set_performance_profile.call_args
        self.assertEqual(kwargs["mode_key"], "cuda_perf")
        self.assertEqual(kwargs["profile_name"] if "profile_name" in kwargs else app.set_performance_profile.call_args.args[0], "performance")
        self.assertTrue(kwargs["use_fused_kernel"])
        self.assertEqual(app.config.video.quality_preset, "performance")

    @mock.patch("nvbroadcast.app.save_config")
    def test_doczeus_mode_uses_ultra_quality_preset(self, _save):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.quality_preset = "quality"
        app._video_effects = SimpleNamespace(quality="quality")
        app._window = None
        app.set_performance_profile = mock.Mock()

        changed = NVBroadcastApp.apply_mode_key(app, "doczeus")

        self.assertTrue(changed)
        app.set_performance_profile.assert_called_once()
        self.assertEqual(
            app.set_performance_profile.call_args.kwargs["quality_preset"],
            "ultra",
        )
        self.assertEqual(app._video_effects.quality, "ultra")
        self.assertEqual(app.config.video.quality_preset, "ultra")

    @mock.patch("nvbroadcast.app.save_config")
    def test_restore_settings_normalizes_stale_named_mode_quality(self, save_config):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.mode_key = "killer"
        app.config.video.quality_preset = "quality"
        app.config.video.eye_contact_mode = "gaze_lock"
        app.config.performance_profile = "performance"
        app.config.compositing = "cupy"

        app._video_effects = SimpleNamespace(
            _model_type="rvm",
            _quality="quality",
            _gpu_index=0,
            enabled=False,
            mode="blur",
            intensity=0.0,
            set_compositing=mock.Mock(),
            set_profile_infer_height=mock.Mock(),
            set_engine_mode=mock.Mock(),
            _apply_edge_config=mock.Mock(),
        )
        app._beautifier = SimpleNamespace(
            enabled=False,
            skin_smooth=0.0,
            denoise=0.0,
            enhance=0.0,
            sharpen=0.0,
            edge_darken=0.0,
            set_compositing=mock.Mock(),
        )
        app._perf_monitor = SimpleNamespace(set_gpu_index=mock.Mock())
        app._window = SimpleNamespace(
            restore_settings=mock.Mock(),
            set_status=mock.Mock(),
        )
        app._eye_contact = SimpleNamespace(
            enabled=False,
            intensity=0.0,
            mode="natural",
        )
        app._relighter = SimpleNamespace(enabled=False, intensity=0.0)
        app._autoframe = SimpleNamespace(enabled=False, zoom_level=1.0)
        app._video_pipeline = None
        app._vcam_available = False
        app._refresh_inference_policy = mock.Mock()
        app._audio_pipeline_should_publish = lambda: False

        NVBroadcastApp._restore_settings(app)

        self.assertEqual(app.config.video.quality_preset, "performance")
        self.assertEqual(app._video_effects._quality, "performance")
        self.assertEqual(app._eye_contact.mode, "gaze_lock")
        save_config.assert_called_once_with(app.config)

    @mock.patch("nvbroadcast.app.save_config")
    def test_setup_complete_syncs_named_mode_quality_preset(self, save_config):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.auto_start = False
        app._video_effects = SimpleNamespace(
            _gpu_index=0,
            _quality="quality",
            _skip_interval=1,
            _apply_edge_config=mock.Mock(),
            set_compositing=mock.Mock(),
            set_profile_infer_height=mock.Mock(),
            set_engine_mode=mock.Mock(),
        )
        app._beautifier = SimpleNamespace(set_compositing=mock.Mock())
        app._window = SimpleNamespace(
            rebuild_mode_selector=mock.Mock(),
            _sync_quality_selector=mock.Mock(),
            _gpu_selector=None,
            _edge_dilate=SimpleNamespace(_scale=SimpleNamespace(set_value=mock.Mock())),
            _edge_blur=SimpleNamespace(_scale=SimpleNamespace(set_value=mock.Mock())),
            _edge_strength=SimpleNamespace(_scale=SimpleNamespace(set_value=mock.Mock())),
            _edge_midpoint=SimpleNamespace(_scale=SimpleNamespace(set_value=mock.Mock())),
            set_status=mock.Mock(),
        )

        NVBroadcastApp._on_setup_complete(app, None, "performance", 0, "cupy")

        self.assertEqual(app.config.mode_key, "cuda_perf")
        self.assertEqual(app.config.video.quality_preset, "performance")
        self.assertEqual(app._video_effects._quality, "performance")
        app._window._sync_quality_selector.assert_called_once_with()
        save_config.assert_called_once_with(app.config)


class AutoStartBroadcastStateTests(unittest.TestCase):
    """_auto_start must respect the profile's saved broadcast state."""

    def _make_app(self, broadcast_started):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app.config = AppConfig()
        app.config.video.camera_device = "/dev/video0"
        app.config.video.output_format = "YUY2"
        app.config.broadcast_started = broadcast_started
        app._streaming = False
        app._vcam_available = True
        app.start_pipeline = mock.Mock()
        app._window = SimpleNamespace(
            _streaming=False,
            set_status=mock.Mock(),
            _stream_btn=SimpleNamespace(
                set_label=mock.Mock(),
                remove_css_class=mock.Mock(),
                add_css_class=mock.Mock(),
            ),
        )
        return app

    def test_auto_start_skipped_when_profile_saved_stopped(self):
        app = self._make_app(broadcast_started=False)

        NVBroadcastApp._auto_start(app)

        app.start_pipeline.assert_not_called()
        app._window.set_status.assert_called_once()
        self.assertFalse(app._window._streaming)

    def test_auto_start_runs_when_profile_saved_running(self):
        app = self._make_app(broadcast_started=True)

        NVBroadcastApp._auto_start(app)

        app.start_pipeline.assert_called_once_with("/dev/video0", "YUY2")
        self.assertTrue(app._window._streaming)

    def test_auto_start_preserves_legacy_behavior_when_state_unknown(self):
        # Legacy configs have no broadcast_started key -> None; they keep the
        # previous always-start behavior.
        app = self._make_app(broadcast_started=None)

        NVBroadcastApp._auto_start(app)

        app.start_pipeline.assert_called_once_with("/dev/video0", "YUY2")
        self.assertTrue(app._window._streaming)


if __name__ == "__main__":
    unittest.main()

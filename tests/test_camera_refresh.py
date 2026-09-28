"""Regression coverage for refreshing cameras after application startup."""

from types import SimpleNamespace
import unittest
from unittest import mock

from nvbroadcast.core.config import AppConfig
from nvbroadcast.ui.device_selector import DeviceSelector
from nvbroadcast.ui.window import (
    NVBroadcastWindow,
    _CAMERA_RETRY_STATUS,
    _CameraCapabilitySnapshot,
    _CameraModeSnapshot,
    _CameraRefreshSnapshot,
)


CAMERA_ZERO = {"name": "Integrated Camera", "device": "/dev/video0"}
CAMERA_TWO = {"name": "USB Camera", "device": "/dev/video2"}


def _camera_snapshot(*cameras, requested_mode=(1280, 720, 30)):
    capabilities = tuple(
        _CameraCapabilitySnapshot(
            camera["device"],
            (_CameraModeSnapshot(1280, 720, (30,)),),
            requested_mode,
        )
        for camera in cameras
    )
    return _CameraRefreshSnapshot(
        tuple((camera["name"], camera["device"]) for camera in cameras),
        capabilities,
        requested_mode,
    )


class _DeferredThread:
    """Thread double whose target runs only when the test requests it."""

    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True

    def run(self):
        self.target()


class _FakeCameraSelector:
    """Small stateful selector double matching the refresh-facing API."""

    def __init__(self, devices=None, selected=""):
        self._devices = list(devices or [])
        self._selected = selected
        self.set_devices_calls = []
        self.set_selected_device_calls = []

    def get_selected_device(self):
        return self._selected

    def set_devices(self, devices):
        devices = list(devices)
        self.set_devices_calls.append(devices)
        if devices == self._devices:
            return False

        selected = self._selected
        self._devices = devices
        available = {device["device"] for device in devices}
        self._selected = (
            selected
            if selected in available
            else devices[0]["device"] if devices else ""
        )
        return True

    def set_selected_device(self, device):
        self.set_selected_device_calls.append(device)
        if any(candidate["device"] == device for candidate in self._devices):
            self._selected = device
            return True
        return False

    def set_selected_index(self, index):
        if 0 <= index < len(self._devices):
            self._selected = self._devices[index]["device"]


class CameraRefreshTests(unittest.TestCase):
    def make_window(self, devices=None, selected="", status="Ready"):
        window = NVBroadcastWindow.__new__(NVBroadcastWindow)
        config = AppConfig()
        config.video.camera_device = "/dev/video0"
        window._app = SimpleNamespace(config=config)
        window._camera_selector = _FakeCameraSelector(devices, selected)
        window._camera_refresh_source_id = 0
        window._camera_refresh_generation = 1
        window._camera_refresh_in_flight = False
        window._camera_refresh_pending_reason = None
        window._camera_refresh_mapped = True
        window._camera_refresh_shutdown = False
        window._status_bar = SimpleNamespace(
            get_text=mock.Mock(return_value=status)
        )
        window.sync_video_input_controls = mock.Mock()
        window.set_status = mock.Mock()
        return window

    def test_apply_discovers_late_camera_and_syncs_capabilities(self):
        window = self.make_window()

        self.assertTrue(window._apply_camera_devices(_camera_snapshot(CAMERA_ZERO)))

        self.assertEqual(window._camera_selector._devices, [CAMERA_ZERO])
        self.assertEqual(
            window._camera_selector.get_selected_device(),
            "/dev/video0",
        )
        window.sync_video_input_controls.assert_called_once_with(
            window._app.config,
            camera_device="/dev/video0",
            camera_modes=(_CameraModeSnapshot(1280, 720, (30,)),),
            supported_mode=(1280, 720, 30),
        )

    def test_empty_result_clears_stale_camera_entries(self):
        window = self.make_window([CAMERA_ZERO], selected="/dev/video0")

        self.assertFalse(window._apply_camera_devices(_camera_snapshot()))

        self.assertEqual(window._camera_selector._devices, [])
        self.assertEqual(window._camera_selector.get_selected_device(), "")
        window.sync_video_input_controls.assert_not_called()

    def test_map_discovery_clears_stale_no_camera_status(self):
        window = self.make_window(status=_CAMERA_RETRY_STATUS)

        self.assertFalse(
            window._finish_camera_refresh(
                window._camera_refresh_generation,
                "map",
                _camera_snapshot(CAMERA_ZERO),
            )
        )

        window.set_status.assert_called_once_with(
            "Camera detected. Source list refreshed."
        )

    def test_initial_map_preserves_current_status(self):
        for status in ("Ready", "Streaming: /dev/video0 -> /dev/video10"):
            with self.subTest(status=status):
                window = self.make_window(status=status)

                self.assertFalse(
                    window._finish_camera_refresh(
                        window._camera_refresh_generation,
                        "map",
                        _camera_snapshot(CAMERA_ZERO),
                    )
                )

                window.set_status.assert_not_called()

    def test_map_with_existing_camera_preserves_current_status(self):
        window = self.make_window([CAMERA_ZERO], selected="/dev/video0")

        self.assertFalse(
            window._finish_camera_refresh(
                window._camera_refresh_generation,
                "map",
                _camera_snapshot(CAMERA_ZERO),
            )
        )

        window.set_status.assert_not_called()

    def test_apply_preserves_selected_camera_when_order_changes(self):
        window = self.make_window(
            [CAMERA_ZERO, CAMERA_TWO],
            selected="/dev/video2",
        )

        self.assertTrue(
            window._apply_camera_devices(
                _camera_snapshot(CAMERA_TWO, CAMERA_ZERO)
            )
        )

        self.assertEqual(
            window._camera_selector.get_selected_device(),
            "/dev/video2",
        )
        window.sync_video_input_controls.assert_called_once_with(
            window._app.config,
            camera_device="/dev/video2",
            camera_modes=(_CameraModeSnapshot(1280, 720, (30,)),),
            supported_mode=(1280, 720, 30),
        )

    @mock.patch("nvbroadcast.ui.window.GLib.idle_add")
    @mock.patch(
        "nvbroadcast.ui.window._probe_camera_refresh",
        return_value=_camera_snapshot(CAMERA_ZERO),
    )
    @mock.patch("nvbroadcast.ui.window.threading.Thread")
    def test_overlapping_refreshes_are_serialized(
        self,
        thread_class,
        probe_camera_refresh,
        idle_add,
    ):
        window = self.make_window()
        workers = []
        idle_callbacks = []

        def build_thread(**kwargs):
            thread = _DeferredThread(**kwargs)
            workers.append(thread)
            return thread

        thread_class.side_effect = build_thread
        idle_add.side_effect = lambda callback, *args: idle_callbacks.append(
            (callback, args)
        ) or 91

        self.assertTrue(window._request_camera_refresh("map"))
        self.assertFalse(window._request_camera_refresh("manual"))
        self.assertEqual(len(workers), 1)
        self.assertTrue(workers[0].started)
        self.assertEqual(window._camera_refresh_pending_reason, "manual")

        workers[0].run()

        probe_camera_refresh.assert_called_once_with((1280, 720, 30))
        self.assertEqual(len(idle_callbacks), 1)
        self.assertEqual(window._camera_selector._devices, [])
        self.assertEqual(len(workers), 1)

        callback, args = idle_callbacks.pop()
        self.assertFalse(callback(*args))

        self.assertEqual(window._camera_selector._devices, [CAMERA_ZERO])
        self.assertEqual(len(workers), 2)
        self.assertTrue(workers[1].started)
        self.assertTrue(window._camera_refresh_in_flight)

    def test_changed_config_discards_snapshot_and_queues_fresh_probe(self):
        window = self.make_window()
        window._camera_refresh_in_flight = True
        window._apply_camera_devices = mock.Mock()
        window._request_camera_refresh = mock.Mock(return_value=True)
        snapshot = _camera_snapshot(CAMERA_ZERO)
        window._app.config.video.width = 640
        window._app.config.video.height = 480
        window._app.config.video.fps = 60

        self.assertFalse(
            window._finish_camera_refresh(
                window._camera_refresh_generation,
                "map",
                snapshot,
            )
        )

        window._apply_camera_devices.assert_not_called()
        window._request_camera_refresh.assert_called_once_with("map")

    @mock.patch("nvbroadcast.ui.window.GLib.idle_add")
    @mock.patch("nvbroadcast.ui.window.select_camera_mode")
    @mock.patch("nvbroadcast.ui.window.list_camera_modes")
    @mock.patch("nvbroadcast.ui.window.list_camera_devices")
    @mock.patch("nvbroadcast.ui.window.clear_camera_probe_cache")
    @mock.patch("nvbroadcast.ui.window.threading.Thread")
    def test_worker_snapshot_avoids_main_thread_probes_beyond_cache_size(
        self,
        thread_class,
        clear_camera_probe_cache,
        list_camera_devices,
        list_camera_modes,
        select_camera_mode,
        idle_add,
    ):
        cameras = [
            {"name": f"Camera {index}", "device": f"/dev/video{index}"}
            for index in range(9)
        ]
        list_camera_devices.return_value = cameras
        list_camera_modes.return_value = [
            {"width": 640, "height": 480, "fps": [60]},
        ]
        select_camera_mode.return_value = {
            "format": "raw",
            "width": 640,
            "height": 480,
            "fps": 60,
        }
        window = self.make_window()
        window._res_selector = _FakeCameraSelector()
        window._fps_selector = _FakeCameraSelector()
        window._format_selector = mock.Mock()
        window._vcam_entry = None
        window._updating_ui = False
        window.sync_video_input_controls = (
            NVBroadcastWindow.sync_video_input_controls.__get__(
                window,
                NVBroadcastWindow,
            )
        )
        workers = []
        idle_callbacks = []
        thread_class.side_effect = lambda **kwargs: workers.append(
            _DeferredThread(**kwargs)
        ) or workers[-1]
        idle_add.side_effect = lambda callback, *args: idle_callbacks.append(
            (callback, args)
        ) or 91

        window._request_camera_refresh("map")
        workers[0].run()

        clear_camera_probe_cache.assert_called_once_with()
        list_camera_devices.assert_called_once_with()
        self.assertEqual(list_camera_modes.call_count, 9)
        self.assertEqual(select_camera_mode.call_count, 9)
        self.assertEqual(len(idle_callbacks), 1)

        # Simulate an evicted/cold cache. GTK reconciliation must use only the
        # immutable worker snapshot and must not invoke either probe function.
        list_camera_modes.side_effect = AssertionError("GTK reprobed camera modes")
        select_camera_mode.side_effect = AssertionError("GTK reselected camera mode")
        callback, args = idle_callbacks.pop()
        self.assertFalse(callback(*args))

        self.assertEqual(window._camera_selector.get_selected_device(), "/dev/video0")
        self.assertEqual(window._res_selector.get_selected_device(), "640x480")
        self.assertEqual(window._fps_selector.get_selected_device(), "60")
        self.assertEqual(
            (
                window._app.config.video.width,
                window._app.config.video.height,
                window._app.config.video.fps,
            ),
            (1280, 720, 30),
        )
        self.assertEqual(list_camera_modes.call_count, 9)
        self.assertEqual(select_camera_mode.call_count, 9)

    @mock.patch("nvbroadcast.ui.window.GLib.idle_add")
    @mock.patch(
        "nvbroadcast.ui.window._probe_camera_refresh",
        return_value=_camera_snapshot(CAMERA_ZERO),
    )
    @mock.patch("nvbroadcast.ui.window.threading.Thread")
    def test_unmap_invalidates_late_worker_result(
        self,
        thread_class,
        _probe_camera_refresh,
        idle_add,
    ):
        window = self.make_window()
        workers = []
        idle_callbacks = []
        thread_class.side_effect = lambda **kwargs: workers.append(
            _DeferredThread(**kwargs)
        ) or workers[-1]
        idle_add.side_effect = lambda callback, *args: idle_callbacks.append(
            (callback, args)
        ) or 91

        window._request_camera_refresh("map")
        workers[0].run()
        generation = window._camera_refresh_generation

        window._on_camera_refresh_unmapped()
        self.assertGreater(window._camera_refresh_generation, generation)

        callback, args = idle_callbacks.pop()
        self.assertFalse(callback(*args))

        self.assertEqual(window._camera_selector._devices, [])
        window.sync_video_input_controls.assert_not_called()
        window.set_status.assert_not_called()
        self.assertFalse(window._camera_refresh_in_flight)
        self.assertEqual(len(workers), 1)

    @mock.patch("nvbroadcast.ui.window.GLib.idle_add")
    @mock.patch(
        "nvbroadcast.ui.window._probe_camera_refresh",
        return_value=_camera_snapshot(CAMERA_ZERO),
    )
    @mock.patch("nvbroadcast.ui.window.threading.Thread")
    def test_shutdown_permanently_invalidates_late_worker_result(
        self,
        thread_class,
        _probe_camera_refresh,
        idle_add,
    ):
        window = self.make_window()
        workers = []
        idle_callbacks = []
        thread_class.side_effect = lambda **kwargs: workers.append(
            _DeferredThread(**kwargs)
        ) or workers[-1]
        idle_add.side_effect = lambda callback, *args: idle_callbacks.append(
            (callback, args)
        ) or 91

        window._request_camera_refresh("map")
        workers[0].run()
        window.stop_camera_refresh(shutdown=True)

        callback, args = idle_callbacks.pop()
        self.assertFalse(callback(*args))

        self.assertTrue(window._camera_refresh_shutdown)
        self.assertEqual(window._camera_selector._devices, [])
        self.assertFalse(window._request_camera_refresh("manual"))
        self.assertEqual(len(workers), 1)

    @mock.patch("nvbroadcast.ui.window.GLib.timeout_add_seconds", return_value=73)
    def test_schedule_registers_one_one_shot_retry_source(self, timeout_add_seconds):
        window = self.make_window()
        window._request_camera_refresh = mock.Mock(return_value=True)

        window._schedule_camera_refresh()
        window._schedule_camera_refresh()

        timeout_add_seconds.assert_called_once()
        seconds, callback = timeout_add_seconds.call_args.args
        self.assertEqual(seconds, 3)
        self.assertEqual(window._camera_refresh_source_id, 73)
        self.assertFalse(callback())
        self.assertEqual(window._camera_refresh_source_id, 0)
        window._request_camera_refresh.assert_called_once_with("retry")

    @mock.patch("nvbroadcast.ui.window.GLib.source_remove")
    def test_stop_invalidates_generation_and_removes_retry_source(self, source_remove):
        window = self.make_window()
        window._camera_refresh_source_id = 73
        generation = window._camera_refresh_generation

        window.stop_camera_refresh()
        window.stop_camera_refresh()

        source_remove.assert_called_once_with(73)
        self.assertEqual(window._camera_refresh_source_id, 0)
        self.assertEqual(window._camera_refresh_generation, generation + 2)

    def test_manual_refresh_queues_worker_without_inline_enumeration(self):
        window = self.make_window()
        window._cancel_camera_refresh_timer = mock.Mock()
        window._request_camera_refresh = mock.Mock(return_value=True)

        window._on_camera_refresh_clicked(None)

        window._cancel_camera_refresh_timer.assert_called_once_with()
        window._request_camera_refresh.assert_called_once_with("manual")
        window.set_status.assert_called_once_with("Refreshing camera list...")

    @mock.patch("nvbroadcast.ui.window.select_camera_mode")
    @mock.patch("nvbroadcast.ui.window.list_camera_modes")
    def test_unsupported_saved_mode_uses_consistent_supported_controls(
        self,
        list_camera_modes,
        select_camera_mode,
    ):
        window = self.make_window([CAMERA_TWO], selected="/dev/video2")
        window._res_selector = _FakeCameraSelector()
        window._fps_selector = _FakeCameraSelector()
        window._format_selector = mock.Mock()
        window._vcam_entry = None
        window._updating_ui = False
        window.sync_video_input_controls = (
            NVBroadcastWindow.sync_video_input_controls.__get__(
                window,
                NVBroadcastWindow,
            )
        )
        window._app.switch_camera = mock.Mock()
        config = window._app.config
        config.video.width = 1280
        config.video.height = 720
        config.video.fps = 30
        list_camera_modes.return_value = [
            {"width": 640, "height": 480, "fps": [60]},
        ]
        select_camera_mode.return_value = {
            "format": "raw",
            "width": 640,
            "height": 480,
            "fps": 60,
        }

        window.sync_video_input_controls(config, camera_device="/dev/video2")

        list_camera_modes.assert_called_once_with("/dev/video2")
        select_camera_mode.assert_called_once_with(
            "/dev/video2",
            1280,
            720,
            30,
        )
        self.assertEqual(window._res_selector.get_selected_device(), "640x480")
        self.assertEqual(window._fps_selector.get_selected_device(), "60")
        self.assertEqual(
            window._fps_selector._devices,
            [{"name": "60 fps", "device": "60"}],
        )
        self.assertEqual(
            (config.video.width, config.video.height, config.video.fps),
            (1280, 720, 30),
        )
        window._app.switch_camera.assert_not_called()


class DeviceSelectorRefreshTests(unittest.TestCase):
    def test_identical_devices_reuse_existing_dropdown_model(self):
        devices = [CAMERA_ZERO, CAMERA_TWO]
        selector = SimpleNamespace(
            _devices=devices,
            _dropdown=mock.Mock(),
            _handler_id=17,
            get_selected_device=mock.Mock(return_value="/dev/video2"),
            _update_tooltip=mock.Mock(),
        )

        self.assertFalse(DeviceSelector.set_devices(selector, list(devices)))

        selector._dropdown.set_model.assert_not_called()
        selector._dropdown.set_selected.assert_not_called()
        selector._update_tooltip.assert_not_called()

    @mock.patch("nvbroadcast.ui.device_selector.Gtk.StringList.new")
    def test_changed_devices_preserve_selection_by_stable_identifier(
        self, new_string_list
    ):
        model = object()
        new_string_list.return_value = model
        selector = SimpleNamespace(
            _devices=[CAMERA_ZERO, CAMERA_TWO],
            _dropdown=mock.Mock(),
            _handler_id=17,
            get_selected_device=mock.Mock(return_value="/dev/video2"),
            _update_tooltip=mock.Mock(),
        )

        changed = DeviceSelector.set_devices(
            selector,
            [CAMERA_TWO, CAMERA_ZERO],
        )

        self.assertTrue(changed)
        new_string_list.assert_called_once_with(
            ["USB Camera", "Integrated Camera"]
        )
        selector._dropdown.set_model.assert_called_once_with(model)
        selector._dropdown.set_selected.assert_called_once_with(0)
        selector._dropdown.handler_block.assert_called_once_with(17)
        selector._dropdown.handler_unblock.assert_called_once_with(17)


if __name__ == "__main__":
    unittest.main()

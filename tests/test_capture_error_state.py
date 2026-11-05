"""Capture failures must stop transport and leave a usable Start control.

Extract real lifecycle methods with mocked native/UI boundaries so these
regressions never open a camera, initialize a model, or require a display.
"""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def selected_class(relative, class_name, methods, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    source = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == class_name)
    body = [node for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", level=0,
                       names=[ast.alias(name="annotations")]),
        ast.ClassDef(name=class_name, bases=[], keywords=[], body=body,
                     decorator_list=[]),
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    return namespace[class_name]


class CaptureErrorStateTests(unittest.TestCase):
    def setUp(self):
        self.glib = SimpleNamespace(
            timeout_add=mock.Mock(return_value=91),
            idle_add=mock.Mock(return_value=92),
            source_remove=mock.Mock(), PRIORITY_HIGH=-100,
        )
        gst = SimpleNamespace(init=mock.Mock(), State=SimpleNamespace(NULL="NULL"))
        namespace = dict(
            threading=threading, Gst=gst, GLib=self.glib,
            DEFAULT_WIDTH=1280, DEFAULT_HEIGHT=720, DEFAULT_FPS=30,
            VIRTUAL_CAM_DEVICE="/dev/video10",
        )
        self.Pipeline = selected_class(
            "src/nvbroadcast/video/pipeline.py", "VideoPipeline",
            {"__init__", "_on_error", "_fail_capture",
             "set_capture_error_callback", "_stop_recording_after_capture_error",
             "_queue_rebuild", "_cancel_rebuild", "_rebuild_pipeline",
             "stop", "_stop_preview_timer", "_poll_teardown", "is_recording"}, namespace,
        )
        self.App = selected_class(
            "src/nvbroadcast/app.py", "NVBroadcastApp",
            {"_on_capture_error", "stop_pipeline", "_sync_tray_status",
             "start_pipeline", "_queue_pipeline_restart", "_restart_after_stop",
             "_clear_finished_teardown", "_on_recording_finalized", "is_recording"},
            {"GLib": self.glib},
        )
        self.Window = selected_class(
            "src/nvbroadcast/ui/window.py", "NVBroadcastWindow",
            {"_on_stream_toggle"}, {},
        )

    def pipeline(self):
        pipeline = self.Pipeline()
        pipeline._running = True
        pipeline._pipeline = mock.Mock()
        pipeline._pipeline.get_by_name.return_value = None
        pipeline._close_macos_virtual_camera = mock.Mock()
        pipeline._capture_error_callback = mock.Mock()
        return pipeline

    @staticmethod
    def error(text="generated camera disconnected", source=None):
        return SimpleNamespace(
            src=source,
            parse_error=mock.Mock(return_value=(SimpleNamespace(message=text), "")),
        )

    def app(self, pipeline):
        app = self.App()
        app._video_pipeline = pipeline
        app._pipeline_teardown = None
        app._streaming = True
        app._pending_auto_start_gpu = None
        app._pending_start = None
        app._idle_active = False
        app._idle_strikes = 0
        app._restart_source_id = 0
        app._eye_contact = mock.Mock()
        app._tray = mock.Mock()
        app._meeting_active = False
        app._meeting_finalizing = False
        window = self.Window()
        window._app = app
        window._streaming = True
        window._stream_btn = mock.Mock()
        window._format_selector = SimpleNamespace(get_selected_device=lambda: "YUY2")
        window._camera_selector = SimpleNamespace(get_selected_device=lambda: "/dev/video2")
        window.set_status = mock.Mock()
        window.on_recording_finalized = mock.Mock()
        app._window = window
        return app

    def test_terminal_error_stops_capture_without_an_active_recording(self):
        pipeline = self.pipeline()
        capture = pipeline._pipeline
        callback = pipeline._capture_error_callback

        pipeline._on_error(None, self.error(), generation=0)

        self.assertFalse(pipeline._running)
        self.assertIsNone(pipeline._pipeline)
        self.assertIs(pipeline._teardown_capture, capture)
        self.assertFalse(pipeline._teardown_done)
        callback.assert_called_once_with(pipeline, 0, "generated camera disconnected")
        capture.set_state.assert_not_called()
        self.assertFalse(pipeline._poll_teardown())
        capture.set_state.assert_called_once_with("NULL")
        self.assertTrue(pipeline._teardown_done)

    def test_duplicate_terminal_messages_notify_only_once(self):
        pipeline = self.pipeline()
        pipeline._on_error(None, self.error(), generation=0)
        pipeline._on_error(None, self.error("second message"), generation=0)
        pipeline._capture_error_callback.assert_called_once_with(
            pipeline, 0, "generated camera disconnected",
        )
        self.assertEqual(self.glib.timeout_add.call_count, 1)

    def test_retired_generation_error_does_not_stop_current_capture(self):
        pipeline = self.pipeline()
        pipeline._capture_generation = 3
        message = self.error()
        pipeline._on_error(None, message, generation=2)
        message.parse_error.assert_not_called()
        self.assertTrue(pipeline._running)
        pipeline._capture_error_callback.assert_not_called()

    def test_advertised_format_retry_remains_recoverable(self):
        pipeline = self.pipeline()
        source = mock.Mock()
        pipeline._pipeline.get_by_name.return_value = source
        pipeline._capture_candidates = [{"format": "mjpeg"}, {"format": "raw"}]

        pipeline._on_error(None, self.error(source=source), generation=0)

        self.assertTrue(pipeline._running)
        self.assertTrue(pipeline._capture_retry_pending)
        self.assertTrue(pipeline._rebuild_pending)
        self.assertEqual(pipeline._capture_candidate_index, 1)
        pipeline._capture_error_callback.assert_not_called()

    def test_secondary_error_while_retry_is_pending_is_ignored(self):
        pipeline = self.pipeline()
        pipeline._capture_retry_pending = True
        pipeline._on_error(None, self.error(), generation=0)
        self.assertTrue(pipeline._running)
        pipeline._capture_error_callback.assert_not_called()
        self.glib.timeout_add.assert_not_called()

    def test_late_gpu_recovery_idle_cannot_reopen_terminal_capture(self):
        pipeline = self.pipeline()
        pipeline._gpu_capture_active = True
        pipeline._on_error(None, self.error("recoverable transport error"), generation=0)
        self.assertTrue(pipeline._running)
        self.assertTrue(pipeline._gpu_path_demoted)
        pipeline._capture_error_callback.assert_not_called()
        recovery = self.glib.idle_add.call_args.args[0]

        pipeline._on_error(None, self.error(), generation=0)
        timeouts_after_stop = self.glib.timeout_add.call_count
        self.assertFalse(recovery())

        self.assertFalse(pipeline._running)
        self.assertFalse(pipeline._rebuild_pending)
        self.assertIsNone(pipeline._pipeline)
        self.assertEqual(self.glib.timeout_add.call_count, timeouts_after_stop)

    def test_failed_rebuild_stops_and_reports_the_capture(self):
        pipeline = self.pipeline()
        pipeline._pipeline = None
        pipeline._rebuild_pending = True
        pipeline.build = mock.Mock(side_effect=RuntimeError("generated rebuild failure"))
        with self.assertRaisesRegex(RuntimeError, "generated rebuild failure"):
            pipeline._rebuild_pipeline()
        self.assertFalse(pipeline._running)
        self.assertFalse(pipeline._rebuild_pending)
        pipeline._capture_error_callback.assert_called_once_with(
            pipeline, 0, "generated rebuild failure",
        )

    def test_user_stop_prevents_late_gpu_idle_recovery(self):
        pipeline = self.pipeline()
        pipeline._gpu_capture_active = True
        pipeline._on_error(None, self.error("recoverable transport error"), generation=0)
        recovery = self.glib.idle_add.call_args.args[0]
        pipeline.stop()
        timeouts_after_stop = self.glib.timeout_add.call_count

        self.assertFalse(recovery())

        self.assertFalse(pipeline._running)
        self.assertFalse(pipeline._rebuild_pending)
        self.assertIsNone(pipeline._pipeline)
        self.assertEqual(self.glib.timeout_add.call_count, timeouts_after_stop)
        pipeline._capture_error_callback.assert_not_called()

    def test_pending_internal_rebuild_can_resume_during_teardown(self):
        pipeline = self.pipeline()
        pipeline._rebuild_pending = True
        pipeline.stop(clear_rebuild_request=False)
        self.assertFalse(pipeline._running)
        self.assertFalse(pipeline._teardown_done)
        timeouts_before_retry = self.glib.timeout_add.call_count

        pipeline._queue_rebuild()

        self.assertTrue(pipeline._rebuild_pending)
        self.assertEqual(pipeline._rebuild_source_id, 91)
        self.assertEqual(self.glib.timeout_add.call_count, timeouts_before_retry + 1)
        self.glib.timeout_add.assert_called_with(
            10, pipeline._rebuild_pipeline, priority=self.glib.PRIORITY_HIGH,
        )

    def test_terminal_failure_retains_incomplete_recording_result(self):
        pipeline = self.pipeline()
        pipeline._recording = True
        callback = mock.Mock()
        pipeline._recording_finalize_callback = callback
        pipeline._recording_error_callback = mock.Mock()
        completions = []

        def finalize(**_kwargs):
            completions.append(pipeline._recording_finalize_callback)
            pipeline._recording = False

        pipeline.stop_recording = mock.Mock(side_effect=finalize)
        pipeline._on_error(None, self.error(), generation=0)
        pipeline.stop_recording.assert_called_once_with(wait=False)
        pipeline._recording_error_callback.assert_called_once_with(
            "generated camera disconnected",
        )
        self.assertIs(pipeline._recording_finalize_callback, callback)
        completions[0](True, "")
        callback.assert_called_once_with(False, "generated camera disconnected")

    def test_recording_error_callback_failure_still_stops_and_notifies(self):
        pipeline = self.pipeline()
        pipeline._recording = True
        pipeline._recording_error_callback = mock.Mock(side_effect=RuntimeError("generated UI error"))
        pipeline.stop_recording = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "generated UI error"):
            pipeline._on_error(None, self.error(), generation=0)
        self.assertFalse(pipeline._running)
        pipeline._capture_error_callback.assert_called_once()

    def test_finalization_cannot_stop_a_replacement_generation(self):
        pipeline = self.pipeline()

        def replacement(_message):
            pipeline._capture_generation += 1

        pipeline._stop_recording_after_capture_error = replacement
        pipeline._on_error(None, self.error(), generation=0)
        self.assertTrue(pipeline._running)
        self.assertEqual(pipeline._capture_generation, 1)
        pipeline._capture_error_callback.assert_not_called()
        self.glib.timeout_add.assert_not_called()

    def test_app_ignores_retired_pipeline_and_generation_notifications(self):
        retired = self.pipeline()
        active = self.pipeline()
        active._capture_generation = 2
        app = self.app(active)
        app._on_capture_error(retired, 0, "retired capture")
        app._on_capture_error(active, 1, "retired generation")
        self.assertIs(app._video_pipeline, active)
        self.assertTrue(app._streaming)
        self.assertTrue(active._running)
        app._window.set_status.assert_not_called()

    def test_error_restores_start_then_waits_for_teardown_before_retry(self):
        pipeline = self.pipeline()
        app = self.app(pipeline)
        pipeline.set_capture_error_callback(app._on_capture_error)
        pipeline._on_error(None, self.error(), generation=0)
        self.assertFalse(app._streaming)
        self.assertFalse(app._window._streaming)
        self.assertIsNone(app._video_pipeline)
        self.assertIs(app._pipeline_teardown, pipeline)
        app._window._stream_btn.set_label.assert_called_with("Start Broadcast")
        self.assertIn("Camera stopped:", app._window.set_status.call_args.args[0])
        self.assertFalse(app._tray.update_status.call_args.args[0])

        def started(*_args):
            app._video_pipeline = self.pipeline()
            app._streaming = True

        app._do_start_pipeline = mock.Mock(side_effect=started)
        app._window._on_stream_toggle(app._window._stream_btn)
        app._do_start_pipeline.assert_not_called()
        self.assertFalse(app._window._streaming)
        self.assertTrue(app._restart_after_stop())
        self.assertFalse(pipeline._poll_teardown())
        self.assertFalse(app._restart_after_stop())
        app._do_start_pipeline.assert_called_once_with("/dev/video2", "YUY2")
        self.assertTrue(app._streaming)
        self.assertTrue(app._window._streaming)
        app._window._stream_btn.set_label.assert_called_with("Stop Broadcast")

    def test_late_recording_completion_does_not_stop_replacement_pipeline(self):
        pipeline = self.pipeline()
        app = self.app(pipeline)
        pipeline.set_capture_error_callback(app._on_capture_error)
        pipeline._recording = True
        pipeline._recording_error_callback = mock.Mock()
        pipeline._recording_finalize_callback = lambda success, error: app._on_recording_finalized(
            "rec", success, error,
        )
        completions = []

        def finalize(**_kwargs):
            completions.append(pipeline._recording_finalize_callback)
            pipeline._recording = False

        pipeline.stop_recording = mock.Mock(side_effect=finalize)
        pipeline._on_error(None, self.error(), generation=0)
        replacement = self.pipeline()
        app._video_pipeline = replacement
        app._streaming = True
        completions[0](True, "")
        self.assertIs(app._video_pipeline, replacement)
        self.assertTrue(app._streaming)
        self.assertTrue(replacement._running)
        app._window.on_recording_finalized.assert_called_once_with(
            False, "generated camera disconnected",
        )


if __name__ == "__main__":
    unittest.main()

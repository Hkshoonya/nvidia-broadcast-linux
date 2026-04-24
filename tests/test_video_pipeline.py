import unittest
from unittest import mock

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from nvbroadcast.video.pipeline import VideoPipeline


class VideoPipelineRebuildTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

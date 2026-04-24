import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()

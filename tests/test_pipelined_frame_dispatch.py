"""App-level dispatch of pipelined inline inference without devices or models."""

from types import SimpleNamespace
import unittest
from unittest import mock

from nvbroadcast import app as app_module
from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.video.inference_stage import PipelinedInferenceStage


class PipelinedFrameDispatchTests(unittest.TestCase):
    def _app(self, *, inline=True, enabled=True):
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app._inline_inference = inline
        app._video_effects = SimpleNamespace(enabled=enabled)
        app._video_pipeline = object()
        app._inference_stage_owner = None
        app._last_pipelined_output = None
        self.inferred = []

        def infer(frame, width, height):
            self.inferred.append(frame)
            return f"alpha({frame.decode()})"

        app._inference_stage = PipelinedInferenceStage(
            infer, lambda value: value + "+refined")
        self.addCleanup(app._inference_stage.shutdown)
        self.rendered = []

        def process(frame, width, height, staged=None):
            self.rendered.append((frame, None if staged is None else staged.prepared))
            return b"out:" + frame

        app._process_frame = process
        return app

    def test_each_output_is_composited_with_its_own_frame_alpha(self):
        app = self._app()
        outputs = [app._process_effect_frame(f"f{i}".encode(), 4, 2) for i in range(6)]
        # Cold start renders f0 itself, then repeats it while two frames fill
        # the stages, then outputs follow capture order with their own alpha.
        self.assertEqual(outputs, [b"out:f0", b"out:f0", b"out:f0",
                                   b"out:f1", b"out:f2", b"out:f3"])
        for frame, prepared in self.rendered:
            self.assertEqual(prepared, f"alpha({frame.decode()})+refined")
        self.assertEqual([frame for frame, _ in self.rendered],
                         [b"f0", b"f1", b"f2", b"f3"])

    def test_serial_policy_drains_and_renders_inline(self):
        app = self._app()
        app._process_effect_frame(b"f0", 4, 2)
        app._process_effect_frame(b"f1", 4, 2)
        app._inline_inference = False
        self.assertEqual(app._process_effect_frame(b"f2", 4, 2), b"out:f2")
        self.assertEqual(self.rendered[-1], (b"f2", None))
        self.assertFalse(app._inference_stage.active)
        self.assertIsNone(app._last_pipelined_output)

    def test_disabled_effects_never_pipeline(self):
        app = self._app(enabled=False)
        self.assertEqual(app._process_effect_frame(b"f0", 4, 2), b"out:f0")
        self.assertEqual(self.rendered, [(b"f0", None)])
        self.assertEqual(self.inferred, [])

    def test_env_opt_out_keeps_serial_path(self):
        app = self._app()
        with mock.patch.dict(app_module.os.environ,
                             {"NVBROADCAST_PIPELINED_INFERENCE": "0"}):
            self.assertEqual(app._process_effect_frame(b"f0", 4, 2), b"out:f0")
        self.assertEqual(self.inferred, [])

    def test_new_capture_pipeline_never_shows_previous_capture_frames(self):
        app = self._app()
        for index in range(4):
            app._process_effect_frame(f"old{index}".encode(), 4, 2)
        app._video_pipeline = object()
        # The old queue and the old output are dropped: the new capture's
        # first frame renders with its own alpha.
        self.assertEqual(app._process_effect_frame(b"new0", 4, 2), b"out:new0")

    def test_stale_output_is_not_repeated_while_refilling(self):
        app = self._app()
        clock = [50.0]
        with mock.patch.object(app_module.time, "monotonic",
                               side_effect=lambda: clock[0]), \
             mock.patch("nvbroadcast.video.inference_stage.time.monotonic",
                        side_effect=lambda: clock[0]):
            for index in range(4):
                app._process_effect_frame(f"a{index}".encode(), 4, 2)
            clock[0] += 60.0
            self.assertEqual(app._process_effect_frame(b"b0", 4, 2), b"out:b0")
            # A fresh output may bridge the refill.
            self.assertEqual(app._process_effect_frame(b"b1", 4, 2), b"out:b0")


if __name__ == "__main__":
    unittest.main()

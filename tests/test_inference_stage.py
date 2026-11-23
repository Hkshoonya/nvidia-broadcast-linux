"""Ordering and freshness contract of the pipelined inference stage."""

import threading
import unittest
from unittest import mock

from nvbroadcast.video import inference_stage
from nvbroadcast.video.inference_stage import PipelinedInferenceStage


class PipelinedInferenceStageTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.lock = threading.Lock()

        def prepare(frame, width, height):
            with self.lock:
                self.calls.append(frame)
            return ("alpha", frame, width, height)

        self.stage = PipelinedInferenceStage(prepare)
        self.addCleanup(self.stage.shutdown)

    def test_each_frame_returns_with_its_own_prepared_result(self):
        self.assertIsNone(self.stage.advance("f0", 4, 2))
        for index in range(1, 6):
            staged = self.stage.advance(f"f{index}", 4, 2)
            self.assertEqual(staged.frame, f"f{index - 1}")
            self.assertEqual(staged.prepared, ("alpha", f"f{index - 1}", 4, 2))
        self.assertEqual(self.stage.take_oldest().frame, "f5")
        self.assertEqual(self.calls, [f"f{index}" for index in range(6)])

    def test_cold_start_can_take_the_current_frame(self):
        self.assertIsNone(self.stage.advance("f0", 4, 2))
        staged = self.stage.take_oldest()
        self.assertEqual(staged.frame, "f0")
        self.assertFalse(self.stage.active)
        self.assertIsNone(self.stage.advance("f1", 4, 2))

    def test_size_change_never_returns_the_other_size(self):
        self.stage.advance("small", 4, 2)
        self.assertIsNone(self.stage.advance("large", 8, 4))
        self.assertEqual(self.stage.advance("next", 8, 4).frame, "large")
        # The stale job still ran to completion before the next one started.
        self.assertEqual(self.calls, ["small", "large", "next"])

    def test_stale_pending_frame_is_dropped(self):
        clock = [100.0]
        with mock.patch.object(inference_stage.time, "monotonic",
                               side_effect=lambda: clock[0]):
            self.stage.advance("old", 4, 2)
            clock[0] += PipelinedInferenceStage.MAX_PENDING_AGE_S + 0.1
            self.assertIsNone(self.stage.advance("new", 4, 2))
            clock[0] += 0.01
            self.assertEqual(self.stage.advance("after", 4, 2).frame, "new")

    def test_drain_discards_in_flight_frame(self):
        self.stage.advance("f0", 4, 2)
        self.stage.drain()
        self.assertFalse(self.stage.active)
        self.assertIsNone(self.stage.advance("f1", 4, 2))

    def test_inference_errors_reach_the_caller_for_that_frame(self):
        def failing(frame, *_args):
            raise RuntimeError(f"failed {frame}")

        stage = PipelinedInferenceStage(failing, lambda value: value)
        self.addCleanup(stage.shutdown)
        stage.advance("f0", 4, 2)
        stage.advance("f1", 4, 2)
        with self.assertRaisesRegex(RuntimeError, "failed f0"):
            stage.advance("f2", 4, 2)


class ChainedStageTests(unittest.TestCase):
    def test_chained_stages_keep_per_stage_order_and_pairing(self):
        seen = {"infer": [], "refine": [], "matte": []}
        lock = threading.Lock()

        def record(name, value):
            with lock:
                seen[name].append(value)

        def infer(frame, width, height):
            record("infer", frame)
            return [frame]

        def refine(value):
            record("refine", value[0])
            return value + ["refined"]

        def matte(value):
            record("matte", value[0])
            return value + ["matte"]

        stage = PipelinedInferenceStage(infer, refine, matte)
        self.addCleanup(stage.shutdown)
        self.assertEqual(stage.depth, 3)
        outputs = []
        for index in range(8):
            staged = stage.advance(f"f{index}", 4, 2)
            if staged is not None:
                outputs.append((staged.frame, staged.prepared))
        # Three frames stay in flight; each returns with its own results.
        self.assertEqual([frame for frame, _ in outputs],
                         [f"f{index}" for index in range(5)])
        for frame, prepared in outputs:
            self.assertEqual(prepared, [frame, "refined", "matte"])
        stage.drain()
        for name in seen:
            self.assertEqual(seen[name], [f"f{index}" for index in range(8)])

    def test_refill_returns_none_until_full(self):
        stage = PipelinedInferenceStage(
            lambda frame, *_: frame, lambda value: value)
        self.addCleanup(stage.shutdown)
        self.assertIsNone(stage.advance("f0", 4, 2))
        self.assertIsNone(stage.advance("f1", 4, 2))
        self.assertEqual(stage.advance("f2", 4, 2).frame, "f0")


if __name__ == "__main__":
    unittest.main()

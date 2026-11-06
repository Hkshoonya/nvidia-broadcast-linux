"""Face-result provenance with scalar frames and no native imports."""
import ast
import copy
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref


SOURCE = Path(__file__).resolve().parents[1] / "src/nvbroadcast/video/face_landmarks.py"


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Frame:
    def __init__(self, sequence):
        self.sequence = sequence

    def copy(self):
        return _Frame(self.sequence)


class _Event:
    def __init__(self, manager=None):
        self.manager = manager
        self.waits = 0

    def set(self):
        pass

    def clear(self):
        pass

    def wait(self):
        self.waits += 1
        if self.waits > 1:
            self.manager._worker_stop = True


def _closed_manager_class():
    tree = ast.parse(SOURCE.read_text())
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "SharedFaceLandmarker")
    names = {"__init__", "ready", "latest", "completed_for_frame",
             "request_async", "detect", "_async_loop"}
    methods = [copy.deepcopy(node) for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name="SharedFaceLandmarker", bases=[], keywords=[],
                     body=methods, decorator_list=[]),
    ], type_ignores=[])
    scope = {
        "weakref": weakref,
        "threading": SimpleNamespace(Lock=_Lock, Event=_Event),
        "FaceTrackingStatus": lambda state: SimpleNamespace(state=state),
    }
    exec(compile(ast.fix_missing_locations(module), "closed-face-provenance", "exec"), scope)
    return scope["SharedFaceLandmarker"]


class FaceLandmarkCompletionTests(unittest.TestCase):
    def make_manager(self):
        cls = _closed_manager_class()
        manager = cls.__new__(cls)
        manager._init = lambda: None
        manager.__init__()
        manager._initialized = True
        manager._landmarker = object()
        manager.detection_calls = []

        def detect(frame):
            manager.detection_calls.append(frame.sequence)
            return ("geometry", frame.sequence)

        manager._run_detection = detect
        return manager

    def finish_pending(self, manager):
        manager._worker_stop = False
        manager._worker_event = _Event(manager)
        manager._async_loop()

    def test_initial_state_has_no_completed_frame(self):
        manager = self.make_manager()
        self.assertEqual(manager.completed_for_frame(_Frame(0)), (False, None))
        self.assertEqual(manager.detection_calls, [])

    def test_current_async_completion_identifies_original_source_not_copy(self):
        manager = self.make_manager()
        source = _Frame(3)
        self.assertIsNone(manager.request_async(source, reuse_frames=1))
        self.assertEqual(manager.detection_calls, [])
        self.finish_pending(manager)
        self.assertEqual(manager.completed_for_frame(source), (True, ("geometry", 3)))
        self.assertEqual(manager.completed_for_frame(_Frame(3)), (False, None))
        self.assertEqual(manager.detection_calls, [3])

    def test_reuse_identity_does_not_claim_current_inference(self):
        manager = self.make_manager()
        first, reused = _Frame(1), _Frame(2)
        manager.request_async(first, reuse_frames=1)
        self.finish_pending(manager)
        self.assertEqual(manager.request_async(reused, reuse_frames=2), ("geometry", 1))
        self.assertIs(manager._last_frame_ref(), reused)
        self.assertEqual(manager.completed_for_frame(reused), (False, None))
        self.assertEqual(manager.completed_for_frame(first), (True, ("geometry", 1)))
        self.assertIsNone(manager._pending_job)
        self.assertEqual(manager.detection_calls, [1])

    def test_completed_no_face_is_distinct_from_no_current_completion(self):
        manager = self.make_manager()
        old, current = _Frame(1), _Frame(2)
        manager.request_async(old, reuse_frames=1)
        self.finish_pending(manager)
        manager._run_detection = lambda frame: None
        self.assertEqual(manager.request_async(current, reuse_frames=1), ("geometry", 1))
        self.finish_pending(manager)
        self.assertEqual(manager.completed_for_frame(current), (True, None))
        self.assertEqual(manager.completed_for_frame(old), (False, None))
        self.assertIsNone(manager.latest())

    def test_newer_frame_result_cannot_be_claimed_for_other_frame(self):
        manager = self.make_manager()
        current, newer = _Frame(7), _Frame(8)
        manager.request_async(current, reuse_frames=1)
        self.finish_pending(manager)
        manager.request_async(newer, reuse_frames=1)
        self.finish_pending(manager)
        self.assertEqual(manager.completed_for_frame(current), (False, None))
        self.assertEqual(manager.completed_for_frame(newer), (True, ("geometry", 8)))

    def test_requests_remain_nonblocking_with_one_newest_pending_job(self):
        manager = self.make_manager()
        frames = [_Frame(index) for index in range(5)]
        for frame in frames:
            manager.request_async(frame, reuse_frames=1)
        self.assertEqual(manager.detection_calls, [])
        self.assertEqual(manager._pending_job.sequence, 4)
        self.finish_pending(manager)
        self.assertEqual(manager.detection_calls, [4])
        for frame in frames[:-1]:
            self.assertEqual(manager.completed_for_frame(frame), (False, None))
        self.assertEqual(manager.completed_for_frame(frames[-1]), (True, ("geometry", 4)))

    def test_synchronous_detection_publishes_genuine_source_identity(self):
        manager = self.make_manager()
        source = _Frame(9)
        self.assertEqual(manager.detect(source, reuse_frames=1), ("geometry", 9))
        self.assertEqual(manager.completed_for_frame(source), (True, ("geometry", 9)))
        self.assertEqual(manager.detection_calls, [9])

    def test_source_reference_does_not_retain_a_finished_frame(self):
        manager = self.make_manager()
        source = _Frame(1)
        reference = weakref.ref(source)
        manager.request_async(source, reuse_frames=1)
        self.finish_pending(manager)
        del source
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(manager.completed_for_frame(_Frame(1)), (False, None))


if __name__ == "__main__":
    unittest.main()

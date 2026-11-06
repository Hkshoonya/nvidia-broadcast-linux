"""Frame-matched Relighting dispatch without native imports or image data.

The processing method is extracted from the application with its two native
imports replaced by scalar test boundaries. No camera, model or pixels exist
in these tests. The manager's completion-publication contract is tested apart.
"""

import ast
import copy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


APP_SOURCE = Path(__file__).resolve().parents[1] / "src/nvbroadcast/app.py"
_PAYLOAD = b"generated scalar output token"


def _app_class(namespace):
    tree = ast.parse(APP_SOURCE.read_text())
    source = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "NVBroadcastApp")
    names = {"_process_frame", "_face_effect_load_score", "_landmark_reuse_frames"}
    body = [copy.deepcopy(node) for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in body} != names:
        raise AssertionError("Processing and cadence methods must be available")
    for method in body:
        if method.name == "_process_frame":
            imports = [node for node in method.body
                       if isinstance(node, (ast.Import, ast.ImportFrom))]
            if any(not isinstance(node, ast.Import)
                   or any(alias.name not in {"cv2", "numpy"} for alias in node.names)
                   for node in imports):
                raise AssertionError("Unexpected native import boundary")
            method.body = [node for node in method.body if node not in imports]
        if any(isinstance(node, (ast.Import, ast.ImportFrom))
               for node in ast.walk(method)):
            raise AssertionError("Extracted methods cannot import packages")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", level=0,
                       names=[ast.alias(name="annotations")]),
        ast.ClassDef(name="NVBroadcastApp", bases=[], keywords=[], body=body,
                     decorator_list=[]),
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(APP_SOURCE), "exec"), namespace)
    return namespace["NVBroadcastApp"]


class _ScalarFrame:
    """Carries identity and an output token, never pixel storage."""

    def __init__(self, *, writeable):
        self.flags = SimpleNamespace(writeable=writeable)
        self.copy_count = 0

    def reshape(self, *shape):
        if shape != (720, 1280, 4):
            raise AssertionError("Unexpected generated dimensions")
        return self

    def copy(self):
        self.copy_count += 1
        return _ScalarFrame(writeable=True)

    def tobytes(self):
        return _PAYLOAD


class RelightingFrameFreshnessTests(unittest.TestCase):
    def run_frame(self, completion, *, relight=True, beauty=False, cached=None,
                  ready=True):
        old = object() if cached is None else cached
        if cached is _EMPTY:
            old = None
        current = object()
        different = object()
        input_frame = _ScalarFrame(writeable=False)
        composite_frame = _ScalarFrame(writeable=True)
        alpha = object()
        state = {"requested_raw": None, "completed_frame": None,
                 "completed_result": None, "latest": old}
        events = []

        def request(frame, *, reuse_frames):
            events.append("request")
            state["requested_raw"] = frame
            self.assertIsNot(frame, input_frame)
            self.assertIsNot(frame, composite_frame)
            self.assertTrue(frame.flags.writeable)
            return old

        def completed_for_frame(frame):
            events.append("completion_lookup")
            self.assertIs(frame, state["requested_raw"])
            if state["completed_frame"] is frame:
                return True, state["completed_result"]
            return False, None

        tracker = SimpleNamespace(
            ready=ready,
            request_async=mock.Mock(side_effect=request),
            completed_for_frame=mock.Mock(side_effect=completed_for_frame),
            detect=mock.Mock(side_effect=AssertionError("No synchronous detection")),
            latest=mock.Mock(side_effect=AssertionError("No unpaired latest result")),
            detection_status=SimpleNamespace(state="tracking"),
        )

        def background(frame, *args, **kwargs):
            events.append("background")
            if ready:
                self.assertIs(frame, state["requested_raw"])
            else:
                state["requested_raw"] = frame
            if completion == "current":
                state.update(completed_frame=frame, completed_result=current,
                             latest=current)
            elif completion == "no_face":
                state.update(completed_frame=frame, completed_result=None, latest=None)
            elif completion == "different":
                state.update(completed_frame=object(), completed_result=different,
                             latest=different)
            elif completion == "steady":
                state.update(completed_frame=frame, completed_result=old)
            elif completion != "none":
                raise AssertionError("Unknown generated completion schedule")
            return composite_frame

        def face_stage(name):
            def process(frame, *args, **kwargs):
                events.append(name)
                self.assertIs(frame, composite_frame)
                return frame
            return mock.Mock(side_effect=process)

        factory = mock.Mock(return_value=tracker)
        App = _app_class({
            "np": SimpleNamespace(frombuffer=mock.Mock(return_value=input_frame),
                                  uint8=object()),
            "cv2": SimpleNamespace(
                flip=mock.Mock(side_effect=AssertionError("Unexpected transform"))),
            "get_shared_landmarker": factory,
        })
        app = App()
        app._perf_monitor = SimpleNamespace(tick=mock.Mock())
        app._beautifier = SimpleNamespace(enabled=beauty,
                                         process_frame_array=face_stage("beauty"))
        app._eye_contact = SimpleNamespace(enabled=True,
                                          process_frame=face_stage("eye"),
                                          set_tracking_status=mock.Mock())
        app._relighter = SimpleNamespace(enabled=relight,
                                         process_frame=face_stage("relight"))
        app._autoframe = SimpleNamespace(enabled=False)
        app._mirror = False
        app._inline_inference = True
        app._video_effects = SimpleNamespace(
            enabled=True, _use_fused_kernel=False,
            set_fused_face_overlay=mock.Mock(),
            process_frame_array=mock.Mock(side_effect=background),
            latest_final_matte_u8=mock.Mock(return_value=alpha),
        )
        output = app._process_frame(b"generated input marker", 1280, 720)
        self.assertIs(output, _PAYLOAD)
        self.assertEqual(input_frame.copy_count, 1)
        factory.assert_called_once_with()
        if ready:
            tracker.request_async.assert_called_once()
        else:
            tracker.request_async.assert_not_called()
        tracker.detect.assert_not_called()
        tracker.latest.assert_not_called()
        return SimpleNamespace(app=app, tracker=tracker, old=old, current=current,
                               different=different, output=output, alpha=alpha,
                               composite=composite_frame, events=events,
                               raw=state["requested_raw"])

    def test_current_frame_completion_replaces_only_relighting_snapshot(self):
        run = self.run_frame("current")

        run.app._eye_contact.process_frame.assert_called_once_with(
            run.composite, landmarks=run.old)
        run.app._relighter.process_frame.assert_called_once_with(
            run.composite, run.alpha, landmarks=run.current)
        run.tracker.completed_for_frame.assert_called_once_with(run.raw)
        self.assertEqual(run.tracker.request_async.call_args.kwargs["reuse_frames"], 1)
        self.assertLess(run.events.index("background"), run.events.index("completion_lookup"))
        self.assertLess(run.events.index("eye"), run.events.index("completion_lookup"))
        self.assertLess(run.events.index("completion_lookup"), run.events.index("relight"))

    def test_no_current_completion_keeps_captured_cached_result(self):
        run = self.run_frame("none")

        run.app._relighter.process_frame.assert_called_once_with(
            run.composite, run.alpha, landmarks=run.old)

    def test_current_no_face_completion_drops_stale_relighting(self):
        run = self.run_frame("no_face")

        run.app._relighter.process_frame.assert_not_called()
        run.app._eye_contact.process_frame.assert_called_once_with(
            run.composite, landmarks=run.old)

    def test_different_frame_completion_cannot_replace_this_frames_snapshot(self):
        run = self.run_frame("different")

        run.app._relighter.process_frame.assert_called_once_with(
            run.composite, run.alpha, landmarks=run.old)

    def test_steady_completed_result_preserves_consumer_inputs_and_output_token(self):
        run = self.run_frame("steady")

        run.app._eye_contact.process_frame.assert_called_once_with(
            run.composite, landmarks=run.old)
        run.app._relighter.process_frame.assert_called_once_with(
            run.composite, run.alpha, landmarks=run.old)
        self.assertIs(run.output, _PAYLOAD)

    def test_disabled_relighting_does_not_read_completion_or_change_eye_input(self):
        run = self.run_frame("current", relight=False)

        run.tracker.completed_for_frame.assert_not_called()
        run.app._relighter.process_frame.assert_not_called()
        run.app._eye_contact.process_frame.assert_called_once_with(
            run.composite, landmarks=run.old)

    def test_first_current_completion_is_usable_after_an_empty_cached_result(self):
        run = self.run_frame("current", cached=_EMPTY)

        run.app._eye_contact.process_frame.assert_not_called()
        run.app._relighter.process_frame.assert_called_once_with(
            run.composite, run.alpha, landmarks=run.current)

    def test_empty_cache_and_different_completion_do_not_apply_relighting(self):
        run = self.run_frame("different", cached=_EMPTY)

        run.app._relighter.process_frame.assert_not_called()

    def test_unavailable_tracker_keeps_video_and_skips_face_effects(self):
        run = self.run_frame("none", cached=_EMPTY, ready=False)

        run.tracker.request_async.assert_not_called()
        run.tracker.completed_for_frame.assert_called_once_with(run.raw)
        run.app._eye_contact.process_frame.assert_not_called()
        run.app._eye_contact.set_tracking_status.assert_called_once_with("unavailable")
        run.app._relighter.process_frame.assert_not_called()
        self.assertIs(run.output, _PAYLOAD)

    def test_beauty_and_eye_keep_their_original_snapshot(self):
        run = self.run_frame("current", beauty=True)

        self.assertIs(run.app._beautifier.process_frame_array.call_args.kwargs["landmarks"],
                      run.old)
        run.app._eye_contact.process_frame.assert_called_once_with(
            run.composite, landmarks=run.old)
        run.app._relighter.process_frame.assert_called_once_with(
            run.composite, run.alpha, landmarks=run.current)
        self.assertLess(run.events.index("beauty"), run.events.index("completion_lookup"))

    def test_relighting_requests_every_frame_without_changing_other_profiles(self):
        # This is the pre-fix policy table when Relighting is disabled.
        prior = {
            (False, False, False): 2, (False, False, True): 2,
            (False, True, False): 1, (False, True, True): 2,
            (True, False, False): 2, (True, False, True): 2,
            (True, True, False): 2, (True, True, True): 2,
        }
        App = _app_class({})
        for beauty, eye, relight, autoframe in product((False, True), repeat=4):
            with self.subTest(beauty=beauty, eye=eye, relight=relight,
                              autoframe=autoframe):
                app = App()
                app._beautifier = SimpleNamespace(enabled=beauty)
                app._eye_contact = SimpleNamespace(enabled=eye)
                app._relighter = SimpleNamespace(enabled=relight)
                app._autoframe = SimpleNamespace(enabled=autoframe)
                expected = 1 if relight else prior[(beauty, eye, autoframe)]
                self.assertEqual(app._landmark_reuse_frames(), expected)


_EMPTY = object()


if __name__ == "__main__":
    unittest.main()

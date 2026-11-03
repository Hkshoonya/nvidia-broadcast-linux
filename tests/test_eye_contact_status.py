"""Application-level Eye Contact diagnostics without devices or native inference."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.video.eye_contact import EyeContactCorrector
from nvbroadcast.video.face_landmarks import FaceTrackingStatus


class EyeContactAppStatusTests(unittest.TestCase):
    @staticmethod
    def _app():
        app = NVBroadcastApp.__new__(NVBroadcastApp)
        app._eye_contact = EyeContactCorrector()
        app._eye_contact.enabled = True
        app._streaming = True
        app._beautifier = SimpleNamespace(enabled=False)
        app._relighter = SimpleNamespace(enabled=False)
        app._autoframe = SimpleNamespace(enabled=False)
        app._video_effects = SimpleNamespace(enabled=False)
        app._mirror = False
        app._perf_monitor = SimpleNamespace(tick=mock.Mock())
        app._landmark_reuse_frames = mock.Mock(return_value=1)
        return app

    def test_missing_async_result_reports_actual_detector_outcome_without_inline_detection(self):
        frame = np.full((12, 16, 4), 77, dtype=np.uint8)
        app = self._app()
        tracker = SimpleNamespace(
            ready=True, request_async=mock.Mock(return_value=None),
            detection_status=FaceTrackingStatus("waiting"),
        )
        with (
            mock.patch("nvbroadcast.app.get_shared_landmarker", return_value=tracker),
            mock.patch("nvbroadcast.video.eye_contact.get_shared_landmarker") as inline,
            mock.patch.object(app._eye_contact, "process_frame") as correct,
        ):
            for source, expected in (("waiting", "waiting"), ("no_face", "no_face"),
                                     ("error", "tracking_error"), ("tracking", "waiting")):
                with self.subTest(source=source):
                    # A previous active frame must not mask the current absence.
                    app._eye_contact._set_status("correcting", 2)
                    tracker.detection_status = FaceTrackingStatus(source)
                    output = app._process_frame(frame.tobytes(), 16, 12)
                    self.assertEqual(output, frame.tobytes())
                    self.assertEqual(app.eye_contact_status.state, expected)
                    self.assertEqual(app.eye_contact_status.corrected_eyes, 0)
            correct.assert_not_called()
            inline.assert_not_called()
            self.assertEqual(tracker.request_async.call_count, 4)

    def test_unavailable_tracker_is_distinct_and_does_not_request_work(self):
        app = self._app()
        tracker = SimpleNamespace(ready=False, request_async=mock.Mock())
        frame = np.zeros((12, 16, 4), dtype=np.uint8)
        with mock.patch("nvbroadcast.app.get_shared_landmarker", return_value=tracker):
            self.assertEqual(app._process_frame(frame.tobytes(), 16, 12), frame.tobytes())
        self.assertEqual(app.eye_contact_status.state, "tracker_unavailable")
        tracker.request_async.assert_not_called()

    def test_disabled_and_stopped_states_override_old_processing_snapshot(self):
        app = self._app()
        app._eye_contact._set_status("correcting", 2)
        self.assertEqual(app.eye_contact_status.state, "correcting")
        app._streaming = False
        self.assertEqual(app.eye_contact_status.state, "stopped")
        app._eye_contact.enabled = False
        self.assertEqual(app.eye_contact_status.state, "disabled")
        app._eye_contact.enabled = True
        app._streaming = True
        self.assertEqual(app.eye_contact_status.state, "waiting")


if __name__ == "__main__":
    unittest.main()

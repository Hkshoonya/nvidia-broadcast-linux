import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nvbroadcast.core.meeting_store import MeetingSession


class MeetingStoreTests(unittest.TestCase):
    def test_save_and_list_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import nvbroadcast.core.meeting_store as store

            with mock.patch.object(store, "MEETINGS_DIR", Path(tmpdir)):
                session = MeetingSession(
                    session_id="20260327-123000",
                    created_at=123,
                    title="Planning Meeting",
                    summary="30 minute meeting | 2 action items",
                    transcript_preview="hello world",
                    duration_seconds=1800,
                    notes_path="/tmp/notes.md",
                    transcript_path="/tmp/transcript.txt",
                    transcript_srt_path="/tmp/transcript.srt",
                    audio_path="/tmp/meeting.wav",
                    video_path="/tmp/meeting.mp4",
                )
                store.save_session(session)
                sessions = store.list_sessions()
                self.assertEqual(len(sessions), 1)
                self.assertEqual(sessions[0].title, "Planning Meeting")


if __name__ == "__main__":
    unittest.main()

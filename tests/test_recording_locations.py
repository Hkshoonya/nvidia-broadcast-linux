import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from nvbroadcast.core.recordings import recordings_directory, legacy_recordings_directory
from nvbroadcast.app import NVBroadcastApp
from nvbroadcast.ui.window import NVBroadcastWindow


class RecordingLocationTests(unittest.TestCase):
    @mock.patch("nvbroadcast.core.recordings.GLib.get_user_special_dir")
    def test_desktop_directory_wins_over_snap_private_home(self, special):
        special.return_value = "/home/test/My Videos"
        with mock.patch.dict(os.environ, {"SNAP": "/snap/app/1", "SNAP_REAL_HOME": "/home/test"}), \
             mock.patch("pathlib.Path.home", return_value=Path("/home/test/snap/app/1")):
            self.assertEqual(recordings_directory(), Path("/home/test/My Videos"))

    @mock.patch("nvbroadcast.core.recordings.GLib.get_user_special_dir", return_value=None)
    def test_snap_fallback_uses_real_home(self, _special):
        with mock.patch.dict(os.environ, {"SNAP": "/snap/app/1", "SNAP_REAL_HOME": "/home/test"}), \
             mock.patch("pathlib.Path.home", return_value=Path("/home/test/snap/app/1")):
            self.assertEqual(recordings_directory(), Path("/home/test/Videos"))

    @mock.patch("nvbroadcast.core.recordings.GLib.get_user_special_dir", return_value="relative")
    def test_native_fallback_ignores_snap_environment_without_snap(self, _special):
        with mock.patch.dict(os.environ, {"SNAP_REAL_HOME": "/wrong"}, clear=True), \
             mock.patch("pathlib.Path.home", return_value=Path("/home/test")):
            self.assertEqual(recordings_directory(), Path("/home/test/Videos"))

    def test_previous_snap_recordings_remain_available_without_moving_files(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            old = home / "Videos"
            old.mkdir()
            recording = old / "existing.mp4"
            recording.write_bytes(b"unchanged")
            with mock.patch.dict(os.environ, {"SNAP": "/snap/app/1"}), \
                 mock.patch("pathlib.Path.home", return_value=home), \
                 mock.patch("nvbroadcast.core.recordings.recordings_directory", return_value=home / "Host Videos"):
                self.assertEqual(legacy_recordings_directory(), old)
                self.assertEqual(recording.read_bytes(), b"unchanged")

    def test_rec_creates_desktop_folder_and_passes_full_path_to_recorder(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "My Videos" / "Recordings"
            pipeline = mock.Mock(is_recording=False, recording_finalizing=False)
            app = SimpleNamespace(_video_pipeline=pipeline, _meeting_active=False,
                                  _meeting_finalizing=False, _idle_active=False,
                                  config=SimpleNamespace(audio=SimpleNamespace(mic_device="test-mic")))
            with mock.patch("nvbroadcast.app.recordings_directory", return_value=destination):
                filepath = NVBroadcastApp.start_recording(app)
            self.assertEqual(Path(filepath).parent, destination)
            self.assertTrue(destination.is_dir())
            pipeline.start_recording.assert_called_once_with(
                filepath, wait_for_codecs=False, mic_device="test-mic"
            )

    def test_uncreatable_folder_reports_failure_without_starting_recorder(self):
        pipeline = mock.Mock(is_recording=False, recording_finalizing=False)
        app = SimpleNamespace(_video_pipeline=pipeline, _meeting_active=False, _meeting_finalizing=False)
        with mock.patch("nvbroadcast.app.recordings_directory", return_value=Path("/missing/Videos")), \
             mock.patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")):
            self.assertEqual(NVBroadcastApp.start_recording(app), "")
        self.assertIn("Cannot create recordings folder", app._recording_start_error)
        pipeline.start_recording.assert_not_called()

    def test_failed_finalize_does_not_offer_incomplete_file_as_last_recording(self):
        window = SimpleNamespace(_app=SimpleNamespace(last_recording_path="/new.mp4"),
                                 _saved_recording_path="/previous.mp4",
                                 _open_recording_btn=mock.Mock(), set_status=mock.Mock())
        NVBroadcastWindow.on_recording_finalized(window, False, "failed")
        self.assertEqual(window._saved_recording_path, "/previous.mp4")
        window._open_recording_btn.set_sensitive.assert_not_called()
        window.set_status.assert_called_once_with("Recording may be incomplete")

    def test_open_recording_passes_spaced_filename_to_async_desktop_launcher(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "my recording.mp4"
            path.touch()
            window = SimpleNamespace(_recordings_menu=mock.Mock(), set_status=mock.Mock(),
                                     _on_recording_location_opened=mock.Mock())
            with mock.patch("nvbroadcast.ui.window.Gtk.FileLauncher.new") as create:
                NVBroadcastWindow._open_recording_location(window, str(path))
            self.assertEqual(create.call_args.args[0].get_path(), str(path))
            create.return_value.launch.assert_called_once_with(
                window, None, window._on_recording_location_opened
            )
            window.set_status.assert_not_called()

    def test_removed_recording_reports_missing_path_without_launching(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "removed.mp4")
            window = SimpleNamespace(_recordings_menu=mock.Mock(), set_status=mock.Mock())
            with mock.patch("nvbroadcast.ui.window.Gtk.FileLauncher.new") as create:
                NVBroadcastWindow._open_recording_location(window, path)
            create.assert_not_called()
            window.set_status.assert_called_once_with(f"Recording location no longer exists: {path}")

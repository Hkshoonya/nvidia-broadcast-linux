import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nvbroadcast.audio.pipeline import AudioPipeline


class LogFilePermissionTests(unittest.TestCase):
    """Persistent logs capture device names, settings and library errors;
    they must be private to the user, never group/world readable."""

    def test_audio_helper_log_is_private(self):
        pipeline = AudioPipeline(use_helper_process=False)
        pipeline._debug_audio = False
        # Import voice_fx (and transitively cupy) before XDG_CACHE_HOME is
        # patched — cupy resolves its kernel cache from it at import time.
        pipeline.voice_fx
        fake_proc = mock.Mock()
        fake_proc.poll.return_value = None

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}), \
                 mock.patch.object(pipeline, "_stop_helper_process"), \
                 mock.patch.object(pipeline, "_stop_stale_helper_processes"), \
                 mock.patch("nvbroadcast.audio.pipeline.subprocess.Popen",
                            return_value=fake_proc), \
                 mock.patch("nvbroadcast.audio.pipeline.time.sleep"):
                self.assertTrue(pipeline._start_helper_process())

            log_path = Path(tmp) / "nvbroadcast" / "audio-helper.log"
            self.assertTrue(log_path.exists())
            mode = stat.S_IMODE(log_path.stat().st_mode)
            self.assertEqual(mode & 0o077, 0,
                             f"audio-helper.log is not private: {oct(mode)}")

    def test_app_log_is_private(self):
        # _redirect_output_to_log rewires the calling process's fds, so run
        # it in a child interpreter with a piped (non-tty) stdout.
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["XDG_STATE_HOME"] = tmp
            env.pop("NVBROADCAST_NO_LOG_FILE", None)
            result = subprocess.run(
                [sys.executable, "-c",
                 "from nvbroadcast.__main__ import _redirect_output_to_log; "
                 "_redirect_output_to_log(); print('log check')"],
                env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            state_dir = Path(tmp) / "nvbroadcast"
            log_path = state_dir / "nvbroadcast.log"
            self.assertTrue(log_path.exists())
            self.assertIn("log check", log_path.read_text())
            for path in (state_dir, log_path):
                mode = stat.S_IMODE(path.stat().st_mode)
                self.assertEqual(mode & 0o077, 0,
                                 f"{path.name} is not private: {oct(mode)}")


if __name__ == "__main__":
    unittest.main()

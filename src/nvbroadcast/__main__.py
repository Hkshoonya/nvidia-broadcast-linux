# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Entry point for NVIDIA Broadcast."""

import os
import sys


def _redirect_output_to_log():
    """Send stdout/stderr to a log file when not attached to a terminal.

    Desktop launchers point stdout at /dev/null, which made every
    `[NV Broadcast]` print (and GStreamer's native stderr) unrecoverable.
    Redirecting at the fd level with dup2 captures native output too,
    which a Python-level tee would miss. Terminal runs are left alone.
    """
    try:
        if os.environ.get("NVBROADCAST_NO_LOG_FILE") == "1":
            return
        if sys.stdout is not None and sys.stdout.isatty():
            return

        from nvbroadcast.core.constants import LOG_FILE, LOG_MAX_BYTES, STATE_DIR

        # Logs capture window titles, device names and library errors —
        # keep the directory and files private to the user.
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(STATE_DIR, 0o700)
        except OSError:
            pass
        try:
            if LOG_FILE.stat().st_size > LOG_MAX_BYTES:
                os.replace(LOG_FILE, LOG_FILE.with_suffix(".log.old"))
        except OSError:
            pass

        fd = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            # O_CREAT mode only applies to new files; fix up files created
            # by earlier builds with wider permissions.
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        # The fds now point at a regular file, so Python's stdio would
        # block-buffer and a crash could lose the most recent lines.
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.reconfigure(line_buffering=True)

        from datetime import datetime

        from nvbroadcast import __version__

        print(
            f"[NV Broadcast] --- started {datetime.now().isoformat(timespec='seconds')} "
            f"v{__version__} pid={os.getpid()} ---",
            flush=True,
        )
    except Exception:
        pass  # Logging must never prevent startup.


def main():
    _redirect_output_to_log()
    from nvbroadcast.app import NVBroadcastApp

    app = NVBroadcastApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())

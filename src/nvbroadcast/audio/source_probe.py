# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""Find an audio source that can deliver a buffer in this runtime."""

from __future__ import annotations

import os
import subprocess
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def probe_audio_source() -> tuple[str | None, str]:
    """Prefer PulseAudio, then PipeWire, after a bounded live capture probe.

    The probe runs in a child because a failed native source can hang or crash
    during teardown. Factory presence alone does not prove sandbox access.
    """
    probe_code = (
        "import gi, sys\n"
        "gi.require_version('Gst', '1.0')\n"
        "from gi.repository import Gst\n"
        "Gst.init(None)\n"
        "pipe = Gst.parse_launch(sys.argv[1] + "
        "' num-buffers=1 ! audio/x-raw ! fakesink sync=false')\n"
        "if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:\n"
        "    sys.exit('audio source could not start')\n"
        "msg = pipe.get_bus().timed_pop_filtered(\n"
        "    Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)\n"
        "if msg is None:\n"
        "    sys.exit('audio source did not produce a buffer')\n"
        "if msg.type == Gst.MessageType.ERROR:\n"
        "    sys.exit(msg.parse_error()[0].message)\n"
        "pipe.set_state(Gst.State.NULL)\n"
    )
    failures = []
    for source in ("pulsesrc", "pipewiresrc"):
        if Gst.ElementFactory.find(source) is None:
            failures.append(f"{source} is not installed")
            continue
        if source == "pipewiresrc":
            runtime_dir = os.environ.get("PIPEWIRE_RUNTIME_DIR") or os.environ.get(
                "XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"
            )
            remote = os.environ.get("PIPEWIRE_REMOTE", "pipewire-0")
            if not os.path.exists(os.path.join(runtime_dir, remote)):
                failures.append("PipeWire socket is unavailable")
                continue
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe_code, source],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if result.returncode == 0:
                return source, ""
            failures.append(f"{source}: {result.stderr.strip() or 'capture failed'}")
        except subprocess.TimeoutExpired:
            failures.append(f"{source} capture probe timed out")
        except OSError as exc:
            failures.append(f"{source} capture probe failed: {exc}")
    return None, "; ".join(failures)

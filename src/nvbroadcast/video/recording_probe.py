"""Isolated GStreamer encode probe used by the recording pipeline."""

from __future__ import annotations

import argparse
import json

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


RESULT_PREFIX = "NVBROADCAST_RECORDING_PROBE="


def run(graph: str) -> dict[str, str | bool | int]:
    """Execute a short graph and require EOS plus encoded output."""
    Gst.init(None)
    probe = None
    encoded_buffers = 0

    def _count_buffer(_sink, _buffer, _pad):
        nonlocal encoded_buffers
        encoded_buffers += 1

    try:
        probe = Gst.parse_launch(graph)
        probe_sink = probe.get_by_name("probe_sink")
        if probe_sink is None:
            return {"usable": False, "error": "encode probe has no named sink"}
        probe_sink.set_property("signal-handoffs", True)
        probe_sink.connect("handoff", _count_buffer)
        if probe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            return {"usable": False, "error": "pipeline could not enter PLAYING"}
        message = probe.get_bus().timed_pop_filtered(
            2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if message is None:
            return {"usable": False, "error": "encode probe timed out"}
        if message.type == Gst.MessageType.ERROR:
            error, _debug = message.parse_error()
            return {"usable": False, "error": error.message}
        if message.type != Gst.MessageType.EOS:
            return {"usable": False, "error": "encode probe ended without EOS"}
        if encoded_buffers < 1:
            return {
                "usable": False,
                "error": "encoder reached EOS without producing a buffer",
            }
        return {"usable": True, "error": "", "buffers": encoded_buffers}
    except Exception as exc:
        return {"usable": False, "error": str(exc) or exc.__class__.__name__}
    finally:
        if probe is not None:
            probe.set_state(Gst.State.NULL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph")
    args = parser.parse_args()
    result = run(args.graph)
    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
    return 0 if result["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

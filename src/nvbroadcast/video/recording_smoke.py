"""Bounded packaged-runtime smoke test for the recording codec path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GstPbutils

from nvbroadcast.video.pipeline import VideoPipeline


def _decode_streams(path: Path) -> dict[str, dict[str, int | None]]:
    decoded: dict[str, dict[str, int | None]] = {
        "audio": {"buffers": 0, "bytes": 0, "first_pts": None, "end": 0},
        "video": {"buffers": 0, "bytes": 0, "first_pts": None, "end": 0},
    }

    def _count(_sink, buffer, _pad, stream_type: str):
        stream = decoded[stream_type]
        stream["buffers"] += 1
        stream["bytes"] += buffer.get_size()
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return
        duration = (
            buffer.duration if buffer.duration != Gst.CLOCK_TIME_NONE else 0
        )
        first_pts = stream["first_pts"]
        stream["first_pts"] = (
            buffer.pts if first_pts is None else min(first_pts, buffer.pts)
        )
        stream["end"] = max(stream["end"], buffer.pts + duration)

    player = Gst.ElementFactory.make("playbin")
    audio_sink = Gst.ElementFactory.make("fakesink")
    video_sink = Gst.ElementFactory.make("fakesink")
    if not all((player, audio_sink, video_sink)):
        raise RuntimeError("Packaged runtime cannot construct decode sinks")
    for stream_type, sink in (("audio", audio_sink), ("video", video_sink)):
        sink.set_property("sync", False)
        sink.set_property("signal-handoffs", True)
        sink.connect("handoff", _count, stream_type)
    player.set_property("audio-sink", audio_sink)
    player.set_property("video-sink", video_sink)
    player.set_property("uri", Gst.filename_to_uri(str(path)))
    try:
        if player.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Packaged runtime could not start MP4 decode")
        message = player.get_bus().timed_pop_filtered(
            8 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS
        )
        if message is None:
            raise RuntimeError("Timed out decoding packaged recording")
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            raise RuntimeError(f"Packaged recording decode failed: {error.message}; {debug}")
    finally:
        player.set_state(Gst.State.NULL)

    minimum_buffers = {"audio": 50, "video": 18}
    for stream_type, stream in decoded.items():
        if stream["buffers"] < 1 or stream["bytes"] < 1:
            raise RuntimeError(f"Decoded {stream_type} stream is empty: {decoded}")
        if stream["buffers"] < minimum_buffers[stream_type]:
            raise RuntimeError(
                f"Decoded {stream_type} stream dropped too many buffers: {decoded}"
            )
        if stream["first_pts"] is None:
            raise RuntimeError(f"Decoded {stream_type} stream has no timestamps: {decoded}")
        stream["span"] = stream["end"] - stream["first_pts"]
        if stream["span"] <= Gst.SECOND:
            raise RuntimeError(f"Decoded {stream_type} stream is too short: {decoded}")
    if abs(decoded["audio"]["span"] - decoded["video"]["span"]) >= Gst.SECOND // 2:
        raise RuntimeError(f"Decoded audio/video spans diverge: {decoded}")
    return decoded


def run(output: Path) -> dict:
    """Encode, finalize, discover, and fully decode one synthetic recording."""
    Gst.init(None)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = VideoPipeline()
    pipeline._width, pipeline._height, pipeline._fps = 640, 360, 15
    pipeline._recording_audio_source = lambda: (
        "audiotestsrc is-live=true num-buffers=100 wave=sine",
        "",
    )
    pipeline.start_recording(str(output))
    if not pipeline.recording_has_audio:
        raise RuntimeError(
            "Packaged recording started without audio: "
            f"{pipeline.recording_audio_error}"
        )

    pixels = bytes((20, 40, 60, 255)) * (pipeline._width * pipeline._height)
    for index in range(24):
        flow = pipeline._push_recording_frame(
            pixels, Gst.SECOND // pipeline._fps
        )
        if flow != Gst.FlowReturn.OK:
            raise RuntimeError(f"Recording appsrc rejected frame {index}: {flow}")
        time.sleep(1 / pipeline._fps)
    if not pipeline.stop_recording():
        raise RuntimeError(
            f"Packaged recording did not finalize: {pipeline.recording_audio_error}"
        )

    discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)
    info = discoverer.discover_uri(Gst.filename_to_uri(str(output)))
    if info.get_result() != GstPbutils.DiscovererResult.OK:
        raise RuntimeError(f"Packaged recording discovery failed: {info.get_result()}")
    if len(info.get_video_streams()) != 1 or len(info.get_audio_streams()) != 1:
        raise RuntimeError(
            "Packaged recording does not contain exactly one video and audio stream"
        )
    decoded = _decode_streams(output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "duration_ns": info.get_duration(),
        "video_encoder": pipeline._recording_encoder_name,
        "audio_encoder": pipeline._recording_aac_encoder,
        "decoded": decoded,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = run(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="NVB recording smoke ") as directory:
            result = run(Path(directory) / "camera and audio.mp4")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

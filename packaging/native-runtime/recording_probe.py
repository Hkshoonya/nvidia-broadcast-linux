#!/usr/bin/env python3
"""Encode/decode generated media with the installed application's codec choice."""

import json
from pathlib import Path
import tempfile

import av
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from nvbroadcast.video.pipeline import VideoPipeline


def main():
    Gst.init([])
    application = VideoPipeline()
    application._width, application._height, application._fps = 320, 240, 24
    video_name, video_graph = application._find_recording_encoder()
    audio_candidates, failures = application._find_recording_aac_encoders()
    if not audio_candidates:
        raise RuntimeError("No usable application AAC encoder: " + "; ".join(failures))
    audio_name, audio_format = audio_candidates[0]
    with tempfile.TemporaryDirectory(prefix="nvb-generated-recording-") as directory:
        recording = Path(directory) / "generated.mp4"
        pipeline = Gst.parse_launch(
            f'mp4mux name=mux ! filesink location="{recording}" '
            'videotestsrc num-buffers=24 ! video/x-raw,format=BGRA,width=320,height=240,framerate=24/1 ! '
            f'{video_graph} ! h264parse ! queue ! mux. '
            'audiotestsrc num-buffers=48 samplesperbuffer=1024 wave=sine ! audioconvert ! audioresample ! '
            f'audio/x-raw,format={audio_format},rate=48000,channels=1 ! {audio_name} bitrate=128000 ! '
            'aacparse ! queue ! mux.')
        try:
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("generated recording could not start")
            message = pipeline.get_bus().timed_pop_filtered(30 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is None:
                raise RuntimeError("generated recording timed out")
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"generated recording: {error}; {debug}")
        finally:
            pipeline.set_state(Gst.State.NULL)
        with av.open(str(recording)) as source:
            video_frames = list(source.decode(video=0))
        with av.open(str(recording)) as source:
            audio_frames = list(source.decode(audio=0))
        samples = sum(frame.samples for frame in audio_frames)
        peak = max(float(np.abs(frame.to_ndarray()).max()) for frame in audio_frames)
        if len(video_frames) != 24 or samples < 48000 or peak < 0.01:
            raise RuntimeError("generated recording has missing video or silent/incomplete audio")
        print("RESULT=" + json.dumps({"status": "pass", "video_encoder": video_name,
            "audio_encoder": audio_name, "video_frames": len(video_frames),
            "audio_samples": samples, "audio_peak": peak, "file_bytes": recording.stat().st_size}))


if __name__ == "__main__":
    main()

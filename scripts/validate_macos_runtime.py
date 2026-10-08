#!/usr/bin/env python3
"""Check an installed Mac runtime with generated media, never physical capture."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import tempfile


def run_checks() -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This check requires an Apple Silicon Mac.")

    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    gi.require_version("Gst", "1.0")
    from gi.repository import Adw, Gst, Gtk
    import numpy as np
    from onnx import TensorProto, helper
    import onnxruntime as ort
    import pyvirtualcam
    import nvbroadcast
    from nvbroadcast.runtime.variants import validate_current_runtime

    Gst.init([])
    required = ("avfvideosrc", "osxaudiosrc", "osxaudiosink", "videotestsrc",
                "audiotestsrc", "videoconvert", "audioconvert", "audioresample",
                "h264parse", "mp4mux", "decodebin", "appsink", "fakesink")
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError("Missing GStreamer plugins: " + ", ".join(missing))
    video_encoder = next((name for name in ("x264enc", "openh264enc")
                          if Gst.ElementFactory.find(name)), None)
    audio_encoder = next((name for name in ("avenc_aac", "voaacenc", "fdkaacenc")
                          if Gst.ElementFactory.find(name)), None)
    if not video_encoder or not audio_encoder:
        raise RuntimeError("A working H.264 and AAC encoder is required.")
    ownership_problems = validate_current_runtime("cpu")
    if ownership_problems:
        raise RuntimeError("Invalid CPU runtime ownership: " + "; ".join(ownership_problems))
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "input"], ["output"])],
        "cpu-check",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": np.array([1, 2, 3, 4], dtype=np.float32)})[0]
    np.testing.assert_array_equal(result, [1, 4, 9, 16])

    def complete(pipeline, label):
        try:
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(f"{label} could not start.")
            message = pipeline.get_bus().timed_pop_filtered(
                30 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is None:
                raise RuntimeError(f"{label} timed out.")
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"{label}: {error.message}; {debug}")
        finally:
            pipeline.set_state(Gst.State.NULL)

    with tempfile.TemporaryDirectory(prefix="nvbroadcast-mac-generated-") as tmp:
        recording = Path(tmp) / "generated.mp4"
        encode = Gst.parse_launch(
            f'mp4mux name=mux ! filesink location="{recording}" '
            'videotestsrc num-buffers=24 ! '
            'video/x-raw,width=320,height=240,framerate=24/1 ! videoconvert ! '
            f'{video_encoder} ! h264parse ! queue ! mux. '
            'audiotestsrc num-buffers=48 samplesperbuffer=1024 ! '
            'audio/x-raw,rate=48000,channels=1 ! audioconvert ! audioresample ! '
            f'{audio_encoder} ! queue ! mux.'
        )
        complete(encode, "Generated recording")
        decode = Gst.parse_launch(
            f'filesrc location="{recording}" ! decodebin name=decoder '
            'decoder. ! queue ! video/x-raw ! fakesink name=video sync=false signal-handoffs=true '
            'decoder. ! queue ! audio/x-raw ! fakesink name=audio sync=false signal-handoffs=true'
        )
        buffers = {"video": 0, "audio": 0}
        def count(_sink, _buffer, _pad, kind):
            buffers[kind] += 1
        for kind in buffers:
            decode.get_by_name(kind).connect("handoff", count, kind)
        complete(decode, "Generated recording decode")
        if buffers["video"] != 24 or buffers["audio"] <= 0:
            raise RuntimeError(f"Incomplete audiovisual recording: {buffers}.")
        digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    return {
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": sys.version, "interpreter": sys.executable,
        "application_source": str(Path(nvbroadcast.__file__).resolve()),
        "gtk": Gtk.get_major_version(), "adwaita": Adw.get_major_version(),
        "gstreamer": Gst.version_string(), "onnxruntime": ort.__version__,
        "providers": session.get_providers(), "cpu_result": result.tolist(),
        "obs_api": str(pyvirtualcam.PixelFormat.BGR),
        "encoders": {"video": video_encoder, "audio": audio_encoder},
        "generated_recording": {"decoded_buffers": buffers, "sha256": digest},
        "physical_devices_started": False,
        "limits": ["No camera/microphone permission or physical capture test",
                   "No OBS virtual output, live effects, CoreML execution or app FPS test"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Installed Mac runtime passed CPU inference and generated audiovisual recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

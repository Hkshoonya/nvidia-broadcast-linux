#!/usr/bin/env python3
"""Check an installed Mac runtime with generated media, never physical capture."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time


def _decoder_factories(Gst, pipeline):
    records = []
    iterator = pipeline.iterate_recurse()
    for _ in range(1000):
        status, element = iterator.next()
        if status == Gst.IteratorResult.DONE:
            return sorted(records, key=lambda item: item["element"])
        if status == Gst.IteratorResult.RESYNC:
            iterator.resync()
            records = []
            continue
        if status != Gst.IteratorResult.OK:
            raise RuntimeError("Could not inspect the selected decoder elements.")
        factory = element.get_factory()
        if factory and "Decoder" in (factory.get_metadata("klass") or ""):
            caps = {}
            for name in ("sink", "src"):
                pad = element.get_static_pad(name)
                current = pad.get_current_caps() if pad else None
                caps[name] = current.to_string() if current else None
            records.append({"element": element.get_name(), "factory": factory.get_name(),
                            "plugin": factory.get_plugin_name(), "rank": int(factory.get_rank()),
                            "caps": caps})
    raise RuntimeError("Decoder element inspection exceeded its bound.")


def _complete(Gst, pipeline, label, buffers=None):
    report = {"status": "fail", "label": label, "warnings": []}
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"{label} could not start.")
        deadline = time.monotonic() + 30
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{label} timed out.")
            message = pipeline.get_bus().timed_pop_filtered(
                int(remaining * Gst.SECOND),
                Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if message is None:
                raise RuntimeError(f"{label} timed out.")
            if message.type == Gst.MessageType.WARNING:
                warning, debug = message.parse_warning()
                if len(report["warnings"]) < 20:
                    report["warnings"].append({"message": warning.message, "debug": debug})
                continue
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"{label}: {error.message}; {debug}")
            report.update(status="pass", terminal_message="eos")
            break
    except Exception as error:
        report["error"] = str(error)
    finally:
        try:
            # Inspect before NULL removes decodebin's dynamically selected elements.
            report["decoder_factories"] = _decoder_factories(Gst, pipeline)
            report["sink_caps"] = {}
            for kind in ("video", "audio"):
                sink = pipeline.get_by_name(kind)
                caps = sink.get_static_pad("sink").get_current_caps() if sink else None
                report["sink_caps"][kind] = caps.to_string() if caps else None
        except Exception as error:
            report.update(status="fail", diagnostics_error=str(error))
        finally:
            pipeline.set_state(Gst.State.NULL)
        if buffers is not None:
            report["decoded_buffers"] = dict(buffers)
    return report


def _decode_recording(Gst, recording, *, software):
    if software:
        description = (
            f'filesrc location="{recording}" ! qtdemux name=demux '
            'demux.video_0 ! queue ! h264parse ! avdec_h264 ! '
            'video/x-raw ! fakesink name=video sync=false signal-handoffs=true '
            'demux.audio_0 ! queue ! aacparse ! avdec_aac ! '
            'audio/x-raw ! fakesink name=audio sync=false signal-handoffs=true')
    else:
        description = (
            f'filesrc location="{recording}" ! decodebin name=decoder '
            'decoder. ! queue ! video/x-raw ! fakesink name=video sync=false signal-handoffs=true '
            'decoder. ! queue ! audio/x-raw ! fakesink name=audio sync=false signal-handoffs=true')
    try:
        pipeline = Gst.parse_launch(description)
    except Exception as error:
        return {"status": "fail", "error": str(error), "decoder_factories": []}
    buffers = {"video": 0, "audio": 0}
    def count(_sink, _buffer, _pad, kind):
        buffers[kind] += 1
    for kind in buffers:
        pipeline.get_by_name(kind).connect("handoff", count, kind)
    return _complete(Gst, pipeline, "Software comparison" if software else "Automatic decode", buffers)


def require_complete_decode(report):
    buffers = report.get("decoded_buffers", {})
    if report.get("status") != "pass" or buffers.get("video") != 24 or buffers.get("audio", 0) <= 0:
        raise RuntimeError(f"Incomplete audiovisual recording: {report}.")


@contextmanager
def _diagnostic_directory(root):
    if root is None:
        # Ordinary setup/verification must not accumulate synthetic recordings.
        with tempfile.TemporaryDirectory(prefix="nvbroadcast-mac-check-") as temporary:
            yield Path(temporary)
        return
    root = Path(root)
    # The caller chooses the output parent; refuse a redirected diagnostics child.
    # Resolving the parent permits macOS's normal /var -> /private/var alias.
    root = root.parent.resolve(strict=True) / root.name
    if root.is_symlink():
        raise ValueError("Media diagnostics directory must not be a symlink.")
    root.mkdir(mode=0o700, exist_ok=True)
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("Media diagnostics directory is not contained in its output parent.")
    yield Path(tempfile.mkdtemp(prefix="attempt-", dir=root))


def check_generated_media(Gst, video_encoder, audio_encoder, diagnostics_root=None, *, diagnostic_repeats=1):
    if not 1 <= diagnostic_repeats <= 5:
        raise ValueError("Diagnostic decode repetitions must be between 1 and 5.")
    with _diagnostic_directory(diagnostics_root) as directory:
        recording = directory / "generated.mp4"
        evidence = {"status": "fail", "physical_devices_started": False,
                    "acceptance": "Automatic decode must produce exactly 24 video buffers and nonzero audio; software comparison cannot rescue failure."}
        try:
            encode = Gst.parse_launch(
                f'mp4mux name=mux ! filesink location="{recording}" '
                'videotestsrc num-buffers=24 ! '
                # Match the production CPU recording branch's 8-bit 4:2:0 input.
                'video/x-raw,width=320,height=240,framerate=24/1 ! videoconvert ! '
                'video/x-raw,format=I420 ! '
                f'{video_encoder} ! h264parse ! queue ! mux. '
                'audiotestsrc num-buffers=48 samplesperbuffer=1024 ! '
                'audio/x-raw,rate=48000,channels=1 ! audioconvert ! audioresample ! '
                f'{audio_encoder} ! queue ! mux.')
            evidence["encode"] = _complete(Gst, encode, "Generated recording")
            if evidence["encode"]["status"] != "pass":
                raise RuntimeError(f"Generated recording failed: {evidence['encode']}")
            digest = hashlib.sha256(recording.read_bytes()).hexdigest()
            evidence["recording"] = {"file": recording.name, "sha256": digest, "bytes": recording.stat().st_size}
            automatic = [_decode_recording(Gst, recording, software=False) for _ in range(diagnostic_repeats)]
            evidence["automatic"] = automatic[0]
            evidence["automatic_repeats"] = automatic
            failed = any(item.get("status") != "pass" or item.get("decoded_buffers", {}).get("video") != 24
                         or item.get("decoded_buffers", {}).get("audio", 0) <= 0 for item in automatic)
            compare_software = failed or diagnostic_repeats > 1
            missing = [name for name in ("qtdemux", "h264parse", "aacparse", "avdec_h264", "avdec_aac")
                       if Gst.ElementFactory.find(name) is None] if compare_software else []
            evidence["software"] = ({"status": "not-requested"} if not compare_software else
                                    {"status": "unavailable", "missing_plugins": missing} if missing else
                                    _decode_recording(Gst, recording, software=True))
            if hashlib.sha256(recording.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Generated file changed between decoder checks.")
            for report in automatic:
                require_complete_decode(report)
            if diagnostic_repeats > 1 and missing:
                raise RuntimeError("Software comparison plugins are unavailable: " + ", ".join(missing))
            if compare_software and not missing:
                require_complete_decode(evidence["software"])
            evidence["status"] = "pass"
            return {"decoded_buffers": evidence["automatic"]["decoded_buffers"], "sha256": digest,
                    "decoder_diagnostics": {"retained": diagnostics_root is not None,
                        "directory": directory.name if diagnostics_root is not None else None,
                        "automatic": evidence["automatic"], "automatic_repeats": automatic,
                        "software": evidence["software"]}}
        except Exception as error:
            evidence["error"] = str(error)
            raise
        finally:
            (directory / "diagnostics.json").write_text(json.dumps(evidence, indent=2) + "\n")
            if diagnostics_root is not None:
                print("Generated-media diagnostics: " + str(directory), flush=True)


def run_checks(diagnostics_root: Path | None = None, *, diagnostic_repeats=1) -> dict:
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

    generated = check_generated_media(Gst, video_encoder, audio_encoder, diagnostics_root,
                                      diagnostic_repeats=diagnostic_repeats)
    return {
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": sys.version, "interpreter": sys.executable,
        "application_source": str(Path(nvbroadcast.__file__).resolve()),
        "gtk": Gtk.get_major_version(), "adwaita": Adw.get_major_version(),
        "gstreamer": Gst.version_string(), "onnxruntime": ort.__version__,
        "providers": session.get_providers(), "cpu_result": result.tolist(),
        "obs_api": str(pyvirtualcam.PixelFormat.BGR),
        "encoders": {"video": video_encoder, "audio": audio_encoder},
        "generated_recording": generated,
        "physical_devices_started": False,
        "limits": ["No camera/microphone permission or physical capture test",
                   "No OBS virtual output, live effects, CoreML execution or app FPS test"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-repeats", type=int, choices=range(1, 6), default=1,
                        help="Repeat automatic decoding of one synthetic file; values above 1 also compare software decoding.")
    args = parser.parse_args()
    diagnostics_root = (args.output.parent / "generated-media-diagnostics"
                        if os.environ.get("NVBROADCAST_RETAIN_MEDIA_DIAGNOSTICS") == "1" else None)
    result = run_checks(diagnostics_root,
                        diagnostic_repeats=args.diagnostic_repeats)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Installed Mac runtime passed CPU inference and generated audiovisual recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

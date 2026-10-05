#!/usr/bin/env python3
"""Probe the complete private CPU runtime; run its Python with -I -B."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib


def inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def probe(root: Path, version: str, window: bool, lock: Path) -> dict:
    root = root.resolve(strict=True)
    assert sys.flags.isolated and sys.flags.no_user_site
    assert Path(sys.prefix).resolve() == root
    assert Path(sys.base_prefix).resolve() == root
    assert sys.version.split()[0] == version, sys.version
    assert all(inside(Path(p), root) for p in sys.path), sys.path
    module_paths = {}
    for name in ("ssl", "sqlite3", "lzma", "bz2", "cairo", "gi", "gi._gi",
                 "numpy", "cv2", "mediapipe", "onnxruntime", "pyrnnoise",
                 "faster_whisper", "sounddevice", "soundfile", "av", "nvbroadcast.app"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert inside(path, root), (name, path)
        module_paths[name] = str(path)

    from nvbroadcast.runtime.artifact import ArtifactEnvironment
    from nvbroadcast.runtime.variants import detect_runtime_variant, RuntimeVariant
    environment = ArtifactEnvironment.current()
    problems = environment.dependency_closure_problems()
    assert not problems, problems
    assert detect_runtime_variant() == RuntimeVariant.CPU
    assert environment.installed.get("onnxruntime") == ("1.24.4",)
    assert "onnxruntime-gpu" not in environment.installed
    assert all(len(versions) == 1 for versions in environment.installed.values())
    expected = {p["name"]: (p["version"],) for p in tomllib.loads(lock.read_text())["packages"]}
    # pip belongs to the pinned interpreter archive, not the application lock.
    assert {k: v for k, v in environment.installed.items() if k != "pip"} == expected
    from nvbroadcast.core.dependency_installer import _running_in_native_package
    assert _running_in_native_package(), "GUI must retain package ownership guards"
    import ssl
    import sqlite3
    context = ssl.create_default_context()
    trust = ssl.get_default_verify_paths()
    # OpenSSL lazily reads hashed CA directories during a handshake. An empty
    # get_ca_certs() result alone is therefore not evidence of broken trust.
    hashed_certificates = list(Path(trust.capath).glob("????????.[0-9]")) if trust.capath else []
    assert context.get_ca_certs() or any(p.is_file() for p in hashed_certificates), trust
    with sqlite3.connect(":memory:") as connection:
        assert connection.execute("select 1 + 2").fetchone() == (3,)

    from nvbroadcast.core.resources import find_app_icon, find_bundled_backgrounds, find_ui_css
    resources = [find_app_icon(), find_ui_css(), *find_bundled_backgrounds()]
    assert len(resources) == 5 and all(p and inside(p, root) for p in resources), resources
    from nvbroadcast.video.effects import VideoEffects
    effects = VideoEffects(compositing="cupy")
    assert effects._compositing == "cpu" and effects._cpu_inference
    executed = subprocess.run([sys.executable, "-I", "-B", "-m", "nvbroadcast.runtime", "--variant", "cpu"],
                              capture_output=True, text=True, timeout=45)
    assert executed.returncode == 0, executed.stdout + executed.stderr

    import gi
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    gi.require_version("Gst", "1.0")
    from gi.repository import Adw, GLib, Gst, Gtk
    Gst.init([])
    from nvbroadcast.video.pipeline import VideoPipeline
    from nvbroadcast.audio.pipeline import AudioPipeline
    VideoPipeline()
    AudioPipeline()
    for graph in ("videotestsrc num-buffers=3 ! videoconvert ! fakesink",
                  "audiotestsrc num-buffers=3 ! audioconvert ! audioresample ! pulsesink sync=false"):
        pipeline = Gst.parse_launch(graph)
        try:
            assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
            message = pipeline.get_bus().timed_pop_filtered(10 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            assert message is not None and message.type == Gst.MessageType.EOS, graph
        finally:
            pipeline.set_state(Gst.State.NULL)

    mapped = False
    if window:
        Gtk.init()
        widget = Gtk.Window(title="Private runtime feasibility probe")
        widget.set_child(Gtk.Label(label="Private Python / GTK / GStreamer"))
        widget.present()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not widget.get_mapped():
            while GLib.MainContext.default().pending():
                GLib.MainContext.default().iteration(False)
            time.sleep(0.01)
        mapped = widget.get_mapped()
        widget.destroy()
        assert mapped, "GTK window did not map"

    app = Path(importlib.import_module("nvbroadcast").__file__).parent
    source_hashes = {str(p.relative_to(app)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(app.rglob("*.py"))}
    os_release = platform.freedesktop_os_release()
    return {"status": "pass", "os": os_release.get("ID"), "os_version": os_release.get("VERSION_ID"),
            "glibc": platform.libc_ver()[1], "python": sys.version.split()[0], "runtime_prefix": str(root),
            "python_paths": sys.path, "module_paths": module_paths,
            "distributions": {d.metadata["Name"]: d.version for d in metadata.distributions()},
            "gtk": f"{Gtk.get_major_version()}.{Gtk.get_minor_version()}.{Gtk.get_micro_version()}",
            "adwaita": f"{Adw.get_major_version()}.{Adw.get_minor_version()}.{Adw.get_micro_version()}",
            "gstreamer": Gst.version_string(), "video_and_audio_pipelines": "eos", "gtk_window_mapped": mapped,
            "cpu_execution": executed.stdout.strip(), "app_python_hashes": source_hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--lock", type=Path,
                        default=Path(__file__).with_name("pylock.linux-x86_64-cp313-cpu.toml"))
    args = parser.parse_args()
    print("RESULT=" + json.dumps(probe(args.runtime, args.python_version, args.window, args.lock), sort_keys=True))


if __name__ == "__main__":
    main()

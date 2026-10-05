#!/usr/bin/env python3
"""Probe a complete private runtime; run its Python with -I -B."""

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

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def closure_problems(environment, variant: str) -> list[str]:
    """Limit the CUDA substitution to faster-whisper's single CPU ORT edge."""
    if variant == "cpu":
        return environment.dependency_closure_problems()
    unexpected = []
    for distribution in environment.distributions:
        owner = canonicalize_name(distribution.metadata["Name"])
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate(environment.markers):
                continue
            if canonicalize_name(requirement.name) == "onnxruntime" and owner != "faster-whisper":
                unexpected.append(f"unreviewed CPU ORT dependency: {owner}: {raw}")
    return unexpected + environment.dependency_closure_problems(
        substitutions={"onnxruntime": "onnxruntime-gpu"})


def probe(root: Path, version: str, window: bool, lock: Path, variant: str = "cpu",
          cuda_unavailable: bool = False) -> dict:
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
    problems = closure_problems(environment, variant)
    assert not problems, problems
    assert detect_runtime_variant() == RuntimeVariant(variant)
    owner, excluded = (("onnxruntime", "onnxruntime-gpu") if variant == "cpu" else
                       ("onnxruntime-gpu", "onnxruntime"))
    assert environment.installed.get(owner) == ("1.24.4",)
    assert excluded not in environment.installed
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
    effects = VideoEffects(compositing="cupy" if variant == "cpu" else "cpu")
    assert effects._compositing == "cpu" and effects._cpu_inference
    from nvbroadcast.runtime.probe import ProbeProvider, probe_execution_provider
    cpu = probe_execution_provider(ProbeProvider.CPU, use_cache=False)
    assert cpu.success, cpu.failure_detail
    cuda = None
    cupy_execution = None
    if variant == "cuda":
        cuda = probe_execution_provider(ProbeProvider.CUDA, use_cache=False)
        assert cuda.success != cuda_unavailable, cuda.failure_detail
        if cuda.success:
            import cupy as cp
            assert inside(Path(cp.__file__), root)
            # Force a fresh compiled kernel, not just import/provider presence.
            values = cp.arange(16, dtype=cp.float32)
            squared = cp.asnumpy(values * values).tolist()
            assert squared == [float(i * i) for i in range(16)], squared
            cupy_execution = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                              "output": squared}
            cp.get_default_memory_pool().free_all_blocks()

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
            "variant": variant, "cpu_execution": cpu.to_payload(),
            "cuda_execution": cuda.to_payload() if cuda else None,
            "cuda_unavailable_expected": cuda_unavailable, "cupy_execution": cupy_execution,
            "app_python_hashes": source_hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-unavailable", action="store_true",
                        help="require CUDA rejection, used only in containers with no GPU device")
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if args.cuda_unavailable and args.variant != "cuda":
        parser.error("--cuda-unavailable requires --variant cuda")
    lock = args.lock or Path(__file__).with_name(f"pylock.linux-x86_64-cp313-{args.variant}.toml")
    print("RESULT=" + json.dumps(probe(args.runtime, args.python_version, args.window, lock,
                                      args.variant, args.cuda_unavailable), sort_keys=True))


if __name__ == "__main__":
    main()

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Isolated Linux audio service for the processed virtual microphone."""

import argparse
import base64
import json
import signal
import threading
import time

from nvbroadcast.audio.pipeline import AudioPipeline
from nvbroadcast.audio.voice_fx import VoiceFXSettings


def _decode_state(payload: str) -> dict:
    raw = base64.urlsafe_b64decode(payload.encode("ascii"))
    return json.loads(raw.decode("utf-8"))


def _build_pipeline(state: dict) -> AudioPipeline:
    pipeline = AudioPipeline(manage_virtual_mic=False, use_helper_process=False)
    pipeline.configure(
        mic_device=state.get("mic_device", ""),
        sample_rate=int(state.get("sample_rate", 48000)),
    )
    pipeline.effects.enabled = bool(state.get("noise_removal", False))
    pipeline.effects.intensity = float(state.get("noise_intensity", 1.0))
    pipeline.voice_fx.enabled = bool(state.get("voice_fx_enabled", False))
    pipeline.voice_fx.use_gpu = bool(state.get("voice_fx_use_gpu", True))

    settings = state.get("voice_fx_settings", {})
    pipeline.voice_fx.settings = VoiceFXSettings(
        bass_boost=float(settings.get("bass_boost", 0.0)),
        treble=float(settings.get("treble", 0.0)),
        warmth=float(settings.get("warmth", 0.0)),
        compression=float(settings.get("compression", 0.0)),
        gate_threshold=float(settings.get("gate_threshold", 0.0)),
        gain=float(settings.get("gain", 0.0)),
    )
    pipeline.build()
    return pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the isolated nvbroadcast audio worker")
    parser.add_argument("--state-b64", required=True, help="Base64-encoded JSON pipeline state")
    args = parser.parse_args(argv)

    stop_event = threading.Event()

    def _handle_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    pipeline = _build_pipeline(_decode_state(args.state_b64))
    pipeline.start()
    if not pipeline._running:
        return 1

    try:
        while not stop_event.wait(0.2):
            pass
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

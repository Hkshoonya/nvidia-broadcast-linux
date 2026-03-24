# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Audio device enumeration — list available microphones and speakers."""

import subprocess
import json


def list_microphones() -> list[dict[str, str]]:
    """List available microphone devices via PipeWire/PulseAudio.

    Returns list of {"name": "Display Name", "device": "device_id"}.
    """
    mics = []

    # Try PipeWire first (pw-dump)
    try:
        result = subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)
        for node in data:
            if node.get("type") != "PipeWire:Interface:Node":
                continue
            props = node.get("info", {}).get("props", {})
            media_class = props.get("media.class", "")
            if media_class in ("Audio/Source", "Audio/Source/Virtual"):
                name = props.get("node.description", props.get("node.name", "Unknown"))
                device_id = str(node.get("id", ""))
                # Skip our own virtual mic
                if "nvbroadcast" in name.lower():
                    continue
                mics.append({"name": name, "device": device_id})
    except Exception:
        pass

    # Fallback: pactl
    if not mics:
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    device_id = parts[1]
                    if "monitor" not in device_id.lower():
                        name = device_id.replace(".", " ").replace("_", " ")
                        mics.append({"name": name, "device": device_id})
        except Exception:
            pass

    if not mics:
        mics.append({"name": "Default Microphone", "device": ""})

    return mics


def list_speakers() -> list[dict[str, str]]:
    """List available speaker/output devices."""
    speakers = []
    try:
        result = subprocess.run(
            ["pactl", "list", "sinks", "short"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                device_id = parts[1]
                name = device_id.replace(".", " ").replace("_", " ")
                speakers.append({"name": name, "device": device_id})
    except Exception:
        speakers.append({"name": "Default Speaker", "device": ""})

    return speakers

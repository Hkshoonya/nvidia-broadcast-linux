# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Audio device enumeration — list available microphones and speakers."""

import subprocess
import json

from nvbroadcast.audio.virtual_mic import VIRTUAL_MIC_SINK_NAME


def _dedupe_devices(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Preserve order while removing duplicate device/name pairs."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for entry in entries:
        key = (entry.get("device", ""), entry.get("name", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _pw_nodes() -> list[dict]:
    try:
        result = subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=5
        )
        return json.loads(result.stdout)
    except Exception:
        return []


def _pactl_short(kind: str) -> list[tuple[str, str]]:
    """Return `(index, name)` pairs from `pactl list <kind> short`."""
    try:
        result = subprocess.run(
            ["pactl", "list", kind, "short"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []

    entries: list[tuple[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            entries.append((parts[0], parts[1]))
    return entries


def default_speaker_device() -> str:
    """Return the current default speaker sink identifier, if available."""
    try:
        result = subprocess.run(
            ["pactl", "info"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Default Sink:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass

    speakers = list_speakers()
    if speakers:
        return speakers[0].get("device", "")
    return ""


def list_microphones() -> list[dict[str, str]]:
    """List available microphone devices via PipeWire/PulseAudio.

    Returns list of {"name": "Display Name", "device": "device_id"}.
    """
    mics = []

    # Try PipeWire first (pw-dump)
    try:
        for node in _pw_nodes():
            if node.get("type") != "PipeWire:Interface:Node":
                continue
            props = node.get("info", {}).get("props", {})
            media_class = props.get("media.class", "")
            if media_class in ("Audio/Source", "Audio/Source/Virtual"):
                name = props.get("node.description", props.get("node.name", "Unknown"))
                device_id = props.get("node.name", str(node.get("id", "")))
                # Skip our own virtual mic
                if "nvbroadcast" in name.lower() or device_id == "nvbroadcast_mic":
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
                    if "monitor" not in device_id.lower() and device_id != "nvbroadcast_mic":
                        name = device_id.replace(".", " ").replace("_", " ")
                        mics.append({"name": name, "device": device_id})
        except Exception:
            pass

    mics = _dedupe_devices(mics)
    if not mics:
        mics.append({"name": "Default Microphone", "device": ""})

    return mics


def list_speakers() -> list[dict[str, str]]:
    """List available speaker/output devices via PipeWire/PulseAudio."""
    speakers = []

    # Try PipeWire first
    try:
        for node in _pw_nodes():
            if node.get("type") != "PipeWire:Interface:Node":
                continue
            props = node.get("info", {}).get("props", {})
            media_class = props.get("media.class", "")
            if media_class in ("Audio/Sink", "Audio/Sink/Virtual"):
                name = props.get("node.description", props.get("node.name", "Unknown"))
                device_id = props.get("node.name", str(node.get("id", "")))
                if device_id == VIRTUAL_MIC_SINK_NAME or "nvbroadcast" in name.lower():
                    continue
                speakers.append({"name": name, "device": device_id})
    except Exception:
        pass

    # Fallback: pactl
    if not speakers:
        try:
            result = subprocess.run(
                ["pactl", "list", "sinks", "short"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    device_id = parts[1]
                    if device_id == VIRTUAL_MIC_SINK_NAME or "nvbroadcast" in device_id.lower():
                        continue
                    name = device_id.replace(".", " ").replace("_", " ")
                    speakers.append({"name": name, "device": device_id})
        except Exception:
            pass

    speakers = _dedupe_devices(speakers)
    if not speakers:
        speakers.append({"name": "Default Speaker", "device": ""})

    return speakers


def resolve_pipewire_target(device: str) -> str:
    """Resolve a saved audio device selection to a PipeWire target-object."""
    if not device:
        return ""
    for node in _pw_nodes():
        if node.get("type") != "PipeWire:Interface:Node":
            continue
        props = node.get("info", {}).get("props", {})
        if str(node.get("id", "")) == device:
            return props.get("node.name", device)
        if props.get("node.name") == device:
            return device
    return device


def resolve_speaker_sink(device: str) -> str:
    """Resolve a speaker selection to a concrete sink target."""
    resolved = device or default_speaker_device()
    return resolve_pipewire_target(resolved)


def resolve_speaker_monitor_name(device: str) -> str:
    """Resolve a speaker sink selection to its monitor source name."""
    target = resolve_speaker_sink(device)
    if not target:
        return ""

    monitor_name = f"{target}.monitor"
    for _source_id, source_name in _pactl_short("sources"):
        if source_name == monitor_name:
            return source_name
    return monitor_name


def resolve_speaker_monitor(device: str) -> str:
    """Resolve a speaker sink selection to its monitor source name."""
    monitor_name = resolve_speaker_monitor_name(device)
    if not monitor_name:
        return ""

    for source_id, source_name in _pactl_short("sources"):
        if source_name == monitor_name:
            return source_id
    return monitor_name

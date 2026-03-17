# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/doczeus)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""PipeWire virtual microphone management."""

import subprocess
import signal


_pw_loopback_process = None


def create_virtual_mic() -> bool:
    """Create a PipeWire virtual microphone source.

    Creates a loopback device that appears as "NVIDIA Broadcast Microphone"
    to all applications.
    """
    global _pw_loopback_process

    if _pw_loopback_process is not None:
        return True

    try:
        _pw_loopback_process = subprocess.Popen(
            [
                "pw-loopback",
                "--capture-props",
                'media.class=Audio/Sink '
                'node.name=nvbroadcast_sink '
                'node.description="NVIDIA Broadcast Audio Input"',
                "--playback-props",
                'media.class=Audio/Source '
                'node.name=nvbroadcast_mic '
                'node.description="NVIDIA Broadcast Microphone"',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[NVIDIA Broadcast] Virtual microphone created (NVIDIA Broadcast Microphone)")
        return True
    except FileNotFoundError:
        print("[NVIDIA Broadcast] pw-loopback not found. Install pipewire.")
        return False
    except Exception as e:
        print(f"[NVIDIA Broadcast] Failed to create virtual mic: {e}")
        return False


def destroy_virtual_mic():
    """Remove the virtual microphone."""
    global _pw_loopback_process

    if _pw_loopback_process is not None:
        _pw_loopback_process.send_signal(signal.SIGTERM)
        _pw_loopback_process.wait(timeout=5)
        _pw_loopback_process = None
        print("[NVIDIA Broadcast] Virtual microphone removed")


def is_virtual_mic_active() -> bool:
    """Check if the virtual microphone is running."""
    return _pw_loopback_process is not None and _pw_loopback_process.poll() is None


def list_audio_devices() -> list[dict[str, str]]:
    """List available audio input devices via PipeWire."""
    devices = []
    try:
        result = subprocess.run(
            ["pw-cli", "list-objects", "Node"],
            capture_output=True,
            text=True,
        )
        # Parse pw-cli output for audio sources
        current = {}
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "id " in line and "type PipeWire:Interface:Node" in line:
                if current.get("name") and current.get("class") == "Audio/Source":
                    devices.append(current)
                current = {}
            elif "node.name" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    current["device"] = parts[1].strip().strip('"')
            elif "node.description" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    current["name"] = parts[1].strip().strip('"')
            elif "media.class" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    current["class"] = parts[1].strip().strip('"')

        if current.get("name") and current.get("class") == "Audio/Source":
            devices.append(current)

    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return devices

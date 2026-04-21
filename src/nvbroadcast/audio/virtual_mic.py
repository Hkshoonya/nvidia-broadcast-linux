# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""Virtual microphone management.

Prefers PipeWire's Pulse-compatible virtual sink/source modules because they
behave more reliably with browsers and meeting apps on mainstream Linux
desktops. Falls back to pw-loopback when pactl is not available.
"""

import shutil
import signal
import subprocess
import time


_pw_loopback_process = None
_pulse_sink_module_id: int | None = None
_pulse_source_module_id: int | None = None

VIRTUAL_MIC_SINK_NAME = "nvbroadcast_sink"
VIRTUAL_MIC_SOURCE_NAME = "nvbroadcast_mic"
VIRTUAL_MIC_DESCRIPTION = "nvbroadcast"
VIRTUAL_MIC_INPUT_DESCRIPTION = "nvbroadcast input"


def _run_pactl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pactl", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def virtual_mic_backend() -> str:
    """Return the preferred virtual mic backend available on this system."""
    if shutil.which("pactl") is not None:
        return "pulse"
    if shutil.which("pw-loopback") is not None:
        return "pw-loopback"
    return ""


def has_virtual_mic_backend() -> bool:
    """Return whether any supported virtual mic backend is available."""
    return bool(virtual_mic_backend())


def virtual_mic_sink_name() -> str:
    """Return the sink node name that receives processed audio."""
    return VIRTUAL_MIC_SINK_NAME


def _pulse_virtual_mic_active() -> bool:
    if _pulse_sink_module_id is None or _pulse_source_module_id is None:
        return False
    result = _run_pactl(["list", "short", "sources"])
    return VIRTUAL_MIC_SOURCE_NAME in result.stdout


def _create_pulse_virtual_mic() -> bool:
    global _pulse_sink_module_id, _pulse_source_module_id

    if _pulse_virtual_mic_active():
        return True

    _pulse_sink_module_id = None
    _pulse_source_module_id = None

    sink = _run_pactl(
        [
            "load-module",
            "module-null-sink",
            f"sink_name={VIRTUAL_MIC_SINK_NAME}",
            f"sink_properties=device.description={VIRTUAL_MIC_INPUT_DESCRIPTION}",
            "rate=48000",
            "channels=2",
            "channel_map=front-left,front-right",
        ]
    )
    if sink.returncode != 0:
        print(f"[NVIDIA Broadcast] Failed to create Pulse virtual sink: {sink.stderr.strip()}")
        return False

    try:
        _pulse_sink_module_id = int(sink.stdout.strip())
    except ValueError:
        print("[NVIDIA Broadcast] Pulse sink module returned an invalid id")
        return False

    source = _run_pactl(
        [
            "load-module",
            "module-remap-source",
            f"master={VIRTUAL_MIC_SINK_NAME}.monitor",
            f"source_name={VIRTUAL_MIC_SOURCE_NAME}",
            f"source_properties=device.description={VIRTUAL_MIC_DESCRIPTION}",
            "channels=2",
            "master_channel_map=front-left,front-right",
            "channel_map=front-left,front-right",
        ]
    )
    if source.returncode != 0:
        print(f"[NVIDIA Broadcast] Failed to create Pulse virtual source: {source.stderr.strip()}")
        _destroy_pulse_virtual_mic()
        return False

    try:
        _pulse_source_module_id = int(source.stdout.strip())
    except ValueError:
        print("[NVIDIA Broadcast] Pulse source module returned an invalid id")
        _destroy_pulse_virtual_mic()
        return False

    time.sleep(0.1)
    print("[NVIDIA Broadcast] Virtual microphone created (pulse)")
    return True


def _destroy_pulse_virtual_mic():
    global _pulse_sink_module_id, _pulse_source_module_id

    for module_id in (_pulse_source_module_id, _pulse_sink_module_id):
        if module_id is None:
            continue
        _run_pactl(["unload-module", str(module_id)])
    _pulse_source_module_id = None
    _pulse_sink_module_id = None


def _create_pw_loopback_virtual_mic() -> bool:
    global _pw_loopback_process

    if _pw_loopback_process is not None and _pw_loopback_process.poll() is None:
        return True
    _pw_loopback_process = None

    try:
        _pw_loopback_process = subprocess.Popen(
            [
                "pw-loopback",
                "-c",
                "2",
                "-m",
                "[ FL, FR ]",
                "--capture-props",
                'media.class=Audio/Sink/Virtual '
                f'node.name={VIRTUAL_MIC_SINK_NAME} '
                f'node.description="{VIRTUAL_MIC_INPUT_DESCRIPTION}"',
                "--playback-props",
                'media.class=Audio/Source '
                f'node.name={VIRTUAL_MIC_SOURCE_NAME} '
                f'node.description="{VIRTUAL_MIC_DESCRIPTION}"',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.1)
        if _pw_loopback_process.poll() is not None:
            _pw_loopback_process = None
            print("[NVIDIA Broadcast] Failed to create pw-loopback virtual microphone")
            return False
        print("[NVIDIA Broadcast] Virtual microphone created (pw-loopback)")
        return True
    except FileNotFoundError:
        print("[NVIDIA Broadcast] pw-loopback not found. Install pipewire.")
        return False
    except Exception as e:
        print(f"[NVIDIA Broadcast] Failed to create pw-loopback virtual mic: {e}")
        return False


def create_virtual_mic() -> bool:
    """Create the virtual microphone source/sink pair."""
    backend = virtual_mic_backend()
    if backend == "pulse":
        return _create_pulse_virtual_mic()
    if backend == "pw-loopback":
        return _create_pw_loopback_virtual_mic()
    print("[NVIDIA Broadcast] No supported virtual mic backend found")
    return False


def destroy_virtual_mic():
    """Remove the virtual microphone."""
    global _pw_loopback_process

    # If we created Pulse modules in this process, unload them directly.
    # Avoid probing pactl first here; teardown should be deterministic and
    # should not depend on a live source query succeeding.
    if _pulse_sink_module_id is not None or _pulse_source_module_id is not None:
        _destroy_pulse_virtual_mic()
        print("[NVIDIA Broadcast] Virtual microphone removed")
        return

    if _pw_loopback_process is not None:
        if _pw_loopback_process.poll() is None:
            _pw_loopback_process.send_signal(signal.SIGTERM)
            _pw_loopback_process.wait(timeout=5)
        _pw_loopback_process = None
        print("[NVIDIA Broadcast] Virtual microphone removed")


def is_virtual_mic_active() -> bool:
    """Check if the virtual microphone is running."""
    return _pulse_virtual_mic_active() or (
        _pw_loopback_process is not None and _pw_loopback_process.poll() is None
    )


def list_audio_devices() -> list[dict[str, str]]:
    """List available audio input devices via PipeWire."""
    devices = []
    try:
        result = subprocess.run(
            ["pw-cli", "list-objects", "Node"],
            capture_output=True,
            text=True,
        )
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

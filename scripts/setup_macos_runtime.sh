#!/bin/bash
# Complete a source PKG installation as the logged-in user, never Installer root.
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
umask 077

if (( EUID == 0 )); then
    echo "[NV Broadcast] ERROR: Run runtime setup as your logged-in user, without sudo." >&2
    exit 1
fi
if [[ "$(/usr/bin/uname -s)" != "Darwin" || "$(/usr/bin/uname -m)" != "arm64" ]]; then
    echo "[NV Broadcast] ERROR: Runtime setup requires an Apple Silicon Mac." >&2
    exit 1
fi
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ID=$(<"$INSTALL_DIR/macos-package-version")
if [[ ! "$PACKAGE_ID" =~ ^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+$ ]]; then
    echo "[NV Broadcast] ERROR: Invalid package version. Reinstall the package." >&2
    exit 1
fi
RUNTIME_ID=$(<"$INSTALL_DIR/macos-runtime-id")
if [[ ! "$RUNTIME_ID" =~ ^[a-f0-9]{64}$ ]]; then
    echo "[NV Broadcast] ERROR: Invalid package runtime identity. Reinstall the package." >&2
    exit 1
fi
BREW="/opt/homebrew/bin/brew"
prerequisite_guidance() {
    echo "[NV Broadcast] Install prerequisites as your logged-in user, without sudo:" >&2
    echo "  /opt/homebrew/bin/brew install python@3.13 pygobject3 gtk4 libadwaita gstreamer" >&2
}
if [[ ! -x "$BREW" ]]; then
    echo "[NV Broadcast] ERROR: Apple Silicon Homebrew is required at /opt/homebrew." >&2
    echo "Install Homebrew from https://brew.sh, then rerun setup." >&2
    exit 1
fi
BREW_PREFIX=$("$BREW" --prefix)
if [[ "$BREW_PREFIX" != "/opt/homebrew" ]]; then
    echo "[NV Broadcast] ERROR: This package requires the Apple Silicon /opt/homebrew prefix." >&2
    exit 1
fi
export PATH="$BREW_PREFIX/bin:$PATH"
export GST_PLUGIN_PATH="$BREW_PREFIX/lib/gstreamer-1.0"
export GI_TYPELIB_PATH="$BREW_PREFIX/lib/girepository-1.0"

# Homebrew's generic python3 can be newer than our supported wheel matrix. Also
# test the actual GI extension ABI; finding a version number alone is insufficient.
PYTHON=""
for minor in 13 12 11; do
    candidate="$BREW_PREFIX/opt/python@3.$minor/bin/python3.$minor"
    if [[ ! -x "$candidate" ]]; then
        continue
    fi
    if "$candidate" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit("Python 3.11-3.13 required")
import cairo
import gi
for namespace, version in (("Gtk", "4.0"), ("Adw", "1"), ("Gst", "1.0"),
                           ("GstApp", "1.0"), ("GstVideo", "1.0")):
    gi.require_version(namespace, version)
from gi.repository import Adw, Gst, GstApp, GstVideo, Gtk
Gst.init([])
required = ("avfvideosrc", "osxaudiosrc", "osxaudiosink", "videoconvert",
            "audioconvert", "audioresample", "x264enc", "h264parse", "mp4mux")
missing = [name for name in required if Gst.ElementFactory.find(name) is None]
if Gst.DeviceProviderFactory.find("osxaudiodeviceprovider") is None:
    missing.append("osxaudiodeviceprovider")
if not any(Gst.ElementFactory.find(name) for name in ("avenc_aac", "voaacenc")):
    missing.append("avenc_aac or voaacenc")
if missing:
    raise SystemExit("Missing GStreamer plugins: " + ", ".join(missing))
PY
    then
        PYTHON="$candidate"
        break
    fi
    echo "[NV Broadcast] $candidate does not provide the required Python/GI/GStreamer stack." >&2
done
if [[ -z "$PYTHON" ]]; then
    echo "[NV Broadcast] ERROR: A supported Python with working GTK/Adw/GStreamer bindings is required." >&2
    prerequisite_guidance
    exit 1
fi

# Never delete or replace an existing user environment. A failed setup leaves no
# ready marker; the launcher cannot accidentally start a partial installation.
BASE_DIR="$HOME/Library/Application Support/NVBroadcast"
for path in "$HOME/Library" "$HOME/Library/Application Support" "$BASE_DIR"; do
    if [[ -L "$path" ]]; then
        echo "[NV Broadcast] ERROR: Refusing symlinked runtime parent: $path" >&2
        exit 1
    fi
done
mkdir -p "$BASE_DIR"
RUNTIME_DIR="$BASE_DIR/$PACKAGE_ID-$RUNTIME_ID"
if [[ -e "$RUNTIME_DIR" || -L "$RUNTIME_DIR" ]]; then
    if [[ ! -L "$RUNTIME_DIR" && -x "$RUNTIME_DIR/.venv/bin/python" && \
          -f "$RUNTIME_DIR/runtime-ready" && \
          "$(<"$RUNTIME_DIR/runtime-ready")" == "$PACKAGE_ID:$RUNTIME_ID" ]]; then
        "$RUNTIME_DIR/.venv/bin/python" -m nvbroadcast.runtime --variant cpu
        "$RUNTIME_DIR/.venv/bin/python" -m pip check
        echo "[NV Broadcast] This user's runtime is already ready. Run: nvbroadcast"
        exit 0
    fi
    echo "[NV Broadcast] ERROR: Existing runtime is incomplete or redirected: $RUNTIME_DIR" >&2
    echo "Stop your NV Broadcast processes, review/remove only this directory, then rerun setup." >&2
    exit 1
fi
mkdir "$RUNTIME_DIR"
PROJECT_DIR="$RUNTIME_DIR/install-source"
mkdir "$PROJECT_DIR"
# setuptools builds need a writable source directory; the system PKG stays
# admin-owned. Build from a per-user copy of the package's exact source inputs.
cp -R "$INSTALL_DIR/src" "$INSTALL_DIR/data" "$PROJECT_DIR/"
cp "$INSTALL_DIR/pyproject.toml" "$INSTALL_DIR/LICENSE" "$INSTALL_DIR/NOTICE" \
    "$INSTALL_DIR/README.md" "$INSTALL_DIR/CONTRIBUTORS.md" "$PROJECT_DIR/"
"$PYTHON" -m venv "$RUNTIME_DIR/.venv" --system-site-packages
RUNTIME_PYTHON="$RUNTIME_DIR/.venv/bin/python"
"$RUNTIME_PYTHON" -m pip install --upgrade "pip>=26.2" "setuptools>=83.0.0" wheel
"$RUNTIME_PYTHON" "$INSTALL_DIR/scripts/install_runtime_variant.py" \
    --project "$PROJECT_DIR" --variant cpu --meeting-backends faster
"$RUNTIME_PYTHON" -m pip check
# Recheck the binding import in the created environment before making it ready.
"$RUNTIME_PYTHON" - <<'PY'
import gi
for namespace, version in (("Gtk", "4.0"), ("Adw", "1"), ("Gst", "1.0")):
    gi.require_version(namespace, version)
from gi.repository import Adw, Gst, Gtk
Gst.init([])
PY
printf '%s\n' "$PACKAGE_ID:$RUNTIME_ID" > "$RUNTIME_DIR/runtime-ready"
echo "[NV Broadcast] Per-user CPU runtime ready. Run: nvbroadcast"

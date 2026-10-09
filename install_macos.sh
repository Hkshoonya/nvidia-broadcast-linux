#!/usr/bin/env bash
# NV Broadcast — macOS Installer
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0
#
# Installs NV Broadcast on macOS using Homebrew.
# Validates CPUExecutionProvider; CoreML acceleration is not qualified here.
# Virtual camera via pyvirtualcam + OBS Studio.

set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
umask 077

if (( EUID == 0 )); then
    echo "Error: Run the macOS source installer as your logged-in user, without sudo." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════╗"
echo "║   NV Broadcast — macOS Installer         ║"
echo "║   by doczeus | AI Powered                ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [[ "$(/usr/bin/uname -s)" != "Darwin" ]]; then
    echo -e "${RED}Error: This installer is for macOS only.${NC}"
    echo "For Linux, use: ./install.sh"
    exit 1
fi

MACOS_ARCH=$(/usr/bin/uname -m)
if [[ "$MACOS_ARCH" != "arm64" ]]; then
    echo -e "${RED}Error: NV Broadcast supports Apple Silicon Macs only.${NC}"
    echo "A secure current MediaPipe wheel is not available for Intel macOS."
    exit 1
fi

# Current Homebrew dependencies are supported on Apple Silicon macOS 15+.
# The lower wheel/PKG platform declaration does not qualify Homebrew setup.
MACOS_VERSION=$(/usr/bin/sw_vers -productVersion)
MACOS_VER="${MACOS_VERSION%%.*}"
if [[ ! "$MACOS_VER" =~ ^[0-9]+$ || "$MACOS_VER" -lt 15 ]]; then
    echo -e "${RED}Error: macOS 15 (Sequoia) or newer required for the supported Homebrew runtime.${NC}"
    exit 1
fi

echo -e "${GREEN}[1/7]${NC} Checking prerequisites..."

# Use the same native Homebrew prefix as the packaged runtime setup. An Intel
# brew or unrelated Python earlier in PATH cannot supply the required GI ABI.
BREW="/opt/homebrew/bin/brew"
if [[ ! -x "$BREW" ]]; then
    echo "Error: Apple Silicon Homebrew is required at /opt/homebrew." >&2
    echo "Install Homebrew from https://brew.sh as your logged-in user, then rerun this installer." >&2
    exit 1
fi
BREW_PREFIX=$("$BREW" --prefix)
if [[ "$BREW_PREFIX" != "/opt/homebrew" ]]; then
    echo "Error: This installer requires the Apple Silicon /opt/homebrew prefix." >&2
    exit 1
fi
export PATH="$BREW_PREFIX/bin:$PATH"
export GST_PLUGIN_PATH="$BREW_PREFIX/lib/gstreamer-1.0"
export GI_TYPELIB_PATH="$BREW_PREFIX/lib/girepository-1.0"

echo -e "  macOS: $MACOS_VERSION"
echo -e "  Arch: $MACOS_ARCH"

# ── Step 2: Install system dependencies ──────────────────────────────────────

echo ""
echo -e "${GREEN}[2/7]${NC} Installing system dependencies via Homebrew..."

# GStreamer now includes the former separate gst-plugins-* formulae.
"$BREW" install --quiet python@3.13 pygobject3 gtk4 libadwaita gstreamer

echo -e "  GStreamer, GTK4, Libadwaita installed"

# Select only a supported Homebrew interpreter with usable native bindings,
# before copying source or replacing an existing installer-owned environment.
check_desktop_stack() {
    "$1" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit("Python 3.11-3.13 required")
import ensurepip
import venv
import cairo
import gi
for namespace, version in (("Gtk", "4.0"), ("Adw", "1"), ("Gst", "1.0"),
                           ("GstApp", "1.0"), ("GstVideo", "1.0")):
    gi.require_version(namespace, version)
from gi.repository import Adw, Gst, GstApp, GstVideo, Gtk
Gst.init([])
required = ("avfvideosrc", "osxaudiosrc", "osxaudiosink", "videoconvert",
            "audioconvert", "audioresample", "x264enc", "h264parse", "aacparse", "mp4mux")
missing = [name for name in required if Gst.ElementFactory.find(name) is None]
if Gst.DeviceProviderFactory.find("osxaudiodeviceprovider") is None:
    missing.append("osxaudiodeviceprovider")
if not any(Gst.ElementFactory.find(name) for name in ("avenc_aac", "voaacenc")):
    missing.append("avenc_aac or voaacenc")
if missing:
    raise SystemExit("Missing GStreamer plugins: " + ", ".join(missing))
PY
}

PYTHON=""
for minor in 13 12 11; do
    candidate="$BREW_PREFIX/opt/python@3.$minor/bin/python3.$minor"
    if [[ -x "$candidate" ]] && check_desktop_stack "$candidate"; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "Error: A supported Python with working GTK/Adw/GStreamer bindings is required." >&2
    echo "  /opt/homebrew/bin/brew install python@3.13 pygobject3 gtk4 libadwaita gstreamer" >&2
    exit 1
fi
echo -e "  Python: $("$PYTHON" --version) ($PYTHON)"

# ── Step 3: Create Python venv ───────────────────────────────────────────────

echo ""
echo -e "${GREEN}[3/7]${NC} Setting up Python environment..."

INSTALL_DIR="$HOME/.local/share/nvbroadcast"
mkdir -p "$INSTALL_DIR"

# Resolve inputs relative to this checkout; missing required files are errors.
cp -R "$SCRIPT_DIR/src" "$SCRIPT_DIR/data" "$SCRIPT_DIR/configs" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/pyproject.toml" "$SCRIPT_DIR/LICENSE" "$SCRIPT_DIR/NOTICE" \
    "$SCRIPT_DIR/README.md" "$SCRIPT_DIR/CONTRIBUTORS.md" "$INSTALL_DIR/"
mkdir -p "$INSTALL_DIR/scripts"
cp "$SCRIPT_DIR/scripts/install_runtime_variant.py" "$INSTALL_DIR/scripts/"
mkdir -p "$INSTALL_DIR/models"
if [[ -d "$SCRIPT_DIR/models" ]]; then
    cp -R "$SCRIPT_DIR/models/." "$INSTALL_DIR/models/"
fi

# Stop old runtime before replacing installer-owned environment.
pkill -f "^${INSTALL_DIR}/venv/bin/python -m nvbroadcast( |$)" 2>/dev/null || true
# Recreate environment so CPU remains sole runtime owner.
rm -rf -- "$INSTALL_DIR/venv"
"$PYTHON" -m venv "$INSTALL_DIR/venv" --system-site-packages
RUNTIME_PYTHON="$INSTALL_DIR/venv/bin/python"

"$RUNTIME_PYTHON" -m pip install --upgrade "pip>=26.2" "setuptools>=83.0.0" wheel -q

# ── Step 4: Install pip dependencies ─────────────────────────────────────────

echo ""
echo -e "${GREEN}[4/7]${NC} Installing Python dependencies..."

"$RUNTIME_PYTHON" "$INSTALL_DIR/scripts/install_runtime_variant.py" \
    --project "$INSTALL_DIR" --variant cpu --meeting-backends faster

if "$RUNTIME_PYTHON" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info < (3, 14) else 1)
PY
then
    "$RUNTIME_PYTHON" -m pip install -q "openai-whisper>=20231117" 2>/dev/null || \
        echo -e "${YELLOW}  openai-whisper install failed; faster-whisper remains the supported local backend.${NC}"
else
    echo -e "${YELLOW}  Skipping openai-whisper on Python 3.14+; faster-whisper remains installed.${NC}"
fi

# The runtime installer checks CPU ownership and inference execution. Recheck
# dependency closure and native bindings after optional pip packages, too.
"$RUNTIME_PYTHON" -m pip check
check_desktop_stack "$RUNTIME_PYTHON"
"$RUNTIME_PYTHON" -m nvbroadcast.runtime --variant cpu

echo -e "  Python packages installed"

# ── Step 5: Create launcher ──────────────────────────────────────────────────

echo ""
echo -e "${GREEN}[5/7]${NC} Creating launcher..."

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/nvbroadcast" << 'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
if (( EUID == 0 )); then
    echo "Error: Run NV Broadcast as your logged-in user, without sudo." >&2
    exit 1
fi
INSTALL_DIR="$HOME/.local/share/nvbroadcast"

# Set GStreamer plugin path for Homebrew
export PATH="/opt/homebrew/bin:$PATH"
export GST_PLUGIN_PATH="/opt/homebrew/lib/gstreamer-1.0"
export GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0"

cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python" -m nvbroadcast "$@"
LAUNCHER
chmod +x "$HOME/.local/bin/nvbroadcast"
echo -e "  Launcher: ~/.local/bin/nvbroadcast"

# ── Step 6: Install OBS (optional, for virtual camera) ──────────────────────

echo ""
echo -e "${GREEN}[6/7]${NC} Virtual camera setup..."

OBS_AVAILABLE=false
if command -v obs &>/dev/null || [[ -d "/Applications/OBS.app" ]]; then
    OBS_AVAILABLE=true
    echo -e "  OBS Studio found"
else
    echo -e "${YELLOW}  OBS Studio not installed.${NC}"
    read -p "  Install OBS for virtual camera support? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        "$BREW" install --cask obs
        OBS_AVAILABLE=true
        echo -e "  OBS installed"
    else
        echo -e "  Skipped. Virtual camera will not be available."
        echo -e "  Install later: brew install --cask obs"
    fi
fi

if [[ "$OBS_AVAILABLE" == "true" ]]; then
    echo -e "${YELLOW}  One-time OBS setup:${NC} open OBS, start Virtual Camera,"
    echo "  stop Virtual Camera, then close OBS. NV Broadcast can then publish"
    echo "  to the 'OBS Virtual Camera' device without OBS running."
fi

# ── Step 7: Create config ───────────────────────────────────────────────────

echo ""
echo -e "${GREEN}[7/7]${NC} Creating configuration..."

CONFIG_DIR="$HOME/Library/Application Support/nvbroadcast"
mkdir -p "$CONFIG_DIR"

if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
    cat > "$CONFIG_DIR/config.toml" << 'CONFIG'
compute_gpu = 0
performance_profile = "balanced"
compositing = "cpu"
auto_start = true
minimize_on_close = true
first_run = true

[video]
camera_device = ""
width = 1280
height = 720
fps = 30
output_format = "YUY2"
model = "rvm"
quality_preset = "quality"
background_removal = false
background_mode = "blur"
background_image = ""
blur_intensity = 0.7
auto_frame = false
auto_frame_zoom = 1.5

[video.edge]
dilate_size = 3
blur_size = 5
sigmoid_strength = 14.0
sigmoid_midpoint = 0.45

[audio]
mic_device = ""
noise_removal = false
noise_intensity = 1.0
speaker_denoise = false
CONFIG
fi

echo -e "  Config: $CONFIG_DIR/config.toml"

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗"
echo "║   Installation complete!                 ║"
echo "╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Run:  nvbroadcast"
echo ""
echo "  Make sure ~/.local/bin is in your PATH:"
# Print the command literally so the user's shell expands HOME and PATH later.
# shellcheck disable=SC2016
echo '  export PATH="$HOME/.local/bin:$PATH"'
echo ""
echo -e "  ${GREEN}Apple Silicon detected${NC} — verified CPU runtime"
echo ""
echo -e "  ${YELLOW}Note:${NC} GPU modes (Killer/Zeus/DocZeus/CUDA) require"
echo "  an NVIDIA GPU and are Linux-only."
echo "  macOS uses CPU Quality/Balanced/Light modes."
echo ""

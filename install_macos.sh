#!/usr/bin/env bash
# NV Broadcast — macOS Installer
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0
#
# Installs NV Broadcast on macOS using Homebrew.
# CPU-only inference (CoreML on Apple Silicon when available).
# Virtual camera via pyvirtualcam + OBS Studio.

set -euo pipefail

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

if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This installer is for macOS only.${NC}"
    echo "For Linux, use: ./install.sh"
    exit 1
fi

# Check macOS version (need 12+ for modern GStreamer/GTK4)
MACOS_VER=$(sw_vers -productVersion | cut -d. -f1)
if [[ "$MACOS_VER" -lt 12 ]]; then
    echo -e "${RED}Error: macOS 12 (Monterey) or newer required.${NC}"
    exit 1
fi

echo -e "${GREEN}[1/7]${NC} Checking prerequisites..."

# Check Homebrew
if ! command -v brew &>/dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check Python 3.11+
PYTHON=""
for p in python3.13 python3.12 python3.11 python3; do
    if command -v "$p" &>/dev/null; then
        ver=$("$p" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 11 ]]; then
            PYTHON="$p"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo -e "${YELLOW}Python 3.11+ not found. Installing via Homebrew...${NC}"
    brew install python@3.12
    PYTHON="python3.12"
fi

echo -e "  Python: $($PYTHON --version)"
echo -e "  macOS: $(sw_vers -productVersion)"
echo -e "  Arch: $(uname -m)"

# ── Step 2: Install system dependencies ──────────────────────────────────────

echo ""
echo -e "${GREEN}[2/7]${NC} Installing system dependencies via Homebrew..."

brew install --quiet \
    gstreamer \
    gst-plugins-base \
    gst-plugins-good \
    gst-plugins-bad \
    gtk4 \
    libadwaita \
    pygobject3 \
    gobject-introspection \
    pkg-config

echo -e "  GStreamer, GTK4, Libadwaita installed"

# ── Step 3: Create Python venv ───────────────────────────────────────────────

echo ""
echo -e "${GREEN}[3/7]${NC} Setting up Python environment..."

INSTALL_DIR="$HOME/.local/share/nvbroadcast"
mkdir -p "$INSTALL_DIR"

# Copy source
cp -r src pyproject.toml requirements.txt data models configs "$INSTALL_DIR/" 2>/dev/null || true
mkdir -p "$INSTALL_DIR/models"

# Create venv
$PYTHON -m venv "$INSTALL_DIR/venv" --system-site-packages
source "$INSTALL_DIR/venv/bin/activate"

pip install --upgrade pip setuptools wheel -q

# ── Step 4: Install pip dependencies ─────────────────────────────────────────

echo ""
echo -e "${GREEN}[4/7]${NC} Installing Python dependencies..."

pip install -q \
    numpy \
    opencv-python-headless \
    onnxruntime \
    mediapipe \
    requests \
    pyvirtualcam \
    openai-whisper \
    scipy

# Try CoreML support for Apple Silicon
if [[ "$(uname -m)" == "arm64" ]]; then
    echo -e "  Apple Silicon detected — installing CoreML provider..."
    pip install -q coremltools 2>/dev/null || true
fi

echo -e "  Python packages installed"

# ── Step 5: Create launcher ──────────────────────────────────────────────────

echo ""
echo -e "${GREEN}[5/7]${NC} Creating launcher..."

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/nvbroadcast" << 'LAUNCHER'
#!/usr/bin/env bash
INSTALL_DIR="$HOME/.local/share/nvbroadcast"
source "$INSTALL_DIR/venv/bin/activate"

# Set GStreamer plugin path for Homebrew
export GST_PLUGIN_PATH="$(brew --prefix)/lib/gstreamer-1.0"
export GI_TYPELIB_PATH="$(brew --prefix)/lib/girepository-1.0"

cd "$INSTALL_DIR"
exec python -m nvbroadcast "$@"
LAUNCHER
chmod +x "$HOME/.local/bin/nvbroadcast"
echo -e "  Launcher: ~/.local/bin/nvbroadcast"

# ── Step 6: Install OBS (optional, for virtual camera) ──────────────────────

echo ""
echo -e "${GREEN}[6/7]${NC} Virtual camera setup..."

if command -v obs &>/dev/null || [[ -d "/Applications/OBS.app" ]]; then
    echo -e "  OBS Studio found — virtual camera available"
else
    echo -e "${YELLOW}  OBS Studio not installed.${NC}"
    read -p "  Install OBS for virtual camera support? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        brew install --cask obs
        echo -e "  OBS installed — virtual camera available"
    else
        echo -e "  Skipped. Virtual camera will not be available."
        echo -e "  Install later: brew install --cask obs"
    fi
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
camera_device = "0"
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
echo '  export PATH="$HOME/.local/bin:$PATH"'
echo ""
if [[ "$(uname -m)" == "arm64" ]]; then
    echo -e "  ${GREEN}Apple Silicon detected${NC} — CPU modes with CoreML acceleration"
else
    echo -e "  Intel Mac — CPU modes only"
fi
echo ""
echo -e "  ${YELLOW}Note:${NC} GPU modes (Killer/Zeus/DocZeus/CUDA) require"
echo "  an NVIDIA GPU and are Linux-only."
echo "  macOS uses CPU Quality/Balanced/Light modes."
echo ""

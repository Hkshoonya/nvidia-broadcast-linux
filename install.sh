#!/usr/bin/env bash
# NVIDIA Broadcast for Linux - Installer
# by doczeus | AI Powered
#
# Supports: Ubuntu, Debian, Pop!_OS, Linux Mint, Fedora, RHEL, CentOS,
#           Arch, Manjaro, EndeavourOS, openSUSE, Gentoo, Void, NixOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${HOME}/.local"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "========================================="
echo "  NVIDIA Broadcast for Linux"
echo "  by doczeus | AI Powered"
echo "========================================="
echo ""

# ─── Distro Detection ───────────────────────────────────────────────────────

detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO_ID="${ID}"
        DISTRO_ID_LIKE="${ID_LIKE:-}"
        DISTRO_NAME="${PRETTY_NAME:-$ID}"
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        DISTRO_ID="${DISTRIB_ID,,}"
        DISTRO_NAME="${DISTRIB_DESCRIPTION:-$DISTRIB_ID}"
    else
        DISTRO_ID="unknown"
        DISTRO_NAME="Unknown Linux"
    fi

    # Determine package manager family
    if command -v apt &>/dev/null; then
        PKG_MANAGER="apt"
    elif command -v dnf &>/dev/null; then
        PKG_MANAGER="dnf"
    elif command -v yum &>/dev/null; then
        PKG_MANAGER="yum"
    elif command -v pacman &>/dev/null; then
        PKG_MANAGER="pacman"
    elif command -v zypper &>/dev/null; then
        PKG_MANAGER="zypper"
    elif command -v emerge &>/dev/null; then
        PKG_MANAGER="portage"
    elif command -v xbps-install &>/dev/null; then
        PKG_MANAGER="xbps"
    elif command -v nix-env &>/dev/null; then
        PKG_MANAGER="nix"
    else
        PKG_MANAGER="unknown"
    fi

    echo "  Distro: $DISTRO_NAME"
    echo "  Package manager: $PKG_MANAGER"
}

# ─── Package Name Mapping ────────────────────────────────────────────────────

# Maps generic package names to distro-specific names
get_packages() {
    case "$PKG_MANAGER" in
        apt)
            # Debian, Ubuntu, Pop!_OS, Linux Mint
            PKGS_VIRTUAL_CAM="v4l-utils v4l2loopback-dkms"
            PKGS_GTK="gir1.2-gtk-4.0 gir1.2-adw-1"
            PKGS_GST="gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"
            PKGS_PYTHON="python3-gi python3-gi-cairo"
            # PipeWire: pipewire-bin (Ubuntu 24.04+) or pipewire-utils (older/Debian)
            if apt-cache show pipewire-bin &>/dev/null 2>&1; then
                PKGS_PIPEWIRE="pipewire-bin"
            elif apt-cache show pipewire-utils &>/dev/null 2>&1; then
                PKGS_PIPEWIRE="pipewire-utils"
            else
                PKGS_PIPEWIRE=""
                echo "  WARNING: pipewire package not found. Install pw-loopback manually."
            fi
            PKGS_VENV="python3-venv"
            ;;
        dnf|yum)
            # Fedora, RHEL, CentOS, Rocky, AlmaLinux
            PKGS_VIRTUAL_CAM="v4l-utils v4l2loopback"
            PKGS_GTK="gtk4-devel libadwaita-devel"
            PKGS_GST="gstreamer1-devel gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad-free"
            PKGS_PYTHON="python3-gobject python3-gobject-cairo"
            PKGS_PIPEWIRE="pipewire-utils"
            PKGS_VENV=""  # Included in python3 on Fedora
            ;;
        pacman)
            # Arch, Manjaro, EndeavourOS
            PKGS_VIRTUAL_CAM="v4l-utils v4l2loopback-dkms"
            PKGS_GTK="gtk4 libadwaita"
            PKGS_GST="gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad"
            PKGS_PYTHON="python-gobject"
            PKGS_PIPEWIRE="pipewire"
            PKGS_VENV=""  # Included in python on Arch
            ;;
        zypper)
            # openSUSE
            PKGS_VIRTUAL_CAM="v4l-utils v4l2loopback-kmp-default"
            PKGS_GTK="gtk4-devel libadwaita-devel typelib-1_0-Gtk-4_0 typelib-1_0-Adw-1"
            PKGS_GST="gstreamer-devel gstreamer-plugins-base gstreamer-plugins-good gstreamer-plugins-bad"
            PKGS_PYTHON="python3-gobject python3-gobject-cairo"
            PKGS_PIPEWIRE="pipewire-tools"
            PKGS_VENV=""
            ;;
        *)
            # Unknown — set empty and show manual instructions
            PKGS_VIRTUAL_CAM=""
            PKGS_GTK=""
            PKGS_GST=""
            PKGS_PYTHON=""
            PKGS_PIPEWIRE=""
            PKGS_VENV=""
            ;;
    esac
}

# Install packages using the detected package manager
install_packages() {
    local pkgs="$1"
    if [ -z "$pkgs" ]; then
        return
    fi

    case "$PKG_MANAGER" in
        apt)     sudo apt install -y $pkgs ;;
        dnf)     sudo dnf install -y $pkgs ;;
        yum)     sudo yum install -y $pkgs ;;
        pacman)  sudo pacman -S --noconfirm --needed $pkgs ;;
        zypper)  sudo zypper install -y $pkgs ;;
        *)
            echo "ERROR: Cannot auto-install packages with $PKG_MANAGER."
            echo "Please install manually: $pkgs"
            return 1
            ;;
    esac
}

# Check if a package is installed
is_pkg_installed() {
    local pkg="$1"
    case "$PKG_MANAGER" in
        apt)     dpkg -s "$pkg" &>/dev/null ;;
        dnf|yum) rpm -q "$pkg" &>/dev/null ;;
        pacman)  pacman -Qi "$pkg" &>/dev/null ;;
        zypper)  rpm -q "$pkg" &>/dev/null ;;
        *)       return 1 ;;
    esac
}

# ─── Pre-flight Checks ──────────────────────────────────────────────────────

echo "[Pre-flight] Checking system requirements..."

detect_distro
ERRORS=()

# Check Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    ERRORS+=("This installer only supports Linux")
fi

# Check Python 3.11+
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
        ERRORS+=("Python >= 3.11 required (found $PY_VER)")
    else
        echo "  Python $PY_VER ... OK"
    fi
else
    ERRORS+=("python3 not found")
fi

# Check python3-venv
if ! python3 -m venv --help &>/dev/null 2>&1; then
    case "$PKG_MANAGER" in
        apt)    ERRORS+=("python3-venv not found (install: sudo apt install python3.${PY_MINOR}-venv)") ;;
        dnf)    ERRORS+=("python3-venv not found (install: sudo dnf install python3-devel)") ;;
        pacman) ERRORS+=("python3-venv not found (should be included with python)") ;;
        *)      ERRORS+=("python3-venv not found") ;;
    esac
fi

# Check pip
if ! python3 -m pip --version &>/dev/null 2>&1; then
    case "$PKG_MANAGER" in
        apt)    ERRORS+=("pip not found (install: sudo apt install python3-pip)") ;;
        dnf)    ERRORS+=("pip not found (install: sudo dnf install python3-pip)") ;;
        pacman) ERRORS+=("pip not found (install: sudo pacman -S python-pip)") ;;
        *)      ERRORS+=("pip not found") ;;
    esac
fi

# Check PipeWire
if command -v pw-loopback &>/dev/null; then
    echo "  pw-loopback ... OK"
elif command -v pw-cli &>/dev/null; then
    echo "  PipeWire ... OK (pw-loopback may be in a separate package)"
else
    echo "  WARNING: PipeWire not found. Virtual microphone will not work."
fi

# Check NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  NVIDIA GPU ... OK ($GPU_NAME)"
else
    echo "  WARNING: nvidia-smi not found. GPU acceleration will not be available."
    echo "           ONNX Runtime will fall back to CPU (much slower)."
fi

# Check DKMS
if command -v dkms &>/dev/null; then
    echo "  DKMS ... OK"
else
    echo "  WARNING: dkms not found. v4l2loopback may fail to build."
    echo "           Install with your package manager: dkms"
fi

# Check kernel headers
KERNEL_VER=$(uname -r)
if [ -d "/usr/src/linux-headers-${KERNEL_VER}" ] || [ -d "/lib/modules/${KERNEL_VER}/build" ]; then
    echo "  Kernel headers ... OK"
else
    echo "  WARNING: Kernel headers for ${KERNEL_VER} may be missing."
    echo "           v4l2loopback needs them to build."
    case "$PKG_MANAGER" in
        apt)    echo "           Install: sudo apt install linux-headers-${KERNEL_VER}" ;;
        dnf)    echo "           Install: sudo dnf install kernel-devel-${KERNEL_VER}" ;;
        pacman) echo "           Install: sudo pacman -S linux-headers" ;;
        zypper) echo "           Install: sudo zypper install kernel-devel" ;;
    esac
fi

# Abort on errors
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "FATAL: Cannot continue due to missing requirements:"
    for err in "${ERRORS[@]}"; do
        echo "  - $err"
    done
    echo ""
    echo "Fix the above issues and re-run this script."
    exit 1
fi

echo ""
echo "All requirements met. Proceeding with installation..."

# ─── Step 1: System Dependencies ────────────────────────────────────────────

echo ""
echo "[1/7] Checking system packages..."

get_packages

ALL_PKGS="$PKGS_VIRTUAL_CAM $PKGS_GTK $PKGS_GST $PKGS_PYTHON $PKGS_PIPEWIRE $PKGS_VENV"

if [ "$PKG_MANAGER" = "unknown" ]; then
    echo ""
    echo "  Your package manager ($PKG_MANAGER) is not auto-supported."
    echo "  Please install these dependencies manually:"
    echo ""
    echo "  Virtual camera:  v4l-utils, v4l2loopback (DKMS)"
    echo "  GTK4 UI:         GTK4, libadwaita, GObject introspection"
    echo "  GStreamer:        gstreamer, plugins-base, plugins-good, plugins-bad"
    echo "  Python bindings: PyGObject (python-gobject / python3-gi)"
    echo "  Audio:           PipeWire with pw-loopback"
    echo ""
    echo "  After installing, re-run this script."
    echo ""
    read -rp "  Continue without system packages? [y/N] " skip_sys
    if [[ ! "$skip_sys" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    MISSING_PKGS=()
    for pkg in $ALL_PKGS; do
        if is_pkg_installed "$pkg"; then
            echo "  $pkg ... installed"
        else
            MISSING_PKGS+=("$pkg")
            echo "  $pkg ... MISSING"
        fi
    done

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo ""
        echo "Installing ${#MISSING_PKGS[@]} missing package(s)..."
        install_packages "${MISSING_PKGS[*]}"
    else
        echo "All system packages are installed."
    fi
fi

# ─── Compositing Engine Selection ────────────────────────────────────────────

echo ""
echo "[Compositing] How should blur/blend compositing run?"
echo ""

# Auto-detect available options
HAS_GL=false
HAS_NVIDIA=false
GPU_VRAM=0

if command -v nvidia-smi &>/dev/null; then
    HAS_NVIDIA=true
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
fi
if command -v gst-inspect-1.0 &>/dev/null; then
    if gst-inspect-1.0 glvideomixer &>/dev/null 2>&1 && gst-inspect-1.0 glupload &>/dev/null 2>&1; then
        HAS_GL=true
    fi
fi

echo "  Your system:"
if [ "$HAS_NVIDIA" = true ]; then
    echo "    NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) (${GPU_VRAM} MB)"
else
    echo "    NVIDIA GPU: not detected"
fi
echo "    GStreamer GL: $( [ "$HAS_GL" = true ] && echo "available" || echo "not available" )"
echo ""

echo "  1) CPU compositing (works everywhere, ~200% CPU usage)"
if [ "$HAS_GL" = true ]; then
    echo "  2) GStreamer OpenGL GPU (recommended, ~60% CPU)"
else
    echo "  2) GStreamer OpenGL GPU [not available — missing GL plugins]"
fi
if [ "$HAS_NVIDIA" = true ]; then
    echo "  3) CuPy CUDA GPU (best quality, ~30% CPU — downloads ~800MB)"
else
    echo "  3) CuPy CUDA GPU [not available — needs NVIDIA GPU]"
fi
echo ""

# Auto-determine default
DEFAULT_COMP=1
if [ "$HAS_GL" = true ]; then
    DEFAULT_COMP=2
fi

read -rp "  Select [1/2/3] (default: ${DEFAULT_COMP}): " comp_choice
comp_choice="${comp_choice:-$DEFAULT_COMP}"

case "$comp_choice" in
    3)
        if [ "$HAS_NVIDIA" != true ]; then
            echo ""
            echo "  ERROR: CuPy CUDA requires an NVIDIA GPU."
            echo "  Falling back to CPU compositing."
            COMPOSITING="cpu"
        else
            COMPOSITING="cupy"
            echo ""
            echo "  Installing CuPy CUDA compositing (~800MB download)..."
            echo "  This may take a few minutes..."
            echo ""
            CUPY_LOG=$("$VENV_DIR/bin/pip" install cupy-cuda12x nvidia-cuda-nvrtc-cu12 2>&1)
            CUPY_EXIT=$?
            if [ $CUPY_EXIT -eq 0 ]; then
                # Verify CuPy actually works
                CUPY_TEST=$("$VENV_DIR/bin/python" -c "
import cupy as cp
import numpy as np
a = np.ones((10,10), dtype=np.float32)
b = cp.asarray(a)
c = (b * 2.0).astype(cp.uint8)
print('OK')
" 2>&1)
                if [ "$CUPY_TEST" = "OK" ]; then
                    echo "  CuPy CUDA compositing installed and verified!"
                else
                    echo ""
                    echo "  WARNING: CuPy installed but CUDA kernel compilation failed."
                    echo ""
                    echo "  Common causes:"
                    echo "    - Missing NVIDIA CUDA toolkit: sudo apt install nvidia-cuda-toolkit"
                    echo "    - Driver too old: need NVIDIA driver 525+ for CUDA 12"
                    echo "    - CUDA version mismatch: pip install cupy-cuda11x (for CUDA 11)"
                    echo ""
                    echo "  Error details:"
                    echo "  $CUPY_TEST" | tail -3
                    echo ""
                    if [ "$HAS_GL" = true ]; then
                        echo "  Falling back to GStreamer OpenGL GPU compositing."
                        COMPOSITING="gstreamer_gl"
                    else
                        echo "  Falling back to CPU compositing."
                        COMPOSITING="cpu"
                    fi
                fi
            else
                echo ""
                echo "  WARNING: CuPy installation failed."
                echo ""
                echo "  Common causes:"
                echo "    - No internet connection"
                echo "    - pip version too old: $VENV_DIR/bin/pip install --upgrade pip"
                echo "    - Disk space: CuPy needs ~800MB free"
                echo "    - Python version: CuPy supports Python 3.9-3.12"
                echo ""
                echo "  Install log (last 5 lines):"
                echo "$CUPY_LOG" | tail -5
                echo ""
                echo "  To retry later: $VENV_DIR/bin/pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12"
                echo ""
                if [ "$HAS_GL" = true ]; then
                    echo "  Falling back to GStreamer OpenGL GPU compositing."
                    COMPOSITING="gstreamer_gl"
                else
                    echo "  Falling back to CPU compositing."
                    COMPOSITING="cpu"
                fi
            fi
        fi
        ;;
    2)
        if [ "$HAS_GL" = true ]; then
            COMPOSITING="gstreamer_gl"
            echo "  GStreamer OpenGL GPU compositing selected."
        else
            echo ""
            echo "  GStreamer GL plugins not available."
            echo "  To install them:"
            case "$PKG_MANAGER" in
                apt)    echo "    sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-gl" ;;
                dnf)    echo "    sudo dnf install gstreamer1-plugins-bad-free-gl" ;;
                pacman) echo "    sudo pacman -S gst-plugins-bad gst-plugin-opengl" ;;
                *)      echo "    Install GStreamer GL/OpenGL plugins for your distro" ;;
            esac
            echo ""
            echo "  Using CPU compositing instead."
            COMPOSITING="cpu"
        fi
        ;;
    *)
        COMPOSITING="cpu"
        echo "  CPU compositing selected."
        ;;
esac
echo ""
echo "  Compositing engine: $COMPOSITING"

# ─── Step 2: v4l2loopback Configuration ─────────────────────────────────────

echo ""
echo "[2/7] Configuring virtual camera (v4l2loopback)..."

V4L2_CONF="/etc/modprobe.d/nvbroadcast-v4l2loopback.conf"
V4L2_LOAD="/etc/modules-load.d/nvbroadcast-v4l2loopback.conf"

# Remove old BluCast configs if present
sudo rm -f /etc/modprobe.d/blucast-v4l2loopback.conf 2>/dev/null
sudo rm -f /etc/modules-load.d/blucast-v4l2loopback.conf 2>/dev/null

if [ ! -f "$V4L2_CONF" ]; then
    echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4' | sudo tee "$V4L2_CONF" > /dev/null
    echo "Created $V4L2_CONF"
fi

if [ ! -f "$V4L2_LOAD" ]; then
    echo "v4l2loopback" | sudo tee "$V4L2_LOAD" > /dev/null
    echo "Created $V4L2_LOAD (auto-load on boot)"
fi

if ! lsmod | grep -q v4l2loopback; then
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4 2>/dev/null || \
        echo "WARNING: Could not load v4l2loopback. You may need to reboot or install kernel headers."
else
    echo "v4l2loopback already loaded"
fi

if [ -e /dev/video10 ]; then
    echo "Virtual camera device: /dev/video10"
else
    echo "WARNING: /dev/video10 not found. You may need to reboot."
fi

# ─── Step 3: Python Environment ─────────────────────────────────────────────

echo ""
echo "[3/7] Setting up Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
    echo "Created virtual environment"
fi
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR" -q
echo "Python packages installed."

# Verify critical Python packages
echo ""
echo "Verifying Python dependencies..."
FAILED_PY=()
for mod in numpy cv2 mediapipe onnxruntime PIL; do
    if "$VENV_DIR/bin/python" -c "import $mod" 2>/dev/null; then
        echo "  $mod ... OK"
    else
        FAILED_PY+=("$mod")
        echo "  $mod ... FAILED"
    fi
done

# pyrnnoise has a different import name
if "$VENV_DIR/bin/python" -c "from pyrnnoise import rnnoise" 2>/dev/null; then
    echo "  pyrnnoise ... OK"
else
    FAILED_PY+=("pyrnnoise")
    echo "  pyrnnoise ... FAILED"
fi

if [ ${#FAILED_PY[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Some Python packages failed to import: ${FAILED_PY[*]}"
    echo "The app may not function correctly. Check errors above."
fi

# Verify GPU acceleration
echo ""
echo "Verifying GPU acceleration..."
GPU_RESULT=$("$VENV_DIR/bin/python" -c "
import onnxruntime as ort
providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in providers:
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        print('CUDA_OK')
    except:
        print('CUDA_FAIL')
else:
    print('CPU_ONLY')
" 2>/dev/null)

if [ "$GPU_RESULT" = "CUDA_OK" ]; then
    echo "  CUDA acceleration ... OK"
elif [ "$GPU_RESULT" = "CPU_ONLY" ]; then
    echo "  WARNING: CUDA not available, will run on CPU (slower)"
    echo "           Install NVIDIA CUDA toolkit for GPU acceleration"
else
    echo "  WARNING: CUDA libraries may not load correctly"
    echo "           App will fall back to CPU if needed"
fi

# Write initial config with installer choices
CONFIG_DIR="$HOME/.config/nvbroadcast"
mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    cat > "$CONFIG_DIR/config.toml" << CONF
compute_gpu = 0
performance_profile = "balanced"
compositing = "${COMPOSITING}"
auto_start = true
minimize_on_close = true
first_run = false

[video]
camera_device = "/dev/video0"
width = 1280
height = 720
fps = 30
output_format = "YUY2"
model = "rvm"
quality_preset = "balanced"
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
CONF
    echo "Initial config created with compositing=$COMPOSITING"
fi

# ─── Step 4: Create Launcher Scripts ─────────────────────────────────────────

echo ""
echo "[4/7] Creating launcher scripts..."

mkdir -p "$INSTALL_PREFIX/bin"

# Remove old BluCast launchers
rm -f "$INSTALL_PREFIX/bin/blucast" "$INSTALL_PREFIX/bin/blucast-vcam" 2>/dev/null

cat > "$INSTALL_PREFIX/bin/nvbroadcast" << 'LAUNCHER'
#!/usr/bin/env bash
NVBROADCAST_DIR="PLACEHOLDER_DIR"
exec "$NVBROADCAST_DIR/.venv/bin/python" -m nvbroadcast "$@"
LAUNCHER
sed -i "s|PLACEHOLDER_DIR|${SCRIPT_DIR}|g" "$INSTALL_PREFIX/bin/nvbroadcast"
chmod +x "$INSTALL_PREFIX/bin/nvbroadcast"

cat > "$INSTALL_PREFIX/bin/nvbroadcast-vcam" << 'LAUNCHER'
#!/usr/bin/env bash
NVBROADCAST_DIR="PLACEHOLDER_DIR"
exec "$NVBROADCAST_DIR/.venv/bin/python" -m nvbroadcast.vcam_service "$@"
LAUNCHER
sed -i "s|PLACEHOLDER_DIR|${SCRIPT_DIR}|g" "$INSTALL_PREFIX/bin/nvbroadcast-vcam"
chmod +x "$INSTALL_PREFIX/bin/nvbroadcast-vcam"

echo "Installed: $INSTALL_PREFIX/bin/nvbroadcast"
echo "Installed: $INSTALL_PREFIX/bin/nvbroadcast-vcam"

# ─── Step 5: Desktop Entry ──────────────────────────────────────────────────

echo ""
echo "[5/7] Installing desktop entry..."

mkdir -p "$INSTALL_PREFIX/share/applications"

# Remove old BluCast desktop entry
rm -f "$INSTALL_PREFIX/share/applications/com.blucast.Broadcast.desktop" 2>/dev/null

cp "$SCRIPT_DIR/data/com.doczeus.NVBroadcast.desktop" "$INSTALL_PREFIX/share/applications/"
sed -i "s|Exec=nvbroadcast|Exec=$INSTALL_PREFIX/bin/nvbroadcast|g" \
    "$INSTALL_PREFIX/share/applications/com.doczeus.NVBroadcast.desktop"

ICON_DIR="$INSTALL_PREFIX/share/icons/hicolor/scalable/apps"
mkdir -p "$ICON_DIR"
cp "$SCRIPT_DIR/data/icons/com.doczeus.NVBroadcast.svg" "$ICON_DIR/"

if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$INSTALL_PREFIX/share/applications" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache &>/dev/null; then
    gtk-update-icon-cache "$INSTALL_PREFIX/share/icons/hicolor" 2>/dev/null || true
fi

echo "Desktop entry and icon installed."

# ─── Step 6: Systemd User Service ───────────────────────────────────────────

echo ""
echo "[6/7] Installing systemd user service..."

SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

# Remove old BluCast service
rm -f "$SYSTEMD_DIR/blucast-vcam.service" 2>/dev/null

# Detect GStreamer plugin path
GST_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
if [ ! -d "$GST_PLUGIN_PATH" ]; then
    GST_PLUGIN_PATH="/usr/lib64/gstreamer-1.0"
fi
if [ ! -d "$GST_PLUGIN_PATH" ]; then
    GST_PLUGIN_PATH="/usr/lib/gstreamer-1.0"
fi

cat > "$SYSTEMD_DIR/nvbroadcast-vcam.service" << EOF
[Unit]
Description=NVIDIA Broadcast Virtual Camera Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=$INSTALL_PREFIX/bin/nvbroadcast-vcam
Restart=on-failure
RestartSec=3
Environment=GST_PLUGIN_PATH=$GST_PLUGIN_PATH

[Install]
WantedBy=graphical-session.target
EOF

if systemctl --user daemon-reload 2>/dev/null; then
    systemctl --user enable nvbroadcast-vcam.service 2>/dev/null || true
    echo "Systemd service installed and enabled (auto-starts on login)"
else
    echo "Service file installed (run 'systemctl --user daemon-reload && systemctl --user enable nvbroadcast-vcam' from your desktop session)"
fi

# ─── Step 7: Desktop Autostart ──────────────────────────────────────────────

echo ""
echo "[7/7] Setting up autostart..."

AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cat > "$AUTOSTART_DIR/com.doczeus.NVBroadcast.desktop" << EOF
[Desktop Entry]
Name=NVIDIA Broadcast
Comment=AI-powered virtual camera - by doczeus
Exec=$INSTALL_PREFIX/bin/nvbroadcast
Icon=com.doczeus.NVBroadcast
Terminal=false
Type=Application
X-GNOME-Autostart-enabled=true
Hidden=false
EOF
echo "Autostart entry installed (launches on login)"

echo ""
echo "========================================="
echo "  Installation Complete!"
echo "  NVIDIA Broadcast for Linux"
echo "========================================="
echo ""
echo "  Detected: $DISTRO_NAME ($PKG_MANAGER)"
echo ""
echo "  Setup once, forget forever:"
echo "    - App auto-starts on login"
echo "    - Closing the window minimizes to background"
echo "    - Virtual camera stays active for browsers/apps"
echo "    - All settings are remembered"
echo ""
echo "  First run:"
echo "    nvbroadcast"
echo ""
echo "  To fully quit: click the exit icon in the header bar"
echo ""

#!/usr/bin/env bash
# NVIDIA Broadcast for Linux - Installer
# by doczeus | AI Powered
#
# Supports: Ubuntu, Debian, Pop!_OS, Linux Mint, Fedora, RHEL, CentOS,
#           Arch, Manjaro, EndeavourOS, openSUSE, Gentoo, Void, NixOS
set -eE
trap 'echo ""; echo "ERROR: Installation failed at line $LINENO (exit code $?)"; echo "Please report this issue with the output above."; exit 1' ERR

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
            PKGS_TRAY="gir1.2-ayatanaappindicator3-0.1"
            PKGS_TOOLS="psmisc"  # provides fuser (camera power save)
            PKGS_PULSE="pulseaudio-utils"  # provides pactl for speaker routing
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
            PKGS_TRAY="libayatana-appindicator-gtk3"
            PKGS_TOOLS="psmisc"
            PKGS_PULSE="pulseaudio-utils"
            PKGS_PIPEWIRE="pipewire-utils"
            PKGS_VENV=""  # Included in python3 on Fedora
            ;;
        pacman)
            # Arch, Manjaro, EndeavourOS
            PKGS_VIRTUAL_CAM="v4l-utils v4l2loopback-dkms"
            PKGS_GTK="gtk4 libadwaita"
            PKGS_GST="gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad"
            PKGS_PYTHON="python-gobject"
            PKGS_TRAY="libayatana-appindicator"
            PKGS_TOOLS="psmisc"
            PKGS_PULSE="libpulse"
            PKGS_PIPEWIRE="pipewire"
            PKGS_VENV=""  # Included in python on Arch
            ;;
        zypper)
            # openSUSE
            PKGS_VIRTUAL_CAM="v4l-utils v4l2loopback-kmp-default"
            PKGS_GTK="gtk4-devel libadwaita-devel typelib-1_0-Gtk-4_0 typelib-1_0-Adw-1"
            PKGS_GST="gstreamer-devel gstreamer-plugins-base gstreamer-plugins-good gstreamer-plugins-bad"
            PKGS_PYTHON="python3-gobject python3-gobject-cairo"
            PKGS_TRAY="typelib-1_0-AyatanaAppIndicator3-0_1"
            PKGS_TOOLS="psmisc"
            PKGS_PULSE="pulseaudio-utils"
            PKGS_PIPEWIRE="pipewire-tools"
            PKGS_VENV=""
            ;;
        *)
            # Unknown — set empty and show manual instructions
            PKGS_VIRTUAL_CAM=""
            PKGS_GTK=""
            PKGS_GST=""
            PKGS_PYTHON=""
            PKGS_TRAY=""
            PKGS_TOOLS=""
            PKGS_PULSE=""
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

ALL_PKGS="$PKGS_VIRTUAL_CAM $PKGS_GTK $PKGS_GST $PKGS_PYTHON $PKGS_TRAY $PKGS_TOOLS $PKGS_PULSE $PKGS_PIPEWIRE $PKGS_VENV"

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
    echo "  System tray:     libayatana-appindicator (GTK3 AppIndicator)"
    echo "  Tools:           psmisc (fuser command for camera power save)"
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
        if ! install_packages "${MISSING_PKGS[*]}"; then
            echo "WARNING: Some system packages failed to install. The app may still work."
            echo "  Missing: ${MISSING_PKGS[*]}"
        fi
    else
        echo "All system packages are installed."
    fi
fi

# Auto-detect GPU capabilities for optional packages
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

# ─── Step 2: v4l2loopback Configuration ─────────────────────────────────────

echo ""
echo "[2/7] Configuring virtual camera (v4l2loopback)..."

V4L2_CONF="/etc/modprobe.d/nvbroadcast-v4l2loopback.conf"
V4L2_LOAD="/etc/modules-load.d/nvbroadcast-v4l2loopback.conf"

# Remove old BluCast configs if present
sudo rm -f /etc/modprobe.d/blucast-v4l2loopback.conf 2>/dev/null || true
sudo rm -f /etc/modules-load.d/blucast-v4l2loopback.conf 2>/dev/null || true

if [ ! -f "$V4L2_CONF" ]; then
    if echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4' | sudo tee "$V4L2_CONF" > /dev/null; then
        echo "Created $V4L2_CONF"
    else
        echo "WARNING: Could not create $V4L2_CONF (sudo failed). Virtual camera may not auto-load."
    fi
fi

if [ ! -f "$V4L2_LOAD" ]; then
    if echo "v4l2loopback" | sudo tee "$V4L2_LOAD" > /dev/null; then
        echo "Created $V4L2_LOAD (auto-load on boot)"
    else
        echo "WARNING: Could not create $V4L2_LOAD (sudo failed)."
    fi
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
"$VENV_DIR/bin/pip" install "$SCRIPT_DIR" -q
echo "Core packages installed."

# Verify critical Python packages
echo ""
echo "Verifying core dependencies..."
FAILED_PY=()
for mod in numpy cv2 mediapipe onnxruntime PIL psutil onnx; do
    if "$VENV_DIR/bin/python" -c "import $mod" 2>/dev/null; then
        echo "  $mod ... OK"
    else
        FAILED_PY+=("$mod")
        echo "  $mod ... FAILED"
    fi
done

if "$VENV_DIR/bin/python" -c "from pyrnnoise import rnnoise" 2>/dev/null; then
    echo "  pyrnnoise ... OK"
else
    FAILED_PY+=("pyrnnoise")
    echo "  pyrnnoise ... FAILED"
fi

if [ ${#FAILED_PY[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Some packages failed: ${FAILED_PY[*]}"
fi

# Verify GPU acceleration
echo ""
echo "Verifying GPU acceleration..."
GPU_RESULT=$("$VENV_DIR/bin/python" -c "
import onnxruntime as ort
providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in providers:
    print('CUDA_OK')
elif 'TensorrtExecutionProvider' in providers:
    print('TRT_OK')
else:
    print('CPU_ONLY')
" 2>/dev/null)

if [ "$GPU_RESULT" = "CUDA_OK" ] || [ "$GPU_RESULT" = "TRT_OK" ]; then
    echo "  CUDA acceleration ... OK"
else
    echo "  WARNING: CUDA not available, will run on CPU (slower)"
fi

# ─── Optional Packages ────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────"
echo "  Optional Packages"
echo "─────────────────────────────────────────"
echo ""
echo "  These unlock premium features. You can install them now or later."
echo "  If skipped, the app will prompt when you select a mode that needs them."
echo ""

# CuPy (GPU compositing + fused kernel)
CUPY_INSTALLED=false
if "$VENV_DIR/bin/python" -c "import cupy" 2>/dev/null; then
    echo "  [installed] CuPy CUDA — GPU compositing + fused CUDA kernel"
    CUPY_INSTALLED=true
else
    echo "  1) CuPy CUDA (~800MB) — Unlocks:"
    echo "     - Killer mode (48fps, fused CUDA kernel compositing)"
    echo "     - Zeus mode (33fps, GPU-optimized inference)"
    echo "     - DocZeus mode (full quality + 0.1ms fused compositing)"
    echo "     - GPU alpha blending (150x faster than CPU)"
    echo ""
    if [ "$HAS_NVIDIA" = true ]; then
        read -rp "  Install CuPy? [Y/n] " install_cupy
        install_cupy="${install_cupy:-Y}"
        if [[ "$install_cupy" =~ ^[Yy]$ ]]; then
            echo "  Installing CuPy (this may take a few minutes)..."
            if "$VENV_DIR/bin/pip" install cupy-cuda12x nvidia-cuda-nvrtc-cu12 -q 2>&1; then
                CUPY_TEST=$("$VENV_DIR/bin/python" -c "import cupy; a=cupy.ones(10); print('OK')" 2>&1)
                if [ "$CUPY_TEST" = "OK" ]; then
                    echo "  CuPy installed and verified!"
                    CUPY_INSTALLED=true
                else
                    echo "  WARNING: CuPy installed but verification failed."
                    echo "  You can retry later: $VENV_DIR/bin/pip install cupy-cuda12x"
                fi
            else
                echo "  WARNING: CuPy installation failed. Skipping."
                echo "  Retry later: $VENV_DIR/bin/pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12"
            fi
        else
            echo "  Skipped. Install later: $VENV_DIR/bin/pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12"
        fi
    else
        echo "  [skipped] No NVIDIA GPU detected."
    fi
fi
echo ""

# TensorRT (Zeus/Killer inference optimization)
TRT_INSTALLED=false
TRT_SUPPORTED=false
if "$VENV_DIR/bin/python" -c "import tensorrt" 2>/dev/null; then
    echo "  [installed] TensorRT — Optimized inference for Zeus/Killer modes"
    TRT_INSTALLED=true
    TRT_SUPPORTED=true
else
    echo "  2) TensorRT (~4GB) — Unlocks:"
    echo "     - Optimized model inference (future TRT engine support)"
    echo "     - Potential 2-5x inference speedup on supported models"
    echo ""
    if [ "$HAS_NVIDIA" = true ]; then
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 8 ] && [ "$PY_MINOR" -le 13 ]; then
            TRT_SUPPORTED=true
            read -rp "  Install TensorRT? [y/N] " install_trt
            install_trt="${install_trt:-N}"
            if [[ "$install_trt" =~ ^[Yy]$ ]]; then
                echo "  Installing TensorRT (this may take several minutes)..."
                if "$VENV_DIR/bin/pip" install tensorrt-cu12 onnx -q 2>&1; then
                    echo "  TensorRT installed!"
                    TRT_INSTALLED=true
                else
                    echo "  WARNING: TensorRT installation failed. Skipping."
                    echo "  Retry later: $VENV_DIR/bin/pip install tensorrt-cu12"
                fi
            else
                echo "  Skipped. Install later: $VENV_DIR/bin/pip install tensorrt-cu12"
            fi
        else
            echo "  [skipped] TensorRT wheels are not available for Python $PY_VER yet."
            echo "            Supported Python versions: 3.8-3.13"
            echo "            Use DocZeus or CUDA modes, or install Python 3.13 for TensorRT."
        fi
    else
        echo "  [skipped] No NVIDIA GPU detected."
    fi
fi
echo ""

# Summary of optional packages
echo "  Optional packages summary:"
echo "    CuPy:     $( [ "$CUPY_INSTALLED" = true ] && echo "INSTALLED (Killer/Zeus/DocZeus modes)" || echo "NOT INSTALLED (CPU modes only)" )"
if [ "$TRT_INSTALLED" = true ]; then
    echo "    TensorRT: INSTALLED (optimized inference)"
elif [ "$TRT_SUPPORTED" = true ]; then
    echo "    TensorRT: NOT INSTALLED (optional for Zeus/Killer)"
else
    echo "    TensorRT: UNSUPPORTED ON PYTHON $PY_VER (requires Python 3.8-3.13)"
fi

# Set compositing based on what's installed
if [ "$CUPY_INSTALLED" = true ]; then
    COMPOSITING="cupy"
elif [ "$HAS_GL" = true ]; then
    COMPOSITING="gstreamer_gl"
else
    COMPOSITING="cpu"
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
rm -f "$INSTALL_PREFIX/share/applications/com.blucast.Broadcast.desktop" 2>/dev/null || true

if [ -f "$SCRIPT_DIR/data/com.doczeus.NVBroadcast.desktop" ]; then
    cp "$SCRIPT_DIR/data/com.doczeus.NVBroadcast.desktop" "$INSTALL_PREFIX/share/applications/"
    sed -i "s|Exec=nvbroadcast|Exec=$INSTALL_PREFIX/bin/nvbroadcast|g" \
        "$INSTALL_PREFIX/share/applications/com.doczeus.NVBroadcast.desktop"
else
    echo "WARNING: Desktop entry file not found at $SCRIPT_DIR/data/com.doczeus.NVBroadcast.desktop"
fi

ICON_DIR="$INSTALL_PREFIX/share/icons/hicolor/scalable/apps"
mkdir -p "$ICON_DIR"
if [ -f "$SCRIPT_DIR/data/icons/com.doczeus.NVBroadcast.svg" ]; then
    cp "$SCRIPT_DIR/data/icons/com.doczeus.NVBroadcast.svg" "$ICON_DIR/"
else
    echo "WARNING: Icon file not found at $SCRIPT_DIR/data/icons/com.doczeus.NVBroadcast.svg"
fi

# Ensure icon theme index exists (needed for gtk-update-icon-cache)
if [ ! -f "$INSTALL_PREFIX/share/icons/hicolor/index.theme" ]; then
    if [ -f /usr/share/icons/hicolor/index.theme ]; then
        cp /usr/share/icons/hicolor/index.theme "$INSTALL_PREFIX/share/icons/hicolor/"
    fi
fi

if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$INSTALL_PREFIX/share/applications" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache &>/dev/null; then
    gtk-update-icon-cache "$INSTALL_PREFIX/share/icons/hicolor" 2>/dev/null || true
fi

# Ensure ~/.local/share is in XDG_DATA_DIRS so GNOME finds the .desktop file
if [[ ":${XDG_DATA_DIRS}:" != *":$INSTALL_PREFIX/share:"* ]]; then
    PROFILE_FILE="$HOME/.profile"
    if [ -f "$HOME/.bash_profile" ]; then
        PROFILE_FILE="$HOME/.bash_profile"
    fi
    if ! grep -q 'XDG_DATA_DIRS.*\.local/share' "$PROFILE_FILE" 2>/dev/null; then
        echo "" >> "$PROFILE_FILE"
        echo "# Added by NV Broadcast installer — show app in desktop menu" >> "$PROFILE_FILE"
        echo 'export XDG_DATA_DIRS="$HOME/.local/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"' >> "$PROFILE_FILE"
        echo "  Added XDG_DATA_DIRS to $PROFILE_FILE (takes effect on next login)"
    fi
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
echo "  Installation Complete! v1.1.1"
echo "  NVIDIA Broadcast for Linux"
echo "  by doczeus | AI Powered"
echo "========================================="
echo ""
echo "  System: $DISTRO_NAME ($PKG_MANAGER)"
echo "  Compositing: $COMPOSITING"
echo "  CuPy: $( [ "$CUPY_INSTALLED" = true ] && echo "YES (Killer/Zeus/DocZeus modes)" || echo "NO (install later for GPU modes)" )"
if [ "$TRT_INSTALLED" = true ]; then
    echo "  TensorRT: YES"
elif [ "$TRT_SUPPORTED" = true ]; then
    echo "  TensorRT: NO (install later for Zeus/Killer optimization)"
else
    echo "  TensorRT: UNSUPPORTED ON PYTHON $PY_VER (requires Python 3.8-3.13)"
fi
echo ""
echo "  Available modes:"
if [ "$CUPY_INSTALLED" = true ]; then
    if [ "$TRT_SUPPORTED" = true ]; then
        echo "    Killer  — 48fps fused CUDA (fastest)"
        echo "    Zeus    — 33fps GPU-optimized"
    else
        echo "    Killer  — unavailable on Python $PY_VER (TensorRT requires 3.8-3.13)"
        echo "    Zeus    — unavailable on Python $PY_VER (TensorRT requires 3.8-3.13)"
    fi
    echo "    DocZeus — 23fps full quality + fused kernel"
fi
echo "    CUDA Max/Balanced/Perf — standard GPU modes"
echo "    CPU Quality/Light/Low  — CPU fallback"
echo ""
echo "  v1.1.1 patch highlights:"
echo "    Virtual Camera Stability — safer Linux loopback sink startup"
echo "    Lower Live Lag           — shared face landmarks and ROI relighting"
echo "    Better Replace Edges     — tighter shoulders, hair, and arm gaps"
echo "    Meeting Transcription    — faster startup and cleaner saved audio"
echo "    Resolution Safety        — save changes without hanging the stream"
echo ""
echo "  To install optional packages later:"
echo "    CuPy:     $VENV_DIR/bin/pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12"
echo "    TensorRT: $VENV_DIR/bin/pip install tensorrt-cu12"
echo ""
echo "  First run:"
if [[ ":$PATH:" != *":$INSTALL_PREFIX/bin:"* ]]; then
    echo "    WARNING: $INSTALL_PREFIX/bin is not on your PATH."
    echo "    Add this to your ~/.bashrc or ~/.zshrc:"
    echo "      export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "    Then run: source ~/.bashrc"
    echo ""
    echo "    Or run directly:"
    echo "      $INSTALL_PREFIX/bin/nvbroadcast"
else
    echo "    nvbroadcast"
fi
echo ""
# Verify critical files were created
INSTALL_OK=true
for f in "$INSTALL_PREFIX/bin/nvbroadcast" \
         "$INSTALL_PREFIX/share/applications/com.doczeus.NVBroadcast.desktop" \
         "$HOME/.config/autostart/com.doczeus.NVBroadcast.desktop"; do
    if [ ! -f "$f" ]; then
        echo "  WARNING: Missing: $f"
        INSTALL_OK=false
    fi
done
if [ "$INSTALL_OK" = true ]; then
    echo "  All files installed successfully."
fi
echo ""
echo "  Setup once, forget forever."
echo ""

#!/usr/bin/env bash
# NVIDIA Broadcast for Linux - Installer
# by doczeus | AI Powered
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${HOME}/.local"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "========================================="
echo "  NVIDIA Broadcast for Linux"
echo "  by doczeus | AI Powered"
echo "========================================="
echo ""

# --- Pre-flight: System Requirements ---
echo "[Pre-flight] Checking system requirements..."

ERRORS=()

# Check Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    ERRORS+=("This installer only supports Linux")
fi

# Check Debian/Ubuntu (apt-based)
if ! command -v apt &>/dev/null; then
    ERRORS+=("apt package manager not found (requires Debian/Ubuntu-based distro)")
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

# Check python3-venv (version-specific package on Debian/Ubuntu)
if ! python3 -m venv --help &>/dev/null 2>&1; then
    ERRORS+=("python3-venv not found (install: sudo apt install python3.${PY_MINOR}-venv)")
fi

# Check pip
if ! python3 -m pip --version &>/dev/null 2>&1; then
    ERRORS+=("pip not found (install: sudo apt install python3-pip)")
fi

# Check PipeWire
if command -v pw-cli &>/dev/null; then
    echo "  PipeWire ... OK"
elif command -v pipewire &>/dev/null; then
    echo "  PipeWire ... OK (pw-cli not in PATH)"
else
    ERRORS+=("PipeWire not found (required for virtual microphone)")
fi

# Check pw-loopback
if command -v pw-loopback &>/dev/null; then
    echo "  pw-loopback ... OK"
else
    ERRORS+=("pw-loopback not found (install: sudo apt install pipewire-utils)")
fi

# Check NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  NVIDIA GPU ... OK ($GPU_NAME)"
else
    echo "  WARNING: nvidia-smi not found. GPU acceleration will not be available."
    echo "           ONNX Runtime will fall back to CPU (much slower)."
fi

# Check DKMS (needed for v4l2loopback)
if command -v dkms &>/dev/null; then
    echo "  DKMS ... OK"
else
    ERRORS+=("dkms not found (install: sudo apt install dkms)")
fi

# Check kernel headers
KERNEL_VER=$(uname -r)
if [ -d "/usr/src/linux-headers-${KERNEL_VER}" ] || [ -d "/lib/modules/${KERNEL_VER}/build" ]; then
    echo "  Kernel headers ... OK"
else
    echo "  WARNING: Kernel headers for ${KERNEL_VER} may be missing."
    echo "           v4l2loopback-dkms needs them to build. Install with:"
    echo "           sudo apt install linux-headers-${KERNEL_VER}"
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

# --- Step 1: System Dependencies ---
echo ""
echo "[1/7] Checking system packages..."

# All system packages needed by NVIDIA Broadcast
REQUIRED_PKGS=(
    # Virtual camera
    v4l-utils
    v4l2loopback-dkms
    # GTK4 / Libadwaita UI
    gir1.2-gtk-4.0
    gir1.2-adw-1
    # GStreamer
    gir1.2-gstreamer-1.0
    gir1.2-gst-plugins-base-1.0
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    # Python GObject bindings
    python3-gi
    python3-gi-cairo
    # PipeWire utilities
    pipewire-utils
)

MISSING_PKGS=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    if dpkg -s "$pkg" &>/dev/null; then
        echo "  $pkg ... installed"
    else
        MISSING_PKGS+=("$pkg")
        echo "  $pkg ... MISSING"
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo ""
    echo "Installing ${#MISSING_PKGS[@]} missing package(s)..."
    sudo apt install -y "${MISSING_PKGS[@]}"
else
    echo "All system packages are installed."
fi

# --- Step 2: v4l2loopback Configuration ---
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
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4
    echo "Loaded v4l2loopback module"
else
    echo "v4l2loopback already loaded"
fi

if [ -e /dev/video10 ]; then
    echo "Virtual camera device: /dev/video10"
else
    echo "WARNING: /dev/video10 not found. You may need to reboot."
fi

# --- Step 3: Python Environment ---
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
    # Test that CUDA actually loads
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

# --- Step 4: Create Launcher Scripts ---
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

# --- Step 5: Desktop Entry ---
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

# --- Step 6: Systemd User Service ---
echo ""
echo "[6/7] Installing systemd user service..."

SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

# Remove old BluCast service
rm -f "$SYSTEMD_DIR/blucast-vcam.service" 2>/dev/null

cat > "$SYSTEMD_DIR/nvbroadcast-vcam.service" << EOF
[Unit]
Description=NVIDIA Broadcast Virtual Camera Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=$INSTALL_PREFIX/bin/nvbroadcast-vcam
Restart=on-failure
RestartSec=3
Environment=GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0

[Install]
WantedBy=graphical-session.target
EOF

if systemctl --user daemon-reload 2>/dev/null; then
    systemctl --user enable nvbroadcast-vcam.service 2>/dev/null || true
    echo "Systemd service installed and enabled (auto-starts on login)"
else
    echo "Service file installed (run 'systemctl --user daemon-reload && systemctl --user enable nvbroadcast-vcam' from your desktop session)"
fi

# --- Step 7: Desktop autostart (GUI app) ---
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

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

# --- Step 1: System Dependencies ---
echo "[1/6] Checking system dependencies..."

MISSING_PKGS=""
for pkg in v4l-utils v4l2loopback-dkms gir1.2-gtk-4.0 gir1.2-adw-1 \
           gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
           gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
           gstreamer1.0-plugins-bad python3-gi python3-gi-cairo; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [ -n "$MISSING_PKGS" ]; then
    echo "Installing missing packages:$MISSING_PKGS"
    sudo apt install -y $MISSING_PKGS
else
    echo "All system packages are installed."
fi

# --- Step 2: v4l2loopback Configuration ---
echo ""
echo "[2/6] Configuring virtual camera (v4l2loopback)..."

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
echo "[3/6] Setting up Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
fi
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR" -q
echo "Python packages installed."

# --- Step 4: Create Launcher Scripts ---
echo ""
echo "[4/6] Creating launcher scripts..."

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
echo "[5/6] Installing desktop entry..."

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
echo "[6/6] Installing systemd user service..."

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

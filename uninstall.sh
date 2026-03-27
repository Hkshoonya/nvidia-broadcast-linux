#!/usr/bin/env bash
# NVIDIA Broadcast for Linux - Uninstaller
# by doczeus | AI Powered
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${HOME}/.local"

echo "========================================="
echo "  NVIDIA Broadcast for Linux - Uninstall"
echo "  by doczeus | AI Powered"
echo "========================================="
echo ""

# Confirm
read -rp "This will remove NVIDIA Broadcast and all its configuration. Continue? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled."
    exit 0
fi

# --- Step 1: Stop and disable systemd service ---
echo ""
echo "[1/6] Stopping systemd service..."

if systemctl --user is-active nvbroadcast-vcam.service &>/dev/null; then
    systemctl --user stop nvbroadcast-vcam.service
    echo "Stopped nvbroadcast-vcam service"
fi

if systemctl --user is-enabled nvbroadcast-vcam.service &>/dev/null; then
    systemctl --user disable nvbroadcast-vcam.service
    echo "Disabled nvbroadcast-vcam service"
fi

rm -f "$HOME/.config/systemd/user/nvbroadcast-vcam.service"
if systemctl --user daemon-reload 2>/dev/null; then
    echo "Systemd daemon reloaded"
fi

# --- Step 2: Remove autostart entry ---
echo ""
echo "[2/6] Removing autostart entry..."

rm -f "$HOME/.config/autostart/com.doczeus.NVBroadcast.desktop"
echo "Autostart entry removed"

# --- Step 3: Remove desktop entry and icon ---
echo ""
echo "[3/6] Removing desktop entry and icon..."

rm -f "$INSTALL_PREFIX/share/applications/com.doczeus.NVBroadcast.desktop"
rm -f "$INSTALL_PREFIX/share/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg"

if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$INSTALL_PREFIX/share/applications" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache &>/dev/null; then
    gtk-update-icon-cache "$INSTALL_PREFIX/share/icons/hicolor" 2>/dev/null || true
fi
echo "Desktop entry and icon removed"

# --- Step 4: Remove launcher scripts ---
echo ""
echo "[4/6] Removing launcher scripts..."

rm -f "$INSTALL_PREFIX/bin/nvbroadcast"
rm -f "$INSTALL_PREFIX/bin/nvbroadcast-vcam"
echo "Launcher scripts removed"

# --- Step 5: Preserve system-level virtual camera configuration ---
echo ""
echo "[5/6] Preserving system virtual camera configuration..."
echo "System-wide v4l2loopback config and drivers were left untouched."
echo "Remove them manually only if you are sure nothing else on the system needs them."

# --- Step 6: Remove Python virtual environment ---
echo ""
echo "[6/6] Removing Python virtual environment..."

if [ -d "$SCRIPT_DIR/.venv" ]; then
    rm -rf "$SCRIPT_DIR/.venv"
    echo "Virtual environment removed"
else
    echo "No virtual environment found, skipping"
fi

echo ""
echo "========================================="
echo "  Uninstall Complete!"
echo "  NVIDIA Broadcast for Linux"
echo "========================================="
echo ""
echo "  What was removed:"
echo "    - Systemd service (nvbroadcast-vcam)"
echo "    - Desktop autostart entry"
echo "    - Desktop menu entry and icon"
echo "    - Launcher scripts (nvbroadcast, nvbroadcast-vcam)"
echo "    - Python virtual environment"
echo ""
echo "  Preserved on purpose:"
echo "    - v4l2loopback kernel module and its package"
echo "    - /etc/modprobe.d and /etc/modules-load.d system configuration"
echo "    - Shared desktop/runtime packages (GTK4, GStreamer, PipeWire, Python GI)"
echo "    - These may be used by other applications and by your desktop session"
echo ""
echo "  The source code in $SCRIPT_DIR is untouched."
echo "  Standard installs now live inside $SCRIPT_DIR/.venv, not as an editable source link."
echo "  You can safely delete the source tree after uninstall if no longer needed."
echo ""

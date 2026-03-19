#!/usr/bin/env bash
# NVIDIA Broadcast for Linux - Uninstaller
# by doczeus | AI Powered
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${HOME}/.local"

# System packages installed by install.sh / setup_deps.sh
SYSTEM_PACKAGES=(
    v4l-utils
    v4l2loopback-dkms
    gir1.2-gtk-4.0
    gir1.2-adw-1
    gir1.2-gstreamer-1.0
    gir1.2-gst-plugins-base-1.0
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    python3-gi
    python3-gi-cairo
    libgstreamer1.0-dev
    libgstreamer-plugins-base1.0-dev
    libgtk-4-dev
    libadwaita-1-dev
    pipewire-utils
)

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

# Ask about system packages
echo ""
echo "The installer added these system packages:"
echo "  GStreamer, GTK4/Libadwaita, v4l2loopback, Python GI bindings"
echo ""
echo "WARNING: Other applications may depend on these packages."
echo "         Only remove them if you are sure nothing else needs them."
echo ""
read -rp "Also remove system packages installed by NVIDIA Broadcast? [y/N] " remove_packages
REMOVE_PACKAGES=false
if [[ "$remove_packages" =~ ^[Yy]$ ]]; then
    REMOVE_PACKAGES=true
fi

# --- Step 1: Stop and disable systemd service ---
echo ""
echo "[1/7] Stopping systemd service..."

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
echo "[2/7] Removing autostart entry..."

rm -f "$HOME/.config/autostart/com.doczeus.NVBroadcast.desktop"
echo "Autostart entry removed"

# --- Step 3: Remove desktop entry and icon ---
echo ""
echo "[3/7] Removing desktop entry and icon..."

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
echo "[4/7] Removing launcher scripts..."

rm -f "$INSTALL_PREFIX/bin/nvbroadcast"
rm -f "$INSTALL_PREFIX/bin/nvbroadcast-vcam"
echo "Launcher scripts removed"

# --- Step 5: Remove v4l2loopback configuration ---
echo ""
echo "[5/7] Removing v4l2loopback configuration..."

if [ -f /etc/modprobe.d/nvbroadcast-v4l2loopback.conf ] || [ -f /etc/modules-load.d/nvbroadcast-v4l2loopback.conf ]; then
    sudo rm -f /etc/modprobe.d/nvbroadcast-v4l2loopback.conf
    sudo rm -f /etc/modules-load.d/nvbroadcast-v4l2loopback.conf
    echo "v4l2loopback config removed (module will not auto-load on next boot)"
else
    echo "No v4l2loopback config found, skipping"
fi

# --- Step 6: Remove Python virtual environment ---
echo ""
echo "[6/7] Removing Python virtual environment..."

if [ -d "$SCRIPT_DIR/.venv" ]; then
    rm -rf "$SCRIPT_DIR/.venv"
    echo "Virtual environment removed"
else
    echo "No virtual environment found, skipping"
fi

# --- Step 7: Remove system packages (optional) ---
echo ""
echo "[7/7] System packages..."

if [ "$REMOVE_PACKAGES" = true ]; then
    # Filter to only packages that are actually installed
    INSTALLED_PKGS=()
    for pkg in "${SYSTEM_PACKAGES[@]}"; do
        if dpkg -s "$pkg" &>/dev/null; then
            INSTALLED_PKGS+=("$pkg")
        fi
    done

    if [ ${#INSTALLED_PKGS[@]} -gt 0 ]; then
        echo "Removing: ${INSTALLED_PKGS[*]}"
        sudo apt remove -y "${INSTALLED_PKGS[@]}"
        echo ""
        read -rp "Run 'apt autoremove' to clean up unused dependencies? [y/N] " do_autoremove
        if [[ "$do_autoremove" =~ ^[Yy]$ ]]; then
            sudo apt autoremove -y
        fi
        echo "System packages removed"
    else
        echo "No NVIDIA Broadcast system packages found installed, skipping"
    fi
else
    echo "Skipped (kept system packages)"
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
echo "    - v4l2loopback configuration"
echo "    - Python virtual environment"
if [ "$REMOVE_PACKAGES" = true ]; then
echo "    - System packages (GStreamer, GTK4, v4l2loopback, etc.)"
fi
echo ""
if [ "$REMOVE_PACKAGES" = false ]; then
echo "  NOT removed (you chose to keep system packages):"
echo "    - v4l2loopback-dkms, GStreamer, GTK4, etc."
echo "    - To remove later: sudo apt remove <package>"
echo ""
fi
echo "  The source code in $SCRIPT_DIR is untouched."
echo "  You can safely delete it if no longer needed."
echo ""

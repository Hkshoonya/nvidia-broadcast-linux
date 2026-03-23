#!/usr/bin/env bash
# Install system dependencies for NVIDIA Broadcast
# by doczeus | AI Powered
set -e

echo "=== NVIDIA Broadcast Dependency Installer ==="

# GStreamer packages
echo "[1/5] Installing GStreamer packages..."
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0

# GTK4 / Libadwaita
echo "[2/5] Installing GTK4/Adwaita packages..."
sudo apt install -y \
    libgtk-4-dev \
    libadwaita-1-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-4.0 \
    gir1.2-adw-1

# v4l2 / virtual camera
echo "[3/5] Installing video tools..."
sudo apt install -y \
    v4l-utils \
    v4l2loopback-dkms

# PipeWire (virtual microphone) — package name varies by distro
echo "[4/5] Installing PipeWire utilities..."
if apt-cache show pipewire-bin &>/dev/null 2>&1; then
    sudo apt install -y pipewire-bin
elif apt-cache show pipewire-utils &>/dev/null 2>&1; then
    sudo apt install -y pipewire-utils
else
    echo "WARNING: Could not find pipewire package. Install manually if pw-loopback is missing."
fi

# Python venv + packages
echo "[5/5] Setting up Python environment..."
python3 -m venv .venv --system-site-packages
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

echo ""
echo "=== Done! ==="
echo "Activate: source .venv/bin/activate"
echo "Run GUI:  python -m nvbroadcast"
echo "Run VCam: python -m nvbroadcast.vcam_service"
echo ""
echo "AI models will be auto-downloaded on first use."

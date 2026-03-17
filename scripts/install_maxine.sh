#!/usr/bin/env bash
# Download and install NVIDIA Maxine SDKs from NGC
# Requires NGC CLI or manual download from developer.nvidia.com
set -e

echo "=== NVIDIA Maxine SDK Installer ==="
echo ""
echo "The Maxine SDKs must be downloaded from NVIDIA's developer portal."
echo "Visit: https://developer.nvidia.com/maxine"
echo ""
echo "Required SDKs:"
echo "  1. Video Effects SDK (background removal, super-res)"
echo "     -> Extract to /usr/local/VideoFX/"
echo ""
echo "  2. Audio Effects SDK (noise removal, echo cancellation)"
echo "     -> Extract to /usr/local/AudioFX/"
echo ""
echo "  3. AR SDK (face/body tracking)"
echo "     -> Extract to /usr/local/ARFX/"
echo ""
echo "After downloading the tarballs, run:"
echo "  sudo tar -xzf NVIDIA_VFX_SDK_*.tar.gz -C /usr/local/"
echo "  sudo tar -xzf NVIDIA_AFX_SDK_*.tar.gz -C /usr/local/"
echo "  sudo tar -xzf NVIDIA_AR_SDK_*.tar.gz -C /usr/local/"
echo ""
echo "Then download the AI models:"
echo "  cd /usr/local/VideoFX && sudo ./download_models.sh"
echo "  cd /usr/local/AudioFX && sudo ./download_models.sh"
echo "  cd /usr/local/ARFX && sudo ./download_models.sh"
echo ""

# Check if SDKs are already installed
for sdk_path in "/usr/local/VideoFX" "/usr/local/AudioFX" "/usr/local/ARFX"; do
    if [ -d "$sdk_path" ]; then
        echo "Found: $sdk_path"
    else
        echo "Missing: $sdk_path"
    fi
done

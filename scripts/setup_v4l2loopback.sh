#!/usr/bin/env bash
# Configure v4l2loopback for NVbroadcast virtual camera
set -e

DEVICE_NUM="${NVBROADCAST_VCAM_DEVICE_NUM:-10}"
if [ -n "${NVBROADCAST_VCAM_DEVICE:-}" ]; then
    case "$NVBROADCAST_VCAM_DEVICE" in
        /dev/video*) DEVICE_NUM="${NVBROADCAST_VCAM_DEVICE#/dev/video}" ;;
        *) echo "WARNING: Ignoring invalid NVBROADCAST_VCAM_DEVICE=${NVBROADCAST_VCAM_DEVICE}; expected /dev/videoN." ;;
    esac
fi
case "$DEVICE_NUM" in
    ''|*[!0-9]*)
        echo "WARNING: Invalid virtual camera number '${DEVICE_NUM}', using 10."
        DEVICE_NUM=10
        ;;
esac
LABEL="NVbroadcast"
DEVICE="/dev/video${DEVICE_NUM}"

echo "=== NVbroadcast Virtual Camera Setup ==="

# Make persistent across reboots
CONF_FILE="/etc/modprobe.d/nvbroadcast-v4l2loopback.conf"
if [ -e "$DEVICE" ]; then
    if ! command -v v4l2-ctl &>/dev/null || \
       ! v4l2-ctl -D -d "$DEVICE" 2>/dev/null | \
           grep -Eiq 'Driver name[[:space:]]*:[[:space:]]*v4l2[[:space:]_-]*loopback'; then
        echo "ERROR: ${DEVICE} already exists and is not a v4l2loopback virtual camera."
        echo "Choose an unused video number with NVBROADCAST_VCAM_DEVICE_NUM."
        exit 1
    fi
fi

if [ ! -f "$CONF_FILE" ] || grep -Eq 'card_label="(NVIDIA Broadcast|NVIDIA Broadcast Virtual Camera|NV Broadcast|NVbroadcast)"' "$CONF_FILE"; then
    echo "Creating persistent config at ${CONF_FILE}..."
    echo "options v4l2loopback devices=1 video_nr=${DEVICE_NUM} card_label=\"${LABEL}\" exclusive_caps=1 max_buffers=4" | sudo tee "$CONF_FILE"
    echo "v4l2loopback" | sudo tee /etc/modules-load.d/nvbroadcast-v4l2loopback.conf
    echo "Config saved. Virtual camera will persist across reboots."
fi

# Check if module is loaded
if lsmod | grep -q v4l2loopback; then
    LIVE_NAME="$(cat "/sys/class/video4linux/video${DEVICE_NUM}/name" 2>/dev/null || true)"
    if [ "$LIVE_NAME" = "$LABEL" ]; then
        echo "v4l2loopback is already loaded with ${LABEL}"
        if [ -e "$DEVICE" ]; then
            echo "Virtual camera device ${DEVICE} already exists"
            v4l2-ctl -d "$DEVICE" --all 2>/dev/null | head -5 || true
        fi
        exit 0
    fi

    echo "v4l2loopback is already loaded"
    if [ -n "$LIVE_NAME" ]; then
        echo "Current ${DEVICE} name: ${LIVE_NAME}"
    fi

    LOOPBACK_COUNT="unknown"
    if command -v v4l2-ctl &>/dev/null; then
        LOOPBACK_COUNT="$(v4l2-ctl --list-devices 2>/dev/null | grep -c 'v4l2loopback' || true)"
    fi

    DEVICE_IN_USE=false
    if command -v fuser &>/dev/null && [ -e "$DEVICE" ] && fuser -s "$DEVICE" 2>/dev/null; then
        DEVICE_IN_USE=true
    fi

    if [ "$LOOPBACK_COUNT" = "1" ] && [ "$DEVICE_IN_USE" = false ]; then
        echo "Reloading v4l2loopback to apply camera name ${LABEL}..."
        sudo modprobe -r v4l2loopback
        sudo modprobe v4l2loopback \
            devices=1 \
            video_nr="${DEVICE_NUM}" \
            card_label="${LABEL}" \
            exclusive_caps=1 \
            max_buffers=4
        echo "Virtual camera reloaded at ${DEVICE}"
        exit 0
    fi

    echo "Skipping live reload because ${DEVICE} is in use or another loopback device exists."
    echo "Close OBS/browser/meeting apps and reboot to apply camera name ${LABEL}."
    exit 0
fi

# Load module
echo "Loading v4l2loopback with device ${DEVICE}..."
sudo modprobe v4l2loopback \
    devices=1 \
    video_nr="${DEVICE_NUM}" \
    card_label="${LABEL}" \
    exclusive_caps=1 \
    max_buffers=4

echo "Virtual camera created at ${DEVICE}"

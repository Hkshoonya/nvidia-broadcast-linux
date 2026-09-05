#!/usr/bin/env bash
# NVIDIA Broadcast for Linux - Installer
# by doczeus | AI Powered
#
# Supports: Ubuntu, Debian, Pop!_OS, Linux Mint, Fedora, RHEL, CentOS,
#           Arch, Manjaro, EndeavourOS, openSUSE, Gentoo, Void, NixOS
set -eE
export PYTHONNOUSERSITE=1
trap 'rc=$?; echo ""; echo "ERROR: Installation failed at line $LINENO (exit code $rc)"; echo "Please report this issue with the output above."; exit $rc' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${HOME}/.local"
VENV_DIR="${SCRIPT_DIR}/.venv"
RUNTIME_REQUEST="auto"
WITH_MEETING=false
PYTHON_REQUEST=""

usage() {
    echo "Usage: $0 [--runtime auto|cpu|cuda] [--with-meeting] [--python /path/to/python]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --runtime)
            [ "$#" -ge 2 ] || { echo "ERROR: --runtime requires auto, cpu, or cuda."; exit 2; }
            RUNTIME_REQUEST="$2"
            shift 2
            ;;
        --runtime=*)
            RUNTIME_REQUEST="${1#*=}"
            shift
            ;;
        --with-meeting)
            WITH_MEETING=true
            shift
            ;;
        --python)
            [ "$#" -ge 2 ] || { echo "ERROR: --python requires an interpreter path."; exit 2; }
            [ -n "$2" ] || { echo "ERROR: --python requires an interpreter path."; exit 2; }
            PYTHON_REQUEST="$2"
            shift 2
            ;;
        --python=*)
            PYTHON_REQUEST="${1#*=}"
            [ -n "$PYTHON_REQUEST" ] || { echo "ERROR: --python requires an interpreter path."; exit 2; }
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

case "$RUNTIME_REQUEST" in
    auto|cpu|cuda) ;;
    *) echo "ERROR: Invalid runtime '$RUNTIME_REQUEST'; expected auto, cpu, or cuda."; exit 2 ;;
esac

guard_source_environment() {
    if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: The selected Python interpreter is unavailable."
        exit 1
    fi

    local guard_status
    if "$PYTHON_BIN" "$SCRIPT_DIR/scripts/check_source_venv_processes.py" --venv "$VENV_DIR"; then
        return
    else
        guard_status=$?
    fi

    if [ "$guard_status" -eq 1 ]; then
        echo "Stop NVBroadcast and the virtual-camera service, then rerun this installer."
        if command -v systemctl &>/dev/null; then
            echo "For the user service: systemctl --user stop nvbroadcast-vcam.service"
        fi
    else
        echo "Resolve the process-inspection error above, then rerun this installer."
    fi
    exit 1
}

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

select_python_interpreter() {
    local require_desktop_bindings="$1"
    local selection
    local selector_args=(--package-manager "$PKG_MANAGER")
    if [ -n "$PYTHON_REQUEST" ]; then
        selector_args+=(--python "$PYTHON_REQUEST")
    fi
    if [ "$require_desktop_bindings" = true ]; then
        selector_args+=(--require-desktop-bindings)
    fi
    if ! selection="$("$BASH" "$SCRIPT_DIR/scripts/select_python_interpreter.sh" "${selector_args[@]}")"; then
        return 1
    fi
    IFS=$'\t' read -r PYTHON_BIN PY_VER PY_MAJOR PY_MINOR <<< "$selection"
    if [ -z "$PYTHON_BIN" ] || [ -z "$PY_VER" ] || [ -z "$PY_MAJOR" ] || [ -z "$PY_MINOR" ]; then
        echo "ERROR: Python interpreter selection returned incomplete data."
        return 1
    fi
}

if ! select_python_interpreter false; then
    exit 1
fi

# Refuse before system or environment mutations while an existing source
# process can still import code from the installer-owned environment.
guard_source_environment

APP_VERSION="$(SCRIPT_DIR="$SCRIPT_DIR" "$PYTHON_BIN" - <<'PY' 2>/dev/null || echo unknown
from pathlib import Path
import os
import tomllib
data = tomllib.loads((Path(os.environ["SCRIPT_DIR"]) / "pyproject.toml").read_text())
print(data.get("project", {}).get("version", "unknown"))
PY
)"

# Check Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    ERRORS+=("This installer only supports Linux")
fi

# The selector accepts CPython 3.11 and newer (gated on desktop-binding
# availability, not a hardcoded ceiling) and validates venv support.
echo "  Python $PY_VER ($PYTHON_BIN) ... OK"

# Recheck venv and its pip bootstrap defensively in case the selected
# interpreter changed on disk after selection.
if ! "$PYTHON_BIN" -I -c 'import ensurepip, venv; ensurepip.version()' &>/dev/null; then
    case "$PKG_MANAGER" in
        apt)    ERRORS+=("Python $PY_VER venv support disappeared (install: sudo apt install python${PY_VER}-venv)") ;;
        dnf)    ERRORS+=("Python $PY_VER venv support disappeared (install: sudo dnf install python${PY_VER})") ;;
        pacman) ERRORS+=("Python $PY_VER venv support disappeared (reinstall the official python package)") ;;
        *)      ERRORS+=("Python $PY_VER venv support disappeared") ;;
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

# System PyGObject packages are interpreter-specific. Re-evaluate after those
# packages are present so an automatic selection cannot create a venv whose
# GUI, GStreamer, or Libadwaita imports will fail at first launch.
INITIAL_PYTHON_BIN="$PYTHON_BIN"
INITIAL_PYTHON_VER="$PY_VER"
if ! select_python_interpreter true; then
    exit 1
fi
if [ "$PYTHON_BIN" != "$INITIAL_PYTHON_BIN" ]; then
    echo "  Python $INITIAL_PYTHON_VER lacks compatible desktop bindings; using Python $PY_VER ($PYTHON_BIN)."
else
    echo "  Python desktop bindings ... OK"
fi

PY_RUNTIME_NOTICE="$(
PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON_BIN" - <<'PY' 2>/dev/null || true
from nvbroadcast.core.platform import python_runtime_advisory
notice = python_runtime_advisory()
if notice:
    _, title, body = notice
    print(title)
    print(body)
PY
)"

if [ -n "$PY_RUNTIME_NOTICE" ]; then
    echo ""
    echo "NOTICE:"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        echo "  $line"
    done <<< "$PY_RUNTIME_NOTICE"
    echo ""
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

MACHINE_ARCH="$(uname -m)"
case "$RUNTIME_REQUEST" in
    cpu) SELECTED_RUNTIME_VARIANT="cpu" ;;
    cuda)
        case "$MACHINE_ARCH" in
            x86_64|amd64) SELECTED_RUNTIME_VARIANT="cuda" ;;
            *)
                echo "ERROR: CUDA runtime variant supports Linux x86_64 only; found ${MACHINE_ARCH}."
                exit 1
                ;;
        esac
        ;;
    auto)
        case "$MACHINE_ARCH:$HAS_NVIDIA" in
            x86_64:true|amd64:true) SELECTED_RUNTIME_VARIANT="cuda" ;;
            *) SELECTED_RUNTIME_VARIANT="cpu" ;;
        esac
        ;;
esac
echo "  Runtime variant: $SELECTED_RUNTIME_VARIANT (requested: $RUNTIME_REQUEST)"

# ─── Step 2: v4l2loopback Configuration ─────────────────────────────────────

echo ""
echo "[2/7] Configuring virtual camera (v4l2loopback)..."

V4L2_DEVICE_NUM="${NVBROADCAST_VCAM_DEVICE_NUM:-10}"
if [ -n "${NVBROADCAST_VCAM_DEVICE:-}" ]; then
    case "$NVBROADCAST_VCAM_DEVICE" in
        /dev/video*) V4L2_DEVICE_NUM="${NVBROADCAST_VCAM_DEVICE#/dev/video}" ;;
        *) echo "WARNING: Ignoring invalid NVBROADCAST_VCAM_DEVICE=${NVBROADCAST_VCAM_DEVICE}; expected /dev/videoN." ;;
    esac
fi
case "$V4L2_DEVICE_NUM" in
    ''|*[!0-9]*)
        echo "WARNING: Invalid virtual camera number '${V4L2_DEVICE_NUM}', using 10."
        V4L2_DEVICE_NUM=10
        ;;
esac
V4L2_DEVICE="/dev/video${V4L2_DEVICE_NUM}"
V4L2_LABEL="NVbroadcast"
V4L2_CONF="/etc/modprobe.d/nvbroadcast-v4l2loopback.conf"
V4L2_LOAD="/etc/modules-load.d/nvbroadcast-v4l2loopback.conf"
V4L2_OPTIONS="options v4l2loopback devices=1 video_nr=${V4L2_DEVICE_NUM} card_label=\"${V4L2_LABEL}\" exclusive_caps=1 max_buffers=4"

if [ -e "$V4L2_DEVICE" ]; then
    if ! command -v v4l2-ctl &>/dev/null || \
       ! v4l2-ctl -D -d "$V4L2_DEVICE" 2>/dev/null | \
           grep -Eiq 'Driver name[[:space:]]*:[[:space:]]*v4l2[[:space:]_-]*loopback'; then
        echo "ERROR: ${V4L2_DEVICE} already exists and is not a v4l2loopback virtual camera."
        echo "Choose an unused video number with NVBROADCAST_VCAM_DEVICE_NUM."
        exit 1
    fi
fi

# Remove old BluCast configs if present
sudo rm -f /etc/modprobe.d/blucast-v4l2loopback.conf 2>/dev/null || true
sudo rm -f /etc/modules-load.d/blucast-v4l2loopback.conf 2>/dev/null || true

if [ ! -f "$V4L2_CONF" ] || grep -Eq 'card_label="(NVIDIA Broadcast|NVIDIA Broadcast Virtual Camera|NV Broadcast|NVbroadcast)"' "$V4L2_CONF"; then
    if echo "$V4L2_OPTIONS" | sudo tee "$V4L2_CONF" > /dev/null; then
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
    sudo modprobe v4l2loopback devices=1 video_nr="${V4L2_DEVICE_NUM}" card_label="${V4L2_LABEL}" exclusive_caps=1 max_buffers=4 2>/dev/null || \
        echo "WARNING: Could not load v4l2loopback. You may need to reboot or install kernel headers."
else
    LIVE_VCAM_NAME="$(cat "/sys/class/video4linux/video${V4L2_DEVICE_NUM}/name" 2>/dev/null || true)"
    if [ "$LIVE_VCAM_NAME" = "$V4L2_LABEL" ]; then
        echo "v4l2loopback already loaded with ${V4L2_LABEL}"
    else
        echo "v4l2loopback already loaded"
        if [ -n "$LIVE_VCAM_NAME" ]; then
            echo "Current ${V4L2_DEVICE} name: ${LIVE_VCAM_NAME}"
        fi

        LOOPBACK_COUNT="unknown"
        if command -v v4l2-ctl &>/dev/null; then
            LOOPBACK_COUNT="$(v4l2-ctl --list-devices 2>/dev/null | grep -c 'v4l2loopback' || true)"
        fi

        VCAM_IN_USE=false
        if command -v fuser &>/dev/null && [ -e "$V4L2_DEVICE" ] && fuser -s "$V4L2_DEVICE" 2>/dev/null; then
            VCAM_IN_USE=true
        fi

        if [ "$LOOPBACK_COUNT" = "1" ] && [ "$VCAM_IN_USE" = false ]; then
            echo "Reloading v4l2loopback to apply camera name ${V4L2_LABEL}..."
            if sudo modprobe -r v4l2loopback 2>/dev/null && \
               sudo modprobe v4l2loopback devices=1 video_nr="${V4L2_DEVICE_NUM}" card_label="${V4L2_LABEL}" exclusive_caps=1 max_buffers=4 2>/dev/null; then
                echo "Reloaded v4l2loopback with camera name ${V4L2_LABEL}"
            else
                echo "WARNING: Could not reload v4l2loopback. Reboot after installation to apply the camera name."
            fi
        else
            echo "Skipping live v4l2loopback reload because ${V4L2_DEVICE} is in use or another loopback device exists."
            echo "Close OBS/browser/meeting apps and reboot to apply camera name ${V4L2_LABEL}."
        fi
    fi
fi

if [ -e "$V4L2_DEVICE" ]; then
    echo "Virtual camera device: $V4L2_DEVICE"
else
    echo "WARNING: $V4L2_DEVICE not found. You may need to reboot."
fi

# ─── Step 3: Python Environment ─────────────────────────────────────────────

echo ""
echo "[3/7] Setting up Python environment..."

installed_runtime_variant() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "none"
        return
    fi
    "$VENV_DIR/bin/python" - <<'PY' 2>/dev/null || echo "unknown"
from nvbroadcast.runtime.variants import detect_runtime_variant
variant = detect_runtime_variant()
print(variant.value if variant else "mixed-or-missing")
PY
}

installed_python_base() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "none"
        return
    fi
    "$VENV_DIR/bin/python" -I - <<'PY' 2>/dev/null || echo "unknown"
import os
import sys
print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))
PY
}

CURRENT_RUNTIME_VARIANT="$(installed_runtime_variant)"
CURRENT_PYTHON_BASE="$(installed_python_base)"
SELECTED_PYTHON_BASE="$("$PYTHON_BIN" -I -c 'import os, sys; print(os.path.realpath(sys.executable))')"

# Recheck immediately before removing or upgrading the environment. This
# narrows the window in which a source process could start during preflight.
guard_source_environment

REPLACE_VENV=false
if [ -d "$VENV_DIR" ] && [ "$CURRENT_PYTHON_BASE" != "$SELECTED_PYTHON_BASE" ]; then
    echo "Replacing environment created by ${CURRENT_PYTHON_BASE} with selected interpreter ${SELECTED_PYTHON_BASE}..."
    REPLACE_VENV=true
fi
if [ "$CURRENT_RUNTIME_VARIANT" != "none" ] && [ "$CURRENT_RUNTIME_VARIANT" != "$SELECTED_RUNTIME_VARIANT" ]; then
    echo "Replacing ${CURRENT_RUNTIME_VARIANT} environment with ${SELECTED_RUNTIME_VARIANT} runtime variant..."
    REPLACE_VENV=true
fi
if [ "$REPLACE_VENV" = true ]; then
    rm -rf -- "$VENV_DIR"
fi

create_virtual_environment() {
    "$PYTHON_BIN" -m venv "$VENV_DIR" --system-site-packages
    echo "Created virtual environment"
}

prepare_virtual_environment() {
    if [ ! -d "$VENV_DIR" ]; then
        create_virtual_environment
    fi
    "$VENV_DIR/bin/pip" install --upgrade \
        "pip>=26.2" "setuptools>=83.0.0" wheel -q
}

prepare_virtual_environment

install_runtime_variant() {
    local meeting_backends="none"
    if [ "$WITH_MEETING" = true ]; then
        meeting_backends="all"
    fi
    "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/install_runtime_variant.py" \
        --project "$SCRIPT_DIR" --variant "$1" \
        --meeting-backends "$meeting_backends"
}

CUDA_EXTRA_INSTALLED=false
CUDA_ACCEL_AVAILABLE=false
if ! install_runtime_variant "$SELECTED_RUNTIME_VARIANT"; then
    if [ "$RUNTIME_REQUEST" = "auto" ] && [ "$SELECTED_RUNTIME_VARIANT" = "cuda" ]; then
        echo "WARNING: CUDA runtime installation failed. Recreating clean CPU environment."
        rm -rf -- "$VENV_DIR"
        SELECTED_RUNTIME_VARIANT="cpu"
        prepare_virtual_environment
        install_runtime_variant "$SELECTED_RUNTIME_VARIANT"
    else
        echo "ERROR: Failed to install requested ${SELECTED_RUNTIME_VARIANT} runtime variant."
        exit 1
    fi
fi
if [ "$SELECTED_RUNTIME_VARIANT" = "cuda" ]; then CUDA_EXTRA_INSTALLED=true; fi
if [ "$WITH_MEETING" = true ]; then
    echo "Core packages, meeting backends, and ${SELECTED_RUNTIME_VARIANT} runtime installed."
else
    echo "Core packages and ${SELECTED_RUNTIME_VARIANT} runtime installed."
fi

# Verify critical Python packages
echo ""
echo "Verifying core dependencies..."
FAILED_PY=()
CORE_PY_MODULES=(numpy cv2 onnxruntime PIL psutil onnx mediapipe)
for mod in "${CORE_PY_MODULES[@]}"; do
    if "$VENV_DIR/bin/python" -c "import $mod" 2>/dev/null; then
        echo "  $mod ... OK"
    else
        FAILED_PY+=("$mod")
        echo "  $mod ... FAILED"
    fi
done

if "$VENV_DIR/bin/python" -c "import av; import av.option" 2>/dev/null; then
    echo "  av ... OK"
else
    FAILED_PY+=("av")
    echo "  av ... FAILED"
fi

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
if [ "$SELECTED_RUNTIME_VARIANT" = "cuda" ]; then
    # The shared runtime installer has already required a successful fresh-
    # process session and pinned-model inference on CUDA.
    echo "  CUDA execution probe ... OK"
    CUDA_ACCEL_AVAILABLE=true
else
    echo "  CPU execution probe ... OK"
fi

# ─── Optional Packages ────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────"
echo "  Optional Packages"
echo "─────────────────────────────────────────"
echo ""
echo "  These unlock premium features. You can install them now or later."
echo "  If skipped, the app will prompt when you select a mode that needs them."
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 14 ]; then
    echo "  Python $PY_VER note: some premium paths use safer defaults on this interpreter."
fi
echo ""

# CuPy compositing retry. The full CUDA inference runtime is installed by the
# project cuda extra above; this fallback only repairs missing GPU blending.
CUPY_INSTALLED=false
if "$VENV_DIR/bin/python" -c "import cupy" 2>/dev/null; then
    echo "  [installed] CuPy CUDA — GPU compositing runtime"
    CUPY_INSTALLED=true
else
    echo "  1) CuPy CUDA compositing retry (~800MB) — Repairs:"
    echo "     - Fused CUDA kernel compositing for DocZeus/Killer"
    echo "     - GPU alpha blending when CUDA inference is already available"
    echo "     - Lower CPU cost for background replacement"
    echo ""
    if [ "$SELECTED_RUNTIME_VARIANT" = "cuda" ]; then
        read -rp "  Install CuPy compositing runtime? [Y/n] " install_cupy
        install_cupy="${install_cupy:-Y}"
        if [[ "$install_cupy" =~ ^[Yy]$ ]]; then
            echo "  Installing CuPy (this may take a few minutes)..."
            if "$VENV_DIR/bin/pip" install "cupy-cuda12x>=14.1.1,<15" nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 -q 2>&1; then
                if CUPY_TEST=$("$VENV_DIR/bin/python" -c "from nvbroadcast.core.platform import preload_nvidia_runtime_libs; preload_nvidia_runtime_libs(); import cupy; a=cupy.ones(10); print('OK')" 2>&1); then
                    if [ "$CUPY_TEST" = "OK" ]; then
                        echo "  CuPy installed and verified!"
                        CUPY_INSTALLED=true
                    else
                        echo "  WARNING: CuPy installed but verification returned unexpected output."
                        echo "  Output: $CUPY_TEST"
                        echo "  You can retry later: $VENV_DIR/bin/pip install 'cupy-cuda12x>=14.1.1,<15' nvidia-cuda-runtime-cu12"
                    fi
                else
                    echo "  WARNING: CuPy installed but verification failed."
                    if [ -n "${CUPY_TEST:-}" ]; then
                        echo "  Output: $CUPY_TEST"
                    fi
                    echo "  You can retry later: $VENV_DIR/bin/pip install 'cupy-cuda12x>=14.1.1,<15' nvidia-cuda-runtime-cu12"
                fi
            else
                echo "  WARNING: CuPy installation failed. Skipping."
                echo "  Retry later: $VENV_DIR/bin/pip install 'cupy-cuda12x>=14.1.1,<15' nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12"
            fi
        else
            echo "  Skipped. Install later: $VENV_DIR/bin/pip install 'cupy-cuda12x>=14.1.1,<15' nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12"
        fi
    else
        echo "  [skipped] No NVIDIA GPU detected."
    fi
fi
echo ""

# TensorRT (Zeus/Killer inference optimization)
verify_tensorrt_execution() {
    local probe_output
    if probe_output="$("$VENV_DIR/bin/python" -m nvbroadcast.runtime \
        --variant cuda --provider tensorrt 2>&1)"; then
        echo "  TensorRT execution probe ... OK"
        return 0
    fi

    echo "  WARNING: TensorRT is importable, but its execution provider failed the pinned-model probe."
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        echo "    $line"
    done <<< "$probe_output"
    return 1
}

TRT_INSTALLED=false
TRT_SUPPORTED=false
TRT_PROBE_FAILED=false
TRT_UNVERIFIED=false
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 8 ] && [ "$PY_MINOR" -le 13 ]; then
    TRT_SUPPORTED=true
fi

if "$VENV_DIR/bin/python" -c "import tensorrt" 2>/dev/null; then
    if [ "$SELECTED_RUNTIME_VARIANT" = "cuda" ]; then
        if verify_tensorrt_execution; then
            echo "  [installed] TensorRT — provider execution verified for Zeus/Killer modes"
            TRT_INSTALLED=true
        else
            TRT_PROBE_FAILED=true
        fi
    else
        echo "  [detected] TensorRT package — execution not verified without the CUDA runtime variant"
        TRT_UNVERIFIED=true
    fi
else
    echo "  2) TensorRT (~4GB) — Unlocks:"
    echo "     - Optimized model inference (future TRT engine support)"
    echo "     - Potential 2-5x inference speedup on supported models"
    echo ""
    if [ "$SELECTED_RUNTIME_VARIANT" = "cuda" ]; then
        if [ "$TRT_SUPPORTED" = true ]; then
            read -rp "  Install TensorRT? [y/N] " install_trt
            install_trt="${install_trt:-N}"
            if [[ "$install_trt" =~ ^[Yy]$ ]]; then
                echo "  Installing TensorRT (this may take several minutes)..."
                if "$VENV_DIR/bin/pip" install tensorrt-cu12 onnx -q 2>&1; then
                    if verify_tensorrt_execution; then
                        echo "  TensorRT installed and its execution provider was verified!"
                        TRT_INSTALLED=true
                    else
                        TRT_PROBE_FAILED=true
                    fi
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
        echo "  [skipped] The CUDA runtime variant is not selected."
    fi
fi
echo ""

# Summary of optional packages
echo "  Optional packages summary:"
if [ "$CUDA_ACCEL_AVAILABLE" = true ]; then
    echo "    CUDA runtime: INSTALLED (GPU inference available)"
elif [ "$CUDA_EXTRA_INSTALLED" = true ]; then
    echo "    CUDA runtime: INSTALLED (provider check still reported CPU fallback)"
else
    echo "    CUDA runtime: NOT INSTALLED (CPU inference fallback)"
fi
if [ "$CUPY_INSTALLED" = true ] && [ "$CUDA_ACCEL_AVAILABLE" = true ]; then
    echo "    CuPy:     INSTALLED (CUDA modes available)"
elif [ "$CUPY_INSTALLED" = true ]; then
    echo "    CuPy:     INSTALLED (CUDA modes still need GPU inference runtime)"
else
    echo "    CuPy:     NOT INSTALLED (CPU modes only)"
fi
if [ "$TRT_INSTALLED" = true ]; then
    echo "    TensorRT: INSTALLED (provider execution verified)"
elif [ "$TRT_PROBE_FAILED" = true ]; then
    echo "    TensorRT: NOT READY (provider execution probe failed)"
elif [ "$TRT_UNVERIFIED" = true ]; then
    echo "    TensorRT: PRESENT BUT UNVERIFIED (select the CUDA runtime to probe it)"
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
export PYTHONNOUSERSITE=1
NVBROADCAST_DIR="PLACEHOLDER_DIR"
exec "$NVBROADCAST_DIR/.venv/bin/python" -m nvbroadcast "$@"
LAUNCHER
sed -i "s|PLACEHOLDER_DIR|${SCRIPT_DIR}|g" "$INSTALL_PREFIX/bin/nvbroadcast"
chmod +x "$INSTALL_PREFIX/bin/nvbroadcast"

cat > "$INSTALL_PREFIX/bin/nvbroadcast-vcam" << 'LAUNCHER'
#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
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
Description=NVbroadcast Virtual Camera Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=$INSTALL_PREFIX/bin/nvbroadcast-vcam
Restart=on-failure
RestartSec=3
Environment=PYTHONNOUSERSITE=1
Environment=GST_PLUGIN_PATH=$GST_PLUGIN_PATH

[Install]
WantedBy=graphical-session.target
EOF

if systemctl --user daemon-reload 2>/dev/null; then
    if [ "${NVBROADCAST_ENABLE_HEADLESS_SERVICE:-0}" = "1" ]; then
        systemctl --user enable nvbroadcast-vcam.service 2>/dev/null || true
        echo "Systemd service installed and enabled for headless mode"
    else
        systemctl --user disable nvbroadcast-vcam.service 2>/dev/null || true
        systemctl --user stop nvbroadcast-vcam.service 2>/dev/null || true
        echo "Systemd service installed but disabled by default"
        echo "  Enable only for no-GUI/headless use: systemctl --user enable --now nvbroadcast-vcam.service"
    fi
else
    echo "Service file installed (run 'systemctl --user daemon-reload && systemctl --user enable --now nvbroadcast-vcam' only for headless/no-GUI use)"
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
echo "  Installation Complete! v$APP_VERSION"
echo "  NVIDIA Broadcast for Linux"
echo "  by doczeus | AI Powered"
echo "========================================="
echo ""
echo "  System: $DISTRO_NAME ($PKG_MANAGER)"
echo "  Compositing: $COMPOSITING"
echo "  CUDA inference: $( [ "$CUDA_ACCEL_AVAILABLE" = true ] && echo "YES" || echo "NO (CPU fallback)" )"
if [ "$CUPY_INSTALLED" = true ] && [ "$CUDA_ACCEL_AVAILABLE" = true ]; then
    echo "  CuPy: YES (CUDA modes available)"
elif [ "$CUPY_INSTALLED" = true ]; then
    echo "  CuPy: YES (CUDA modes still need GPU inference runtime)"
else
    echo "  CuPy: NO (install later for GPU modes)"
fi
if [ "$TRT_INSTALLED" = true ]; then
    echo "  TensorRT: YES (provider execution verified)"
elif [ "$TRT_PROBE_FAILED" = true ]; then
    echo "  TensorRT: NOT READY (provider execution probe failed)"
elif [ "$TRT_UNVERIFIED" = true ]; then
    echo "  TensorRT: UNVERIFIED (rerun with --runtime cuda to probe it)"
elif [ "$TRT_SUPPORTED" = true ]; then
    echo "  TensorRT: NO (install later for Zeus/Killer optimization)"
else
    echo "  TensorRT: UNSUPPORTED ON PYTHON $PY_VER (requires Python 3.8-3.13)"
fi
if [ -n "$PY_RUNTIME_NOTICE" ]; then
    echo ""
    echo "  Python runtime notice:"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        echo "    $line"
    done <<< "$PY_RUNTIME_NOTICE"
fi
echo ""
echo "  Available modes:"
if [ "$CUPY_INSTALLED" = true ] && [ "$CUDA_ACCEL_AVAILABLE" = true ]; then
    if [ "$TRT_INSTALLED" = true ]; then
        echo "    Killer  — 48fps fused CUDA (fastest)"
        echo "    Zeus    — 33fps GPU-optimized"
    else
        echo "    Killer  — unavailable until TensorRT passes its execution probe"
        echo "    Zeus    — unavailable until TensorRT passes its execution probe"
    fi
    echo "    DocZeus — 23fps full quality + fused kernel"
elif [ "$CUPY_INSTALLED" = true ]; then
    echo "    Killer/Zeus/DocZeus — unavailable until CUDA inference runtime installs"
fi
if [ "$CUDA_ACCEL_AVAILABLE" = true ] && [ "$CUPY_INSTALLED" = true ]; then
    echo "    CUDA Max/Balanced/Perf — standard GPU modes"
elif [ "$CUDA_ACCEL_AVAILABLE" = true ]; then
    echo "    CUDA Max/Balanced/Perf — unavailable until CuPy installs"
else
    echo "    CUDA Max/Balanced/Perf — unavailable until CUDA runtime installs"
fi
echo "    CPU Quality/Light/Low  — CPU fallback"
echo ""
echo "  Recent patch highlights:"
echo "    Virtual Camera Stability — safer Linux loopback sink startup"
echo "    Lower Live Lag           — shared face landmarks and ROI relighting"
echo "    Better Replace Edges     — tighter shoulders, hair, and arm gaps"
echo "    Meeting Transcription    — faster startup and cleaner saved audio"
echo "    Resolution Safety        — save changes without hanging the stream"
echo ""
echo "  To install optional packages later:"
echo "    Runtime switch: stop NVBroadcast, then run $SCRIPT_DIR/install.sh --runtime cpu|cuda"
echo "    CuPy:     $VENV_DIR/bin/pip install 'cupy-cuda12x>=14.1.1,<15' nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12"
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

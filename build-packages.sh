#!/usr/bin/env bash
# NV Broadcast - Package Builder
# Builds .deb and .rpm packages from the current source tree.
# Version is read from pyproject.toml automatically.
#
# Usage:
#   ./build-packages.sh          # Build both .deb and .rpm
#   ./build-packages.sh deb      # Build .deb only
#   ./build-packages.sh rpm      # Build .rpm only
#
# Output:
#   dist/deb/nvbroadcast_<version>-<rev>_all.deb
#   dist/rpm/nvbroadcast-<version>-<rev>.noarch.rpm

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Read version from pyproject.toml ─────────────────────────────────────────

VERSION=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
" 2>/dev/null || python3 -c "
import re
with open('pyproject.toml') as f:
    m = re.search(r'version\s*=\s*\"(.+?)\"', f.read())
    print(m.group(1))
")

if [ -z "$VERSION" ]; then
    echo "ERROR: Could not read version from pyproject.toml"
    exit 1
fi

# Package revision is stable unless explicitly overridden by CI.
REV="${PACKAGE_REV:-1}"

echo "========================================="
echo "  NV Broadcast Package Builder"
echo "  Version: ${VERSION}-${REV}"
echo "========================================="
echo ""

BUILD_TARGET="${1:-all}"

# ─── Build .deb ───────────────────────────────────────────────────────────────

build_deb() {
    echo "[DEB] Building .deb package..."

    local BUILD_DIR
    BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nvbroadcast-deb-build.XXXXXX")
    local PKG_DIR="${BUILD_DIR}/nvbroadcast_${VERSION}-${REV}_all"
    mkdir -p "$PKG_DIR/DEBIAN"

    # Generate binary control file (strip source-only fields, add version)
    cat > "$PKG_DIR/DEBIAN/control" << CTRL
Package: nvbroadcast
Version: ${VERSION}-${REV}
Section: video
Priority: optional
Architecture: all
Maintainer: doczeus <harshit@kshoonya.com>
Depends: python3 (>= 3.11), python3-venv, python3-gi, python3-gi-cairo, gir1.2-gtk-4.0, gir1.2-adw-1, gir1.2-gstreamer-1.0, gir1.2-gst-plugins-base-1.0, gstreamer1.0-plugins-base, gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad, v4l-utils, v4l2loopback-dkms, psmisc, pipewire-bin | pipewire-utils, pulseaudio-utils
Recommends: gir1.2-ayatanaappindicator3-0.1
Homepage: https://github.com/Hkshoonya/nvidia-broadcast-linux
Description: NV Broadcast - Unofficial NVIDIA Broadcast for Linux
 AI-powered virtual camera with background removal, blur, replacement,
 video enhancement, auto-framing, and noise cancellation.
 9 processing modes including Killer, Zeus, and DocZeus with fused CUDA.
 Requires NVIDIA GPU with driver 525+ for GPU acceleration.
CTRL

    # Scripts
    cp packaging/debian/postinst "$PKG_DIR/DEBIAN/"
    cp packaging/debian/prerm "$PKG_DIR/DEBIAN/"
    cp packaging/debian/postrm "$PKG_DIR/DEBIAN/"
    sed -i '/^#DEBHELPER#$/d' \
        "$PKG_DIR/DEBIAN/postinst" \
        "$PKG_DIR/DEBIAN/prerm" \
        "$PKG_DIR/DEBIAN/postrm"
    chmod 755 "$PKG_DIR/DEBIAN/postinst" "$PKG_DIR/DEBIAN/prerm" "$PKG_DIR/DEBIAN/postrm"

    # Application files -> /opt/nvbroadcast
    install -d "$PKG_DIR/opt/nvbroadcast"
    cp -r src pyproject.toml LICENSE README.md "$PKG_DIR/opt/nvbroadcast/"
    install -Dm 755 scripts/install_runtime_variant.py \
        "$PKG_DIR/opt/nvbroadcast/scripts/install_runtime_variant.py"
    find "$PKG_DIR/opt/nvbroadcast/src" -type d \
        \( -name "__pycache__" -o -name "*.egg-info" \) \
        -prune -exec rm -rf {} +
    install -d "$PKG_DIR/opt/nvbroadcast/models"
    cp -r data "$PKG_DIR/opt/nvbroadcast/"
    [ -d configs ] && cp -r configs "$PKG_DIR/opt/nvbroadcast/" || true

    # Desktop entry
    install -d "$PKG_DIR/usr/share/applications"
    cp data/com.doczeus.NVBroadcast.desktop "$PKG_DIR/usr/share/applications/"
    sed -i "s|Exec=nvbroadcast|Exec=/usr/bin/nvbroadcast|g" "$PKG_DIR/usr/share/applications/com.doczeus.NVBroadcast.desktop"

    # AppStream metadata
    install -d "$PKG_DIR/usr/share/metainfo"
    cp data/com.doczeus.NVBroadcast.metainfo.xml "$PKG_DIR/usr/share/metainfo/"

    # Icon
    install -d "$PKG_DIR/usr/share/icons/hicolor/scalable/apps"
    cp data/icons/com.doczeus.NVBroadcast.svg "$PKG_DIR/usr/share/icons/hicolor/scalable/apps/"

    # Debian package documentation
    install -d "$PKG_DIR/usr/share/doc/nvbroadcast"
    install -m 644 packaging/debian/copyright "$PKG_DIR/usr/share/doc/nvbroadcast/copyright"
    gzip -9n -c packaging/debian/changelog > \
        "$PKG_DIR/usr/share/doc/nvbroadcast/changelog.Debian.gz"

    # Launcher scripts
    install -d "$PKG_DIR/usr/bin"
    cat > "$PKG_DIR/usr/bin/nvbroadcast" << 'LAUNCHER'
#!/bin/bash
export PYTHONNOUSERSITE=1
exec /opt/nvbroadcast/.venv/bin/python -m nvbroadcast "$@"
LAUNCHER
    chmod 755 "$PKG_DIR/usr/bin/nvbroadcast"

    cat > "$PKG_DIR/usr/bin/nvbroadcast-vcam" << 'LAUNCHER'
#!/bin/bash
export PYTHONNOUSERSITE=1
exec /opt/nvbroadcast/.venv/bin/python -m nvbroadcast.vcam_service "$@"
LAUNCHER
    chmod 755 "$PKG_DIR/usr/bin/nvbroadcast-vcam"

    # Systemd service
    install -d "$PKG_DIR/usr/lib/systemd/user"
    cat > "$PKG_DIR/usr/lib/systemd/user/nvbroadcast-vcam.service" << 'SVC'
[Unit]
Description=NVbroadcast Virtual Camera Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=/usr/bin/nvbroadcast-vcam
Restart=on-failure
RestartSec=3
Environment=PYTHONNOUSERSITE=1

[Install]
WantedBy=graphical-session.target
SVC

    # Normalize the payload so local umask/ownership cannot leak into a system package.
    find "$PKG_DIR" -type d -exec chmod 755 {} +
    find "$PKG_DIR" -type f -exec chmod 644 {} +
    chmod 755 \
        "$PKG_DIR/DEBIAN/postinst" \
        "$PKG_DIR/DEBIAN/prerm" \
        "$PKG_DIR/DEBIAN/postrm" \
        "$PKG_DIR/usr/bin/nvbroadcast" \
        "$PKG_DIR/usr/bin/nvbroadcast-vcam"

    # Build .deb
    mkdir -p dist/deb
    dpkg-deb -Zxz --root-owner-group --build \
        "$PKG_DIR" \
        "dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb"

    echo "[DEB] Built: dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb"
    dpkg-deb --info "dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb" | head -10

    rm -rf "$BUILD_DIR"
}

# ─── Build .rpm ───────────────────────────────────────────────────────────────

build_rpm() {
    echo "[RPM] Building .rpm package..."

    if ! command -v rpmbuild &>/dev/null; then
        echo "[RPM] SKIP: rpmbuild not found. Install with: sudo apt install rpm"
        return
    fi

    local RPM_DIR
    RPM_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nvbroadcast-rpm-build.XXXXXX")
    mkdir -p "$RPM_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

    # Create source tarball
    local TAR_DIR="nvbroadcast-${VERSION}"
    local TAR_PATH="$RPM_DIR/SOURCES/${TAR_DIR}.tar.gz"
    local TAR_ROOT="$RPM_DIR/source"
    mkdir -p "$TAR_ROOT/$TAR_DIR"
    cp -r src pyproject.toml LICENSE README.md data "$TAR_ROOT/$TAR_DIR/"
    install -Dm 755 scripts/install_runtime_variant.py \
        "$TAR_ROOT/$TAR_DIR/scripts/install_runtime_variant.py"
    [ -d configs ] && cp -r configs "$TAR_ROOT/$TAR_DIR/" || true
    find "$TAR_ROOT/$TAR_DIR/src" -type d \
        \( -name "__pycache__" -o -name "*.egg-info" \) \
        -prune -exec rm -rf {} +
    (cd "$TAR_ROOT" && tar czf "$TAR_PATH" "$TAR_DIR")

    # Copy and update spec with current version
    sed "s/^Version:.*/Version:        ${VERSION}/" packaging/rpm/nvbroadcast.spec | \
        sed "s/^Release:.*/Release:        ${REV}%{?dist}/" > "$RPM_DIR/SPECS/nvbroadcast.spec"

    # Build
    rpmbuild \
        --nodeps \
        --define "_topdir $RPM_DIR" \
        --define "_userunitdir /usr/lib/systemd/user" \
        -bb "$RPM_DIR/SPECS/nvbroadcast.spec" 2>&1 | tail -5

    # Copy output
    mkdir -p dist/rpm
    find "$RPM_DIR/RPMS" -name "*.rpm" -exec cp {} dist/rpm/ \;

    echo "[RPM] Built:"
    ls -la dist/rpm/nvbroadcast-*.rpm 2>/dev/null || echo "  (no RPM found — check build errors above)"

    rm -rf "$RPM_DIR"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

# ─── Build .pkg (macOS) ──────────────────────────────────────────────────────

build_pkg() {
    echo "[PKG] Building .pkg package for macOS..."

    if [[ "$(uname)" != "Darwin" ]]; then
        echo "[PKG] SKIP: .pkg can only be built on macOS (needs pkgbuild/productbuild)"
        return
    fi

    local BUILD_DIR
    BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nvbroadcast-pkg-build.XXXXXX")
    local INSTALL_ROOT="${BUILD_DIR}/root"
    local SCRIPTS_DIR="${BUILD_DIR}/scripts"
    mkdir -p "$INSTALL_ROOT/opt/nvbroadcast"
    mkdir -p "$INSTALL_ROOT/usr/local/bin"
    mkdir -p "$SCRIPTS_DIR"

    # Application files -> /opt/nvbroadcast
    cp -r src pyproject.toml LICENSE README.md "$INSTALL_ROOT/opt/nvbroadcast/"
    install -Dm 755 scripts/install_runtime_variant.py \
        "$INSTALL_ROOT/opt/nvbroadcast/scripts/install_runtime_variant.py"
    find "$INSTALL_ROOT/opt/nvbroadcast/src" -type d \
        \( -name "__pycache__" -o -name "*.egg-info" \) \
        -prune -exec rm -rf {} +
    mkdir -p "$INSTALL_ROOT/opt/nvbroadcast/models"
    cp -r data "$INSTALL_ROOT/opt/nvbroadcast/"
    [ -d configs ] && cp -r configs "$INSTALL_ROOT/opt/nvbroadcast/" || true
    cp install_macos.sh "$INSTALL_ROOT/opt/nvbroadcast/"

    # Launcher script -> /usr/local/bin
    cat > "$INSTALL_ROOT/usr/local/bin/nvbroadcast" << 'LAUNCHER'
#!/bin/bash
export PYTHONNOUSERSITE=1
INSTALL_DIR="/opt/nvbroadcast"
if [ -d "$INSTALL_DIR/.venv" ]; then
    source "$INSTALL_DIR/.venv/bin/activate"
fi

# GStreamer plugin path for Homebrew
if command -v brew &>/dev/null; then
    export GST_PLUGIN_PATH="$(brew --prefix)/lib/gstreamer-1.0"
    export GI_TYPELIB_PATH="$(brew --prefix)/lib/girepository-1.0"
fi

cd "$INSTALL_DIR"
exec python3 -m nvbroadcast "$@"
LAUNCHER
    chmod 755 "$INSTALL_ROOT/usr/local/bin/nvbroadcast"

    # Post-install script — sets up venv and installs pip deps
    cat > "$SCRIPTS_DIR/postinstall" << 'POSTINST'
#!/bin/bash
set -e
export PYTHONNOUSERSITE=1
INSTALL_DIR="/opt/nvbroadcast"

echo "[NV Broadcast] Setting up Python environment..."

if [ "$(uname -m)" != "arm64" ]; then
    echo "[NV Broadcast] ERROR: v1.4.0 supports Apple Silicon Macs only."
    exit 1
fi

# Find a Python version covered by the Apple Silicon release matrix
PYTHON=""
for p in python3.13 python3.12 python3.11 python3; do
    if command -v "$p" &>/dev/null; then
        ver=$("$p" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] 2>/dev/null && \
           [ "$minor" -ge 11 ] 2>/dev/null && \
           [ "$minor" -le 13 ] 2>/dev/null; then
            PYTHON="$p"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[NV Broadcast] WARNING: Python 3.11-3.13 not found. Run: brew install python@3.12"
    exit 0
fi

# Stop old runtime before replacing installer-owned environment.
pkill -f "^${INSTALL_DIR}/.venv/bin/python -m nvbroadcast( |$)" 2>/dev/null || true
# Recreate environment so CPU remains sole runtime owner.
rm -rf -- "$INSTALL_DIR/.venv"
$PYTHON -m venv "$INSTALL_DIR/.venv" --system-site-packages 2>/dev/null || true
source "$INSTALL_DIR/.venv/bin/activate"
pip install --upgrade \
    "pip>=26.1.2" "setuptools>=83.0.0" wheel -q 2>/dev/null || true
python "$INSTALL_DIR/scripts/install_runtime_variant.py" \
    --project "$INSTALL_DIR" --variant cpu --meeting-backends faster

# CoreML for Apple Silicon
if [ "$(uname -m)" = "arm64" ]; then
    pip install -q coremltools 2>/dev/null || true
fi

echo "[NV Broadcast] Installation complete. Run: nvbroadcast"
POSTINST
    chmod 755 "$SCRIPTS_DIR/postinstall"

    # Keep the package payload independent of the builder's local umask.
    find "$INSTALL_ROOT" -type d -exec chmod 755 {} +
    find "$INSTALL_ROOT" -type f -exec chmod 644 {} +
    chmod 755 "$INSTALL_ROOT/usr/local/bin/nvbroadcast" "$SCRIPTS_DIR/postinstall"

    # Build component package
    mkdir -p dist/pkg
    pkgbuild \
        --root "$INSTALL_ROOT" \
        --ownership recommended \
        --identifier "com.doczeus.nvbroadcast" \
        --version "${VERSION}.${REV}" \
        --scripts "$SCRIPTS_DIR" \
        --install-location "/" \
        "${BUILD_DIR}/nvbroadcast-component.pkg"

    # Build product package (adds welcome/license UI)
    cat > "${BUILD_DIR}/distribution.xml" << DIST
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>NV Broadcast ${VERSION}</title>
    <organization>com.doczeus</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="true" rootVolumeOnly="true" hostArchitectures="arm64"/>
    <volume-check>
        <allowed-os-versions>
            <os-version min="13.0"/>
        </allowed-os-versions>
    </volume-check>
    <choices-outline>
        <line choice="default">
            <line choice="com.doczeus.nvbroadcast"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="com.doczeus.nvbroadcast" visible="false">
        <pkg-ref id="com.doczeus.nvbroadcast"/>
    </choice>
    <pkg-ref id="com.doczeus.nvbroadcast" version="${VERSION}.${REV}" onConclusion="none">nvbroadcast-component.pkg</pkg-ref>
</installer-gui-script>
DIST

    productbuild \
        --distribution "${BUILD_DIR}/distribution.xml" \
        --package-path "$BUILD_DIR" \
        "dist/pkg/NVBroadcast-${VERSION}-${REV}.pkg"

    echo "[PKG] Built: dist/pkg/NVBroadcast-${VERSION}-${REV}.pkg"
    rm -rf "$BUILD_DIR"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

case "$BUILD_TARGET" in
    deb) build_deb ;;
    rpm) build_rpm ;;
    pkg) build_pkg ;;
    all)
        build_deb; echo ""
        build_rpm; echo ""
        build_pkg
        ;;
    *)   echo "Usage: $0 [deb|rpm|pkg|all]"; exit 1 ;;
esac

echo ""
echo "========================================="
echo "  Packages built: v${VERSION}-${REV}"
echo "========================================="
ls -lh dist/deb/*.deb dist/rpm/*.rpm dist/pkg/*.pkg 2>/dev/null || true
echo ""
echo "  Install .deb:  sudo dpkg -i dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb && sudo apt -f install"
echo "  Install .rpm:  sudo dnf install dist/rpm/nvbroadcast-${VERSION}-${REV}*.rpm"
echo "  Install .pkg:  open dist/pkg/NVBroadcast-${VERSION}-${REV}.pkg  (macOS)"

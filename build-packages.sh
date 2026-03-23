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

# Revision tracking (increments per build, resets on version bump)
REV_FILE="packaging/.revision"
LAST_VER=""
REV=1
if [ -f "$REV_FILE" ]; then
    LAST_VER=$(head -1 "$REV_FILE" | cut -d: -f1)
    LAST_REV=$(head -1 "$REV_FILE" | cut -d: -f2)
    if [ "$LAST_VER" = "$VERSION" ]; then
        REV=$((LAST_REV + 1))
    fi
fi
echo "${VERSION}:${REV}" > "$REV_FILE"

echo "========================================="
echo "  NV Broadcast Package Builder"
echo "  Version: ${VERSION}-${REV}"
echo "========================================="
echo ""

BUILD_TARGET="${1:-all}"

# ─── Build .deb ───────────────────────────────────────────────────────────────

build_deb() {
    echo "[DEB] Building .deb package..."

    local BUILD_DIR="/tmp/nvbroadcast-deb-build"
    local PKG_DIR="${BUILD_DIR}/nvbroadcast_${VERSION}-${REV}_all"
    rm -rf "$BUILD_DIR"
    mkdir -p "$PKG_DIR/DEBIAN"

    # Generate binary control file (strip source-only fields, add version)
    cat > "$PKG_DIR/DEBIAN/control" << CTRL
Package: nvbroadcast
Version: ${VERSION}-${REV}
Architecture: all
Maintainer: doczeus <harshit@kshoonya.com>
Depends: python3 (>= 3.11), python3-venv, python3-gi, python3-gi-cairo, gir1.2-gtk-4.0, gir1.2-adw-1, gir1.2-gstreamer-1.0, gir1.2-gst-plugins-base-1.0, gstreamer1.0-plugins-base, gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad, v4l-utils, v4l2loopback-dkms, psmisc
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
    chmod 755 "$PKG_DIR/DEBIAN/postinst" "$PKG_DIR/DEBIAN/prerm" "$PKG_DIR/DEBIAN/postrm"

    # Application files -> /opt/nvbroadcast
    install -d "$PKG_DIR/opt/nvbroadcast"
    cp -r src pyproject.toml requirements.txt LICENSE README.md "$PKG_DIR/opt/nvbroadcast/"
    find "$PKG_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    install -d "$PKG_DIR/opt/nvbroadcast/models"
    cp -r data "$PKG_DIR/opt/nvbroadcast/"
    [ -d configs ] && cp -r configs "$PKG_DIR/opt/nvbroadcast/" || true

    # Desktop entry
    install -d "$PKG_DIR/usr/share/applications"
    cp data/com.doczeus.NVBroadcast.desktop "$PKG_DIR/usr/share/applications/"
    sed -i "s|Exec=nvbroadcast|Exec=/usr/bin/nvbroadcast|g" "$PKG_DIR/usr/share/applications/com.doczeus.NVBroadcast.desktop"

    # Icon
    install -d "$PKG_DIR/usr/share/icons/hicolor/scalable/apps"
    cp data/icons/com.doczeus.NVBroadcast.svg "$PKG_DIR/usr/share/icons/hicolor/scalable/apps/"

    # Launcher scripts
    install -d "$PKG_DIR/usr/bin"
    cat > "$PKG_DIR/usr/bin/nvbroadcast" << 'LAUNCHER'
#!/bin/bash
exec /opt/nvbroadcast/.venv/bin/python -m nvbroadcast "$@"
LAUNCHER
    chmod 755 "$PKG_DIR/usr/bin/nvbroadcast"

    cat > "$PKG_DIR/usr/bin/nvbroadcast-vcam" << 'LAUNCHER'
#!/bin/bash
exec /opt/nvbroadcast/.venv/bin/python -m nvbroadcast.vcam_service "$@"
LAUNCHER
    chmod 755 "$PKG_DIR/usr/bin/nvbroadcast-vcam"

    # Systemd service
    install -d "$PKG_DIR/usr/lib/systemd/user"
    cat > "$PKG_DIR/usr/lib/systemd/user/nvbroadcast-vcam.service" << 'SVC'
[Unit]
Description=NV Broadcast Virtual Camera Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=/usr/bin/nvbroadcast-vcam
Restart=on-failure
RestartSec=3

[Install]
WantedBy=graphical-session.target
SVC

    # Build .deb
    mkdir -p dist/deb
    dpkg-deb --build "$PKG_DIR" "dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb"

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

    local RPM_DIR="/tmp/nvbroadcast-rpm-build"
    rm -rf "$RPM_DIR"
    mkdir -p "$RPM_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

    # Create source tarball
    local TAR_DIR="nvbroadcast-${VERSION}"
    local TAR_PATH="$RPM_DIR/SOURCES/${TAR_DIR}.tar.gz"
    mkdir -p "/tmp/${TAR_DIR}"
    cp -r src pyproject.toml requirements.txt LICENSE README.md data "/tmp/${TAR_DIR}/"
    [ -d configs ] && cp -r configs "/tmp/${TAR_DIR}/" || true
    (cd /tmp && tar czf "$TAR_PATH" "$TAR_DIR")
    rm -rf "/tmp/${TAR_DIR}"

    # Copy and update spec with current version
    sed "s/^Version:.*/Version:        ${VERSION}/" packaging/rpm/nvbroadcast.spec | \
        sed "s/^Release:.*/Release:        ${REV}%{?dist}/" > "$RPM_DIR/SPECS/nvbroadcast.spec"

    # Build
    rpmbuild --define "_topdir $RPM_DIR" -bb "$RPM_DIR/SPECS/nvbroadcast.spec" 2>&1 | tail -5

    # Copy output
    mkdir -p dist/rpm
    find "$RPM_DIR/RPMS" -name "*.rpm" -exec cp {} dist/rpm/ \;

    echo "[RPM] Built:"
    ls -la dist/rpm/nvbroadcast-*.rpm 2>/dev/null || echo "  (no RPM found — check build errors above)"

    rm -rf "$RPM_DIR"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

case "$BUILD_TARGET" in
    deb) build_deb ;;
    rpm) build_rpm ;;
    all) build_deb; echo ""; build_rpm ;;
    *)   echo "Usage: $0 [deb|rpm|all]"; exit 1 ;;
esac

echo ""
echo "========================================="
echo "  Packages built: v${VERSION}-${REV}"
echo "========================================="
ls -lh dist/deb/*.deb dist/rpm/*.rpm 2>/dev/null
echo ""
echo "  Install .deb: sudo dpkg -i dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb && sudo apt -f install"
echo "  Install .rpm: sudo dnf install dist/rpm/nvbroadcast-${VERSION}-${REV}*.rpm"

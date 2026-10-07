#!/usr/bin/env bash
# NV Broadcast - Package Builder
# Builds .deb and .rpm packages from the current source tree.
# Version is read from pyproject.toml automatically.
#
# Usage:
#   ./build-packages.sh          # Build both .deb and .rpm
#   ./build-packages.sh deb      # Build .deb only
#   ./build-packages.sh rpm      # Build .rpm only
#   ./build-packages.sh upgrade-helper  # Bind upgrader to built .deb and .rpm
#
# Output:
#   dist/deb/nvbroadcast_<version>-<rev>_all.deb
#   dist/rpm/nvbroadcast-<version>-<rev>[.<dist>].noarch.rpm

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
BUILT_RPM_PATH=""

package_source_date_epoch() {
    local epoch="${SOURCE_DATE_EPOCH:-}"
    if [ -z "$epoch" ]; then
        epoch="$(python3 - <<'PY'
from email.utils import parsedate_to_datetime
from pathlib import Path

entry = next(
    line for line in Path("packaging/debian/changelog").read_text().splitlines()
    if line.startswith(" -- ")
)
print(int(parsedate_to_datetime(entry.rsplit("  ", 1)[-1]).timestamp()))
PY
)"
    fi
    if [[ ! "$epoch" =~ ^[0-9]+$ ]]; then
        echo "ERROR: SOURCE_DATE_EPOCH must be a non-negative Unix timestamp" >&2
        return 1
    fi
    printf '%s\n' "$epoch"
}

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
Depends: python3 (>= 3.11), python3-venv, python3-gi, python3-gi-cairo, gir1.2-gtk-4.0, gir1.2-adw-1, gir1.2-gstreamer-1.0, gir1.2-gst-plugins-base-1.0, gstreamer1.0-plugins-base, gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad, gstreamer1.0-plugins-ugly, v4l-utils, v4l2loopback-dkms, psmisc, pipewire-bin | pipewire-utils, pulseaudio-utils
Recommends: gir1.2-ayatanaappindicator3-0.1
Homepage: https://nvbroadcast.com
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
    cp -r src pyproject.toml LICENSE NOTICE README.md CONTRIBUTORS.md \
        "$PKG_DIR/opt/nvbroadcast/"
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
    install -m 644 NOTICE CONTRIBUTORS.md "$PKG_DIR/usr/share/doc/nvbroadcast/"
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
        "$PKG_DIR/opt/nvbroadcast/scripts/install_runtime_variant.py" \
        "$PKG_DIR/usr/bin/nvbroadcast" \
        "$PKG_DIR/usr/bin/nvbroadcast-vcam"

    # Build .deb
    mkdir -p dist/deb
    # dpkg-deb uses SOURCE_DATE_EPOCH to clamp generated file and archive
    # timestamps. Use the Debian changelog date for direct local builds too.
    local deb_source_date_epoch
    deb_source_date_epoch="$(package_source_date_epoch)"
    SOURCE_DATE_EPOCH="$deb_source_date_epoch" dpkg-deb -Zxz --root-owner-group --build \
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
        echo "[RPM] ERROR: rpmbuild not found. Install with: sudo apt install rpm" >&2
        return 1
    fi

    local rpm_source_date_epoch
    rpm_source_date_epoch="$(package_source_date_epoch)"

    local RPM_DIR
    RPM_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nvbroadcast-rpm-build.XXXXXX")
    mkdir -p "$RPM_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

    # Create source tarball
    local TAR_DIR="nvbroadcast-${VERSION}"
    local TAR_PATH="$RPM_DIR/SOURCES/${TAR_DIR}.tar.gz"
    local TAR_ROOT="$RPM_DIR/source"
    mkdir -p "$TAR_ROOT/$TAR_DIR"
    cp -r src pyproject.toml LICENSE NOTICE README.md CONTRIBUTORS.md data \
        "$TAR_ROOT/$TAR_DIR/"
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
    if ! SOURCE_DATE_EPOCH="$rpm_source_date_epoch" rpmbuild \
        --nodeps \
        --define "_topdir $RPM_DIR" \
        --define "_userunitdir /usr/lib/systemd/user" \
        --define "_buildhost nvbroadcast" \
        --define "source_date_epoch_from_changelog 0" \
        --define "use_source_date_epoch_as_buildtime 1" \
        --define "clamp_mtime_to_source_date_epoch 1" \
        -bb "$RPM_DIR/SPECS/nvbroadcast.spec" > "$RPM_DIR/rpmbuild.log" 2>&1; then
        tail -30 "$RPM_DIR/rpmbuild.log" >&2
        rm -rf "$RPM_DIR"
        return 1
    fi
    tail -5 "$RPM_DIR/rpmbuild.log"

    # Require an artifact produced by this invocation, even when dist/rpm
    # contains a package from an earlier build.
    local rpm_file
    rpm_file="$(find "$RPM_DIR/RPMS" -type f \
        -name "nvbroadcast-${VERSION}-${REV}*.noarch.rpm" -print -quit)"
    if [ -z "$rpm_file" ] || [ ! -s "$rpm_file" ]; then
        echo "[RPM] ERROR: rpmbuild produced no RPM artifact" >&2
        rm -rf "$RPM_DIR"
        return 1
    fi

    # Copy output
    mkdir -p dist/rpm
    cp "$rpm_file" dist/rpm/
    BUILT_RPM_PATH="dist/rpm/$(basename "$rpm_file")"

    echo "[RPM] Built: $BUILT_RPM_PATH"

    rm -rf "$RPM_DIR"
}

# ─── Render native-package upgrade helper ────────────────────────────────────

build_upgrade_helper() {
    local DEB_PATH="dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb"
    local RPM_PATH="$BUILT_RPM_PATH"
    local -a rpm_candidates=()

    # `all` binds the RPM built in this invocation. A standalone helper build
    # requires exactly one matching artifact, including any RPM dist suffix.
    if [ -z "$RPM_PATH" ]; then
        if [ -d dist/rpm ]; then
            mapfile -d '' -t rpm_candidates < <(find dist/rpm -maxdepth 1 \
                -type f -name "nvbroadcast-${VERSION}-${REV}*.noarch.rpm" -print0)
        fi
        if [ "${#rpm_candidates[@]}" -ne 1 ]; then
            echo "[UPGRADE] ERROR: Expected one matching RPM artifact; found ${#rpm_candidates[@]}." >&2
            return 1
        fi
        RPM_PATH="${rpm_candidates[0]}"
    fi

    if [ ! -f "$DEB_PATH" ] || [ ! -f "$RPM_PATH" ]; then
        echo "[UPGRADE] ERROR: Build the exact .deb and .rpm before the upgrade helper."
        return 1
    fi

    python3 scripts/render_native_upgrade_helper.py \
        --template scripts/native_package_upgrade.sh.in \
        --deb "$DEB_PATH" \
        --rpm "$RPM_PATH" \
        --version "$VERSION" \
        --revision "$REV" \
        --output dist/nvbroadcast-native-upgrade
    echo "[UPGRADE] Built: dist/nvbroadcast-native-upgrade"
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
    mkdir -p "$INSTALL_ROOT/opt/nvbroadcast/scripts"
    mkdir -p "$INSTALL_ROOT/usr/local/bin"
    mkdir -p "$SCRIPTS_DIR"

    # Application files -> /opt/nvbroadcast
    cp -r src pyproject.toml LICENSE NOTICE README.md CONTRIBUTORS.md \
        "$INSTALL_ROOT/opt/nvbroadcast/"
    install -m 755 scripts/install_runtime_variant.py \
        "$INSTALL_ROOT/opt/nvbroadcast/scripts/install_runtime_variant.py"
    find "$INSTALL_ROOT/opt/nvbroadcast/src" -type d \
        \( -name "__pycache__" -o -name "*.egg-info" \) \
        -prune -exec rm -rf {} +
    mkdir -p "$INSTALL_ROOT/opt/nvbroadcast/models"
    cp -r data "$INSTALL_ROOT/opt/nvbroadcast/"
    [ -d configs ] && cp -r configs "$INSTALL_ROOT/opt/nvbroadcast/" || true
    cp install_macos.sh "$INSTALL_ROOT/opt/nvbroadcast/"

    install -m 755 scripts/setup_macos_runtime.sh \
        "$INSTALL_ROOT/opt/nvbroadcast/scripts/setup_macos_runtime.sh"
    printf '%s\n' "${VERSION}-${REV}" > \
        "$INSTALL_ROOT/opt/nvbroadcast/macos-package-version"

    # A package never runs Homebrew or its user-owned Python as Installer root.
    # The logged-in user completes runtime setup explicitly after installation.
    cat > "$INSTALL_ROOT/usr/local/bin/nvbroadcast" << 'LAUNCHER'
#!/bin/bash
set -euo pipefail
if (( EUID == 0 )); then
    echo "[NV Broadcast] ERROR: Launch as your logged-in user, without sudo." >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
INSTALL_DIR="/opt/nvbroadcast"
PACKAGE_ID=$(<"$INSTALL_DIR/macos-package-version")
if [[ ! "$PACKAGE_ID" =~ ^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+$ ]]; then
    echo "[NV Broadcast] ERROR: Invalid package version. Reinstall the package." >&2
    exit 1
fi
RUNTIME_ID=$(<"$INSTALL_DIR/macos-runtime-id")
if [[ ! "$RUNTIME_ID" =~ ^[a-f0-9]{64}$ ]]; then
    echo "[NV Broadcast] ERROR: Invalid package runtime identity. Reinstall the package." >&2
    exit 1
fi
RUNTIME_DIR="$HOME/Library/Application Support/NVBroadcast/$PACKAGE_ID-$RUNTIME_ID"
if [[ ! -x "$RUNTIME_DIR/.venv/bin/python" || \
      ! -f "$RUNTIME_DIR/runtime-ready" || \
      "$(<"$RUNTIME_DIR/runtime-ready")" != "$PACKAGE_ID:$RUNTIME_ID" ]]; then
    echo "[NV Broadcast] ERROR: The per-user runtime is not ready." >&2
    echo "Run without sudo: $INSTALL_DIR/scripts/setup_macos_runtime.sh" >&2
    exit 1
fi
export GST_PLUGIN_PATH="/opt/homebrew/lib/gstreamer-1.0"
export GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0"
cd "$INSTALL_DIR"
exec "$RUNTIME_DIR/.venv/bin/python" -m nvbroadcast "$@"
LAUNCHER
    chmod 755 "$INSTALL_ROOT/usr/local/bin/nvbroadcast"

    # Reject redirected or user-writable payload destinations before root writes.
    cat > "$SCRIPTS_DIR/preinstall" << 'PREINST'
#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
if [[ "${3:-/}" != "/" ]]; then
    echo "[NV Broadcast] ERROR: Install on the running system volume (/)." >&2
    exit 1
fi
if [[ "$(/usr/bin/uname -m)" != "arm64" ]]; then
    echo "[NV Broadcast] ERROR: This package supports Apple Silicon Macs only." >&2
    exit 1
fi
MACOS_VERSION=$(/usr/bin/sw_vers -productVersion)
if [[ ! "$MACOS_VERSION" =~ ^([0-9]+)\. || "${BASH_REMATCH[1]}" -lt 13 ]]; then
    echo "[NV Broadcast] ERROR: macOS 13 (Ventura) or newer required." >&2
    exit 1
fi
check_destination() {
    local path="$1" owner mode access_details
    if [[ -L "$path" ]]; then
        echo "[NV Broadcast] ERROR: Refusing symlinked package destination: $path" >&2
        exit 1
    fi
    if [[ -e "$path" ]]; then
        # BSD ls -e adds ACL entries below its first line. Conservatively reject
        # existing ACLs, including write grants that the POSIX mode omits.
        access_details=$(/bin/ls -lde "$path")
        if [[ "$access_details" == *$'\n'* ]]; then
            echo "[NV Broadcast] ERROR: Existing destination ACL requires administrator review: $path" >&2
            exit 1
        fi
        read -r owner mode < <(/usr/bin/stat -f '%u %Lp' "$path")
        if [[ "$owner" != 0 || ! "$mode" =~ ^[0-7]+$ ]] || \
                (( (8#$mode & 0022) != 0 )); then
            echo "[NV Broadcast] ERROR: Package destination must be admin-owned and not group/other writable: $path" >&2
            echo "Ask an administrator to review this path; the installer will not change ownership." >&2
            exit 1
        fi
    fi
}
for path in /opt /opt/nvbroadcast /usr /usr/local /usr/local/bin \
        /usr/local/bin/nvbroadcast; do
    check_destination "$path"
done
# Existing package subtrees must not redirect payload writes either. The old
# installer-owned .venv is deliberately untouched and is not a new payload path.
for subtree in src scripts data models configs; do
    if [[ -e "/opt/nvbroadcast/$subtree" || -L "/opt/nvbroadcast/$subtree" ]]; then
        while IFS= read -r path; do
            check_destination "$path"
        done < <(/usr/bin/find -P "/opt/nvbroadcast/$subtree" -print)
    fi
done
for path in /opt/nvbroadcast/pyproject.toml /opt/nvbroadcast/LICENSE \
        /opt/nvbroadcast/NOTICE /opt/nvbroadcast/README.md \
        /opt/nvbroadcast/CONTRIBUTORS.md /opt/nvbroadcast/install_macos.sh \
        /opt/nvbroadcast/macos-package-version /opt/nvbroadcast/macos-runtime-id; do
    check_destination "$path"
done
PREINST
    chmod 755 "$SCRIPTS_DIR/preinstall"

    cat > "$SCRIPTS_DIR/postinstall" << 'POSTINST'
#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
if [[ "${3:-/}" != "/" ]]; then
    echo "[NV Broadcast] ERROR: Install on the running system volume (/)." >&2
    exit 1
fi
INSTALL_DIR="/opt/nvbroadcast"
if [[ ! -x "$INSTALL_DIR/scripts/setup_macos_runtime.sh" || \
      ! -f "$INSTALL_DIR/macos-package-version" || \
      ! -f "$INSTALL_DIR/macos-runtime-id" || \
      ! -x /usr/local/bin/nvbroadcast ]]; then
    echo "[NV Broadcast] ERROR: Package payload is incomplete. Reinstall the package." >&2
    exit 1
fi
echo "[NV Broadcast] Application source and launcher installed."
echo "[NV Broadcast] Runtime setup is still required for your logged-in user."
echo "[NV Broadcast] Run without sudo: $INSTALL_DIR/scripts/setup_macos_runtime.sh"
echo "[NV Broadcast] Then launch: nvbroadcast"
POSTINST
    chmod 755 "$SCRIPTS_DIR/postinstall"

    # Keep the package payload independent of the builder's local umask.
    find "$INSTALL_ROOT" -type d -exec chmod 755 {} +
    find "$INSTALL_ROOT" -type f -exec chmod 644 {} +
    chmod 755 \
        "$INSTALL_ROOT/usr/local/bin/nvbroadcast" \
        "$INSTALL_ROOT/opt/nvbroadcast/scripts/install_runtime_variant.py" \
        "$INSTALL_ROOT/opt/nvbroadcast/scripts/setup_macos_runtime.sh" \
        "$SCRIPTS_DIR/preinstall" \
        "$SCRIPTS_DIR/postinstall"

    # Bind per-user runtimes to this exact normalized package source. A rebuilt
    # candidate with the same version must not silently reuse a stale venv.
    python3 - "$INSTALL_ROOT/opt/nvbroadcast" <<'IDENTITY'
from pathlib import Path
import hashlib
import stat
import sys

root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"Unexpected package source symlink: {path}")
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix().encode()
    if relative == b"macos-runtime-id":
        continue
    payload = path.read_bytes()
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(stat.S_IMODE(path.stat().st_mode).to_bytes(4, "big"))
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
(root / "macos-runtime-id").write_text(digest.hexdigest() + "\n")
IDENTITY
    chmod 644 "$INSTALL_ROOT/opt/nvbroadcast/macos-runtime-id"

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
    upgrade-helper) build_upgrade_helper ;;
    all)
        build_deb; echo ""
        build_rpm; echo ""
        build_upgrade_helper; echo ""
        build_pkg
        ;;
    *)   echo "Usage: $0 [deb|rpm|pkg|upgrade-helper|all]"; exit 1 ;;
esac

echo ""
echo "========================================="
echo "  Packages built: v${VERSION}-${REV}"
echo "========================================="
ls -lh dist/deb/*.deb dist/rpm/*.rpm dist/pkg/*.pkg dist/nvbroadcast-native-upgrade 2>/dev/null || true
echo ""
echo "  Install .deb:  sudo dpkg -i dist/deb/nvbroadcast_${VERSION}-${REV}_all.deb && sudo apt -f install"
echo "  Install .rpm:  sudo dnf install dist/rpm/nvbroadcast-${VERSION}-${REV}*.rpm"
echo "  Install .pkg:  open dist/pkg/NVBroadcast-${VERSION}-${REV}.pkg  (macOS)"
echo "  Upgrade native Linux package: sudo dist/nvbroadcast-native-upgrade <package>"

#!/usr/bin/env bash
# NV Broadcast - macOS Camera Extension Builder
# Copyright (c) 2026 doczeus. Proprietary license (see macos/LICENSE).
#
# Builds the CoreMediaIO Camera Extension without Xcode GUI.
# Uses xcodebuild with a generated project or swift build for the helper.
#
# Requirements: macOS 12.3+, Xcode CLT or Xcode.app
#
# Usage:
#   ./macos/build.sh                    # Build extension + helper
#   ./macos/build.sh --sign "Dev ID"    # Build + code sign for distribution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/.build"
SIGN_IDENTITY="${1:-}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════╗"
echo "║  NV Broadcast — macOS Extension Builder      ║"
echo "║  Proprietary — doczeus                       ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Pre-flight ───────────────────────────────────────────────────────────────

if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: Must run on macOS${NC}"
    exit 1
fi

if ! command -v xcodebuild &>/dev/null; then
    echo -e "${RED}Error: Xcode Command Line Tools required${NC}"
    echo "Install: xcode-select --install"
    exit 1
fi

XCODE_VER=$(xcodebuild -version 2>/dev/null | head -1 || echo "unknown")
MACOS_VER=$(sw_vers -productVersion)
echo "  Xcode: $XCODE_VER"
echo "  macOS: $MACOS_VER"
echo "  Arch:  $(uname -m)"
echo ""

# ── Generate Xcode project ──────────────────────────────────────────────────

echo -e "${GREEN}[1/4]${NC} Generating Xcode project..."

PROJECT_DIR="$BUILD_DIR/NVBroadcast.xcodeproj"
rm -rf "$BUILD_DIR"
mkdir -p "$PROJECT_DIR"

# Read version from pyproject.toml
VERSION=$(python3 -c "
import re
with open('$ROOT_DIR/pyproject.toml') as f:
    m = re.search(r'version\s*=\s*\"(.+?)\"', f.read())
    print(m.group(1))
" 2>/dev/null || echo "0.2.0")

# Copy sources into build dir
EXTENSION_SRC="$BUILD_DIR/NVBroadcastExtension"
mkdir -p "$EXTENSION_SRC"
cp "$SCRIPT_DIR/NVBroadcastExtension/"*.swift "$EXTENSION_SRC/"
cp "$SCRIPT_DIR/Shared/"*.swift "$EXTENSION_SRC/"
cp "$SCRIPT_DIR/NVBroadcastExtension/Info.plist" "$EXTENSION_SRC/"
cp "$SCRIPT_DIR/NVBroadcastExtension/NVBroadcastExtension.entitlements" "$EXTENSION_SRC/"

# Generate pbxproj using xcodegen-style approach via swift package
# For CI simplicity, compile directly with swiftc
echo "  Sources ready"

# ── Compile extension ────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}[2/4]${NC} Compiling Camera Extension..."

EXTENSION_OUT="$BUILD_DIR/NVBroadcastExtension.systemextension"
EXTENSION_BIN="$EXTENSION_OUT/Contents/MacOS/NVBroadcastExtension"

mkdir -p "$EXTENSION_OUT/Contents/MacOS"
mkdir -p "$EXTENSION_OUT/Contents"

# Compile all Swift files into a single binary
SWIFT_FILES=(
    "$EXTENSION_SRC/Constants.swift"
    "$EXTENSION_SRC/ExtensionProvider.swift"
    "$EXTENSION_SRC/DeviceSource.swift"
    "$EXTENSION_SRC/StreamSource.swift"
    "$EXTENSION_SRC/main.swift"
)

FRAMEWORKS=(
    "-framework" "Foundation"
    "-framework" "CoreMediaIO"
    "-framework" "CoreMedia"
    "-framework" "CoreVideo"
    "-framework" "IOSurface"
)

# Detect minimum deployment target
DEPLOY_TARGET="12.3"

swiftc \
    -O \
    -target "$(uname -m)-apple-macosx${DEPLOY_TARGET}" \
    -module-name NVBroadcastExtension \
    "${SWIFT_FILES[@]}" \
    "${FRAMEWORKS[@]}" \
    -o "$EXTENSION_BIN" \
    2>&1

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Compilation failed${NC}"
    exit 1
fi

# Copy Info.plist
cp "$EXTENSION_SRC/Info.plist" "$EXTENSION_OUT/Contents/"

# Set version in Info.plist
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$EXTENSION_OUT/Contents/Info.plist" 2>/dev/null || true

echo "  Compiled: $EXTENSION_BIN"
echo "  Size: $(du -sh "$EXTENSION_BIN" | cut -f1)"

# ── Code sign ────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}[3/4]${NC} Code signing..."

if [[ "$SIGN_IDENTITY" == "--sign" && -n "${2:-}" ]]; then
    IDENTITY="$2"
    echo "  Signing with: $IDENTITY"
    codesign --force --sign "$IDENTITY" \
        --entitlements "$EXTENSION_SRC/NVBroadcastExtension.entitlements" \
        --options runtime \
        --timestamp \
        "$EXTENSION_OUT"
    echo "  Signed and timestamped"
elif [[ -n "$SIGN_IDENTITY" && "$SIGN_IDENTITY" != "--sign" ]]; then
    echo "  Signing with: $SIGN_IDENTITY"
    codesign --force --sign "$SIGN_IDENTITY" \
        --entitlements "$EXTENSION_SRC/NVBroadcastExtension.entitlements" \
        --options runtime \
        --timestamp \
        "$EXTENSION_OUT"
    echo "  Signed and timestamped"
else
    echo "  Ad-hoc signing (development only)"
    codesign --force --sign - \
        --entitlements "$EXTENSION_SRC/NVBroadcastExtension.entitlements" \
        "$EXTENSION_OUT"
    echo -e "  ${YELLOW}Note: Distribution requires: ./macos/build.sh --sign \"Developer ID Application: Name\"${NC}"
fi

# ── Package ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}[4/4]${NC} Creating distribution..."

DIST_DIR="$ROOT_DIR/dist/macos"
mkdir -p "$DIST_DIR"

# Copy extension to dist
cp -r "$EXTENSION_OUT" "$DIST_DIR/"

# Copy Python frame bridge helper
cp "$SCRIPT_DIR/NVBroadcastHelper/frame_bridge.py" "$DIST_DIR/"

# Create install script
cat > "$DIST_DIR/install-extension.sh" << 'INSTALL'
#!/usr/bin/env bash
# Install NV Broadcast Camera Extension
set -e

EXT_DIR="/Library/SystemExtensions"
EXT_NAME="NVBroadcastExtension.systemextension"

echo "Installing NV Broadcast Camera Extension..."
echo "This requires administrator privileges."

sudo mkdir -p "$EXT_DIR"
sudo cp -r "$EXT_NAME" "$EXT_DIR/"

echo ""
echo "Extension installed. It will appear as 'NV Broadcast' in video apps."
echo "You may need to approve it in System Settings > Privacy & Security."
INSTALL
chmod +x "$DIST_DIR/install-extension.sh"

echo "  Extension: $DIST_DIR/NVBroadcastExtension.systemextension"
echo "  Helper:    $DIST_DIR/frame_bridge.py"
echo "  Installer: $DIST_DIR/install-extension.sh"

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗"
echo "║  Build complete: v${VERSION}                    ║"
echo "╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Test locally:  cd dist/macos && sudo ./install-extension.sh"
echo "  Distribute:    ./macos/build.sh --sign \"Developer ID Application: ...\""
echo ""

#!/bin/bash
# Complete a source PKG installation as the logged-in user, never Installer root.
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
umask 077

if (( EUID == 0 )); then
    echo "[NV Broadcast] ERROR: Run runtime setup as your logged-in user, without sudo." >&2
    exit 1
fi
if [[ "$(/usr/bin/uname -s)" != "Darwin" || "$(/usr/bin/uname -m)" != "arm64" ]]; then
    echo "[NV Broadcast] ERROR: Runtime setup requires an Apple Silicon Mac." >&2
    exit 1
fi
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
BREW="/opt/homebrew/bin/brew"
prerequisite_guidance() {
    echo "[NV Broadcast] Install prerequisites as your logged-in user, without sudo:" >&2
    echo "  /opt/homebrew/bin/brew update" >&2
    echo "  /opt/homebrew/bin/brew install python@3.13 pygobject3 gtk4 libadwaita gstreamer pkgconf ninja" >&2
    echo "  /opt/homebrew/bin/brew upgrade python@3.13" >&2
    echo "If private bindings need compiling, install Apple's Command Line Tools: xcode-select --install" >&2
}
if [[ ! -x "$BREW" ]]; then
    echo "[NV Broadcast] ERROR: Apple Silicon Homebrew is required at /opt/homebrew." >&2
    echo "Install Homebrew from https://brew.sh, then rerun setup." >&2
    exit 1
fi
BREW_PREFIX=$("$BREW" --prefix)
if [[ "$BREW_PREFIX" != "/opt/homebrew" ]]; then
    echo "[NV Broadcast] ERROR: This package requires the Apple Silicon /opt/homebrew prefix." >&2
    exit 1
fi
export PATH="$BREW_PREFIX/bin:$PATH"
export GST_PLUGIN_PATH="$BREW_PREFIX/lib/gstreamer-1.0"
export GI_TYPELIB_PATH="$BREW_PREFIX/lib/girepository-1.0"

# Keep support for secure, compatible system bindings on Python 3.11-3.13.
# Current Homebrew PyGObject formulae can target newer Python ABIs, so a secure
# CPython 3.13 may instead receive private bindings inside this user's venv.
check_python_version() {
    "$1" - <<'PY'
import sys
# PSF's October 2026 certificate incident mitigation; check before imports.
minimum_patch = {(3, 11): 18, (3, 12): 16, (3, 13): 18}.get(sys.version_info[:2])
if minimum_patch is None or sys.version_info[2] < minimum_patch:
    raise SystemExit("Python 3.11.18+, 3.12.16+, or 3.13.18+ required; update Homebrew Python")
PY
}
check_native_stack() {
    "$1" - <<'PY'
import cairo
import gi
for namespace, version in (("Gtk", "4.0"), ("Adw", "1"), ("Gst", "1.0"),
                           ("GstApp", "1.0"), ("GstVideo", "1.0")):
    gi.require_version(namespace, version)
from gi.repository import Adw, Gst, GstApp, GstVideo, Gtk
Gst.init([])
required = ("avfvideosrc", "osxaudiosrc", "osxaudiosink", "videoconvert",
            "audioconvert", "audioresample", "x264enc", "h264parse", "mp4mux")
missing = [name for name in required if Gst.ElementFactory.find(name) is None]
if Gst.DeviceProviderFactory.find("osxaudiodeviceprovider") is None:
    missing.append("osxaudiodeviceprovider")
if not any(Gst.ElementFactory.find(name) for name in ("avenc_aac", "voaacenc")):
    missing.append("avenc_aac or voaacenc")
if missing:
    raise SystemExit("Missing GStreamer plugins: " + ", ".join(missing))
PY
}
PYTHON=""
PRIVATE_BINDINGS_PYTHON=""
for minor in 13 12 11; do
    candidate="$BREW_PREFIX/opt/python@3.$minor/bin/python3.$minor"
    if [[ ! -x "$candidate" ]] || ! check_python_version "$candidate"; then
        continue
    fi
    if check_native_stack "$candidate"; then
        PYTHON="$candidate"
        break
    fi
    if [[ "$minor" == 13 ]]; then
        PRIVATE_BINDINGS_PYTHON="$candidate"
    fi
    echo "[NV Broadcast] $candidate does not provide the required Python/GI/GStreamer stack." >&2
done
PRIVATE_BINDINGS=0
if [[ -z "$PYTHON" && -n "$PRIVATE_BINDINGS_PYTHON" ]]; then
    # Build only for the app's supported CPython ABI, never Homebrew's generic
    # python3. System packages and bindings for other interpreters stay intact.
    export PKG_CONFIG_PATH="$BREW_PREFIX/lib/pkgconfig:$BREW_PREFIX/opt/libffi/lib/pkgconfig"
    if [[ ! -x "$BREW_PREFIX/bin/pkg-config" || ! -x "$BREW_PREFIX/bin/ninja" ]] || \
       ! "$BREW_PREFIX/bin/pkg-config" --exists cairo girepository-2.0 gtk4 libadwaita-1 gstreamer-1.0; then
        echo "[NV Broadcast] ERROR: Private Python bindings require Homebrew native headers, pkgconf and ninja." >&2
        prerequisite_guidance
        exit 1
    fi
    PYTHON="$PRIVATE_BINDINGS_PYTHON"
    PRIVATE_BINDINGS=1
fi
if [[ -z "$PYTHON" ]]; then
    echo "[NV Broadcast] ERROR: A supported Python with working GTK/Adw/GStreamer bindings is required." >&2
    prerequisite_guidance
    exit 1
fi

# Never delete or replace an existing user environment. A failed setup leaves no
# ready marker; the launcher cannot accidentally start a partial installation.
BASE_DIR="$HOME/Library/Application Support/NVBroadcast"
for path in "$HOME/Library" "$HOME/Library/Application Support" "$BASE_DIR"; do
    if [[ -L "$path" ]]; then
        echo "[NV Broadcast] ERROR: Refusing symlinked runtime parent: $path" >&2
        exit 1
    fi
done
mkdir -p "$BASE_DIR"
RUNTIME_DIR="$BASE_DIR/$PACKAGE_ID-$RUNTIME_ID"
if [[ -e "$RUNTIME_DIR" || -L "$RUNTIME_DIR" ]]; then
    if [[ ! -L "$RUNTIME_DIR" && -x "$RUNTIME_DIR/.venv/bin/python" && \
          -f "$RUNTIME_DIR/runtime-ready" && \
          "$(<"$RUNTIME_DIR/runtime-ready")" == "$PACKAGE_ID:$RUNTIME_ID" ]]; then
        "$RUNTIME_DIR/.venv/bin/python" -m nvbroadcast.runtime --variant cpu
        "$RUNTIME_DIR/.venv/bin/python" -m pip check
        echo "[NV Broadcast] This user's runtime is already ready. Run: nvbroadcast"
        exit 0
    fi
    echo "[NV Broadcast] ERROR: Existing runtime is incomplete or redirected: $RUNTIME_DIR" >&2
    echo "Stop your NV Broadcast processes, review/remove only this directory, then rerun setup." >&2
    exit 1
fi
mkdir "$RUNTIME_DIR"
PROJECT_DIR="$RUNTIME_DIR/install-source"
mkdir "$PROJECT_DIR"
# setuptools builds need a writable source directory; the system PKG stays
# admin-owned. Build from a per-user copy of the package's exact source inputs.
cp -R "$INSTALL_DIR/src" "$INSTALL_DIR/data" "$PROJECT_DIR/"
cp "$INSTALL_DIR/pyproject.toml" "$INSTALL_DIR/LICENSE" "$INSTALL_DIR/NOTICE" \
    "$INSTALL_DIR/README.md" "$INSTALL_DIR/CONTRIBUTORS.md" "$PROJECT_DIR/"
"$PYTHON" -m venv "$RUNTIME_DIR/.venv" --system-site-packages
RUNTIME_PYTHON="$RUNTIME_DIR/.venv/bin/python"
# Bootstrap the trust-store fix from the PSF-published wheel hash before resolving packages.
"$RUNTIME_PYTHON" -m pip install --upgrade --no-deps "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
"$RUNTIME_PYTHON" -m pip install --upgrade "pip>=26.2.2" "setuptools>=84.1.0" "certifi>=2026.10.19" wheel
if (( PRIVATE_BINDINGS )); then
    # These pure-Python builder pins are shared with the reviewed native builder.
    # Disable build isolation/dependency resolution so source builds cannot fetch
    # unreviewed build tools. URL hashes authenticate every new build input.
    export PATH="$RUNTIME_DIR/.venv/bin:$PATH"
    "$RUNTIME_PYTHON" - "$INSTALL_DIR/scripts/macos-bindings-inputs.json" "$RUNTIME_DIR" <<'PY'
import json
from pathlib import Path
import sys
inputs = json.loads(Path(sys.argv[1]).read_text())
if inputs["schema_version"] != 1 or inputs["target"] != "macos-arm64-cp313-cpu":
    raise SystemExit("Invalid macOS binding input identity; reinstall the package")
def requirements(items):
    return "".join(
        f'{item["name"]} @ {item["url"]}#sha256={item["sha256"]}\n' for item in items)
output = Path(sys.argv[2])
(output / "binding-build-requirements.txt").write_text(requirements(inputs["bootstrap"]))
# PyGObject metadata needs Pycairo installed before its own build starts.
for name in ("pycairo", "PyGObject"):
    item, = (item for item in inputs["bindings"] if item["name"] == name)
    (output / f"{name.lower()}-requirements.txt").write_text(requirements([item]))
PY
    # Replace any private packaging copy installed by wheel's bootstrap, rather
    # than leaving competing dist-info versions. pip preserves system-site files.
    "$RUNTIME_PYTHON" -m pip install --no-deps --force-reinstall \
        -r "$RUNTIME_DIR/binding-build-requirements.txt"
    # Install into this fresh venv even if an incompatible system distribution
    # is visible through --system-site-packages. Never uninstall system files.
    "$RUNTIME_PYTHON" -m pip install --no-deps --no-build-isolation --ignore-installed \
        -r "$RUNTIME_DIR/pycairo-requirements.txt"
    "$RUNTIME_PYTHON" -m pip install --no-deps --no-build-isolation --ignore-installed \
        -r "$RUNTIME_DIR/pygobject-requirements.txt"
fi
check_native_stack "$RUNTIME_PYTHON"
"$RUNTIME_PYTHON" "$INSTALL_DIR/scripts/install_runtime_variant.py" \
    --project "$PROJECT_DIR" --variant cpu --meeting-backends faster
"$RUNTIME_PYTHON" -m pip check
# Recheck the complete native stack after app dependency installation.
check_native_stack "$RUNTIME_PYTHON"
printf '%s\n' "$PACKAGE_ID:$RUNTIME_ID" > "$RUNTIME_DIR/runtime-ready"
echo "[NV Broadcast] Per-user CPU runtime ready. Run: nvbroadcast"

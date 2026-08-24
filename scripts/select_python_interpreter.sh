#!/usr/bin/env bash
# Select an existing CPython interpreter for the source installer.

set -u

PACKAGE_MANAGER="unknown"
PYTHON_REQUEST=""
REQUIRE_DESKTOP_BINDINGS=false

usage() {
    echo "Usage: $0 [--python /path/to/python] [--package-manager NAME] [--require-desktop-bindings]" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python)
            [ "$#" -ge 2 ] || { echo "ERROR: --python requires an interpreter path." >&2; exit 2; }
            [ -n "$2" ] || { echo "ERROR: --python requires an interpreter path." >&2; exit 2; }
            PYTHON_REQUEST="$2"
            shift 2
            ;;
        --package-manager)
            [ "$#" -ge 2 ] || { echo "ERROR: --package-manager requires a value." >&2; exit 2; }
            PACKAGE_MANAGER="$2"
            shift 2
            ;;
        --require-desktop-bindings)
            REQUIRE_DESKTOP_BINDINGS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

case "$PYTHON_REQUEST" in
    *$'\t'*|*$'\n'*)
        echo "ERROR: --python cannot contain a tab or newline." >&2
        exit 2
        ;;
esac

INSPECT_EXECUTABLE=""
INSPECT_IMPLEMENTATION=""
INSPECT_MAJOR=""
INSPECT_MINOR=""
INSPECT_VERSION=""
INSPECT_REASON=""

inspect_python() {
    local requested="$1"
    local executable details extra directory basename canonical_directory

    INSPECT_EXECUTABLE=""
    INSPECT_IMPLEMENTATION=""
    INSPECT_MAJOR=""
    INSPECT_MINOR=""
    INSPECT_VERSION=""
    INSPECT_REASON="not-found"

    if ! executable="$(command -v -- "$requested" 2>/dev/null)" || [ ! -x "$executable" ]; then
        return 1
    fi

    case "$executable" in
        /*) ;;
        */*)
            directory="${executable%/*}"
            basename="${executable##*/}"
            if canonical_directory="$(cd -- "$directory" 2>/dev/null && pwd -P)"; then
                executable="${canonical_directory}/${basename}"
            fi
            ;;
        *) executable="${PWD}/${executable}" ;;
    esac

    if ! details="$("$executable" -I -c \
        'import platform, sys; print(platform.python_implementation(), sys.version_info.major, sys.version_info.minor, sep="\t")' \
        2>/dev/null)"; then
        INSPECT_REASON="unusable"
        return 1
    fi

    IFS=$'\t' read -r INSPECT_IMPLEMENTATION INSPECT_MAJOR INSPECT_MINOR extra <<< "$details"
    if [ -n "${extra:-}" ] ||
       [[ ! "$INSPECT_MAJOR" =~ ^[0-9]+$ ]] ||
       [[ ! "$INSPECT_MINOR" =~ ^[0-9]+$ ]]; then
        INSPECT_REASON="invalid-version"
        return 1
    fi

    INSPECT_EXECUTABLE="$executable"
    INSPECT_VERSION="${INSPECT_MAJOR}.${INSPECT_MINOR}"

    if [ "$INSPECT_IMPLEMENTATION" != "CPython" ]; then
        INSPECT_REASON="not-cpython"
        return 1
    fi
    if [ "$INSPECT_MAJOR" -ne 3 ] || [ "$INSPECT_MINOR" -lt 11 ] || [ "$INSPECT_MINOR" -gt 13 ]; then
        INSPECT_REASON="unsupported-version"
        return 1
    fi
    if ! "$INSPECT_EXECUTABLE" -I -c \
        'import ensurepip, venv; ensurepip.version()' &>/dev/null; then
        INSPECT_REASON="missing-venv"
        return 1
    fi
    if [ "$REQUIRE_DESKTOP_BINDINGS" = true ] &&
       ! "$INSPECT_EXECUTABLE" -I -c '
import gi
gi.require_version("Adw", "1")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, GdkPixbuf, Gst, GstVideo, Gtk
' &>/dev/null; then
        INSPECT_REASON="missing-desktop-bindings"
        return 1
    fi

    INSPECT_REASON=""
    return 0
}

print_apt_guidance() {
    local version
    if command -v apt-cache &>/dev/null; then
        for version in 3.13 3.12 3.11; do
            if apt-cache show "python${version}" &>/dev/null &&
               apt-cache show "python${version}-venv" &>/dev/null; then
                echo "  sudo apt install python${version} python${version}-venv" >&2
                echo "  ./install.sh --python /usr/bin/python${version}" >&2
                return
            fi
        done
    fi
    echo "  apt-cache policy python3.13 python3.12 python3.11" >&2
    echo "Install the newest listed version and its matching -venv package." >&2
}

print_install_guidance() {
    echo "Install CPython 3.13, 3.12, or 3.11 from your distro's official repositories." >&2
    case "$PACKAGE_MANAGER" in
        apt)
            echo "Ubuntu/Debian family:" >&2
            print_apt_guidance
            ;;
        dnf)
            echo "Fedora/RHEL family:" >&2
            echo "  dnf repoquery python3.13" >&2
            echo "If that package is listed by an enabled official repository:" >&2
            echo "  sudo dnf install python3.13" >&2
            echo "  ./install.sh --python /usr/bin/python3.13" >&2
            echo "Use python3.12 or python3.11 if 3.13 is unavailable in the enabled official repositories." >&2
            ;;
        yum)
            echo "RHEL/CentOS family:" >&2
            echo "  yum list available python3.13" >&2
            echo "If that package is listed by an enabled official repository:" >&2
            echo "  sudo yum install python3.13" >&2
            echo "  ./install.sh --python /usr/bin/python3.13" >&2
            echo "Use python3.12 or python3.11 if 3.13 is unavailable in the enabled official repositories." >&2
            ;;
        zypper)
            echo "openSUSE family:" >&2
            echo "  zypper search --match-exact python313" >&2
            echo "If that package is listed by an enabled official repository:" >&2
            echo "  sudo zypper install python313" >&2
            echo "  ./install.sh --python /usr/bin/python3.13" >&2
            echo "Use the python312 or python311 package if python313 is unavailable." >&2
            ;;
        pacman)
            echo "Arch family:" >&2
            echo "  pacman -Si python" >&2
            echo "  sudo pacman -S python" >&2
            echo "Rerun only if the official package is Python 3.11-3.13; otherwise pass an independently managed interpreter with --python." >&2
            ;;
        portage)
            echo "Gentoo:" >&2
            echo "  sudo emerge --ask dev-lang/python:3.13" >&2
            echo "  ./install.sh --python /usr/bin/python3.13" >&2
            ;;
        xbps)
            echo "Void Linux:" >&2
            echo "  xbps-query -R python3" >&2
            echo "  sudo xbps-install -S python3" >&2
            echo "Rerun only if the official package is Python 3.11-3.13." >&2
            ;;
        nix)
            echo "NixOS:" >&2
            echo "  nix-shell -p python313 --run './install.sh --python \"\$(command -v python3.13)\"'" >&2
            ;;
        *)
            echo "After installation, rerun: ./install.sh --python /path/to/python3" >&2
            ;;
    esac
    echo "The installer will create only this project's .venv and will not replace the system Python." >&2
}

print_desktop_binding_guidance() {
    echo "The installed desktop bindings do not support the selected interpreter." >&2
    case "$PACKAGE_MANAGER" in
        apt) echo "Required distro packages: python3-gi, python3-gi-cairo, gir1.2-gtk-4.0, gir1.2-adw-1, and the GStreamer typelibs." >&2 ;;
        dnf|yum) echo "Required distro packages: python3-gobject, python3-gobject-cairo, gtk4, libadwaita, and GStreamer typelibs." >&2 ;;
        pacman) echo "Required distro packages: python-gobject, gtk4, libadwaita, and GStreamer." >&2 ;;
        zypper) echo "Required distro packages: python3-gobject, python3-gobject-cairo, GTK4/Libadwaita typelibs, and GStreamer." >&2 ;;
        *) echo "Install PyGObject plus the GTK4, Libadwaita, and GStreamer typelibs for this exact interpreter." >&2 ;;
    esac
    echo "Distro bindings usually target the distro's default Python. Choose a compatible interpreter that can import them, or provision PyGObject for the requested interpreter, then rerun." >&2
    echo "The installer will not add a third-party repository or replace the system Python." >&2
}

emit_selection() {
    printf '%s\t%s\t%s\t%s\n' \
        "$INSPECT_EXECUTABLE" "$INSPECT_VERSION" "$INSPECT_MAJOR" "$INSPECT_MINOR"
}

if [ -n "$PYTHON_REQUEST" ]; then
    if inspect_python "$PYTHON_REQUEST"; then
        emit_selection
        exit 0
    fi

    case "$INSPECT_REASON" in
        not-found) echo "ERROR: --python interpreter '$PYTHON_REQUEST' was not found or is not executable." >&2 ;;
        not-cpython) echo "ERROR: --python must select CPython; found ${INSPECT_IMPLEMENTATION:-an unknown implementation}." >&2 ;;
        unsupported-version) echo "ERROR: --python must select CPython 3.11-3.13; found ${INSPECT_VERSION:-an unknown version}." >&2 ;;
        missing-venv) echo "ERROR: $PYTHON_REQUEST is CPython $INSPECT_VERSION but its venv module is unavailable." >&2 ;;
        missing-desktop-bindings)
            echo "ERROR: $PYTHON_REQUEST cannot import the required GTK4, Libadwaita, and GStreamer bindings." >&2
            print_desktop_binding_guidance
            exit 1
            ;;
        *) echo "ERROR: Could not validate --python interpreter '$PYTHON_REQUEST'." >&2 ;;
    esac
    print_install_guidance
    exit 1
fi

SAW_MISSING_DESKTOP_BINDINGS=false
for candidate in python3.13 python3.12 python3.11 python3; do
    if inspect_python "$candidate"; then
        emit_selection
        exit 0
    fi
    if [ "$INSPECT_REASON" = "missing-desktop-bindings" ]; then
        SAW_MISSING_DESKTOP_BINDINGS=true
    fi
done

if [ "$REQUIRE_DESKTOP_BINDINGS" = true ] && [ "$SAW_MISSING_DESKTOP_BINDINGS" = true ]; then
    echo "ERROR: No CPython 3.11-3.13 interpreter can import the required desktop bindings." >&2
    print_desktop_binding_guidance
    exit 1
fi

echo "ERROR: No fully supported CPython interpreter with venv support was found." >&2
print_install_guidance
exit 1

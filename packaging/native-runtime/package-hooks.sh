#!/bin/sh
# Native package payloads are complete. Never resolve dependencies here.
set -eu
base=/usr/lib/nvbroadcast
case "$1" in
    prepare)
        for directory in /usr /usr/lib "$base"; do
            if [ -L "$directory" ]; then
                echo "NVBroadcast: refusing redirected package directory: $directory" >&2
                exit 78
            fi
        done
        # An in-place package upgrade cannot preserve a running interpreter.
        # Refuse before unpacking; never terminate unrelated user processes.
        for process in /proc/[0-9]*/exe; do
            executable=$(readlink "$process" 2>/dev/null || true)
            # procfs can deny exe dereferencing across users without denying
            # cmdline reads (including ordinary container capability sets).
            # Launchers always use an absolute interpreter path. Legacy venv
            # executables can also resolve to a shared system interpreter.
            argument=$(tr '\000' '\n' < "${process%/exe}/cmdline" 2>/dev/null | head -n 1) || argument=''
            for candidate in "$executable" "$argument"; do
                case "$candidate" in
                    /usr/lib/nvbroadcast/runtime/bin/python*|/opt/nvbroadcast/.venv/bin/python*)
                        echo 'NVBroadcast is running. Quit it and its virtual-camera service, then retry.' >&2
                        exit 75
                        ;;
                esac
            done
        done
        mkdir -p "$base"
        printf '%s\n' '@VERSION@' > "$base/.transaction"
        ;;
    finish)
        "$base/check-packages" --configure
        "$base/runtime/bin/python" -I -B "$base/validate-install.py" --variant '@VARIANT@'
        # Legacy generated environments were outside the native file database.
        # Only delete the conventional, unredirected private environment.
        if [ -f "$base/.legacy-runtime" ] && [ ! -L /opt/nvbroadcast ]; then
            rm -rf -- /opt/nvbroadcast/.venv
            rm -f "$base/.legacy-runtime"
            rmdir /opt/nvbroadcast 2>/dev/null || true
        fi
        rm -f "$base/.transaction"
        ;;
    remove)
        if [ ! -e /usr/bin/nvbroadcast ]; then
            rm -f "$base/.transaction" "$base/.legacy-runtime"
            rmdir "$base" 2>/dev/null || true
        fi
        ;;
    *) exit 64 ;;
esac

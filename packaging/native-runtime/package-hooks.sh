#!/bin/sh
# Native package payloads are complete. Never resolve dependencies here.
set -eu
umask 022
base=/usr/lib/nvbroadcast
state_dir=/var/lib/nvbroadcast
runtime_lock=$state_dir/runtime.lock
check_paths() {
        for directory in /usr /usr/lib "$base" "$base/runtime" /var /var/lib "$state_dir"; do
            if [ -L "$directory" ]; then
                echo "NVBroadcast: refusing redirected package directory: $directory" >&2
                exit 78
            fi
            if [ -e "$directory" ]; then
                owner=$(stat -c '%u' -- "$directory")
                permissions=$(stat -c '%a' -- "$directory")
                if [ ! -d "$directory" ] || [ "$owner" != 0 ] || [ "$((0$permissions & 0022))" -ne 0 ]; then
                    echo "NVBroadcast: package directory must be root-owned and not writable by other users: $directory" >&2
                    exit 78
                fi
            fi
        done
        for state in "$base/.transaction" "$base/.legacy-runtime" "$runtime_lock"; do
            if [ -L "$state" ] || { [ -e "$state" ] && [ ! -f "$state" ]; }; then
                echo 'NVBroadcast: refusing redirected transaction marker.' >&2
                exit 78
            fi
            if [ -e "$state" ]; then
                owner=$(stat -c '%u:%g:%h' -- "$state")
                permissions=$(stat -c '%a' -- "$state")
                if [ "$owner" != 0:0:1 ] || [ "$((0$permissions & 0022))" -ne 0 ]; then
                    echo 'NVBroadcast: refusing unsafe transaction marker ownership or permissions.' >&2
                    exit 78
                fi
            fi
        done
}
check_paths
case "$1" in
    prepare)
        mkdir -p "$state_dir"
        if [ ! -e "$runtime_lock" ]; then
            (set -C; : > "$runtime_lock")
            chmod 644 "$runtime_lock"
        fi
        exec 9< "$runtime_lock"
        if ! flock --exclusive --nonblock 9; then
            echo 'NVBroadcast is running or another native transaction is preparing. Quit it and retry.' >&2
            exit 75
        fi
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
        marker=$(mktemp "$base/.transaction.XXXXXX")
        printf '%s\n' '@VERSION@' > "$marker"
        chmod 644 "$marker"
        mv -Tf -- "$marker" "$base/.transaction"
        ;;
    finish)
        "$base/check-packages" --configure
        "$base/runtime/bin/python" -I -B "$base/validate-install.py" --variant '@VARIANT@'
        # Legacy generated environments were outside the native file database.
        # Only delete the conventional, unredirected private environment.
        if [ -f "$base/.legacy-runtime" ]; then
            trusted=true
            for legacy in /opt /opt/nvbroadcast /opt/nvbroadcast/.venv; do
                if [ -L "$legacy" ] || { [ -e "$legacy" ] && [ ! -d "$legacy" ]; }; then
                    trusted=false
                elif [ -e "$legacy" ]; then
                    owner=$(stat -c '%u' -- "$legacy")
                    permissions=$(stat -c '%a' -- "$legacy")
                    if [ "$owner" != 0 ] || [ "$((0$permissions & 0022))" -ne 0 ]; then
                        trusted=false
                    fi
                fi
            done
            if [ "$trusted" = true ]; then
                rm -rf --one-file-system -- /opt/nvbroadcast/.venv
                rmdir /opt/nvbroadcast 2>/dev/null || true
            else
                echo 'NVBroadcast: preserved administrator-modified legacy environment for manual review.' >&2
            fi
            rm -f "$base/.legacy-runtime"
        fi
        rm -f "$base/.transaction"
        ;;
    remove)
        if [ ! -e /usr/bin/nvbroadcast ]; then
            rm -f "$base/.transaction" "$base/.legacy-runtime"
            rmdir "$base" 2>/dev/null || true
        fi
        # Keep the zero-byte lock inode stable across uninstall/reinstall. It
        # contains no application data and is never a Python runtime payload.
        ;;
    *) exit 64 ;;
esac

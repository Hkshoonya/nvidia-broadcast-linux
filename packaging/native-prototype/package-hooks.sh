#!/bin/sh
# NON-RELEASE prototype: native transactions never resolve Python dependencies.
set -eu
base=/usr/lib/nvbroadcast
kind='@KIND@'
family='@FAMILY@'

case "$1" in
    prepare)
        mkdir -p "$base"
        # A killed package-manager transaction must not leave a launchable,
        # partially unpacked runtime. Only the application's final hook clears it.
        printf '%s\n' '@VERSION@' > "$base/.transaction"
        if [ "$kind" != runtime ] && [ -f /opt/nvbroadcast/scripts/install_runtime_variant.py ]; then
            if [ "$family" = deb ]; then
                owner=$(dpkg-query -S /opt/nvbroadcast/scripts/install_runtime_variant.py 2>/dev/null || true)
                [ "$owner" != 'nvbroadcast: /opt/nvbroadcast/scripts/install_runtime_variant.py' ] || touch "$base/.legacy-runtime"
            else
                owner=$(rpm -qf --qf '%{NAME}' /opt/nvbroadcast/scripts/install_runtime_variant.py 2>/dev/null || true)
                [ "$owner" != nvbroadcast ] || touch "$base/.legacy-runtime"
            fi
        fi
        ;;
    finish)
        if [ "$kind" != runtime ]; then
            "$base/check-packages" --configure
            if [ -f "$base/.legacy-runtime" ]; then
                # The old installer generated this private environment outside
                # the package database. Delete it only after the replacement
                # application and runtime have both been configured.
                # A locally redirected legacy prefix is administrator-owned.
                # Do not follow that parent symlink into another directory.
                if [ ! -L /opt/nvbroadcast ]; then
                    rm -rf -- /opt/nvbroadcast/.venv
                    rmdir /opt/nvbroadcast 2>/dev/null || true
                fi
                rm -f "$base/.legacy-runtime"
            fi
            rm -f "$base/.transaction"
        fi
        ;;
    remove)
        # Never erase another installed variant's files or a transaction marker
        # while its launcher is still present. The package manager owns payloads.
        if [ ! -e /usr/bin/nvbroadcast ]; then
            rm -f "$base/.transaction" "$base/.legacy-runtime"
            rmdir "$base" 2>/dev/null || true
        fi
        ;;
    *) exit 64 ;;
esac

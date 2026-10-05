#!/usr/bin/env bash
# Private synthetic audio/display servers. No host sockets or devices are used.
set -euo pipefail
umask 077
export XDG_RUNTIME_DIR=/tmp/nvb-runtime
export PULSE_RUNTIME_PATH=/tmp/nvb-runtime/pulse
export PULSE_STATE_PATH=/tmp/nvb-pulse-state
export PULSE_CONFIG_PATH=/tmp/nvb-pulse-config
export PULSE_SERVER=unix:/tmp/nvb-runtime/pulse-native
mkdir -p "$XDG_RUNTIME_DIR" "$PULSE_RUNTIME_PATH" "$PULSE_STATE_PATH" "$PULSE_CONFIG_PATH"

pulseaudio -n --daemonize=no --exit-idle-time=-1 --disable-shm=yes --use-pid-file=no \
    --load='module-native-protocol-unix socket=/tmp/nvb-runtime/pulse-native auth-anonymous=1' \
    --load='module-null-sink sink_name=prototype_sink' >/tmp/nvb-pulse.log 2>&1 &
PULSE_PID=$!
trap 'kill "$PULSE_PID" 2>/dev/null || true; wait "$PULSE_PID" 2>/dev/null || true' EXIT
for attempt in {1..50}; do
    if [[ -S /tmp/nvb-runtime/pulse-native ]]; then
        break
    fi
    if ! kill -0 "$PULSE_PID" 2>/dev/null; then
        cat /tmp/nvb-pulse.log >&2
        exit 1
    fi
    sleep 0.1
done
if [[ ! -S /tmp/nvb-runtime/pulse-native ]]; then
    cat /tmp/nvb-pulse.log >&2
    exit 1
fi
dbus-run-session -- xvfb-run -a -e /dev/stderr "$@"

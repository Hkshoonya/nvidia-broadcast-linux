#!/bin/bash
set -euo pipefail
umask 077
if (( EUID == 0 )); then
    echo 'Run candidate setup as your logged-in user, without sudo.' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export GST_PLUGIN_PATH=/opt/homebrew/lib/gstreamer-1.0
export GI_TYPELIB_PATH=/opt/homebrew/lib/girepository-1.0
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python=/opt/homebrew/opt/python@3.13/bin/python3.13
if [[ ! -x "$python" ]]; then
    echo 'Candidate requires Homebrew python@3.13, pygobject3, gtk4, libadwaita and gstreamer.' >&2
    exit 1
fi
identity=$(<"$bundle/macos-runtime-id")
if [[ ! "$identity" =~ ^[a-f0-9]{64}$ ]]; then
    echo 'Invalid candidate payload identity.' >&2
    exit 1
fi
runtime="$HOME/Library/Application Support/NVBroadcast Offline Candidate/$identity"
exec "$python" "$bundle/runtime.py" install --bundle "$bundle" --runtime "$runtime"

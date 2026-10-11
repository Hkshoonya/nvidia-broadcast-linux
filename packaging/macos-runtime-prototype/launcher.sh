#!/bin/bash
set -euo pipefail
if (( EUID == 0 )); then
    echo 'Run the candidate as your logged-in user, without sudo.' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export GST_PLUGIN_PATH=/opt/homebrew/lib/gstreamer-1.0
export GI_TYPELIB_PATH=/opt/homebrew/lib/girepository-1.0
bundle=/opt/nvbroadcast-offline-candidate
identity=$(<"$bundle/macos-runtime-id")
if [[ ! "$identity" =~ ^[a-f0-9]{64}$ ]]; then
    echo 'Invalid candidate payload identity.' >&2
    exit 1
fi
runtime="$HOME/Library/Application Support/NVBroadcast Offline Candidate/$identity"
if [[ ! -f "$runtime/runtime-ready" || "$(<"$runtime/runtime-ready")" != "$identity" ]]; then
    echo "Run first: $bundle/setup.sh" >&2
    exit 1
fi
# Keep candidate application state separate from the stable app as well.
export XDG_CONFIG_HOME="$HOME/Library/Application Support/NVBroadcast Offline Candidate/config"
export XDG_CACHE_HOME="$HOME/Library/Caches/NVBroadcast Offline Candidate"
cd "$bundle"
exec "$runtime/.venv/bin/python" -m nvbroadcast "$@"

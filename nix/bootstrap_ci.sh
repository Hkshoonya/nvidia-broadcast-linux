#!/usr/bin/env bash
# Official, checksum-pinned bootstrap for disposable Linux x86_64 CI only.
set -euo pipefail

if [ "${GITHUB_ACTIONS:-}" != true ] || [ "${RUNNER_ENVIRONMENT:-}" != github-hosted ] || \
   [ "${RUNNER_OS:-}" != Linux ] || [ "${RUNNER_ARCH:-}" != X64 ]; then
  echo "This bootstrap runs only on a disposable Linux x86_64 GitHub runner." >&2
  exit 1
fi
if command -v nix >/dev/null 2>&1; then
  echo "Refusing to replace an existing Nix installation." >&2
  exit 1
fi
if systemctl cat nix-daemon.service >/dev/null 2>&1 || systemctl cat nix-daemon.socket >/dev/null 2>&1; then
  echo "Refusing to replace an existing Nix daemon." >&2
  exit 1
fi

mkdir -p dist/nix-cpu
installer="$RUNNER_TEMP/nvbroadcast-nix-install"
curl --fail --location --proto '=https' --tlsv1.2 --retry 2 \
  --connect-timeout 20 --max-time 120 --max-filesize 100000 \
  https://releases.nixos.org/nix/nix-2.35.4/install --output "$installer"
printf '%s  %s\n' 118c3dfa04bdc946b3f577365b9a295607142c9b920e2c07b184e266d62b1ee1 "$installer" | sha256sum --check --status

# The verified script itself checks this Linux tarball's SHA-256 before unpack.
grep -Fq 'hash=ff66236d2f7e2a21f3350e73668e021ac6a8881ef277a38e16e952bfb22718c3' "$installer"
grep -Fq 'nix-2.35.4-x86_64-linux.tar.xz' "$installer"
cat > "$RUNNER_TEMP/nvbroadcast-nix.conf" <<'CONF'
experimental-features = nix-command flakes
sandbox = true
require-sigs = true
substituters = https://cache.nixos.org/
trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY=
trusted-users = root
builders =
access-tokens =
accept-flake-config = false
allow-import-from-derivation = false
max-jobs = 2
cores = 2
max-build-log-size = 8388608
ssl-cert-file = /etc/ssl/certs/ca-certificates.crt
CONF

# No action injects github.token, and no credential/config is passed to Nix.
touch dist/nix-cpu/installation-owned-by-job
env -u GITHUB_TOKEN -u GH_TOKEN -u NIX_CONFIG -u NIX_PATH \
  sh "$installer" --daemon --yes --no-channel-add --daemon-user-count 4 \
  --nix-extra-conf-file "$RUNNER_TEMP/nvbroadcast-nix.conf"
sudo mkdir -p /etc/systemd/system/nix-daemon.service.d
printf '[Service]\nKillMode=control-group\nTimeoutStopSec=15\n' | \
  sudo tee /etc/systemd/system/nix-daemon.service.d/nvbroadcast-bounded-ci.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart nix-daemon.service
test "$(systemctl show --property=KillMode --value nix-daemon.service)" = control-group
echo /nix/var/nix/profiles/default/bin >> "$GITHUB_PATH"
printf 'NIX_USER_CONF_FILES=/dev/null\nNIX_CONFIG=\nNIX_PATH=\n' >> "$GITHUB_ENV"
cat > dist/nix-cpu/bootstrap.json <<'JSON'
{"version":"2.35.4","installer_sha256":"118c3dfa04bdc946b3f577365b9a295607142c9b920e2c07b184e266d62b1ee1","linux_tarball_sha256":"ff66236d2f7e2a21f3350e73668e021ac6a8881ef277a38e16e952bfb22718c3","daemon":"fresh job-owned nix-daemon.service; KillMode=control-group; 15-second stop timeout","scope":"Checksum-verified official bootstrap over HTTPS; separate from hermetic application build. No project credentials, cache upload or release publication."}
JSON

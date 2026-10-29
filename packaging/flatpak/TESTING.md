# CPU Flatpak candidate test

This is an **unsigned development candidate**, not a supported Flatpak release
or a Flathub submission. Please use the exact successful CI run and source SHA
linked by the maintainer in issue #95. Do not substitute a moving branch build.
The initial scope is Linux x86_64, CPU processing, GNOME Platform 50.

## Obtain and verify the named artifact

Use a fresh directory and copy `RUN_ID`, `RUN_ATTEMPT`, `EXPECTED_SOURCE`, and
`ARTIFACT_NAME` from the maintainer's candidate notice. The source is the exact
CI checkout recorded in `bundle-provenance.json`; on a pull request this may be
GitHub's tested merge commit, not the PR branch head. The notice must give it
explicitly. Check the run belongs to `Hkshoonya/nvidia-broadcast-linux`, is the
**Flatpak Development Build** workflow, and succeeded at that attempt.

```bash
gh run view "$RUN_ID" --repo Hkshoonya/nvidia-broadcast-linux
gh run download "$RUN_ID" --repo Hkshoonya/nvidia-broadcast-linux \
  --name "$ARTIFACT_NAME"
sha256sum --check SHA256SUMS
python3 - "$EXPECTED_SOURCE" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path('bundle-provenance.json').read_text())
if p['source_revision'] != sys.argv[1]:
    raise SystemExit('Unexpected source revision; stop here')
if p['application_ref'] != 'app/com.nvbroadcast.NVBroadcast/x86_64/master':
    raise SystemExit('Unexpected application reference; stop here')
if p['signature'] != 'unsigned development bundle':
    raise SystemExit('Unexpected candidate type; stop here')
print('Verified source:', p['source_revision'])
print('Application commit:', p['application_commit'])
print('Bundle:', p['bundle']['name'], p['bundle']['sha256'])
PY
```

Stop if any check fails. Checksums detect transfer changes; the trusted
repository/run selection binds the candidate to the maintainer's build. This
procedure does not claim a project GPG signature. CI artifacts expire after
14 days; request a fresh named candidate if the download has expired.

## Install without changing a native or Snap runtime

Quit any running Broadcast instance first so it releases the camera and virtual
devices. Installing this Flatpak does not replace a native package or Snap. Its
ID and private preferences are separate. Existing development-ID settings can
be copied explicitly using [DISTRIBUTION.md](DISTRIBUTION.md), or keep the new
candidate's defaults for this test.

Install Flatpak using your distribution's normal instructions. The bundle needs
the separately installed GNOME Platform runtime; it is not an offline runtime
bundle. If Flathub is not yet configured for your user, add its runtime remote:

```bash
flatpak remote-add --user --if-not-exists flathub \
  https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak install --user flathub org.gnome.Platform//50 \
  org.freedesktop.Platform.codecs-extra//25.08-extra
flatpak update --user org.gnome.Platform//50 \
  org.freedesktop.Platform.codecs-extra//25.08-extra
flatpak install --user ./nvbroadcast-development-cpu-x86_64.flatpak
flatpak info --user --show-commit com.nvbroadcast.NVBroadcast
flatpak info --user --show-commit org.gnome.Platform//50
flatpak info --user --show-commit org.freedesktop.Platform.codecs-extra//25.08-extra
```

The printed installed application commit must equal the value from the verified
provenance. The separately updated Platform commit must match the
`runtime_inputs.runtimes` Platform entry in that provenance. A bundle install
alone may leave an older shared runtime installed. The codec extension commit
must also match `runtime_inputs.extensions`. If the current remote has
moved to a different commit, report both values for review; do not downgrade or
force a particular runtime globally to make the comparison pass.

Before opening the app, check its real TLS contexts without contacting a server
or accessing a camera/microphone:

```bash
flatpak run --user --command=python3 com.nvbroadcast.NVBroadcast - \
  "$(cat bundle-provenance.json)" <<'PY'
import hashlib, importlib.metadata, json, os, re, ssl, sys
from pathlib import Path
import certifi, pip, pip._vendor.certifi, requests.certs, setuptools
from httpx._config import create_ssl_context
provenance = json.loads(sys.argv[1])
p = provenance['runtime_inputs']
expected_python = tuple(map(int, provenance['private_python']['python']['version'].split('.')))
assert expected_python >= (3, 13, 18) and sys.version_info[:3] == expected_python
assert Path(sys.prefix).resolve() == Path('/app')
assert Path(sys.executable).resolve() == Path('/app/bin/python3.13').resolve()
assert Path('/app/bin/nvbroadcast').read_text().splitlines()[0] == '#!/app/bin/python3.13'
assert Path(ssl.__file__).resolve().is_relative_to('/app/')
assert Path(pip.__file__).resolve().is_relative_to('/app/')
assert importlib.metadata.version('pip') == '26.2.2'
platform = next(x for x in p['runtimes'] if '/org.gnome.Platform/' in x['ref'])
rejected = set(p['rejected_certificate_sha256'])
defaults = ssl.get_default_verify_paths()
default_ca = Path(defaults.cafile)
assert hashlib.sha256(default_ca.read_bytes()).hexdigest() == platform['ca_sha256']
# A fresh SSLContext omits unused capath certificates. Inspect each hashed
# certificate file eagerly, including raw SSL_CERT_DIR path lists, offline.
configured = os.environ.get(defaults.openssl_capath_env, defaults.openssl_capath) or ''
for name in configured.split(os.pathsep):
    if not name or not Path(name).is_dir():
        continue
    files = sorted(x for x in Path(name).iterdir()
                   if re.fullmatch(r'[0-9a-fA-F]{8}\.[0-9]+', x.name))
    assert len(files) <= 4096
    eager = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    total = 0
    for path in files:
        assert path.is_file() and path.stat().st_size <= 1024 * 1024
        total += path.stat().st_size
        assert total <= 16 * 1024 * 1024
        eager.load_verify_locations(cafile=str(path))
    roots = {hashlib.sha256(x).hexdigest() for x in eager.get_ca_certs(binary_form=True)}
    assert not files or roots
    assert not roots.intersection(rejected)
assert Path(setuptools.__file__).resolve().is_relative_to('/app/')
assert importlib.metadata.version('setuptools') == '84.1.0'
assert Path(certifi.where()).resolve().is_relative_to('/app/')
assert requests.certs.where() == certifi.where()
for context in (ssl.create_default_context(), create_ssl_context(),
                ssl.create_default_context(cafile=certifi.where()),
                ssl.create_default_context(cafile=pip._vendor.certifi.where())):
    roots = {hashlib.sha256(x).hexdigest() for x in context.get_ca_certs(binary_form=True)}
    assert roots and not roots.intersection(rejected)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
print('Private interpreter, application TLS contexts and setuptools override passed')
PY
```

Stop and report any failure. This is a targeted trust/dependency check, not a
full audit of the shared runtime. After it passes, start with:

```bash
flatpak run com.nvbroadcast.NVBroadcast
```

For virtual-camera testing, the host must already provide v4l2loopback; see the
[host prerequisite](README.md#host-prerequisite). The sandbox cannot install a
kernel module. Do not use another producer on the same loopback device during
the test.

## Focused acceptance on a normal Wayland desktop

Report the distribution/version, desktop/version, `XDG_SESSION_TYPE`, CPU,
camera and microphone model, exact candidate source/app commit, resolution, and
processing mode. These checks need a normal logged-in Wayland session; the
headless Wayland tests already completed cannot replace physical desktop input.

1. **Camera and controls:** Select the physical camera and a CPU mode, then
   Start Broadcast. Verify moving video, Camera/Audio tabs, minimum window size,
   and Show Preview. Try Blur, Remove, Replace, and effects off. Record visible
   errors and approximate FPS; do not enable unavailable NVIDIA modes.
2. **Virtual camera and reconnect:** Select Broadcast's virtual camera in a
   separate calling/recording application. Keep that consumer open while
   changing effects and preview visibility. Close/reopen the consumer, then
   stop/start Broadcast. Confirm processed video returns. If practical, unplug
   and reconnect the physical USB camera and confirm recovery.
3. **Selected physical microphone:** Select the intended microphone explicitly.
   Use Mic Test Original, then Mic Test Processed with noise removal. Confirm
   speech remains understandable. If a second physical microphone is available,
   switch to it and confirm that the selected device is actually captured.
4. **Virtual microphone:** Select Broadcast's virtual microphone in a separate
   application. Confirm intelligible processed speech, then stop/start Broadcast
   and reconnect the consumer. Check that it resumes without duplicate or
   permanently silent virtual devices.
5. **Recording and meeting:** Record about 15 seconds of speech, switch effects
   once, stop, and play the MP4 from Recordings. Confirm complete moving video
   and audible selected-microphone speech. Finish a short microphone-only Meeting
   and check the saved recording/transcript after any first-use model download.
6. **Desktop integration:** If a StatusNotifier tray is provided, test menu
   actions and recovery after its watcher restarts. If a GlobalShortcuts portal
   is provided, grant a shortcut and test it with another app focused. GNOME
   commonly needs a tray extension, and some desktops lack the shortcuts portal.
   An explicit unavailable message on such a desktop is a capability boundary;
   a promised action failing on a supported backend is a defect.
7. **Soak:** Keep the camera and virtual consumers active for ten minutes, with
   occasional hand motion and one stop/start or consumer reconnect. Report any
   crash, stuck frame, growing delay, repeated device duplication, or recording
   finalization failure.

Report pass/fail/unavailable for each item and the exact error where applicable.
A missing second microphone is a coverage limitation, not a fabricated pass.
No private video, voice recording, or full personal desktop capture is needed
in a public issue report. Logs can be shared after removing personal paths and
identifiers. Keep failed evidence bound to the exact candidate rather than
silently retesting a different artifact.

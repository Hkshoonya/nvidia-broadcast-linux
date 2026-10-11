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
flatpak install --user flathub org.gnome.Platform//50
flatpak install --user ./nvbroadcast-development-cpu-x86_64.flatpak
flatpak info --user --show-commit com.nvbroadcast.NVBroadcast
```

The printed installed application commit must equal the value from the verified
provenance. Start with:

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

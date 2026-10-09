# NV Broadcast v1.5.3

**Store candidate promoted; stable publication remains pending applicable
candidate checks and feedback.** The immutable `v1.5.3` tag and verified draft
packages are built. Snap candidate is revision **190 on AMD64 / 189 on ARM64**;
stable remains v1.5.2 at **185 / 184**. The affected feedback window started
8 October 2026 at **20:55:02 UTC**; its earliest 48-hour milestone is
**10 October at 20:55:02 UTC**, subject to remaining checks and regressions.
This is not a guaranteed publication time. Actual channel and publication
receipts are tracked in [the release readiness record](RELEASE_READINESS_1.5.3.md).

This maintenance release brings together the recording, compact-window,
background-edge, and runtime-recovery fixes merged since v1.5.2.

## Recording that survives capture changes

MP4 recording receives camera frames when all effects are disabled and keeps
running while effects or the capture backend change. Hiding the preview also
preserves recording. A failed replacement capture stops the recorder and
reports an incomplete result so the retained file can be inspected.

The app verifies its H.264 and AAC encoding path before recording, uses the
selected microphone, and reports missing codecs or audio failures. Rec saves
to the desktop Videos directory. The Recordings menu opens that directory,
the last finalized recording, or recordings saved by older Snap revisions.

Meeting capture now reports WAV/audio failures and finalizes its saved media
asynchronously. Faster-whisper downloads use pinned revisions and verified
file hashes; a corrupted cached model is rejected before inference.

## Controls at smaller window sizes

Camera and Audio controls adapt to narrow windows, including when Meeting
Notes is open. At the minimum size, Show Preview restores the camera picture
while the controls remain reachable. The tray's Start/Stop state follows the
actual broadcast state, including startup failures.

Camera discovery retries when the application opens without a camera. A newly
connected camera appears automatically, and Refresh handles later changes.
Discovery does not start a broadcast without the user's configured opt-in.

## Processing and background quality

CPU modes create CPU inference sessions. GPU availability checks use the
selected device, and temporary GPU memory pressure is reported as retryable.
Cold CUDA and TensorRT checks allow time for first-use kernel compilation.
The AMD64 Snap bundles CUDA support without the TensorRT SDK libraries;
TensorRT is available only in installations with its required runtime.

Background processing preserves more moving finger gaps and avoids several
bright-edge and white-clothing contamination cases. Dilate and Softness now
affect every background mode. Remove's tested direct-window and backlit
open-hand checks improved; fine-hair and extreme-setting qualification remains
tracked in [#91](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/91).
These changes do not promise perfect matting in every lighting condition.

## Installation and recovery

Linux source updates build and verify a new runtime before selecting it.
Failed updates preserve the working environment, and `./install.sh
--rollback-runtime` verifies and selects the previous generation. Managed
runtimes cannot be modified by the in-app dependency installer.

Source installations can select supported Python 3.14 systems with matching
desktop bindings. TensorRT uses pinned ABI 10 libraries without requiring
Python bindings; its direct wheel download shows the actual transfer.
GStreamer initialization also works with the newer 1.28 typelib and the older
private PyGObject binding used by the runtime experiments.

macOS now has a Developer ID Installer-signed, notarized and stapled PKG for
Apple Silicon with the supported macOS 15+ Homebrew stack. Installer places
admin-owned source and the packaged launcher; runtime setup runs separately
as the regular user. CPU processing and OBS Virtual Camera are supported.
Mic Test, video recording with microphone speech, and microphone-only Meeting
capture are available. System-audio loopback, processed virtual-microphone
output and CoreML acceleration remain outside the qualified Mac scope; the
proprietary Camera Extension prototype is not installed by the package.

The maintainer reported passing physical camera, compact controls,
Blur/Remove, built-in-microphone Mic Test and complete speech recording,
processed OBS video in another application, and microphone-only Meeting
recording/transcription on a MacBook Air M2 after following the final signed
package instructions. A second physical microphone was unavailable; measured
FPS and upgrade/uninstall lifecycle results were not supplied.

The Python 3.14 source-installer contribution came from
[`@KadotyGamer`](https://github.com/KadotyGamer) in
[PR #103](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/103).
All previously accepted contributors remain in the packaged About credits,
`CONTRIBUTORS.md`, and `NOTICE`.

## Package scope and upgrade instructions

The candidate retains the existing native/source and Snap installation
contracts. The complete offline CPU/CUDA runtime packages under `packaging/`
are maintainer prototypes. Production native installers still require network
dependency resolution; offline runtime locks, RPM signing and production
runtime-pack work remain tracked in #53 and #60. The final macOS PKG's Installer
signing/notarization is verified separately below.

A carried-forward RPM reporting limitation remains in this immutable tag:
if CUDA runtime setup and its clean CPU fallback both fail, the RPM postinstall
scriptlet can still report success. The focused follow-up in
[PR #142](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/142) is merged
for subsequent work and reports that failure explicitly; it is not included in
v1.5.3. Successful CUDA and CPU installation paths are unaffected by this
reporting defect.

DEB and RPM also retain a prerequisite-failure reporting limitation: failed
environment cleanup, creation, or bootstrap dependency setup can be hidden by
a later successful command. The same function bodies occur in v1.5.2 and this
tag. [PR #143](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/143)
merged on 9 October to address those failures for subsequent work; this tag
is unchanged.

Upgrades from affected native v1.4.0 or older installations must use the
`nvbroadcast-native-upgrade` helper and package from the **same release**,
verified against that release's checksums and provenance. The helper is bound
to the exact DEB/RPM bytes. See [artifact verification](RELEASE_VERIFICATION.md).
Final v1.5.3 download links will be added only after publication.

Flatpak remains an x86_64 CPU development package pending its desktop,
microphone, identity, and distribution checks. Project licensing metadata and
retained development-artifact notices are verified; the binary-wheel/source-build
distribution route remains unresolved. No Flathub release or GPU/aarch64 Flatpak
support is announced here.

The [actual tag package workflow](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/37835368212)
passed macOS signing, Accepted notarization, ticket stapling, Gatekeeper and
exact-payload checks, then attested the final packages and created a draft
release. The verified `NVBroadcast-1.5.3-1.pkg` checksum is
`8a4df3d903f80d90b866adcd6571f34368b015aeae8023f35aba8472c61af2e2`.
The [candidate promotion workflow](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/37842894804)
completed for the immutable tag. Public Store SHA3-384 digests and sizes match
the actual reviewed files and their own source-bound attestations. Store
candidate revision 190 is installed: accepted local Snap assertions, mounted
source/runtime input checks and exclusive CPU inference passed. The maintainer
confirmed smooth camera preview and usable compact controls. With GPU memory
available on 9 October, revision 190 passed actual CUDA inference with CPU
fallback disabled and fresh CuPy NVRTC compilation on RTX 5070. All 78 wheel
RECORD inputs and 82 loaded native package files match the qualified review
inputs. These small runtime probes do not measure camera FPS or long-run
performance. Earlier x10 speech-recording evidence remains explicitly retained
through exact application and native runtime input matching; no new physical
recording is claimed for revision 190.
Public downloads and Snap stable promotion remain pending the applicable
candidate checks, feedback and publication steps; candidate promotion is not
stable publication.

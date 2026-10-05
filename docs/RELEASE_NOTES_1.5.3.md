# NV Broadcast v1.5.3

**Candidate notes. This version has not been published.** Final package,
hardware, licensing, and release checks are tracked in
[the release readiness record](RELEASE_READINESS_1.5.3.md).

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

The Python 3.14 source-installer contribution came from
[`@KadotyGamer`](https://github.com/KadotyGamer) in
[PR #103](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/103).
All previously accepted contributors remain in the packaged About credits,
`CONTRIBUTORS.md`, and `NOTICE`.

## Package scope and upgrade instructions

The candidate retains the existing native/source and Snap installation
contracts. The complete offline CPU/CUDA runtime packages under `packaging/`
are maintainer prototypes. Production native installers still require network
dependency resolution; their full offline, signing, and runtime-pack work
remains tracked in #53 and #60.

Upgrades from affected native v1.4.0 or older installations must use the
`nvbroadcast-native-upgrade` helper and package from the **same release**,
verified against that release's checksums and provenance. The helper is bound
to the exact DEB/RPM bytes. See [artifact verification](RELEASE_VERIFICATION.md).
Final v1.5.3 download links will be added only after publication.

Flatpak remains an x86_64 CPU development package pending its desktop,
microphone, licensing, identity, and distribution checks. No Flathub release
or GPU/aarch64 Flatpak support is announced here. macOS Developer ID signing
and notarization are also pending; account approval alone does not establish
signed-package readiness.

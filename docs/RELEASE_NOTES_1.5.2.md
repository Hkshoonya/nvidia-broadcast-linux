# NV Broadcast v1.5.2

NV Broadcast v1.5.2 is the first public release in the v1.5 series. v1.5.0
was withdrawn before stable rollout, and v1.5.1 remained a GitHub draft and
Snap candidate while its native-upgrade, runtime, and packaging corrections
were validated. v1.5.2 preserves all of that work and adds the reviewed changes
merged afterward, so users do not need to install either intermediate build.

## Camera startup recovery

Some V4L2 cameras advertise an MJPEG mode that fails only when GStreamer asks
the driver to allocate capture buffers. Earlier builds stopped the camera
pipeline at that point even when the same camera advertised a working raw mode
at the requested geometry.

v1.5.2 now:

- builds an ordered list of advertised formats for the exact requested
  resolution and frame rate;
- retries the next advertised encoding when the physical source fails before
  producing its first valid frame;
- preserves the requested frame rate, resolution, and output geometry;
- ignores delayed errors from retired pipeline generations;
- remembers a format only after a valid frame arrives, using a bounded
  process-local cache; and
- retains the complete GStreamer diagnostic when every advertised candidate
  fails.

This recovery was contributed by Cédric Prezelin
([`@Tenshock`](https://github.com/Tenshock)) in
[PR #76](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/76).

## Source installation reliability

The source installer now discovers supported CPython interpreters in the order
3.13, 3.12, then 3.11, or accepts an explicit `--python` path. It verifies
`venv` and `ensurepip`, then checks that GTK4, Libadwaita, and GStreamer Python
bindings match the selected interpreter after distro packages are installed.

Only the project-owned `.venv` is recreated when its base interpreter or
CPU/CUDA runtime owner is incompatible. The installer does not replace the
system Python or add a third-party package repository, and unsupported setups
stop with package-manager-specific recovery guidance instead of reporting a
partially working installation.

## Included v1.5 improvements

v1.5.2 includes the complete v1.5.1 candidate and its corrected v1.5 runtime:

- one managed CPU or CUDA ONNX Runtime distribution owner per environment;
- pinned fresh-process CPU, CUDA, and TensorRT execution probes that reject
  silent CPU fallback where GPU execution is required;
- saved microphone restoration and live audio-pipeline rebuilding;
- clean GUI and headless handling when no physical camera is available;
- explicit per-profile auto-start with failure-safe visible controls;
- MediaPipe 1.0.0 face effects on Linux arm64;
- enforced OpenCV, Protobuf, Pillow, and development dependency security
  floors;
- clean final DEB and RPM removal of installer-generated runtime and build
  files; and
- package checksums plus GitHub provenance attestations for tag-built release
  artifacts.

## Required native-package upgrade path

Public DEB and RPM releases through v1.4.0 contain legacy pre-removal scripts
that can terminate their own package transaction. Those scripts run before a
new package can replace them. For that reason, users upgrading an existing
native v1.4.0 or older installation must use the release asset named
`nvbroadcast-native-upgrade`.

1. Download the v1.5.2 DEB or RPM, `nvbroadcast-native-upgrade`, and
   `SHA256SUMS.packages` from the same GitHub Release.
2. Verify both downloaded files against `SHA256SUMS.packages` and, where the
   GitHub CLI is available, verify their build attestations as described in
   [Verifying Release Artifacts](RELEASE_VERIFICATION.md).
3. Make the helper executable and pass it the downloaded package:

   ```bash
   chmod 755 ./nvbroadcast-native-upgrade
   sudo ./nvbroadcast-native-upgrade ./nvbroadcast_1.5.2-1_all.deb
   ```

   Fedora users pass the RPM instead:

   ```bash
   chmod 755 ./nvbroadcast-native-upgrade
   sudo ./nvbroadcast-native-upgrade ./nvbroadcast-1.5.2-1.noarch.rpm
   ```

Do not invoke `apt`, `dpkg`, `dnf`, or `rpm` directly for an upgrade from an
affected native v1.4.0 or older installation. Do not mix a helper and package
from different releases. The v1.5.2 helper is generated from the exact release
artifacts, embeds their SHA-256 hashes, copies the selected package into a
root-owned temporary directory, and validates its identity, version, release,
and architecture before invoking the package manager.

Clean native v1.5.2 installations may use the normal package-manager command.
Snap users update normally through the Store and do not use this helper.

## Release and contributor safeguards

- Every Snap Store review, candidate, and stable action must resolve to an
  existing release tag and matching source commit before any build or upload.
  Branch, mismatched-tag, and mismatched-source dispatches fail closed.
- Tag workflows preserve the GitHub Release as a draft. Publishing the GitHub
  Release and moving both Snap architectures to stable remain explicit actions
  after artifact inspection and candidate testing.
- Contributor credits are cumulative and packaged in the application. The
  About window credits John Maingi (`@JohnMaingi-IXP`), Jon Fuller
  (`@perfectra1n`), Cédric Prezelin (`@Tenshock`), and Cenkay Çoban
  (`@pastor0711`), and pull-request validation prevents a future release from
  silently dropping an accepted external contributor.

## Validation and remaining boundary

The unpublished v1.5.1 Snap candidate completed more than seven days of soak.
Because v1.5.2 changes physical-camera startup, the exact v1.5.2 Snap revisions
must complete a fresh 72-hour candidate soak, including no-camera and
busy-camera recovery, live source changes, processing-mode changes, auto
framing, difficult lighting, and hair-edge behavior, before stable promotion.

This release does not close the broader supply-chain work in
[Issue #60](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/60).
Native installers still resolve part of their Python environment online,
native artifacts are not independently reproducible, a complete SBOM is not
yet published, and RPM signing plus macOS signing/notarization remain open.
Intel macOS is also not included because a secure current MediaPipe wheel is
not available for that architecture; the macOS package supports Apple Silicon.

# NV Broadcast v1.5.1

NV Broadcast v1.5.1 is the native-package upgrade and release-gate hotfix for
the v1.5 runtime, device-recovery, and provenance update. It supersedes v1.5.0,
which was withdrawn before Snap stable rollout after lifecycle testing found an
upgrade blocker inherited from older native packages.

## Native-package upgrades

Public DEB releases through v1.4.0 contain a broad pre-removal command that can
terminate their own package transaction. The old script runs before a new DEB
can replace it, so v1.5.1 includes a separate release asset named
`nvbroadcast-native-upgrade`.

The helper also accepts dpkg's install-requested recovery states, including the
half-configured state left by an earlier failed direct upgrade, and completes
the transaction by unpacking the hash-verified v1.5.1 DEB before asking apt to
repair and configure dependencies.

RPM releases through v1.4.0 contain the equivalent broad pre-uninstall command.
The helper bypasses it only when the installed RPM reports the exact known
one-line script. It resolves the verified target RPM's declared dependencies
with DNF, validates the transaction, and uses RPM's supported `--nopreun` flag
for that migration only. Unknown or locally modified vulnerable scripts are
rejected; installations with a safe pre-uninstall script use an ordinary DNF
transaction.

For an upgrade from a native v1.4.0 or older installation:

1. Download the v1.5.1 DEB or RPM, `nvbroadcast-native-upgrade`, and
   `SHA256SUMS.packages` from the same GitHub Release.
2. Verify both downloaded files against `SHA256SUMS.packages` and, where the
   GitHub CLI is available, verify their build attestations as documented in
   [Verifying Release Artifacts](RELEASE_VERIFICATION.md).
3. Make the downloaded helper executable and run it with the package path:

   ```bash
   chmod 755 ./nvbroadcast-native-upgrade
   sudo ./nvbroadcast-native-upgrade ./nvbroadcast_1.5.1-1_all.deb
   ```

   Fedora users pass the RPM instead:

   ```bash
   chmod 755 ./nvbroadcast-native-upgrade
   sudo ./nvbroadcast-native-upgrade ./nvbroadcast-1.5.1-1.noarch.rpm
   ```

Do not use a helper or package from a different release. The v1.5.1 helper
contains the exact SHA-256 of both v1.5.1 native artifacts, copies the selected
artifact into a root-owned temporary directory, then validates its package
name, version, release, and architecture before invoking the package manager.

Clean v1.5.1 installations do not contain the legacy script and may use the
normal `apt` or `dnf` installation command.

## Uninstall cleanup

- DEB and RPM installation removes temporary `build` and `egg-info` metadata
  created while installing the local application project.
- Final DEB removal or purge removes the installer-managed
  `/opt/nvbroadcast` tree, including its generated virtual environment.
- Final RPM removal performs the same cleanup only when the RPM transaction
  reports no remaining installed version; upgrades preserve the new payload.
- Per-user configuration and model caches remain in user-owned configuration
  and cache directories. Existing system v4l2loopback configuration remains
  preserved as documented by the uninstall path.

## Release publication guard

- A tag push may attach size-eligible Snap assets only while preserving the
  GitHub Release as a draft.
- Store review and candidate workflow dispatches cannot run the GitHub release
  attachment job merely because they were dispatched from a tag.
- Stable publication remains a separate, explicit gate after exact-artifact
  verification and candidate soak.

## Included v1.5 improvements

v1.5.1 includes all application changes prepared in v1.5.0:

- exactly one CPU or CUDA ONNX Runtime distribution owner per managed runtime;
- fresh-process CPU, CUDA, and TensorRT execution probes with GPU fallback
  disabled where acceleration is required;
- saved microphone restoration and live capture-pipeline rebuilding;
- clean GUI and headless handling when no physical camera is available;
- explicit per-profile auto-start and failure-safe visible start state;
- MediaPipe 1.0.0 face effects on Linux arm64;
- stricter Snap dependency closure, reviewed revision promotion, checksums, and
  GitHub artifact attestations.

## Release boundary

The patch fixes the confirmed lifecycle and publication-gate defects. It does
not complete the broader supply-chain work in Issue #60: native installers
still resolve parts of their Python environment online, native artifacts are
not independently reproducible, RPM signing and macOS signing/notarization are
not complete, and a full SBOM is not yet published.

The release remains draft and Snap remains outside stable until the exact
v1.5.1 artifacts pass clean install, legacy upgrade, reinstall, launch/runtime,
final removal, checksum, provenance, supported-platform, and candidate-soak
gates in the [Release Checklist](RELEASE_CHECKLIST.md).

# NV Broadcast v1.5.4

**Candidate in preparation on 28 October 2026; not published.** No v1.5.4 tag,
public downloads or Store revisions are claimed by these notes. Public native
packages and Snap stable remain on v1.5.3 while the new candidate is verified.

This maintenance update brings security-sensitive Python dependency updates,
fresh Eye Contact tracking, Arch runtime compatibility and clearer installer
failure handling together in the existing release paths.

## Updated Python dependencies and certificate bundles

The source and Snap requirements now use the repaired certificate bundles and
newer Pillow, filelock, setuptools and pip inputs introduced in #171. The
locked native and macOS development inputs are updated too. CI audits exact
frozen Python versions alongside the ordinary resolved dependency audit.

The October 26 input refresh selected certifi 2026.10.19, Pillow 13.0.0,
filelock 4.2.1, setuptools 84.1.0 and pip 26.2.2. Actual v1.5.4 package contents
must be inspected and audited before promotion; these source changes do not
establish that every OS library, codec, model or system trust store is audited.

## Eye Contact follows current camera frames

The shared face tracker no longer treats a reused Python frame object ID as
permanent evidence that the image has not changed. Eye Contact can update its
eye positions while the camera picture keeps moving. Natural and Gaze Lock
retain their existing blink, occlusion and intensity behavior.

The fix has generated-frame and installed Apple Silicon regression coverage.
Physical confirmation of the specific M5/Tahoe report in #150 remains open;
that issue's GTK/GStreamer assertions are not claimed fixed by this change.

## Arch CUDA compatibility and source-install reliability

Arch packages such as `python-onnxruntime-opt-cuda` can expose CUDA while their
Python distribution is still named `onnxruntime`. Live runtime detection now
recognizes that combination. Duplicate runtime owners remain rejected, and
packaged-artifact validation keeps its strict CPU/CUDA distribution contract.
GPU availability still requires real provider execution checks.

Thanks to [@mauriciozaffari](https://github.com/mauriciozaffari) for the Arch
contribution and RTX 2070 SUPER verification in #156. Cumulative contributor
credits remain available in the app and [CONTRIBUTORS.md](../CONTRIBUTORS.md).

The managed source installer keeps its own protobuf discoverable when Fedora’s
legacy `google` namespace hook would otherwise hide it. Distro-only namespace
modules and system GTK/GStreamer remain available. Native runtime setup stops
on failed prerequisite steps and reports an error if CUDA setup and its clean
CPU fallback both fail. These installer changes do not promote the experimental
offline runtime adapter.

The macOS PKG setup also handles fresh Homebrew installations whose PyGObject
formula targets Python 3.14/3.15. It can build hash-pinned PyGObject and Pycairo
for supported Python 3.13 inside the per-user runtime, using Homebrew's native
GTK/GStreamer libraries. Existing working bindings for supported Python
3.11–3.13 remain usable. This change retains the patched Python floor and
macOS CPU ONNX Runtime constraint; actual fresh-Mac compilation and media
qualification remain required before releasing the PKG.

## Candidate verification and distribution scope

The following gates remain pending for the actual v1.5.4 candidate:

- Complete current-head PR checks, version/metadata checks, dependency audits,
  release smoke and both Snap architecture builds.
- Independently inspect actual package/source identities and packaged supplier
  versions. Bind checksums, signatures and provenance to the exact release
  source and generated artifacts.
- Verify the installed strict Snap's camera, microphone, controls, processing
  modes, Eye Contact movement and complete spoken recording. Qualify the
  changed runtime inputs on available CPU/CUDA hardware and record unavailable
  combinations explicitly.
- Tie candidate feedback and the affected-runtime observation period to its
  exact source and Store revisions. Stable promotion follows review of the
  results and accepted limitations.
- Complete signing/notarization and affected native-package upgrade/install
  checks before distributing the corresponding DEB/RPM/PKG artifacts.

The regular Snap update can proceed independently of the experimental offline
native-runtime and Flatpak distribution work. #60 still tracks the broader
runtime delivery, supported-platform and activation requirements. #95 still
tracks the external runtime refresh and physical Wayland acceptance required
for a supported Flatpak. No Flatpak or NixOS support is announced here.

The remaining fine-hair/edge scope in #91 and the broader roadmap in #3 stay
open. Existing v1.5.3 tags, packages and recorded qualification retain their
original identities. Candidate scope and release verification follow the
[release checklist](RELEASE_CHECKLIST.md).

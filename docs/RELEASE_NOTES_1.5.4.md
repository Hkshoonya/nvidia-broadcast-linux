# NV Broadcast v1.5.4

**Tagged 29 October 2026; not published.** The v1.5.4 tag is frozen at
`755ee61384583303cf0ab469d02600c85f1e2b3b`. Its GitHub release remains a draft.
Public native downloads and Snap stable remain on v1.5.3; the separately built
Store candidate still needs its own verification, delivery and acceptance.

This maintenance update brings security-sensitive Python dependency updates,
fresh Eye Contact tracking, Arch runtime compatibility and clearer installer
failure handling together in the existing release paths.

## Updated Python dependencies and certificate bundles

The source and Snap requirements now use the repaired certificate bundles and
newer Pillow, filelock, setuptools and pip inputs introduced in #171. The
locked native and macOS development inputs are updated too. CI audits exact
frozen Python versions alongside the ordinary resolved dependency audit.

The October 26 input refresh selected certifi 2026.10.19, Pillow 13.0.0,
filelock 4.2.1, setuptools 84.1.0 and pip 26.2.2. The actual tag-built Snaps
passed supplier inspection and the recorded Python advisory checks. Those
checks do not establish that every OS library, codec, driver, model or system
trust store is audited.

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
macOS CPU ONNX Runtime constraint. The exact tag-built PKG passed fresh
binding compilation/imports, CPU inference and synthetic media checks on the
Apple Silicon runner.

## Candidate verification and distribution scope

The following checks are complete within their recorded scope:

- Source tests, version metadata, release smoke and dependency audits passed.
  Both strict Snap architectures passed dependency closure and confined
  synthetic recording checks.
- Default DEB/RPM installation, reinstall, upgrade, failure recovery and removal
  passed **108 checks** on disposable Ubuntu 24.04 and Fedora 44 fixtures with
  documented cached prerequisites. The final tag packages preserve those inputs.
- The [tag package run](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/40792768856)
  passed project-key RPM signing and actual Apple Silicon installation, private
  binding compilation, CPU inference and synthetic media decoding. The PKG
  passed Developer ID Installer signing as **WeMakeSense LLC (T39RXSKKZ6)**,
  notarization, stapling and Gatekeeper acceptance. Independent readback verified
  package/source identities and the signatures' scoped evidence.
- The native checksum manifest matches the DEB, signed RPM, upgrade helper,
  project public key and PKG. All six subjects, including the manifest itself,
  passed cryptographic verification against the exact tag, source, workflow,
  hosted runner and original run/attempt.
- The [tag Snap run](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/40792768843)
  passed source/bytecode, suppliers, closure, notices, RECORD, architecture,
  bundled-certificate and provenance checks. Recorded Python advisory snapshots
  checked 69 AMD64 and 60 ARM64 packages with zero vulnerabilities or lookup
  errors. All **572 AMD64** and **479 ARM64** packaged ELF files match accepted
  runtime inputs. Earlier AMD64 CPU/CUDA and fresh CuPy probes on the available
  RTX 3080 and RTX 5070 transfer only through that verified byte comparison.
- The owner confirmed the requested Linux CPU/DocZeus preview, Eye Contact
  movement and spoken recording, plus signed M2 Mac setup, camera/effects and
  spoken recording. Tag payload/native-binding evidence matches that Mac
  candidate. Earlier unchanged controls, lighting and OBS/Meeting results retain
  their original scope; no new measurement is inferred from that carry-forward.

The draft ARM64 Snap asset matches its attested file. The AMD64 Snap exceeds
GitHub's **2 GiB** release-asset limit and remains in the tag workflow's Actions
artifact. Neither file identifies a later Store rebuild merely because its
version is also 1.5.4.

The remaining release gates are:

- Complete Store review with valid authentication, then verify both actual Store
  rebuilds, their own bytes/provenance and uploaded revision identities against
  the accepted runtime inputs.
- Verify candidate delivery and installation. Tie feedback and the affected
  runtime observation period to those exact source/artifact/revision identities.
- Review the observation evidence and accepted limitations before the explicit
  stable/publication decision. A shorter urgent-security window requires a
  documented decision; the brief owner test is not a completed 48–72-hour soak.
- Publish the existing verified draft only after the applicable release gates,
  then verify public assets/update discovery and update the website downloads.

Numeric FPS, a second physical microphone, M5/Tahoe, physical Linux ARM devices,
TensorRT coverage and a completed 48–72-hour observation window are outside the
recorded acceptance. Synthetic media checks qualify their tested paths; they do
not replace physical device acceptance or a complete security assessment.

The regular Snap update can proceed independently of the experimental offline
native-runtime and Flatpak distribution work. #60 still tracks the broader
runtime delivery, supported-platform and activation requirements. #95 still
tracks the external runtime refresh and physical Wayland acceptance required
for a supported Flatpak. No Flatpak or NixOS support is announced here.

The remaining fine-hair/edge scope in #91 and the broader roadmap in #3 stay
open. Existing v1.5.3 tags, packages and recorded qualification retain their
original identities. Candidate scope and release verification follow the
[release checklist](RELEASE_CHECKLIST.md).

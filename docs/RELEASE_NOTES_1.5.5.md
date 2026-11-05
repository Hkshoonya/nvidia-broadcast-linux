# NV Broadcast v1.5.5

**Release preparation — not published.** The proposed release starts from
`f014576ca257f1c085d9a02d94d94f3b477e1152`. This proposed candidate has version metadata and release notes prepared; no
v1.5.5 tag, package, Store revision or release acceptance is claimed here. [v1.5.4](https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v1.5.4)
remains the current public release, with its frozen artifacts and verification
record intact.

This maintenance update focuses on responsive startup, processing-mode checks
and clearer face-effect recovery. The ordinary Snap, online DEB/RPM and signed
Apple Silicon PKG paths remain the release targets.

## Responsive processing-mode and first-run checks

Provider checks run away from the desktop interface. Mode controls reuse
completed results, and first-run setup permits an explicit CPU choice while
GPU checks are pending. Saved Auto choices, automatic startup and retries wait
for verified results tied to the selected GPU and current window.

Closing setup or quitting the app cancels its owned provider-check children.
Cancellation is not cached as evidence that a runtime is unavailable. Existing
CPU/GPU execution validation and cold-start time limits remain in place; this
change does not enable an unverified processing mode
([PR #191](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/191)).

## Contain face-model initialization failures

The MediaPipe face landmarker and Auto Frame detector are created in an
isolated child before their native models are enabled in the app. A failed,
killed or timed-out initialization check leaves the affected optional effect
unavailable for that session instead of allowing that initialization failure
to stop Broadcast.

This contains the reported startup failure; it does not claim to repair the
underlying MediaPipe compatibility problem or every later native execution
failure. Contributed by
[@Nirusaki-Malaal](https://github.com/Nirusaki-Malaal)
([PR #180](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/180),
fixes [#179](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/179)).

## Clearer Eye Contact status and Fedora installation

The existing Eye Contact controls show whether tracking is waiting, a face is
missing, tracking failed, the eyes cannot be adjusted or a correction was
applied. Generated-frame comparisons preserve the previous gaze-correction
pixels. Natural/Gaze Lock behavior and conservative blink/occlusion limits are
unchanged. The specific M5/Tahoe quality report remains open under
[#150](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/150)
([PR #189](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/189)).

Fedora source installation requests the available `python3-gobject` and
`python3-cairo` packages. Guidance explains why an alternate Python alone does
not provide matching GTK/GStreamer bindings. Normal supported source
installation remains CPython 3.11–3.14
([#183](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/183)).

## Control scrolling

Scrolling over an effect slider leaves its value unchanged and lets GTK
scroll the containing page, including horizontal and Shift-modified input.
Keyboard adjustment and pointer dragging still change the slider. This also
prevents unintended noise-strength changes from restarting the virtual
microphone ([PR #199](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/199)).

## Contribution recognition

The About window presents Core Maintainers, Project Sponsors, Major Feature
Contributors and Contributors, with concise scopes for accepted work and a
separate sponsor section. Verified fixes remain eligible for recognition. Grouping
reflects reviewed technical depth and impact rather than line counts,
pull-request totals or sponsorship; it grants no repository permissions.

All accepted external credits remain. The registry, README and
[contributor record](../CONTRIBUTORS.md) retain public names, evidence and the
distinction between source acceptance and delivery. In particular, the
experimental Python 3.15 CPU tooling and native-model checks credited to
Mradul Kumar were accepted outside the frozen v1.5.4 source; recognition does not claim
that those changes were in that release or complete their pending platform
and package acceptance.

## RPM signature compatibility

The release signer requests an additional legacy header signature when
`rpmsign` supports it. Default RPM6 signatures on format6 packages can pass a
current reader while an RPM4 reader reports only valid digests, without a
verified publisher signature. The compatibility request preserves the modern
signature and the existing package-header, payload and key checks. Older
signing tools retain their existing command.

A disposable RSA signing fixture verifies the signature and file payload
with RPM6.0.2 and RPM4.18.2 readers. This test uses no publisher key; the final
publisher-signed release RPM still requires its own verification and identity
record. Package format and build environment are unchanged by this fix.

## Reduce Face Relighting frame cost

Face Relighting reduces repeated frame/color work while keeping its existing
lighting masks, strength and analysis cadence. Generated comparisons preserve
exact pixels and background-analysis state against the previous implementation
([PR #195](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/195)).
Full live FPS acceptance remains pending; this change does not claim that all
processing delays are resolved.

## Respect the selected GPU during compositing

Mixed CPU/GPU effects explicitly use the selected Compute GPU for uploads,
compositing and GPU-matte bookkeeping, including on a fresh capture thread.
Each operation restores the thread's previous device even after a failure;
existing CPU fallback remains available. A separate cache boundary protects
device switches without serializing rendering with asynchronous inference.

Generated routing tests cover device selection, cache reuse/invalidation,
concurrent switches and error recovery while preserving the existing pixel
calculations. Native multi-GPU smoke and physical FPS acceptance remain pending.
This corrects device ownership; it does not establish the cause of every slow
background-processing frame or claim an FPS improvement.

## Preserve capture timing in recordings

Processed video keeps the capture event's timing on the recorder's shared
clock instead of acquiring a timestamp after the effects finish. The camera
may have been running for minutes before Rec starts; its timestamps are rebased
onto the new recording rather than copied directly. Frames from an earlier Rec
session or retired camera pipeline are discarded.

If a capture clock is unavailable when Rec starts, that session keeps arrival
timing so it can still record video. Missing source timestamps also retain an
arrival fallback. These cases do not claim proven capture-time alignment, and
older or duplicate timestamps are dropped rather than moved forward.

A generated H.264/AAC regression compares video markers with audio pulses after
injected 0, 100 and 300 ms processing waits. Its legacy control exposes the added
offset; the mapped case preserves relative event timing. This does not measure
physical camera/microphone latency or make expensive live effects run faster.
Physical A/V and FPS acceptance on the final package remains pending; v1.5.5
is still not published.

## Optional tooling stays separate

The source tree also contains explicitly opted-in development work:

- The experimental Linux x86_64 Python 3.15 CPU source route checks its pinned
  native runtime before activation. GPU support and Fedora 45 physical-device
  acceptance remain separate under
  [#184](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/184).
  Contributed by [@Nirusaki-Malaal](https://github.com/Nirusaki-Malaal)
  ([PR #185](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/185)).
- Optional native-candidate Python inventory generation defaults off. Its
  [first protected handoff](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/41733325775)
  passed on historical `1.5.3-9.cpu` packages, recording 59 installed Python
  distributions per DEB/RPM inventory. Independent small-evidence readback
  verified the metadata chain; hosted jobs checked the large package archives
  and signatures. This is partial Python inventory, not complete native,
  interpreter, codec, model or license coverage, and does not qualify v1.5.5
  packages ([#60](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/60),
  [scope](PYTHON_SUPPLIER_INVENTORY.md)).
- Private macOS bindings for the unsigned offline prototype do not replace the
  default signed online PKG. The authenticated Flatpak CPU/x86_64 development
  candidate retains its physical Wayland and public-distribution requirements
  under [#95](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/95).
  This release announces no new Flatpak or NixOS support.
- Snap revision promotion verifies exact release-tag provenance and captured
  revision IDs. Optional edge publication remains disabled by default and
  requires its own activation and credential qualification
  ([promotion policy](SNAP_EDGE_PROMOTION.md)).

## Verification still to record for this release

Source tests and prior package checks retain their exact original scope. The
installed v1.5.4 Snap revision 192 and earlier Mac/device acceptance are
historical evidence, not acceptance of these new app bytes.

Before publication, record the frozen v1.5.5 commit and its actual package
identities, current-head checks, dependency review, package source readback,
startup/quit behavior and camera/microphone acceptance. Existing generated-media
smokes cover codecs, recording and CPU inference; they do not initialize both
real MediaPipe face models. The changed model paths require a targeted check
in the actual packaged runtime.

Each new package needs its own checksum and provenance; the new PKG needs its
own signing, notarization, stapling and Gatekeeper results. Separately built
Store revisions require their own identity and acceptance evidence. Observe
the affected behavior according to the [release checklist](RELEASE_CHECKLIST.md),
carrying forward only justified unchanged-path evidence. The remaining qualification and publication steps have not yet occurred.

Numeric FPS, a second physical microphone, M5/Tahoe, physical Linux ARM
hardware and TensorRT coverage must retain their actual recorded limits.
Cumulative contributor credits remain in the app and
[CONTRIBUTORS.md](../CONTRIBUTORS.md).

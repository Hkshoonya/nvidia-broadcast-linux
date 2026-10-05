# v1.5.3 release readiness

Assessment started **5 October 2026**, from main commit
`254ec93b5351c8e50815b3bf03dce6ae91ef1726`. The latest published version remains
[v1.5.2, published 4 September 2026](https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v1.5.2).

**Status: release preparation; not approved for publication.** The version
metadata and candidate notes describe the next maintenance release. No tag,
Store promotion, website download change, or public Flatpak submission is part
of this preparation. The planned metadata date must be reconciled with the
actual publication date before tagging.

## Scope and freeze

The application scope is the merged fixes through #135: recording, compact
windows, camera discovery, processing recovery, matting improvements, pinned
meeting downloads, Python 3.14, and recoverable source upgrades. New features
and the production runtime-pack redesign remain outside this candidate.

Only a recorded release blocker should change application behavior after this
cut. Record any such amendment and rerun its affected checks. Experiments in
`packaging/runtime-prototype` and `packaging/native-prototype` remain development
tooling, with independent source/input identities; they are not replacements
for the production installation paths.

The release audit found one notice error under #100: RobustVideoMatting was
labeled MIT, while its referenced v1.0.0 source tag contains GPLv3. Correcting
that source label changes distributed notice bytes and invalidates earlier
candidate package hashes. Rebuild the final packages and bound upgrade helper.
The [licensing review packet](LICENSING_REVIEW_1.5.3.md) records the exact source
identity, proposed review path, and model/codec questions; it does not alter
the project's attribution terms or supply legal clearance.

## Evidence and remaining decisions

Prior development-package tests guide qualification but do not transfer
automatically to newly built v1.5.3 artifacts.

| Area | Existing evidence | Remaining release work |
| --- | --- | --- |
| Unit, packaging, and CI | #135: 940 local tests passed; six PR checks passed; merged source tree equals the tested head | Run checks on the v1.5.3 head and manual release-build workflows |
| Small window UI | Maintainer tested Camera/Audio and restored camera preview in an installed development Snap | Repeat on the exact release candidate |
| Recording | Installed Snap speech playback passed; exact development DEB/RPM/Flatpak physical-camera recordings passed, with generated selected audio; extended Flatpak recording passed | Final artifact tests, native/Flatpak desktop and physical-microphone acceptance, codec review in #100 |
| CPU/GPU behavior | Installed CPU fallback, GPU recovery, live CUDA, and cold-cache probe fixes were verified before their merges | Exact candidate CPU/CUDA/TensorRT checks on applicable devices; record memory pressure separately from a broken installation |
| Matte quality (#91) | Patched Remove passed direct-window and backlit moving-hand feedback | Final-package Blur/Replace, fine hair, white clothing, and extreme sliders; document any accepted residual limitation |
| Source recovery (#53) | Failed-install preservation, CPU/CUDA transitions, rollback and source window startup tested in #133 | Candidate source update/rollback check; production native runtime packs remain a separate unfinished scope |
| Native artifacts (#60) | CPU prototypes passed eight lifecycle cells; unsigned archives reproduced in recorded builders | Build the production DEB/RPM, bind the exact upgrade helper, verify final manifests/provenance and installer transactions; do not claim hermetic production payloads |
| License/redistribution (#100) | Exact repository/release license mismatch and codec inventories documented | Qualified review and maintainer decision; complete applicable texts and consistent package declarations before clearance |
| Flatpak (#95) | Development build, dependency closure, model trust, recording, physical camera, and initial virtual-camera read passed | CPU release conditions below; remains excluded until applicable gates pass |
| macOS | Developer account approved; signing setup deferred by maintainer | Developer ID configuration, signed/notarized/stapled artifact checks before claiming a signed macOS release |
| Snap edge automation (#90) | Current public channels reported aligned on reviewed 185/184 | Verify connected builder configuration and scoped edge credentials before merging the edge-promotion draft |
| NixOS (#18) | Package/module evaluation and Xvfb startup passed | No real NixOS machine available; draft remains unsupported pending physical-device acceptance |

## Flatpak decision for this release

The first possible release scope is **x86_64 CPU**. CUDA/TensorRT and aarch64
need separate implementations and qualification; their absence does not by
itself prevent an accurately described CPU-only package.

Still required for that CPU package:

1. Final installed-package desktop acceptance: real Wayland camera and virtual
   camera, effects, microphone processing and virtual output, shortcuts/tray,
   recording, and client reconnect behavior. Current recorded desktop testing
   used X11; generated tone checks do not establish audible physical speech.
2. Resolve #100, including the bundled codec/dependency notices and the
   project's additional attribution terms. Preserve accepted contributor credit.
3. Confirm a permanent application ID, naming/non-affiliation wording, real
   application screenshots, and passing final artifact lint.
4. Review payload size and the distribution route. The development manifest
   uses prebuilt Python wheels; Flathub's current source-build requirement also
   applies to dependencies, unless an applicable exception is granted.
5. Keep Flathub submission work human-authored. Its current policy prohibits AI
   tools from opening or automating submissions and from generating submission
   commit messages, descriptions, comments, or replies. This upstream readiness
   document and local packaging work are not a Flathub submission.

Primary policy references checked on 5 October 2026:
[Flathub requirements](https://docs.flathub.org/docs/for-app-authors/requirements)
and [application verification](https://docs.flathub.org/docs/for-app-authors/verification).

## Security and publication status

No new security clearance follows from a version bump or passing unit tests.
Run the release checklist's dependency audit, Bandit, package/source inspection,
model trust, confinement, installer privilege-boundary, and artifact provenance
checks against the exact release inputs. This desktop application has no hosted
authentication service in the release scope. Remote downloads, privileged
native hooks, optional runtime installation, and bundled native libraries still
need their applicable checks.

At this point, final-artifact security and hardware qualification are pending,
and license clearance is unresolved. **The candidate is not yet verified for
public release.** Follow [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the
recorded merge, tag, attestation, candidate, soak, and publication sequence.

# v1.5.3 release readiness

Assessment started **5 October 2026**, from main commit
`254ec93b5351c8e50815b3bf03dce6ae91ef1726`. The latest published version remains
[v1.5.2, published 4 September 2026](https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v1.5.2).

**Status: maintainer-authorized release execution, 8 October 2026.** The
maintainer authorized verified PR merges and the v1.5.3 release. Planned stable
publication is **10 October 2026**, after applicable candidate feedback and
affected-behavior checks. This authorization permits the main merge, fixed tag,
draft assets, verified Store candidate and eventual stable publication once
their technical gates pass; it does not record those actions as completed.
Public website downloads stay on v1.5.2 until publication, and supported public
Flatpak distribution remains excluded. Keep actual workflow, artifact, Store
revision, device acceptance and publication receipts in the release record.

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

The 8 October qualification amendment corrects a reproduced false positive in
dependency validation: selected CPU/CUDA and meeting extras, and extras requested
by transitive dependencies, could be omitted while a candidate was reported
complete. Source rollback still verifies older generations using their own
interpreter and application imports. The shipped checker and installer bytes
change, so earlier package hashes and installed-runtime evidence do not qualify
this amendment; rebuild and repeat the affected checks on the final head.

The same amendment fixes the legacy macOS source installer's prerequisite
selection and required native-plugin checks before replacing its existing venv.
It retains the legacy in-place replacement limit documented in the README;
PKG per-user setup and notarization behavior are separate. Flatpak CI now retains
unsigned development bundles only after import, OSTree integrity, Git source,
resource, notice, and sandbox-metadata checks. This retention does not satisfy
the public-distribution or physical-device gates below.

The release audit found one notice error under #100: RobustVideoMatting was
labeled MIT, while its referenced v1.0.0 source tag contains GPLv3. Correcting
that source label changes distributed notice bytes and invalidates earlier
candidate package hashes. Rebuild the final packages and bound upgrade helper.
The [licensing review packet](LICENSING_REVIEW_1.5.3.md) records the exact source
identity, proposed review path, and model/codec questions; it does not alter
the project's attribution terms or supply legal clearance.

On 5 October, the maintainer confirmed that a legal friend completed the
review and directed work to proceed. The candidate retains the current terms,
adds the missing complete GPLv3 text, and aligns references with the existing
GPLv3-or-later grant. The technical acceptance recorded in #100 is complete;
that issue remains open only until the reviewed candidate in PR #136 merges
into main. Final release artifacts still need source-bound notice,
contributor-credit, metadata, and component/provider checks. Those release checks
do not add a new #100 closure gate or imply unreported legal clearance.

The first ARM64 release workflow exposed three GPU retry/memory-pressure tests
that inherited the runner's ARM64 platform restriction instead of declaring
their Linux x86_64 scenario. The candidate now sets that scenario explicitly
in those tests; application platform restrictions are unchanged. All 43
dependency-installer tests and the three cases under a simulated ARM64 host
passed locally. The actual ARM64 workflow must pass on the amended head.

## Evidence and remaining decisions

Prior development-package tests guide qualification but do not transfer
automatically to newly built v1.5.3 artifacts.

| Area | Existing evidence | Remaining release work |
| --- | --- | --- |
| Unit, packaging, and CI | #135: 940 local tests passed; six PR checks passed; merged source tree equals the tested head | Run checks on the v1.5.3 head and manual release-build workflows |
| Small window UI | Maintainer tested Camera/Audio and restored camera preview in an installed development Snap | Repeat on the exact release candidate |
| Recording | Installed Snap speech playback passed; exact development DEB/RPM/Flatpak physical-camera recordings passed, with generated selected audio; extended Flatpak recording passed | Final artifact tests, native/Flatpak desktop and physical-microphone acceptance, applicable codec/provider redistribution checks |
| CPU/GPU behavior | Installed CPU fallback, GPU recovery, live CUDA, and cold-cache probe fixes were verified before their merges | Exact candidate CPU/CUDA/TensorRT checks on applicable devices; record memory pressure separately from a broken installation |
| Matte quality (#91) | Patched Remove passed direct-window and backlit moving-hand feedback | Final-package Blur/Replace, fine hair, white clothing, and extreme sliders; document any accepted residual limitation |
| Source recovery (#53) | Failed-install preservation, CPU/CUDA transitions, rollback and source window startup tested in #133 | Candidate source update/rollback check; production native runtime packs remain a separate unfinished scope |
| Native artifacts (#60) | CPU prototypes passed eight lifecycle cells; unsigned archives reproduced in recorded builders | Build the production DEB/RPM, bind the exact upgrade helper, verify final manifests/provenance and installer transactions; do not claim hermetic production payloads |
| License/redistribution (#100) | Maintainer confirmed a legal friend reviewed the existing terms; complete GPLv3 text and consistent grant references added; issue acceptance is complete | Merge PR #136 into main, then resolve #100 with the merge and qualification references. Continue exact final-artifact notice/component checks in the release record |
| Flatpak (#95) | Development build, dependency closure, model trust, recording, physical camera, and initial virtual-camera read passed | CPU release conditions below; remains excluded until applicable gates pass |
| macOS | Maintainer requested GitHub-runner signing and a physical Mac test on 7 October; package/audio blockers and signing tooling are being qualified | Configure protected Installer/notary credentials, pass installed-runtime and actual signed/notarized/stapled package checks, then physical Mac acceptance; see [MACOS_SIGNING.md](MACOS_SIGNING.md) |
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
2. Verify complete license/notice contents, accepted contributor credit, and
   bundled codec/dependency scope on the exact Flatpak artifact under the
   maintainer-confirmed arrangement. #100's separate remaining gate is the
   PR #136 main merge.
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

Exact-source checks, native artifact/source/notices, public DEB/RPM CPU
upgrades, unsigned installed macOS runtime and both Snap confined generated
recordings passed at reviewed source `0e4d2b5`, integrated with exact tree
equality at `51bcb12`. All six subsequent PR checks passed against the current
main/release integration. Fresh actual-Snap Python advisory lookups found no
known findings or skipped upstream identities in their bounded scope.

The date amendment exposed a native reproducibility defect: clamping mtimes
does not normalize generated files whose timestamps precede a future release
epoch. The DEB builder now normalizes the complete staging tree, and RPM
normalizes source staging and its final buildroot. Future-epoch repeat-build
checks cover both formats. This changes archive metadata without changing
application, runtime-checker, installer or dependency behavior. Carry forward unchanged
functional evidence, verify rebuilt/tag-bound package metadata, payloads,
checksums and provenance, and complete the affected candidate tests. Earlier
package hashes remain historical. Both original Apple submissions now report
Accepted in authenticated status-only runs, but their signed uploads were not
retained and cannot qualify the current PKG. The final tag-built macOS package
must complete its own signing/notarization/staple/Gatekeeper gates once.

The maintainer completed the installed v1.5.3 Snap x10 check: camera preview,
compact-window Camera/Audio controls, Show Preview, CPU/DocZeus availability,
moving open-hand Blur/Remove and saved video with audible speech all passed.
Actual confined CPU/CUDA inference and fresh CuPy compilation also passed;
the narrow Snap CUDA profile does not bundle TensorRT libraries. A nondefault
physical microphone was not explicitly confirmed. Mac prerequisites are now
installed on the maintainer's M2 Mac; the final signed package's physical
camera/microphone test and the tag/candidate/soak sequence remain pending.
**Stable publication is authorized but not yet technically qualified.** Follow
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) without restarting unchanged
behavior merely for metadata, while recording the affected installer/candidate
window and any accepted non-blocking limitations.


## macOS blocker amendment, 7 October 2026

The maintainer resumed macOS signing and chose GitHub macOS runners plus a test
on their own Mac. Source review reproduced missing CoreAudio microphone routes
and incomplete package/runtime bootstrap; it also found root postinstall
executing user-owned Homebrew Python and network pip. These are recorded
release blockers, so the candidate freeze permits their targeted amendment.

The amendment uses native CoreAudio for selected-microphone recording and Mic
Test, keeps unsupported system-audio/processed virtual-mic routes explicit,
and separates admin-owned package installation from user runtime provisioning.
The new signing path requires a timestamped Installer identity, accepted
notarization, a stapled ticket and Gatekeeper acceptance. Tag artifact hashes
and attestations bind final signed bytes; unsigned macOS packages cannot enter
that release path. Prior package hashes/source identities remain historical
baseline evidence and do not qualify the amended payload. Rebuild affected
packages, verify Linux regressions, run actual Mac package/runtime CI, and
record signing plus physical-device acceptance before claiming readiness.

# v1.5.3 release readiness

Assessment started **5 October 2026**, from main commit
`254ec93b5351c8e50815b3bf03dce6ae91ef1726`. The latest published version remains
[v1.5.2, published 4 September 2026](https://github.com/Hkshoonya/nvidia-broadcast-linux/releases/tag/v1.5.2).

**Status: merged, tagged, draft packages verified and Store candidate promoted;
installed candidate CPU/CUDA/CuPy checks passed; feedback and stable publication
remain pending, 9 October 2026.** PR #136 merged into main at
`048d4f4005a56e6a7b235739f0176e0264bc3fa7`, exactly matching the tested integration
tree `f2ffa891c607a3effb653a545eb150c2a4f6c5e0`. The immutable `v1.5.3` tag points
to that source. The maintainer authorized this merge, tag, verified Store
candidate and eventual stable publication once the technical gates pass.
[Tag package run 37835368212](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/37835368212)
completed native build/test, actual macOS signing/notarization, package
attestation and draft-release creation. A draft is not a public release.

The affected feedback window started **8 October 2026 at 20:55:02 UTC**. Its
earliest 48-hour milestone is **10 October 2026 at 20:55:02 UTC**, subject to
remaining candidate checks and regressions; no publication time is guaranteed.
Website download changes are prepared for publication;
the deployed website stays on the current public release until then. Supported
public Flatpak distribution remains excluded. Keep actual artifact, Store
revision, device acceptance and publication receipts in the release record;
verify public asset URLs and Store versions before deploying the website update.

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
changed, so earlier package hashes and installed-runtime evidence did not qualify
this amendment. The final tag packages were rebuilt and the affected checks
completed as recorded below; the earlier hashes remain historical evidence.

The same amendment fixes the legacy macOS source installer's prerequisite
selection and required native-plugin checks before replacing its existing venv.
It retains the legacy in-place replacement limit documented in the README;
PKG per-user setup and notarization behavior are separate. Flatpak CI now retains
unsigned development bundles only after import, OSTree integrity, Git source,
resource, notice, and sandbox-metadata checks. This retention does not satisfy
the public-distribution or physical-device gates below.

The release audit found one notice error under #100: RobustVideoMatting was
labeled MIT, while its referenced v1.0.0 source tag contains GPLv3. Correcting
that source label changed distributed notice bytes and invalidated earlier
candidate package hashes. The final packages and bound upgrade helper were
rebuilt and their exact source, notices and provenance verified below.
The [licensing review packet](LICENSING_REVIEW_1.5.3.md) records the exact source
identity, proposed review path, and model/codec questions; it does not alter
the project's attribution terms or supply legal clearance.

On 5 October, the maintainer confirmed that a legal friend completed the
review and directed work to proceed. The candidate retains the current terms,
adds the missing complete GPLv3 text, and aligns references with the existing
GPLv3-or-later grant. The technical acceptance recorded in #100 is complete;
PR #136 has now merged into main and #100 was resolved on 8 October with the
merge and qualification references. Final package/source checks preserve the
complete license, notice and contributor records. Component/provider scope
remains recorded with each package; those release checks do not add a new #100
closure gate or imply unreported legal clearance.

The first ARM64 release workflow exposed three GPU retry/memory-pressure tests
that inherited the runner's ARM64 platform restriction instead of declaring
their Linux x86_64 scenario. The candidate now sets that scenario explicitly
in those tests; application platform restrictions are unchanged. All 43
dependency-installer tests and the three cases under a simulated ARM64 host
passed locally. The final tag workflow's actual ARM64 Linux gate has now passed.

## Evidence and remaining decisions

Prior development-package tests guide qualification but do not transfer
automatically to newly built v1.5.3 artifacts.

| Area | Existing evidence | Remaining release work |
| --- | --- | --- |
| Unit, packaging, and CI | The frozen release merge/tag tree equals the tested integration; applicable PR checks, tag Build Packages run 37835368212 and paired Snap candidate promotion run 37842894804 passed | Record remaining candidate acceptance, affected feedback and stable/publication receipts; subsequent main fixes do not change the frozen release |
| Small window UI | Installed Store candidate 190 owner moving preview, compact Camera/Audio controls and Show Preview passed; physical signed-PKG camera/compact controls passed | Record affected feedback and any regressions |
| Recording | Earlier x10 Snap speech playback is retained through exact application/native input matching; final signed-PKG built-in-microphone Mic Test, complete speech Rec and microphone-only Meeting transcription passed; prior native/Flatpak capture and generated-media evidence is retained | No new physical recording was claimed for Store 190; native/Flatpak physical-microphone and desktop follow-up remains in #112, separate from the completed Mac scope; preserve provider scope and unavailable-second-mic limitation |
| CPU/GPU behavior | Installed Store 190 exclusive CPU inference passed; on 9 October, CUDA inference with CPU fallback disabled and fresh CuPy NVRTC execution passed on RTX 5070. All 78 wheel RECORD inputs and 82 loaded native package files match the qualified review inputs | Record affected feedback. These small execution probes do not measure camera FPS or long-run performance. TensorRT SDK is outside this Snap profile; earlier x10 evidence remains separately identified |
| Matte quality (#91) | Patched Remove passed direct-window and backlit moving-hand feedback | Final-package Blur/Replace, fine hair, white clothing, and extreme sliders; document any accepted residual limitation |
| Source recovery (#53) | On 9 October, the exact frozen source passed CPU runtime upgrade from reviewed generation `96c83fb`, real failed-index preservation and rollback/return with each generation's own interpreter/imports; the older checker lacks `root_extras`, exercising the compatibility path | Source CUDA transitions, full system/device installation and production native runtime packs remain outside this final CPU qualification; #53 remains open |
| Native artifacts (#60) | Final tag DEB/RPM, bound helper, source/notices and hosted provenance passed independent checks; fresh public v1.5.2 to exact final CPU upgrades passed 17 steps on Ubuntu 24.04 and 16 on Fedora 44; eight historical prototype lifecycle cells are retained separately | Record candidate/publication receipts; production runtime payloads remain non-hermetic, with locked/offline dependencies, RPM signing and full lifecycle work separate |
| License/redistribution (#100) | Existing terms reviewed under maintainer-confirmed arrangement; complete GPLv3 and grant references shipped; PR #136 merged and #100 resolved | Preserve exact artifact notice/component evidence and the recorded review scope |
| Flatpak (#95) | Development build, dependency closure, model trust, recording, physical camera, and initial virtual-camera read passed | CPU release conditions below; remains excluded until applicable gates pass |
| macOS | Exact tag PKG passed actual Installer/team signing, Accepted notarization, stapling, Gatekeeper, payload/checksum and hosted provenance; owner M2 camera, compact controls, CPU effects, built-in-mic speech Rec, OBS output and microphone-only Meeting passed | Candidate/soak/publication record; a second mic was unavailable, numeric FPS and physical upgrade/uninstall were not supplied; keep macOS 13/14, Intel and CoreML outside qualified scope |
| Snap edge automation (#90) | Candidate is 190/189; stable and edge remain v1.5.2 at 185/184 in the verified Store receipt | Verify connected builder configuration and scoped edge credentials before merging the edge-promotion draft |
| NixOS (#18) | Draft integration `e1fee563` reports v1.5.3; offline package build passed 1,140 tests and exclusive CPU inference, module evaluation and all six GitHub checks passed | No real NixOS machine available; draft remains unsupported pending physical camera, v4l2loopback, microphone and virtual-microphone acceptance; it is outside the frozen release |

## Flatpak decision for this release

The first possible release scope is **x86_64 CPU**. CUDA/TensorRT and aarch64
need separate implementations and qualification; their absence does not by
itself prevent an accurately described CPU-only package.

Still required for that CPU package:

1. Final installed-package desktop acceptance: real Wayland camera and virtual
   camera, effects, microphone processing and virtual output, shortcuts/tray,
   recording, and client reconnect behavior. Current recorded desktop testing
   used X11; generated tone checks do not establish audible physical speech.
2. Preserve the verified license/notice contents, accepted contributor credit,
   and recorded dependency scope when selecting a public artifact. The retained
   development artifact's exact project source/notices passed inspection;
   #136 merged and #100 was resolved. That completed metadata work does not
   establish Flathub policy acceptance or the remaining distribution gates.
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

The final source-recovery qualification used a disposable Ubuntu 24.04
container as ordinary UID 1000, with no host mounts, display/audio sockets or
physical devices. It ran each source installer's actual runtime section, then
the complete frozen `install.sh --rollback-runtime` entry point twice. The
forced unavailable-index failure left the selection bytes and generation
directory set unchanged. Both rollbacks reran pinned CPU inference; all 62
installed application Python files matched each generation's source. The
older baseline `96c83fb` is a reviewed source generation, not the public
v1.5.2 source tag. No dependency or inference stubs were used. This completes
the final CPU source update/rollback check, without claiming source CUDA,
offline delivery, physical devices or power-loss recovery.

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
completed its own signing/notarization/staple/Gatekeeper gates once.

The maintainer previously completed the v1.5.3 development Snap x10 check:
camera preview, compact-window Camera/Audio controls, Show Preview, CPU/DocZeus availability,
moving open-hand Blur/Remove and saved video with audible speech all passed.
Actual confined CPU/CUDA inference and fresh CuPy compilation also passed;
the narrow Snap CUDA profile does not bundle TensorRT libraries. A nondefault
physical microphone was not explicitly confirmed for that Snap session.

[Candidate promotion run 37842894804](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/37842894804),
attempt 1, completed successfully from immutable tag source `048d4f4`. The
public Store receipt at **8 October 20:56:23 UTC** verifies latest/candidate
**AMD64 revision 190 / ARM64 revision 189**, both v1.5.3. Their Store SHA3-384
digests and sizes match fresh reads of the actual reviewed files, which retain
their own exact-run/tag/source cryptographic attestations. This verifies the
Store record against those files without claiming a fresh CDN transfer or
detached Snap assertion check. Stable and edge remain v1.5.2 at **185 / 184**.
The stable promotion step was skipped. The preserved candidate run, Store
response and digest bindings are in
`dist/qualification/snap/candidate-run-37842894804-attempt-1/` in the tag checkout.

Store candidate revision **190 is now installed**. The accepted local Snap
assertion chain binds its revision, size and SHA3-384 to the qualified reviewed
artifact. Actual mounted checks verified 704 files, including all 64 application
source files and 79 source/resource/notice copies. Fresh installed CPU inference
returned `[1, 4, 9, 16]` with profiling proving exclusive CPUExecutionProvider
execution. All 78 installed wheel RECORD inputs and the seven native files
actually mapped by that CPU check match the qualified hashes. This uses
snapd's accepted assertion database and bounded mounted inputs; the root-owned
backing archive was not independently rehashed. The maintainer confirmed the
moving preview, compact Camera/Audio controls and Show Preview work. Their
Store-190 physical recording check was not repeated.

On 8 October, the exact-190 GPU checks did not start because selected-device
memory was below the 4,096 MiB reserve. On 9 October, with 11,757 MiB free on
RTX 5070, both confined checks passed using fresh private caches. CUDA completed
in 62.350282 seconds wall time with CPU fallback disabled and profiling showing
only CUDA kernel execution. CuPy compiled and executed a fresh NVRTC kernel in
2.076677 seconds. Both returned `[1, 4, 9, 16]`; neither timed out or emitted
stderr. Fresh hashes bind all 78 wheel RECORD inputs and 82 mapped native
package files to the qualified review inputs. These startup checks do not
measure preview FPS, all processing models or long-run GPU performance.

The earlier x10 physical evidence and distinct tag/review archive hashes remain
recorded separately. Installed identity, provider/native bindings and owner
acceptance receipts are in `dist/release-1.5.3/candidate/installed/`; the new
GPU evidence is under `gpu/retry-20261009T171844Z/`. No CPU repeat, capture,
configuration change or process unloading was needed. Remaining feedback and
regressions are recorded against candidate revisions 190/189.

The immutable tag retains a known RPM reporting limitation reproduced in both
public v1.5.2 and final v1.5.3: if CUDA setup and its clean CPU fallback both
fail, POSTIN can return success and continue integration. Successful CUDA,
successful fallback CPU and direct CPU outcomes are unchanged. The targeted
[PR #142](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/142) follow-up
merged on 8 October at 21:06:07 UTC as main commit
`1f8865833d7be9ab2bbee028ee05009dcf8656f1`, adding explicit failed-fallback
status for subsequent work; it is not included in v1.5.3. The release tag
remains `048d4f4`. This is a carried-forward reporting limitation, not a new
release regression.

The 9 October installer review reproduced a separate mandatory-stage failure
path in both DEB and RPM: cleanup, venv creation or bootstrap pip failure can be
hidden by a later successful runtime command. The affected function bodies are
identical in v1.5.2, v1.5.3 and pre-fix main. The six explicit failure guards in
[PR #143](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/143) address it
for subsequent work. It merged on 9 October at 17:28:06 UTC as main commit
`086f1a5815f65a1f1ce0cb217184cc6e361cce57`. All 18 controlled failure cases failed
before the fix and passed afterward; independent review and all five CI checks
passed. The immutable release tag is
unchanged; package-manager rollback and atomic activation remain separate.

The final signed macOS package is 488,100 bytes, SHA-256
`8a4df3d903f80d90b866adcd6571f34368b015aeae8023f35aba8472c61af2e2`, signed by
**Developer ID Installer: WeMakeSense LLC (T39RXSKKZ6)**. New Apple submission
`4e82228e-8dc6-401c-9788-445964a562e4` reports Accepted and its log identifies the
exact pre-staple upload. Native CI verified the trusted timestamp, matching
Installer/team identity, unchanged payload, stapled ticket, Gatekeeper's
Notarized Developer ID acceptance and final-byte checksum. Independent
inspection verified all 83 source inputs, equal unsigned/pre-staple/final
payloads and scripts, signing evidence hashes, actual XAR signature and the
hosted tag attestation. The Mac runtime identity is
`7a5e963ab32bab096655feed0da13a348796039507d3096ddb2aa65c0de1dcf2`.

Following the exact signed-package instructions, the maintainer reported a
successful physical test on their MacBook Air M2: setup/app startup, camera and
compact controls, CPU Blur/Remove, built-in-microphone Mic Test and complete
speech recording/playback, processed OBS video in another app, and
microphone-only Meeting recording/transcription. No second microphone was
available. The owner's checksum/runtime output was not separately supplied;
package identity is bound by the verified instructions and downloaded
artifact. Numeric FPS, physical upgrade/uninstall and other Mac versions were
not reported. These limits remain distinct from the passed acceptance scope.

**Store-candidate promotion is complete; stable publication is authorized but
applicable candidate checks and feedback remain pending.** Follow
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) without
restarting unchanged behavior merely for metadata. Record the exact candidate
revisions, affected test window, accepted limitations and public publication
receipts. The prepared website update must be finalized against actual public
asset URLs and channel versions after publication.


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
baseline evidence and do not qualify the amended payload. Those affected
rebuild, Linux regression, actual Mac package/runtime CI, signing and available
physical-device checks are now complete in the scopes recorded above. The
remaining candidate feedback and publication steps still apply.

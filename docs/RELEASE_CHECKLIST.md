# Release Checklist

Use this checklist for every stable release. Record the release tag, source
commit, workflow runs, Snap revisions, validation results, and approver in the
release tracking issue or pull request. A passing build from another commit is
not evidence for the release candidate.

## 1. Prepare the release tree

- Start the release branch from the current `origin/main` and record its commit.
- Choose the version and planned publication date. Update them consistently in
  `pyproject.toml`, `src/nvbroadcast/__init__.py`, `snap/snapcraft.yaml`, Debian
  and RPM metadata, AppStream metadata, installer/build copy, the changelog,
  and the release notes.
- Keep `docs/index.html` download commands on the latest public version while
  the new release is still a candidate. Keep the `published` version in
  `tests/test_packaging_metadata.py` aligned with those public links.
- Confirm release notes distinguish completed work from open follow-up work and
  credit merged community contributions accurately.
- Review the complete diff for unrelated changes, generated files, credentials,
  private data, and release claims that exceed the performed verification.

## 2. Run pre-tag gates

- Run `git diff --check`, metadata drift tests, Ruff, Bash syntax checks,
  actionlint, and ShellCheck where applicable.
- Run the complete non-hardware test suite and `scripts/release_smoke.py`.
- Run `pip check`, Bandit, and the dependency vulnerability audit against a
  clean resolution of the release tree.
- Build and inspect the Debian and RPM packages locally or in disposable clean
  builders. Do not install build dependencies into a user's production system.
- Render the native-package upgrade helper from those exact DEB/RPM files.
  Verify its embedded hashes, executable syntax, package-identity checks,
  fail-closed legacy-script handling, checksum entry, and attestation subject.
- Open the release pull request and require all PR checks, supported Python
  jobs, and review requirements to pass on its current head.
- Run the `Build Packages` and `Build & Publish Snap` workflows manually from
  the exact release commit, without a Store action. Confirm Linux amd64/arm64,
  Apple Silicon, Python-version, package, and strict Snap jobs pass.
- Run CPU, CUDA, and TensorRT execution probes on applicable real hardware.
  Record unsupported combinations explicitly instead of treating a skipped
  hardware test as a pass.

Stop if a required check fails, a high or critical exploitable vulnerability is
unresolved, package contents differ from expectations, or the exact candidate
commit changes. Fix the release branch and restart affected gates.

## 3. Merge and create the candidate

- Merge the release pull request only after approval. Record the resulting
  `main` commit and confirm it contains exactly the reviewed release changes.
- Re-run any required checks invalidated by the merge commit.
- Create `vX.Y.Z` only after explicit release authorization. Verify the tag
  resolves to the recorded release commit and matches every package version.
- Let the tag-triggered `Build Packages` and `Build & Publish Snap` workflows
  finish. They must create draft GitHub release assets; tagging alone must not
  publish the release or move the Snap stable channel.
- Download and inspect the DEB, RPM, and PKG. Verify every entry in
  `SHA256SUMS.packages` and verify each package attestation against the exact
  tag, commit, repository, signer workflow, and a GitHub-hosted runner.
- Verify every tag-built Snap attestation. When a Snap fits GitHub's release
  asset limit, also verify its `SHA256SUMS.snap` entry. A missing large Snap
  release asset is expected only when the workflow reports the size limit.
- Upload the exact tag-built Snap revisions for Store review when required.
  After approval, promote the recorded amd64 and arm64 revisions together to
  `candidate`; do not substitute a rebuild from another ref.

## 4. Test and soak the candidate

- Install or upgrade from the built DEB and RPM on clean supported Linux
  systems. Verify launch, camera selection, virtual camera output, microphone
  selection, audio output, recording, profiles, updates, and uninstall/upgrade.
- Exercise the exact previous public DEB and RPM upgrade paths through the
  release helper. Confirm normal direct clean install/reinstall, helper-based
  legacy upgrade, package-manager failure recovery, and final removal leave no
  installer-generated `/opt/nvbroadcast` residue.
- Test the Apple Silicon PKG on supported macOS, including prerequisite checks,
  OBS Virtual Camera output, upgrade behavior, and clean failure paths.
- Test both Snap architectures from `candidate`. On amd64, exercise CPU, CUDA,
  and TensorRT where supported; on arm64, verify the CPU path and face effects.
- Test no-camera startup, busy-camera recovery, live device changes, quality and
  processing-mode changes, auto framing at zero and non-zero zoom, stable and
  moving background modes, hair/facial-hair edges, and difficult lighting.
- Monitor candidate feedback and open regressions for the planned soak window.
  Use 48-72 hours for a release with runtime, camera, audio, or packaging
  changes unless an urgent security fix requires a documented shorter window.

Reset the candidate or delay publication for crashes, data loss, startup or
upgrade failures, security regressions, broken core camera/audio paths, or an
unexplained platform-specific failure. Record accepted non-blocking limitations
in the release notes.

## 5. Publish stable

- Obtain explicit publication approval after reviewing the exact commit,
  workflow results, artifact verification, Store revisions, soak results, and
  unresolved issues.
- Run `Promote Reviewed Snap Revisions` from the release tag with operation
  `stable` and empty revision inputs. Confirm it verifies and promotes the
  complete tested candidate set for both architectures.
- Confirm the stable Store revisions match candidate and Store metadata was
  refreshed. Install the stable Snap once on each available architecture.
- Publish the existing draft GitHub release for the same tag. Use the reviewed
  release notes and confirm all intended package and checksum assets are
  present before making it public.
- Confirm a Store review dispatch from the release tag neither enters the Snap
  attachment job nor changes the GitHub Release out of draft state.
- Verify the public release page, package checksums, provenance instructions,
  and in-app update detection.

## 6. Complete post-release metadata

- In a follow-up change, update all `docs/index.html` direct download commands
  and visible current-version copy to `X.Y.Z` only after the public assets
  exist. Change `published` in `tests/test_packaging_metadata.py` to the same
  version.
- Run the metadata tests and verify each published website download URL returns
  the expected file rather than an error page.
- Merge the website update and confirm the GitHub Pages deployment succeeds.
- Verify upgrade discovery from the previous release and check the Snap stable
  listing from a logged-out browser session.
- Update the release tracking issue and relevant bug reports with concise,
  verified outcomes. Keep broader work such as hermetic builds, SBOMs,
  reproducibility, and native signing open until its own acceptance criteria
  are complete.

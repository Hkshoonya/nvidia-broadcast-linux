# Verifying Release Artifacts

Release artifacts built after this verification policy lands include two forms
of integrity evidence:

- `SHA256SUMS.packages` covers the DEB, RPM, and macOS PKG assets.
- `SHA256SUMS.snap` covers Snap files small enough to attach to GitHub Releases.
- GitHub artifact attestations bind each package digest to the repository,
  source commit, triggering event, and workflow that produced it.

These controls apply only to artifacts built by workflows containing this
policy. They do not retroactively attest older releases.

## Verify A Checksum

Download the package and its matching checksum manifest into the same
directory. On Linux, name the exact native package you downloaded:

```bash
ARTIFACT=nvbroadcast_X.Y.Z-1_all.deb
awk -v file="$ARTIFACT" '$2 == file { print; found=1 } END { exit !found }' \
  SHA256SUMS.packages | sha256sum -c -
```

For a downloaded Snap, use:

```bash
ARTIFACT=nvbroadcast_X.Y.Z_amd64.snap
awk -v file="$ARTIFACT" '$2 == file { print; found=1 } END { exit !found }' \
  SHA256SUMS.snap | sha256sum -c -
```

On macOS, select the downloaded PKG entry and pass it to `shasum`:

```bash
ARTIFACT=NVBroadcast-X.Y.Z-1.pkg
awk -v file="$ARTIFACT" '$2 == file { print; found=1 } END { exit !found }' \
  SHA256SUMS.packages | shasum -a 256 -c -
```

Treat a missing entry, a checksum mismatch, or a command that verifies no files
as a failed verification. Do not install that download.

## Verify Build Provenance

Install a current [GitHub CLI](https://cli.github.com/) release, authenticate if
GitHub requests it, and set `TAG` to the release tag shown on GitHub. Resolve
that tag to its current commit before verifying the artifact:

```bash
TAG=vX.Y.Z
SOURCE_COMMIT="$(gh api "repos/Hkshoonya/nvidia-broadcast-linux/commits/$TAG" --jq .sha)"
```

Verify the downloaded artifact against this repository, the expected signer
workflow, the exact release tag and commit, and a GitHub-hosted runner.

For DEB, RPM, or PKG files:

```bash
gh attestation verify ./PATH_TO_PACKAGE \
  --repo Hkshoonya/nvidia-broadcast-linux \
  --signer-workflow Hkshoonya/nvidia-broadcast-linux/.github/workflows/build-packages.yml \
  --source-ref "refs/tags/$TAG" \
  --source-digest "$SOURCE_COMMIT" \
  --deny-self-hosted-runners
```

For Snap files downloaded from GitHub Releases:

```bash
gh attestation verify ./PATH_TO_SNAP \
  --repo Hkshoonya/nvidia-broadcast-linux \
  --signer-workflow Hkshoonya/nvidia-broadcast-linux/.github/workflows/snap.yml \
  --source-ref "refs/tags/$TAG" \
  --source-digest "$SOURCE_COMMIT" \
  --deny-self-hosted-runners
```

Successful verification proves that the artifact digest was signed through
GitHub Actions for this repository using Sigstore-backed identity. It links the
artifact to the source commit and workflow shown in the result. It does not, by
itself, prove that the source is vulnerability-free or functionally correct.

The Snap Store separately signs and distributes Store revisions. The GitHub
attestation applies to the matching file produced by this repository's build
workflow. Before promotion to the stable Store channel, both candidate
architectures must verify against the Snap workflow attestation, the exact
version tag, that tag's commit, and a GitHub-hosted runner. Store-delivered snaps
then continue through Snap's own verification chain.

## Remaining Supply-Chain Work

This is the provenance foundation tracked by issue #60, not completion of that
issue. The current native installers still resolve parts of their Python
environment online during installation, so the complete installed environment
is not yet hermetic or independently reproducible. Target-specific dependency
locks, offline wheelhouses, complete SBOMs, independent rebuild comparison, RPM
signing, and macOS signing/notarization remain separate acceptance work.

Official references:

- [GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations)
- [`gh attestation verify`](https://cli.github.com/manual/gh_attestation_verify)

Maintainers should use the [Release Checklist](RELEASE_CHECKLIST.md) to keep
pre-tag validation, candidate testing, publication, and post-release metadata
updates tied to the same release commit.

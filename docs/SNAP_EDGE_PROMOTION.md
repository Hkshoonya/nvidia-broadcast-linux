# Optional Snap edge promotion

The reviewed promotion workflow can copy the two verified candidate revisions
to `latest/edge`. This is optional release tooling. Candidate and stable
publication do not require edge activation.

Edge publication is **disabled by default**. An unset or false repository
variable `SNAP_EDGE_PROMOTION_ENABLED` stops an edge dispatch before any Store
command and prevents injecting the edge secret. Merging this tooling does not
activate it, create credentials, dispatch a release, or change Store channels.
Repository changes can still be observed by an independently connected builder;
the workflow cannot control that builder's private configuration.

Before enabling edge, a maintainer must complete and record both checks:

1. Inspect the publisher's authenticated [builds page](https://snapcraft.io/nvbroadcast/builds).
   Record any connected source/Launchpad recipe URL, Git ref, automatic-build
   setting, Store upload/publication channels, and queued jobs. Disconnect or
   make the alternate builder non-publishing and clear publishing jobs. An empty
   public recipe search or GitHub webhook list does not establish this state.
2. Create a separate, expiring `SNAP_EDGE_TOKEN` restricted to `nvbroadcast`,
   `latest/edge`, and the access/release permissions needed to inspect and
   release existing revisions. Do not grant upload or metadata-update access,
   broaden candidate/stable credentials, or reuse them for edge. Verify the
   effective restrictions using nonsecret account/scope metadata before saving
   the token as a GitHub Actions secret. Never commit or log its value.

Snapcraft documents the `--snaps`, `--channels`, `--acls`, and `--expires`
restrictions in its [authentication guide](https://snapcraft.io/docs/snapcraft-authentication/).
A secret's name alone does not prove its scope. Only after recording the
builder and credential checks should a maintainer set the repository variable
`SNAP_EDGE_PROMOTION_ENABLED` to exactly `true`. Unset it to disable the path.
This variable records an operational decision; it is not an automatic audit.

Use a future reviewed release tag that **contains this workflow**, with its
matching tested amd64/arm64 revisions already in candidate. Dispatch
`.github/workflows/snap-promote.yml` from that tag with `operation=edge` and
empty revision inputs. The unconditional preflight requires the tag, checkout,
package version, and workflow source SHA to agree before any Store credential
is used. Existing immutable release tags are not changed to add this feature.
The already published v1.5.4 tag therefore does not acquire edge support by
merging a later change to main.

The workflow downloads both exact Store revisions and verifies GitHub-hosted
artifact provenance against the release tag, source commit, and Snap build
workflow. It then rechecks candidate and releases only those captured revision
IDs. A later candidate change cannot substitute unverified bytes. Both existing
target-channel revisions must be available as a rollback baseline. Failed or
uncertain partial releases attempt to restore revisions still owned by the
operation, without knowingly overwriting a different publisher's revision.
Store operations across architectures are not atomic; an external publisher
can race a status check or rollback, which is why removing alternate publishing
paths remains necessary. Review any failed job's final Store state.

Edge never uploads Store metadata. Stable keeps its existing metadata update
after both target revisions have been verified. Provenance verification proves
artifact/source identity, not physical camera testing; record acceptance of the
actual candidate revision pair before dispatch.

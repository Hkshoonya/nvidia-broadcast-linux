# Complete native Linux runtime adapter

This adapter builds the current application's complete CPU or CUDA Python
environment before creating a native package. It uses the reviewed private
CPython 3.13.18 supplier archive and dependency hashes from the runtime
investigation, while building the application from an explicit Git commit.
Both formats retain the public package name `nvbroadcast`. Architecture is
`amd64` for DEB and `x86_64` for RPM, with a distinct `N.cpu` or `N.cuda`
package revision. The payload lives under `/usr/lib/nvbroadcast/runtime`.

The explicit `build-packages.sh native-runtime` entry point does not replace
the existing public release recipes. Promotion requires authenticated CI
builder delivery, RPM signing, final acceptance, and the remaining issue #60
platform gates. These candidates must not replace frozen v1.5.3 downloads.

## Build contract

`builders.json` pins the inspected local Docker image IDs of the three qualified
local OCI images and records their installed native package inventories.
The builder refuses a different image, including a rebuilt image with the
same friendly tag. Their historical snapshot import is authenticated for CI
consumption. Their reproducibility scope is fixed input consumption and repeated
assembly using these recorded tools; their apt/dnf origins are not independent
source rebuilds or snapshotted repository rebuild recipes.

CI must first obtain these exact images through one of these reviewed paths:

1. An OCI registry reference pinned to its immutable manifest digest, with
   provenance authenticating the builder source and the workflow that produced
   it. After pulling, the inspected local image ID must equal the pin.
2. A `docker save` archive with a pinned SHA-256 and authenticated provenance,
   checked before `docker load`. Its resulting inspected image ID must also match.

Do not regenerate floating apt/dnf images in a release workflow and remove the
identity check to accommodate them. An intentional builder update changes the
pins after repeating wheel reproduction, input audit and lifecycle acceptance.

### Delivering the historical builders

`builder-delivery.json` pins one exported OCI archive containing the three exact
approved image indexes. Its 620,776,960 bytes have SHA-256
`7b9579055f63c7f63516060c545c7d8207d07fb04ea8c038bc1c8185f790e3d4`.
The infrastructure asset ID is `631786999`; its uploaded name, size and GitHub
digest have been checked against the lock.
Qualification run `38175085983` and promotion run `38175299710` completed at
main source `3e1378faa9e2fb4ca4fee669288ecc8376d37b09`; the promoted lock pins
that signing/source authority.
The [import qualification record](qualification/2026-10-11-builder-import.json)
retains the evidence identities and cryptographic readback policy. A local hash
or the unsigned BuildKit metadata alone proves no publisher identity or original
build.

`builder_delivery.py inspect` validates the archive hash/size, every OCI blob,
descriptor size, index/platform graph, and the Docker compatibility manifest.
It rejects additional blobs, external content, tags, links, duplicate JSON/tar
entries and unexpected paths. BuildKit's embedded two-byte empty config must
match its separately hashed blob exactly. Verification and loading use the same
private copy, so replacing the supplied path cannot change the loaded bytes.

The historical `.Id` values are OCI **index** digests because their original
Docker daemon uses the containerd image store. A classic store can assign a
different config digest to equivalent layers; that is rejected. Qualification
requires containerd, all three original image IDs, Linux amd64, and exact native
package inventories (512 bindings, 355 DEB, 210 RPM entries). Inventory queries
run offline with a read-only root filesystem, dropped capabilities and no new
privileges. No apt/dnf command rebuilds or updates these images.

```bash
python3 packaging/native-runtime/builder_delivery.py qualify \
  --archive dist/builder-delivery/approved-images.tar \
  --report dist/builder-delivery/local-qualification.json
```

The bounded delivery sequence is:

1. Review the verifier, lock, qualification workflow and local qualification
   evidence. Create a separate infrastructure **draft prerelease** with tag
   `infra-native-builders-20261011`, targeting the reviewed source commit, and
   upload `nvbroadcast-native-builders-linux-amd64-20261011.oci.tar`. Never use
   `--clobber`, reuse a `v*` tag, or modify a frozen v1.5.3 asset. This archive is
   infrastructure and is not an application release or installation candidate.
2. Read back the exact asset ID, name, size and GitHub SHA-256. Commit that asset
   ID into the reviewed lock. A null or different ID rejects a guessed or
   unreviewed asset. A changed asset requires another explicit lock review.
3. Run **Qualify historical native builders** from main using that same asset
   ID. It has only `contents: read`, checks the release's tag/name/ID/size/hash,
   then loads and qualifies the images on a disposable GitHub-hosted runner.
   No signing key, OIDC token or package publication permission is available.
   If GitHub's read token cannot download the draft, publish only this separate
   prerelease with `make_latest: false` after reviewing its inputs, then retry.
   Preserve the qualification JSON and source commit/run/attempt as evidence.
4. Run **Authenticate historical native builder import** from main, passing
   the successful qualification run ID. Both runs must use the same exact main
   source commit. The promotion validates that run's success, repository, event,
   workflow path and attempt; checks its evidence artifact ID/digest; and compares
   both JSON records with the freshly inspected archive and reviewed inventories.
   Its separate signing job never loads or executes the images. It uses
   `contents: read`, `actions: read`, `id-token: write` and `attestations: write`;
   no RPM key, registry credential or new long-lived secret is needed. The pinned
   official attestation action signs the **historical builder import** claim,
   and a subsequent step verifies the minted identity and exact source/signing
   commit. The initial qualification and promotion both completed successfully
   on October 11, 2026, with independent cryptographic readback.
5. After successful promotion, review the lock change to `promoted` with the
   exact promotion workflow source commit. Release consumers must use
   `builder_delivery.py verify --load`, which verifies the archive against
   GitHub's cryptographic attestation before Docker load. Policy binds repository,
   `native-builders-promote.yml`, its exact source/signing commit, main ref,
   custom historical-import predicate type and a GitHub-hosted runner.

The import authenticates this project's deliberate adoption of these exact
historical bytes and subsequent delivery. It does **not** recover authenticated
provenance for their original network dependency installation. Independently
rebuilding them requires snapshotted package repositories and a new reviewed
builder generation. The delivery gate is complete for this pinned historical
snapshot; hosted CPU assembly and remaining package acceptance are separate.

The containerd requirement follows Docker's
[image-store documentation](https://docs.docker.com/engine/storage/containerd/).
Attestation verification follows the GitHub CLI's
[identity and source policy](https://cli.github.com/manual/gh_attestation_verify).

With the authenticated lock, **Build authenticated native candidate** is
the manual main-only consumer. It fetches the pinned archive, cryptographically
verifies the historical import before loading it, and builds both native formats
with application and adapter fixed to that run's `GITHUB_SHA`. It receives only
contents/attestations read permissions and retains complete unsigned DEB/RPM
artifacts plus bounded source, runtime and builder provenance. It does not publish
a release, access host devices, install native packages or receive signing keys.
Signing and device/lifecycle acceptance remain separate gates.

Host-side candidate workflow scripts use `/usr/bin/python3 -I -S`. The production
boundary tests need `packaging`; they zip-import only the packaging 26.0 universal
wheel already pinned in `runtime-prototype/pylock.build.toml`, after checking its
SHA-256 and size. No pip resolver, global install or ambient Python package is
used. The artifact includes the isolated interpreter/dependency verification
record in `host-tools.json`.

The CPU job requires **12 GiB actually free** on its workspace, temporary and
home filesystems before fetching builders. The Docker setup action keeps its
isolated image store under home. The October 11 complete local CPU build
used 5,905,301,504 disk bytes, with a separately measured 559,071,232-byte cache
(some hardlinks overlap), up to 2.49 GB of builder images, and up to 1.25 GB for
transport plus a private verification copy. The preflight includes headroom and
records actual free space; it fails without pruning unrelated runner tools or
images. The [first successful hosted CPU run](qualification/2026-10-23-hosted-cpu.json)
completed with 92,098,985,984 free bytes observed for each checked location.
That capacity observation applies to the recorded CPU run. The workflows now
accept an explicit `variant=cpu|cuda`, defaulting to CPU. CUDA requires 40 GiB
free before each hosted stage. This is a conservative allocation budget for
builder images, input cache, runtime/staging trees and private package copies,
not a measured hosted peak. The producer records free bytes before and after
assembly; insufficient capacity fails without pruning.

Trusted `candidate_policy.py` selects all bounds from the explicit workflow
input. Artifact metadata cannot change the selected variant or limits:

| Bound | CPU | CUDA |
| --- | --- | --- |
| Individual package | at most 1 GiB | strictly below 2 GiB |
| Aggregate Actions ZIP | at most 2 GiB | at most 4 GiB + 64 MiB |
| Decoded regular payload | at most 4 GiB | at most 6 GiB |
| Complete decoded tar stream | at most 5 GiB | at most 7 GiB |
| Free disk allocation gate | at least 12 GiB | at least 40 GiB |

The CUDA transport limit holds both packages and bounded provenance. Both
variants retain the 1 GiB individual decoded member, 1 MiB tar extension,
50,000 payload entries, 16 MiB metadata member and 32 MiB aggregate provenance
limits. Source lock, version suffix, archive layout and installed input selection
must all match the same explicit variant.

The following invocation runs on Linux x86_64 as an ordinary user. The cache
may contain verified input archives and previously downloaded dependency
wheels. Missing inputs are downloaded unless `--offline` is specified. Network
access is always disabled for wheel building, payload assembly and packaging.

```bash
python3 packaging/native-runtime/build.py \
  --variant cpu --source HEAD --package-revision 2 \
  --cache "$PWD/dist/native-cache" --output "$PWD/dist/native-cpu" \
  --bindings-image sha256:7295102d8391699de28d86df555bddae343c5540c32bb21cbd5af10f92fc07ad \
  --deb-image sha256:1275fea2392e54fa240ca3e3917b5a050cbcd081797f1208ec7a9feff9204e4b \
  --rpm-image sha256:e494d493b7f5a50fbcf741b86c46ca6f1c25a9bb90016068ff76bcfb4c78708c
```

The output must be a new directory. `packages.json` binds application source identity,
variant, version, artifact sizes/hashes, builder IDs and runtime manifest hash.
Both native formats use zstd: level 8 for CPU and level 10 for CUDA. After
packaging, the builder requires each archive to be a nonempty regular file
strictly below 2 GiB, matching [GitHub's release-asset limit](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
An oversized archive fails the build before that package is added to
`packages.json`; the failed output and build log remain available for diagnosis.
The RPM signing helper checks the actual signed file against the same limit
before publishing its output directory. These size checks do not replace
signature, provenance or installed-package qualification. Hosted CUDA finalization
uses the explicit variant policy described below.
The payload contains interpreter supplier metadata/licenses, exact wheels'
installation records, a complete PEP 751 dependency lock and builder inventory.
The dependency lock has no moving application pin: the application's wheel is
built offline from the requested Git archive, and its exact hash is added to
the final lock before installation. All installed Python source files are
compared against that Git commit before either native package is built. The
adapter's Git revision, tree and executable input hashes are recorded separately
in `build-provenance.json`, `packages.json` and installed `package.json`. Building
with uncommitted executable adapter inputs is refused. The application source
and adapter revisions may differ and must not be described interchangeably.

Dependency wheels retain their reviewed dependency epoch; only the application
wheel uses the current source epoch. This distinction preserves dependency
wheel hashes across application releases. A hash mismatch aborts instead of
resolving or accepting a new dependency. CPU and CUDA locks each contain exactly
one ONNX Runtime owner. CUDA's faster-whisper CPU ORT requirement has one scoped
provider substitution; another requesting package is rejected.

To rewrap a previously assembled runtime, add `--runtime /path/to/runtime`
and its independently verified `--manifest-sha256`. Both complete contents and
application source are checked again. Never copy an identity reported by an
untrusted payload and treat that as an authenticated manifest.

## Native installation and recovery

Both packages own a complete private interpreter and Python dependency closure.
They depend on native GTK, GStreamer, PipeWire/PulseAudio command utilities,
audio and device libraries supplied by the
operating system. Offline installation therefore assumes those declared native
dependencies are already installed or available in a trusted local OS mirror.
The packages never invoke pip, uv, a network downloader, or a resolver from a
maintainer script. Model downloads requested later by an application feature
retain their existing separate model verification policy.

The launcher runs private Python with `-I -B`; system and user Python modules
cannot enter the private import path, and launch does not mutate package files.
Before replacing the private interpreter, the native prepare hook refuses a
running application. Quit Broadcast and its virtual-camera service before an
upgrade. Each native launcher holds a shared lock for its process lifetime;
prepare takes the same lock exclusively while checking legacy/orphan processes
and creating the transaction marker. This serializes new native launches with
the start of unpack. The marker then blocks launches until configuration ends.
The root-owned zero-byte `/var/lib/nvbroadcast/runtime.lock` inode is retained
across removal/reinstallation to preserve lock identity. Package/state parent
directories and marker types, ownership and permissions are checked before writes.
Configuration verifies every runtime file, directory, symlink, permission and
root owner, checks distribution closure including the selected CPU/CUDA and
meeting-support extras, verifies the managed faster-whisper version and
single-provider ownership, then
clears that marker. Corruption leaves it in place and reports a configuration
failure. RPM script errors may leave a package registered as installed; the
marker and verified upgrade helper prevent interpreting that as usable success.

The new prefix avoids legacy RPM removal hooks that erase `/opt/nvbroadcast`.
Generated legacy `.venv` files are removed only after the new package passes
configuration, and only when native ownership of the old installer was detected.
Redirected, writable or user-owned legacy environments are preserved for manual
review. Legacy launchers do not participate in the new lock; their migration
still requires dedicated lifecycle qualification.
User settings, recordings and shared virtual-camera driver configuration are
preserved. Native repair uses exact authenticated packages through dpkg or RPM;
these package-owned environments never use the source-generation activator.
An RPM killed during payload unpack can leave temporary files that a reinstall
does not remove. The native RPM now includes `nvbroadcast-recover-rpm`, described
in [authenticated RPM recovery](../../docs/NATIVE_RPM_RECOVERY.md). It uses
system Python and the project public key embedded in its signed package payload.
It verifies the supplied RPM, reads its payload without extracting it, and
preserves only recognized fragments with matching bytes and one transaction ID
bounded by the transaction marker. Neither retained build staging nor external
content/checksum manifests are inputs. An optional signed previous RPM recognizes
paths removed by an interrupted upgrade. Unknown private-prefix paths abort the
entire plan, and unrelated system files are not touched.

The default command only inspects. Explicit preservation retains the verified
RPM, public key, original fragments and durable audit under
`/usr/lib/nvbroadcast-recovery`, while keeping the launch-blocking marker.
The user then replays that exact saved package through RPM and checks native
configuration/integrity before launch. Successful preservation alone is not
successful repair. The helper is shipped by this native adapter; existing
v1.5.3 public packages are not changed.

This adapter blocks interrupted launches but does not keep the previous runtime
launchable during a package-manager transaction. Package metadata, native
transactions, abrupt power-loss/fsync behavior and a self-contained application
updater need separate qualification. The complete CUDA payload is approximately
5 GB before compression; size pruning and additional native platforms remain
explicit support-policy work.

## Signatures and upgrade helper

Sign the RPM before rendering a helper or creating public checksum manifests.
The existing helper renderer supports the new exact variant and architecture:

```bash
python3 scripts/render_native_upgrade_helper.py \
  --template scripts/native_package_upgrade.sh.in \
  --deb dist/native-cpu/deb/nvbroadcast_1.5.3-2.cpu_amd64.deb \
  --rpm dist/native-cpu/rpm/build/RPMS/x86_64/nvbroadcast-1.5.3-2.cpu.x86_64.rpm \
  --version 1.5.3 --revision 2.cpu \
  --output dist/native-cpu/nvbroadcast-native-upgrade
```

The helper is bound to the exact artifact SHA-256 values. It retains the narrowly
recognized legacy-script repairs, compares native RPM revisions correctly, and
checks the completed private runtime before reporting success. Existing numeric
revision helpers continue to use the legacy `all` / `noarch` artifact contract.
Use the rendered helper's `--offline` option to forbid APT downloads and disable
DNF repositories. This matters even when every dependency is present: DNF may
otherwise refresh repository metadata before considering a local RPM. Missing
native dependencies fail clearly; they are not silently downloaded in this mode.

## Qualification

### Finalized CUDA clean-VM functional checks — 26 October 2026

**Functional results only; release promotion is blocked.** The tested
`1.5.3-7.cuda` packages retain certifi `2026.7.22` and Pillow `12.3.0`, which
match the [certifi trust-store advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2)
and [Pillow decoder advisory](https://github.com/advisories/GHSA-5x8w-8jj5-q52h).
Replacement inputs need rebuild, audit and requalification. Passing lifecycle
checks below do not qualify refreshed packages or authorize release promotion.

The exact finalized DEB and project-signed RPM from source `e765a951` passed
**50 Ubuntu 24.04 and 54 Fedora 44 checks** in fresh KVM cloud machines. Vendor
image signatures/checksums, native package inventories, package/probe hashes
and all 62 installed application Python files were independently verified.
OS dependencies were fetched before a complete VM shutdown; installation ran
after restart with outbound routing disabled. Application-only dependency and
launcher checks preceded installation of synthetic display/audio tools.

Both machines passed CPU execution with expected CUDA rejection, actual
`NVBroadcastApp` window mapping, generated recording/decode, a clean reboot,
repeated runtime/window/media checks and package removal. Each of four
recordings contained 24 decoded video frames and nonzero generated audio.
Removal verified all **17,688 DEB / 17,689 RPM** package-owned files absent,
while user sentinels, system Python and the runtime lock remained intact.
Independent readback checked all **104 execution logs**, six distinct boot
identities, network boundaries, unchanged inputs and stopped VM processes.

Fedora's synthetic audio fixture explicitly replaces only `pipewire-pulseaudio`
with a private PulseAudio daemon after app-only dependency/launcher checks.
This does not establish default PipeWire desktop audio. The Btrfs filesystem
also exposed a test error: Linux device numbers can change across reboot.
The corrected check binds filesystem UUID/subvolume/inode and lock metadata
across boots, then keeps exact device/inode verification within the uninstall
boot. The original failure and observed device renumbering are retained.

The [permanent functional record](qualification/2026-10-26-clean-vm.json)
binds exact artifacts, images, logs, corrected fixture failures and the known
security hold. Physical devices, GPU execution, interactive desktop behavior,
CPU-variant VMs, upgrade/downgrade and power-loss acceptance remain separate.
These results do not close #60 or modify the installed Snap or v1.5.3 assets.

The [security-refresh record](qualification/2026-10-26-security-refresh.json)
adds fresh unsigned `1.5.3-8.cpu` and `1.5.3-8.cuda` packages from `0721ae7`,
with CPython 3.13.18 and repaired Python dependencies. Installed container
checks passed 106 steps without a GPU and 92 further steps on Ubuntu/RTX 3080
and Fedora/RTX 5070, including actual CUDA/CuPy execution, generated recordings,
running-launcher refusals and CPU rollback. Follow-up `5d78452` changes only
the prototype application-wheel hashes; it does not relabel these packages.
Revision 7's hold remains. Refreshed signing, protected hosted artifacts,
clean-VM/device acceptance and sustained-performance checks remain separate;
these development results do not authorize release promotion.

### CPU/CUDA switching and explicit GPU execution

`qualify_switching.py` checks a reviewed pair of local native development builds
with the same application/adapter source and package revision. Supply the
independently checked SHA-256 of each build's `packages.json` and an exact
provisioned fixture image ID:

```bash
python3 packaging/native-runtime/qualify_switching.py \
  --cpu-packages dist/native-cpu --cpu-manifest-sha256 CPU_PACKAGES_JSON_SHA256 \
  --cuda-packages dist/native-cuda --cuda-manifest-sha256 CUDA_PACKAGES_JSON_SHA256 \
  --source FULL_APPLICATION_AND_ADAPTER_COMMIT --family deb \
  --image sha256:QUALIFIED_FIXTURE_IMAGE_ID --output dist/switching-deb
```

The default prints a validated plan. Add `--execute` to run it in a disposable,
network-disabled container; use a fresh output directory for every run. The
sequence installs CPU, switches to CUDA, rolls back to CPU and removes the
package. It verifies running-launcher refusals in both directions and checks that
each refusal preserves the installed variant. Each installed state checks
its own version, runtime manifest, source files and single ONNX Runtime owner.
Synthetic GTK/media and decoded recording checks use an ordinary container user.
The test also checks preservation of settings, recordings, system Python and the
native launch-lock inode.

Without a GPU selector, CUDA must report unavailable while CPU execution works.
For a separate hardware run, add `--gpu-device GPU-UUID` with the exact NVIDIA
device UUID. That run requires actual CUDA execution with CPU fallback disabled
and a fresh CuPy kernel. It exposes the selected GPU to the fixture but no host
camera, microphone, display, home or audio socket. Repeat with `--family rpm`
and the Fedora fixture for the RPM adapter. These local development checks do
not establish RPM publisher trust, hosted provenance, physical camera/microphone
acceptance, interrupted-switch recovery or release readiness.

The [2026-10-25 CUDA switching record](qualification/2026-10-25-cuda-switching.json)
binds fresh `1.5.3-5.cpu` and `1.5.3-5.cuda` packages to application/adapter
source `42fd9e6`. All six Ubuntu/Fedora runs passed 46 checks each: no GPU,
RTX 3080 and RTX 5070, with NVIDIA driver 595.91.07. Independent readback verified
276 execution logs, 30 runtime reports and 18 decoded synthetic recordings.
Hardware runs proved CUDA provider execution and fresh CuPy output; CPU rollback
restored the CPU owner. A separate full CUDA rebuild produced byte-identical
unsigned DEB/RPM packages and all six compared metadata files using the same
recorded builders. That level-8 CUDA DEB exceeds GitHub's release-asset limit.
A separate level-10 experiment reduced it to
2,070,585,298 bytes with identical decoded control/data streams; that experimental
package was not substituted into that matrix. The level-10 packages from the
committed recipe and their qualification are recorded separately below.

The [CUDA compression qualification record](qualification/2026-10-25-cuda-compression.json)
binds fresh `1.5.3-6.cpu` and `1.5.3-6.cuda` packages to application/adapter
source `1b9f3342`. The CUDA DEB is 2,070,585,394 bytes and the RPM is
2,058,427,518 bytes: respectively 76,898,254 and 89,056,130 bytes below the
strict 2 GiB limit. Complete runtime/native manifests and source identities
were verified; compression did not require dependency pruning.

All six new-package Ubuntu/Fedora runs passed 46 checks each, covering no GPU,
RTX 3080 and RTX 5070 with driver 595.91.07. Independent readback verified
276 execution logs, 30 runtime reports and 18 synthetic recordings, including
actual CUDA/CuPy execution, both running-launcher refusal directions, CPU
rollback, preservation and cleanup. A second full fresh CUDA build reproduced
both unsigned archives and all six compared metadata files byte for byte using
the same pinned builders. These are local unsigned packages; protected hosted
CUDA finalization, signed distribution and broader platform/device acceptance
remain separate. The installed Snap and public v1.5.3 artifacts are unchanged.

### Protected native finalization

**Sign and qualify approved native candidate** (`native-cpu-finalize.yml`)
continues an existing successful main-only build of the explicitly selected variant. It is a manual development
candidate workflow, with no release, tag, registry or application-asset writes.
Before dispatch, review the producer run and select its exact attempt, source
commit, unexpired artifact ID, GitHub artifact SHA-256 and package revision:

```bash
gh workflow run native-cpu-finalize.yml --ref main \
  -f producer_run=RUN_ID -f producer_attempt=ATTEMPT \
  -f source_commit=FULL_APPLICATION_AND_ADAPTER_COMMIT \
  -f artifact_id=ARTIFACT_ID -f artifact_digest=sha256:APPROVED_ARTIFACT_SHA256 \
  -f package_revision=POSITIVE_INTEGER -f variant=cuda
```

The workflow has four trust boundaries:

1. Validation authenticates the historical builder import before loading its
   decoders, checks the live producer repository/run/attempt/workflow identity,
   and requires the source to be a real ancestor of the trusted tool checkout.
   It enforces the explicit archive digest, a closed ZIP inventory and bounded
   decoding, then compares the full runtime, application and package scripts
   with the approved source. No candidate interpreter or scriptlet runs here.
2. A separate `rpm-signing` job rechecks the validation receipt and package
   bytes, uses the existing pinned project publisher key, and regenerates the
   upgrade helper from the final signed RPM and unchanged DEB. Only its signing
   step receives private-key/passphrase secrets. This job never installs or
   executes the candidate.
3. Separate Ubuntu/Fedora jobs verify the signed bundle before installation.
   They build disposable fixtures from pinned official amd64 base manifests
   and record actual image IDs and native package inventories. Distribution
   dependency resolution happens during fixture preparation; this is not an
   independently reproduced OS repository snapshot. Candidate execution is
   network-disconnected, with no signing environment, host devices, home,
   display or audio sockets. Tests cover complete installed integrity,
   ordinary-user CPU/media/GTK behavior, decoded generated recording, removal
   and preservation. Fedora also authenticates the RPM, interrupts an actual
   fresh installation, preserves its fragments and exact package, replays it
   and verifies the repaired installation.
4. Completion requires successful receipts for those exact package bytes and
   fixture images. It checks required step outcomes, bounded log hashes, the
   actual SIGKILL observation, preservation identity and complete repair.
   Only then is the complete `native-{variant}-qualified-*` development bundle kept
   for 30 days. Its packages, public key, upgrade helper, checksums, validation,
   signing and qualification evidence remain together.

Transfers between jobs use immutable artifact IDs plus receipt/manifest hashes,
including across a failed-job rerun. Intermediate artifacts have one-day
retention and explicit `validated` or `signed-awaiting-qualification` names;
neither is an accepted installation candidate. Every stage receives the explicit
variant, checks it before signing or execution, and applies the same package
bounds again to the final signed bytes. Historical receipts without a variant
are accepted only with the CPU default.

Hosted containers use `runc` with NVIDIA visibility set to `void`, without GPU
devices. CUDA qualification requires actual CPU provider execution and a failed
CUDA provider probe, including after interrupted RPM installation and repair.
It does not establish GPU execution; the local hardware matrix remains separate
evidence. Physical devices, clean virtual machines and production promotion are
also separate. Frozen v1.5.3 release assets remain unchanged. The successful
hosted CUDA qualification below covers the exact recorded revision 7 artifacts.

Local equivalents are `validate_candidate.py`, `finalize_candidate.py`,
`fixtures/build_fixture.py` and `qualify_candidate.py`. Fixture and qualification
commands are plan-only unless passed `--execute`. Run these from a reviewed
checkout; producer files provide expected identities through `git show`, while
the checked-out tools provide all executed probes. `--evidence-output` exports
only public receipts and referenced logs, excluding private working snapshots
and probe inputs.

The [2026-10-25 hosted CUDA record](qualification/2026-10-25-hosted-cuda.json)
records producer run `40240561476` and protected finalizer `40241470218` for
`e765a951` / `1.5.3-7.cuda`. Source/payload validation, existing project-key
signing, all **14 DEB, 19 RPM and 24 interrupted-install recovery checks**, and
final retention passed. Disconnected Ubuntu 24.04/Fedora 44 containers proved
CPU provider execution and expected CUDA rejection; the repaired RPM runtime
passed the same provider check after an observed mid-unpack SIGKILL and replay.

Independent download/readback verified the retained archive digest, all **75
files**, **57 execution logs** and the interruption transcript. A fresh RPM
database containing only the pinned project public key verified the actual
signed RPM. Both generated recordings decoded 24 video frames and nonzero
synthetic audio. A separate fresh offline build from the same source and
recorded builders reproduced both unsigned archives and all six compared
metadata files **byte for byte**. The final DEB is 2,070,585,242 bytes; the signed
RPM is 2,058,428,352 bytes. Both remain strictly below 2 GiB. The qualified
[development artifact](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/40241470218/artifacts/12617194457)
expires on **24 November 2026**. GPU execution on these exact signed bytes,
physical devices, clean-VM acceptance and release promotion remain separate.

The [2026-10-23 finalization and reproducibility record](qualification/2026-10-23-cpu-finalization.json)
records the successful protected workflow from `a007338`, qualifying the
`71903d7` CPU candidate with the existing project RPM identity. All 14 DEB,
19 RPM and 23 interrupted-install recovery checks passed. Independent download
verification checked the final archive, all 74 files, 56 execution logs and the
interruption transcript; a fresh public-key-only RPM database verified the
signature. The record also binds a separate full offline rebuild to the same
builder images: both unsigned packages and all six compared metadata files were
byte-identical to the hosted build. It retains the two corrected finalization
failures and their regression evidence. The qualified artifact expires on
2026-11-22; physical devices, CUDA, clean-VM acceptance and production promotion
remain separate.

The [2026-10-23 hosted CPU record](qualification/2026-10-23-hosted-cpu.json)
binds the `1.5.3-4.cpu` DEB/RPM candidates to application and adapter commit
`71903d7`. Authenticated hosted assembly, independent source/payload verification
and RPM signing passed. Installed smoke passed 14 Ubuntu DEB and 19 Fedora RPM
steps, including ordinary-user CPU execution, GTK and decoded video/audio
recording. Signed RPM first-install interruption, bootstrap recovery, exact
replay and full validation passed 23 steps. These are isolated-container results;
CUDA, physical devices, clean-VM acceptance and native release promotion remain
separate. The record retains all three corrected workflow/harness failures.

The [2026-10-11 authenticated RPM recovery record](qualification/2026-10-11-rpm-recovery.json)
binds the signed CPU revision 3 candidate (application `0295b190`, adapter
`16d49c5`) and signed revision 2 baseline. All six recovery/preflight scenarios
passed 80 steps, including real fresh-install and upgrade interruptions with no
build staging mounted. The signed `%pre` bootstrap remained available while
RPM held `/usr/bin` files under temporary names; explicit preservation and exact
saved-package replay completed full runtime validation. Independent review adds
38 lifecycle stages covering launch/process refusals, upgrade, reinstall,
rollback and erase, plus actual scriptlet/dependency checks. The record retains
the original failure evidence and exact limits; it does not promote public
packages or qualify CUDA, physical devices or power-loss recovery.

The [2026-10-11 CPU record](qualification/2026-10-11-current-main-cpu.json)
binds freshly built `2.cpu` and `3.cpu` packages to application and adapter
commit `0295b190`, including PR #146 and PR #151. Both disconnected fixtures
passed actual mid-unpack interruption and strict repair. Across ten installed
runtime probes, all 62 application Python files matched that commit.

| Fixture | Lifecycle stages | Generated recording decoded successfully |
| --- | ---: | --- |
| Ubuntu 24.04 | 39 | 24 video frames; 49,152 audio samples (`x264enc` / `voaacenc`) |
| Fedora 44 | 40 | 24 video frames; 51,200 audio samples (`openh264enc` / `fdkaacenc`) |

These unsigned local candidates use one newly assembled offline CPU runtime.
The signed recovery candidate above reuses that verified application runtime
with its separately qualified adapter. The hosted CPU record above adds the
newer exact-source assembly; CUDA switching and desktop/device acceptance
remain separate gates. The [2026-10-09 record](qualification/2026-10-09-cpu.json)
retains the earlier qualification of application commit `c2dd082`.

The [published-package migration record](qualification/2026-10-11-legacy-migration.json)
adds 18 checks per family from the exact public `v1.5.3` DEB/RPM to those
`2.cpu` candidates. `qualify_legacy.py` installs the pinned legacy artifact with
its normal online hook, then disconnects the disposable container before
attempting migration. It checks refusal while an existing legacy interpreter
is running, unchanged files after refusal, offline upgrade, full runtime
integrity, removal of the package-owned legacy environment, preserved settings,
recordings and administrator files, and ordinary-user inference/media/GTK.
Both installed application trees match all 62 Python files from `0295b190`.
This closes the tested CPU migration path; it does not qualify a concurrent
legacy launcher race, CUDA migration, physical devices or clean virtual machines.

```bash
python3 packaging/native-runtime/qualify_legacy.py \
  --family deb --image sha256:YOUR_PREPROVISIONED_FIXTURE_DIGEST \
  --baseline /verified-release/nvbroadcast_1.5.3-1_all.deb \
  --baseline-sha256 VERIFIED_PUBLIC_RELEASE_SHA256 --baseline-version 1.5.3-1 \
  --packages dist/native-cpu --output dist/native-cpu-legacy-migration-deb
```

`qualify.py` consumes two versioned package sets and verifies their hashes before
launching disconnected disposable fixtures. It performs install, upgrade,
rollback, reinstall, deliberate corruption and exact repair, actual native
package-manager interruption and repair, removal/purge, and preservation of
user configuration/recording sentinels and system Python. Ordinary-user probes
exercise CPU inference, application imports, synthetic video/audio EOS and a
mapped GTK window. No physical device, host display/audio socket or user home
is mounted. This is container lifecycle evidence, not clean-VM desktop/device
acceptance or real GPU qualification.
The recording probe encodes generated moving video and a generated tone with
the application's selected H.264/AAC encoders, then decodes the MP4 and checks
frame count, audio length and signal level. Lock contention includes the actual
installed Python launcher. Mid-unpack interruption must observe a native staging
file larger than 1 MiB before killing the manager; a marker-only interruption
must not be reported as unpack recovery.

```bash
python3 packaging/native-runtime/qualify.py \
  --packages dist/native-cpu --upgrade dist/native-cpu-next \
  --family deb --image YOUR_PREPROVISIONED_TEST_IMAGE \
  --output dist/native-cpu-qualification-deb
PYTHONPATH=src python3 -m pytest -q \
  tests/test_native_runtime_production.py tests/test_native_package_upgrade.py
```

The test image must supply declared OS dependencies plus Xvfb, D-Bus, a private
PulseAudio daemon and ordinary UID 1000. Its actual immutable ID is saved in
the results. Production defaults remain gated until the delivery and supported
platform acceptance above are complete; this adapter alone does not close #60.

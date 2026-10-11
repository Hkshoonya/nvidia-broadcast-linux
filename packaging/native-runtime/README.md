# Complete native Linux runtime adapter

This adapter builds the current application's complete CPU or CUDA Python
environment before creating a native package. It uses the reviewed private
CPython 3.13.16 supplier archive and dependency hashes from the runtime
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
same friendly tag. These images are not yet authenticated for CI consumption.
Their reproducibility scope is fixed input consumption and repeated
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
The `candidate` status and absent attestation deliberately prevent a release
consumer from calling it authenticated. A local hash or the unsigned BuildKit
metadata inside the archive proves no publisher identity or original build.

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
   commit. This step remains pending until the actual workflow succeeds.
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
builder generation. Until steps 4–5 pass, CI release promotion remains blocked.

The containerd requirement follows Docker's
[image-store documentation](https://docs.docker.com/engine/storage/containerd/).
Attestation verification follows the GitHub CLI's
[identity and source policy](https://cli.github.com/manual/gh_attestation_verify).

After the authenticated lock lands, **Build authenticated native CPU candidate**
is the manual main-only consumer. It fetches the pinned archive, cryptographically
verifies the historical import before loading it, and builds both native formats
with application and adapter fixed to that run's `GITHUB_SHA`. It receives only
contents/attestations read permissions and retains complete unsigned DEB/RPM
artifacts plus bounded source, runtime and builder provenance. It does not publish
a release, access host devices, install native packages or receive signing keys.
Signing and device/lifecycle acceptance remain separate gates.

Host-side CPU workflow scripts use `/usr/bin/python3 -I -S`. The production
boundary tests need `packaging`; they zip-import only the packaging 26.0 universal
wheel already pinned in `runtime-prototype/pylock.build.toml`, after checking its
SHA-256 and size. No pip resolver, global install or ambient Python package is
used. The artifact includes the isolated interpreter/dependency verification
record in `host-tools.json`.

The CPU job requires **12 GiB actually free** on its workspace and temporary
filesystems before fetching builders. The October 11 complete local CPU build
used 5,905,301,504 disk bytes, with a separately measured 559,071,232-byte cache
(some hardlinks overlap), up to 2.49 GB of builder images, and up to 1.25 GB for
transport plus a private verification copy. The preflight includes headroom and
records actual free space; it fails without pruning unrelated runner tools or
images. GitHub documents 14 GB storage for its standard public Ubuntu runners,
so actual hosted capacity still needs confirmation in the first run. CUDA is
not offered by this workflow and its larger footprint is not qualified here.

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
does not remove. The qualification harness preserves only recognized fragments
whose transaction timestamp and bytes match the selected authenticated payload,
then replays the package and verifies the exact installed file set. Its bounded
recovery helper currently needs retained build staging/content manifests and is
qualification tooling; a shipped authenticated end-user recovery interface is
still a promotion gate. Unknown fragments are rejected, never broadly deleted.

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
Signing the final candidate bytes, authenticated CI builder delivery, shipped
recovery, CUDA switching and desktop/device acceptance remain
separate gates. The [2026-10-09 record](qualification/2026-10-09-cpu.json)
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

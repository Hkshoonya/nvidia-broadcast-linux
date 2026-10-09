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
same friendly tag. These images are not yet published in an authenticated
registry. Their reproducibility scope is fixed input consumption and repeated
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

`qualify.py` consumes two versioned package sets and verifies their hashes before
launching disconnected disposable fixtures. It performs install, upgrade,
rollback, reinstall, deliberate corruption and exact repair, actual native
package-manager interruption and repair, removal/purge, and preservation of
user configuration/recording sentinels and system Python. Ordinary-user probes
exercise CPU inference, application imports, synthetic video/audio EOS and a
mapped GTK window. No physical device, host display/audio socket or user home
is mounted. This is container lifecycle evidence, not clean-VM desktop/device
acceptance or real GPU qualification.

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

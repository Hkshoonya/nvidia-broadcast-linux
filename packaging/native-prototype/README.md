# Complete CPU native-package lifecycle experiment

This is the next bounded investigation for #53, #60, and the package alternatives
in draft PR #77. It wraps the complete, hash-verified
[private runtime](../runtime-prototype/README.md) in architecture-specific DEB
and RPM packages. It does not replace the release packaging recipes.

Both candidates are implemented and tested before choosing a production model:

| Candidate | Packages | Ownership |
| --- | --- | --- |
| Self-contained | `nvbroadcast-cpu` | Application, resources, private interpreter, and all CPU dependencies |
| Split | `nvbroadcast` + `nvbroadcast-runtime-cpu` | Application files from the installed wheel's verified `RECORD`; the runtime package owns the remaining payload |

Directories may have shared ownership. Regular files and symlinks cannot have
two owners. The split application requires the exact runtime package version;
its launcher checks that same pair. The runtime packages provide a versioned
`nvbroadcast-runtime` capability and conflict through that capability. The
self-contained package also provides versioned `nvbroadcast` and replaces the
older legacy package. The illustrative versions `1.5.2-900~prototype1/2` (DEB)
and `1.5.2-900.prototype1/2` (RPM) are local experiment versions, not releases.
They must not be published to a user-facing package repository.

## Reproduced migration defect and the chosen test prefix

The existing RPM's final-removal script recursively deletes
`/opt/nvbroadcast`. A replacement package using that directory can be installed
successfully and then deleted by the old package's `%postun` during an
`Obsoletes` transaction. A real installed `2fcd999` development package reproduced
this: DNF exited successfully, but `rpm -V nvbroadcast-cpu` reported a missing
file owned by the replacement package. This was not inferred from a mocked
removal script.

The prototypes install under `/usr/lib/nvbroadcast/runtime`, outside that
legacy cleanup boundary. Their launchers use the absolute private Python path
with `-I -B`. The application recognizes this private native prefix so its CPU
runtime guidance directs users to the package manager instead of the source
installer. Generated console scripts inside the original runtime are preserved
byte-for-byte; the adapters provide the launchers used by the desktop and user
service.

The split layout upgrades the existing `nvbroadcast` package name. Its old
generated virtual environment is not owned by dpkg/RPM, so a bounded migration
hook removes that old private environment after both new packages are configured.
The hook detects legacy ownership before unpacking. It does not remove files
from user home directories. Earlier public packages with the historical broad
`pkill` bug still require the existing legacy upgrade helper; those versions
are not qualified by this experiment.

## Transaction and integrity checks

`build.py` requires an externally supplied assembly-manifest digest, rehashes
the runtime, verifies application `RECORD` entries, and builds in containers
with networking disabled. The payload bytes are the same in both layouts.
Adapters add only native metadata, launchers, integration files, version
identity files, and lifecycle hooks. Install hooks never run pip, uv, or a
network fetcher.

Each prepare hook writes a transaction marker. The launcher refuses to run
while that marker exists or if the installed native versions are inconsistent.
The application's final configuration hook clears the marker after checking
its package set. Interrupted transactions are repaired by replaying the exact
DEB set with dpkg, or the RPM set with `rpm --replacepkgs`, followed by native
dependency and runtime checks. This is package-manager recovery; it is not an
atomic filesystem generation switch or a guarantee that the previous runtime
remains launchable during an interrupted native upgrade.

`run_lifecycle.py` verifies all twelve package digests before starting any test.
For each clean or legacy-installed image, both layouts exercise:

- Local, network-disabled APT/DNF installation, upgrade, exact-version rollback,
  and reinstall.
- Split application/runtime version mismatch rejection by the package solver.
- SIGKILL of the actual package-manager process group after its prepare marker,
  refusal to launch, exact-package repair, and dependency rechecking.
- Complete installed-runtime hash/mode/root-ownership comparison and separate
  native package-file ownership checks after each successful transaction.
- Ordinary-user imports, dependency closure, CPU model execution, video/audio
  EOS, a mapped GTK test window, and the installed launcher's `--help` path.
  Every installed application Python file must match this checkout.
- Removal, Debian purge, explicit removal of a retained split runtime, absence
  of orphaned runtime files, unchanged system Python bytes, and preserved
  configuration/recording sentinel files in a user home.

The test containers expose no host camera, microphone, display, audio socket,
user home, or Docker socket. Synthetic Xvfb/D-Bus/PulseAudio fixtures come from
the private-runtime investigation. All test containers are removed on completion
or failure. Results retain logs and numeric evidence, not recordings.

The two package revisions have different adapter version metadata but consume
the same Python payload. They test native package transactions, dependency
matching, and ownership, not an upgrade between different Python/ML library
versions. SIGKILL tests one recorded interruption boundary; it is not exhaustive
power-loss or disk-full testing.

## Reproduce in local containers

First assemble the current private runtime using the commands in
[its README](../runtime-prototype/README.md), including its pinned application
revision. Build the clean runtime test images there too. No host package install
is needed; run these commands as an ordinary user with Docker access:

```bash
NVB_WORK="$PWD/dist/native-lifecycle"
NVB_PAYLOAD="$PWD/dist/native-runtime-prototype/payload"
mkdir -p "$NVB_WORK"

docker build -f packaging/native-prototype/Dockerfile.builder.rpm \
  -t nvb-native-package-builder:local packaging/native-prototype
NVB_RPM_BUILDER=$(docker image inspect --format '{{.Id}}' nvb-native-package-builder:local)
NVB_DEB_BUILDER=$(docker image inspect --format '{{.Id}}' nvb-private-runtime-test:ubuntu24-20261005)
NVB_MANIFEST=$(sha256sum "$NVB_PAYLOAD/manifest.json" | cut -d ' ' -f 1)

python3 packaging/native-prototype/build.py \
  --runtime "$NVB_PAYLOAD/runtime" --manifest-sha256 "$NVB_MANIFEST" \
  --deb-image "$NVB_DEB_BUILDER" --rpm-image "$NVB_RPM_BUILDER" \
  --output "$NVB_WORK/packages"
```

Create a matrix of inspected local image IDs. A clean matrix can be generated
from the runtime test images:

```bash
python3 - <<'PY'
import json, subprocess
from pathlib import Path
cells = []
for name, family in (("ubuntu24", "deb"), ("fedora44", "rpm")):
    image = subprocess.check_output([
        "docker", "image", "inspect", "--format", "{{.Id}}",
        f"nvb-private-runtime-test:{name}-20261005",
    ], text=True).strip()
    cells.append({"name": name, "family": family, "image": image, "legacy": False})
Path("dist/native-lifecycle/matrix.json").write_text(json.dumps(cells, indent=2) + "\n")
PY

python3 packaging/native-prototype/run_lifecycle.py \
  --packages "$NVB_WORK/packages" --runtime "$NVB_PAYLOAD/runtime" \
  --matrix "$NVB_WORK/matrix.json" --output "$NVB_WORK/checks"
```

For legacy migration, add cells with `legacy: true` and an inspected image ID
containing a real installed legacy `nvbroadcast` package and its generated
virtual environment. The recorded run used the Ubuntu 24.04 and Fedora 44
development packages from source `2fcd99989d211bcd8c3e65e8bd98a634cb2c8c7f`;
their exact installed-image identities and native inventories are part of the
evidence. These local images are not public distribution fixtures. Testing a
different baseline requires recording that baseline separately.

Native host prerequisites and the synthetic desktop fixture must be present
before the network-disabled transaction. The first legacy attempts correctly
refused installation because the older images lacked PortAudio. Prepare derived
fixtures using `../runtime-prototype/Dockerfile.runtime.deb` and
`Dockerfile.legacy-fixture.rpm` with the legacy image as `BASE_IMAGE`, then record
both the original and derived identities. The latter swaps PipeWire's Pulse
replacement for the private PulseAudio fixture inside the derived container;
the application package and generated Python environment stay installed.
This native dependency fetch is a separate, network-enabled preparation step;
it does not rebuild or replace the legacy Python environment. Fedora container
defaults omit documentation files, so the harness explicitly clears `tsflags`
for the complete package-file comparison.

Use fresh output directories for independent builds and test runs. The input
runtime is mounted read-only by the wrappers. Native repositories are not
snapshotted; compare actual builder identities, tool inventories, package bytes,
and canonical payloads rather than assuming a later image is identical.

The fast boundary tests require neither Docker nor package-manager privileges:

```bash
python -m pytest -q tests/test_native_runtime_prototype.py tests/test_private_runtime_inputs.py
```

## Recorded result — 5 October 2026

All eight final lifecycle cases passed against the exact unsigned artifacts:

| System and starting state | Self-contained | Split application/runtime |
| --- | --- | --- |
| Ubuntu 24.04, clean | Pass | Pass |
| Ubuntu 24.04, installed legacy package | Pass | Pass |
| Fedora 44, clean | Pass | Pass |
| Fedora 44, installed legacy package | Pass | Pass |

Every case passed install, upgrade, rollback, reinstall, SIGKILL interruption,
blocked launch, exact-file repair, dependency checking, runtime execution,
removal, and user-file preservation. All four split cases rejected a mismatched
application version. Both Debian layouts passed purge; the renamed Debian
candidate also survived purging the old package while the new runtime remained
installed. All eight ended without an orphaned runtime directory.

Two separate wrapping runs used different recorded builder images. All twelve
DEB/RPM archives and all twelve canonical native content manifests matched
byte-for-byte. The same unmodified 16,556-entry runtime manifest underlies both
layouts. This is deterministic wrapping of a pinned upstream payload with the
recorded tools, not independent source rebuilding of Python or its dependencies.
The private runtime also passed its seven-distribution feasibility matrix.

| Format | Self-contained download | Split application | Split runtime |
| --- | ---: | ---: | ---: |
| DEB | 371.697 MiB | 0.380 MiB | 371.371 MiB |
| RPM | 371.332 MiB | 0.405 MiB | 370.896 MiB |

These are unpruned prototype sizes. The small application portion alone does
not settle the variant-switch or package-maintenance tradeoff. The local suite
passed **940 tests, with 38 skips and 981 subtests**; release smoke and wheel
asset checks passed. The checked-in result records revisions, builder/fixture
identities, archive hashes, input hashes, each lifecycle step, and evidence
digests. Full logs remain in the local test output directories.

## Scope still required for production

The [recorded results](results-2026-10-05.json) describe only these unsigned CPU
prototypes. They do not choose between the two package models or qualify the
current public installers. Remaining work includes real CUDA payloads and
CPU/CUDA switching, virtual-capability selection for split variants, Zypper and
other declared target versions, native dependency/ELF validation, interpreter
and dependency-content upgrades, and running-application/service restart policy.

RPM automatic dependency generation and build-root postprocessing are disabled
here to preserve the already assembled payload. Explicit native requirements
cover the tested import/media fixture, not all recording codecs and virtual
devices. Passing this experiment does not satisfy Fedora packaging policy or
replace RPATH/shared-library analysis. The full interpreter still contains pip,
debug/development material, and unused Tk/Tcl components. Pruning must happen in
the assembler with a new verified manifest, not silently inside the adapter.

Final signatures, repository trust, native supply-chain/SBOM and redistribution
review (#100), production sizes, real desktop/device/model/GPU acceptance, and
signed-artifact tests remain release gates. The RPM `LicenseRef` deliberately
marks the prototype's combined license inventory as unreviewed. No public
release readiness or legal clearance is claimed.

Package relationship semantics are described in the primary
[Debian policy](https://www.debian.org/doc/debian-policy/ch-relationships.html)
and [RPM spec documentation](https://rpm.org/docs/6.0.x/manual/spec.html).
The recorded package-manager results, including the legacy deletion failure,
are the evidence for the behavior tested here.

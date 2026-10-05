# Private native CPU runtime feasibility

This is the executable Stage 0 investigation for issues #53 and #60 and the
design in PR #77. It builds a complete private CPython 3.13 CPU environment,
including PyGObject, pycairo, the application, and the managed faster-whisper
dependency closure. It is **not a native package or release installer**.

The subsequent [native-package lifecycle experiment](../native-prototype/README.md)
consumes this payload in complete self-contained and split DEB/RPM prototypes.
Its current application pin also recognizes the new native prefix used to avoid
legacy removal-script collisions. The results below retain the identities of
the earlier private-runtime-only run; the package experiment records its own
updated input and artifact identities.

The application change found by this investigation is small: pass an empty argv
list to `Gst.init` throughout startup, media pipelines, and isolated media
probes. The private PyGObject 3.48.2 binding rejects `None` with the newer
GStreamer 1.28 typelib. The original `VideoPipeline()` constructor fails before
capturing video on Fedora 44. The changed constructor works with the older and
newer GStreamer versions in the recorded matrix.

No runtime supplier or DEB/RPM package-shape decision is finalized here. Current
native installation scripts, Snap, Flatpak, source Python support, and the
OpenAI Whisper compatibility path retain their existing behavior.

## What is pinned and tested

- `inputs.json` pins the complete upstream Python archive, uv archive, binding
  source archives, and application Git revision. SHA-256 is checked before
  extraction or use. Python metadata and all upstream license records are kept
  under `share/nvbroadcast-runtime-provenance` in the payload.
- `pylock.build.toml` fixes the wheel-building Python tools.
  `pylock.linux-x86_64-cp313-cpu.toml` fixes all 58 application/binding packages.
  The interpreter archive also includes pip. CPU ONNX Runtime is the only ORT
  owner. No GPU dependency substitution is involved in this target.
- The three local wheels are built without network access on Ubuntu 22.04.
  The application comes from `git archive`, not an arbitrary working directory;
  this prevents local file permissions and uncommitted files changing the wheel.
  The source epoch comes from the pinned application revision, and the scripts
  use a fixed umask.
- `assemble.py` rechecks the inputs and uses a Docker container with
  `--network=none`, `--no-index`, `--no-deps`, and `--require-hashes`.
  It writes a manifest of file bytes, permissions, and relative symlink targets.
  Existing output directories are rejected. Failed work stays available for
  diagnosis; it is never activated in an installed application.
- `run_matrix.py` mounts the payload read-only in clean native images and runs
  Python with `-I -B`. Deliberately incorrect `PYTHONHOME` and `PYTHONPATH` values
  must not leak system or user packages into the private environment. The exact
  distribution inventory, source hashes, imports, resources, package ownership
  guard, pinned CPU execution probe, video/audio test pipelines, and a mapped GTK
  window are checked. The manifest must match before and after the probes.
- The fixture uses UID 1000, a private D-Bus session, Xvfb, and a private
  PulseAudio null sink. It exposes no host camera, microphone, audio socket, or
  display. Ubuntu 26.04's PortAudio initializes its PulseAudio backend at import
  time, so a real test server is needed even for an import check.

The upstream [Python distribution documentation](https://gregoryszorc.com/docs/python-build-standalone/main/distributions.html)
describes the full archive's `PYTHON.json` and license metadata. This prototype
uses that full archive and retains those records; the smaller install-only
archive omits them. License inventory is technical evidence for issue #100,
not a redistribution clearance.

## Reproduce

Run on Linux x86_64 as an ordinary user with Python 3.12+, zstd, Git, and access
to Docker. Input downloads and native image builds require networking. Wheel
builds, payload assembly, and application probes disable it. Commands below
run from the repository root and use a fresh ignored work directory.
The pinned application commit must be available locally; fetch that revision
first if working in a shallow checkout.

```bash
NVB_WORK="$PWD/dist/native-runtime-prototype"
NVB_BUILDER=nvb-private-runtime-builder:local

python3 packaging/runtime-prototype/prepare.py fetch --directory "$NVB_WORK"
python3 packaging/runtime-prototype/prepare.py fetch-lock \
  --lock packaging/runtime-prototype/pylock.build.toml \
  --directory "$NVB_WORK/bootstrap"

docker build -f packaging/runtime-prototype/Dockerfile.bindings \
  -t "$NVB_BUILDER" packaging/runtime-prototype
NVB_BUILDER_ID=$(docker image inspect --format '{{.Id}}' "$NVB_BUILDER")

python3 packaging/runtime-prototype/build_local.py \
  --directory "$NVB_WORK" --image "$NVB_BUILDER_ID"
python3 packaging/runtime-prototype/prepare.py fetch-lock \
  --lock packaging/runtime-prototype/pylock.linux-x86_64-cp313-cpu.toml \
  --local-wheels "$NVB_WORK/wheels" --directory "$NVB_WORK/application"

python3 packaging/runtime-prototype/assemble.py \
  --directory "$NVB_WORK" --output "$NVB_WORK/payload" --image "$NVB_BUILDER_ID"
python3 packaging/runtime-prototype/run_matrix.py \
  --directory "$NVB_WORK/checks" --runtime "$NVB_WORK/payload/runtime" --build-images
```

The all-target run intentionally returns failure while any recorded cell is
unqualified, including Rocky 9. `matrix-results.json` and per-cell logs identify
the failing layer. Use `--cells ubuntu22 ubuntu24 ubuntu26 debian12 debian13
fedora43 fedora44` to rerun just the qualified target set. Omit `--build-images`
to reuse the already built images; each actual image ID is recorded in results.
`matrix.json` pins base digests; native package inventories are recorded too.

Native repositories are not snapshotted yet. A later builder may produce
different wheel bytes even with the same base image. The checked-in hashes must
then fail, rather than silently accepting a different artifact. Repeating two
builds in the **same recorded builder image** is the scope of the current
reproducibility result, not a claim about future repository states.

For an intentional lock refresh after changing the application revision or
dependency inputs, use the pinned resolver and review its output before replacing
the checked-in lock:

```bash
python3 packaging/runtime-prototype/resolve.py \
  --directory "$NVB_WORK" --output "$NVB_WORK/pylock.candidate.toml"
```

`resolve.py` consumes the built application's canonical dependency metadata and
the managed faster-whisper version from the pinned source tree. It only accepts
binary wheels and targets CPython 3.13 / x86_64 / manylinux 2.28. This maintainer
operation uses the network; it is not called by assembly or end-user installs.
Use a new work directory for repeat builds. Compare wheel hashes and the two
assembly `manifest.json` files; mtimes are deliberately outside that content and
permissions comparison. Neither signed-package reproducibility nor byte-identical
compressed archives has been established by this test.

## Recorded result and remaining gates

See [the recorded results](results-2026-10-05.json) for exact revisions, input
and lock digests, built wheel hashes, builder and runtime image identities,
payload manifest identity, and the distribution matrix.

The 2026-10-05 run passed Ubuntu 22.04/24.04/26.04, Debian 12/13, and
Fedora 43/44. All seven verified the same 62 application Python files and 59
installed distributions, including the interpreter's pip. Two separate wheel
builds matched, and two separate assemblies produced the same content and
permissions manifest. The unpruned payload contains 1.289 GiB of regular-file
data; a reference zstd-compressed tar is 371.8 MiB. Those are prototype payload
sizes, not DEB/RPM download sizes.

Local validation passed 931 tests with 38 skips and 971 subtests. The Python
dependency audit reported no known vulnerabilities in 58 public distributions;
the local application was skipped because it is not published on PyPI.

The investigation exposed and corrected missing GI 1.x compatibility libraries
and GLES libraries in the minimal test-image dependency lists. The Xvfb runner
also needs Docker's `--init` so its readiness signal reaches the wrapper.
OpenSSL certificate directories load lazily; the probe checks for available
hashed CA files as well as eagerly loaded certificates. It does not claim a
remote TLS handshake was tested offline.

Rocky Linux 9's default repositories do not provide PortAudio under the requested
native dependency contract. That cell fails at image construction. Adding EPEL,
bundling PortAudio, or changing the supported target matrix requires a separate
decision and verification. A Rocky result would not qualify all EL derivatives.

Remaining work for the broader issues includes:

- Complete the package-model comparison after the linked CPU DEB/RPM lifecycle
  experiment, including CPU/CUDA switching, Zypper, and production policy.
  The earlier tiny-package experiment did not exercise this full closure.
- Define the production launcher and immutable installation/activation contract.
  This prototype explicitly invokes private Python with `-I`; it does not change
  the generated console script or the existing installed launcher.
- Qualify CUDA separately and finish private interpreter update ownership,
  stripping/pruning, size targets, native toolchain snapshots, and supplier review.
  The current full Python archive retains development/debug material and unused
  Tk/Tcl components; it is not an optimized package payload.
- Produce and verify final package signatures, provenance, SBOMs, and signed
  artifact tests. No release assets, signing credentials, or Store channels are
  changed by these scripts.
- Perform real desktop, camera, microphone, speech-model, effect-model, Wayland,
  virtual-device, and client acceptance on the final native packages. Synthetic
  container probes do not replace those checks or the separate #91/#95/#112 gates.
- Resolve the recorded licensing review in #100. Neither these hash checks nor
  the Python vulnerability scan assess all bundled native libraries or establish
  public-launch readiness.

The ordinary test suite exercises the input verification boundaries without
network access or Docker:

```bash
python -m pytest -q tests/test_private_runtime_inputs.py
```

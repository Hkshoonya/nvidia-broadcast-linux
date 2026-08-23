# NV Broadcast v1.5.0

NV Broadcast v1.5.0 makes runtime selection deterministic and verifies that
the selected inference provider can execute real model work before the app
describes it as ready. It also strengthens package and Snap validation, adds
release provenance, and improves recovery around microphones, missing cameras,
profiles, and failed starts.

## Deterministic CPU and CUDA runtime ownership

- Every managed environment now has exactly one ONNX Runtime distribution
  owner: `onnxruntime` for CPU or `onnxruntime-gpu` for CUDA, never both.
- The source installer accepts an explicit runtime choice:

  ```bash
  ./install.sh --runtime auto
  ./install.sh --runtime cpu
  ./install.sh --runtime cuda
  ```

- `auto` selects CUDA only on Linux x86_64 when NVIDIA hardware is detected.
  Other systems select CPU.
- An automatic CUDA setup failure discards the partial environment and creates
  a clean CPU environment. An explicit CUDA request fails clearly instead of
  silently changing the requested variant.
- Runtime ownership changes require NV Broadcast to be stopped. The installer
  recreates its managed virtual environment rather than replacing an imported
  inference runtime inside the running process.
- Debian and RPM installs select their runtime during installation or upgrade.
  macOS remains CPU/CoreML, amd64 Snap owns the CUDA runtime, and arm64 Snap
  owns the CPU runtime.
- Package-managed and Snap environments remain immutable from the app. Their
  runtimes are repaired or upgraded through their package path, not by an
  in-app `pip` operation.
- Source users can request the complete meeting stack without allowing
  `faster-whisper` to introduce a second ONNX Runtime owner:

  ```bash
  ./install.sh --runtime auto --with-meeting
  ```

## Execution-proven acceleration

- Runtime readiness now launches an isolated child interpreter and executes a
  checksum-pinned ONNX model instead of trusting provider enumeration.
- The probe validates the exact result shape and values and inspects ONNX
  Runtime profiling evidence to confirm that CPU, CUDA, or TensorRT executed
  the graph.
- CUDA and TensorRT probes disable CPU fallback. A provider that loads but
  silently places work on CPU is rejected.
- Native provider errors, bounded standard output, and bounded standard error
  are retained for installer and UI recovery messages.
- Newly installed GPU support is not activated in a process that has already
  imported another runtime. The app requires a restart before the mode can be
  selected.
- The same probe contract is used by source installation, platform readiness,
  the runtime CLI, CI, and release smoke checks.

Source users can run the installed-environment probe directly:

```bash
.venv/bin/python -m nvbroadcast.runtime --variant cpu
.venv/bin/python -m nvbroadcast.runtime --variant cuda
```

## Package and Snap integrity

- Strict Snap builds reject a runtime `pip` installer, missing dependencies,
  incompatible versions, duplicate core packages, mixed ONNX Runtime owners,
  invalid import provenance, and forbidden execution providers before upload.
- Native package permission normalization preserves the executable runtime
  repair helper used by package installation and recovery paths.
- Snap startup uses the Core 24 runtime Python and the GNOME content interface
  without shadow copies of shared GTK or GStreamer libraries.
- Candidate promotion validates exact amd64 and arm64 revisions and restores
  the previous candidate revisions if a partial release fails.
- Stable promotion operates on the complete tested candidate set and verifies
  both downloaded Store revisions against the repository, signer workflow,
  exact release tag, tag commit, and GitHub-hosted runner before promotion.
- Tag-built DEB, RPM, PKG, and Snap artifacts receive Sigstore-backed GitHub
  artifact attestations. Native packages and Snap files small enough to attach
  to GitHub Releases also receive deterministic SHA-256 manifests.
- Manual stable Snap publication is accepted only when the workflow ref,
  checked-out tag, source commit, `GITHUB_SHA`, and Snap version agree.
- Version tags create draft GitHub release artifacts for inspection. They do
  not automatically publish a GitHub release or move the Snap Store stable
  channel.

Verification commands are documented in
[Verifying Release Artifacts](RELEASE_VERIFICATION.md). Provenance binds an
artifact digest to this repository, source commit, and workflow; it does not
prove that the source is vulnerability-free or that two independent builds
are byte-for-byte reproducible.

## Device and startup recovery

- The microphone selector now restores the saved capture device after
  enumeration. If that device is unavailable, the visible fallback is saved
  instead of leaving configuration and UI state different.
- Changing the microphone while audio is active rebuilds the capture pipeline
  so the new device takes effect immediately.
- GUI and headless startup stop cleanly when no usable physical camera is
  available. The app no longer constructs a capture pipeline for an invalid
  saved device or `/dev/video0` fallback.
- Headless startup exits with a focused error before creating a virtual camera
  when no physical input can be resolved.
- Saving a profile now offers an explicit "Start broadcast when this profile
  is selected" option. It defaults off, and older profiles never auto-start
  unless the option is saved explicitly.
- Launch-time Auto Start remains an application setting. A profile saved while
  stopped cannot suppress the user's application-level startup preference.
- Button, launch, profile, legacy tray, and StatusNotifierItem start paths
  update their visible state only after the pipeline actually starts. Busy or
  missing cameras no longer leave a phantom "Stop Broadcast" state.

## Platform and dependency maintenance

- MediaPipe 1.0.0 restores Auto Frame, Eye Contact, Face Relighting, and
  face-aware Video Enhancement on Linux arm64.
- OpenCV now requires at least 4.8.1.78, Protobuf at least 6.33.5, Pillow at
  least 12.3.0, and development testing pytest at least 9.0.3.
- Python 3.11 and newer remain supported for Linux CPU use. The release matrix
  covers Linux Python 3.11 through 3.14 and Apple Silicon Python 3.11 through
  3.13.
- TensorRT modes remain unavailable on Python 3.14 because current ONNX Runtime
  TensorRT provider wheels target TensorRT ABI 10 while Python 3.14 TensorRT
  packages provide ABI 11. CUDA modes remain the supported GPU path there.
- Intel macOS remains excluded because no secure current MediaPipe wheel is
  available for that architecture. Supported macOS output continues through
  OBS Virtual Camera on Apple Silicon with macOS 13 or newer.

## Community

- Cédric Prezelin ([`@Tenshock`](https://github.com/Tenshock)) contributed the
  dependency security update, deterministic runtime-ownership foundation, and
  no-camera startup recovery through PRs #51, #62, and #70.
- Cenkay Coban ([`@pastor0711`](https://github.com/pastor0711)) contributed the
  explicit profile auto-start option and failure-safe start-state handling in
  PR #52.
- The microphone persistence fix follows the report tracked in Issue #66.
- Runtime ownership and release reproducibility continue through Issues #53
  and #60. This release completes bounded stages of those issues without
  closing their later runtime activation, hermetic build, SBOM, or native
  signing work.

## Release validation

The exact v1.5.0 release tree must pass the pre-tag gates below. Tag-only
artifact, attestation, and Store gates must then pass before publication:

- Complete non-hardware unit and packaging suite.
- Release smoke suite, wheel construction, and required asset inspection.
- Python 3.11 and Python 3.14 clean-environment compatibility checks.
- Debian and RPM package builds and metadata inspection.
- Apple Silicon package build and Python 3.11-3.13 CI matrix.
- Strict amd64 and arm64 Snap construction and runtime-closure validation.
- Ruff, actionlint, ShellCheck, Bash syntax, Bandit, dependency closure, and
  known-vulnerability audit.
- Real CUDA and TensorRT execution probes on supported NVIDIA hardware.
- Checksum verification for every attached release asset and provenance
  verification for every tag-built release artifact.

The release remains draft, and Snap remains outside stable, until all
applicable gates pass on the exact release commit.

Maintainers should record each gate and publication step in the
[Release Checklist](RELEASE_CHECKLIST.md).

## Remaining supply-chain work

v1.5.0 establishes verifiable provenance but does not complete Issue #60.
Native installers still resolve parts of their Python environments online.
Target-specific dependency locks or audited wheelhouses, clean-VM lifecycle
tests, complete SBOMs, independent rebuild comparison, RPM signing, and macOS
Developer ID signing and notarization remain open release-engineering work.

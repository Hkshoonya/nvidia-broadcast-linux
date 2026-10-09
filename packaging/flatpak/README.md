# Flatpak development package

This directory prepares an upstream Flatpak build without making it part of a
stable release. The current manifest is an `x86_64`, CPU ONNX Runtime baseline
for sandbox and integration testing. It is an upstream development build. The intended first distribution is a
reviewed CPU/x86_64 bundle from this project; desktop acceptance and authenticated
publication still gate that release. It is not a Flathub submission. See
[DISTRIBUTION.md](DISTRIBUTION.md) for the exact scope and current policy review.

## What is included

- GNOME Platform and SDK 50 with Python 3.13.
- GTK 4, libadwaita, GStreamer, V4L2, PipeWire, and PulseAudio utilities from
  the maintained runtime.
- Hash-pinned Python 3.13 wheels generated from `requirements.txt`.
- CPU ONNX Runtime, MediaPipe, RNNoise, and faster-whisper meeting support.
- A runtime alias for the GNOME Platform's versioned `libsndfile`, required by
  Python SoundFile when `ldconfig` is unavailable in the sandbox.
- Flatpak-specific guards for immutable dependencies, host `systemctl`, host
  Firefox profiles, and host v4l2loopback module reloads.

CUDA, TensorRT, and Linux `aarch64` are not represented by this baseline. Do not
advertise GPU acceleration for this package until the CUDA driver path, NVIDIA
wheel redistribution terms, runtime size, and real inference execution have all
been validated in the sandbox.

## Sandbox permissions

The manifest grants network access for release checks, checksum-verified
app-owned model downloads, and pinned, SHA-256-verified faster-whisper model
retrieval, plus Wayland with X11 fallback, PulseAudio compatibility, and access
to the StatusNotifier watcher. The finished-runtime smoke check verifies that
the faster-whisper trust manifest is packaged. Installed ordinary-user tests on
2 October 2026 passed first-use faster-whisper download, corrupt-cache rejection, and CPU
inference for the tiny model; other models and each final release artifact still
need their applicable checks (see issue #95). Recording follows the
desktop's XDG Videos directory, so the manifest grants only
`--filesystem=xdg-videos:create` to keep those files visible after the app exits,
including when the user customizes that directory. The manifest does not grant the rest of the
host home, a session-bus wildcard, the system bus, or permission to run host
commands.

`--device=all` is currently required because the app reads physical
`/dev/video*` devices and writes to a host-created v4l2loopback device. Flatpak
cannot expose those nodes with a filesystem permission and does not provide a
portal for virtual-camera output. This broad device permission must remain a
visible security tradeoff during review.

## Desktop integration

The tray requires a desktop StatusNotifierWatcher; GNOME users generally need
an extension providing one. Global shortcuts require the host's
GlobalShortcuts portal. When the desktop lacks that portal, the sandbox reports
shortcuts unavailable rather than editing host desktop settings. These are
conditional desktop capabilities; a missing portal does not prevent camera,
audio, or recording use. Test tray recovery after a watcher restart and the
shortcut grant/activation flow on desktops that provide those features.

## Host prerequisite

Flatpak cannot install or load kernel modules. Before hardware testing, create
the virtual camera on the host. On Ubuntu, the existing project setup is:

```bash
sudo apt install v4l2loopback-dkms v4l-utils
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVBroadcast" exclusive_caps=1 max_buffers=4
```

The app can use an existing loopback node but cannot reset it from inside the
sandbox. Firefox profile changes must also be made from the host.

## Generate dependencies

Use the GNOME 50 builder image so dependency resolution uses the same Python
3.13 ABI as the runtime:

```bash
docker run --rm --privileged \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/packaging/flatpak:/output" \
  -w /workspace \
  ghcr.io/flathub-infra/flatpak-github-actions:gnome-50 \
  flatpak-pip-generator \
  --requirements-file packaging/flatpak/requirements.txt \
  --runtime org.gnome.Sdk//50 \
  --yaml --wheel-arches=x86_64 \
  --prefer-wheels=numpy,Pillow,opencv-contrib-python,mediapipe,protobuf,pyrnnoise,av,onnx,psutil,scipy,onnxruntime,ctranslate2,tokenizers,hf-xet,cffi,ml-dtypes,charset-normalizer,markupsafe,matplotlib,contourpy,fonttools,kiwisolver,pyyaml,wrapt \
  --output /output/python3-flatpak-requirements
```

Generated sources must retain immutable HTTPS URLs and SHA-256 hashes. A
dependency refresh requires a package build, import smoke test, `pip check`, and
dependency audit before review. Do not generate dependency modules with
`--cleanup scripts`: Flatpak applies module cleanup globally at the end of the
build, so `/bin` cleanup would also remove the app's `nvbroadcast` launcher.

## Build and smoke test

With Flatpak and flatpak-builder installed locally:

```bash
flatpak-builder --user --install-deps-from=flathub --force-clean \
  flatpak-build packaging/flatpak/com.nvbroadcast.NVBroadcast.yml
flatpak-builder --run flatpak-build \
  packaging/flatpak/com.nvbroadcast.NVBroadcast.yml \
  python3 -c 'import cv2, mediapipe, onnxruntime, pyrnnoise, nvbroadcast'
```

The repository also validates this manifest in the maintained GNOME 50 Docker
builder, which avoids changing the host package set. The scoped Flatpak workflow
runs when packaging inputs change and can be dispatched manually after other
source changes or before a release.

Successful workflow runs retain a `flatpak-development-cpu-x86_64` artifact for
14 days. It contains an unsigned `.flatpak` bundle, `SHA256SUMS`, and
`bundle-provenance.json` recording the checked-out Git revision, pinned builder
image, application OSTree commit, runtime reference, and build-input hashes.
Before retaining the bundle, the workflow imports it into a fresh temporary
repository, checks the commit and all OSTree objects, and compares packaged app
source, resources, project license, notices, and sandbox metadata with the build.
It also compares source and build inputs with the recorded Git revision, rejecting
staged or unstaged edits and untracked shipping inputs.
This check does not install the app or access physical devices.

Download the artifact from the completed **Flatpak Development Build** run and
verify its files before using it for an agreed hardware test:

```bash
sha256sum --check SHA256SUMS
```

The bundle does not include the GNOME Platform runtime. Development artifacts
are test inputs; uploading one to GitHub Actions does not qualify it as a stable
release or submit it to Flathub. The public-distribution gates below still apply.

Before review, validate the manifest and finished artifacts with the current
Flathub linter:

```bash
flatpak-builder-lint manifest packaging/flatpak/com.nvbroadcast.NVBroadcast.yml
flatpak-builder-lint appstream \
  flatpak-build/files/share/metainfo/com.nvbroadcast.NVBroadcast.metainfo.xml
flatpak-builder-lint builddir flatpak-build
```

The package includes Flatpak-specific desktop metadata and real application
controls screenshots under `docs/screenshots/`. They show an isolated idle
application, with no personal camera or microphone input. Metadata pins
the exact screenshot source commit on GitHub over HTTPS; confirm those URLs
resolve to the reviewed PNG bytes before publishing a bundle. The full build-directory linter must pass without suppressing
`metainfo-missing-screenshots`.

## Public-distribution gates

The first CPU/x86_64 bundle requires final physical camera, virtual-camera,
Wayland desktop, physical/processed/virtual microphone, effects, recording,
shortcuts/tray, reconnect, and hardware-soak acceptance. Repeat applicable
model-trust checks against the exact candidate. The previous installed
tiny-model first-use download, corrupt-cache rejection, and CPU inference
checks are evidence for their original artifact, not every later build.

The permanent Flatpak application ID is `com.nvbroadcast.NVBroadcast`, grounded
in the project's controlled `nvbroadcast.com` domain. Native packages retain
`com.doczeus.NVBroadcast`; no native preferences move. See the explicit,
non-destructive development-ID migration instructions in DISTRIBUTION.md.
Flatpak metadata uses the NV Broadcast project name, CPU-specific features,
an independent camera icon without the NVIDIA eye motif, and a clear statement
that this community project is not affiliated with NVIDIA. This is a technical
identity/trademark presentation review, not a new legal opinion. The accepted
license metadata work from #100/#136 remains complete.

The roughly 1.2 GB application payload has been measured by component. Meeting
support remains included in the first CPU package: its principal engine is
about 140.5 MB, while OpenCV, MediaPipe, audio processing, and numerical libraries
account for most of the remaining payload. The separate GNOME runtime is shared
with other apps. DISTRIBUTION.md records exact bytes and the distinction between
installed and compressed size.

Flathub's current source-build and AI-assisted-manifest restrictions mean this
upstream wheel-based manifest is ineligible for a Flathub submission as written.
A direct upstream Flatpak bundle is a separate distribution route and does not
claim Flathub approval. A future Flathub attempt requires human-maintained,
policy-compliant packaging and an honest disclosure of AI-assisted application
material; an agent must not submit or write Flathub review replies.

CUDA and TensorRT, plus aarch64, are separate future variants. A GPU variant
requires real driver/model execution and NVIDIA wheel redistribution review;
aarch64 requires its own dependency graph and hardware tests. Neither is
advertised by the initial x86_64 CPU package.

Installed development-package physical-camera recording and initial
virtual-camera output passed on 5 October on X11. Final desktop/device gates
remain tracked in issue #95. A container build verifies package closure and
sandbox execution; it does not establish the missing physical desktop checks.

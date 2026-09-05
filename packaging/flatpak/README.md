# Flatpak development package

This directory prepares an upstream Flatpak build without making it part of a
stable release. The current manifest is an `x86_64`, CPU ONNX Runtime baseline
for sandbox and integration testing. It is not ready for public distribution or
submission to Flathub.

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
app-owned model downloads, and faster-whisper model retrieval, plus Wayland
with X11 fallback, PulseAudio compatibility, and access to the StatusNotifier
watcher. The faster-whisper model path is not hash-pinned by this packaging work
and remains a public-distribution security gate. The manifest does not grant
host filesystem access, a session-bus wildcard, the system bus, or permission
to run host commands.

`--device=all` is currently required because the app reads physical
`/dev/video*` devices and writes to a host-created v4l2loopback device. Flatpak
cannot expose those nodes with a filesystem permission and does not provide a
portal for virtual-camera output. This broad device permission must remain a
visible security tradeoff during review.

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
  flatpak-build packaging/flatpak/com.doczeus.NVBroadcast.yml
flatpak-builder --run flatpak-build \
  packaging/flatpak/com.doczeus.NVBroadcast.yml \
  python3 -c 'import cv2, mediapipe, onnxruntime, pyrnnoise, nvbroadcast'
```

The repository also validates this manifest in the maintained GNOME 50 Docker
builder, which avoids changing the host package set. The scoped Flatpak workflow
runs when packaging inputs change and can be dispatched manually after other
source changes or before a release.

Before review, validate the manifest and finished artifacts with the current
Flathub linter:

```bash
flatpak-builder-lint manifest packaging/flatpak/com.doczeus.NVBroadcast.yml
flatpak-builder-lint appstream \
  flatpak-build/files/share/metainfo/com.doczeus.NVBroadcast.metainfo.xml
flatpak-builder-lint builddir flatpak-build
```

The development package currently reports `metainfo-missing-screenshots` for
the build directory. Add real, current application screenshots at stable HTTPS
URLs only after desktop and effect-path testing; do not use promotional artwork
as an application screenshot.

## Public-distribution blockers

Before any Flatpak release, all of these gates must be closed:

1. Test physical camera capture, v4l2loopback output, background effects, mode
   switching, recording, microphone processing, virtual microphone output,
   model downloads, and global shortcuts on a real Wayland desktop.
2. Decide the permanent application ID. `com.doczeus.NVBroadcast` requires
   control of the corresponding domain for store verification; a GitHub-derived
   ID would require a coordinated metadata, D-Bus, and migration change.
3. Resolve the canonical attribution and license metadata work before claiming
   that the AppStream `GPL-3.0-or-later` declaration matches the distributed
   license text.
4. Capture representative application screenshots after hardware testing and
   close the current `metainfo-missing-screenshots` linter error.
5. Review the name, icon, screenshots, and description for NVIDIA trademark and
   affiliation clarity.
6. Pin and verify the faster-whisper model revision or document and approve its
   external model-download trust policy.
7. Validate CUDA and TensorRT from the NVIDIA driver through a real model
   execution, then decide whether their binary wheels may be redistributed.
8. Build and test a separate `aarch64` dependency set; do not infer support from
   the `x86_64` build.
9. Review the roughly 1.2 GB application payload measured for this baseline and
   decide whether meeting transcription should become an optional extension.
10. Re-check current Flathub submission and automated-content policies. This
   upstream development manifest is not a Flathub submission.

Passing a container build proves dependency closure and sandbox startup only.
It does not prove camera, microphone, GPU, desktop-portal, or host-driver
behavior.

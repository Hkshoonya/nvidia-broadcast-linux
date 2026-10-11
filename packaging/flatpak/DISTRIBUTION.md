# CPU Flatpak distribution decision — 9 October 2026

The first upstream Flatpak target is **Linux x86_64 with CPU processing**. Keep
meeting transcription, microphone processing, and recording in the package.
CUDA/TensorRT and aarch64 need separate dependency graphs and qualifications.
A completed build is still a candidate until issue #95's remaining desktop and
physical-device acceptance and its publication checks pass.

## Identity and presentation

Use `com.nvbroadcast.NVBroadcast`: the project controls `nvbroadcast.com`, its
canonical HTTPS website. This follows Flatpak's [reverse-DNS application ID
convention](https://docs.flatpak.org/en/latest/conventions.html#application-ids).
Domain control does not itself claim Flathub verification or admission.

Only the Flatpak manifest, exported desktop entry, AppStream metadata, icon,
application/D-Bus identity, and corresponding resource lookup use this ID.
Native packages retain `com.doczeus.NVBroadcast` and their existing settings.
The Flatpak title is **NV Broadcast**, matching the application's established
project name. The CPU description explicitly excludes NVIDIA processing modes
and states that the project is independent of NVIDIA Corporation. The scoped
Flatpak icon retains the project's camera illustration and removes the eye
motif. Original native artwork is unchanged.

The screenshots are actual GTK application windows captured on an isolated
Linux display, using real UI source and CSS, with device discovery and periodic
hardware work disabled. They show controls before broadcasting, no generated
camera image, and no personal desktop, speech, or camera input. The application
was captured at 990×690 and 630×690 logical pixels after GTK decorations; image
pixels are unedited. These illustrate the interface, not physical-device or
AI-quality acceptance. Source capture details are in
[`docs/screenshots/README.md`](../../docs/screenshots/README.md).

### Migrating the unpublished development ID

The earlier ID was never a supported public release. Flatpak treats the new ID
as a separate app with separate portal permissions and private directories.
There is no automatic read, move, overwrite, or removal of old preferences.
A tester who wants existing settings can quit both Flatpak instances, inspect
and back up the old directory, then copy only their app settings once:

```bash
(
set -eu
old_config="$HOME/.var/app/com.doczeus.NVBroadcast/config/nvbroadcast"
new_config="$HOME/.var/app/com.nvbroadcast.NVBroadcast/config/nvbroadcast"
test -d "$old_config"
test ! -e "$new_config"
install -d -m 700 "$(dirname "$new_config")"
cp -a --no-clobber "$old_config" "$new_config"
)
```

The subshell stops on a failed check and refuses an existing destination.
The old app data remains available. Grant any new portal permissions normally;
do not copy permission databases or runtime binaries. Existing recordings in
Videos stay in place. Models can download again into the new private cache.
Native/source/Snap settings are outside this migration.

## Measured payload and meeting support

A read-only inventory of the retained GNOME 50 v1.5.3 development build measured
**1,148,117,176 logical file bytes** under `/app`, including split debug files.
This excludes the separate GNOME runtime and downloaded models. Its compressed
development bundle was about 243 MB; installed footprint and network transfer
are different quantities. Major installed components are:

| Component | Logical bytes | Used by |
| --- | ---: | --- |
| OpenCV and its bundled libraries | 212,556,948 | Camera and image processing |
| SciPy and its bundled libraries | 131,679,704 | Audio/numerical processing |
| MediaPipe | 89,498,848 | Face and camera effects |
| PyAV and its bundled libraries | 121,505,873 | Audio/media processing |
| CTranslate2 and its bundled libraries | 140,528,921 | Local meeting transcription |
| ONNX Runtime | 53,723,633 | CPU model inference |
| NumPy and its bundled libraries | 68,346,161 | Shared numerical processing |
| SymPy | 68,562,186 | ONNX Runtime dependency |

Keep meeting support in this first scope. Making it optional would remove a
minority of the payload and introduce a new extension interface, dependency
resolution, upgrade behavior, and acceptance matrix. A later size reduction
must preserve working features and license notices and measure its actual
benefit. No unsupported cleanup of native libraries or dependency metadata is
part of this decision. Every final candidate still needs its own measured
compressed size and exact digest.

## Upstream bundle route

Flatpak supports [single-file bundles](https://docs.flatpak.org/en/latest/single-file-bundles.html).
Use this project's reviewed release assets for the initial distribution. Each
published bundle must have an exact source commit, manifest/builder pins,
SHA-256 checksum, and independently verifiable build provenance. Review and
qualify that exact artifact before attaching it to a public release. The CI
exporter remains fail-closed and labels its output an **unsigned development
bundle**; this change does not silently promote it to a supported release.

The GNOME Platform runtime is a separate prerequisite; a bundle is not a
complete offline installation. An upstream bundle also does not establish an
automatic update repository. A user must install the next reviewed bundle to
update until a separately authenticated repository is provided. Flatpak
recommends repository hosting for regular update delivery. Native host
v4l2loopback installation remains a documented prerequisite.

## Flathub policy review

Checked the primary [Flathub requirements](https://docs.flathub.org/docs/for-app-authors/requirements)
and [submission instructions](https://docs.flathub.org/docs/for-app-authors/submission)
on **9 October 2026**. The current upstream manifest uses prebuilt ONNX Runtime,
OpenCV, MediaPipe, and other Python wheels. Flathub requires source-available
applications and bundled dependencies to build from source unless an applicable
exception is granted. None has been obtained for this package.

The current policy also forbids AI-generated or AI-assisted Flathub manifests,
requires disclosure of generated application/documentation material and its
approximate extent, and bars agents from submitting or writing Flathub
submission/review interactions. Our upstream development manifest is
AI-assisted. It must not be relabeled as human-written or submitted as-is.
A future Flathub route needs a human-maintained policy-compliant manifest,
truthful disclosure, and human submission/review work. Upstream code and
packaging improvements can continue independently.

The maintainer's completed project licensing review and #100/#136 remain
accepted. This source-build/policy route decision does not reopen that review
or claim a Flathub exemption. Before any future submission, re-check policies
and hardware evidence because both may change.

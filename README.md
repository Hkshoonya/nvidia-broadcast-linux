<p align="center">
  <img src="data/icons/com.doczeus.NVBroadcast.svg" width="120" alt="NV Broadcast">
</p>

<h1 align="center">NV Broadcast</h1>

<p align="center">
  <strong>by DocZeus | AI Powered</strong>
</p>

<p align="center">
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/stargazers"><img src="https://img.shields.io/github/stars/Hkshoonya/nvidia-broadcast-linux?style=for-the-badge&color=76b900&labelColor=1a1a1a" alt="Stars"></a>
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-76b900?style=for-the-badge&labelColor=1a1a1a" alt="License"></a>
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/issues"><img src="https://img.shields.io/github/issues/Hkshoonya/nvidia-broadcast-linux?style=for-the-badge&color=76b900&labelColor=1a1a1a" alt="Issues"></a>
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/workflows/build-packages.yml"><img src="https://img.shields.io/github/actions/workflow/status/Hkshoonya/nvidia-broadcast-linux/build-packages.yml?style=for-the-badge&color=76b900&labelColor=1a1a1a&label=Build" alt="Build"></a>
  <a href="https://nvbroadcast.com"><img src="https://img.shields.io/badge/Website-nvbroadcast.com-76b900?style=for-the-badge&labelColor=1a1a1a" alt="Website"></a>
  <a href="https://github.com/sponsors/Hkshoonya?metadata_campaign=nvbroadcast_readme"><img src="https://img.shields.io/badge/Sponsor-DocZeus-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor"></a>
</p>

<p align="center">
  <em>Open-source, unofficial NVIDIA Broadcast alternative for Linux and macOS. AI-powered virtual camera and audio tools.</em>
</p>

---

> [!IMPORTANT]
> **Canonical upstream:** [Hkshoonya/nvidia-broadcast-linux](https://github.com/Hkshoonya/nvidia-broadcast-linux), created and maintained by [DocZeus](https://github.com/Hkshoonya). Forks and mirrors are welcome under the project license, but downstream versions should identify themselves clearly, link to this repository, and preserve the required author attribution. See [Canonical Project and Attribution](docs/CANONICAL_PROJECT.md).

## Why I Built This

I left Windows. Millions of Linux users left Windows. But we all missed one thing — **NVIDIA Broadcast**.

That one app that made every video call look professional. Background blur that actually worked. Virtual backgrounds that didn't look like a PowerPoint slide. Noise cancellation that silenced your mechanical keyboard.

On Linux? Nothing. You had to cobble together 5 different tools, fight with v4l2loopback configs, and still get janky edges. **That's over now.**

I built this because I believe Linux users deserve the same broadcast-quality experience. Not a half-baked wrapper. Not a "good enough" hack. A real, proper implementation that uses your NVIDIA GPU to do what NVIDIA Broadcast does on Windows — **but open source, and in some ways, better.**

> *"Not saying this is perfect. But I believe it will be."*
>
> This is fast. This is optimized. And the quality already rivals Windows Broadcast. With the community behind it, we'll surpass it.
>
> **— DocZeus**

---

## What's new?

See [CHANGELOG.md](./CHANGELOG.md) for latest updates!

---

## What It Does

<table>
<tr>
<td width="50%">

### Camera Effects
- **Background Blur** — AI-powered, person stays crystal sharp
- **Background Replace** — Any image via native file picker
- **Green Screen** — Solid green for OBS chroma key
- **Auto Frame** — Face tracking with Center Face and Stable Background modes
- **Global Effect Hotkeys** — Rebind effect toggles without bringing the app forward
- **Video Enhancement** — Skin smooth, enhance, sharpen, denoise, vignette
- **Eye Contact Correction** — Natural and Gaze Lock modes redirect gaze to camera
- **Face Relighting** — Fill light guided by the scene
- **Recording to MP4** — NVENC hardware encode (x264 fallback)
- **User Profiles** — 5 built-in (Meeting, Streaming, etc.) + custom save/load
- **Performance Overlay** — Real-time FPS, GPU usage, VRAM, temperature
- **Multi-Model** — RVM (person), IS-Net (objects), BiRefNet (best edges)
- **Mirror** — Horizontal flip toggle

</td>
<td width="50%">

### Audio Effects
- **Mic Noise Removal** — Kills keyboard, fan, environment noise
- **Speaker Denoise** — Clean up incoming audio
- **Voice Effects** — Bass boost, treble, warmth, compression, noise gate, gain
- **6 Voice Presets** — Natural, Radio, Podcast, Deep Voice, Bright, Studio
- **Microphone Selection** — Full PipeWire/PulseAudio device enumeration
- **Speaker Detection** — All output devices via PipeWire
- **Audio Level Monitor** — Real-time VU meter with peak hold
- **Mic Test** — Record 30s / 45s / 60s and play back to test your setup
- **AI Meeting Transcription** — Local Whisper speech-to-text (GPU-accelerated)
- **AI Meeting Summarizer** — Action items, questions, key points (fully local)
- **Meeting Mode** — Video+audio recording with live transcription and AI summary

### System Integration
- **Virtual Camera** — Works in Chrome, Firefox, Zoom, Discord, OBS
- **Auto-Start** — Launches on login, runs in background
- **Setup Wizard** — Auto-detects system, configures optimally
- **Multi-GPU** — Select which GPU runs AI effects
- **Multi-Distro** — Ubuntu, Fedora, Arch, openSUSE, and more
- **Resolution/FPS** — 360p-4K, 15-60fps, auto-detected from camera

</td>
</tr>
</table>

---

## Processing Modes

9 modes from maximum speed to CPU fallback:

| Mode | Inference | Compositing | 1080p Speed | CPU | GPU | Best For |
|------|-----------|-------------|-------------|-----|-----|----------|
| **Killer** | 360p + fused CUDA | Fused kernel (0.1ms) | **20ms / 48fps** | 24% | 41% | Maximum speed |
| **Zeus** | 480p optimized | CuPy GPU | 30ms / 33fps | 22% | 39% | Speed + quality balance |
| **DocZeus** | 720p full quality | Fused kernel (0.1ms) | 44ms / 23fps | 22% | 46% | **Best quality/speed** |
| **CUDA Max** | 720p | CuPy GPU | 45ms / 22fps | 22% | 46% | Maximum quality |
| **CUDA Balanced** | 720p, skip 2 | CuPy GPU | 29ms / 34fps | 24% | 39% | Daily use |
| **CUDA Perf** | 720p, skip 2 | CuPy GPU | 30ms / 34fps | 23% | 39% | Light GPU load |
| **CPU Quality** | 720p | OpenCV SIMD | 66ms / 15fps | 17% | 27% | No CuPy fallback |
| **CPU Light** | 720p, skip 2 | OpenCV SIMD | 30ms / 34fps | 23% | 20% | Save GPU for games |
| **CPU Low End** | 720p, skip 3 | OpenCV SIMD | 27ms / 37fps | 21% | 20% | Older hardware |

> **Edge Refine** toggle available for Killer and Zeus modes — adds ~27ms but recovers 89.9% of max quality edges.
>
> Switch modes anytime from the **Mode** dropdown. No restart needed.
>
> CUDA modes require the CUDA mode runtime: CuPy for compositing plus ONNX Runtime with `CUDAExecutionProvider` for model inference. Source, `.deb`, `.rpm`, and amd64 Snap installs handle this automatically on NVIDIA systems. The arm64 Snap build stays CPU-safe because ONNX Runtime GPU wheels are not published for Linux arm64 yet.

---

## Architecture

```
                              NV Broadcast Pipeline
                        ─────────────────────────────────

  ┌───────────┐     ┌─────────────────────────────────────────┐     ┌──────────────┐
  │  Webcam   ├────▶│          GStreamer Pipeline             ├────▶│ Virtual Cam  │
  │(360p-4K)  │     │                                         │     │ /dev/video10 │
  └───────────┘     │ JPEG Decode ─▶ Color Convert ─▶ appsink │     └──────┬───────┘
                    └────────────────────┬────────────────────┘            │
                                         │                         ┌───────▼───────┐
                         ┌───────────────▼────────────────┐        │ Chrome / Zoom │
                         │      Async Effects Thread      │        │ Firefox / OBS │
                         │     (never blocks capture)     │        │ Discord/Meet  │
                         │                                │        └───────────────┘
                         │  ┌──────────────────────────┐  │
                         │  │     AI Segmentation      │  │
                         │  │                          │  │
                         │  │  Pre-downsample to 720p  │  │
                         │  │  (or 480/360 for Zeus/   │  │
                         │  │   Killer modes)          │  │
                         │  │                          │  │
                         │  │  ┌────┐ ┌─────┐ ┌────┐   │  │
                         │  │  │RVM │ │ISNet│ │BiR │   │  │
                         │  │  └──┬─┘ └──┬──┘ └─┬──┘   │  │
                         │  │     └──────┼──────┘      │  │
                         │  │            ▼             │  │
                         │  │       Alpha Refine       │  │
                         │  │     (sigmoid+dilate)     │  │
                         │  └────────────┬─────────────┘  │
                         │               │                │
                         │  ┌────────────▼─────────────┐  │
                         │  │   Edge Refiner (opt.)    │  │
                         │  │   720p 2nd pass RVM      │  │
                         │  │   (Zeus/Killer only)     │  │
                         │  └────────────┬─────────────┘  │
                         │               │                │
                         │  ┌────────────▼─────────────┐  │
                         │  │       Compositing        │  │
                         │  │                          │  │
                         │  │   ┌────────┐ ┌───────┐   │  │
                         │  │   │ Fused  │ │ CuPy  │   │  │
                         │  │   │ CUDA   │ │ CUDA  │   │  │
                         │  │   │ 0.1ms  │ │ 15ms  │   │  │
                         │  │   └────────┘ └───────┘   │  │
                         │  └────────────┬─────────────┘  │
                         │               │                │
                         │  ┌────────────▼─────────────┐  │
                         │  │    Video Enhancement     │  │
                         │  │   5 effects + presets    │  │
                         │  │    GPU batch (CuPy)      │  │
                         │  └────────────┬─────────────┘  │
                         │               ▼                │
                         │     Mirror flip (optional)     │
                         └───────────────┬────────────────┘
                                         │
                             ┌───────────▼────────────┐
                             │ Preview (GTK4 Texture) │
                             │ Pause / Hide / Resize  │
                             └────────────────────────┘

  ┌───────────┐      ┌─────────────────────────────────┐      ┌──────────────┐
  │    Mic    ├─────▶│    DeepFilterNet3 AI Denoise    ├─────▶│ Virtual Mic  │
  │           │      │   RNNoise fallback at 48kHz     │      │  (PipeWire)  │
  └───────────┘      └─────────────────────────────────┘      └──────────────┘
```

### Fused CUDA Kernel (DocZeus/Killer)

A custom CUDA kernel that performs alpha blend + enhance + sharpen + vignette in **one GPU pass**:

```cuda
// Single kernel: fg*alpha + bg*(1-alpha) + enhance + vignette
// 0.1ms at 1080p — 150x faster than CuPy's multi-kernel approach
extern "C" __global__ void fused_composite(
    fg, bg, alpha, face_mask, vignette, output,
    total_pixels, enhance_i, vignette_i, brightness, contrast, warmth
);
```

### Edge Refinement Network

When Edge Refine is toggled ON (Zeus/Killer modes):

1. **Fast pass**: RVM at 360p/480p → coarse alpha (18-21ms)
2. **Refine pass**: RVM ResNet50 at 720p → quality alpha (30ms, every 2nd frame)
3. **Blend**: On refine frames use quality alpha; on skip frames 80% quality + 20% coarse for tracking
4. **Result**: 89.9% quality recovery with minimal cost

---

## AI Models

| Model | Segments | Speed (RTX 5060) | VRAM | License | Auto-Download |
|-------|----------|-----------------|------|---------|---------------|
| **RVM** (default) | Person only | ~29ms (720p) | 660 MB | GPL-3.0 | Yes |
| **IS-Net** | Any object | ~55ms | 1.8 GB | Apache 2.0 | Yes |
| **BiRefNet** | Best edges | ~187ms | 6+ GB | MIT | Yes |

### Quality Presets (RVM only)

| Preset | Backbone | Downsample | Best For |
|--------|----------|-----------|----------|
| Performance | MobileNetV3 | 0.25 | Video calls |
| Balanced | MobileNetV3 | 0.5 | Daily use |
| Quality | ResNet50 | 0.375 | Presentations |
| Ultra | ResNet50 | 0.5 | Recording |

---

## Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1060 | RTX 3060 or newer |
| **VRAM** | 2 GB | 4 GB+ |
| **CPU** | 4 cores | 8+ cores (if using CPU compositing) |
| **Webcam** | Any USB camera | 720p+ with MJPEG or raw V4L2 modes |
| **Mic** | Any audio input | — |

### Software
- **Linux** with NVIDIA driver 525+ (Pop!_OS, Ubuntu, Fedora, Arch, openSUSE, etc.)
- **Python** 3.11+ (the Linux source installer accepts any CPython 3.11+ interpreter that can import GTK4/Libadwaita/GStreamer bindings; Python 3.12/3.13 currently give the broadest premium-feature compatibility, see [Python runtime notice](#optional-tensorrt-for-zeuskiller-modes))
- **PipeWire** (virtual microphone)
- **PulseAudio utilities** (`pactl`) for speaker-monitor routing and device resolution
- **GStreamer** 1.20+ with plugins-base, plugins-good, plugins-bad
- **GTK4** and **Libadwaita**
- **v4l2loopback** kernel module
- **DKMS** and **kernel headers** (to build v4l2loopback)

---

## Installation

### Linux — One Command Install

```bash
git clone https://github.com/Hkshoonya/nvidia-broadcast-linux.git
cd nvidia-broadcast-linux
./install.sh
```

The Linux source installer uses an already-installed compatible interpreter in
this order: CPython 3.13, 3.12, then 3.11. It creates only the repository's
`.venv`; it does not replace the distro's system Python or add a package
repository. Compatibility includes `venv`/`ensurepip` support and access to the
distro's GTK4, Libadwaita, and GStreamer Python bindings. To select a specific
compatible interpreter:

```bash
./install.sh --python /usr/bin/python3.12
```

If no compatible interpreter with `venv` support is installed, the installer
stops before changing the system and prints guidance for the detected distro.

### macOS — One Command Install

```bash
git clone https://github.com/Hkshoonya/nvidia-broadcast-linux.git
cd nvidia-broadcast-linux
./install_macos.sh
```

Requires an Apple Silicon Mac with macOS 13+, Homebrew, Python 3.11-3.13, and OBS Studio for virtual-camera output. The installer provisions GStreamer and GTK4 and can install OBS. After installing OBS, open it once, start and stop **Virtual Camera**, then close OBS so its camera backend is registered for NV Broadcast.
CPU modes use CoreML acceleration. Intel macOS is not included in v1.5.2 because no secure current MediaPipe wheel is available for that architecture. GPU modes (Killer/Zeus/DocZeus/CUDA) are Linux-only and require an NVIDIA GPU.

### Linux — Snap Package

```bash
sudo snap install nvbroadcast
```

Snap users typically receive background refreshes from `snapd`. When the app sees a newer stable release, the in-app update button opens the Snap Store listing so the user can move directly into the store-managed upgrade path.

The amd64 Snap build includes the CUDA mode runtime for NVIDIA systems. The arm64 Snap build stays CPU-safe because the required ONNX Runtime GPU wheels are not available for Linux arm64 yet. Background effects, virtual camera and microphone output, recording, and local meeting tools remain available. If CUDA modes are still unavailable on amd64 Snap, use the source installer, `.deb`, or `.rpm` release package as the fallback.

Native `.deb` or `.rpm` users upgrading from v1.4.0 or older must use the
`nvbroadcast-native-upgrade` asset shipped with v1.5.2 and later. Verify the
helper and package against `SHA256SUMS.packages`, then follow
[Verifying Release Artifacts](docs/RELEASE_VERIFICATION.md). The old package's
pre-removal script runs before a newer package can replace it, so a direct
package-manager upgrade is not safe on those versions.

Packaged releases are intended to include the local meeting transcription runtime. Source installs from this repo can still use the in-app runtime installer flow for optional components.

### Linux Installer Details

The installer:
1. **Detects your distro** and package manager
2. **Checks all requirements** (Python, PipeWire, GPU, DKMS, kernel headers)
3. **Installs missing packages** with the correct names for your distro
4. **Installs NVIDIA CUDA mode runtime packages** when an NVIDIA GPU is detected
5. **Asks about compositing** — CPU, GStreamer GL, or CuPy CUDA
6. **Sets up virtual camera**, launcher scripts, desktop entry, systemd service
7. **Verifies GPU acceleration** and writes initial config
8. **Lets optional runtimes install later** inside the app without blocking the rest of the UI

### Update Behavior

- **Git checkout / manual Linux packages** — the app checks GitHub Releases and opens the matching release download page when a newer stable build is available
- **macOS package installs** — the app prefers the latest `.pkg` release asset when one is published; the package updates an installation whose Homebrew, Python, GStreamer, GTK, and OBS prerequisites are already configured by `install_macos.sh`
- **Snap installs** — the app opens the Snap Store listing; stable refreshes are normally handled by `snapd`

### Verify Release Downloads

New release workflows publish SHA-256 manifests and Sigstore-backed GitHub
provenance for DEB, RPM, PKG, and attached Snap artifacts. Verify both the
checksum and the expected signer workflow before installing a manually
downloaded package. See [Verifying Release Artifacts](docs/RELEASE_VERIFICATION.md)
for the exact Linux, macOS, and GitHub CLI commands and the remaining
reproducibility limits.

### Optional: TensorRT (for Zeus/Killer modes)

```bash
.venv/bin/pip install tensorrt-cu12 onnx
```

TensorRT Python wheels are currently published for Python `3.8` through `3.13`
on Linux `x86_64`. If you are on Python `3.14+`, use `DocZeus` or the CUDA
modes instead.

### Supported Distros

| Distro | Package Manager | Status |
|--------|----------------|--------|
| Ubuntu, Debian, Pop!_OS, Mint | apt | Full auto-install |
| Fedora, RHEL, CentOS, Rocky | dnf/yum | Full auto-install |
| Arch, Manjaro, EndeavourOS | pacman | Full auto-install |
| openSUSE | zypper | Full auto-install |
| Gentoo, Void, NixOS | portage/xbps/nix | Manual instructions shown |

<details>
<summary>Click to expand manual install steps</summary>

```bash
# 1. System dependencies
sudo apt install -y \
    python3-gi python3-gi-cairo \
    gir1.2-gtk-4.0 gir1.2-adw-1 \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    v4l-utils v4l2loopback-dkms \
    pipewire-bin pulseaudio-utils

# 2. Python venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# 3. Install exactly one ONNX Runtime variant
pip install -e ".[cpu]"

# For NVIDIA GPU acceleration on Linux x86_64, choose CUDA instead:
pip install -e ".[cuda]"

# Optional: preserve the OpenAI Whisper compatibility backend:
pip install -e ".[cpu,meeting]"  # or .[cuda,meeting]

# 4. Optional: CuPy-only retry for GPU compositing
pip install "cupy-cuda12x>=14.1.1,<15" nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

# 5. Virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVbroadcast" exclusive_caps=1 max_buffers=4

# Use another output node if /dev/video10 conflicts:
NVBROADCAST_VCAM_DEVICE_NUM=11 ./scripts/setup_v4l2loopback.sh

# 6. Run
python -m nvbroadcast
```

The `.[meeting]` compatibility extra retains the guarded `openai-whisper`
backend on Python versions below 3.14 without selecting an ONNX Runtime owner.
Combine it with exactly one runtime variant as `.[cpu,meeting]` or
`.[cuda,meeting]`. Because `faster-whisper` declares `onnxruntime` directly,
plain pip extras cannot safely install both meeting backends while preserving
strict runtime ownership. Use the source installer for the complete meeting
stack:

```bash
./install.sh --runtime auto --with-meeting
```

This installs support dependencies through project metadata, installs
`faster-whisper` with `--no-deps`, and installs guarded OpenAI Whisper on Python
versions below 3.14. Native packages keep their lighter `faster-whisper` policy;
the macOS installer keeps its best-effort OpenAI Whisper fallback.

</details>

---

## Usage

### Setup Once, Forget Forever

```bash
nvbroadcast          # Launch GUI (first time: setup wizard)
```

1. **Setup wizard** detects your system and configures the best mode
2. App starts and auto-begins streaming
3. Configure effects, select resolution/FPS/mode
4. **Close the window** — app minimizes to background, virtual camera stays active
5. Open **Chrome / Zoom / Discord** — select **"NVbroadcast"** on Linux or **"OBS Virtual Camera"** on macOS
6. **Next login** — app starts automatically with all your settings remembered

### Controls

| Control | Description |
|---------|-------------|
| **Resolution** | 360p to 4K — auto-detected from camera, applied safely after restart |
| **FPS** | 15-60fps — adapts to selected resolution |
| **Mode** | 9 modes: Killer, Zeus, DocZeus, CUDA, CPU |
| **Mirror** | Horizontal flip on/off |
| **Edge Refine** | Neural edge refinement (Zeus/Killer) |
| **Pause View** | Freeze preview display |
| **Hide Preview** | Collapse preview for more control space |
| **Drag Divider** | Resize preview vs controls area |

### Headless Mode

```bash
nvbroadcast-vcam                    # No GUI, just the virtual camera
nvbroadcast-vcam --format i420      # Firefox-compatible format
```

### As a System Service

Use this only for no-GUI/headless passthrough workflows. Do not run the
headless service at the same time as the GUI app, because both need exclusive
access to the physical camera and `NVbroadcast` virtual camera.

```bash
systemctl --user enable --now nvbroadcast-vcam

# If you use the GUI app instead:
systemctl --user disable --now nvbroadcast-vcam
```

The headless command is a passthrough producer for OBS/browser workflows. For
full background effects, start the main NVbroadcast app first, then select the
`NVbroadcast` camera in OBS or your meeting app.

---

## Troubleshooting

<details>
<summary><strong>OBS shows v4l2loopback-000, an old camera name, or a blank feed</strong></summary>

OBS can only display frames after NVbroadcast is actively writing to the virtual
camera. Start the main app for background effects, then select `NVbroadcast` in
OBS. Do not run `nvbroadcast-vcam` and the main app at the same time.

If the visible camera name is still old after an update, close OBS, browsers,
meeting apps, and NVbroadcast, then reboot. Advanced users can reload the
loopback device instead:

```bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVbroadcast" exclusive_caps=1 max_buffers=4
```

</details>

<details>
<summary><strong>Chrome doesn't see the virtual camera</strong></summary>

1. Go to `chrome://flags`
2. Search **"PipeWire"**
3. **Disable** "PipeWire Camera" flag
4. Restart Chrome

</details>

<details>
<summary><strong>"Device busy" error</strong></summary>

Another app is using the camera. Close it or run:
```bash
fuser -k /dev/video0
```

</details>

<details>
<summary><strong>No GPU acceleration (running on CPU)</strong></summary>

Stop NVBroadcast, then ask the source installer to recreate its environment as the CUDA variant:
```bash
./install.sh --runtime cuda
```

For a manually managed source environment, recreate the virtual environment and
install `.[cuda]`. Never overlay `.[cuda]` on an existing `.[cpu]` environment.
Bare `pip install .` is runtime-neutral and intended for downstream packagers
that provide exactly one ONNX Runtime owner themselves.

Verify ownership and execute the pinned probe model in a fresh process with CPU
fallback disabled:
```bash
.venv/bin/python -m nvbroadcast.runtime --variant cuda
```

The command succeeds only when `CUDAExecutionProvider` creates a session,
executes the probe graph on CUDA, and returns the expected output. Provider
enumeration by itself is not treated as GPU readiness. To verify TensorRT in a
CUDA-owned environment, run:

```bash
.venv/bin/python -m nvbroadcast.runtime --variant cuda --provider tensorrt
```

On Python `3.14+`, TensorRT may still be unavailable, but CUDA modes can run
when the default CUDA probe succeeds.

The amd64 Snap includes CUDA inference, compositing, and frame conversion, but
intentionally uses GStreamer's CPU MJPEG decoder. Bundling the optional
nvImageCodec and nvJPEG runtimes would add about 256 MB to the Snap. Source,
Debian, and RPM installs using the CUDA extra can use GPU MJPEG decoding.

</details>

<details>
<summary><strong>Resolution changes do not apply immediately</strong></summary>

Resolution changes are now saved safely and applied after you stop and start the app again. This avoids the live-pipeline hang path that some cameras and loopback setups hit during hot restarts.

If a camera still behaves oddly after restart, verify its real supported modes:
```bash
v4l2-ctl -d /dev/video0 --list-formats-ext   # Check supported resolutions
```

</details>

---

## Project Structure

```
nvidia-broadcast-linux/
├── src/nvbroadcast/
│   ├── __init__.py              # Package version (1.5.2)
│   ├── app.py                   # GTK4 app: modes, effects, pipeline management
│   ├── vcam_service.py          # Headless virtual camera service
│   ├── __main__.py              # CLI entry point
│   ├── ai/
│   │   ├── transcriber.py       # Local meeting transcription
│   │   └── summarizer.py        # Local meeting notes and summary extraction
│   ├── core/
│   │   ├── config.py            # TOML config, performance profiles, compositing backends
│   │   ├── constants.py         # App ID, paths, GPU config
│   │   ├── dependency_installer.py  # Optional runtime installer flow
│   │   ├── global_hotkeys.py     # Portal and GNOME global effect shortcuts
│   │   ├── gpu.py               # GPU detection, CUDA device mapping
│   │   ├── meeting_store.py     # On-device meeting history and retention
│   │   ├── model_download.py     # Verified per-user AI model cache
│   │   ├── platform.py          # OS/runtime feature detection
│   │   ├── resources.py         # Packaged resource lookup
│   │   └── updates.py           # GitHub release/update helpers
│   ├── runtime/
│   │   ├── artifact.py          # Installed dependency and artifact inspection
│   │   ├── probe.py             # Fresh-process provider execution probe
│   │   └── variants.py          # CPU/CUDA runtime ownership contracts
│   ├── video/
│   │   ├── effects.py           # Multi-model engine, fused CUDA kernel, edge refiner
│   │   ├── pipeline.py          # GStreamer pipeline, async effects, frame throttling
│   │   ├── beautify.py          # Video enhancement (5 effects + GPU batch)
│   │   ├── autoframe.py         # MediaPipe face tracking with smooth zoom/pan
│   │   ├── eye_contact.py       # Eye contact correction
│   │   ├── face_landmarks.py    # Shared MediaPipe face landmark worker
│   │   ├── perf_monitor.py      # FPS/GPU performance monitor
│   │   ├── relighting.py        # Face relighting effect
│   │   ├── vcam_monitor.py       # Safe virtual-camera consumer detection
│   │   └── virtual_camera.py    # v4l2loopback + camera capability query
│   ├── audio/
│   │   ├── deepfilter.py        # DeepFilterNet3 ONNX speech enhancement
│   │   ├── devices.py           # Mic/speaker enumeration and routing
│   │   ├── effects.py           # Denoiser selection and RNNoise fallback
│   │   ├── level_monitor.py     # Audio level meter
│   │   ├── meeting_capture.py   # Mixed mic + speaker meeting capture
│   │   ├── mic_test.py          # Processed mic recording/playback test
│   │   ├── pipeline.py          # GStreamer audio pipeline
│   │   ├── monitor.py           # Speaker output denoise
│   │   ├── service.py           # Background audio helper service
│   │   ├── virtual_mic.py       # PipeWire/Pulse virtual microphone
│   │   └── voice_fx.py          # Voice EQ, gate, compression, presets
│   └── ui/
│       ├── window.py            # Main window: resizable paned layout, 9 modes
│       ├── setup_wizard.py      # First-run wizard
│       ├── controls.py          # Effect toggles, sliders, file picker
│       ├── device_selector.py   # Dropdown selector (single-connect fix)
│       ├── sni_tray.py          # Native StatusNotifierItem tray
│       ├── tray.py              # Optional legacy tray integration
│       ├── video_preview.py     # Live video preview
│       └── style.css            # App styling with Adwaita/system theme integration
├── models/                      # AI models (auto-downloaded)
│   ├── rvm_mobilenetv3_fp32.onnx
│   ├── rvm_resnet50_fp32.onnx
│   ├── rvm_mobilenetv3_fp16.onnx   # Lightweight refiner model
│   ├── rvm_resnet50_fp32_trt.onnx  # TensorRT shape-inferred
│   └── rvm_mobilenetv3_fp32_trt.onnx
├── configs/                     # v4l2loopback and PipeWire templates
├── data/                        # Desktop, metainfo, service, icons, backgrounds
├── docs/                        # GitHub Pages site and release notes
├── macos/                       # CoreMediaIO camera extension and helper bridge
├── packaging/                   # Debian and RPM package metadata
├── scripts/                     # Model/setup/release/quality tooling
├── snap/                        # Snapcraft package metadata and store assets
├── tests/                       # Unit and integration tests
├── install.sh                   # Multi-distro installer
├── install_macos.sh             # macOS installer
├── uninstall.sh                 # Clean removal
├── build-packages.sh            # Debian/RPM/macOS package builder
├── pyproject.toml               # Package config (v1.5.2)
└── README.md
```

---

## Contributing

Contributions, feedback, and ideas are **warmly welcome**.

Recognized external contributors receive cumulative credit in every later app
release and in [CONTRIBUTORS.md](CONTRIBUTORS.md), including a public record of
the work or technical finding behind each credit. Original project authorship,
external contributions, community testing, and financial sponsorship are
recognized separately so that one form of support never replaces another.

### How to Contribute

1. **Fork** this repository
2. **Create a branch** (`git checkout -b feature/amazing-thing`)
3. **Commit** with clear messages
4. **Open a Pull Request**

### Report Issues

Found a bug? [Open an issue](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/new).

### Ideas for Contribution

- [x] Eye contact correction *(v0.3.0)*
- [x] Virtual lighting / face relighting *(v0.3.0)*
- [x] System tray indicator *(v0.2.0)*
- [x] Multi-camera support *(v0.3.0)*
- [x] Recording mode *(v0.3.0)*
- [x] Performance overlay (FPS, GPU usage) *(v0.3.0)*
- [x] GStreamer NVDEC/NVENC hardware codec pipeline *(v0.3.0)*
- [ ] NVIDIA Maxine SDK integration
- [ ] Flatpak packaging
- [x] Snap packaging

### Recognized Contributors

The permanent contributor record currently recognizes
[John Maingi](https://github.com/JohnMaingi-IXP),
[Jon Fuller](https://github.com/perfectra1n),
[Cédric Prezelin](https://github.com/Tenshock), and
[Cenkay Çoban](https://github.com/pastor0711). See
[CONTRIBUTORS.md](CONTRIBUTORS.md) for the contribution summary and source
references behind each credit.

## Future Upgrades

- **Meeting lip-sync compensation** — explicit audio/video delay calibration so heavy live video stacks still land naturally in calls
- **Per-device auto benchmark** — benchmark each camera mode and effect stack once, then pin the best stable settings for that machine
- **Speaker diarization** — separate “me” vs “remote speaker” in live meeting transcripts and saved notes
- **Local live captions** — optional on-screen captions and confidence-aware subtitle output for streams and calls
- **Multi-person framing** — presenter mode for interviews, podcasts, and side-by-side calls
- **AI meeting memory** — on-device semantic search across prior meetings, summaries, action items, and decisions
- **Scene-aware relighting** — stronger face light that reacts to background direction, exposure, and skin tone without flattening the face
- **Quality advisor** — explain exactly which effect, resolution, or backend is costing FPS on the current hardware

---

## Sustain NV Broadcast

I created and lead NV Broadcast with help from community bug reports, testing, documentation, ideas, and code contributions. As usage grows, so does the work required to review that input and test GPU runtimes, cameras, audio, packages, and releases across real systems.

The first sustainability goal is **10 monthly sponsors**. Reaching it will help reserve predictable maintenance time each month for bug triage, compatibility fixes, package testing, and reliable releases. Core features will remain open source.

Financial supporters make sustained maintainer time possible, while contributors strengthen the project through reports, testing, documentation, ideas, and code. Sponsorship funds maintenance for the whole community, and roadmap priorities remain based on security, impact, reproducibility, community needs, and what is technically right for the project.

<p align="center">
  <a href="https://github.com/sponsors/Hkshoonya?metadata_campaign=nvbroadcast_readme">
    <img src="https://img.shields.io/badge/Sponsor_DocZeus-Support_Development-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor">
  </a>
</p>

### Founding Sponsor

- [@Mattsky](https://github.com/Mattsky) - supporting NV Broadcast while its sustainability program is being established

### 💎 Featured Sponsors

<!-- featured --><em>No featured sponsors yet - <a href="https://github.com/sponsors/Hkshoonya?metadata_campaign=nvbroadcast_readme">become a Project Backer</a> and your name appears here.</em><!-- featured -->

### 💚 Backers &amp; Supporters

<!-- sponsors --><a href="https://github.com/Mattsky" title="Mattsky"><img src="https://avatars.githubusercontent.com/u/2619664?u=a3e9b73765da4dd8f3472520e40c9588c65a7803&v=4" width="55" alt="Mattsky"></a>&nbsp;<!-- sponsors -->

<p align="center">
  <a href="https://github.com/sponsors/Hkshoonya?metadata_campaign=nvbroadcast_readme">GitHub Sponsors</a> ·
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/issues">Report bugs</a> ·
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/discussions">Share ideas</a>
</p>

---

## License

- **Python app & Linux code:** GPL-3.0 — see [LICENSE](LICENSE)
- **macOS Camera Extension** (`macos/`): Proprietary — see [macos/LICENSE](macos/LICENSE)

Any redistribution or derivative work **must retain the original author attribution**.

---

<p align="center">
  <img src="data/icons/doczeus-logo.svg" width="48" alt="DocZeus">
</p>

<p align="center">
  <strong>Created with passion by <a href="https://github.com/Hkshoonya">DocZeus</a></strong><br>
  <em>Because Linux users deserve broadcast-quality video too.</em>
</p>

<p align="center">
  <sub>Copyright (c) 2026 DocZeus. All rights reserved under GPL-3.0.</sub>
</p>

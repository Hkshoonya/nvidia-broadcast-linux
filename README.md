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
  <a href="https://github.com/sponsors/Hkshoonya"><img src="https://img.shields.io/badge/Sponsor-DocZeus-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor"></a>
</p>

<p align="center">
  <em>NV Broadcast — Unofficial NV Broadcast and other OS. Open source. GPU accelerated. Built with passion.</em>
</p>

---

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

## What's New

### v1.1.1 — Stability Patch

- **Virtual Camera Stability** — Safer Linux `v4l2loopback` sink startup and retry handling
- **Lower Live Lag** — Shared face landmarks and face-ROI relighting reduce delay in heavier effect stacks
- **Better Replace Edges** — Tighter shoulders, ear-side hair, and under-arm gaps during background replace
- **Meeting Transcription Reliability** — Faster startup, shorter chunking, and cleaner saved meeting audio
- **Resolution Change Safety** — Resolution changes are saved safely and applied after restart instead of hanging the stream

### v1.1.0 — Meeting Assistant Update

- **Meeting Assistant Sidebar** — Collapsible live transcript and rolling summary inside the app
- **Meeting History** — Local session history stays on-device for 7 days with automatic cleanup
- **Two-Way Meeting Audio** — Meeting capture records both sides for better local notes and transcripts
- **Background Runtime Installs** — Optional CUDA, TensorRT, and meeting runtimes install in the background with progress
- **Improved Setup Guidance** — First-run flow explains modes, downloads, and skip/install choices more clearly

### v1.0.0 — AI Release

- **AI Meeting Transcription** — Local Whisper speech-to-text (tiny/base/small/medium models, GPU-accelerated)
- **AI Meeting Summarizer** — Extracts action items, questions, key points from transcripts (fully local)
- **Voice Effects** — Bass boost, treble, warmth, compression, noise gate, gain (GPU + CPU)
- **6 Voice Presets** — Natural, Radio, Podcast, Deep Voice, Bright, Studio
- **Microphone Selection** — Full PipeWire/PulseAudio device enumeration
- **Speaker Detection** — All output devices via PipeWire
- **Audio Level Monitor** — Real-time VU meter with peak hold
- **Mic Test** — Record 5s and play back to test your setup
- **Meeting Mode** — Combined video+audio recording with live transcription and AI summary
- **Recording Fix** — MP4 now includes audio track (NVENC video + AAC audio)
- **Voice FX GPU Acceleration** — CuPy CUDA for warmth/gate/gain, scipy for filters (2.8ms/chunk)

### v0.3.0

- **Eye Contact Correction** — MediaPipe iris tracking redirects your gaze to look at camera
- **Face Relighting** — Matches face brightness and warmth to background
- **Recording Mode** — NVENC hardware encode to MP4 (x264 fallback on non-NVIDIA)
- **Performance Overlay** — Real-time FPS, GPU usage, VRAM, temperature monitoring
- **User Profiles** — 5 built-in (Meeting, Streaming, Presentation, Gaming, Clean) + custom save/load
- **Multi-Camera Support** — Hot-switch between cameras without restarting
- **Apple-Inspired UI** — Glassmorphism cards, collapsible sections, smooth transitions
- **Shared FaceLandmarker** — Single MediaPipe instance shared across all face effects (3x faster)
- **macOS Support** — CPU modes with CoreML, AVFoundation camera, Homebrew installer
- **CI Pipeline** — GitHub Actions builds .deb, .rpm, .pkg + Swift Camera Extension on macOS

### v0.2.0

### Premium GPU Modes
- **Killer Mode** — Fused CUDA kernel + 360p inference = **48fps at 1080p** (20ms/frame)
- **Zeus Mode** — 480p optimized inference = **33fps at 1080p** (30ms/frame)
- **DocZeus Mode** — Fused CUDA kernel compositing = **CUDA Max quality at 150x faster blend** (0.1ms vs 15ms)

### Edge Refinement Neural Network
- Toggle-activated second-pass inference at 720p for Zeus/Killer modes
- Uses RVM ResNet50 at full resolution with morphological edge band blending
- **89.9% quality recovery** — brings fast modes close to max quality edges

### Video Enhancement
- **5 independent effects**: Skin Smooth, Denoise, Enhance, Sharpen, Edge Darken
- **4 presets**: Natural, Broadcast, Glamour, Custom
- Per-effect toggle + intensity slider
- MediaPipe FaceLandmarker at half-res, every 5th frame
- GPU batch processing (CuPy) for enhance + sharpen + vignette

### Resolution & FPS Selector
- Auto-detects camera capabilities via v4l2
- Shows only supported resolutions (360p to 4K)
- FPS dropdown adapts per resolution (e.g., 4K shows 30fps, 1080p shows 30+60fps)
- Validated before pipeline start — no more cap negotiation hangs

### UI Improvements
- **Resizable preview** — drag the divider between preview and controls
- **Pause View** — freeze the preview display (camera keeps running)
- **Hide Preview** — collapse preview entirely for more control space
- **Mirror toggle** — horizontal flip for webcam view
- **Scrollable controls** — all settings accessible regardless of window size
- **Grouped cards** — Input, Processing, Background, Auto Frame, Beauty

### Performance Optimizations
- **Pre-downsampling**: Frames above 720p are downsampled before inference (124ms -> 29ms at 1080p)
- **Async effects processing**: Capture thread never blocks — zero preview latency
- **Python-side frame throttling**: No pipeline restart for mode/profile changes
- **Fused CUDA kernel**: Single GPU pass for alpha blend + enhance + vignette (0.1ms)

---

## What It Does

<table>
<tr>
<td width="50%">

### Camera Effects
- **Background Blur** — AI-powered, person stays crystal sharp
- **Background Replace** — Any image via native file picker
- **Green Screen** — Solid green for OBS chroma key
- **Auto Frame** — Face tracking with smooth zoom/pan
- **Video Enhancement** — Skin smooth, enhance, sharpen, denoise, vignette
- **Eye Contact Correction** — MediaPipe iris tracking redirects gaze to camera
- **Face Relighting** — Matches face brightness and warmth to background
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
- **Mic Test** — Record 5s and play back to test your setup
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

---

## Architecture

```
                         NV Broadcast v1.0.0
                         ─────────────────────────────────

  ┌───────────┐      ┌──────────────────────────────────────────┐      ┌──────────────┐
  │  Webcam   │─────▶│           GStreamer Pipeline              │─────▶│ Virtual Cam  │
  │(360p-4K)  │      │                                          │      │ /dev/video10 │
  └───────────┘      │  JPEG Decode ─▶ Color Convert ─▶ appsink │      └──────┬───────┘
                     └──────────────────────┬───────────────────┘             │
                                            │                        ┌───────▼───────┐
                            ┌───────────────▼──────────────┐         │ Chrome / Zoom │
                            │    Async Effects Thread       │         │ Firefox / OBS │
                            │   (never blocks capture)      │         │ Discord/Meet  │
                            │                               │         └───────────────┘
                            │  ┌─────────────────────────┐  │
                            │  │   AI Segmentation        │  │
                            │  │                          │  │
                            │  │  Pre-downsample to 720p  │  │
                            │  │  (or 480/360 for Zeus/   │  │
                            │  │   Killer modes)          │  │
                            │  │                          │  │
                            │  │  ┌────┐ ┌─────┐ ┌────┐  │  │
                            │  │  │RVM │ │ISNet│ │BiR │  │  │
                            │  │  └──┬─┘ └──┬──┘ └─┬──┘  │  │
                            │  │     └───┬──┘      │     │  │
                            │  │         ▼         │     │  │
                            │  │   Alpha Refine    │     │  │
                            │  │  (sigmoid+dilate) │     │  │
                            │  └────────┬──────────┘     │  │
                            │           │                 │  │
                            │  ┌────────▼───────────────┐ │  │
                            │  │  Edge Refiner (opt.)   │ │  │
                            │  │  720p 2nd pass RVM     │ │  │
                            │  │  (Zeus/Killer only)    │ │  │
                            │  └────────┬───────────────┘ │  │
                            │           │                 │  │
                            │  ┌────────▼───────────────┐ │  │
                            │  │     Compositing        │ │  │
                            │  │                        │ │  │
                            │  │  ┌────────┐ ┌───────┐  │ │  │
                            │  │  │ Fused  │ │ CuPy  │  │ │  │
                            │  │  │ CUDA   │ │ CUDA  │  │ │  │
                            │  │  │ 0.1ms  │ │ 15ms  │  │ │  │
                            │  │  └────────┘ └───────┘  │ │  │
                            │  └────────────────────────┘ │  │
                            │           │                 │  │
                            │  ┌────────▼───────────────┐ │  │
                            │  │   Video Enhancement  │ │  │
                            │  │  5 effects + presets   │ │  │
                            │  │  GPU batch (CuPy)      │ │  │
                            │  └────────────────────────┘ │  │
                            │           │                 │  │
                            │  Mirror flip (optional)     │  │
                            └───────────┬─────────────────┘
                                        │
                            ┌───────────▼──────────────┐
                            │ Preview (GTK4 Texture)    │
                            │ Pause / Hide / Resize     │
                            └──────────────────────────┘

  ┌───────────┐      ┌─────────────────────────────────┐      ┌──────────────┐
  │    Mic    │─────▶│     RNNoise AI Denoise          │─────▶│ Virtual Mic  │
  │           │      │     (48kHz, 10ms frames)        │      │  (PipeWire)  │
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
| **Webcam** | Any USB camera | 720p+ with MJPEG |
| **Mic** | Any audio input | — |

### Software
- **Linux** with NVIDIA driver 525+ (Pop!_OS, Ubuntu, Fedora, Arch, openSUSE, etc.)
- **Python** 3.11+
- **PipeWire** (virtual microphone)
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

### macOS — One Command Install

```bash
git clone https://github.com/Hkshoonya/nvidia-broadcast-linux.git
cd nvidia-broadcast-linux
./install_macos.sh
```

Requires macOS 12+, Homebrew, Python 3.11+. Installs GStreamer, GTK4 via Homebrew.
CPU modes with CoreML acceleration on Apple Silicon. GPU modes (Killer/Zeus/DocZeus/CUDA) are Linux-only and require an NVIDIA GPU.

### Linux Installer Details

The installer:
1. **Detects your distro** and package manager
2. **Checks all requirements** (Python, PipeWire, GPU, DKMS, kernel headers)
3. **Installs missing packages** with the correct names for your distro
4. **Asks about compositing** — CPU, GStreamer GL, or CuPy CUDA
5. **Sets up virtual camera**, launcher scripts, desktop entry, systemd service
6. **Verifies GPU acceleration** and writes initial config
7. **Lets optional runtimes install later** inside the app without blocking the rest of the UI

### Optional: TensorRT (for Zeus/Killer modes)

```bash
.venv/bin/pip install tensorrt tensorrt-cu12 onnx
```

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
    pipewire-bin

# 2. Python venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 3. Install
pip install -e .

# 4. Optional: CuPy for GPU compositing
pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12

# 5. Virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4

# 6. Run
python -m nvbroadcast
```

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
5. Open **Chrome / Zoom / Discord / OBS** — select **"NVIDIA Broadcast"** as your camera
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

```bash
systemctl --user enable nvbroadcast-vcam
systemctl --user start nvbroadcast-vcam
```

---

## Troubleshooting

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

Reinstall CUDA runtime libraries:
```bash
.venv/bin/pip install --force-reinstall nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 \
    nvidia-cusolver-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12
```

</details>

<details>
<summary><strong>Resolution changes do not apply immediately</strong></summary>

Resolution changes are now saved safely and applied after restart. This avoids the live-pipeline hang path that some cameras and loopback setups hit during hot restarts.

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
│   ├── __init__.py              # Package version (1.1.1)
│   ├── app.py                   # GTK4 app: modes, effects, pipeline management
│   ├── vcam_service.py          # Headless virtual camera service
│   ├── core/
│   │   ├── config.py            # TOML config, performance profiles, compositing backends
│   │   ├── constants.py         # App ID, paths, GPU config
│   │   └── gpu.py               # GPU detection, CUDA device mapping
│   ├── video/
│   │   ├── effects.py           # Multi-model engine, fused CUDA kernel, edge refiner
│   │   ├── pipeline.py          # GStreamer pipeline, async effects, frame throttling
│   │   ├── beautify.py          # Video enhancement (5 effects + GPU batch)
│   │   ├── autoframe.py         # MediaPipe face tracking with smooth zoom/pan
│   │   └── virtual_camera.py    # v4l2loopback + camera capability query
│   ├── audio/
│   │   ├── effects.py           # RNNoise denoiser
│   │   ├── pipeline.py          # GStreamer audio pipeline
│   │   ├── monitor.py           # Speaker output denoise
│   │   └── virtual_mic.py       # PipeWire virtual microphone
│   └── ui/
│       ├── window.py            # Main window: resizable paned layout, 9 modes
│       ├── setup_wizard.py      # First-run wizard
│       ├── controls.py          # Effect toggles, sliders, file picker
│       ├── device_selector.py   # Dropdown selector (single-connect fix)
│       ├── video_preview.py     # Live video preview
│       └── style.css            # App styling with Adwaita/system theme integration
├── models/                      # AI models (auto-downloaded)
│   ├── rvm_mobilenetv3_fp32.onnx
│   ├── rvm_resnet50_fp32.onnx
│   ├── rvm_mobilenetv3_fp16.onnx   # Lightweight refiner model
│   ├── rvm_resnet50_fp32_trt.onnx  # TensorRT shape-inferred
│   └── rvm_mobilenetv3_fp32_trt.onnx
├── install.sh                   # Multi-distro installer
├── uninstall.sh                 # Clean removal
├── pyproject.toml               # Package config (v1.1.1)
└── README.md
```

---

## Contributing

Contributions, feedback, and ideas are **warmly welcome**.

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

---

## Sponsor This Project

If NV Broadcast saves you from going back to Windows, consider sponsoring:

<p align="center">
  <a href="https://github.com/sponsors/Hkshoonya">
    <img src="https://img.shields.io/badge/Sponsor_DocZeus-Support_Development-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor">
  </a>
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

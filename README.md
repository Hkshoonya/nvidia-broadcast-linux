<p align="center">
  <img src="data/icons/com.doczeus.NVBroadcast.svg" width="120" alt="NVIDIA Broadcast for Linux">
</p>

<h1 align="center">NVIDIA Broadcast for Linux</h1>

<p align="center">
  <strong>by DocZeus | AI Powered</strong>
</p>

<p align="center">
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/stargazers"><img src="https://img.shields.io/github/stars/Hkshoonya/nvidia-broadcast-linux?style=for-the-badge&color=76b900&labelColor=1a1a1a" alt="Stars"></a>
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-76b900?style=for-the-badge&labelColor=1a1a1a" alt="License"></a>
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/issues"><img src="https://img.shields.io/github/issues/Hkshoonya/nvidia-broadcast-linux?style=for-the-badge&color=76b900&labelColor=1a1a1a" alt="Issues"></a>
  <a href="https://github.com/sponsors/Hkshoonya"><img src="https://img.shields.io/badge/Sponsor-DocZeus-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor"></a>
</p>

<p align="center">
  <em>The NVIDIA Broadcast experience you loved on Windows — now on Linux. Open source. GPU accelerated. Built with passion.</em>
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

## What It Does

<table>
<tr>
<td width="50%">

### Camera Effects
- **Background Blur** — AI-powered, person stays crystal sharp
- **Background Replace** — Any image via native file picker
- **Green Screen** — Solid green for OBS chroma key
- **Auto Frame** — Face tracking with smooth zoom/pan
- **Multi-Model** — RVM (person), IS-Net (objects), BiRefNet (best edges)

</td>
<td width="50%">

### Audio Effects
- **Mic Noise Removal** — Kills keyboard, fan, environment noise
- **Speaker Denoise** — Clean up incoming audio

### System Integration
- **Virtual Camera** — Works in Chrome, Firefox, Zoom, Discord, OBS
- **Auto-Start** — Launches on login, runs in background
- **Setup Wizard** — Auto-detects system, configures optimally
- **Multi-GPU** — Select which GPU runs AI effects
- **Multi-Distro** — Ubuntu, Fedora, Arch, openSUSE, and more

</td>
</tr>
</table>

---

## Architecture

```
                           NVIDIA Broadcast for Linux
                           ─────────────────────────

    ┌───────────┐      ┌─────────────────────────────────┐      ┌──────────────┐
    │  Webcam   │─────▶│         GStreamer Pipeline       │─────▶│ Virtual Cam  │
    │  (USB)    │      │                                  │      │ /dev/video10 │
    └───────────┘      │  ┌─────────────────────────────┐ │      └──────┬───────┘
                       │  │    JPEG Decode (GPU/CPU)     │ │             │
                       │  │  nvjpegdec or software dec   │ │    ┌───────▼───────┐
                       │  └──────────┬──────────────────┘ │    │ Chrome / Zoom │
                       │             │                     │    │ Firefox / OBS │
                       │  ┌──────────▼──────────────────┐ │    │ Discord/Meet  │
                       │  │   AI Segmentation Engine     │ │    └───────────────┘
                       │  │                              │ │
                       │  │  ┌────────┐ ┌──────┐ ┌────┐ │ │
                       │  │  │  RVM   │ │IS-Net│ │Bi- │ │ │
                       │  │  │(person)│ │(obj) │ │Ref │ │ │
                       │  │  └───┬────┘ └──┬───┘ └─┬──┘ │ │
                       │  │      └────┬────┘       │    │ │
                       │  │           ▼            │    │ │
                       │  │    Alpha Matte (GPU)    │    │ │
                       │  └──────────┬──────────────────┘ │
                       │             │                     │
                       │  ┌──────────▼──────────────────┐ │
                       │  │   Compositing Engine         │ │
                       │  │                              │ │
                       │  │  ┌──────┐ ┌─────┐ ┌──────┐  │ │
                       │  │  │ CuPy │ │ GL  │ │ CPU  │  │ │
                       │  │  │ CUDA │ │(GSt)│ │(cv2) │  │ │
                       │  │  └──────┘ └─────┘ └──────┘  │ │
                       │  └──────────────────────────────┘ │
                       └───────────────────────────────────┘

    ┌───────────┐      ┌─────────────────────────────────┐      ┌──────────────┐
    │    Mic    │─────▶│     RNNoise AI Denoise          │─────▶│ Virtual Mic  │
    │           │      │     (48kHz, 10ms frames)        │      │  (PipeWire)  │
    └───────────┘      └─────────────────────────────────┘      └──────────────┘
```

### The Processing Pipeline

Every frame flows through this path:

```
Camera (30fps MJPEG)
  │
  ▼
JPEG Decode ────────── nvjpegdec (GPU) or jpegdec (CPU)
  │
  ▼
Color Convert ──────── BGRA for processing
  │
  ▼
AI Segmentation ────── ONNX Runtime + CUDA (GPU inference)
  │                    ├── RVM: Person matting, temporal state, ~5ms
  │                    ├── IS-Net: General objects, ~55ms
  │                    └── BiRefNet: Best edges, ~187ms (needs 8GB+ VRAM)
  │
  ▼
Alpha Refinement ───── Dilate → Blur → Sigmoid (configurable)
  │
  ▼
Compositing ────────── Blend foreground + background
  │                    ├── CuPy CUDA: ~1ms (GPU arrays, near-zero CPU)
  │                    ├── GStreamer GL: OpenGL compositor (GPU)
  │                    └── CPU (cv2): ~4ms (SIMD-optimized fallback)
  │
  ▼
Virtual Camera ─────── v4l2loopback → /dev/video10
  │
  ▼
Preview ────────────── GTK4 texture → app window (~30fps)
```

### Key Technical Decisions

| Component | What It Does | Why It's Fast |
|-----------|-------------|---------------|
| **Multi-Model Engine** | Swappable AI backends (RVM, IS-Net, BiRefNet) | Each model optimized for different use cases |
| **GPU JPEG Decode** | `nvjpegdec` decodes camera frames on GPU | Saves ~60% CPU vs software `jpegdec` |
| **CuPy CUDA Blend** | Compositing runs entirely on GPU via CUDA arrays | 1ms blend vs 4ms CPU — near-zero CPU overhead |
| **Dual-Mode Pipeline** | Passthrough = zero CPU; Effects = GPU processing | GStreamer handles passthrough in pure C |
| **Async Teardown** | Pipeline stop/start runs in background threads | UI never freezes during mode switches |
| **CUDA Device Ordering** | `CUDA_DEVICE_ORDER=PCI_BUS_ID` forced at import | GPU selector matches `nvidia-smi` numbering |
| **VRAM Management** | `kSameAsRequested` arena + 2GB limit + HEURISTIC algo | No pre-allocation waste, ~660MB for RVM |
| **Alpha Refinement** | Dilate → Gaussian blur → sigmoid curve | Configurable per-system via Advanced panel |

---

## Processing Modes

The installer and setup wizard auto-detect your system and recommend the best mode:

| Mode | Compositing | FPS | CPU Usage | Best For |
|------|-------------|-----|-----------|----------|
| **CUDA GPU - Maximum** | CuPy CUDA | 30 | ~30% | High-end NVIDIA systems |
| **GPU OpenGL - Best** | GStreamer GL | 30 | ~60% | Systems with GL plugins |
| **GPU OpenGL - Balanced** | GStreamer GL | 20 | ~45% | Daily use (recommended) |
| **CPU - High Quality** | OpenCV SIMD | 30 | ~250% | No GPU compositor |
| **CPU - Light** | OpenCV SIMD | 15 | ~60% | Save CPU for other apps |
| **Low-End** | OpenCV SIMD | 10 | ~30% | Older hardware |

> You can switch modes anytime from the **Mode** dropdown in the app. No restart needed.

---

## AI Models

| Model | Segments | Speed (RTX 5060) | VRAM | License | Auto-Download |
|-------|----------|-----------------|------|---------|---------------|
| **RVM** (default) | Person only | ~5ms (100+ fps) | 660 MB | GPL-3.0 | Yes |
| **IS-Net** | Any object | ~55ms (18 fps) | 1.8 GB | Apache 2.0 | Yes |
| **BiRefNet** | Best edges | ~187ms (5 fps) | 6+ GB | MIT | Yes |

RVM uses **recurrent temporal states** (r1-r4) for smooth frame-to-frame consistency. IS-Net and BiRefNet are single-frame models with **EMA temporal smoothing** to reduce flicker.

### Quality Presets (RVM only)

| Preset | Backbone | Downsample | Speed | Best For |
|--------|----------|-----------|-------|----------|
| Performance | MobileNetV3 | 0.25 | ~5ms | Video calls |
| Balanced | MobileNetV3 | 0.5 | ~7ms | Daily use |
| Quality | ResNet50 | 0.375 | ~10ms | Presentations |
| Ultra | ResNet50 | 0.5 | ~12ms | Recording |

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

> The installer auto-detects your distro and installs the correct packages. Supports **apt** (Debian/Ubuntu/Pop!_OS/Mint), **dnf** (Fedora/RHEL), **pacman** (Arch/Manjaro), and **zypper** (openSUSE).

---

## Installation

### One Command Install

```bash
git clone https://github.com/Hkshoonya/nvidia-broadcast-linux.git
cd nvidia-broadcast-linux
./install.sh
```

The installer:
1. **Detects your distro** and package manager
2. **Checks all requirements** (Python, PipeWire, GPU, DKMS, kernel headers)
3. **Installs missing packages** with the correct names for your distro
4. **Asks about compositing** — CPU, GStreamer GL, or CuPy CUDA
5. **Sets up virtual camera**, launcher scripts, desktop entry, systemd service
6. **Verifies GPU acceleration** and writes initial config

```
=========================================
  NVIDIA Broadcast for Linux
  by DocZeus | AI Powered
=========================================

[Pre-flight] Checking system requirements...
  Distro: Pop!_OS 22.04 (apt)
  Python 3.12 ... OK
  NVIDIA GPU ... OK (NVIDIA GeForce RTX 5060)
  DKMS ... OK

[Compositing] How should blur/blend compositing run?
  Your system:
    NVIDIA GPU: GeForce RTX 5060 (8151 MB)
    GStreamer GL: available

  1) CPU compositing (works everywhere, ~200% CPU)
  2) GStreamer OpenGL GPU (recommended, ~60% CPU)
  3) CuPy CUDA GPU (best quality, ~30% CPU — downloads ~800MB)

  Select [1/2/3] (default: 2): 3
  Installing CuPy CUDA compositing...
  CuPy CUDA compositing installed and verified!

[1/7] System packages .............. done
[2/7] Virtual camera ............... done
[3/7] Python environment ........... done
[4/7] Launcher scripts ............. done
[5/7] Desktop entry & icon ......... done
[6/7] Systemd service .............. done
[7/7] Autostart on login ........... done

  Detected: Pop!_OS 22.04 (apt)
  Compositing: cupy
  Setup once, forget forever.
```

### First-Run Setup Wizard

On first launch, a **setup wizard** appears that:

- **Scans your hardware** — CPU cores, GPU name/VRAM, available backends
- **Shows system info** and marks the best option with ★ recommended
- **Lets you choose** from unified GPU/CPU processing modes
- **Auto-installs CuPy** if you select CUDA mode (with progress + error details)
- **Falls back gracefully** if anything fails (CUDA → OpenGL → CPU)

### Supported Distros

| Distro | Package Manager | Status |
|--------|----------------|--------|
| Ubuntu, Debian, Pop!_OS, Mint | apt | Full auto-install |
| Fedora, RHEL, CentOS, Rocky | dnf/yum | Full auto-install |
| Arch, Manjaro, EndeavourOS | pacman | Full auto-install |
| openSUSE | zypper | Full auto-install |
| Gentoo, Void, NixOS | portage/xbps/nix | Manual instructions shown |

### Manual Install

<details>
<summary>Click to expand manual steps</summary>

```bash
# 1. System dependencies (cannot be installed via pip)
sudo apt install -y \
    python3-gi python3-gi-cairo \
    gir1.2-gtk-4.0 gir1.2-adw-1 \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    v4l-utils v4l2loopback-dkms \
    pipewire-bin  # or pipewire-utils on some distros

# 2. Python venv
sudo apt install -y python3.12-venv   # adjust version
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 3. Install nvbroadcast + all Python dependencies
pip install -e .

# 4. Optional: CuPy for GPU compositing (~800MB)
pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12

# 5. Virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4

# 6. Persist virtual camera across reboots
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4' | \
    sudo tee /etc/modprobe.d/nvbroadcast-v4l2loopback.conf
echo "v4l2loopback" | sudo tee /etc/modules-load.d/nvbroadcast-v4l2loopback.conf

# 7. Run
python -m nvbroadcast
```

**What gets installed and why:**

| Package | Source | Purpose |
|---------|--------|---------|
| `python3-gi`, `python3-gi-cairo` | apt | PyGObject bindings for GTK4 and GStreamer |
| `gir1.2-gtk-4.0`, `gir1.2-adw-1` | apt | GTK4 and Libadwaita UI framework |
| `gir1.2-gstreamer-1.0` | apt | GStreamer video/audio pipeline |
| `gstreamer1.0-plugins-bad` | apt | GPU JPEG decode (nvjpegdec), GL compositor |
| `v4l2loopback-dkms` | apt | Kernel module for virtual camera |
| `pipewire-bin` | apt | `pw-loopback` for virtual microphone |
| `numpy`, `opencv-python-headless` | pip | Image processing and array math |
| `mediapipe` | pip | Face detection for auto-frame |
| `onnxruntime-gpu` | pip | GPU inference engine for AI models |
| `pyrnnoise` | pip | AI noise cancellation |
| `nvidia-*-cu12` (9 packages) | pip | CUDA 12 runtime + NVRTC for GPU acceleration |
| `cupy-cuda12x` (optional) | pip | CUDA GPU compositing (~800MB) |

</details>

---

## Usage

### Setup Once, Forget Forever

```bash
nvbroadcast          # Launch GUI (first time: setup wizard)
```

1. **Setup wizard** detects your system and configures the best mode
2. App starts and auto-begins streaming
3. Configure background blur/replace/green-screen, select model and quality
4. **Close the window** — app minimizes to background, virtual camera stays active
5. Open **Chrome / Zoom / Discord / OBS** — select **"NVIDIA Broadcast"** as your camera
6. **Next login** — app starts automatically with all your settings remembered

### Advanced Controls

Expand **Advanced Edge Tuning** in the app for per-system fine-tuning:

| Control | Range | Effect |
|---------|-------|--------|
| **Dilate** | 0–15 | Expand person mask (prevents edges eating into person) |
| **Softness** | 1–25 | Gaussian blur on alpha (smooth edge transitions) |
| **Sharpness** | 1–30 | Sigmoid steepness (higher = crisper boundary) |
| **Midpoint** | 0.1–0.9 | Where edge transition sits (lower = keeps more person) |
| **Frame Skip** | 1–5 | Inference frequency (1 = every frame, higher = less CPU) |
| **Smoothing** | 0–0.5 | Temporal EMA for single-frame models (reduces flicker) |

All settings save automatically to `~/.config/nvbroadcast/config.toml`.

### Headless Mode

```bash
nvbroadcast-vcam                    # No GUI, just the virtual camera
nvbroadcast-vcam --format i420      # Firefox-compatible format
```

### As a System Service

```bash
systemctl --user enable nvbroadcast-vcam   # Auto-start on login
systemctl --user start nvbroadcast-vcam    # Start now
systemctl --user status nvbroadcast-vcam   # Check status
```

### Uninstall

```bash
./uninstall.sh
```

Removes everything the installer created. You'll be asked whether to also remove system packages.

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

Your NVIDIA driver must support CUDA 12+ (driver 525+). Check with `nvidia-smi`.

</details>

<details>
<summary><strong>CuPy CUDA compositing fails</strong></summary>

If CuPy installs but CUDA kernel compilation fails:
```bash
# Install NVRTC (CUDA runtime compiler)
.venv/bin/pip install nvidia-cuda-nvrtc-cu12

# Or install system CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify CuPy works
.venv/bin/python -c "import cupy; a = cupy.ones(10); print(a * 2)"
```

Common causes:
- Missing `nvidia-cuda-nvrtc-cu12` pip package
- NVIDIA driver too old (need 525+ for CUDA 12)
- Wrong CuPy version (use `cupy-cuda12x` for CUDA 12, `cupy-cuda11x` for CUDA 11)

</details>

<details>
<summary><strong>v4l2loopback not loaded after reboot</strong></summary>

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4
```

</details>

<details>
<summary><strong>GPU selector shows wrong GPU</strong></summary>

The app forces `CUDA_DEVICE_ORDER=PCI_BUS_ID` so GPU numbering matches `nvidia-smi`. If it's still wrong, check:
```bash
nvidia-smi -L   # Lists GPUs in nvidia-smi order
```

</details>

---

## Project Structure

```
nvidia-broadcast-linux/
├── src/nvbroadcast/
│   ├── app.py                 # GTK4 app: auto-start, minimize, profile management
│   ├── vcam_service.py        # Headless virtual camera service
│   ├── core/
│   │   ├── config.py          # TOML config, performance profiles, system detection
│   │   ├── constants.py       # App ID, paths, GPU config, brand colors
│   │   └── gpu.py             # GPU detection, CUDA device mapping
│   ├── video/
│   │   ├── effects.py         # Multi-model engine, compositing backends, alpha refinement
│   │   ├── pipeline.py        # GStreamer pipeline (passthrough + effects + GPU decode)
│   │   ├── autoframe.py       # MediaPipe face tracking with smooth zoom/pan
│   │   └── virtual_camera.py  # v4l2loopback device management
│   ├── audio/
│   │   ├── effects.py         # RNNoise denoiser with intensity control
│   │   ├── pipeline.py        # GStreamer audio pipeline (mic → denoise → virtual mic)
│   │   ├── monitor.py         # Speaker output denoise
│   │   └── virtual_mic.py     # PipeWire virtual microphone
│   └── ui/
│       ├── window.py          # Main window: preview, controls, mode/GPU selectors
│       ├── setup_wizard.py    # First-run wizard: system detect, mode selection
│       ├── controls.py        # Effect toggles, sliders, native file picker
│       ├── device_selector.py # Camera/audio device dropdowns
│       ├── video_preview.py   # Live video preview via Gdk.Texture
│       └── style.css          # NVIDIA-branded dark theme
├── models/                    # AI models (auto-downloaded on first use)
│   ├── rvm_mobilenetv3_fp32.onnx   # 15 MB — fast person matting
│   ├── rvm_resnet50_fp32.onnx      # 103 MB — quality person matting
│   ├── isnet-general-use.onnx      # 171 MB — general object segmentation
│   └── blaze_face_short_range.tflite # Face detection for auto-frame
├── configs/                   # v4l2loopback & PipeWire configs
├── install.sh                 # Multi-distro installer with compositing choice
├── uninstall.sh               # Clean removal with optional package removal
├── setup_deps.sh              # System dependency installer
├── pyproject.toml             # Python package config with all dependencies
├── requirements.txt           # Documented pip + system requirements
├── .gitattributes             # Enforce LF line endings
├── LICENSE                    # GPL-3.0 with attribution requirement
└── README.md
```

---

## Contributing

Contributions, feedback, and ideas are **warmly welcome**. This project is built for the Linux community, by the Linux community.

### How to Contribute

1. **Fork** this repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-thing`)
3. **Commit** your changes with clear messages
4. **Open a Pull Request** — describe what you changed and why

### Rules

- All PRs require **review and approval** before merging
- Keep the **original author attribution** intact (see [LICENSE](LICENSE))
- Follow the existing code style and structure
- Add tests for new features when possible
- Be respectful in discussions — we're all here because we love Linux

### Ideas for Contribution

- [ ] NVIDIA Maxine SDK integration (native background effects)
- [ ] Eye contact correction
- [ ] Virtual lighting / face relighting
- [ ] System tray indicator with quick toggle
- [ ] Flatpak / Snap packaging
- [ ] Multi-camera support
- [ ] Recording mode
- [ ] Performance overlay (FPS, GPU usage)
- [ ] Depth-based segmentation (for chair/desk detection)
- [ ] GStreamer CUDA compositor (keep frames in GPU memory end-to-end)

---

## Sponsor This Project

If NVIDIA Broadcast for Linux saves you from going back to Windows, consider sponsoring the development:

<p align="center">
  <a href="https://github.com/sponsors/Hkshoonya">
    <img src="https://img.shields.io/badge/Sponsor_DocZeus-Support_Development-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor">
  </a>
</p>

Every contribution helps keep this project alive and improving.

---

## License

**GPL-3.0** — see [LICENSE](LICENSE) for details.

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

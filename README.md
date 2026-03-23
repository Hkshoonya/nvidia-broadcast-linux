<p align="center">
  <img src="data/icons/com.doczeus.NVBroadcast.svg" width="120" alt="NVIDIA Broadcast for Linux">
</p>

<h1 align="center">NVIDIA Broadcast for Linux</h1>

<p align="center">
  <strong>by Doczeus | AI Powered</strong>
</p>

<p align="center">
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/stargazers"><img src="https://img.shields.io/github/stars/Hkshoonya/nvidia-broadcast-linux?style=for-the-badge&color=76b900&labelColor=1a1a1a" alt="Stars"></a>
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-76b900?style=for-the-badge&labelColor=1a1a1a" alt="License"></a>
  <a href="https://github.com/Hkshoonya/nvidia-broadcast-linux/issues"><img src="https://img.shields.io/github/issues/Hkshoonya/nvidia-broadcast-linux?style=for-the-badge&color=76b900&labelColor=1a1a1a" alt="Issues"></a>
  <a href="https://github.com/sponsors/Hkshoonya"><img src="https://img.shields.io/badge/Sponsor-Doczeus-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor"></a>
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
> **— Doczeus**

---

## What It Does

<table>
<tr>
<td width="50%">

### Camera Effects
- **Background Blur** — AI-powered, person stays crystal sharp
- **Background Replace** — Any image, perfectly composited
- **Green Screen** — Solid green for OBS chroma key
- **Auto Frame** — Face tracking with smooth zoom/pan

</td>
<td width="50%">

### Audio Effects
- **Mic Noise Removal** — Kills keyboard, fan, environment noise
- **Speaker Denoise** — Clean up incoming audio

### System Integration
- **Virtual Camera** — Works in Chrome, Firefox, Zoom, Discord, OBS
- **Auto-Start** — Launches on login, runs in background
- **Setup & Forget** — All settings persist forever

</td>
</tr>
</table>

---

## How It Works

```
                         NVIDIA Broadcast for Linux
                         ─────────────────────────

    ┌──────────┐     ┌────────────────────┐     ┌──────────────────┐
    │  Webcam  │────▶│  GStreamer Pipeline │────▶│  Virtual Camera  │
    │ (USB)    │     │                    │     │  /dev/video10    │
    └──────────┘     │  ┌──────────────┐  │     └────────┬─────────┘
                     │  │ RobustVideo  │  │              │
                     │  │  Matting     │  │     ┌────────▼─────────┐
                     │  │ (ONNX+CUDA) │  │     │  Chrome / Zoom   │
                     │  └──────┬───────┘  │     │  Firefox / OBS   │
                     │         │          │     │  Discord / Meet  │
                     │  ┌──────▼───────┐  │     └──────────────────┘
                     │  │  Alpha Matte │  │
                     │  │  Compositing │  │
                     │  └──────────────┘  │
                     └────────────────────┘

    ┌──────────┐     ┌────────────────────┐     ┌──────────────────┐
    │   Mic    │────▶│  RNNoise Denoise  │────▶│  Virtual Mic     │
    └──────────┘     └────────────────────┘     └──────────────────┘
```

### The Secret Sauce

| Component | What It Does | Why It's Fast |
|-----------|-------------|---------------|
| **RobustVideoMatting** | True alpha mattes (not binary masks) — natural hair/clothing edges | GPU inference via ONNX Runtime + CUDA |
| **Dual-Mode Pipeline** | Passthrough mode = zero CPU; Effects mode = GPU processing | GStreamer handles passthrough in pure C |
| **Frame Skipping** | AI runs every 2nd frame, reuses alpha matte on skipped frames | Halves GPU load without visible quality loss |
| **uint16 Blending** | Integer math compositing instead of float32 | 4x faster than naive approach |

---

## Quality Presets

| Preset | Model | Edge Quality | Speed | Best For |
|--------|-------|:----------:|:-----:|----------|
| **Performance** | MobileNetV3 | Good | ~7ms | Video calls with blur |
| **Balanced** | MobileNetV3 | Better | ~10ms | Daily use |
| **Quality** | ResNet50 | Excellent | ~12ms | Presentations |
| **Ultra** | ResNet50 | Best | ~15ms | Recording & streaming |

All presets run in real-time on any RTX GPU. Select in-app under Background > Quality.

---

## Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1060 | RTX 3060 or newer |
| **VRAM** | 2 GB | 4 GB+ |
| **Webcam** | Any USB camera | 720p+ with MJPEG |
| **Mic** | Any audio input | — |

### Software
- **Linux** with NVIDIA driver 525+ (Pop!_OS, Ubuntu, Fedora, Arch, etc.)
- **Python** 3.11+
- **PipeWire** (virtual microphone)
- **GStreamer** 1.20+ with plugins-base, plugins-good, plugins-bad
- **GTK4** and **Libadwaita**
- **v4l2loopback** kernel module
- **DKMS** and **kernel headers** (to build v4l2loopback)

> The installer checks all requirements before proceeding and will tell you exactly what's missing.

---

## Installation

### One Command Install

```bash
git clone https://github.com/Hkshoonya/nvidia-broadcast-linux.git
cd nvidia-broadcast-linux
./install.sh
```

The installer checks all requirements, installs what's missing, and sets everything up:

```
=========================================
  NVIDIA Broadcast for Linux
  by Doczeus | AI Powered
=========================================

[Pre-flight] Checking system requirements...
  Python 3.12 ... OK
  PipeWire ... OK
  pw-loopback ... OK
  NVIDIA GPU ... OK (NVIDIA GeForce RTX 5060)
  DKMS ... OK
  Kernel headers ... OK

[1/7] System packages .............. ✓
[2/7] Virtual camera (v4l2loopback) . ✓
[3/7] Python environment ............ ✓
[4/7] Launcher scripts .............. ✓
[5/7] Desktop entry & icon .......... ✓
[6/7] Systemd service ............... ✓
[7/7] Autostart on login ............ ✓

Setup once, forget forever.
```

### Manual Install

<details>
<summary>Click to expand manual steps</summary>

```bash
# 1. System dependencies (cannot be installed via pip)
#    These provide GTK4, GStreamer, virtual camera, and virtual microphone
sudo apt install -y \
    python3-gi python3-gi-cairo \
    gir1.2-gtk-4.0 gir1.2-adw-1 \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    v4l-utils v4l2loopback-dkms \
    pipewire-utils

# 2. Python venv (install python3.X-venv if missing)
sudo apt install -y python3.12-venv   # adjust version to match your Python
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 3. Install nvbroadcast + all Python dependencies (including CUDA 12 libs)
pip install -e .

# 4. Virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4

# 5. Make virtual camera persistent across reboots
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4' | \
    sudo tee /etc/modprobe.d/nvbroadcast-v4l2loopback.conf
echo "v4l2loopback" | sudo tee /etc/modules-load.d/nvbroadcast-v4l2loopback.conf

# 6. Run
python -m nvbroadcast
```

**What gets installed and why:**

| Package | Source | Purpose |
|---------|--------|---------|
| `python3-gi`, `python3-gi-cairo` | apt | PyGObject bindings for GTK4 and GStreamer |
| `gir1.2-gtk-4.0`, `gir1.2-adw-1` | apt | GTK4 and Libadwaita UI framework |
| `gir1.2-gstreamer-1.0` | apt | GStreamer video/audio pipeline |
| `gstreamer1.0-plugins-bad` | apt | GPU JPEG decode (nvjpegdec), advanced elements |
| `v4l2loopback-dkms` | apt | Kernel module for virtual camera |
| `pipewire-utils` | apt | `pw-loopback` for virtual microphone |
| `numpy`, `opencv-python-headless` | pip | Image processing and array math |
| `mediapipe` | pip | Face detection for auto-frame |
| `onnxruntime-gpu` | pip | GPU inference engine for AI models |
| `pyrnnoise` | pip | AI noise cancellation |
| `nvidia-*-cu12` (8 packages) | pip | CUDA 12 runtime libraries for GPU acceleration |

</details>

---

## Usage

### Setup Once, Forget Forever

```bash
nvbroadcast          # Launch GUI (first time: configure your effects)
```

1. App starts and auto-begins streaming
2. Configure background blur/replace/green-screen, select quality
3. **Close the window** — app minimizes to background, virtual camera stays active
4. Open **Chrome / Zoom / Discord / OBS** — select **"NVIDIA Broadcast"** as your camera
5. **Next login** — app starts automatically with all your settings remembered

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

Removes everything the installer created: systemd service, autostart, desktop entry, launchers, v4l2loopback config, and the Python virtual environment.

You'll be asked whether to also remove system packages (GStreamer, GTK4, v4l2loopback, etc.) — these are kept by default since other apps may depend on them.

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

The CUDA 12 runtime libraries are included in requirements but may fail on some systems. Reinstall them:
```bash
.venv/bin/pip install --force-reinstall nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 \
    nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
```

Note: Your NVIDIA driver must support CUDA 12+ (driver 525+). The driver's "CUDA Version" in `nvidia-smi` shows the maximum supported — the pip packages provide the actual runtime libraries.

</details>

<details>
<summary><strong>v4l2loopback not loaded after reboot</strong></summary>

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4
```

</details>

---

## Project Structure

```
nvidia-broadcast-linux/
├── src/nvbroadcast/           # Main package (every file carries doczeus copyright)
│   ├── app.py                 # GTK4 application with auto-start & minimize
│   ├── vcam_service.py        # Headless virtual camera service
│   ├── core/                  # Config, GPU detection, constants
│   ├── video/                 # Pipeline, RVM effects, auto-frame, v4l2loopback
│   ├── audio/                 # RNNoise denoising, PipeWire virtual mic
│   └── ui/                    # Window, preview, controls (NVIDIA dark theme)
├── models/                    # AI models (auto-downloaded on first run)
├── configs/                   # v4l2loopback & PipeWire configs
├── install.sh                 # One-command installer (checks all requirements)
├── uninstall.sh               # Clean removal of everything installed
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

- [ ] Better segmentation models (NVIDIA Maxine SDK integration)
- [ ] Eye contact correction
- [ ] Virtual lighting / face relighting
- [ ] System tray indicator with quick toggle
- [ ] Flatpak / Snap packaging
- [ ] Multi-camera support
- [ ] Recording mode
- [ ] Performance overlay (FPS, GPU usage)

---

## Sponsor This Project

If NVIDIA Broadcast for Linux saves you from going back to Windows, consider sponsoring the development:

<p align="center">
  <a href="https://github.com/sponsors/Hkshoonya">
    <img src="https://img.shields.io/badge/Sponsor_Doczeus-Support_Development-76b900?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=1a1a1a" alt="Sponsor">
  </a>
</p>

Every contribution helps keep this project alive and improving.

---

## License

**GPL-3.0** — see [LICENSE](LICENSE) for details.

Any redistribution or derivative work **must retain the original author attribution**.

---

<p align="center">
  <img src="data/icons/doczeus-logo.svg" width="48" alt="Doczeus">
</p>

<p align="center">
  <strong>Created with passion by <a href="https://github.com/Hkshoonya">Doczeus</a></strong><br>
  <em>Because Linux users deserve broadcast-quality video too.</em>
</p>

<p align="center">
  <sub>Copyright (c) 2026 Doczeus. All rights reserved under GPL-3.0.</sub>
</p>

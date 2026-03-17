# NVIDIA Broadcast for Linux

**by doczeus | AI Powered**

An open-source implementation of [NVIDIA Broadcast](https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/) for Linux. Brings AI-powered virtual camera, background removal/blur/replacement, auto-framing, and noise cancellation to Linux desktops - features previously only available on Windows.

> Created by **doczeus** - [github.com/doczeus](https://github.com/doczeus)

---

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Background Blur** | AI-powered Gaussian blur on background, person stays sharp | Working |
| **Background Replace** | Replace background with any custom image | Working |
| **Green Screen** | Solid green background for OBS chroma key | Working |
| **Auto Frame** | Face tracking with smooth auto-zoom/pan | Working |
| **Mic Noise Removal** | AI denoising for microphone input (RNNoise) | Working |
| **Speaker Denoise** | Remove noise from incoming audio | Working |
| **Virtual Camera** | Works with Chrome, Firefox, Zoom, Discord, OBS | Working |
| **GPU Accelerated** | ONNX Runtime on NVIDIA RTX GPU | Working |
| **Quality Presets** | Performance / Balanced / Quality / Ultra | Working |
| **Auto-Start** | Launches on login, minimizes to background | Working |
| **Settings Memory** | All settings persist across sessions | Working |

## How It Works

```
Webcam -> GStreamer Pipeline -> AI Effects (GPU) -> Virtual Camera
                                    |
                            RobustVideoMatting
                            (ONNX Runtime + CUDA)
                                    |
                    +---------------+---------------+
                    |               |               |
              Background       Green Screen    Background
                 Blur            Remove        Replacement
```

The app creates a **v4l2loopback virtual camera** device (`/dev/video10`) that appears as "NVIDIA Broadcast" in all applications. When streaming, browsers and video apps can select it as their camera source.

### Architecture

- **Passthrough Mode** (no effects): Direct GStreamer C pipeline - zero Python overhead, near-zero CPU
- **Effects Mode**: GStreamer appsink -> Python/ONNX processing -> appsrc -> v4l2sink
- **RobustVideoMatting**: Produces true alpha mattes with temporal consistency via recurrent neural network
- **Adaptive Pipeline**: Automatically switches between passthrough and effects mode

### AI Models

| Model | Purpose | Size | Speed (RTX 5060) |
|-------|---------|------|-------------------|
| RVM MobileNetV3 | Fast segmentation | 14 MB | 7ms inference |
| RVM ResNet50 | High-quality segmentation | 103 MB | 15ms inference |
| MediaPipe BlazeFace | Face detection (auto-frame) | 230 KB | <5ms inference |
| RNNoise | Audio denoising | Built-in | <1ms per frame |

Models are downloaded automatically on first use.

---

## Requirements

### Hardware
- **NVIDIA GPU**: Any RTX series (RTX 2060 or newer recommended)
- **Webcam**: Any USB webcam (MJPEG or YUYV supported)
- **Microphone**: Any audio input device

### Software
- **OS**: Pop!_OS 22.04+, Ubuntu 22.04+, Fedora 38+, or any Linux with:
  - NVIDIA Driver 525+ with CUDA support
  - GTK4 and Libadwaita
  - GStreamer 1.20+
  - PipeWire (for audio effects)
  - v4l2loopback kernel module
- **Python**: 3.11+

---

## Installation

### Quick Install (Recommended)

```bash
git clone https://github.com/doczeus/nvidia-broadcast-linux.git
cd nvidia-broadcast-linux
./install.sh
```

The installer handles everything:
1. Installs system dependencies (GStreamer, GTK4, v4l2loopback)
2. Configures virtual camera (persists across reboots)
3. Sets up Python environment with GPU-accelerated ML libraries
4. Creates desktop launcher and system tray entry
5. Enables auto-start on login

### Manual Install

```bash
# 1. System dependencies
sudo apt install v4l-utils v4l2loopback-dkms \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gir1.2-gtk-4.0 gir1.2-adw-1 \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    python3-gi python3-gi-cairo

# 2. Virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4

# 3. Python environment
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e .

# 4. For GPU acceleration (CUDA/cuDNN for ONNX Runtime)
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12

# 5. Run
python -m nvbroadcast
```

### Making v4l2loopback Persistent

```bash
# Auto-load on boot
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4' | \
    sudo tee /etc/modprobe.d/nvbroadcast-v4l2loopback.conf
echo "v4l2loopback" | sudo tee /etc/modules-load.d/nvbroadcast-v4l2loopback.conf
```

---

## Usage

### First Run

```bash
nvbroadcast
```

The app will:
1. Auto-detect your webcam and GPU
2. Start streaming to the virtual camera
3. Show a live preview

### Setup Once, Forget Forever

1. **Configure your effects** (background blur, image, quality preset)
2. **Close the window** - app minimizes to background, virtual camera stays active
3. **Open any video app** (Chrome, Zoom, Discord) - select "NVIDIA Broadcast" as camera
4. **On next login** - app starts automatically with your saved settings

### Headless Mode (No GUI)

```bash
# Just the virtual camera, no window
nvbroadcast-vcam

# With specific settings
nvbroadcast-vcam --device /dev/video0 --format yuy2
nvbroadcast-vcam --format i420  # Better Firefox compatibility
```

### Systemd Service

```bash
# Start/stop
systemctl --user start nvbroadcast-vcam
systemctl --user stop nvbroadcast-vcam

# Enable auto-start on login
systemctl --user enable nvbroadcast-vcam
```

### Chrome Users

If Chrome doesn't detect the virtual camera:
1. Go to `chrome://flags`
2. Search for "PipeWire"
3. Disable "PipeWire Camera" flag
4. Restart Chrome

---

## Quality Presets

| Preset | Model | Resolution | Speed | Best For |
|--------|-------|-----------|-------|----------|
| **Performance** | MobileNetV3, ds=0.25 | Fast | ~7ms | Video calls with effects |
| **Balanced** | MobileNetV3, ds=0.5 | Medium | ~10ms | General use |
| **Quality** | ResNet50, ds=0.375 | High | ~12ms | Presentations |
| **Ultra** | ResNet50, ds=0.5 | Highest | ~15ms | Recording/streaming |

Select in the app under Background > Quality dropdown.

---

## Project Structure

```
nvidia-broadcast-linux/
├── src/nvbroadcast/           # Main Python package
│   ├── __init__.py            # Package metadata & attribution
│   ├── __main__.py            # Entry point
│   ├── app.py                 # GTK4/Adwaita application
│   ├── vcam_service.py        # Headless virtual camera service
│   ├── core/
│   │   ├── config.py          # Settings persistence (TOML)
│   │   ├── constants.py       # App constants, NVIDIA colors
│   │   └── gpu.py             # GPU detection via nvidia-smi
│   ├── video/
│   │   ├── pipeline.py        # GStreamer dual-mode pipeline
│   │   ├── effects.py         # RVM background removal/blur/replace
│   │   ├── autoframe.py       # Face tracking + auto-zoom
│   │   └── virtual_camera.py  # v4l2loopback management
│   ├── audio/
│   │   ├── effects.py         # RNNoise denoising
│   │   ├── pipeline.py        # PipeWire audio pipeline
│   │   ├── virtual_mic.py     # Virtual microphone
│   │   └── monitor.py         # Speaker output denoising
│   └── ui/
│       ├── window.py          # Main window (NVIDIA Broadcast layout)
│       ├── video_preview.py   # Live camera preview
│       ├── controls.py        # Effect toggles, sliders, pickers
│       ├── device_selector.py # Device dropdowns
│       └── style.css          # NVIDIA dark theme
├── models/                    # AI models (auto-downloaded)
├── data/                      # Desktop entry, icons
├── configs/                   # v4l2loopback, PipeWire configs
├── scripts/                   # Setup scripts
├── tests/                     # Integration tests
├── install.sh                 # One-command installer
├── LICENSE                    # GPL-3.0
├── pyproject.toml             # Python package config
└── README.md                  # This file
```

---

## Troubleshooting

### Camera not detected by browser
- Make sure the app is running and streaming (camera LED should be on)
- The virtual camera only appears while the app is actively streaming
- Try refreshing the browser's camera list
- Chrome: disable PipeWire Camera flag (see above)

### "Device busy" error
- Another application is using the camera
- Close other video apps, or run: `fuser -k /dev/video0`

### No GPU acceleration
```bash
# Check if CUDA is available
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Install CUDA libraries if missing
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

### v4l2loopback not loaded
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4
```

### App won't start after reboot
```bash
# Re-run installer
./install.sh

# Or manually reload v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="NVIDIA Broadcast" exclusive_caps=1 max_buffers=4
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Keep the original author attribution intact (see LICENSE)
4. Submit a pull request

---

## Credits

**Created by [doczeus](https://github.com/doczeus)** - AI Powered

### Technologies Used
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) - Video matting model
- [ONNX Runtime](https://onnxruntime.ai/) - GPU-accelerated inference
- [MediaPipe](https://mediapipe.dev/) - Face detection
- [RNNoise](https://github.com/xiph/rnnoise) - Audio noise suppression
- [GStreamer](https://gstreamer.freedesktop.org/) - Media pipeline
- [GTK4](https://gtk.org/) / [Libadwaita](https://gnome.pages.gitlab.gnome.org/libadwaita/) - UI framework
- [v4l2loopback](https://github.com/umlaeute/v4l2loopback) - Virtual camera
- [PipeWire](https://pipewire.org/) - Audio routing

### Disclaimer
This project is not affiliated with or endorsed by NVIDIA Corporation.
NVIDIA, NVIDIA Broadcast, CUDA, TensorRT, and Maxine are trademarks of NVIDIA Corporation.

---

## License

**GPL-3.0** - see [LICENSE](LICENSE) file.

Copyright (c) 2026 **doczeus** ([github.com/doczeus](https://github.com/doczeus))

Any redistribution or derivative work must retain the original author attribution.

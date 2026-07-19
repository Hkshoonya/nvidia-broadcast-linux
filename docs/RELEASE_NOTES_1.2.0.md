# NV Broadcast v1.2.0

This release combines community-contributed audio, performance, correctness,
and desktop-integration work with configurable Linux virtual-camera output.

## Highlights

### Stronger noise removal

- Adds DeepFilterNet3 as the neural noise-removal engine.
- Downloads the model from an immutable upstream revision and verifies its
  SHA-256 checksum before use.
- Verifies generated ONNX cache files before reusing them.
- Falls back to RNNoise when the model cannot be downloaded or initialized.

### Lower background CPU use

- Caps ONNX Runtime thread pools for the real-time workloads that do not
  benefit from additional worker threads.
- Uses blocking CUDA synchronization by default to avoid a spinning CPU core.
- Keeps an environment override for systems that need the previous CUDA wait
  behavior.

### Native Linux tray integration

- Adds a StatusNotifierItem tray implementation for modern Linux desktops.
- Keeps the existing legacy tray as an optional fallback.
- Cleans up the D-Bus tray item explicitly during application shutdown.

### Configurable virtual-camera output

- Adds a persisted Linux output-device setting under Camera.
- Uses the selected `/dev/videoN` in the GUI pipeline, headless
  `nvbroadcast-vcam` service, busy-device checks, and recovery instructions.
- Adds `NVBROADCAST_VCAM_DEVICE_NUM` and `NVBROADCAST_VCAM_DEVICE` overrides to
  the Linux setup scripts.
- Keeps `/dev/video10` as the default while allowing `/dev/video11` or another
  loopback node when OBS or other software already uses that device number.
- Leaves macOS virtual-camera selection under the CoreMediaIO extension rather
  than displaying a Linux device-path control.

### Correctness and privacy

- Writes configuration updates atomically to reduce corruption after a crash
  or interrupted save.
- Creates persistent log directories and files with private user permissions.
- Stores first-run AI models in a writable per-user cache on Linux, Snap, and
  macOS, and verifies every downloaded artifact against a pinned SHA-256 hash.
- Aligns voice-effect parameters between the main app and audio helper.
- Pins PyAV to the compatible 16.x line required by the RNNoise stack.
- Requires patched ONNX, Click, and pip versions for new and upgraded installs.

## Upgrade notes

Existing settings are preserved. Linux users who keep the default loopback
device do not need to change anything. To move the virtual camera during a
source install, run:

```bash
NVBROADCAST_VCAM_DEVICE_NUM=11 ./install.sh
```

Then select `/dev/video11` as the Camera output in the app. Close OBS, browsers,
and meeting applications before changing or reloading a v4l2loopback device.

## Contributor credit

DeepFilterNet3, runtime tuning, native SNI tray integration, and the merged
correctness improvements were contributed by Jon Fuller
([@perfectra1n](https://github.com/perfectra1n)).

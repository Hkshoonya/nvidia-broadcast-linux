# Phase 2: Proprietary macOS Virtual Camera (CoreMediaIO Extension)

## Overview

Replace the pyvirtualcam/OBS bridge with a proprietary CoreMediaIO Camera Extension
that appears as "NV Broadcast" in all macOS video apps (Zoom, FaceTime, Chrome, etc.).

## Architecture

```
Python App (NV Broadcast)
  ├── Renders BGRA frames (effects, background removal)
  └── Writes to shared memory buffer
              ↓
Swift Helper (subprocess)
  ├── Creates IOSurface from shared memory
  ├── Sends Mach port to extension via CFNotification
  └── Zero-copy GPU path
              ↓
NV Broadcast Camera Extension (Swift)
  ├── CMIOExtensionProvider → device → stream
  ├── Receives IOSurface (zero-copy CVPixelBuffer)
  └── Delivers frames to all video apps
              ↓
Zoom / FaceTime / Chrome / OBS / Discord
```

## Key Components

### 1. Camera Extension (Swift)

3-tier CoreMediaIO hierarchy:
- `ExtensionProvider` — manages all virtual camera devices
- `DeviceSource` — "NV Broadcast" device properties (name, model, transport)
- `StreamSource` — frame delivery (CVPixelBuffer via IOSurface)

### 2. IPC: Python → Extension

- **IOSurface + Mach Ports** for pixel data (zero-copy GPU memory)
- **CFNotification** for "new frame ready" signals
- **App Groups** (`group.com.doczeus.nvbroadcast`) for shared state

### 3. Swift Helper Shim

Small CLI tool called by Python via subprocess:
- Creates IOSurface from stdin bytes or shared memory
- Serializes Mach port
- Posts CFNotification to wake extension

## Requirements

- macOS 12.3+ (Camera Extensions introduced)
- Xcode 13.1+
- Paid Apple Developer Account ($99/year) for distribution
  - Free account works for local development
  - `com.apple.developer.system-extension.install` entitlement requires paid account
- Code signing + notarization for distribution

## Entitlements

**App:**
- `com.apple.developer.system-extension.install`
- `com.apple.security.application-groups` → `group.com.doczeus.nvbroadcast`

**Extension:**
- `com.apple.security.app-sandbox`
- `com.apple.security.application-groups` → `group.com.doczeus.nvbroadcast`
- `com.apple.security.device.camera`

## File Structure

```
macos/
├── NVBroadcast.xcodeproj
├── NVBroadcastApp/
│   ├── AppDelegate.swift        # Extension installer/activator
│   └── Info.plist
├── NVBroadcastExtension/
│   ├── main.swift               # Entry point
│   ├── ExtensionProvider.swift  # CMIOExtensionProviderSource
│   ├── DeviceSource.swift       # CMIOExtensionDeviceSource
│   ├── StreamSource.swift       # CMIOExtensionStreamSource
│   ├── IOSurfaceReceiver.swift  # Mach port → CVPixelBuffer
│   ├── Info.plist
│   └── Entitlements.plist
├── NVBroadcastHelper/
│   └── main.swift               # Python→Extension bridge CLI
└── Shared/
    └── Constants.swift          # Bundle IDs, app group
```

## Reference Implementations

- [ldenoue/cameraextension](https://github.com/ldenoue/cameraextension) — best practical example
- [Halle/SinkCam](https://github.com/Halle/SinkCam) — sink+source stream pattern
- [WWDC22 Session 10022](https://developer.apple.com/videos/play/wwdc2022/10022/) — official Apple docs
- [The Offcuts blog](https://theoffcuts.org/) — best tutorial series

## Estimated Effort

- Swift extension + helper: 2-3 weeks
- Python IPC integration: 1 week
- Code signing + notarization: 1-2 days
- Testing across macOS versions: 1 week
- Distribution (.dmg / Homebrew cask): 1 week

## Status

- [x] Phase 1: pyvirtualcam bridge (current — works with OBS)
- [ ] Phase 2: Proprietary CoreMediaIO extension (this document)

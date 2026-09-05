# NV Broadcast Contributors

NV Broadcast was originally created and is maintained by
[DocZeus](https://github.com/Hkshoonya). The project also includes accepted
technical work and findings from the external contributors below. This record
is cumulative and ships alongside the in-application credits; accepted
contributors are not removed from later releases merely because a release
contains no new work from them.

The Git history and linked pull requests or commits remain the source of truth.
This page provides a human-readable summary and does not transfer, replace, or
diminish authorship of any contribution.

## Recognized External Contributors

### John Maingi ([@JohnMaingi-IXP](https://github.com/JohnMaingi-IXP))

- Diagnosed the Whisper installer bug that skipped required HTTP and Hub
  dependencies and proposed the two-step installation strategy in
  [#10](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/10). The pull
  request was closed as superseded rather than merged verbatim; its validated
  finding informed the broader fix shipped across all supported install paths
  in `v1.1.9`.

### Jon Fuller ([@perfectra1n](https://github.com/perfectra1n))

- Added correctness and reliability fixes covering dependency pins, atomic
  configuration saves, voice effects, and persistent logs
  ([#23](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/23)).
- Added DeepFilterNet noise removal, ONNX Runtime efficiency improvements, and
  the native SNI tray integration
  ([#24](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/24),
  [#25](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/25),
  [#28](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/28)).
- Added the device-resident GPU frame path, camera and microphone power saving,
  and stronger adjustable background blur controls
  ([#26](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/26),
  [#27](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/27),
  [#29](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/29)).

### Cédric Prezelin ([@Tenshock](https://github.com/Tenshock))

- Improved documentation, live camera switching, profile changes with
  single-frame backends, GTK diagnostics, and consistent logging
  ([#19](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/19),
  [#40](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/40),
  [#41](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/41),
  [#42](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/42),
  [#43](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/43)).
- Raised dependency security floors, updated GitHub Actions, and helped define
  deterministic runtime ownership
  ([#51](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/51),
  [#61](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/61),
  [#62](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/62)).
- Hardened GUI and headless startup when cameras are missing and fixed MJPEG
  capture fallback to raw camera modes
  ([#70](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/70),
  [#76](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/76)).

### Cenkay Çoban ([@pastor0711](https://github.com/pastor0711))

- Made profile auto-start an explicit user choice and made failed starts return
  safely to a stopped state
  ([#52](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/52)).

### KadotyGamer ([@KadotyGamer](https://github.com/KadotyGamer))

- Added Python 3.14 support to the Linux source installer with checks for
  actual GTK4/Libadwaita/GStreamer binding availability
  ([#103](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/103)).

## Other Community Contributions

Bug reports, reproduction details, hardware testing, review comments,
documentation suggestions, and feature discussions also shape NV Broadcast.
Those contributions remain attributed in their public issue, discussion, pull
request, commit, and release-note history. Private identities should not be
published without permission.

Financial supporters are recognized separately in the README and application.
Sponsorship supports maintenance but does not create or replace code authorship.

## Updating This Record

Every accepted external human contribution must update both
`src/nvbroadcast/contributors.py` and this file. Pull-request checks verify that
every cumulative registry entry appears here with the same public name and
GitHub account.

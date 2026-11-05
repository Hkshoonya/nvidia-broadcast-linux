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

## Recognition criteria

- **Core Maintainers** carry ongoing responsibility for the project and its
  maintenance. This is a project role, separate from original authorship.
- **Major Feature Contributors** receive prominent scope credit for substantial
  feature, architecture, security, reliability, or integration work. A verified
  fix can qualify when its technical depth and impact justify that recognition.
- **Contributors** receive concise, factual credit for accepted focused fixes,
  compatibility work, documentation, and validated technical findings.

The scope and verified impact of accepted work determine its display group.
Line counts, pull-request counts, account age, sponsorship, and unmerged
proposals do not determine a group. A display group grants no maintainer or
review permissions. All accepted credits remain cumulative.

Recognition and delivery status are separate: acceptance into the source tree
is not a claim that work was included in an earlier release, and experimental
work retains its stated limits. Maintainers review the contribution evidence
and grouping; opening a pull request or passing CI does not establish acceptance.

## Core Maintainers

### DocZeus ([@Hkshoonya](https://github.com/Hkshoonya))

**Areas:** Project creation; Maintenance.

Original project creator and maintainer.

- Created NV Broadcast and maintains the
  [canonical project repository](https://github.com/Hkshoonya/nvidia-broadcast-linux).

## Major Feature Contributors

### Jon Fuller ([@perfectra1n](https://github.com/perfectra1n))

**Areas:** GPU video; Audio processing; Runtime reliability.

GPU frame transport, audio processing, and runtime reliability.

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

**Areas:** Camera pipelines; Runtime ownership; Dependency security.

Camera recovery, runtime ownership, and dependency security.

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

### Mradul Kumar ([@Nirusaki-Malaal](https://github.com/Nirusaki-Malaal))

**Areas:** Python compatibility; Native-model recovery.

Experimental Python 3.15 CPU tooling and native-model checks.

**Delivery:** Not included in v1.5.4.

These accepted changes are part of the unpublished v1.5.5 preparation; current
package and platform acceptance remain recorded in the
[release notes](docs/RELEASE_NOTES_1.5.5.md).

- Reported the MediaPipe face-model startup kill on Arch Linux and extended
  the isolated native probes to check model initialization before loading it
  in the GUI, with regression tests for failures and timeouts
  ([#180](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/180)).

- Added the experimental Linux x86_64 source-built Python 3.15 CPU route,
  including pinned AV/ONNX Runtime builds and native runtime checks
  ([#185](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/185)).
  GPU support and physical-device acceptance remain separate. Both #180 and
  #185 were accepted outside the frozen v1.5.4 source; this credit does not place them in that
  published release or attribute later maintainer follow-up work to the author.

## Contributors

### John Maingi ([@JohnMaingi-IXP](https://github.com/JohnMaingi-IXP))

**Areas:** Installer diagnosis.

Validated Whisper installer dependency diagnosis.

- Diagnosed the Whisper installer bug that skipped required HTTP and Hub
  dependencies and proposed the two-step installation strategy in
  [#10](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/10). The pull
  request was closed as superseded rather than merged verbatim; its validated
  finding informed the broader fix shipped across all supported install paths
  in `v1.1.9`.

### Cenkay Çoban ([@pastor0711](https://github.com/pastor0711))

**Areas:** Profiles; Startup recovery.

Explicit profile auto-start and safe failed-start recovery.

- Made profile auto-start an explicit user choice and made failed starts return
  safely to a stopped state
  ([#52](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/52)).

### KadotyGamer ([@KadotyGamer](https://github.com/KadotyGamer))

**Areas:** Source installation; Python compatibility.

Python 3.14 source installation and desktop binding checks.

- Added Python 3.14 support to the Linux source installer with checks for
  actual GTK4/Libadwaita/GStreamer binding availability
  ([#103](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/103)).

### Mauricio Zaffari ([@mauriciozaffari](https://github.com/mauriciozaffari))

**Areas:** CUDA compatibility.

CUDA detection for system-provided onnxruntime packages.

- Fixed CUDA runtime detection for distributions that ship
  `CUDAExecutionProvider` under the plain `onnxruntime` name (e.g. Arch's
  `python-onnxruntime-opt-cuda`), which the variant check previously
  misclassified as CPU, blocking CUDA modes with source-installer guidance
  that does not apply to system packages
  ([#156](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/156)).

### JZomDev ([@JZomDev](https://github.com/JZomDev))

**Areas:** Controls; Audio settings.

Prevent accidental slider changes during page scrolling.

- Reported and fixed wheel input changing sliders under the pointer while
  scrolling controls, including accidental noise-strength changes that
  restarted the virtual microphone. Maintainer review refined native page
  propagation and added wheel, modifier, keyboard and drag regression checks
  ([#199](https://github.com/Hkshoonya/nvidia-broadcast-linux/pull/199)).

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
`src/nvbroadcast/contributors.py` and this file. Maintainers review the public
name, account, contribution area, concise scope, evidence links, and display
group. Verified focused fixes and validated findings retain credit; new
submissions are evaluated on their merits before acceptance.

Pull-request checks verify that every cumulative registry entry appears here
with the same public name, GitHub account, scope and evidence. Keep delivery
notes accurate when finalizing a release; a recognition group is not a release
or platform-support guarantee.

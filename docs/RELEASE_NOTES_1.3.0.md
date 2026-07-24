# NV Broadcast v1.3.0

NV Broadcast v1.3.0 is a video-pipeline, background-quality, power-management,
and macOS camera update. It combines the reviewed community work merged after
v1.2.3 with follow-up fixes validated against a real sunlit camera scene and
Linux virtual-camera output.

## Device-resident NVIDIA video path

- Supported Linux GPU modes can move camera-native I420, NV12, YUY2, or MJPEG
  frames into pinned memory and keep color conversion, eligible RVM input,
  blur matte refinement, compositing, and mirroring on the GPU.
- Source, Debian, and RPM CUDA installs can use nvImageCodec and nvJPEG for
  direct GPU MJPEG decode. The amd64 Snap intentionally retains GStreamer's
  CPU MJPEG decoder to avoid adding roughly 256 MB of optional runtime.
- The virtual-camera leg receives YUY2 directly from the CUDA conversion
  kernel instead of performing another full-frame conversion in GStreamer.
- CPU-only stages still use the established mixed path, so replacement,
  relighting, eye contact, beautification, and auto framing preserve their
  existing behavior.
- GPU JPEG decode, CUDA conversion, and the device-resident frame path each
  retain explicit kill switches and staged automatic fallback.

## Stronger background blur

- The Blur slider now spans a perceptual sigma range up to a substantially
  stronger portrait-style maximum.
- Large blur radii are calculated on a quarter-resolution frame and scaled
  back up, producing softer background detail at lower cost.
- New Dim and Desaturate controls let users reduce background distraction
  without changing the foreground subject.
- GPU and CPU implementations use matching behavior, so fallback does not
  unexpectedly change the selected look.

## Safe camera and microphone power saving

- When the window is hidden and no external application is consuming the
  virtual camera, camera capture can pause while the loopback device remains
  available to meeting applications.
- The virtual microphone follows the same policy and keeps its published
  device alive while physical microphone capture is idle.
- Unknown consumer state, monitoring errors, recording, or a visible window
  always vetoes idling. Three consecutive idle decisions are required before
  capture pauses.
- v4l2loopback client-usage events provide the primary Linux camera-consumer
  signal, avoiding unreliable process counting inside sandboxed namespaces.

## Camera and edge corrections

- Apple Silicon camera discovery now uses GStreamer's AVFoundation provider,
  so the physical camera selected in the UI maps to the device GStreamer
  opens. OBS Virtual Camera is excluded from physical input choices.
- Saved numeric macOS camera preferences from v1.2.3 and older migrate to the
  new stable selection model.
- Native macOS camera frames are negotiated before output scaling, avoiding
  failures when a device does not expose the exact requested output mode.
- Replacement foreground cleanup now detects chroma and tone disagreement
  against nearby solid subject pixels. This removes saturated red or magenta
  outlines caused by strongly lit physical-background objects while leaving
  solid hair, skin, glasses, and clothing unchanged.
- Linux direct-GPU virtual-camera caps now declare progressive scan, fixing
  an appsrc/v4l2loopback negotiation failure that previously caused silent
  fallback to the legacy transport.

## Color and compatibility validation

- Tagged v1.2.3 and current main produced byte-identical mattes and replacement
  output through the established path for the same captured frame, confirming
  that the v1.2.3 backlight adaptation remains intact.
- CPU and GPU decoding of the exact same camera JPEG differed by no more than
  0.002 levels in per-channel mean. The complete GPU JPEG-to-YUY2 round trip
  shifted channel means by no more than 0.053 levels out of 255.
- Backlight adaptation remains inference-only. It helps RVM read foreground
  detail but does not alter displayed camera exposure or white balance.
- Linux arm64 stays on `protobuf>=5.29.6` and omits MediaPipe because the
  available MediaPipe wheel requires a Protobuf branch affected by
  CVE-2026-0994. Auto Frame, Eye Contact, Face Relighting, and face-aware Video
  Enhancement are temporarily unavailable there; background effects, virtual
  devices, recording, and meeting tools remain available.
- Linux amd64/arm64 and supported Apple Silicon Python 3.11-3.13 package
  matrices remain release gates. Linux Python 3.11-3.14 unit matrices continue
  to protect compatible fallback behavior.

## Community

- Jon Fuller ([`@perfectra1n`](https://github.com/perfectra1n)) contributed the
  device-resident GPU video path, camera and microphone power saving, and
  stronger background blur controls.
- Cédric Prezelin ([`@Tenshock`](https://github.com/Tenshock)) moved the release
  history into `CHANGELOG.md` and refreshed the README project structure and
  architecture documentation.
- The Apple Silicon camera fix follows the M3 Pro validation and diagnostics
  provided by [`@13v13reddy`](https://github.com/13v13reddy) on Issue #31.

## Local release verification

- Complete test suite: 395 tests and 13 subtests passed.
- Release smoke suite: 198 tests passed, followed by a clean wheel build and
  required desktop-asset inspection.
- Focused video, GPU frame-path, and replacement-edge suite: 77 tests and
  2 subtests passed.
- Native Debian and RPM packages built locally with the expected `1.3.0-1`
  metadata, root ownership, and normalized payload permissions.
- The clean wheel dependency graph had no broken requirements or known
  vulnerabilities. Bandit found no medium-or-higher Python issues, Gitleaks
  found no secrets in repository files, and both release workflows passed
  `actionlint`.
- Live Linux validation confirmed active GPU transport, background replacement,
  progressive virtual-camera negotiation, and removal of the reported red
  sunlit edge.

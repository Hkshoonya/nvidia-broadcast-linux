# NV Broadcast v1.4.0

NV Broadcast v1.4.0 adds global effect controls, an optional Eye Contact
Gaze Lock mode, live camera switching, and stronger recovery around the GPU
background-processing path. It also includes community fixes for profile
switching and desktop diagnostics.

## Rebindable global effect hotkeys

- Linux users can toggle Background, Auto Frame, Eye Contact, Mirror, and
  microphone noise removal while a meeting, browser, or streaming application
  has focus.
- Desktops with the XDG GlobalShortcuts portal use the portal-native path.
  GNOME systems without that portal can use isolated custom keybindings that
  invoke fixed `Gio.Application` actions.
- Duplicate, malformed, and unsafe bindings are rejected. Recording and
  meeting actions are deliberately not exposed as global shortcuts.
- Shortcut settings persist with the user's profile and can be rebound from
  the application.

## Natural and Gaze Lock Eye Contact

- Eye Contact now provides separate Natural and Gaze Lock modes.
- Natural mode retains the existing subtle camera-facing correction.
- Gaze Lock holds small coordinated eye movements closer to the camera target
  while preserving natural binocular spacing.
- Blink fallback, unstable-landmark rejection, head-yaw compensation, and
  smoothed head-pose updates remain active in both modes.

## Live camera and profile changes

- Selecting a different physical camera now updates the active capture
  pipeline without requiring an application restart.
- Profile switching no longer assumes that every video backend can provide
  multiple current frames. Backends that expose only one frame now apply the
  profile without a stale-frame failure.
- GTK image baseline warnings are removed, and runtime messages use a
  consistent NV Broadcast prefix so actionable diagnostics are easier to find.

## Crash-safe background processing

- RVM quality and processing-mode changes no longer mutate a live model while
  its recurrent tensors still use dimensions from the previous quality.
- The application serializes background-backend reloads, creates and warms the
  replacement session, and swaps it in only when the latest requested
  generation is ready.
- Existing GPU sessions are detached before their resources are released,
  preventing old provider memory from remaining live during replacement
  allocation.
- ONNX Runtime CUDA allocation and arena-exhaustion errors now trigger a clean
  CUDA provider rebuild. If that rebuild cannot initialize, background
  processing falls back to CPU instead of leaving the preview permanently on
  the original camera image.
- GPU-resident frame routing is disabled automatically if inference has moved
  to CPU, preserving a valid mixed processing path.

## Cleaner hair and facial-hair edges

- Ultra quality retains the detail-preserving refinement branches used for
  difficult hair, beard, and motion boundaries.
- Replacement cleanup now also detects neutral or bright background spill
  around soft foreground pixels, not only strongly saturated color spill.
- Cleanup remains bounded by nearby solid-foreground evidence so it does not
  recolor the main subject or alter the original camera frame.
- The existing backlight adaptation remains inference-only and does not change
  displayed camera exposure or white balance.

## Low-latency virtual microphone

- Power Save now applies only to camera capture, matching the setting shown in
  the application.
- Microphone capture remains active while the processed virtual microphone is
  published. This prevents Discord and other Pulse/PipeWire clients from
  retaining a multi-second capture buffer when they connect to an idle source.
- The denoiser and voice-effects path remains isolated in its helper process;
  this change only removes the incompatible microphone suspend/resume cycle.
- Protobuf remains at the patched `>=5.29.6,<7` range so source installations
  that expose host GTK packages do not conflict with current OpenTelemetry
  packages.

## Managed runtime packaging

- Strict Snap and installer-owned DEB/RPM environments are no longer modified
  by the optional-runtime GUI. Missing package-managed runtimes now produce a
  direct compatibility message instead of starting an install that cannot
  succeed.
- Writable, user-owned source and macOS virtual environments retain optional
  CUDA, TensorRT, and meeting-runtime installation.
- Runtimes already bundled by a package remain available even when the current
  environment cannot install or update them.
- Source, native package, and Snap dependency sets use one bounded
  `opencv-contrib-python` distribution so separate wheels do not compete for
  ownership of the `cv2` module.
- Snap stages `packaging` and `setuptools` explicitly. The higher-priority
  amd64 CUDA overlay is installed without transitive dependencies and contains
  only its explicit GPU payload, leaving shared Python packages under one
  owner.
- Snap builds remove the build-time `pip` module, metadata, and executables
  from the final runtime. The compressed artifact is inspected for all three
  before it can be uploaded or published.
- Every built Snap now validates active dependency metadata, version bounds,
  required imports, and duplicate core owners before artifact upload.
- Version tags build draft GitHub release artifacts for inspection and do not
  promote the Snap Store automatically. Store review, candidate, and stable
  actions remain explicit manual workflows.

## Community

- Cédric Prezelin
  ([`@Tenshock`](https://github.com/Tenshock)) contributed live camera source
  switching, the single-frame profile fix, GTK warning cleanup, and consistent
  log prefixes through PRs #40-#43.
- The global hotkey and Gaze Lock work follows the user requests tracked in
  Issues #45 and #38.
- The background mode-switch and edge-quality fix follows live RTX camera
  testing and the regression report resolved in PR #49.
- The virtual-microphone resume correction follows the Discord latency report
  tracked in Issue #44.

## Compatibility and release validation

- The complete non-hardware test suite passed with 449 tests and 15 subtests.
- The focused packaging and Snap runtime-validator suite passed 49 tests.
- The focused background, CUDA recovery, and mode-switch suite passed with
  95 tests and 9 subtests.
- Release smoke passed 249 tests, built the wheel, and verified required
  desktop and package assets.
- Linux package CI passed for amd64 and arm64, including Python 3.11, 3.13,
  and 3.14 coverage plus Debian and RPM artifacts.
- Apple Silicon package CI passed for Python 3.11, 3.12, and 3.13 and built
  the macOS package artifact.
- Snapcraft produced strict amd64 and arm64 release-candidate artifacts.
- Bandit reported no medium-or-higher Python findings, and the clean project
  dependency audit reported no known vulnerabilities.
- Live Linux validation on an RTX 5070 confirmed Ultra, Balanced, Performance,
  and Ultra mode transitions; CUDA provider recovery; retained replacement
  output; and improved moving hair and facial-hair boundaries.

## Snap publication gate

The strict Snap declares the application-specific session D-Bus name
`com.doczeus.NVBroadcast` for its GNOME shortcut fallback. The candidate must
not be promoted until the Snap Store grants the one-time installation
declaration and a strict-confinement shortcut test passes. This release gate is
tracked in
[Issue #48](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/48).

A `v1.4.0` tag creates draft GitHub artifacts for inspection but performs no
Snap Store release. Store review upload, candidate testing, and stable
promotion are separate manual actions, so this declaration gate cannot be
bypassed by tagging the release.

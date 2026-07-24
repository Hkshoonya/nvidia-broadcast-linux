# What's New

## v1.3.0 - GPU Video, Power Save, and Blur Quality Update

- **Device-Resident NVIDIA Video Path** - Supported Linux GPU modes can keep camera decode, RVM input, blur matte processing, compositing, mirroring, and YUY2 conversion on the GPU with pinned host transfers and optional nvImageCodec MJPEG decode
- **Fallbacks Remain Safe** - GPU JPEG, CUDA conversion, and device-resident frame failures demote in stages to established GStreamer and CPU paths instead of taking down the virtual camera
- **Stronger Adjustable Blur** - Background blur now reaches a substantially stronger portrait-style maximum, with separate Dim and Desaturate controls and matched GPU/CPU behavior
- **Camera and Microphone Power Save** - Capture pauses only after repeated, reliable idle detection while virtual devices stay available; unknown consumer state, recording, or a visible window prevents idling
- **macOS Camera Selection Fixed** - Supported Apple Silicon Macs now enumerate AVFoundation devices correctly, exclude OBS Virtual Camera from physical inputs, migrate older saved camera indexes, and negotiate native source modes before scaling
- **Cleaner Sunlit Replacement Edges** - Saturated physical-background colors are removed from soft hair and shoulder pixels without recoloring solid foreground detail
- **GPU Virtual Camera Negotiation Fixed** - The direct YUY2 output now declares progressive scan explicitly, preventing v4l2loopback negotiation failure and silent fallback
- **Community Release** - GPU transport, power saving, and blur controls were contributed by Jon Fuller ([`@perfectra1n`](https://github.com/perfectra1n)); README and architecture documentation were reorganized by Cédric Prezelin ([`@Tenshock`](https://github.com/Tenshock))

> If you are on `v1.2.3` or older, update to `v1.3.0`. This is the recommended stable release for lower GPU-mode CPU overhead, stronger blur controls, safer idle power use, corrected Apple Silicon camera selection, and cleaner replacement edges under difficult sunlight.

## v1.2.3 - Backlit Matting Stability Patch

- **Adaptive Backlight Matting** - Quality and Ultra now compensate inside the RVM model input when a scene contains both strong highlights and deep foreground shadows, improving subject-mask stability in difficult sunlight
- **Camera Image Preserved** - The adaptation never changes the original camera pixels, replacement image, or final frame colors; it only helps the segmentation model read shadow detail
- **Balanced Lighting Unchanged** - Uniformly dark, uniformly bright, and normally balanced scenes stay on neutral processing, with smoothed recovery when lighting changes
- **Focused Regression Coverage** - Tests cover backlight classification, temporal recovery, performance-mode bypass, input-frame integrity, and isolation from the user's saved configuration

> If you are on `v1.2.2` or older, update to `v1.2.3`. This is the recommended stable patch for background blur and replacement in strongly backlit scenes, while retaining the binocular Eye Contact fixes from v1.2.2.

## v1.2.2 - Binocular Eye Contact Patch

- **Shared Camera Target** - Eye Contact now derives one normalized gaze correction from both eyes instead of centering and smoothing each iris independently
- **Natural Iris Spacing Preserved** - Separate, slightly outward target positions retain most of the person's measured binocular disparity instead of pulling both eyes toward a cross-eyed result
- **Lower Stable-Tracking Latency** - Coordinated eye movement receives a faster response and fresh per-frame landmark requests when Eye Contact is the only active face effect, while one-eye landmark jumps keep stronger smoothing
- **Safer Edge Cases** - Conservative head-yaw compensation follows the camera target, extreme side gaze tapers progressively, and a blink or unstable eye skips correction for the pair
- **M3 Pro Follow-up** - [`@13v13reddy`](https://github.com/13v13reddy) confirmed that v1.2.1 made Eye Contact visible and near real time, then supplied the convergence test case fixed in this patch on [Issue #31](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/31)

> If you are on `v1.2.1` or older, update to `v1.2.2`. This is the recommended stable patch for natural binocular eye contact, lower coordinated-gaze latency, and the corrected macOS OBS camera path.

## v1.2.1 - Eye Contact Visibility Patch

- **Visible at the Default Intensity** - Eye Contact now visibly moves a moderate gaze toward the camera instead of often producing an imperceptible sub-pixel change
- **Natural, Bounded Eye Warping** - Horizontal and vertical correction is feathered inside the detected eyelid contour, smoothed between frames, and skipped for blinks or unstable landmark offsets
- **Frame Integrity Preserved** - The localized correction changes color pixels only and leaves the frame alpha channel intact for background removal and virtual-camera output
- **Corrected macOS OBS Output Included** - Upgrading from 1.1.13 also picks up the contiguous BGR OBS Virtual Camera path introduced in 1.2.0 for both passthrough and processed video
- **Reporter Credit** - This patch resolves [Issue #31](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/31), reported on an Apple M3 Pro by [`@13v13reddy`](https://github.com/13v13reddy)

> v1.2.1 fixed imperceptible correction at normal intensity. v1.2.2 supersedes it with shared binocular targeting after the reporter's Apple M3 Pro follow-up.

## v1.2.0 — Community Audio, Performance, and Camera Update

- **Configurable Virtual Camera Output** — Linux users can choose and persist the `/dev/videoN` output used by the app, headless service, installer, and recovery commands, avoiding conflicts with OBS and other loopback cameras
- **DeepFilterNet3 Noise Removal** — The new neural denoiser improves difficult microphone cleanup, verifies its pinned model before loading, and falls back to RNNoise on Linux when it is unavailable
- **Lower Runtime CPU Use** — ONNX Runtime thread pools are capped and CUDA waits use blocking synchronization by default, reducing idle CPU load without changing processing modes
- **Native Desktop Tray** — A StatusNotifierItem tray works natively on modern Linux desktops, with explicit shutdown cleanup and the existing legacy fallback where needed
- **Safer Settings, Logs, and Models** — Config saves are atomic, persistent logs use private permissions, first-run AI models use a writable per-user cache with pinned SHA-256 verification, voice effects behave consistently across app and helper paths, and PyAV is pinned to the compatible 16.x line
- **Contributor Release** — These audio, performance, tray, and correctness improvements were contributed by Jon Fuller ([`@perfectra1n`](https://github.com/perfectra1n)); the configurable output-device work completes [Issue #22](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/22)

> If you are on `v1.1.13` or older, update to `v1.2.0`. This is the recommended stable release for stronger noise removal, lower background CPU use, native tray support, and conflict-free virtual-camera output selection.

## v1.1.13 — Auto Frame Tracking and Framing Patch

- **Center Face Tracking Smoothed** — Auto Frame now follows side-to-side face movement continuously instead of waiting for a large lateral jump before the crop catches up
- **Stable Background Mode Added** — The new Auto Frame `Framing` selector lets users choose `Center Face` tracking or `Stable Background` framing when they do not want the room to drift
- **Minimum Zoom Framing Fixed** — Center Face now keeps a small internal crop margin at the minimum zoom setting, so face centering still works when users reduce zoom as far as the UI allows
- **Mode Switching Fixed** — Switching from Stable Background back to Center Face snaps to the next detected face immediately, then resumes normal smoothing
- **Regression Tests Added** — Release checks now cover lateral tracking smoothness, framing mode switching, config persistence, and minimum-zoom auto-frame behavior

> If you are still on `v1.1.12` or older, update to `v1.1.13`. This is the recommended stable patch for smoother Auto Frame tracking and clearer framing behavior.

## v1.1.12 — OBS Startup, Phone Webcam, and Audio Filtering Patch

- **OBS Startup Race Fixed** — The main app now stops the optional headless virtual-camera service before opening the camera, preventing blank preview and busy-camera conflicts after login
- **Phone Webcam Fallback Improved** — Android/phone-as-webcam sources now get safer mode retries, and busy `/dev/video10` is reported clearly before fallback attempts
- **Audio Filtering Dependency Fixed** — RNNoise mic/speaker cleanup now installs its PyAV dependency with the core app, so audio routing does not silently run unfiltered
- **Complete Runtime Bundle Preserved** — Meeting transcription and summaries keep the bundled `faster-whisper`, `ctranslate2`, `soundfile`, and support-package path; `openai-whisper` remains optional and guarded for compatible Python versions
- **Virtual Camera Setup Recovery Improved** — Existing installs update the loopback configuration safely and avoid live reloads while OBS, browsers, or meeting apps are using the camera
- **Regression Tests Expanded** — Release checks now cover GUI/headless ownership, phone webcam fallback, audio-filter dependency imports, Snap/package metadata, and hardware-independent vcam tests

> If you are still on `v1.1.11` or older, update to `v1.1.12`. This is the recommended stable patch for OBS startup reliability, phone webcam compatibility, audio filtering, and package runtime completeness.

## v1.1.11 — OBS Camera Compatibility and Packaging Reliability Patch

- **OBS White Preview Fixed** — The camera pipeline now handles cameras that expose raw video modes instead of MJPEG for the selected resolution
- **Safer Camera Auto-Detection** — Startup now avoids stale, metadata-only, and virtual-loopback camera nodes after reboot, reducing blank preview and “no effects” cases
- **Headless Virtual Camera Fixed Too** — `nvbroadcast-vcam` uses the same camera compatibility path as the main app, so OBS-only workflows get the same fix
- **CUDA Runtime Packaging Fixed** — Source, Debian, RPM, and amd64 Snap installs now use the correct package paths for the CUDA mode runtime
- **Regression Tests Added** — Release checks now cover raw-camera fallback, camera-node filtering, headless virtual camera behavior, and package metadata consistency

> If you are still on `v1.1.10` or older, update to `v1.1.11`. This is the recommended stable patch for OBS camera compatibility and package install reliability.

## v1.1.10 — Live Edge Quality and Compute Control Update

- **Cleaner Live Background Edges** — Background replace is steadier around hair, shoulders, raised hands, and finger gaps during motion
- **Less Edge Cleanup Cost** — The DocZeus/fused CUDA path now uses the same replace-mode foreground cleanup while the CPU fringe cleanup does less full-frame work
- **New Compute Selector** — Users can choose Auto, GPU Focused, or CPU Focused depending on whether they want automatic tuning, CUDA preference, or lower GPU load
- **CUDA Install Path Fixed** — Source, Debian, RPM, and amd64 Snap installs now install the CUDA mode runtime on NVIDIA systems instead of installing only CuPy and leaving ONNX inference on CPU
- **Clearer GPU Runtime Checks** — CUDA modes now require both CuPy compositing and the ONNX Runtime `CUDAExecutionProvider`, so the app no longer treats a partial GPU install as ready
- **Meeting Dependencies Stay Safe** — Packaged installs continue using the lighter `faster-whisper` runtime, and `openai-whisper` remains guarded for Python versions that support it
- **Sponsors Are Visible** — Public GitHub Sponsors now show in the README, dedicated sponsor wall, and About window so users can see who is backing the project

> If you are still on `v1.1.9` or older, update to `v1.1.10`. This is the recommended stable update for live background edge quality, compute-mode clarity, and release packaging safety.

## v1.1.9 — Meeting Runtime Dependency Hotfix

- **Meeting Transcription Install Fixed** — The app now installs `faster-whisper` safely without skipping the support packages required for local transcription
- **Package Installers Fixed Too** — Debian, RPM, and macOS package install paths now use the same corrected meeting runtime recipe as the in-app installer
- **Clearer Recovery Steps** — Missing-transcription messages now show the complete command instead of only `pip install faster-whisper`
- **No OpenAI Whisper Regression** — `openai-whisper` remains optional and guarded on newer Python versions; the default packaged path stays on the lighter `faster-whisper` runtime

> If you are still on `v1.1.8` or older, update to `v1.1.9`. This is the recommended hotfix for meeting transcription dependency reliability.

## v1.1.8 — Audio Helper and Installer Reliability Patch

- **Fixed Echo-Like Mic Doubling** — Old background audio helpers now exit with the app and are cleaned up before a new helper starts, preventing stale helpers from feeding duplicate `nvbroadcast` mic audio into calls
- **CuPy Installer Flow Fixed** — The source installer no longer aborts the whole install when the optional CuPy verification step fails after package installation
- **Clearer Installer Errors** — Install failures now report the real exit code and show useful CuPy verification output for future troubleshooting
- **Sponsor Links Easier to Find** — The About window and README now point users to GitHub Sponsors more clearly

> If you are still on `v1.1.7` or older, update to `v1.1.8`. This is the recommended stable patch for microphone reliability and source-install GPU setup.

## v1.1.7 — Live Edge and Mic Reliability Patch

- **Cleaner Hair and Hand Edges** — Background replace now holds hair edges, finger gaps, and hands near the body more cleanly during motion
- **Better Live Quality Mode Behavior** — Quality profiles now stay closer to the fresher inline matte path, which reduces the “edges follow motion” look that showed up after the earlier lag fixes
- **Less Face-Effect Spill Into Hair** — Beautify and relighting now use tighter face-tone masks, so head hair and side hair look less bright and washed out
- **Stronger Live Processing Path** — The live GPU/background path, shared face-landmark worker flow, and fused face-overlay handling were tuned further for steadier real-time behavior
- **Mic Always Ready** — The exported `nvbroadcast` microphone now stays available even when voice effects and noise removal are turned off
- **Broader Release Verification** — This release was rechecked across video, audio, meeting transcription, summaries, packaging metadata, and release smoke before shipping

> If you are still on `v1.1.6` or older, update to `v1.1.7`. This is the recommended stable patch for current live edge quality and microphone reliability.

## v1.1.6 — Live Background Performance and Stability Patch

- **Fixed Background Reset Loops** — The live alpha path now uses one dedicated worker instead of bouncing inference across short-lived threads, which stops the CUDA invalid-resource-handle failures that could make replace mode collapse into repeated RVM resets
- **Lower Replace-Mode Live Cost** — Relighting now reuses the same-frame final matte instead of rebuilding it, and replace-mode matte work is cached more aggressively on the live path
- **Better Motion Handling Around Face Effects** — Beautify keeps GPU work local to the face ROI and preserves raw denoise history more carefully, reducing motion smear around the face and glasses
- **Safer Heavy Live Stack Behavior** — The app is better at keeping the background path responsive on heavier Meeting-style stacks instead of compounding lag with duplicate work

> If you are still on `v1.1.5` or older, update to `v1.1.6`. This is the recommended stable patch for the recent live background lag regression.

## v1.1.5 — Stability and Live Quality Patch

- **Safer Effect + Mode Switching** — Video pipeline rebuilds now wait for teardown properly, which reduces camera freezes, device-busy failures, and mode-switch crashes
- **TensorRT Detection Fixed** — Zeus and Killer now recognize current `tensorrt-cu12` installs correctly and handle the TensorRT handoff more safely
- **Duplicate Audio Devices Cleaned Up** — Stale `nvbroadcast` mic and speaker duplicates are deduped, and startup restore no longer churns the virtual-audio path as aggressively
- **Better Motion On Face + Glasses** — Beautify denoise is now limited to the face ROI and keeps raw history, which reduces motion smear and disappearing glasses during movement
- **Replace Mode Overlap Improved** — Raised hands near shoulders and underarms are less likely to blow false holes through the background during motion

> If you are still on `v1.1.4` or older, update to `v1.1.5`. This is the recommended stable patch for switch stability, TensorRT detection, and current live-quality fixes.

## v1.1.4 — Audio and Packaging Reliability Patch

- **Browser-Safe Processed Mic** — The Linux `nvbroadcast` microphone path is now stable for Chrome, Discord, Meet, and similar apps instead of hanging or opening with silence
- **Meeting Runtime Fixed** — The optional Whisper runtime installer now validates real imports before claiming success and includes the missing `httpx` dependency
- **Packaged Meeting Runtime** — Release installers now bundle the local meeting transcription runtime more consistently so packaged installs do not depend on the in-app runtime path as often
- **No External `ffmpeg` Requirement For Saved Meetings** — The app now reads its own saved WAV meeting capture directly for the final transcript pass
- **Release Checks Tightened** — Packaging CI is opted into Node 24 early, and release smoke now covers dependency-installer, transcriber, summarizer, meeting-store, and packaging metadata checks

> If you are still on `v1.1.3` or older, update to `v1.1.4`. This is the recommended stable patch for meeting audio reliability and packaged runtime consistency.

## v1.1.3 — Meeting-First Live Quality Update

- **Lower Visible Delay** — Live video now prefers the newest frames instead of letting stale buffers build up, which reduces the odd “lips move late” effect in meetings
- **Auto - Adaptive Mode** — Hardware-aware tuning now persists across restart, warns on very weak devices, and can recommend lighter capture modes when real-time FPS collapses
- **Better Live Face Effects** — Face relighting is now fill-light biased instead of darkening the face, and eye contact is more conservative so it distorts less
- **Meeting-Selectable Mic** — The processed mic is exported as `nvbroadcast`, so Zoom, Meet, Teams, OBS, and similar apps can select it directly
- **Correct Speaker Routing** — Speaker denoise now honors the selected output device instead of whichever sink the system guessed
- **Release + Update Flow** — The app now surfaces platform-aware upgrade targets for GitHub releases, macOS `.pkg` downloads, and Snap installs

> If you are still on `v1.1.2` or older, update to `v1.1.3`. This is the recommended stable build for live meetings, adaptive tuning, and corrected audio routing.

## v1.1.2 — Priority Stability Update

- **Meeting Transcript Quality** — Better chunk cleanup and a stronger final full-audio pass improve saved transcripts and notes
- **No End-Meeting Freeze** — Meeting transcript, notes, and summary finalization now run off the UI thread
- **Persistent Speaker + Profile State** — Speaker selection and the active profile now restore correctly after restart
- **Reset to Defaults** — One-click recovery back to a known-good baseline
- **Mic Test Fixed** — Recording and playback are more reliable, with `30s`, `45s`, and `60s` capture options plus early stop

> If you are still on `v1.1.0` or `v1.1.1`, update to `v1.1.2`. It is the recommended stable patch for meeting quality, persistence, and audio test reliability.

## v1.1.1 — Stability Patch

- **Virtual Camera Stability** — Safer Linux `v4l2loopback` sink startup and retry handling
- **Lower Live Lag** — Shared face landmarks and face-ROI relighting reduce delay in heavier effect stacks
- **Better Replace Edges** — Tighter shoulders, ear-side hair, and under-arm gaps during background replace
- **Meeting Transcription Reliability** — Faster startup, shorter chunking, and cleaner saved meeting audio
- **Resolution Change Safety** — Resolution changes are saved safely and applied after restart instead of hanging the stream

> If you are still on `v1.1.0`, update to `v1.1.1`. It improved virtual-camera behavior, lower lag, and cleaner live compositing.

## v1.1.0 — Meeting Assistant Update

- **Meeting Assistant Sidebar** — Collapsible live transcript and rolling summary inside the app
- **Meeting History** — Local session history stays on-device for 7 days with automatic cleanup
- **Two-Way Meeting Audio** — Meeting capture records both sides for better local notes and transcripts
- **Background Runtime Installs** — Optional CUDA, TensorRT, and meeting runtimes install in the background with progress
- **Improved Setup Guidance** — First-run flow explains modes, downloads, and skip/install choices more clearly

## v1.0.0 — AI Release

- **AI Meeting Transcription** — Local Whisper speech-to-text (tiny/base/small/medium models, GPU-accelerated)
- **AI Meeting Summarizer** — Extracts action items, questions, key points from transcripts (fully local)
- **Voice Effects** — Bass boost, treble, warmth, compression, noise gate, gain (GPU + CPU)
- **6 Voice Presets** — Natural, Radio, Podcast, Deep Voice, Bright, Studio
- **Microphone Selection** — Full PipeWire/PulseAudio device enumeration
- **Speaker Detection** — All output devices via PipeWire
- **Audio Level Monitor** — Real-time VU meter with peak hold
- **Mic Test** — Record 30s / 45s / 60s and play back to test your setup
- **Meeting Mode** — Combined video+audio recording with live transcription and AI summary
- **Recording Fix** — MP4 now includes audio track (NVENC video + AAC audio)
- **Voice FX GPU Acceleration** — CuPy CUDA for warmth/gate/gain, scipy for filters (2.8ms/chunk)

## v0.3.0

- **Eye Contact Correction** — MediaPipe iris tracking redirects your gaze to look at camera
- **Face Relighting** — Fill light guided by the scene
- **Recording Mode** — NVENC hardware encode to MP4 (x264 fallback on non-NVIDIA)
- **Performance Overlay** — Real-time FPS, GPU usage, VRAM, temperature monitoring
- **User Profiles** — 5 built-in (Meeting, Streaming, Presentation, Gaming, Clean) + custom save/load
- **Multi-Camera Support** — Hot-switch between cameras without restarting
- **Apple-Inspired UI** — Glassmorphism cards, collapsible sections, smooth transitions
- **Shared FaceLandmarker** — Single MediaPipe instance shared across all face effects (3x faster)
- **macOS Support** — CPU modes with CoreML, AVFoundation camera, Homebrew installer
- **CI Pipeline** — GitHub Actions builds .deb, .rpm, .pkg + Swift Camera Extension on macOS

## v0.2.0

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

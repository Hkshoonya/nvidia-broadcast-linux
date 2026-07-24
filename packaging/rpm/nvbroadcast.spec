Name:           nvbroadcast
Version:        1.3.0
Release:        1%{?dist}
Summary:        NV Broadcast - Unofficial NVIDIA Broadcast for Linux
License:        GPL-3.0-or-later
URL:            https://github.com/Hkshoonya/nvidia-broadcast-linux
Source0:        %{name}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel

Requires:       python3 >= 3.11
Requires:       python3-gobject
Requires:       python3-gobject-cairo
Requires:       gtk4
Requires:       libadwaita
Requires:       gstreamer1-plugins-base
Requires:       gstreamer1-plugins-good
Requires:       gstreamer1-plugins-bad-free
Requires:       pipewire-utils
Requires:       pulseaudio-utils
Requires:       v4l-utils
Requires:       psmisc

Recommends:     libayatana-appindicator-gtk3

%description
NV Broadcast is an unofficial NVIDIA Broadcast for Linux and other OS.
AI-powered virtual camera with background removal, blur, replacement,
video enhancement, auto-framing, noise cancellation, and AI meeting
transcription using GPU-accelerated deep learning.

Features:
- AI Meeting Transcription (local Whisper, no cloud needed)
- Voice Effects (real-time audio processing)
- Mic Selection (choose input microphone)
- 9 processing modes (Killer, Zeus, DocZeus, CUDA, CPU)
- Fused CUDA kernel compositing (0.1ms at 1080p)
- Edge refinement neural network
- Video enhancement (5 effects + presets)
- Eye contact correction (gaze redirection to camera)
- Face relighting (AI-powered lighting adjustment)
- Session recording (save processed video to file)
- User profiles (save/load per-user settings)
- Resolution selector (360p to 4K)
- System tray integration
- Camera power save

Requires NVIDIA GPU with driver 525+ for GPU acceleration.

%prep
%autosetup -n %{name}-%{version}
chmod 644 LICENSE README.md

%install
# Application
install -d %{buildroot}/opt/nvbroadcast
cp -r src pyproject.toml requirements.txt LICENSE README.md %{buildroot}/opt/nvbroadcast/
install -d %{buildroot}/opt/nvbroadcast/models
cp -r data %{buildroot}/opt/nvbroadcast/
cp -r configs %{buildroot}/opt/nvbroadcast/ 2>/dev/null || true
find %{buildroot}/opt/nvbroadcast -type d -exec chmod 755 {} +
find %{buildroot}/opt/nvbroadcast -type f -exec chmod 644 {} +

# Desktop entry
install -Dm 644 data/com.doczeus.NVBroadcast.desktop \
    %{buildroot}%{_datadir}/applications/com.doczeus.NVBroadcast.desktop

# AppStream metadata
install -Dm 644 data/com.doczeus.NVBroadcast.metainfo.xml \
    %{buildroot}%{_datadir}/metainfo/com.doczeus.NVBroadcast.metainfo.xml

# Icon
install -Dm 644 data/icons/com.doczeus.NVBroadcast.svg \
    %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg

# Launchers
install -d %{buildroot}%{_bindir}
cat > %{buildroot}%{_bindir}/nvbroadcast << 'EOF'
#!/bin/bash
exec /opt/nvbroadcast/.venv/bin/python -m nvbroadcast "$@"
EOF
chmod 755 %{buildroot}%{_bindir}/nvbroadcast

cat > %{buildroot}%{_bindir}/nvbroadcast-vcam << 'EOF'
#!/bin/bash
exec /opt/nvbroadcast/.venv/bin/python -m nvbroadcast.vcam_service "$@"
EOF
chmod 755 %{buildroot}%{_bindir}/nvbroadcast-vcam

# Systemd user service
install -d %{buildroot}%{_userunitdir}
cat > %{buildroot}%{_userunitdir}/nvbroadcast-vcam.service << 'EOF'
[Unit]
Description=NVbroadcast Virtual Camera Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=/usr/bin/nvbroadcast-vcam
Restart=on-failure
RestartSec=3

[Install]
WantedBy=graphical-session.target
EOF

# v4l2loopback config
install -d %{buildroot}/etc/modprobe.d
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVbroadcast" exclusive_caps=1 max_buffers=4' \
    > %{buildroot}/etc/modprobe.d/nvbroadcast-v4l2loopback.conf
install -d %{buildroot}/etc/modules-load.d
echo 'v4l2loopback' > %{buildroot}/etc/modules-load.d/nvbroadcast-v4l2loopback.conf

%post
# Setup Python venv and install pip deps
if [ ! -d /opt/nvbroadcast/.venv ]; then
    python3 -m venv /opt/nvbroadcast/.venv --system-site-packages
fi
/opt/nvbroadcast/.venv/bin/pip install --upgrade \
    "pip>=26.1.2" "setuptools>=83.0.0" wheel -q
/opt/nvbroadcast/.venv/bin/pip install /opt/nvbroadcast -q
/opt/nvbroadcast/.venv/bin/pip install --no-deps faster-whisper -q 2>/dev/null && \
    /opt/nvbroadcast/.venv/bin/pip install ctranslate2 huggingface-hub httpx tokenizers soundfile av tqdm -q 2>/dev/null || true

# Install CUDA mode runtime if NVIDIA GPU present
if command -v nvidia-smi &>/dev/null; then
    /opt/nvbroadcast/.venv/bin/pip install --upgrade "/opt/nvbroadcast[cuda]" -q 2>/dev/null || true
fi

# Load v4l2loopback
if [ -f /etc/modprobe.d/nvbroadcast-v4l2loopback.conf ] && \
   grep -Eq 'card_label="(NVIDIA Broadcast|NVIDIA Broadcast Virtual Camera|NV Broadcast)"' /etc/modprobe.d/nvbroadcast-v4l2loopback.conf; then
    echo 'options v4l2loopback devices=1 video_nr=10 card_label="NVbroadcast" exclusive_caps=1 max_buffers=4' \
        > /etc/modprobe.d/nvbroadcast-v4l2loopback.conf
fi
modprobe v4l2loopback devices=1 video_nr=10 card_label="NVbroadcast" exclusive_caps=1 max_buffers=4 2>/dev/null || true

%preun
pkill -f "nvbroadcast" 2>/dev/null || true

%files
/opt/nvbroadcast/
%{_bindir}/nvbroadcast
%{_bindir}/nvbroadcast-vcam
%{_datadir}/applications/com.doczeus.NVBroadcast.desktop
%{_datadir}/metainfo/com.doczeus.NVBroadcast.metainfo.xml
%{_datadir}/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg
%{_userunitdir}/nvbroadcast-vcam.service
%config(noreplace) /etc/modprobe.d/nvbroadcast-v4l2loopback.conf
%config(noreplace) /etc/modules-load.d/nvbroadcast-v4l2loopback.conf
%license LICENSE
%doc README.md

%changelog
* Fri Jul 24 2026 doczeus <harshit@kshoonya.com> - 1.3.0-1
- Add a device-resident NVIDIA video path with staged safe fallbacks
- Add stronger blur plus separate Dim and Desaturate controls
- Add conservative camera and microphone power saving
- Fix physical-camera selection and capture negotiation on Apple Silicon
- Remove saturated color spill from sunlit replacement edges
- Fix progressive negotiation for direct GPU virtual-camera output
- Refresh release, store, website, and package metadata

* Mon Jul 20 2026 doczeus <harshit@kshoonya.com> - 1.2.3-1
- Stabilize Quality and Ultra mattes in strongly backlit scenes
- Adapt only model input while preserving displayed camera pixels
- Return smoothly to neutral processing in balanced lighting
- Isolate integration tests from persisted user configuration

* Sun Jul 19 2026 doczeus <harshit@kshoonya.com> - 1.2.2-1
- Derive Eye Contact correction from one shared binocular camera target
- Preserve natural iris spacing with an outward target and head-yaw compensation
- Follow coordinated gaze faster while smoothing one-eye landmark noise
- Taper extreme side gaze and skip asymmetric correction during blinks

* Sun Jul 19 2026 doczeus <harshit@kshoonya.com> - 1.2.1-1
- Make eye contact visibly correct moderate gaze at the default intensity
- Correct both horizontal and vertical gaze inside a feathered eyelid contour
- Smooth gaze movement while rejecting blinks and unstable landmark offsets
- Preserve frame alpha while applying the localized eye correction

* Sun Jul 19 2026 doczeus <harshit@kshoonya.com> - 1.2.0-1
- Add DeepFilterNet3 neural noise removal with verified downloads and RNNoise fallback
- Reduce idle CPU use through bounded ONNX Runtime threads and blocking CUDA synchronization
- Add native StatusNotifierItem tray integration with explicit shutdown cleanup
- Make the Linux virtual-camera output device configurable across app and installer paths
- Use atomic config saves, private persistent logs, and consistent voice effects
- Store first-run AI models in a writable per-user cache with pinned SHA-256 verification

* Sat Jul 04 2026 doczeus <harshit@kshoonya.com> - 1.1.13-1
- Smooth Auto Frame lateral tracking so Center Face no longer waits for a large crop jump
- Add Center Face and Stable Background framing modes for explicit auto-frame behavior
- Keep Center Face framing active at minimum zoom with a small internal tracking crop margin
- Make switching from Stable Background back to Center Face take effect on the next detected face
- Add regression coverage for auto-frame tracking, mode switching, config persistence, and minimum zoom

* Tue Jun 30 2026 doczeus <harshit@kshoonya.com> - 1.1.12-1
- Stop the optional headless virtual-camera service before the main app opens the camera
- Improve phone and Android webcam fallback behavior with clearer busy-camera diagnostics
- Install the PyAV audio dependency with the core app so RNNoise filtering initializes reliably
- Keep meeting transcription and summaries on the bundled faster-whisper runtime path
- Improve virtual-camera setup recovery for existing installs without live reloads while the camera is in use

* Tue Jun 23 2026 doczeus <harshit@kshoonya.com> - 1.1.11-1
- Fix OBS and meeting-app white preview cases on cameras that expose raw video modes instead of MJPEG
- Avoid stale, metadata-only, and virtual-loopback camera nodes after reboot or device-order changes
- Apply the same camera compatibility path to the headless virtual camera command
- Fix CUDA runtime package paths for source, Debian, RPM, and amd64 Snap installs
- Add regression coverage for camera-mode fallback, camera-node filtering, headless virtual camera behavior, and package metadata

* Fri Jun 19 2026 doczeus <harshit@kshoonya.com> - 1.1.10-1
- Improve live background replace edge stability around hair, shoulders, hands, and fingers
- Apply replace-mode fringe cleanup in the fused CUDA compositing path and reduce CPU cleanup cost
- Add Auto/GPU Focused/CPU Focused compute controls for clearer performance tradeoffs
- Install the full CUDA mode runtime on NVIDIA systems so ONNX inference does not stay CPU-only
- Keep meeting runtime packaging on faster-whisper and guard openai-whisper on newer Python versions
- Add auto-updating GitHub Sponsors walls and visible sponsor recognition

* Sat May 23 2026 doczeus <harshit@kshoonya.com> - 1.1.9-1
- Fix meeting transcription runtime installation across app, Debian/RPM, and macOS package paths
- Split faster-whisper installation so its support packages keep their required dependencies
- Update missing-backend help text with the complete faster-whisper install command

* Sat May 23 2026 doczeus <harshit@kshoonya.com> - 1.1.8-1
- Stop stale orphaned audio helpers from feeding duplicate nvbroadcast mic audio after app exits
- Fix the source installer CuPy verification flow so optional GPU install checks do not abort incorrectly
- Report real installer exit codes and clearer optional GPU verification output
- Add clearer project sponsorship links in the app and README

* Wed Apr 29 2026 doczeus <harshit@kshoonya.com> - 1.1.7-1
- Improve live background edges around hair, fingers, and hands near the body
- Reduce face-effect spill into head hair so hair looks less bright and washed out
- Keep the exported nvbroadcast microphone live even when voice effects and noise removal are turned off
- Re-verify audio, video, meeting transcription, summaries, and packaging checks before release

* Wed Apr 29 2026 doczeus <harshit@kshoonya.com> - 1.1.6-1
- Fix the live background alpha path so one dedicated worker owns CUDA inference instead of hopping across short-lived threads
- Stop repeated invalid-resource-handle failures and RVM reset loops that could make replace mode extremely laggy
- Reuse same-frame final mattes for relighting and cache more replace-mode work to cut duplicate live processing cost
- Keep beautify GPU work local to the face ROI and preserve raw denoise history more carefully for motion stability

* Fri Apr 24 2026 doczeus <harshit@kshoonya.com> - 1.1.5-1
- Stabilize effect and mode switching to avoid camera device-busy, freeze, and teardown races
- Recognize current TensorRT cu12 package layouts and improve Zeus/Killer TensorRT runtime handoff
- Deduplicate stale nvbroadcast mic and speaker devices and quiet startup audio restore churn
- Scope beautify denoise to the face ROI and preserve raw history to reduce motion smear on face and glasses
- Reduce false replace-mode shoulder and underarm background breakout during raised-hand overlap

* Tue Apr 21 2026 doczeus <harshit@kshoonya.com> - 1.1.4-1
- Stabilize Linux processed-mic routing for browser and meeting app compatibility
- Fix optional meeting runtime validation and include the missing httpx dependency
- Package the local meeting transcription runtime for release installers
- Remove saved-meeting final-pass dependence on an external ffmpeg binary
- Opt GitHub Actions packaging workflows into Node 24 ahead of runner migration

* Fri Apr 17 2026 doczeus <harshit@kshoonya.com> - 1.1.3-1
- Improve live meeting quality with adaptive mode and safer low-FPS fallback
- Reduce perceived lip-sync lag by preferring fresh frames over stale buffered video
- Make relighting fill-light biased and soften eye contact defaults for live calls
- Persist voice FX, output format, and adaptive capture settings across restart
- Export the processed meeting mic as nvbroadcast and honor the selected speaker for denoise
- Improve update surfacing and package metadata for GitHub releases, macOS pkg, and Snap users

* Thu Apr 02 2026 doczeus <harshit@kshoonya.com> - 1.1.2-1
- Priority stability update for meeting transcription and settings persistence
- Improve final transcript quality and move meeting finalization off the UI thread
- Persist speaker selection and active profile state with reset-to-defaults support
- Fix microphone test record/playback reliability and extend test durations to 30s/45s/60s

* Sat Mar 28 2026 doczeus <harshit@kshoonya.com> - 1.1.1-1
- Stabilize the virtual camera sink path on Linux loopback devices
- Reduce live face-effect latency with shared landmark reuse and ROI relighting
- Tighten replace edges around shoulders, hair, and under-arm gaps
- Improve local meeting transcription startup, chunking, and audio finalization
- Save resolution changes safely without hanging the live stream

* Fri Mar 27 2026 doczeus <harshit@kshoonya.com> - 1.1.0-1
- Add meeting assistant sidebar with live transcript and local session history
- Capture both-way meeting audio for on-device transcription and notes
- Keep local meeting records for 7 days with automatic cleanup
- Add background optional-runtime installer flow for CUDA, TensorRT, and Whisper
- Improve first-run setup guidance and in-app install progress handling

* Fri Mar 27 2026 doczeus <harshit@kshoonya.com> - 1.0.2-1
- Improve background matte quality and mode restore behavior
- Package desktop assets and AppStream metadata in release artifacts
- Stop using editable installs in packaged environments
- Preserve system virtual camera configuration during uninstall

* Mon Mar 23 2026 doczeus <harshit@kshoonya.com> - 1.0.0-1
- AI Meeting Transcription (local Whisper, no cloud)
- Voice Effects (real-time audio processing)
- Mic Selection (choose input microphone)
- Complete audio system overhaul
- Recording pipeline fixes

* Mon Mar 23 2026 doczeus <harshit@kshoonya.com> - 0.3.0-1
- Eye contact correction (gaze redirection)
- Face relighting (AI lighting adjustment)
- Session recording (save processed video)
- User profiles (per-user settings)

* Mon Mar 23 2026 doczeus <harshit@kshoonya.com> - 0.2.0-1
- Premium GPU modes (Killer, Zeus, DocZeus)
- Fused CUDA kernel compositing
- Edge refinement neural network
- Video enhancement (5 effects)
- Resolution/FPS selector
- System tray + camera power save
- Firefox auto-configuration

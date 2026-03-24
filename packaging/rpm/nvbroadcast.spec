Name:           nvbroadcast
Version:        1.0.0
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

%install
# Application
install -d %{buildroot}/opt/nvbroadcast
cp -r src pyproject.toml requirements.txt LICENSE README.md %{buildroot}/opt/nvbroadcast/
install -d %{buildroot}/opt/nvbroadcast/models
cp -r data %{buildroot}/opt/nvbroadcast/
cp -r configs %{buildroot}/opt/nvbroadcast/ 2>/dev/null || true

# Desktop entry
install -Dm 644 data/com.doczeus.NVBroadcast.desktop \
    %{buildroot}%{_datadir}/applications/com.doczeus.NVBroadcast.desktop

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
Description=NV Broadcast Virtual Camera Service
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
echo 'options v4l2loopback devices=1 video_nr=10 card_label="NV Broadcast" exclusive_caps=1 max_buffers=4' \
    > %{buildroot}/etc/modprobe.d/nvbroadcast-v4l2loopback.conf
install -d %{buildroot}/etc/modules-load.d
echo 'v4l2loopback' > %{buildroot}/etc/modules-load.d/nvbroadcast-v4l2loopback.conf

%post
# Setup Python venv and install pip deps
if [ ! -d /opt/nvbroadcast/.venv ]; then
    python3 -m venv /opt/nvbroadcast/.venv --system-site-packages
fi
/opt/nvbroadcast/.venv/bin/pip install --upgrade pip -q
/opt/nvbroadcast/.venv/bin/pip install -e /opt/nvbroadcast -q

# Install CuPy if NVIDIA GPU present
if command -v nvidia-smi &>/dev/null; then
    /opt/nvbroadcast/.venv/bin/pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 -q 2>/dev/null || true
fi

# Load v4l2loopback
modprobe v4l2loopback devices=1 video_nr=10 card_label="NV Broadcast" exclusive_caps=1 max_buffers=4 2>/dev/null || true

%preun
pkill -f "nvbroadcast" 2>/dev/null || true

%files
/opt/nvbroadcast/
%{_bindir}/nvbroadcast
%{_bindir}/nvbroadcast-vcam
%{_datadir}/applications/com.doczeus.NVBroadcast.desktop
%{_datadir}/icons/hicolor/scalable/apps/com.doczeus.NVBroadcast.svg
%{_userunitdir}/nvbroadcast-vcam.service
%config(noreplace) /etc/modprobe.d/nvbroadcast-v4l2loopback.conf
%config(noreplace) /etc/modules-load.d/nvbroadcast-v4l2loopback.conf
%license LICENSE
%doc README.md

%changelog
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

* Sun Mar 23 2026 doczeus <harshit@kshoonya.com> - 0.2.0-1
- Premium GPU modes (Killer, Zeus, DocZeus)
- Fused CUDA kernel compositing
- Edge refinement neural network
- Video enhancement (5 effects)
- Resolution/FPS selector
- System tray + camera power save
- Firefox auto-configuration

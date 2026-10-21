# Optional HUSH noise removal

This downstream integration uses [HUSH](https://github.com/UMCEKO/hush)'s
MIT-licensed Rust Maxine engine, through the
[fixtse worker fork](https://github.com/fixtse/hush/tree/feature/stdio-worker).
NV Broadcast remains GPL-3.0-or-later, originally created by DocZeus.

Requires Linux x86_64, PipeWire for the existing microphone routing, an NVIDIA
Turing-or-newer supported GPU with proprietary drivers, and NVIDIA Maxine AFX
2.1.0 plus a matching 48 kHz model. CPU and macOS users should keep Automatic
or RNNoise. HUSH and NVIDIA's runtime are optional and are not bundled in our
packages. NVIDIA's runtime/model terms still apply; use HUSH's setup UI to
review those terms and provision them, or provide your own SDK/model.

## Reusable installer (recommended)

Clone the NV Broadcast fork and run the installer as your normal desktop user:

```sh
git clone --branch feature/hush-denoiser https://github.com/fixtse/nvidia-broadcast-linux.git
cd nvidia-broadcast-linux
python3 scripts/install_hush_fork.py --stop-hush --runtime auto
```

First install Rust with edition 2024 support, Git, a C compiler, pkg-config,
PipeWire development headers, and a desktop-capable Python 3.11+ with GTK4,
Libadwaita and GStreamer bindings. On Arch/CachyOS, the worker build requires
`base-devel rust git pkgconf pipewire`. The existing `install.sh` handles the
app's distro dependencies and may request sudo and optional runtime choices.
Run HUSH setup first to accept NVIDIA's terms and install the SDK and matching
model. This installer reuses those files and verifies the automatically
selected cached model against the pinned HUSH checksum; it does not download
NVIDIA files or accept terms on your behalf. Explicit `--model` files are
trusted user inputs. Use `--sdk`, `--model`, or `--gpu-index` for other locations
or a different GPU. The default is GPU index 0.

For an existing source installation in the **same checkout**, avoid reinstalling
its dependencies with:

```sh
python3 scripts/install_hush_fork.py --reuse-runtime --stop-hush
```

`--reuse-runtime` verifies the selected generation and upgrades the app package
in place with `pip --no-deps`; this opt-in mode does not provide an app-package
rollback. The default runs the normal generation-based source installer.
Stop NV Broadcast and its audio/camera helpers before either command.
An existing native/Flatpak/Snap installation is not treated as a reusable source
runtime; install the fork from this checkout instead.

The script builds the pinned HUSH fork, smoke-tests the real GPU worker, installs
`~/.local/bin/hush-worker` and `~/.local/bin/nvbroadcast-hush`, adds an app-menu
entry **NV Broadcast (HUSH fork)**, backs up the NV Broadcast config, and enables
HUSH microphone noise removal. It refuses malformed existing TOML rather than
silently resetting settings. The dedicated app launcher sets the worker path
explicitly, so it also works when a desktop session omits `~/.local/bin` from PATH.
The source checkout and the HUSH SDK/model paths must remain in place.

`--stop-hush` stops your existing `hush`/`hushd` processes and disables known
HUSH user services after the worker has passed its smoke test. It leaves the
installed HUSH package and its downloaded files intact. Do not run the old HUSH
GUI/daemon concurrently when using this integration. Removing the old package
is unnecessary; retaining it lets you use its setup UI for future models.
The worker itself is the forked HUSH component; its GUI is not rebuilt.

Launch `~/.local/bin/nvbroadcast-hush`, then select the **NVbroadcast** microphone
in calls. To return to the standalone HUSH package, close NV Broadcast, launch
`hush`, and restore any previously enabled HUSH service as appropriate. To undo
the app setting change, restore the timestamped `config.toml.pre-hush-*` backup
under `~/.config/nvbroadcast` (or your XDG config directory).

## Build the pinned worker

Install Rust (edition 2024 support), a C compiler, pkg-config and PipeWire
and OpenSSL development headers, then:

```sh
git clone https://github.com/fixtse/hush.git
cd hush
git checkout c8a2a171ddcd702568f604170a315c17de47fff3
export NVAFX_SDK=/absolute/path/to/Audio_Effects_SDK
cargo build --locked --release -p hush-engine --bin hush-worker
install -Dm755 target/release/hush-worker "$HOME/.local/bin/hush-worker"
```

The SDK root must contain `nvafx/lib`, `external/cuda/lib`, and
`features/denoiser/lib`. HUSH-managed installs normally put it under
`${XDG_DATA_HOME:-$HOME/.local/share}/hush/sdk/2.1.0`.

## Run

Launch NV Broadcast from a shell with these variables (use actual absolute
paths for your installation):

```sh
export NVAFX_SDK=/absolute/path/to/Audio_Effects_SDK
export NVAFX_MODEL=/absolute/path/to/matching/denoiser_v2_48k.trtpkg
export LD_LIBRARY_PATH="$NVAFX_SDK/nvafx/lib:$NVAFX_SDK/external/cuda/lib:$NVAFX_SDK/features/denoiser/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVBROADCAST_HUSH_WORKER="$HOME/.local/bin/hush-worker"
# Optional: export CUDA_VISIBLE_DEVICES=GPU-... to select a GPU by UUID.
python -m nvbroadcast
```

Select **Microphone → Noise engine → HUSH / NVIDIA Maxine (optional)** and
enable Noise Removal. Strength is sent to Maxine for each frame. Select the
usual NVbroadcast microphone in your calling application; do not run `hushd`
for this integration. Voice effects and recording still use the existing
NV Broadcast pipeline. The worker inherits its GPU selection from the
launch environment, independently of NV Broadcast's video GPU selector.

Missing prerequisites log a contextual message and fall back to DeepFilterNet
or RNNoise. A worker failure during streaming falls back to the already
initialized RNNoise engine, or passes audio through if that is unavailable.
Reselect another engine and then HUSH to retry. Worker startup is bounded to
30 seconds; per-frame I/O is bounded to 250 ms, after which it is retired.
A failure can cause a brief discontinuity; uninterrupted sample alignment
across engine failures is not guaranteed.

## Worker protocol

The worker writes `HSH1` and a little-endian u32 frame length after model load.
Each request contains a little-endian f32 intensity and one frame of mono
f32LE audio at 48 kHz. The response is one equally sized f32LE frame.
Diagnostics go to stderr. NV Broadcast reframes arbitrary capture blocks,
adds one frame of buffering, validates finite output, and terminates the
worker during cleanup. The worker creates no PipeWire devices and never
downloads SDKs or models.

## Validation

The integration was checked with 44 focused Python tests covering worker IPC,
reframing, cleanup, timeout/fallback, configuration, helper service, and audio
pipeline behavior. The Rust worker passed `cargo check --locked` and a native
build against an installed Maxine SDK. An RTX 4090 smoke test processed two
seconds of synthetic 48 kHz audio in 0.137 seconds with finite, correctly sized
output. This checks execution, not perceptual denoising quality. Broader app
and model tests could not complete in the checkout's Python environment because
OpenCV (`cv2`) and ONNX were absent.

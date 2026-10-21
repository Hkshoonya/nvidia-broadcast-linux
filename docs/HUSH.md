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

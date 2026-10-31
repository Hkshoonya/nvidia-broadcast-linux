# Experimental Python 3.15 source installation

Issue [#184](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/184)
tracks installing with Fedora 45's system Python and its matching desktop
bindings. Regular CPython 3.15 on Linux x86_64 can opt into a **source-built CPU
runtime**. This is experimental: upstream PyPI wheels are still missing for
the pinned ONNX Runtime and AV versions. The normal installer continues to
select Python 3.11–3.14.

CUDA, CuPy, TensorRT, free-threaded Python (3.15t), other architectures, and
macOS are outside this experimental path. No packaged runtime or release
support claim changes. Updating AV to 17+ is not a workaround: `pyrnnoise`'s
`audiolab` dependency imports `av.option`, which AV 17 removed.

## Build the missing wheels

Use the same Python, distro, architecture, and FFmpeg libraries for building
and installing. These are local wheels; they do not bundle FFmpeg and are not
portable release artifacts. Keep the FFmpeg shared libraries installed.

Required tools: Git, CMake 3.28+, Make, a C/C++ compiler, pkg-config,
Python 3.15 development headers and venv/ensurepip, and **FFmpeg 8 development
headers** (`libavcodec` 62, plus libavformat, libavdevice, libavutil,
libavfilter, libswresample, and libswscale). Check your enabled distro
repositories for the development packages. On Fedora these include
`python3-devel`, `gcc-c++`, `cmake`, `make`, `git`, and `pkgconf-pkg-config`.
Check `pkg-config --modversion libavcodec` returns `62.x`; FFmpeg 9 headers
are incompatible with AV 16. Fedora 45's
[`ffmpeg-free-devel`](https://packages.fedoraproject.org/pkgs/ffmpeg/ffmpeg-free-devel/)
currently supplies FFmpeg 9, so its package alone is insufficient. Build a
private FFmpeg 8 alongside the system libraries in that case:

```bash
mkdir -p build
curl --fail --location https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz \
  --output build/ffmpeg-8.1.3.tar.xz
# Verify the release signature using the official instructions linked below.
tar -xJf build/ffmpeg-8.1.3.tar.xz -C build
FFMPEG_PREFIX="$PWD/build/ffmpeg8"
(cd build/ffmpeg-8.1.3 && ./configure --prefix="$FFMPEG_PREFIX" \
  --enable-shared --disable-static --disable-programs --disable-doc \
  --disable-x86asm && make -j2 && make install)
export PKG_CONFIG_PATH="$FFMPEG_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$FFMPEG_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

[FFmpeg's release verification instructions](https://ffmpeg.org/download.html#releases)
describe its signing keys and detached signatures. Keep this private prefix
and export `LD_LIBRARY_PATH` when installing and running Broadcast. Launch
from this terminal; desktop launches also need that environment. The system
FFmpeg installation stays intact. This extra native-library setup is one
reason Python 3.15 remains experimental.

```bash
python3.15 scripts/python315_runtime.py build --jobs 2
```

This creates an isolated build venv under `build/python315`, builds the
hash-verified AV 16.1.0 source archive and ONNX Runtime 1.24.4 at commit
`2d924974ef147392ced8409d36bd6d2e7fcc8a74`, and writes wheels under
`build/python315-wheels`. Existing dependency pins stay intact. It checks
native imports, `av.option`, and CPU provider availability after building.
It also runs a small ONNX model to check actual CPU inference.
Compilation can take a long time and needs several GB of free disk and RAM;
`--jobs 2` limits memory use. Build dependencies and CMake's upstream source
dependencies require network access. Failed builds never activate a runtime.

To keep build files elsewhere, pass `--build-dir /path/to/build` and
`--wheelhouse /path/to/wheels`. Reusing the build directory resumes the
ONNX Runtime build; an existing source checkout must match the pinned commit
and have no tracked changes.

## Install with the matching desktop bindings

First ensure that **this exact interpreter** imports the distro desktop
bindings; a different Python's PyGObject extension cannot be reused:

```bash
python3.15 -I -c 'import gi; gi.require_version("Gtk", "4.0"); gi.require_version("Adw", "1"); gi.require_version("Gst", "1.0"); from gi.repository import Gtk, Adw, Gst'
./install.sh --python /usr/bin/python3.15 --experimental-python315 \
  --wheelhouse "$PWD/build/python315-wheels" --runtime cpu
```

Stop Broadcast and its services before running the installer. Both the
experimental flag and a wheelhouse are required for Python 3.15. The installer
checks wheel metadata before changing system packages, installs the exact
local wheels into a new candidate environment, then runs its existing import,
dependency-closure, and actual CPU inference checks before activation. A
failed candidate leaves the active runtime unchanged. With Python 3.15,
`--runtime auto` selects CPU and `--runtime cuda` fails explicitly.

The build helper alone does not qualify the GUI, camera, microphone, or model
inference. After installation, launch Broadcast and test those on the target
system. The optional `--with-meeting` uses faster-whisper; openai-whisper
remains excluded on Python 3.14 and newer by its existing dependency marker.

To return to a previously verified generation:

```bash
./install.sh --rollback-runtime
```

## Moving to normal support

Before removing the opt-in, verify wheels for the complete selected dependency
closure on Python 3.15, preserve the AV/RNNoise interface and ONNX Runtime's
provider ABI, and qualify import, real model inference, GTK/GStreamer, audio,
and camera flows on Fedora 45. CUDA needs separate ONNX Runtime GPU, CuPy,
and nvimgcodec validation. See also the upstream
[ONNX Runtime wheel request](https://github.com/microsoft/onnxruntime/issues/33271).

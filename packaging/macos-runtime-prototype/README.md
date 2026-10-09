# macOS arm64 offline Python runtime candidate

This is an unsigned, opt-in issue #60 experiment. It does not alter the normal
source PKG, signing workflow, stable release, or installed native launcher.
The experimental identifier is `com.doczeus.nvbroadcast.offline-candidate`.

The candidate keeps the tested **Homebrew CPython 3.13, PyGObject, Pycairo,
GTK4, libadwaita and GStreamer** contract. It replaces destination source builds,
pip index queries and dependency resolution with prebuilt wheels selected and
hashed at build time. Homebrew provisioning is still an online prerequisite;
this is not yet a self-contained macOS runtime. Model weights are not included.

## Trust and build boundaries

Run `build.py --directory /absolute/new/output` on an ordinary-user Apple Silicon
macOS runner, using `/opt/homebrew/opt/python@3.13/bin/python3.13`. Start from a
clean committed checkout and provide the native prerequisites first:

```sh
/opt/homebrew/bin/brew install python@3.13 pygobject3 gtk4 libadwaita gstreamer
/opt/homebrew/opt/python@3.13/bin/python3.13 packaging/macos-runtime-prototype/build.py \
  --directory "$PWD/dist/macos-offline-candidate"
```

`inputs.json` pins the resolver and builder wheels by exact supplier URL and
SHA-256. The build uses the committed application source and canonical
`cpu,meeting-support` extras, plus the managed faster-whisper version. It resolves
only on the build host, targeting arm64 CPython 3.13 and a macOS 13 wheel baseline.
The generated PEP 751 lock retains supplier URLs and wheel hashes. An exact
runner-compatible wheel selection becomes `manifest.json` and hash-checked
`requirements.txt` in the package. No sdists enter the runtime. Keep and review
these outputs before treating a particular candidate as reproducible; this
initial experiment intentionally refreshes the dependency lock during builds.

The native probe imports actual GI/Pycairo extensions using the selected Python,
checks their resolved Homebrew origins, and records Python, binding hashes,
Homebrew installed versions and GStreamer plugin versions/paths. Homebrew ABI
changes fail the probe instead of silently choosing a different Python version.
Build-host inventory and destination inventory are separately retained. The
wheel deployment baseline does not prove that current Homebrew bottles run on
macOS 13; qualification applies only to the exact runner OS recorded in evidence.

## Separate installation and offline verification

The root Installer writes only `/opt/nvbroadcast-offline-candidate` and
`/usr/local/bin/nvbroadcast-offline-candidate`. It reuses the shipping installer's
root ownership, permissions, ACL and symlink checks. It never executes Homebrew
or user Python as root. Run setup explicitly without sudo:

```sh
/opt/nvbroadcast-offline-candidate/setup.sh
/usr/local/bin/nvbroadcast-offline-candidate --help
```

Each exact package payload gets its own SHA-256 identity and per-user directory
under `~/Library/Application Support/NVBroadcast Offline Candidate`. Failed or
previous runtimes are not replaced. Candidate app config/cache paths are separate.

Before installing anything, setup verifies every wheel's metadata and SHA-256,
rejects duplicate/extra/missing wheels, requires exactly the CPU ONNX Runtime
owner, and verifies the complete package identity. Pip runs with `--isolated
--no-index --no-deps --require-hashes --only-binary=:all: --ignore-installed`.
All selected wheels must be installed in the private venv at their exact versions;
Homebrew supplies the native bindings through the existing system-site ABI.
The ready marker is written only after dependency closure, ownership, native
imports, real CPU inference and generated H.264/AAC recording/decode pass.

The dedicated workflow performs a fresh install and repeat setup under macOS
`sandbox-exec` with `deny network*`, first proving local/external TCP attempts
fail with a permission error. No signing secrets, production environment,
release upload or default package modification is involved. Artifact evidence
includes the exact source/package identity, selected wheel manifest, full
resolver lock, native inventory, generated-media results and install logs.
PR triggers cover prototype/runtime packaging inputs. Run the manual workflow
on the final candidate revision when unrelated application changes also need
qualification; an older green prototype run does not qualify newer source.

## Qualification limits

Portable unit tests cover tampered/incomplete inventories, duplicate/CUDA owners,
requirement injection, root refusal, failed installation and preservation of old
runtimes. They do not establish that Mach-O wheels or native media imports work.
Only the actual macOS runner can establish the automated runtime/media result.
Physical camera/microphone permissions, OBS virtual output, live effects/FPS,
meeting model acquisition and long-running behavior require later Mac checks.
This candidate is not signed/notarized or a default release replacement.

Primary supplier references consulted 2026-10-09:

- [uv platform resolution and macOS deployment target](https://docs.astral.sh/uv/reference/cli/)
- [uv supported platforms](https://docs.astral.sh/uv/reference/policies/platforms/)
- [pip hash-checked installs](https://pip.pypa.io/en/stable/topics/secure-installs/)
- [pip offline and no-dependency installation options](https://pip.pypa.io/en/stable/cli/pip_install/)
- [Homebrew PyGObject formula and Python ABI dependencies](https://formulae.brew.sh/formula/pygobject3)
- [Pinned uv release](https://github.com/astral-sh/uv/releases/tag/0.11.7)

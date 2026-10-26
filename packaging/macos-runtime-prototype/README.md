# macOS arm64 offline Python runtime candidate

This is an unsigned, opt-in issue #60 experiment. It does not alter the normal
source PKG, signing workflow, stable release, or installed native launcher.
The experimental identifier is `com.doczeus.nvbroadcast.offline-candidate`.
This candidate requires **macOS 14 (Sonoma) or newer**. The canonical PyAV 16
arm64 CPython 3.13 supplier wheels require macOS 14; a real runner resolution for
macOS 13 failed for that reason. The shipping package's support policy is unchanged.

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
  --directory "$PWD/dist/macos-offline-candidate" --cache "$PWD/dist/macos-offline-inputs"
```

`inputs.json` pins the builder wheels and the SHA-256 of the committed
`pylock.macos-arm64-cp313-cpu.toml`. This supplier-only lock selects exactly one
hashed wheel per external distribution for arm64 CPython 3.13 and the macOS 14
wheel baseline. Its initial 54 supplier selections come from the retained
[qualified PR #148 build](https://github.com/Hkshoonya/nvidia-broadcast-linux/actions/runs/37985568418);
the originating resolver-lock hash and source revision are recorded in `inputs.json`.

Normal builds never invoke the resolver. They rebuild the application wheel from
committed source using the pinned builder tools, fetch only exact reviewed wheel
URLs, and verify filenames, platform/ABI tags, hashes, metadata, managed
faster-whisper and bootstrap versions, and the complete dependency closure before
`pkgbuild`. The app's `cpu,meeting-support` extras and transitive dependency extras
are evaluated against the declared Mac target; changed app requirements fail if
the reviewed supplier set no longer satisfies them. No sdists enter the runtime.
The generated full package lock includes the newly built application hash rather
than retaining an older application's same-version wheel.

Use `--offline --cache /absolute/cache` to consume existing authenticated inputs
without downloads. Missing or changed inputs fail instead of accessing an index.
The hosted workflow repeats the complete build with networking denied and
compares every candidate payload file's bytes and mode, including the launcher,
application wheel, supplier inventory and Installer scripts. This verifies fixed Python input
consumption on the same native host; it does not establish independent Homebrew
reproducibility or byte-identical PKG archives.

Dependency refresh is a separate explicit operation:

```sh
/opt/homebrew/opt/python@3.13/bin/python3.13 packaging/macos-runtime-prototype/build.py \
  --refresh-lock --directory "$PWD/dist/macos-lock-proposal" \
  --cache "$PWD/dist/macos-offline-inputs"
```

This uses the pinned resolver, verifies the proposed wheelhouse and current-source
closure, and writes `supplier-lock.toml`, `supplier-selection.json`,
`supplier-closure.json`, `resolved-pylock.toml` and `supplier-lock-update.json`.
It does not build a package or edit the reviewed inputs. Review the supplier
changes, then copy the proposed lock to the checked-in lock path and update its
SHA-256/provenance in `inputs.json` in the same PR. If builder versions change,
review and update the bootstrap pins as part of that change. Keep the new runner
qualification evidence before treating the updated inputs as qualified.

The native probe imports actual GI/Pycairo extensions using the selected Python,
checks their resolved Homebrew origins, and records Python, binding hashes,
Homebrew installed versions and GStreamer plugin versions/paths. Homebrew ABI
changes fail the probe instead of silently choosing a different Python version.
Build-host inventory and destination inventory are separately retained. The
wheel deployment baseline does not prove that current Homebrew bottles run on
every supported OS; qualification applies only to the exact runner OS recorded
in evidence. Selection uses the declared floor, not the builder's newer OS tags.

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
--no-index --no-deps --require-hashes --only-binary=:all: --force-reinstall`.
Reinstalling removes stale local ensurepip metadata while pip's venv boundary
prevents removing global Homebrew distributions.
All selected wheels must be installed in the private venv at their exact versions;
Homebrew supplies the native bindings through the existing system-site ABI.
The ready marker is written only after dependency closure, ownership, native
imports, real CPU inference and generated H.264/AAC recording/decode pass.

The dedicated workflow performs a fresh install and repeat setup under macOS
`sandbox-exec` with `deny network*`, first proving local/external TCP attempts
fail with a permission error. No signing secrets, production environment,
release upload or default package modification is involved. Artifact evidence
includes the exact source/package identity, selected wheel manifest, full
package lock, supplier lock/selection/closure and rebuild receipt, native
inventory, generated-media results and install logs.
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
- [PyAV 16.1.0 supplier wheel filenames and hashes](https://pypi.org/pypi/av/16.1.0/json)

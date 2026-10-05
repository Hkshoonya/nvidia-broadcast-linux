# Complete CPU/CUDA package-switching experiment

This extends the [CPU lifecycle experiment](README.md) for #53, #60 and the
package-model discussion in #77. Both self-contained and split candidates use
real private Python/ML payloads. These are unsigned development artifacts,
not replacements for the production DEB/RPM or a v1.5.3 release payload.

## Dependency ownership

The CUDA target uses Python 3.13.16, ONNX Runtime GPU 1.24.4, CuPy 14.2.0 and
CUDA 12 component wheels. Its lock contains 69 distributions; the interpreter
also contains pip. The CPU baseline and CUDA variant share pinned versions for
their common dependencies, isolating the effect of changing runtime ownership.
TensorRT is not included or qualified.

`faster-whisper==1.2.1` declares CPU `onnxruntime` even though its ONNX use can
be served by the selected GPU distribution. Installing both wheels would give
two distributions ownership of the same import package. The resolver reads
the hash-verified upstream wheel, retains all other requirements, markers,
extras and Python constraints, and overrides only that version's unconditional
CPU ORT edge. The original wheel is not modified. The recorded override is
available alongside the generated lock. An unexpected conditional edge or
different wheel version stops resolution.

The installed closure check permits the ORT substitution only for that reviewed
package. Another package requiring CPU ORT is rejected. All remaining active
requirements and version bounds must resolve, there must be exactly one ORT
distribution, and its identity must match the requested CPU/CUDA variant.
The package builder also rejects a payload with a mislabeled or duplicate owner.
Ordinary `pip check` does not understand this distribution substitution; it is
not the validation used to qualify this fixed, immutable graph.

## Package identity and recovery

| Layout | CPU | CUDA |
| --- | --- | --- |
| Self-contained | `nvbroadcast-cpu` | `nvbroadcast-cuda` |
| Split | `nvbroadcast` + `nvbroadcast-runtime-cpu` | `nvbroadcast` + `nvbroadcast-runtime-cuda` |

The two runtime packages conflict through their shared versioned capability.
The split app still depends on one concrete runtime and checks the exact native
version pair before launch. Its CUDA variant has a distinct `.cuda` version
suffix because its launcher and native dependency differ. Different contents
are never published at the same package name/version/architecture identity.
Switching a split variant therefore replaces both app and runtime; this does
not establish transparent selection through an abstract dependency capability.

`run_switching.py` verifies input archive hashes and complete payload manifests
before creating disposable, network-disabled APT/DNF containers. Both layouts
exercise CPU installation, CUDA replacement, return to CPU, interrupted CUDA
replacement, exact-artifact repair, CPU rollback and removal. Split cases must
reject an app update without its matching CUDA runtime. Every completed state
checks all payload hashes and permissions, native ownership, exact versions,
the absence of the opposite runtime owner, dependency closure, application
source identity, isolated imports, CPU execution, synthetic audio/video EOS,
a mapped GTK window and the installed launcher's help path.

The interruption kills the whole disposable container after its transaction
marker appears and verifies exit 137. Killing only APT's process group can
leave dpkg in a separate session writing files and holding its database lock.
No package database locks are deleted. After restarting the same container,
launch must be refused. Debian recovery explicitly supplies both local CUDA
artifacts with `--fix-broken`, then reinstalls the exact files and runs
`apt-get check`. DNF recovery similarly replays the selected files and checks
dependencies. This tests one process-crash boundary; it is not exhaustive
power-loss, disk-full or filesystem durability testing.

The fixtures expose no host camera, microphone, display, audio server, user
home or Docker socket. User configuration/recording sentinels and system Python
bytes must survive all transactions. GPU execution is checked separately so a
device-less transaction test cannot silently count CPU fallback as CUDA success.

## Reproduce

Prepare the pinned application/binding wheels and builder images using the
[private-runtime instructions](../runtime-prototype/README.md). Use separate
CPU and CUDA work directories. The CUDA assembly consumes the checked-in lock;
refreshing it is an explicit maintainer operation, not an install hook:

```bash
python3 packaging/runtime-prototype/resolve.py --variant cuda \
  --directory "$NVB_CUDA_WORK" --output "$NVB_CUDA_WORK/pylock.candidate.toml"
# Review the candidate against the checked-in CUDA lock before accepting changes.
python3 packaging/runtime-prototype/prepare.py fetch-lock \
  --lock packaging/runtime-prototype/pylock.linux-x86_64-cp313-cuda.toml \
  --local-wheels "$NVB_CUDA_WORK/wheels" --directory "$NVB_CUDA_WORK/application"
python3 packaging/runtime-prototype/assemble.py --variant cuda \
  --directory "$NVB_CUDA_WORK" --output "$NVB_CUDA_WORK/payload" \
  --image "$NVB_BUILDER_ID"
python3 packaging/runtime-prototype/run_matrix.py --variant cuda \
  --runtime "$NVB_CUDA_WORK/payload/runtime" --directory "$NVB_CUDA_WORK/no-gpu" \
  --cells ubuntu22 ubuntu24 ubuntu26 debian12 debian13 fedora43 fedora44
```

The shared `inputs.json` retains its original CPU-baseline target label because
it pins common interpreter/tool/application inputs. The installed provenance's
`selection.json` records the actual variant and hashes of the selected lock
and common input manifest. Neither that baseline label nor the presence of a
CUDA provider is treated as proof of execution.

On a Linux NVIDIA host with the container toolkit, expose only one selected
GPU for a separate matrix cell. CUDA must execute the pinned ONNX probe, and
CuPy must compile and execute a fresh kernel. Caches stay under private `/tmp`;
the runtime and container filesystem remain read-only:

```bash
python3 packaging/runtime-prototype/run_matrix.py --variant cuda --gpu 0 \
  --cells ubuntu24 --runtime "$NVB_CUDA_WORK/payload/runtime" \
  --directory "$NVB_CUDA_WORK/gpu0"
```

Wrap each variant with `build.py --variant cpu|cuda --revisions 1`, supplying
its own verified manifest hash and fresh output directory as in the CPU README.
Then run the exact package sets against inspected clean/legacy image IDs:

```bash
python3 packaging/native-prototype/run_switching.py \
  --cpu-packages "$NVB_CPU_PACKAGES" --cuda-packages "$NVB_CUDA_PACKAGES" \
  --cpu-runtime "$NVB_CPU_PAYLOAD/runtime" \
  --cuda-runtime "$NVB_CUDA_WORK/payload/runtime" \
  --matrix "$NVB_SWITCH_MATRIX" --output "$NVB_SWITCH_RESULTS" --jobs 2
```

Use fresh output directories and retain complete logs. Rebuild packages in
independently recorded builders, compare both archive hashes and canonical
content manifests, and repeat runtime assembly. The recorded CPU payload was
reused from the prior experiment; newer assembly adds a variant-selection
provenance file, so recompute its identity when rebuilding it with this recipe.

## Production decisions still open

The complete CUDA payload is about 4.66 GiB before compression, and a CUDA DEB
is about 2.01 GiB. The split application is small, but a variant switch still
downloads the large runtime. Both designs need a pruning/size budget, updater
ownership, native dependency/ELF qualification, signed artifacts and production
support policy. The current RPM adapters disable automatic dependency
generation and build-root postprocessing to preserve the experimental bytes.

This test does not settle Zypper/EL support, runtime dependency-version upgrades,
running-app/service restarts, TensorRT, real camera/microphone/model acceptance,
or complete codec/NVIDIA/model redistribution requirements. Old source and
private-Python license inventories remain tied to their pinned revisions.
They do not inherit the separate release candidate's licensing changes.

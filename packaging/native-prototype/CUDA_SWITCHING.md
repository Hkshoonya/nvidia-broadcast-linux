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
`apt-get check`. DNF recovery preserves recognized unpack fragments before
replaying the selected files and checking dependencies. The fault can be
injected immediately after the prepare marker or during real RPM unpacking.
These recorded boundaries do not establish exhaustive power-loss, disk-full
or filesystem durability testing.

### Reproduced RPM unpack defect and preservation

RPM 6.0.2 stages payload files with a `;xxxxxxxx` transaction-ID suffix before
renaming them to their final paths. Killing the container during unpacking left
37 unowned files under `runtime/bin` in the recorded split case. DNF installation
and exact-file reinstallation both reported success, and all expected runtime
files matched, but the complete inventory still rejected the extra temporary
files. A successful package-manager return alone was insufficient. The original
fragments and failure inventory were retained before testing recovery.

The self-contained fault also left the checked-package script and both native
launchers staged outside the Python runtime. The harness regenerates these tiny
adapter sources, verifies each against the recorded native package-content
hashes, and exposes them read-only alongside the runtime payload. Preservation
also covers those exact recorded native stems. It does not sweep unrelated
system files. Successful-state verification compares the entire private native
prefix and rejects staged siblings of the package-owned integration files.

`recover_rpm_unpacked.py` handles this bounded recovery step. It requires the
transaction marker, the selected payload and prior manifest, and the actual
interruption's recorded time window. Every unexpected path must have an RPM
suffix within that window, correspond to a selected payload file, belong to
root, and have one link. Regular-file bytes must exactly match a prefix of the
hash-verified read-only payload; symlinks must have the recorded target. It
rejects symlinked parents and any unknown path or changed bytes before moving
anything. Recognized files are moved, never deleted, into a fresh private
directory outside the runtime. Native absolute paths receive separate relative
quarantine paths. Its report records their original paths, modes,
sizes and hashes. Native replay and the complete runtime/ownership verification
must then pass normally; the verifier does not exclude these paths.

The helper assumes all package writers are stopped and the root-owned fixture
is exclusive. It does not claim resistance to concurrent privileged filesystem
changes. The harness kills the entire isolated container, restarts only its
idle process, supplies a read-only payload mount, performs preservation, and
then invokes the package manager. Recovery reports remain with the logs; the
container's private quarantine is removed with that disposable fixture.
Production updater policy, quarantine retention, and automatic recovery are
separate work. The prototype helper is not shipped in the package payload.

RPM's staging and rename behavior is visible in its primary
[file-installer source](https://github.com/rpm-software-management/rpm/blob/7e7498db9b5b0fa1687d289c792331f0dd5cd251/lib/fsm.cc).

The fixtures expose no host camera, microphone, display, audio server, user
home or Docker socket. User configuration/recording sentinels and system Python
bytes must survive all transactions. GPU execution is checked separately so a
device-less transaction test cannot silently count CPU fallback as CUDA success.

The desktop fixture supplies a private ALSA null PCM for PortAudio enumeration
and a separate private PulseAudio null sink for GStreamer audio EOS. This keeps
the sounddevice/MediaPipe imports real without depending on a PipeWire user
session that the disposable container does not run. PortAudio also opens JACK;
Fedora's PipeWire JACK compatibility library could wait indefinitely for that
absent session even with the ALSA null PCM. The fixture therefore sets the
documented [`PIPEWIRE_NOJACK=1` refusal option](https://pipewire.pages.freedesktop.org/pipewire/page_man_pipewire-jack_conf_5.html).
The probe still requires real PortAudio enumeration of an ALSA or private
PulseAudio device and records
its host APIs. Periodic stack dumps and
partial timeout logs retain evidence of blocked native imports. These checks
qualify synthetic ALSA/PortAudio initialization and PulseAudio/GStreamer EOS;
they do not qualify JACK, physical microphones, a normal desktop PipeWire session,
or the application's native audio behavior.

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

Use an RPM-only matrix with `--fault-boundary rpm-unpack` to wait for an actual
temporary payload file before killing the container. A diagnostic reproduction
can disable the preservation step with `--skip-rpm-temporary-recovery` and keep
the failing fixture with `--keep-failed`. The result records its container name;
remove that exact container after capturing the failure evidence. These options
do not change or weaken the final integrity comparison. `--shapes split` limits
a targeted rerun to the affected layout. Timeout logs and periodic Python stack
dumps distinguish a synthetic-probe failure from a package transaction failure.

## Recorded qualification — 2026-10-07

The [evidence record](results-cuda-2026-10-07.json) binds the pinned inputs,
runtime manifests, package archives/content manifests, inspected container
images and raw result/log digests. Payload and package bytes were unchanged
during recovery debugging. The CUDA manifest contains 19,571 entries and
5,004,664,649 regular-file bytes; its independent assembly is identical.
All twelve CPU/CUDA DEB/RPM archive and content-manifest pairs are reproducible
in the separately recorded builders.

| Check | Completed evidence |
| --- | --- |
| No-GPU runtime | All seven distro cells pass with the final private audio fixture; CPU executes and unavailable CUDA is rejected |
| Hardware CUDA | The unchanged payload executes ORT CUDA and freshly compiled CuPy kernels on RTX 3080/Fedora 44 and RTX 5070/Ubuntu 24; recorded 2026-10-05 with the earlier probe harness |
| Clean native lifecycle | Ubuntu 24 and Fedora 44, self-contained and split: four successful traces |
| Legacy native lifecycle | Ubuntu 24 and Fedora 44, both layouts: four successful traces |
| Real RPM unpack faults | Two additional Fedora 44 traces observe a staged payload before exit-137 SIGKILL and complete repair, rollback and removal |
| PortAudio isolation | Forty independent imports in the exact legacy Fedora image enumerate the ALSA null PCM; maximum recorded import time 0.091 seconds |
| Integrity negative control | A package-owned regular launcher replaced with a same-byte, same-mode symlink is rejected; the real regular launcher passes |
| Dependency audit | pip-audit 2.10.1 finds no known vulnerabilities in the 69 pinned public distributions, including interpreter pip; native libraries and licensing are outside this audit |

The final real-unpack self-contained trace preserves 42 fragments, including
both launchers and `check-packages`; the split trace preserves 38, including
`runtime.json`. Counts differ between interruption timings. Both then restore
the exact selected inventory, return to CPU, remove the packages without an
orphaned prefix, and preserve user sentinels and system Python.

The four Ubuntu native traces were retained from the earlier private-ALSA
fixture. Their `libjack-jackd2` library is unaffected by the subsequent
PipeWire-only refusal option; the final seven-distro runtime matrix separately
checks the revised fixture. The six affected Fedora native traces were rerun.
An independent review then identified an inherited verifier gap: an expected
regular integration file could be a same-byte symlink outside the runtime.
The verifier now requires `lstat` regular-file type before hashing it. All six
remaining Fedora CPU rollback states report the stricter verifier's script hash
and type-check flag. Earlier states retain the prior verifier identity; the
focused real-file positive/negative controls qualify that added check. The
record does not relabel earlier traces as executions of the final recipe.

The original RPM integrity failure, incomplete runtime-only remediation,
PortAudio initialization timeout, and an overly strict ALSA-only probe check
on Ubuntu 26 remain recorded as failures. Ubuntu 26 exposes private PulseAudio
devices through PortAudio; the corrected probe accepts a real ALSA or private
PulseAudio device and the targeted rerun and final seven-distro matrix pass.
The local evidence includes the original RPM fragments and complete stack and
failure logs. No production package or host audio configuration was changed.

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

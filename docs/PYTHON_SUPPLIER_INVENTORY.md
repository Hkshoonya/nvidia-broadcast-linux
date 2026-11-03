# App-owned Python supplier inventory

`scripts/generate_python_sbom.py` is an opt-in inventory tool for issue #60.
It reads installed wheel metadata from an already extracted application runtime
and writes an SPDX 2.3 JSON document. It uses the existing `ArtifactEnvironment`
discovery without putting the artifact on `sys.path`, importing its modules,
processing its `.pth` files, or running its Python interpreter. No network
requests, resolution, installation or extraction occur.

This is a **partial Python package inventory**, not a complete release SBOM.
It records the actual installed names/versions and hashes of their `METADATA`
files. Package licenses remain `NOASSERTION`; package/dependency/license analysis,
the CPython interpreter and its non-distribution components, native libraries embedded in wheels, codecs,
OS/content runtimes, drivers and model weights require separate inventory work.
Reading `.dist-info` identifies a declared distribution; it does not establish
that every installed package file matches a supplier wheel.

Use a previously verified, stable extraction and its matching original archive.
The tool computes the archive's SHA-256, but it does not authenticate that archive
or prove its relationship to the extracted directory. The source commit and
optional lock files are explicitly labeled **caller-provided context**. Their
hashes identify the supplied inputs; the tool does not check build provenance,
that a build consumed those locks, or runtime dependency closure. Existing
artifact/provenance and package validation remain responsible for those checks.

From a trusted checkout with the existing `packaging` dependency installed:

```bash
PYTHONPATH=src python3 scripts/generate_python_sbom.py \
  --runtime-root /absolute/path/to/extracted/runtime \
  --architecture amd64 \
  --artifact /absolute/path/to/nvbroadcast-candidate.deb \
  --source-commit FULL_SOURCE_COMMIT \
  --source-date-epoch SOURCE_COMMIT_UNIX_TIMESTAMP \
  --lock /absolute/path/to/assembled/pylock.toml \
  --output /absolute/path/to/new-python-inventory.spdx.json
```

The runtime root is the private interpreter prefix containing
`lib/python*/site-packages`, or an extracted Snap/application root with the
corresponding `usr/lib` or `usr/local/lib` Python paths. The tool recognizes the
existing Linux amd64/arm64 architecture spellings; it does not qualify any new
platform. It refuses external package roots, redirected wheel metadata,
duplicate distribution owners, missing/ambiguous metadata and inventories it
cannot completely represent. Egg metadata is unsupported and fails clearly.
Use a stable extraction while reading; concurrent mutation is not supported.

The output must be a new file outside the inspected runtime. Inputs are never
rewritten. Explicit timestamps and sorted records keep output independent of
the current time or the machine's absolute directories. The document namespace
hash covers the emitted archive identity, source/lock context, metadata and
scope, so changing any recorded input creates a new document identity.

The test suite validates output with `jsonschema==4.26.0` against the unchanged
[official SPDX 2.3 JSON schema](https://github.com/spdx/spdx-spec/blob/aadf3b0b8dbbabdb4d880b0fc714255fea436ff7/schemas/spdx-schema.json),
pinned by SHA-256 in the test. The local schema copy and upstream license are
under `tests/data/spdx-2.3`; validation refuses external references. The SPDX
specification's [package fields](https://spdx.github.io/spdx-spec/v2.3.1/package-information/)
permit an explicitly limited package-level inventory. The document version is
`SPDX-2.3`, as required by the specification's major/minor version field.
SPDX metadata uses `CC0-1.0`; this does not change any inventoried software's
license. The optional lock context uses [PEP 751 format version 1.0](https://peps.python.org/pep-0751/).

The generator does not change installed applications, release assets,
signatures, channels or issue #60's broader acceptance gates.

## Optional experimental native-candidate evidence

The manual `native-cpu-finalize.yml` workflow has a `python_inventory` boolean
input, disabled by default. It applies to the existing experimental CPU/CUDA
native-candidate harness; enabling it does not build a new package, publish a
release, or complete issue #60. The installed runtime and package bytes are
unchanged. No inventory is added to the signed bundle or the DEB/RPM family
qualification inputs.

When enabled, the existing authenticated DEB validator retains only regular
`METADATA` files during its ordinary payload stream. The projection must match
both the complete decoded path inventory and the runtime manifest, including
distribution directories and their directory parents. Missing metadata,
redirected paths, unsupported egg owners and omitted distributions fail. The
same limits apply as the generator: 1,000 distributions, 4 MiB per file and
64 MiB of metadata in total. The projection also retains the authenticated
assembled PEP 751 lock and the producer commit timestamp. A digest in the
validation receipt binds this projection before signing; its small, separate
artifact never enters the signing environment.

Only after every existing final-package qualification gate succeeds does
`retain_python_inventory.py` reverify the signed bundle against the signing
job's externally supplied manifest digest, recheck the qualification evidence,
and authenticate the projection. It generates one document for the final DEB
and one for the final **signed** RPM. Each is generated twice with identical
bytes, validated against the pinned official SPDX schema without remote
references, and checked for exact metadata identities, archive hashes,
relationships, timestamp and explicitly partial scope. The receipt records
the source helper hashes, schema, evidence-tool closure, projection, runtime
manifest, assembled lock and final archives. Package/projection identities are
checked again before a successful receipt is written.

The isolated Ubuntu evidence process uses the existing pure `packaging==26.0`
wheel and the exact schema-validation closure in
`packaging/native-runtime/python-inventory-tools.json`. These are host tools,
not runtime dependencies. The six validator wheels total less than 1 MiB and
include a CPython 3.12 Linux x86_64 RPDS extension; this restriction applies only
to the evidence job. The bounded bootstrap verifies each source-pinned wheel,
refuses redirected, duplicate, executable `.pth` and installation-scheme
members, and checks the complete extracted file set before importing. It uses
no resolver or installer. Projected runtime paths never enter `sys.path`, and
the target interpreter, modules and entry points never execute.

The optional `python-inventory/` directory under the retained qualified
development artifact contains `deb.python-inventory.spdx.json`,
`rpm.python-inventory.spdx.json` and `receipt.json`. Failure of optional
generation fails that retention attempt; it cannot produce a successful
inventory receipt. A retry uses fresh output. The default workflow retains its
existing output and capacity policy; opting in adds a 256 MiB evidence reserve.
This strengthens the native harness's archive/projection binding around the
unchanged generator. It does not turn Python metadata into a complete native,
interpreter, codec, model, license, device or public-release inventory.

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

This source-only tool does not change build workflows, installed applications,
release assets, signatures, channels or issue #60's broader acceptance gates.

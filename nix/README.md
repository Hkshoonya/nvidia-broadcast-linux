# Experimental Nix CPU package qualification

This is an interim x86_64 CPU package and NixOS module. GPU inference, CuPy and
meeting transcription remain outside its closure. The in-app installer cannot
modify the Nix-owned runtime. No real NixOS camera/microphone/virtual-device
acceptance is implied by evaluation or a generated-data build check.

The source refresh pins current Nixpkgs and the official PSF CPython 3.13.18
source, plus exact canonical PyPI inputs for Pillow, certifi, filelock, urllib3
and Setuptools. `security-inputs.json` records source URLs and hashes. Current
application requirement bounds are preserved; the old AV relaxation and obsolete
OpenCV removal are absent. The OpenCV contrib metadata/provider must pass the
actual CPU dependency-closure check. Nixpkgs-native patches and the new Python
microrelease remain subject to an actual hermetic build, not static assertions.

Run `python3 nix/source_files.py` after staging intended source changes. The
manifest covers Git-tracked application/test/documentation inputs and excludes
the flake/Nix expressions themselves. The normal source test and hosted workflow
refuse a stale manifest.

The hosted workflow has two separate stages:

1. With an unmatched lock, a PR or manual `prepare-lock` operation only resolves
   the exact committed Nixpkgs revision and retains its lock and bounded metadata.
   The CPU qualification job is skipped. Review the generated input identity and
   commit that lock before proceeding. A green preparation run is not a package
   build or support/merge qualification.
2. With the reviewed lock committed, qualification refuses lock updates, records
   the official signed-cache dry-run forecast, and applies derivation/download/
   storage/time/log bounds before building the package and its checks. A budget
   failure is preserved as a failure; no requirements or tests are weakened to
   obtain a pass. Only small evidence files are uploaded, never binaries, cache
   contents, release assets or credentials.

Bootstrap runs only on disposable GitHub-hosted Linux x86_64 workers. It verifies
the official Nix 2.35.4 installer before execution; that installer verifies its
binary archive before extraction. This replaces 2.35.2, which is affected by
[daemon repair privilege escalation](https://github.com/NixOS/nix/security/advisories/GHSA-jwvx-xq5j-gc56)
and [multi-user installer ownership](https://github.com/NixOS/nix/security/advisories/GHSA-hf6w-2rg9-v84g)
defects. Bootstrap uses HTTPS and fixed checksums; the application build's sandbox
and locked sources are a separate trust boundary. No local Nix installation is
needed to review or run the source/controller failure tests.

The runtime check uses a clean environment with the finished app's interpreter
and app-only dependencies, separate from pytest and build tools. It checks actual
single-owner installed/imported suppliers, CPU inference, AV/RNNoise processing,
isolated MediaPipe imports and model initialization, resources/source identity,
and the actual packaged recording smoke test. Generated media does not substitute
for physical camera or audio acceptance.

A clean runtime environment defaults both OpenSSL certificate-file variables to
the exact upstream certifi bundle and an empty capath. The proof compares default
stdlib and Requests adapter trust roots and requires certificate/hostname checks,
then verifies CPython's compromised-intermediate fixture fails with code 28 even
when its root is explicitly trusted. External `NIX_SSL_CERT_FILE`, `SSL_CERT_FILE`,
`SSL_CERT_DIR` or Requests trust overrides are user/host configuration outside
that default-context qualification.

Application and supplier notices must survive packaging/fixups. A Python wrapper's
BSD metadata does not establish the complete licensing of bundled FFmpeg/codecs,
native libraries, models or the interpreter. Complete native/license inventory,
current advisory review and real NixOS/device acceptance remain separate work;
this draft tooling is not a public launch or complete release-security claim.

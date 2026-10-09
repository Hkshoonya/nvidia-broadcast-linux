# RPM publisher signatures

The signing implementation uses a dedicated project OpenPGP key. It signs a copy
of the RPM, verifies the signature in an empty RPM database containing only the
reviewed public key, and compares the immutable header and unpacked payload to
the unsigned input. An unsigned RPM with valid checksums is rejected.

This does not retroactively sign v1.5.3. Its qualified release files and tag stay
unchanged. Production key provisioning and an actual signed candidate are
required before claiming that the RPM acceptance item in #60 is complete.

## Key provisioning

Keep a project key separate from personal email-signing keys. Before creating a
new key, check for an existing project key and preserve its identity if present.
No RPM signing key was found in the checked repository, configured GitHub
environments or local secret-key inventory on 9 October 2026; the maintainer has
not yet confirmed whether a separate historical key exists.

The reviewed public key belongs at:

```
packaging/keys/RPM-GPG-KEY-nvbroadcast.asc
packaging/keys/RPM-GPG-KEY-nvbroadcast.fingerprint
```

The fingerprint file must contain the full primary fingerprint. The ASCII key
must contain only public material. Review those public files before use; never
commit a private key, keyring, passphrase, or encrypted private backup.

Configure the GitHub `rpm-signing` environment with:

- `RPM_SIGNING_PRIVATE_KEY`: the armored secret signing key, preferably an
  exported signing subkey with the primary certification key kept offline.
- `RPM_SIGNING_PASSPHRASE`: that key's passphrase.

Restrict this environment to the trusted main branch and intended release tags.
Store the private backup and revocation material securely outside this repo.
Record the public fingerprint in the release documentation and website through
the normal reviewed change. A key rotation needs an explicit published transition;
do not silently replace the trusted fingerprint during an ordinary release.

The workflow confines these secrets to the signing step. It never prints them
or supplies them in process arguments. The signer uses disposable GnuPG/RPM
databases and removes its temporary keyring and passphrase file on completion or
failure. It does not import into the maintainer's normal keyring or system RPM
database. Signing diagnostics deliberately omit tool output that might echo
malformed secret input.

## Build and verification order

1. Build and test the native packages without signing credentials.
2. Sign the exact RPM copy after Linux/runtime checks pass.
3. Require native signature verification with the pinned public key, plus
   unchanged immutable header and payload.
4. Render `nvbroadcast-native-upgrade` against the final signed RPM and exact DEB.
5. Generate release checksums and provenance for the signed outputs and public key.
6. Verify those exact artifacts before publishing.

`Build Packages` runs `sign-linux` for release tags or an explicitly selected
manual `sign_rpm` dispatch. Missing public-key files or credentials fail the job;
there is no unsigned-release fallback. Unsigned build artifacts remain available
for inspection and cannot be selected by the release publication job.

## User verification

Obtain the public key from the project's reviewed release/website and compare its
full fingerprint with the published fingerprint before importing it. Substitute
the actual RPM filename below:

```bash
gpg --show-keys --with-fingerprint RPM-GPG-KEY-nvbroadcast.asc
sudo rpm --import RPM-GPG-KEY-nvbroadcast.asc
rpmkeys --checksig --verbose nvbroadcast-VERSION-RELEASE.ARCH.rpm
```

Require a verified signature as well as valid payload/header digests. A digest-only
success on an unsigned package does not authenticate its publisher. A public key
downloaded beside a package is not an independent trust anchor until its fingerprint
has been checked through the established project identity.

Native RPM signatures complement the release's GitHub source/workflow attestations
and checksums. They do not verify application correctness or all bundled dependency
licenses and vulnerabilities.

Primary references: [RPM signing](https://rpm.org/docs/6.1.x/man/rpmsign.1) and
[GitHub environment secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

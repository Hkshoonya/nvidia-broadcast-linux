# Recover an interrupted native RPM installation

This interface applies to the complete `x86_64` CPU/CUDA RPMs built by
`packaging/native-runtime`, with releases such as `2.cpu` or `2.cuda`.
It does not apply to the existing public v1.5.3 `noarch` package or the Snap.
Those immutable downloads are unchanged. Native-runtime promotion still requires
its other documented release gates.

If RPM stops during unpack, Broadcast's transaction marker blocks launches.
RPM can leave files named `filename;1234abcd` which a reinstall does not remove.
Do not delete these files, edit the marker, or start its partially installed
private Python.

The native RPM includes one standalone `/usr/bin/nvbroadcast-recover-rpm`.
It runs system Python 3.11 or newer with isolated imports and bytecode disabled.
The executable and its embedded public key are delivered inside the project-signed
RPM. RPM may stage even `/usr/bin` files under temporary names until unpacking
finishes, so the signed package also prepares a recovery copy before unpacking.
Its ordinary `%pre` scriptlet runs only after system Python's explicit package
dependency, and writes the exact recovery bytes atomically to a root-only,
content-addressed file in `/var/lib/nvbroadcast/recovery-tools/`. It prints that
path before writing the transaction marker. Neither copy imports the private
runtime. An unavailable Python or unsafe recovery state fails before unpacking.
The public key's SHA-256 is fixed in the recovery source; a downloaded adjacent
key or checksum cannot authorize a different publisher.

Publisher primary fingerprint:

```text
A691DCA6C031FEFD2C34F5AAFE33D0C20C9451D1
```

If `/usr/bin/nvbroadcast-recover-rpm` is absent after a first-install interruption,
use the exact path printed as `NV Broadcast recovery copy:` by the authenticated
package transaction. Run it with system Python, passing the signed target RPM:

```bash
sudo /usr/bin/python3 -IB /var/lib/nvbroadcast/recovery-tools/recover-PRINTED_SHA256.py /path/to/target.rpm
```

The printed hash is part of the filename, not a separate downloaded checksum.
Do not execute a staged `;1234abcd` file, choose an arbitrary file from that
directory, or download an unverified replacement script. If the trusted path is
unknown or either copy's integrity is in doubt, request maintainer-assisted
recovery using the authenticated package. There is no unsigned bypass. First
installations may have no launcher at all; interrupted upgrades retain the old
launcher's refusal while the transaction marker is present.

## Inspect and preserve

Quit Broadcast and its virtual-camera service, and make sure no package-manager
transaction is running. Obtain the exact project-signed RPM whose version and
CPU/CUDA variant were being installed. Run:

```bash
sudo nvbroadcast-recover-rpm /path/to/nvbroadcast-1.5.3-2.cpu.x86_64.rpm
```

Inspection makes no changes to installed files. It snapshots the package into
private temporary storage, checks its actual OpenPGP signature using only the
pinned project public key in a temporary RPM keyring, and reports the verified
package hash and matching fragment paths. It does not trust an adjacent checksum,
the installed runtime manifest, or a filename to authenticate package contents.
The signed package must identify `nvbroadcast`, `x86_64`, and an exact native
CPU/CUDA version matching the root-owned transaction marker.

If an interrupted upgrade left normal files that the new version removed,
provide the previous project-signed package as well:

```bash
sudo nvbroadcast-recover-rpm /path/to/target.rpm \
  --previous-package /path/to/previous.rpm
```

Both packages are independently authenticated. The previous package supplies
only recognized old paths; temporary fragments must match the target package.
Do not substitute a content manifest or build directory.

After a successful inspection, repeat the same command with `--preserve`.
It plans and validates all fragments before moving any. It accepts only
root-owned, single-link regular files whose complete bytes are an exact prefix
of their signed target member, or an exact matching temporary symlink. It rejects
redirected/writable parent directories, unknown private-prefix paths, unsupported
archive members, mixed transaction suffixes and modified fragments. One common
RPM transaction timestamp must fall within 300 seconds before the marker's
timestamp, with one second of clock rounding allowance. Ambiguous or stale
transactions require manual review; never touch the marker to pass that check.

Recovery takes RPM's native transaction `fcntl` lock and Broadcast's existing
launch `flock`, preserving their inode identities. Contention fails without
moving files. RPM documents the lock path as
[`%_rpmlock_path`](https://github.com/rpm-software-management/rpm/blob/master/macros.in).
The implementation reads it from the installed RPM configuration and rejects an
unsafe/missing lock. Its tests exercise a separate real RPM database transaction
while this lock is held. A decoding watchdog bounds the time spent holding locks.

## Replay and verify

Preservation saves an audit, original fragments, the verified RPM and public
key under a new root-only directory in `/usr/lib/nvbroadcast-recovery/`.
It records every planned move before moving any file, flushes the preserved
file data and audit/directory changes, and records completed moves separately.
It never removes the transaction marker. Renames must stay on one filesystem;
an unusual mount layout requiring a cross-filesystem move is refused.
Allow free space for the saved RPM and temporary verification snapshots.

The report includes the exact commands to:

1. Import the saved, pinned public key into the system RPM database if necessary.
2. Replay the saved package with `rpm --upgrade --replacepkgs --oldpackage`.
3. Check the marker is absent, native package state is complete, and the complete
   runtime passes its installed integrity/dependency validator.

Use those printed paths, not a subsequently replaced download. RPM may need
native dependencies already installed by the operating system; recovery does
not download or resolve them. A replay or configuration failure keeps launch
blocked and must not be described as a successful repair. Inspect the error
without deleting the audit or evidence.

Successful configuration first validates the complete private runtime and the
installed recovery executable. Only then does it remove the exact current
bootstrap copy and clear the transaction marker. A normal uninstall does not
create a bootstrap copy. Copies from older failed transactions and unrelated
recovery state are preserved for manual review; successful transactions do not
accumulate copies. Do not remove failed-transaction evidence until it has been
reviewed.

If preservation itself is interrupted, retain the recovery directory and its
`recovery.json`. It records both the full plan and completed moves; an incomplete
record may need reconciliation against the preserved files after an abrupt
shutdown. Do not delete the directory or assume every move completed. Remaining
fragments can be inspected again once native package transactions are stopped.
No promise of atomic multi-file rollback or power-loss recovery is made.

For development from a trusted source checkout, the unrendered script accepts
`--public-key packaging/keys/RPM-GPG-KEY-nvbroadcast.asc`; this still enforces
the same pinned key hash. It does not add a custom signing identity.

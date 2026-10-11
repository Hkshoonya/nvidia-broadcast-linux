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
RPM, and `/usr/bin` precedes the large private runtime in the package payload.
The public key's SHA-256 is fixed in the recovery source; a downloaded adjacent
key or checksum cannot authorize a different publisher.

Publisher primary fingerprint:

```text
A691DCA6C031FEFD2C34F5AAFE33D0C20C9451D1
```

If interruption occurred before the recovery executable was installed, or its
integrity is in doubt, stop and request maintainer-assisted recovery using the
authenticated package. Do not bootstrap by executing a script taken from an
unverified archive. The current interface deliberately has no unsigned bypass.

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

If preservation itself is interrupted, retain the recovery directory and its
`recovery.json`. It records both the full plan and completed moves; an incomplete
record may need reconciliation against the preserved files after an abrupt
shutdown. Do not delete the directory or assume every move completed. Remaining
fragments can be inspected again once native package transactions are stopped.
No promise of atomic multi-file rollback or power-loss recovery is made.

For development from a trusted source checkout, the unrendered script accepts
`--public-key packaging/keys/RPM-GPG-KEY-nvbroadcast.asc`; this still enforces
the same pinned key hash. It does not add a custom signing identity.

# Source runtime updates

The Linux source installer builds every update at a new permanent path below
`.nvbroadcast-runtimes/`. Python environments contain absolute interpreter and
console-script paths, so generations are never renamed after creation. This
also applies to same-variant and same-Python updates.

The installer checks required imports, desktop bindings, effects initialization
(including saved GPU preferences in a CPU runtime), dependency closure,
single ONNX Runtime ownership, and the pinned-model CPU/CUDA execution probe
in fresh candidate processes. It repeats the final checks after optional
packages have been installed. Only then does it atomically replace
`selection.json`, containing both the active and previous generation.

The `nvbroadcast` and `nvbroadcast-vcam` launchers read that selection and exec
the selected interpreter in isolation. Audio subprocesses inherit that
interpreter. Without a selection, the launchers use the legacy `.venv`, which
is neither moved nor changed by the new installer. A malformed selection is
an error; it does not silently fall back to a different runtime.

Stop the application and its audio/virtual-camera services before activation.
The process guard checks both the selected environment and the legacy `.venv`.
A lock serializes selection changes. Candidates remember the selection they
were built against, so a second concurrent installer cannot overwrite a newer
selection. This does not prevent a user from directly starting an old interpreter
between a process check and activation; generations remain in place and are not
mutated, so such a process keeps using its original environment.

## Recovery

Run `./install.sh --rollback-runtime` to recheck and select the previous runtime.
Run it again to return to the newer one, provided that runtime still verifies.
Rollback needs no pip download, but a missing inference-probe model may need its
normal verified model download. It does not roll back system packages, drivers,
configuration, or the checkout itself.

Build failures and interrupted installer exits discard only the unselected
candidate. If the process is killed without cleanup, its directory can remain.
Print the active path with:

```bash
python3 scripts/source_runtime.py --project . active
```

An abandoned candidate can be removed with `source_runtime.py --project . discard
<candidate-path>`. The command refuses the active and previous generations.
Do not remove `selection.json` or selected directories by hand. The source
uninstaller removes the generated environments and the legacy `.venv` after
checking that no relevant application process is running.

Selection-file publication uses file and directory synchronization. An I/O
error after atomic replacement is reported as an ambiguous durability state:
the new selection may already be visible. Inspect `active` before retrying;
cleanup refuses to remove either selected environment. This is not a tested
power-loss guarantee for every payload file and filesystem.

## Scope

This mechanism prevents failed local source upgrades from destroying the
working Python environment. It is not the signed, self-contained runtime-pack
design in #53/#60/#77. The source checkout, selected host Python, distro GI/GTK,
GStreamer libraries, and the installing user remain trusted inputs. Dependency
resolution still uses the configured indexes; these generations do not have
an authenticated whole-payload manifest or a pinned offline dependency graph.

In-app pip installation is disabled for these generations so it cannot mutate
the selected runtime after inference libraries have loaded. Rerun `install.sh`
to add optional packages. Manual pip changes, Makefile development environments,
and package-manager-owned DEB/RPM/Snap/Flatpak environments are outside this
source selection transaction. Do not use this tool to change those packages.

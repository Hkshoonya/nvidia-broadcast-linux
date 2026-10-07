#!/usr/bin/env python3
"""Verify CPU/CUDA native replacement in disposable, network-disabled containers.

No GPU is exposed: CUDA must reject execution and its CPU provider must work.
Real CUDA/CuPy execution is a separate run_matrix.py --gpu check, recorded
independently from package-manager ownership and interruption recovery.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from run_lifecycle import HERE, RUNTIME_TOOLS, PREFIX, SENTINELS, Lifecycle, digest


def check_inputs(packages: Path, runtime: Path, variant: str) -> list[dict]:
    records = json.loads((packages / "packages.json").read_text())
    keys = set()
    for record in records:
        artifact = (packages / record["artifact"]).resolve(strict=True)
        if not artifact.is_relative_to(packages) or digest(artifact) != record["sha256"]:
            raise ValueError("package path or digest mismatch")
        key = record["family"], record["kind"], record["revision"]
        if key in keys or record.get("variant", "cpu") != variant:
            raise ValueError("duplicate package or wrong variant")
        keys.add(key)
    required = {(family, kind, 1) for family in ("deb", "rpm") for kind in ("self", "app", "runtime")}
    if not required <= keys:
        raise ValueError("incomplete CPU/CUDA comparison set")
    from build import inventory
    if inventory(runtime) != json.loads((runtime.parent / "manifest.json").read_text()):
        raise ValueError("runtime differs from manifest")
    return records


class Switching(Lifecycle):
    def __init__(self, cell: dict, shape: str, packages: dict, runtimes: dict, output: Path,
                 keep_failed: bool = False, fault_boundary: str = "prepare",
                 skip_rpm_temporary_recovery: bool = False):
        super().__init__(cell, shape, packages["cpu"], runtimes["cpu"], output)
        self.package_sets, self.runtimes = packages, runtimes
        self.variants = {v: json.loads((p / "packages.json").read_text()) for v, p in packages.items()}
        self.keep_failed = keep_failed
        self.fault_boundary = fault_boundary
        self.skip_rpm_temporary_recovery = skip_rpm_temporary_recovery

    def native_recovery_inputs(self) -> Path:
        """Reproduce tiny adapter files and bind them to the built content hashes."""
        from build import integration, write
        directory = self.output / "native-recovery-inputs"
        source = directory / "root"
        source.mkdir(parents=True)
        contents = {}
        for variant in ("cpu", "cuda"):
            combined = {}
            for record in self.selection(variant):
                content = json.loads((self.package_sets[variant] / record["content"]).read_text())
                for name, entry in content.items():
                    if name in combined and combined[name] != entry:
                        raise ValueError("inconsistent split native content manifests")
                    combined[name] = entry
                if variant == "cuda":
                    if record["kind"] != "runtime":
                        integration(source, record["family"], record["kind"], record["version"], variant)
                    write(source / f"usr/lib/nvbroadcast/{record['kind']}.json", json.dumps({
                        "prototype": True, "family": record["family"], "package": record["name"],
                        "version": record["version"], "variant": variant,
                        "runtime_manifest_sha256": digest(self.runtimes[variant].parent / "manifest.json"),
                    }, sort_keys=True) + "\n")
            contents[variant] = combined
        for name, entry in contents["cuda"].items():
            if Path(name).is_relative_to(PREFIX):
                continue
            generated = source / name.lstrip("/")
            if "sha256" in entry:
                if generated.is_symlink() or digest(generated) != entry["sha256"]:
                    raise ValueError(f"native recovery source differs from recorded package: {name}")
            elif "symlink" in entry:
                if not generated.is_symlink() or os.readlink(generated) != entry["symlink"]:
                    raise ValueError(f"native recovery link differs from recorded package: {name}")
        for variant, value in contents.items():
            (directory / f"{variant}.json").write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        return directory

    def selection(self, variant: str, only: str | None = None) -> list[dict]:
        kinds = {only} if only else ({"self"} if self.shape == "self" else {"app", "runtime"})
        return [p for p in self.variants[variant] if p["family"] == self.cell["family"]
                and p["revision"] == 1 and p["kind"] in kinds]

    def transaction(self, variant: str, only: str | None = None, reinstall: bool = False,
                    repair: bool = False) -> list[str]:
        files = [f"/artifacts/{variant}/{p['artifact']}" for p in self.selection(variant, only)]
        if self.cell["family"] == "deb":
            return ["apt-get", "-y", "--allow-downgrades",
                    *(["--fix-broken"] if repair else []),
                    *(["--reinstall"] if reinstall else []), "install", *files]
        return ["dnf", "-y", "--disable-repo=*", "--setopt=tsflags=",
                "reinstall" if reinstall else "install", "--allowerasing", *files]

    def verify_variant(self, label: str, variant: str) -> None:
        run = self.command(label + "-integrity", [f"{PREFIX}/bin/python", "-I", "-B",
                           "/native-prototype/verify.py", self.cell["family"], self.shape, "1",
                           "--artifacts", f"/artifacts/{variant}", "--manifest", f"/manifests/{variant}.json"])
        self.result["steps"][-1]["verification"] = json.loads(run.stdout.split("RESULT=", 1)[1])
        probe = ["bash", "/runtime-prototype/desktop_probe.sh", f"{PREFIX}/bin/python", "-I", "-B", "-u",
                 "/runtime-prototype/probe.py", "--runtime", PREFIX, "--python-version", "3.13.16",
                 "--window", "--variant", variant]
        if variant == "cuda":
            probe += ["--cuda-unavailable"]
        run = self.command(label + "-probe", probe, user=True)
        reports = [json.loads(line[7:]) for line in run.stdout.splitlines() if line.startswith("RESULT=")]
        source = RUNTIME_TOOLS.parents[1] / "src/nvbroadcast"
        expected = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))}
        if len(reports) != 1 or reports[0]["app_python_hashes"] != expected:
            raise RuntimeError("missing runtime report or packaged application source mismatch")
        self.result["steps"][-1]["verification"] = reports[0]
        other = "cuda" if variant == "cpu" else "cpu"
        forbidden = [f"nvbroadcast-{other}", f"nvbroadcast-runtime-{other}"]
        # dpkg -l also includes residual-config entries, so inspect configured
        # package state rather than mistaking a harmless database record for an owner.
        for name in forbidden:
            command = (["dpkg-query", "-W", "-f=${Status}", name] if self.cell["family"] == "deb" else
                       ["rpm", "-q", name])
            result = self.command(label + "-absent-" + name, command, expected=None)
            if (result.stdout.strip() == "install ok installed" if self.cell["family"] == "deb" else result.returncode == 0):
                raise RuntimeError(f"opposite runtime package remains installed: {name}")
        self.command(label + "-launcher", ["bash", "/runtime-prototype/desktop_probe.sh",
                                           "/usr/bin/nvbroadcast", "--help"], user=True)
        self.command(label + "-dependencies", ["apt-get", "check"] if self.cell["family"] == "deb" else
                     ["dnf", "--disable-repo=*", "check"])

    def interrupt_switch(self) -> None:
        tid_min = int(time.time())
        with (self.output / "interrupted-switch.log").open("w") as log:
            process = subprocess.Popen(["docker", "exec", self.name, *self.transaction("cuda")],
                                       stdout=log, stderr=subprocess.STDOUT)
            killed = False
            observed_path = ""
            try:
                # DNF verifies the complete multi-GiB RPM before any prepare
                # hook runs. Allow that work to finish before injecting a fault.
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline and process.poll() is None:
                    check_command = ["test", "-f", "/usr/lib/nvbroadcast/.transaction"]
                    if self.fault_boundary == "rpm-unpack":
                        check_command = ["sh", "-c", "test -f /usr/lib/nvbroadcast/.transaction && "
                                         "find /usr/lib/nvbroadcast/runtime -type f -name '*;????????' "
                                         "-size +1M -print -quit"]
                    check = subprocess.run(["docker", "exec", self.name, *check_command],
                                           capture_output=True, text=True, timeout=10)
                    if check.returncode == 0 and (self.fault_boundary == "prepare" or check.stdout.strip()):
                        observed_path = check.stdout.strip()
                        # APT may put dpkg in a separate session; killing only
                        # APT's process group leaves the real unpacker alive.
                        # Terminate this disposable container so no descendant
                        # can continue writing or retain a package database lock.
                        subprocess.run(["docker", "kill", "--signal", "KILL", self.name],
                                       capture_output=True, check=True, timeout=15)
                        killed = True
                        break
                    time.sleep(0.05)
                status = process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    subprocess.run(["docker", "kill", self.name], capture_output=True)
                    process.wait(timeout=15)
            state = json.loads(subprocess.check_output(["docker", "inspect", "--format", "{{json .State}}", self.name], text=True))
            if not killed or status == 0 or state["Running"] or state["ExitCode"] != 137:
                raise RuntimeError(f"SIGKILL not established: killed={killed}, exit={status}")
        tid_max = int(time.time())
        self.result["steps"].append({"name": "interrupted-switch", "exit": status,
                                    "container_exit": state["ExitCode"],
                                    "fault_boundary": self.fault_boundary,
                                    "observed_unpack_path": observed_path,
                                    "tid_min": tid_min, "tid_max": tid_max,
                                    "fault": "SIGKILL whole isolated container after new variant's prepare marker"})
        subprocess.run(["docker", "start", self.name], capture_output=True, check=True, timeout=30)
        self.command("interrupted-launch-blocked", ["sh", "-c",
                     'if [ -x /usr/bin/nvbroadcast ]; then exec /usr/bin/nvbroadcast --help; else exit 78; fi'],
                     user=True, expected=78)
        self.command("interrupted-native-inventory", ["dpkg-query", "-W"] if self.cell["family"] == "deb"
                     else ["rpm", "-qa"])
        if self.cell["family"] == "deb":
            self.command("recover-pending-configure", ["dpkg", "--configure", "-a"], expected=None)
        elif not self.skip_rpm_temporary_recovery:
            recovery = self.command("preserve-rpm-unpack-fragments", [f"{PREFIX}/bin/python", "-I", "-B",
                                    "/native-prototype/recover_rpm_unpacked.py", "--runtime", PREFIX,
                                    "--payload", "/payloads/cuda", "--manifest", "/manifests/cuda.json",
                                    "--baseline-manifest", "/manifests/cpu.json", "--tid-min", str(tid_min),
                                    "--tid-max", str(tid_max), "--quarantine", "/var/lib/nvb-prototype-recovery",
                                    "--native-source", "/native-recovery/root",
                                    "--native-manifest", "/native-recovery/cuda.json",
                                    "--native-baseline", "/native-recovery/cpu.json"])
            report = json.loads(recovery.stdout.split("RESULT=", 1)[1])
            if self.fault_boundary == "rpm-unpack" and not report["preserved"]:
                raise RuntimeError("unpack fault did not leave any recorded RPM fragments")
            self.result["steps"][-1]["verification"] = report
            (self.output / "rpm-unpack-preservation.json").write_text(json.dumps(report, indent=2) + "\n")
        # A crash between unpacking the application and its matching runtime
        # leaves an intentionally inconsistent split dependency pair. Repair
        # that state with the explicit local package set, without a network or
        # an unconstrained apt --fix-broken operation that could remove the app.
        self.command("recover-transaction", self.transaction("cuda", repair=True))
        # Force exact-file replay even if the interrupted database already
        # records the target as installed. Native metadata is not a hash check.
        self.command("recover-exact-files", self.transaction("cuda", reinstall=True))
        self.verify_variant("repaired-cuda", "cuda")

    def run(self) -> dict:
        try:
            recovery_inputs = self.native_recovery_inputs()
            mounts = []
            for variant in ("cpu", "cuda"):
                mounts += ["-v", f"{self.package_sets[variant]}:/artifacts/{variant}:ro",
                           "-v", f"{self.runtimes[variant].parent / 'manifest.json'}:/manifests/{variant}.json:ro"]
            mounts += ["-v", f"{self.runtimes['cuda']}:/payloads/cuda:ro"]
            mounts += ["-v", f"{recovery_inputs}:/native-recovery:ro"]
            subprocess.run(["docker", "run", "-d", "--init", "--pull=never", "--network=none",
                            "--name", self.name, "--tmpfs", "/tmp:rw,mode=1777", *mounts,
                            "-v", f"{HERE}:/native-prototype:ro", "-v", f"{RUNTIME_TOOLS}:/runtime-prototype:ro",
                            self.cell["image"], "sleep", "infinity"], capture_output=True, text=True, check=True)
            self.command("prepare-fixture", ["sh", "-c",
                         "getent passwd 1000 >/dev/null || useradd -M -u 1000 -d /tmp/nvb-home nvb-probe; "
                         "mkdir -p /home/nvb-prototype/.config/nvbroadcast /home/nvb-prototype/Videos/Broadcast; "
                         f"printf '%s\\n' 'user preferences' > {SENTINELS[0]}; "
                         f"printf '%s\\n' 'user recording sentinel' > {SENTINELS[1]}; "
                         "chown -R 1000:1000 /home/nvb-prototype"])
            before = self.command("user-files-before", ["sha256sum", *SENTINELS]).stdout
            python_command = ["sh", "-c", 'p=$(command -v python3 || true); [ -z "$p" ] || sha256sum "$p"']
            python_before = self.command("system-python-before", python_command).stdout
            self.command("native-inventory", ["dpkg-query", "-W"] if self.cell["family"] == "deb" else ["rpm", "-qa"])
            self.command("install-cpu", self.transaction("cpu"))
            self.verify_variant("initial-cpu", "cpu")
            if self.shape == "split":
                mismatch = self.command("incomplete-cuda-pair", self.transaction("cuda", only="app"), expected=None)
                if mismatch.returncode == 0:
                    raise RuntimeError("CUDA app without its runtime was accepted")
                self.verify_variant("after-rejected-pair", "cpu")
            self.command("switch-cuda", self.transaction("cuda"))
            self.verify_variant("switched-cuda", "cuda")
            self.command("switch-cpu", self.transaction("cpu"))
            self.verify_variant("switched-back-cpu", "cpu")
            self.interrupt_switch()
            self.command("rollback-cpu", self.transaction("cpu"))
            self.verify_variant("rolled-back-cpu", "cpu")
            names = [p["name"] for p in self.selection("cpu")]
            remove = (["apt-get", "-y", "purge"] if self.cell["family"] == "deb" else
                      ["dnf", "-y", "--disable-repo=*", "remove", "--no-autoremove"])
            self.command("remove", [*remove, *names])
            self.command("no-orphaned-runtime", ["test", "!", "-e", "/usr/lib/nvbroadcast"])
            self.command("no-launcher", ["test", "!", "-e", "/usr/bin/nvbroadcast"])
            if self.command("user-files-after", ["sha256sum", *SENTINELS]).stdout != before:
                raise RuntimeError("variant transactions modified user files")
            if self.command("system-python-after", python_command).stdout != python_before:
                raise RuntimeError("variant transactions modified system Python")
            self.result["status"] = "pass"
        except Exception as error:
            self.result.update(status="fail", error=f"{type(error).__name__}: {error}")
        finally:
            if self.keep_failed and self.result["status"] == "fail":
                self.result["retained_container"] = self.name
            else:
                subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self.save()
        return self.result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in ("cpu", "cuda"):
        parser.add_argument(f"--{variant}-packages", type=Path, required=True)
        parser.add_argument(f"--{variant}-runtime", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--shapes", nargs="+", choices=("self", "split"), default=("self", "split"))
    parser.add_argument("--keep-failed", action="store_true",
                        help="retain a failing isolated container for diagnosis; remove it manually afterwards")
    parser.add_argument("--fault-boundary", choices=("prepare", "rpm-unpack"), default="prepare")
    parser.add_argument("--skip-rpm-temporary-recovery", action="store_true",
                        help="reproduce RPM's unowned temporary-file defect without preserving the fragments")
    args = parser.parse_args()
    packages = {v: getattr(args, f"{v}_packages").resolve(strict=True) for v in ("cpu", "cuda")}
    runtimes = {v: getattr(args, f"{v}_runtime").resolve(strict=True) for v in ("cpu", "cuda")}
    records = {v: check_inputs(packages[v], runtimes[v], v) for v in packages}
    matrix = json.loads(args.matrix.read_text())
    if not matrix or any(not c["image"].startswith("sha256:") or len(c["image"]) != 71 for c in matrix):
        parser.error("matrix must contain inspected immutable image IDs")
    if args.fault_boundary == "rpm-unpack" and any(c["family"] != "rpm" for c in matrix):
        parser.error("rpm-unpack requires an RPM-only matrix")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(Switching(cell, shape, packages, runtimes, output,
                                       args.keep_failed, args.fault_boundary, args.skip_rpm_temporary_recovery).run)
                   for cell in matrix for shape in args.shapes]
        for future in futures:
            result = future.result()
            results.append(result)
            print(f"{result['cell']['name']} {result['shape']}: {result['status']} {result.get('error', '')}", flush=True)
            (output / "results.json").write_text(json.dumps({"packages": records, "results": results}, indent=2) + "\n")
    raise SystemExit(0 if all(r["status"] == "pass" for r in results) else 1)


if __name__ == "__main__":
    main()

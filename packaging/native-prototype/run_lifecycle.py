#!/usr/bin/env python3
"""Exercise real native managers against non-release packages without networking.

Only Docker containers are changed. No host devices, audio/display sockets,
user homes, or writable runtime payloads are mounted into the test systems.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
RUNTIME_TOOLS = HERE.parent / "runtime-prototype"
PREFIX = "/usr/lib/nvbroadcast/runtime"
SENTINELS = ("/home/nvb-prototype/.config/nvbroadcast/preferences.json",
             "/home/nvb-prototype/Videos/Broadcast/user-recording.mp4")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_packages(directory: Path) -> list[dict]:
    packages = json.loads((directory / "packages.json").read_text())
    keys = set()
    for package in packages:
        artifact = (directory / package["artifact"]).resolve(strict=True)
        if not artifact.is_relative_to(directory.resolve()):
            raise ValueError("artifact escapes package directory")
        if digest(artifact) != package["sha256"]:
            raise ValueError("package SHA-256 mismatch")
        key = package["family"], package["kind"], package["revision"]
        if key in keys:
            raise ValueError("duplicate package identity")
        keys.add(key)
    expected = {(family, kind, revision) for family in ("deb", "rpm")
                for kind in ("self", "app", "runtime") for revision in (1, 2)}
    if keys != expected:
        raise ValueError("incomplete package comparison set")
    return packages


class Lifecycle:
    def __init__(self, cell: dict, shape: str, packages: Path, runtime: Path, output: Path):
        self.cell, self.shape, self.packages, self.runtime = cell, shape, packages, runtime
        self.output = output / f"{cell['name']}-{shape}"
        self.output.mkdir(parents=True, exist_ok=False)
        self.name = "nvb-native-lifecycle-" + uuid.uuid4().hex
        self.records = json.loads((packages / "packages.json").read_text())
        self.result = {"cell": cell, "shape": shape, "steps": [], "status": "running"}

    def files(self, revision: int, kind: str | None = None) -> list[str]:
        kinds = {kind} if kind else ({"self"} if self.shape == "self" else {"app", "runtime"})
        return ["/artifacts/" + p["artifact"] for p in self.records
                if p["family"] == self.cell["family"] and p["revision"] == revision and p["kind"] in kinds]

    def command(self, label: str, command: list[str], *, expected: int | None = 0,
                user: bool = False, timeout: int = 240) -> subprocess.CompletedProcess:
        prefix = ["docker", "exec"]
        if user:
            prefix += ["--user", "1000:1000", "-e", "XDG_CACHE_HOME=/tmp/cache",
                       "-e", "XDG_CONFIG_HOME=/tmp/config", "-e", "XDG_STATE_HOME=/tmp/state",
                       "-e", "NVBROADCAST_NO_LOG_FILE=1", "-e", "OPENBLAS_NUM_THREADS=2",
                       "-e", "PYTHONPATH=/usr/lib/python3/dist-packages", "-e", "PYTHONHOME=/usr"]
        started = time.monotonic()
        run = subprocess.run([*prefix, self.name, *command], capture_output=True, text=True, timeout=timeout)
        (self.output / f"{label}.log").write_text(run.stdout + run.stderr)
        self.result["steps"].append({"name": label, "command": command, "exit": run.returncode,
                                     "seconds": round(time.monotonic() - started, 2)})
        self.save()
        if expected is not None and run.returncode != expected:
            raise RuntimeError(f"{label}: expected exit {expected}, got {run.returncode}; see {self.output}")
        return run

    def save(self) -> None:
        (self.output / "result.json").write_text(json.dumps(self.result, indent=2) + "\n")

    def manager(self, action: str, revision: int, *, kind: str | None = None) -> list[str]:
        files = self.files(revision, kind)
        if self.cell["family"] == "deb":
            extra = ["--reinstall"] if action == "reinstall" else []
            # Docker denies networking. APT's --no-download also suppresses
            # acquiring local .debs on some releases ("Pathname ... not absolute").
            return ["apt-get", "-y", "--allow-downgrades", *extra, "install", *files]
        # Minimal Fedora containers set tsflags=nodocs. Include package notices
        # here so the full native file-ownership comparison is meaningful.
        return ["dnf", "-y", "--disable-repo=*", "--setopt=tsflags=", action, *files]

    def verify(self, label: str, revision: int) -> None:
        run = self.command(label + "-integrity", [f"{PREFIX}/bin/python", "-I", "-B",
                           "/native-prototype/verify.py", self.cell["family"], self.shape, str(revision)])
        self.result["steps"][-1]["verification"] = json.loads(run.stdout.split("RESULT=", 1)[1])
        run = self.command(label + "-probe", ["bash", "/runtime-prototype/desktop_probe.sh",
                           f"{PREFIX}/bin/python", "-I", "-B", "-u", "/runtime-prototype/probe.py",
                           "--runtime", PREFIX, "--python-version", "3.13.16", "--window"], user=True)
        reports = [json.loads(line[7:]) for line in run.stdout.splitlines() if line.startswith("RESULT=")]
        if len(reports) != 1:
            raise RuntimeError("runtime probe omitted its report")
        source = RUNTIME_TOOLS.parents[1] / "src/nvbroadcast"
        expected = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))}
        if reports[0]["app_python_hashes"] != expected:
            raise RuntimeError("packaged application does not match this source checkout")
        self.result["steps"][-1]["verification"] = reports[0]
        self.command(label + "-launcher", ["bash", "/runtime-prototype/desktop_probe.sh",
                                           "/usr/bin/nvbroadcast", "--help"], user=True)

    def interrupted_upgrade(self) -> None:
        signal_status = self.command("sigkill-fixture", ["setsid", "--fork", "--wait", "sh", "-c",
                                                       "kill -KILL $$"], expected=None).returncode
        if signal_status == 0:
            raise RuntimeError("fixture did not observe SIGKILL")
        files = self.files(2)
        pm = (["dpkg", "--install", *files] if self.cell["family"] == "deb" else ["rpm", "-Uvh", *files])
        with (self.output / "interrupted-upgrade.log").open("w") as log:
            process = subprocess.Popen(["docker", "exec", self.name, "setsid", "--fork", "--wait", "sh", "-c",
                                        'echo $$ > /tmp/nvb-package-manager.pid; exec "$@"', "pm", *pm],
                                       stdout=log, stderr=subprocess.STDOUT)
            killed = False
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and process.poll() is None:
                    run = subprocess.run(["docker", "exec", self.name, "bash", "-c",
                                          'test -f /usr/lib/nvbroadcast/.transaction && '
                                          'kill -KILL -- "-$(cat /tmp/nvb-package-manager.pid)"'],
                                         capture_output=True, timeout=10)
                    if run.returncode == 0:
                        killed = True
                        break
                    time.sleep(0.05)
                status = process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    subprocess.run(["docker", "kill", self.name], capture_output=True)
                    process.wait(timeout=15)
            if not killed or status != signal_status:
                raise RuntimeError(f"interruption was not established: killed={killed}, exit={status}")
        self.result["steps"].append({"name": "interrupted-upgrade", "exit": status,
                                    "expected_signal_exit": signal_status,
                                    "fault": "SIGKILL package-manager process group after prepare marker"})
        self.command("interrupted-launch-blocked", ["/usr/bin/nvbroadcast", "--help"], expected=78, user=True)
        # Replaying the exact files repairs either an old database with partial
        # new bytes or a partially committed split pair. No dependency resolution.
        repair = (["dpkg", "--install", *files] if self.cell["family"] == "deb" else
                  ["rpm", "-Uvh", "--replacepkgs", *files])
        self.command("interrupted-repair", repair)
        self.command("repaired-dependencies", ["apt-get", "check"] if self.cell["family"] == "deb" else
                     ["dnf", "--disable-repo=*", "check"])
        self.verify("repaired", 2)

    def run(self) -> dict:
        try:
            subprocess.run(["docker", "run", "-d", "--init", "--pull=never", "--network=none",
                            "--name", self.name, "--tmpfs", "/tmp:rw,mode=1777",
                            "-v", f"{self.packages}:/artifacts:ro",
                            "-v", f"{self.runtime.parent / 'manifest.json'}:/payload-manifest.json:ro",
                            "-v", f"{HERE}:/native-prototype:ro", "-v", f"{RUNTIME_TOOLS}:/runtime-prototype:ro",
                            self.cell["image"], "sleep", "infinity"], capture_output=True, text=True, check=True)
            self.command("prepare-fixture", ["sh", "-c",
                         "getent passwd 1000 >/dev/null || useradd -M -u 1000 -d /tmp/nvb-home nvb-probe; "
                         "mkdir -p /home/nvb-prototype/.config/nvbroadcast /home/nvb-prototype/Videos/Broadcast; "
                         f"printf '%s\\n' 'user preferences' > {SENTINELS[0]}; "
                         f"printf '%s\\n' 'user recording sentinel' > {SENTINELS[1]}; "
                         "chown -R 1000:1000 /home/nvb-prototype"])
            before = self.command("user-files-before", ["sha256sum", *SENTINELS]).stdout
            python_before = self.command("system-python-before", ["sh", "-c",
                                         'p=$(command -v python3 || true); [ -z "$p" ] || sha256sum "$p"']).stdout
            inventory_command = ["dpkg-query", "-W"] if self.cell["family"] == "deb" else ["rpm", "-qa"]
            self.command("native-packages-before", inventory_command)
            if self.cell.get("legacy"):
                self.command("legacy-runtime-present", ["test", "-f", "/opt/nvbroadcast/.venv/pyvenv.cfg"])
            self.command("install", self.manager("install", 1))
            self.verify("installed", 1)
            self.command("legacy-runtime-removed", ["test", "!", "-e", "/opt/nvbroadcast/.venv"])
            if self.cell.get("legacy") and self.shape == "self" and self.cell["family"] == "deb":
                self.command("purge-legacy", ["dpkg", "--purge", "nvbroadcast"])
                self.verify("after-legacy-purge", 1)
            self.command("upgrade", self.manager("install", 2))
            self.verify("upgraded", 2)
            self.command("rollback", self.manager("downgrade", 1))
            self.verify("rolled-back", 1)
            if self.shape == "split":
                mismatch = self.command("mismatched-app-rejected", self.manager("install", 2, kind="app"), expected=None)
                if mismatch.returncode == 0:
                    raise RuntimeError("split app upgrade accepted an incompatible runtime")
                self.verify("after-mismatch", 1)
            self.command("reinstall", self.manager("reinstall", 1))
            self.verify("reinstalled", 1)
            self.interrupted_upgrade()
            names = ["nvbroadcast-cpu"] if self.shape == "self" else ["nvbroadcast", "nvbroadcast-runtime-cpu"]
            remove = (["apt-get", "-y", "remove"] if self.cell["family"] == "deb" else
                      ["dnf", "-y", "--disable-repo=*", "remove", "--no-autoremove"])
            self.command("remove-app", [*remove, names[0]])
            self.command("launcher-removed", ["test", "!", "-e", "/usr/bin/nvbroadcast"])
            if self.shape == "split":
                self.command("retained-runtime", ["test", "-x", f"{PREFIX}/bin/python"])
                self.command("remove-runtime", [*remove, names[1]])
            if self.cell["family"] == "deb":
                self.command("purge", ["apt-get", "-y", "purge", *names])
            self.command("no-orphaned-runtime", ["test", "!", "-e", "/usr/lib/nvbroadcast"])
            after = self.command("user-files-after", ["sha256sum", *SENTINELS]).stdout
            if before != after:
                raise RuntimeError("package lifecycle changed user configuration or recordings")
            python_after = self.command("system-python-after", ["sh", "-c",
                                        'p=$(command -v python3 || true); [ -z "$p" ] || sha256sum "$p"']).stdout
            if python_after != python_before:
                raise RuntimeError("system Python executable changed")
            self.command("native-packages-after", inventory_command)
            self.result["status"] = "pass"
        except Exception as error:
            self.result.update(status="fail", error=f"{type(error).__name__}: {error}")
        finally:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self.save()
        return self.result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True, help="JSON cells with name, family, immutable image ID, legacy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    packages = args.packages.resolve(strict=True)
    records = checked_packages(packages)
    matrix = json.loads(args.matrix.read_text())
    for cell in matrix:
        if not cell["image"].startswith("sha256:") or len(cell["image"]) != 71:
            parser.error("matrix must pin local image IDs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(Lifecycle(cell, shape, packages, args.runtime.resolve(strict=True), output).run)
                   for cell in matrix for shape in ("self", "split")]
        for future in futures:
            result = future.result()
            results.append(result)
            print(f"{result['cell']['name']} {result['shape']}: {result['status']} {result.get('error', '')}", flush=True)
            (output / "results.json").write_text(json.dumps({"packages": records, "results": results}, indent=2) + "\n")
    raise SystemExit(0 if all(r["status"] == "pass" for r in results) else 1)


if __name__ == "__main__":
    main()

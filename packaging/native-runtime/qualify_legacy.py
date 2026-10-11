#!/usr/bin/env python3
"""Qualify a published legacy package's migration in a disposable container.

Only baseline installation has network access. The migration, refusal and
runtime checks execute after disconnecting the container from every network.
No host devices, desktop sockets or user configuration are exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from qualify import selected


REPO = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("deb", "rpm"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--baseline-version", required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    if digest(baseline) != args.baseline_sha256:
        raise ValueError("published baseline digest mismatch")
    packages = args.packages.resolve(strict=True)
    target = selected(packages, args.family)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    image = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
    ).strip()
    if image != args.image or not image.startswith("sha256:"):
        raise ValueError("fixture must be selected by its exact local image digest")
    name = "nvb-legacy-migration-" + uuid.uuid4().hex[:12]
    result = {"status": "running", "family": args.family, "image": image,
              "baseline": {"name": baseline.name, "sha256": args.baseline_sha256,
                           "version": args.baseline_version}, "target": target,
              "harness_sha256": digest(Path(__file__)), "steps": [],
              "baseline_online_install": True, "migration_network_disabled": False,
              "hardware_verified": False}

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def execute(label, command, *, failure=False, user=False, detached=False, timeout=600):
        argv = ["docker", "exec"]
        if detached:
            argv.append("-d")
        if user:
            argv += ["--user", "1000:1000", "-e", "HOME=/tmp/nvb-home",
                     "-e", "XDG_CACHE_HOME=/tmp/nvb-home/.cache"]
        log = output / f"{len(result['steps']):02d}-{label}.log"
        started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.run(argv + [name, *command], stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=timeout)
        result["steps"].append({"name": label, "command": command, "log": log.name,
                                "sha256": digest(log), "exit_code": process.returncode,
                                "expected_failure": failure,
                                "seconds": round(time.monotonic() - started, 3)})
        save()
        if (process.returncode == 0) == failure:
            raise RuntimeError(f"unexpected result: {label}; see {log}")
        return log.read_text()

    def identity():
        return (["dpkg-query", "-W", "-f=${Version}", "nvbroadcast"]
                if args.family == "deb" else
                ["rpm", "-q", "--qf", "%{VERSION}-%{RELEASE}", "nvbroadcast"])

    def install(path, *, baseline=False):
        if baseline:
            return (["apt-get", "install", "--yes", "--no-remove", path]
                    if args.family == "deb" else ["dnf", "--assumeyes", "install", path])
        return (["dpkg", "-i", path] if args.family == "deb" else
                ["rpm", "-Uvh", "--replacepkgs", path])

    try:
        subprocess.run(["docker", "run", "-d", "--init", "--pull=never", "--name", name,
                        "--memory", "6g", "--cpus", "2", "-e", "PYTHONNOUSERSITE=1",
                        "-e", "PIP_PROGRESS_BAR=off",
                        "-e", "PULSE_SERVER=unix:/tmp/no-host-pulse-socket",
                        "-v", f"{baseline.parent}:/baseline:ro", "-v", f"{packages}:/candidate:ro",
                        "-v", f"{REPO}:/source:ro", image, "sleep", "infinity"],
                       check=True, stdout=subprocess.DEVNULL)
        inspected = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
        assert not inspected["HostConfig"]["Privileged"]
        assert not inspected["HostConfig"]["Devices"]
        result["mounts"] = inspected["Mounts"]
        execute("fixture-desktop-user", ["sh", "-eu", "-c",
            "getent group 1000 >/dev/null || groupadd -g 1000 nvbqualification; "
            "getent passwd 1000 >/dev/null || useradd -u 1000 -g 1000 -M "
            "-d /tmp/nvb-home -s /bin/sh nvbqualification"])
        execute("baseline-install", install("/baseline/" + baseline.name, baseline=True), timeout=1200)
        if execute("baseline-identity", identity()).strip() != args.baseline_version:
            raise RuntimeError("installed baseline version differs")
        execute("baseline-runtime", ["/opt/nvbroadcast/.venv/bin/python", "-m", "pip", "check"])
        execute("baseline-launcher", ["/usr/bin/nvbroadcast", "--help"], user=True)
        for network in inspected["NetworkSettings"]["Networks"]:
            subprocess.run(["docker", "network", "disconnect", network, name], check=True)
        isolated = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
        assert not isolated["NetworkSettings"]["Networks"]
        result["migration_network_disabled"] = True
        execute("sentinels", ["sh", "-eu", "-c",
            "mkdir -p /tmp/nvb-home/.config/nvbroadcast /tmp/nvb-home/Recordings; "
            "printf preserved > /tmp/nvb-home/.config/nvbroadcast/sentinel; "
            "printf preserved > /tmp/nvb-home/Recordings/sentinel; "
            "printf admin-preserved > /opt/nvbroadcast/admin-sentinel; "
            "chown -R 1000:1000 /tmp/nvb-home; "
            "sha256sum /usr/bin/python3 /usr/bin/nvbroadcast > /tmp/baseline-files.sha256; "
            "sha256sum /usr/bin/python3 > /tmp/system-python.sha256"])
        execute("start-legacy-process", ["/opt/nvbroadcast/.venv/bin/python", "-c",
            "import os,time; open('/tmp/legacy-process.pid','w').write(str(os.getpid())); time.sleep(600)"],
            user=True, detached=True)
        execute("wait-legacy-process", ["sh", "-eu", "-c",
            "i=0; while test ! -s /tmp/legacy-process.pid; do i=$((i+1)); test $i -lt 100; sleep .1; done"])
        target_path = "/candidate/" + target["artifact"]
        refusal = execute("refuse-running-legacy", install(target_path), failure=True)
        if "NVBroadcast is running" not in refusal:
            raise RuntimeError("upgrade failed for an unrelated reason")
        if execute("refusal-keeps-baseline", identity()).strip() != args.baseline_version:
            raise RuntimeError("running-process refusal changed package identity")
        execute("refusal-preserves-files-and-process", ["sh", "-eu", "-c",
            "sha256sum -c /tmp/baseline-files.sha256; test -d /opt/nvbroadcast/.venv; "
            "test ! -e /usr/lib/nvbroadcast/.transaction; kill -0 $(cat /tmp/legacy-process.pid)"])
        execute("stop-test-process", ["sh", "-eu", "-c",
            "pid=$(cat /tmp/legacy-process.pid); kill $pid; "
            "i=0; while kill -0 $pid 2>/dev/null; do i=$((i+1)); test $i -lt 100; sleep .1; done"])
        execute("offline-migration", install(target_path))
        if execute("migrated-identity", identity()).strip() != target["version"]:
            raise RuntimeError("migrated package version differs")
        execute("migrated-integrity", ["/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B",
            "/usr/lib/nvbroadcast/validate-install.py", "--variant", target["variant"]])
        execute("legacy-cleanup-user-preservation", ["sh", "-eu", "-c",
            "test ! -e /opt/nvbroadcast/.venv; test ! -e /usr/lib/nvbroadcast/.transaction; "
            "test ! -e /usr/lib/nvbroadcast/.legacy-runtime; "
            "test \"$(cat /opt/nvbroadcast/admin-sentinel)\" = admin-preserved; "
            "test \"$(cat /tmp/nvb-home/.config/nvbroadcast/sentinel)\" = preserved; "
            "test \"$(cat /tmp/nvb-home/Recordings/sentinel)\" = preserved; "
            "sha256sum -c /tmp/system-python.sha256"])
        execute("migrated-launcher", ["/usr/bin/nvbroadcast", "--help"], user=True)
        execute("migrated-inference-media-window", ["bash", "/source/packaging/runtime-prototype/desktop_probe.sh",
            "/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B", "/source/packaging/runtime-prototype/probe.py",
            "--runtime", "/usr/lib/nvbroadcast/runtime", "--python-version", "3.13.16", "--window",
            "--variant", target["variant"], "--lock",
            "/usr/lib/nvbroadcast/runtime/share/nvbroadcast-runtime-provenance/pylock.toml"], user=True)
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail", error=str(error))
        raise
    finally:
        cleanup = subprocess.run(["docker", "rm", "--force", name], capture_output=True, text=True)
        result["container_removed"] = cleanup.returncode == 0
        if cleanup.returncode and result["status"] == "pass":
            result.update(status="fail", error="qualification container cleanup failed")
        save()
        if result["status"] == "fail" and not cleanup.returncode == 0:
            raise RuntimeError(f"qualification container cleanup failed: {name}")


if __name__ == "__main__":
    main()

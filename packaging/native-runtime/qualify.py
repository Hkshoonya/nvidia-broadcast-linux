#!/usr/bin/env python3
"""Exercise exact native packages in disconnected disposable desktop fixtures.

Fixtures contain native OS dependencies, Xvfb and private synthetic PulseAudio.
No host display, camera, microphone, home or runtime directory is mounted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected(directory: Path, family: str) -> dict:
    packages = [p for p in json.loads((directory / "packages.json").read_text()) if p["family"] == family]
    if len(packages) != 1:
        raise ValueError("expected one package for selected family")
    package = packages[0]
    artifact = (directory / package["artifact"]).resolve(strict=True)
    if not artifact.is_relative_to(directory) or sha(artifact) != package["sha256"]:
        raise ValueError("package artifact identity mismatch")
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--upgrade", type=Path, required=True)
    parser.add_argument("--family", choices=("deb", "rpm"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_dir, second_dir = args.packages.resolve(strict=True), args.upgrade.resolve(strict=True)
    first, second = selected(first_dir, args.family), selected(second_dir, args.family)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    image = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    name = "nvb-native-qualification-" + uuid.uuid4().hex[:12]
    stages = []
    result = {"status": "running", "family": args.family, "image": image,
              "packages": [first, second], "steps": stages}

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    def execute(label: str, command: list[str], *, user=False, failure=False, detached=False):
        prefix = ["docker", "exec"]
        if detached:
            prefix.append("-d")
        if user:
            prefix += ["--user", "1000:1000", "-e", "HOME=/tmp/nvb-home", "-e", "XDG_CACHE_HOME=/tmp/nvb-home/.cache"]
        path = output / f"{len(stages):02d}-{label}.log"
        with path.open("w") as stream:
            process = subprocess.run(prefix + [name, *command], stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        stage = {"name": label, "command": command, "exit_code": process.returncode,
                 "expected_failure": failure, "log": path.name, "sha256": sha(path)}
        stages.append(stage)
        save()
        if (process.returncode == 0) == failure:
            raise RuntimeError(f"unexpected result: {label}; see {path}")
        return path.read_text()

    def install(package: dict, mount: str):
        path = mount + "/" + package["artifact"]
        return (["dpkg", "-i", path] if args.family == "deb" else ["rpm", "-Uvh", "--replacepkgs", "--oldpackage", path])

    def verify(label):
        execute(label + "-integrity", ["/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B",
            "/usr/lib/nvbroadcast/validate-install.py", "--variant", first["variant"]])
        execute(label + "-launcher", ["/usr/bin/nvbroadcast", "--help"], user=True)
        execute(label + "-inference-media-window", ["bash", "/source/packaging/runtime-prototype/desktop_probe.sh",
            "/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B", "/source/packaging/runtime-prototype/probe.py",
            "--runtime", "/usr/lib/nvbroadcast/runtime", "--python-version", "3.13.16", "--window",
            "--variant", first["variant"], "--lock", "/usr/lib/nvbroadcast/runtime/share/nvbroadcast-runtime-provenance/pylock.toml"], user=True)

    try:
        subprocess.run(["docker", "run", "-d", "--init", "--pull=never", "--network=none", "--name", name,
            "-v", f"{first_dir}:/first:ro", "-v", f"{second_dir}:/second:ro", "-v", f"{REPO}:/source:ro",
            image, "sleep", "infinity"], check=True, stdout=subprocess.DEVNULL)
        execute("sentinels", ["bash", "-eu", "-c",
            "mkdir -p /tmp/nvb-home/.config/nvbroadcast /tmp/nvb-home/Recordings; "
            "printf preserved > /tmp/nvb-home/.config/nvbroadcast/sentinel; "
            "printf preserved > /tmp/nvb-home/Recordings/sentinel; chown -R 1000:1000 /tmp/nvb-home; "
            "if test -e /usr/bin/python3; then sha256sum /usr/bin/python3 > /tmp/system-python.sha256; "
            "else touch /tmp/no-system-python; fi"])
        execute("install", install(first, "/first"))
        verify("installed")
        execute("start-held-runtime", ["/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B", "-c",
            'import os,time;open("/tmp/held-runtime.pid","w").write(str(os.getpid()));time.sleep(180)'], user=True, detached=True)
        execute("wait-held-runtime", ["sh", "-c", "until test -s /tmp/held-runtime.pid; do sleep 0.05; done"])
        execute("running-upgrade-refused", install(second, "/second"), failure=True)
        execute("stop-held-runtime", ["sh", "-c", 'kill "$(cat /tmp/held-runtime.pid)"; test ! -e /usr/lib/nvbroadcast/.transaction'])
        execute("preserved-running-selection", ["/usr/bin/nvbroadcast", "--help"], user=True)
        execute("upgrade", install(second, "/second"))
        verify("upgraded")
        execute("rollback", install(first, "/first"))
        verify("rolled-back")
        execute("reinstall", install(first, "/first"))
        # A required-component failure must block completion and launch until repair.
        execute("corrupt-required-file", ["bash", "-eu", "-c",
            "printf '\n# tampered\n' >> /usr/lib/nvbroadcast/runtime/lib/python3.13/site-packages/nvbroadcast/__init__.py; "
            "touch /usr/lib/nvbroadcast/.transaction"])
        execute("corruption-rejected", ["/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B",
            "/usr/lib/nvbroadcast/validate-install.py", "--variant", first["variant"]], failure=True)
        execute("corrupt-launch-blocked", ["/usr/bin/nvbroadcast", "--help"], user=True, failure=True)
        execute("corruption-repair", install(first, "/first"))
        verify("repaired")
        # Kill the actual native transaction after its launch-block marker.
        command = install(second, "/second")
        interruption_log = output / "interruption-manager.log"
        with interruption_log.open("w") as stream:
            manager = subprocess.Popen(["docker", "exec", name, "setsid", "--fork", "--wait", "sh", "-c",
                'echo $$ > /tmp/manager.pid; exec "$@"', "manager", *command], stdout=stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                ready = subprocess.run(["docker", "exec", name, "test", "-f", "/usr/lib/nvbroadcast/.transaction"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if ready.returncode == 0:
                    break
                if manager.poll() is not None:
                    raise RuntimeError("native transaction finished before interruption")
                time.sleep(0.025)
            else:
                raise RuntimeError("native transaction did not write its marker")
            execute("kill-package-manager", ["sh", "-c", 'kill -KILL -"$(cat /tmp/manager.pid)"'])
            manager.wait(timeout=30)
        execute("interrupted-launch-blocked", ["/usr/bin/nvbroadcast", "--help"], user=True, failure=True)
        execute("interrupted-repair", install(second, "/second"))
        verify("interruption-repaired")
        execute("remove", ["dpkg", "--purge", "nvbroadcast"] if args.family == "deb" else ["rpm", "-e", "nvbroadcast"])
        execute("preservation", ["bash", "-eu", "-c",
            "test ! -e /usr/lib/nvbroadcast/runtime; test ! -e /usr/bin/nvbroadcast; "
            "test ! -e /usr/lib/nvbroadcast/.transaction; "
            "test \"$(cat /tmp/nvb-home/.config/nvbroadcast/sentinel)\" = preserved; "
            "test \"$(cat /tmp/nvb-home/Recordings/sentinel)\" = preserved; "
            "if test -f /tmp/no-system-python; then test ! -e /usr/bin/python3; "
            "else sha256sum -c /tmp/system-python.sha256; fi"])
        result["status"] = "pass"
    except BaseException as error:
        result["status"] = "fail"
        result["error"] = str(error)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        save()
    print(json.dumps({"status": result["status"], "steps": len(stages), "image": image}))


if __name__ == "__main__":
    main()

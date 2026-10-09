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
    content = (directory / package["content"]).resolve(strict=True)
    if not content.is_relative_to(directory) or sha(content) != package["content_sha256"]:
        raise ValueError("package content manifest identity mismatch")
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
              "packages": [first, second], "steps": stages,
              "harness": {"file_sha256": sha(Path(__file__)),
                  "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}}

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
        execute("generated-recording", ["bash", "/source/packaging/runtime-prototype/desktop_probe.sh",
            "/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B", "/source/packaging/native-runtime/recording_probe.py"], user=True)
        execute("start-installed-launcher", ["bash", "/source/packaging/runtime-prototype/desktop_probe.sh",
            "sh", "-c", 'echo $$ > /tmp/installed-launcher.pid; exec /usr/bin/nvbroadcast'], user=True, detached=True)
        execute("verify-launcher-lifetime-lock", ["sh", "-eu", "-c",
            'for attempt in $(seq 1 100); do '
            'if test -s /tmp/installed-launcher.pid; then pid=$(cat /tmp/installed-launcher.pid); '
            'if test "$(readlink /proc/$pid/fd/9 2>/dev/null || true)" = /var/lib/nvbroadcast/runtime.lock '
            '&& tr "\\000" " " < /proc/$pid/cmdline | grep -q "python -I -B -m nvbroadcast"; '
            'then echo "Installed Python launcher retains runtime lock FD9"; exit 0; fi; fi; sleep 0.1; done; exit 1'], user=True)
        execute("launcher-lock-upgrade-refused", install(second, "/second"), failure=True)
        execute("stop-installed-launcher", ["sh", "-eu", "-c",
            'kill "$(cat /tmp/installed-launcher.pid)"; '
            'for attempt in $(seq 1 100); do if flock --exclusive --nonblock /var/lib/nvbroadcast/runtime.lock true; '
            'then test ! -e /usr/lib/nvbroadcast/.transaction; exit 0; fi; sleep 0.1; done; exit 1'])
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
        # Observe a large payload staging file before killing the unpacker.
        command = install(second, "/second")
        interruption_log = output / "interruption-manager.log"
        with interruption_log.open("w") as stream:
            manager = subprocess.Popen(["docker", "exec", name, "setsid", "--fork", "--wait", "sh", "-c",
                'echo $$ > /tmp/manager.pid; exec "$@"', "manager", *command], stdout=stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 120
            observed_path = ""
            tid_min = int(time.time())
            while time.monotonic() < deadline:
                pattern = "*;????????" if args.family == "rpm" else "*.dpkg-new"
                ready = subprocess.run(["docker", "exec", name, "sh", "-c",
                    'test -f /usr/lib/nvbroadcast/.transaction && find /usr/lib/nvbroadcast/runtime -type f -name "$1" -size +1M -print -quit',
                    "watch", pattern], capture_output=True, text=True)
                if ready.returncode == 0 and ready.stdout.strip():
                    observed_path = ready.stdout.strip()
                    break
                if manager.poll() is not None:
                    raise RuntimeError("native transaction finished before observing payload unpack")
                time.sleep(0.025)
            else:
                raise RuntimeError("native transaction did not expose a payload staging file")
            execute("kill-package-manager", ["sh", "-c", 'kill -KILL -"$(cat /tmp/manager.pid)"'])
            manager.wait(timeout=30)
            result["interruption"] = {"boundary": "mid-unpack", "observed_path": observed_path,
                "minimum_observed_bytes": 1048576, "manager_exit": manager.returncode,
                "tid_min": tid_min, "tid_max": int(time.time())}
            save()
        execute("interrupted-launch-blocked", ["/usr/bin/nvbroadcast", "--help"], user=True, failure=True)
        if args.family == "rpm":
            stage = "/second/rpm/stage"
            runtime = "/usr/lib/nvbroadcast/runtime"
            # The builder's content hash binds these retained read-only staging
            # bytes. Preserve only authenticated RPM partial-file prefixes;
            # unknown files abort recovery without broad cleanup.
            report_text = execute("preserve-rpm-unpack-fragments", [stage + runtime + "/bin/python", "-I", "-B",
                "/source/packaging/native-prototype/recover_rpm_unpacked.py", "--runtime", runtime,
                "--payload", stage + runtime, "--manifest", stage + "/usr/lib/nvbroadcast/runtime-manifest.json",
                "--baseline-manifest", "/first/rpm/stage/usr/lib/nvbroadcast/runtime-manifest.json",
                "--tid-min", str(result["interruption"]["tid_min"]), "--tid-max", str(result["interruption"]["tid_max"]),
                "--quarantine", "/var/lib/nvb-runtime-recovery", "--native-source", stage,
                "--native-manifest", "/second/" + second["content"], "--native-baseline", "/first/" + first["content"]])
            report = json.loads(report_text.split("RESULT=", 1)[1])
            if not report["preserved"]:
                raise RuntimeError("mid-unpack interruption left no authenticated RPM fragments")
            (output / "rpm-unpack-preservation.json").write_text(json.dumps(report, indent=2) + "\n")
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

#!/usr/bin/env python3
"""Qualify a reviewed native CPU/CUDA package pair in one offline fixture.

Default operation verifies local identities and prints a plan. --execute installs
only in a disposable container. RPMs may be unsigned development builds: this
proof does not establish publisher trust, hosted provenance or release readiness.
An explicit GPU UUID enables a separate hardware run; otherwise no GPU is exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tomllib
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("switching_smoke_helpers", HERE / "smoke_candidate.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
common = smoke.recovery
MAX_PACKAGE_BYTES = 8 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2
RUNTIME = "/usr/lib/nvbroadcast/runtime"
PYTHON = RUNTIME + "/bin/python"
ADAPTER_PATHS = ["packaging/native-runtime/" + name for name in (
    "build.py", "builders.json", "package-hooks.sh", "validate_install.py", "recovery_bootstrap.py",
    "pylock.linux-x86_64-cp313-cpu.toml", "pylock.linux-x86_64-cp313-cuda.toml")]
ADAPTER_PATHS += ["packaging/runtime-prototype/" + name for name in (
    "prepare.py", "assemble.py", "inputs.json", "pylock.build.toml", "build-bindings.sh")]
ADAPTER_PATHS += ["packaging/native-prototype/build.py", "scripts/recover_native_rpm.py",
                  "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc"]
INSTALLED_IDENTITY = '''from pathlib import Path
import hashlib,json,sys
expected=json.loads(Path('/probes/expected.json').read_text())[sys.argv[1]]
base=Path('/usr/lib/nvbroadcast')
actual=json.loads((base/'package.json').read_text())
assert actual==expected['package'], (actual,expected['package'])
assert hashlib.sha256((base/'runtime-manifest.json').read_bytes()).hexdigest()==expected['package']['runtime_manifest_sha256']
assert not (base/'.transaction').exists()
assert not Path('/var/lib/nvbroadcast/recovery-tools').exists()
lock=Path('/var/lib/nvbroadcast/runtime.lock').stat()
identity=str(lock.st_dev)+':'+str(lock.st_ino)
saved=Path('/tmp/runtime-lock.identity')
if saved.exists(): assert saved.read_text()==identity, 'runtime lock replaced'
else: saved.write_text(identity)
print(json.dumps({'status':'pass','package':actual,'runtime_lock_identity':identity}))
'''


def regular(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(fd, "rb")
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        stream.close()
        raise ValueError("expected a bounded regular file")
    return stream


def digest(path, maximum=MAX_PACKAGE_BYTES):
    with regular(path, maximum) as stream:
        before = os.fstat(stream.fileno())
        hasher, count = hashlib.sha256(), 0
        for block in iter(lambda: stream.read(1024**2), b""):
            count += len(block)
            if count > maximum:
                raise ValueError("input grew beyond bound")
            hasher.update(block)
        after = os.fstat(stream.fileno())
        fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if count != before.st_size or fields(before) != fields(after):
            raise ValueError("input changed while hashing")
        return hasher.hexdigest()


def json_file(path, maximum=MAX_METADATA_BYTES, expected=None):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    with regular(path, maximum) as stream:
        raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("metadata grew beyond bound")
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("package manifest identity mismatch")
    return json.loads(raw, object_pairs_hook=unique)



def save_result(path, result):
    """Publish complete JSON so cancellation leaves the last readable state."""
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def contained(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("invalid package path")
    path = directory / relative
    if not path.resolve(strict=True).is_relative_to(directory):
        raise ValueError("package path escapes input directory")
    return path


def snapshot(source, destination, expected):
    with regular(source, MAX_PACKAGE_BYTES) as original, destination.open("xb") as target:
        before = os.fstat(original.fileno())
        count = 0
        for block in iter(lambda: original.read(1024**2), b""):
            count += len(block)
            if count > MAX_PACKAGE_BYTES:
                raise ValueError("package grew beyond bound")
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
        after = os.fstat(original.fileno())
    destination.chmod(0o600)
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after) or digest(destination) != expected:
        raise ValueError("snapshot differs from reviewed input")


def selected(directory, manifest_sha256, family, variant, source):
    directory = directory.resolve(strict=True)
    manifest = directory / "packages.json"
    common.identity(manifest_sha256, 64)
    records = json_file(manifest, expected=manifest_sha256)
    if not isinstance(records, list):
        raise ValueError("expected package record list")
    matches = [p for p in records if p.get("family") == family]
    if len(matches) != 1:
        raise ValueError("expected exactly one selected native package")
    record = matches[0]
    if record["variant"] != variant or record["source"]["revision"] != source or record["adapter"]["revision"] != source:
        raise ValueError("package variant or source mismatch")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+-[1-9][0-9]*\." + variant, record["version"]):
        raise ValueError("invalid native package version")
    for key in ("sha256", "content_sha256", "runtime_manifest_sha256"):
        common.identity(record[key], 64)
    artifact = contained(directory, record["artifact"])
    content = contained(directory, record["content"])
    if digest(artifact) != record["sha256"] or artifact.stat().st_size != record["bytes"]:
        raise ValueError("package artifact identity mismatch")
    if digest(content, MAX_METADATA_BYTES) != record["content_sha256"]:
        raise ValueError("native content identity mismatch")
    return {"directory": str(directory), "manifest_sha256": manifest_sha256,
            "artifact_path": str(artifact), "record": record}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in ("cpu", "cuda"):
        parser.add_argument(f"--{variant}-packages", type=Path, required=True)
        parser.add_argument(f"--{variant}-manifest-sha256", required=True,
                            help="independently reviewed packages.json SHA-256")
    parser.add_argument("--source", required=True, help="exact application and adapter commit")
    parser.add_argument("--family", choices=("deb", "rpm"), required=True)
    parser.add_argument("--image", required=True, help="already provisioned amd64 fixture image ID")
    parser.add_argument("--gpu-device", help="explicit NVIDIA GPU UUID; absent means no GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def make_plan(args):
    common.identity(args.source, 40)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        raise ValueError("fixture must be an explicit image ID")
    common.identity(args.image[7:], 64)
    if args.gpu_device and not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu_device):
        raise ValueError("GPU selection must be an explicit UUID")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("use a fresh qualification output directory")
    if any(c in str(p) for p in (output, args.cpu_packages, args.cuda_packages) for c in ",\n\r"):
        raise ValueError("unsupported bind-mount path")
    packages = {v: selected(getattr(args, v + "_packages"), getattr(args, v + "_manifest_sha256"),
                           args.family, v, args.source) for v in ("cpu", "cuda")}
    cpu, cuda = (packages[v]["record"] for v in ("cpu", "cuda"))
    if cpu["version"].removesuffix(".cpu") != cuda["version"].removesuffix(".cuda"):
        raise ValueError("CPU and CUDA package revision mismatch")
    source = tomllib.loads(smoke.git_bytes(args.source, "pyproject.toml").decode())
    expected_files = {path: hashlib.sha256(smoke.git_bytes(args.source, path)).hexdigest() for path in ADAPTER_PATHS}
    for record in (cpu, cuda):
        if (record["source"]["version"] != source["project"]["version"]
                or record["version"].split("-", 1)[0] != source["project"]["version"]
                or record["adapter"]["files"] != expected_files):
            raise ValueError("package differs from source adapter or version")
    return {"schema_version": 1, "status": "planned-not-executed", "family": args.family,
            "source": args.source, "packages": packages, "image": args.image,
            "gpu_device": args.gpu_device, "output": str(output), "network": "none",
            "sequence": ["cpu", "cuda", "cpu"], "driver_sha256": digest(Path(__file__)),
            "shared_helpers": {p.name: digest(p) for p in (HERE / "smoke_candidate.py", HERE / "recover_candidate.py")},
            "scope": "Local development package switching and explicit GPU execution only; no signing, hosted provenance, physical camera/microphone, clean-VM, interruption or release acceptance.",
            "container_limits": {"memory": "6g", "memory_swap": "6g", "cpus": 2, "pids": 256},
            "maximum_step_log_bytes": common.MAX_LOG_BYTES, "steps": []}


def install_command(family, variant):
    path = f"/inputs/{variant}.{family}"
    return ["dpkg", "-i", path] if family == "deb" else ["rpm", "--upgrade", "--oldpackage", "--replacepkgs", path]


def probe_command(variant, gpu_device, python_version):
    command = ["bash", "/probes/desktop_probe.sh", PYTHON, "-I", "-B", "/probes/probe.py",
               "--runtime", RUNTIME, "--python-version", python_version, "--window", "--variant", variant,
               "--lock", RUNTIME + "/share/nvbroadcast-runtime-provenance/pylock.toml"]
    if variant == "cuda" and not gpu_device:
        command.append("--cuda-unavailable")
    return command


def runtime_report(text, variant, gpu_device, expected_app):
    reports = [json.loads(line[7:]) for line in text.splitlines() if line.startswith("RESULT=")]
    if len(reports) != 1:
        raise ValueError("expected one runtime probe report")
    report = reports[0]
    if report["status"] != "pass" or report["variant"] != variant or report["app_python_hashes"] != expected_app or not report["cpu_execution"]["success"]:
        raise ValueError("runtime probe identity or CPU execution mismatch")
    if variant == "cuda":
        available = bool(gpu_device)
        if report["cuda_execution"]["success"] != available or report["cuda_unavailable_expected"] == available:
            raise ValueError("unexpected CUDA execution outcome")
        if available and not report["cupy_execution"]:
            raise ValueError("missing genuine CuPy execution")
    return report


def container_command(plan, name, snapshots, probes):
    command = ["docker", "run", "-d", "--init", "--pull=never", "--network=none", "--name", name,
               "--memory", "6g", "--memory-swap", "6g", "--cpus", "2", "--pids-limit", "256"]
    if plan["gpu_device"]:
        command += ["--runtime=nvidia", "--gpus", "device=" + plan["gpu_device"],
                    "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
    else:
        command += ["--runtime=runc", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
    for variant, path in snapshots.items():
        command += ["--mount", f"type=bind,source={path},target=/inputs/{variant}.{plan['family']},readonly"]
    command += ["--mount", f"type=bind,source={probes},target=/probes,readonly",
                plan["image"], "sleep", "infinity"]
    return command


def execute(plan):
    output = Path(plan["output"])
    common.inspect_image(plan["image"])
    existing = output.parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < 32 * 1024**3:
        raise ValueError("switching qualification requires 32 GiB free; no automatic pruning")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.chmod(0o700)
    probes = output / "probes"
    probes.mkdir(mode=0o755)
    probes.chmod(0o755)
    tool_source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=30).strip()
    common.identity(tool_source, 40)
    for name, relative in smoke.PROBES.items():
        (probes / name).write_bytes(smoke.git_bytes(tool_source, relative))
        (probes / name).chmod(0o644)
    expected = {v: {"package": {k: item["record"][k] for k in ("variant", "version", "source", "adapter", "runtime_manifest_sha256")}}
                for v, item in plan["packages"].items()}
    for value in expected.values():
        value["package"]["schema_version"] = 1
    (probes / "expected.json").write_text(json.dumps(expected, sort_keys=True) + "\n")
    (probes / "expected.json").chmod(0o644)
    probe_hashes = {p.name: digest(p) for p in probes.iterdir()}
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", plan["source"], "--", "src/nvbroadcast"], cwd=ROOT, text=True, timeout=30).splitlines()
    expected_app = {p.removeprefix("src/nvbroadcast/"): hashlib.sha256(smoke.git_bytes(plan["source"], p)).hexdigest() for p in paths if p.endswith(".py")}
    python_version = json.loads(smoke.git_bytes(plan["source"], "packaging/runtime-prototype/inputs.json"))["python"]["version"]
    snapshots = {v: output / (v + "." + plan["family"]) for v in plan["packages"]}
    for variant, path in snapshots.items():
        record = plan["packages"][variant]
        snapshot(Path(record["artifact_path"]), path, record["record"]["sha256"])
    name = "nvb-native-switch-" + uuid.uuid4().hex[:12]
    plan.update(status="running", container=name, qualification_tool_source=tool_source,
                probe_snapshot_sha256=probe_hashes, runtime_probes={}, recordings={})
    created = False

    def save():
        save_result(output / "result.json", plan)

    def step(label, command, *, user=False, expected=0, detached=False):
        prefix = ["docker", "exec"]
        if detached:
            prefix.append("-d")
        if user:
            prefix += ["--user", "1000:1000", "-e", "HOME=/tmp/nvb-home", "-e", "XDG_CACHE_HOME=/tmp/nvb-home/.cache"]
        log = output / f"{len(plan['steps']):02d}-{label}.log"
        try:
            completed = common.run_capped([*prefix, name, *command], timeout=600, log=log)
        except BaseException as error:
            failed = {"name": label, "error": str(error), "expected_exit": expected, "log": log.name}
            if log.exists():
                failed["sha256"] = digest(log)
            plan["steps"].append(failed)
            save()
            raise
        plan["steps"].append({"name": label, "exit_code": completed.returncode, "expected_exit": expected,
                              "log": log.name, "sha256": digest(log)})
        save()
        if completed.returncode != expected:
            raise RuntimeError(f"unexpected result: {label}; see {log}")
        return (completed.stdout + completed.stderr).decode()

    def verify(label, variant, recording=True):
        step(label + "-identity", [PYTHON, "-I", "-B", "-c", INSTALLED_IDENTITY, variant])
        step(label + "-integrity", [PYTHON, "-I", "-B", "/usr/lib/nvbroadcast/validate-install.py", "--variant", variant])
        step(label + "-native-package-state", ["/usr/lib/nvbroadcast/check-packages"])
        step(label + "-launcher", ["/usr/bin/nvbroadcast", "--help"], user=True)
        text = step(label + "-runtime-media-gtk", probe_command(variant, plan["gpu_device"], python_version), user=True)
        plan["runtime_probes"][label] = runtime_report(text, variant, plan["gpu_device"], expected_app)
        if recording:
            text = step(label + "-generated-recording", ["bash", "/probes/desktop_probe.sh", PYTHON, "-I", "-B", "/probes/recording_probe.py"], user=True)
            reports = [json.loads(line[7:]) for line in text.splitlines() if line.startswith("RESULT=")]
            if len(reports) != 1 or reports[0]["status"] != "pass":
                raise ValueError("missing generated recording result")
            plan["recordings"][label] = reports[0]
        save()

    def refuse_running_switch(current, target):
        label = current + "-to-" + target
        step(label + "-start-launcher", ["bash", "/probes/desktop_probe.sh", "sh", "-c", 'echo $$ > /tmp/installed-launcher.pid; exec /usr/bin/nvbroadcast'], user=True, detached=True)
        step(label + "-verify-launcher-lock", ["sh", "-eu", "-c", 'for attempt in $(seq 1 100); do '
             'if test -s /tmp/installed-launcher.pid; then pid=$(cat /tmp/installed-launcher.pid); '
             'if test "$(readlink /proc/$pid/fd/9 2>/dev/null || true)" = /var/lib/nvbroadcast/runtime.lock '
             '&& tr "\\000" " " < /proc/$pid/cmdline | grep -q "python -I -B -m nvbroadcast"; then exit 0; fi; fi; sleep 0.1; done; exit 1'], user=True)
        refusal = step(label + "-switch-refused", install_command(plan["family"], target), expected=1)
        if "NVBroadcast is running" not in refusal:
            raise ValueError("switch failed without the required running-process refusal")
        step(label + "-stop-launcher", ["sh", "-eu", "-c", 'kill "$(cat /tmp/installed-launcher.pid)"; '
             'for attempt in $(seq 1 100); do if flock --exclusive --nonblock /var/lib/nvbroadcast/runtime.lock true; '
             'then test ! -e /usr/lib/nvbroadcast/.transaction; exit 0; fi; sleep 0.1; done; exit 1'])
        verify("preserved-" + current + "-after-refusal", current, recording=False)

    try:
        save()
        subprocess.run(container_command(plan, name, snapshots, probes), check=True, stdout=subprocess.DEVNULL, timeout=90)
        created = True
        step("confirm-clean-fixture", ["dpkg-query", "-W", "nvbroadcast"] if plan["family"] == "deb" else ["rpm", "-q", "nvbroadcast"], expected=1)
        step("sentinels", ["sh", "-eu", "-c", "mkdir -p /tmp/nvb-home/.config/nvbroadcast /tmp/nvb-home/Recordings; "
             "printf preserved > /tmp/nvb-home/.config/nvbroadcast/sentinel; printf preserved > /tmp/nvb-home/Recordings/sentinel; chown -R 1000:1000 /tmp/nvb-home"])
        step("capture-system-python", ["sh", "-eu", "-c", smoke.SYSTEM_PYTHON_CAPTURE, "capture", "/usr/bin/python3", "/tmp/system-python"])
        step("install-cpu", install_command(plan["family"], "cpu"))
        verify("installed-cpu", "cpu")
        refuse_running_switch("cpu", "cuda")
        step("switch-to-cuda", install_command(plan["family"], "cuda"))
        verify("installed-cuda", "cuda")
        refuse_running_switch("cuda", "cpu")
        step("rollback-to-cpu", install_command(plan["family"], "cpu"))
        verify("rolled-back-cpu", "cpu")
        step("remove-package", ["dpkg", "--purge", "nvbroadcast"] if plan["family"] == "deb" else ["rpm", "--erase", "nvbroadcast"])
        step("package-uninstalled", ["dpkg-query", "-W", "nvbroadcast"] if plan["family"] == "deb" else ["rpm", "-q", "nvbroadcast"], expected=1)
        step("preservation", ["sh", "-eu", "-c", "test ! -e /usr/lib/nvbroadcast/runtime; test ! -e /usr/bin/nvbroadcast; "
             "test ! -e /usr/lib/nvbroadcast/.transaction; test ! -e /var/lib/nvbroadcast/recovery-tools; "
             'test "$(cat /tmp/nvb-home/.config/nvbroadcast/sentinel)" = preserved; test "$(cat /tmp/nvb-home/Recordings/sentinel)" = preserved; '
             'test "$(cat /tmp/runtime-lock.identity)" = "$(stat -c %d:%i /var/lib/nvbroadcast/runtime.lock)"'])
        step("preserve-system-python", ["sh", "-eu", "-c", smoke.SYSTEM_PYTHON_VERIFY, "verify", "/usr/bin/python3", "/tmp/system-python"])
        plan["status"] = "pass"
    except BaseException as error:
        plan["status"], plan["error"] = "fail", str(error)
        raise
    finally:
        try:
            removed = subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=60)
            remaining = subprocess.check_output(["docker", "container", "ls", "--all", "--quiet", "--filter", f"name=^/{name}$"], text=True, timeout=30)
            plan["container_removed"] = not remaining.strip() and (not created or removed.returncode == 0)
            plan["inputs_unchanged"] = all(digest(Path(p["artifact_path"])) == p["record"]["sha256"] for p in plan["packages"].values())
            plan["snapshots_unchanged"] = all(digest(path) == plan["packages"][v]["record"]["sha256"] for v, path in snapshots.items())
            plan["probes_unchanged"] = {p.name: digest(p) for p in probes.iterdir()} == probe_hashes
            if not all(plan[k] for k in ("container_removed", "inputs_unchanged", "snapshots_unchanged", "probes_unchanged")):
                raise RuntimeError("cleanup or input identity recheck failed")
        except BaseException as error:
            plan["status"], plan["cleanup_error"] = "fail", str(error)
            raise
        finally:
            save()
    return plan


def main(argv=None):
    args = parse_arguments(argv)
    plan = make_plan(args)
    if args.execute:
        execute(plan)
        print(json.dumps({"status": plan["status"], "steps": len(plan["steps"]), "result": str(Path(plan["output"]) / "result.json")}))
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

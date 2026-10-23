#!/usr/bin/env python3
"""One signed CPU RPM first-install interruption/recovery proof.

Default operation validates local inputs and prints a plan. --execute is a
separate, explicit action after the exact signed inputs and driver are reviewed.
All package operations run inside one disposable, network-disabled container.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import threading
import time
import uuid


PUBLIC_SHA = "bf9371c4fc83c41f8b1dbc3607c25ffa261b1e5de99fe799cf0a863db16cbd8a"
PRIMARY = "A691DCA6C031FEFD2C34F5AAFE33D0C20C9451D1"
BASE_DRIVER_SHA = "7d65e0d1f0a066f9ecbbf7bb5023262f4046d9ef420e63113b37c56f996a0e53"
ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc"
MAX_RPM_BYTES = 2 * 1024**3
MAX_LOG_BYTES = 8 * 1024**2

# Decode, but never extract or execute, the three binding records. The decoder
# has its own watchdog in addition to the outer Docker command's timeout.
PAYLOAD_BINDINGS = r'''
import hashlib,json,subprocess,sys,tarfile,threading
limits={"usr/lib/nvbroadcast/package.json":1024**2,
        "usr/lib/nvbroadcast/runtime-manifest.json":16*1024**2,
        "usr/bin/nvbroadcast-recover-rpm":128*1024}
found={}
process=subprocess.Popen(["rpm2archive","--nocompression","/inputs/target.rpm"],stdout=subprocess.PIPE)
def stop():
 try: process.kill()
 except ProcessLookupError: pass
timer=threading.Timer(300,stop); timer.daemon=True; timer.start()
try:
 total=0; count=0
 with tarfile.open(fileobj=process.stdout,mode="r|") as archive:
  for member in archive:
   count+=1; total+=member.size
   assert count<=100000 and 0<=member.size and total<=20*1024**3, "payload bounds exceeded"
   name=member.name.removeprefix("./")
   if name not in limits: continue
   assert name not in found and member.isfile() and member.size<=limits[name], "invalid binding record"
   stream=archive.extractfile(member)
   value=stream.read(limits[name]+1)
   assert len(value)==member.size, "short binding record"
   found[name]=value
 while process.stdout.read(1024**2): pass
 assert process.wait(timeout=30)==0, "payload decoder failed"
 assert set(found)==set(limits), "missing binding records"
 print(json.dumps({"package":json.loads(found["usr/lib/nvbroadcast/package.json"]),
   "runtime_manifest_sha256":hashlib.sha256(found["usr/lib/nvbroadcast/runtime-manifest.json"]).hexdigest(),
   "helper_sha256":hashlib.sha256(found["usr/bin/nvbroadcast-recover-rpm"]).hexdigest()}))
finally:
 timer.cancel(); process.stdout.close()
 if process.poll() is None: process.kill(); process.wait(timeout=30)
'''

INSTALLED_BINDINGS = r'''
from pathlib import Path
import hashlib,json
base=Path("/usr/lib/nvbroadcast")
print(json.dumps({"package":json.loads((base/"package.json").read_text()),
 "runtime_manifest_sha256":hashlib.file_digest((base/"runtime-manifest.json").open("rb"),"sha256").hexdigest(),
 "helper_sha256":hashlib.file_digest(Path("/usr/bin/nvbroadcast-recover-rpm").open("rb"),"sha256").hexdigest()}))
'''

KILL_MANAGER = r'''
from pathlib import Path
import json,os,signal,stat,sys
pid=int(Path("/tmp/manager.pid").read_text())
assert pid>1 and os.getpgid(pid)==pid, "not the dedicated manager process group"
command=Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
assert command[:3]==[b"rpm",b"--upgrade",b"/inputs/target.rpm"], "unexpected process"
marker=Path("/usr/lib/nvbroadcast/.transaction")
assert marker.read_text()==sys.argv[2]+"\n", "transaction marker changed"
fragment=Path(sys.argv[1]); info=fragment.lstat()
assert stat.S_ISREG(info.st_mode) and info.st_size>1048576, "no genuine large unpack fragment"
os.killpg(pid,signal.SIGKILL)
print(json.dumps({"manager_pid":pid,"observed_path":str(fragment),"observed_bytes":info.st_size,
                  "signal":"SIGKILL","fresh_install":True}))
'''

PRESERVED_BYTES = r'''
from pathlib import Path
import hashlib,json,os,sys
base=Path(sys.argv[1]).parent
report=json.loads((base/"recovery.json").read_text())
assert report["status"]=="preserved; package replay and validation still required"
assert report["package_sha256"]==sys.argv[2]
assert report["version"]==sys.argv[3] and report["fingerprint"]==sys.argv[4]
entries={entry["original"]:entry for entry in report["entries"]}
assert report["preserved"]
for original in report["preserved"]:
 entry=entries[original]; file=base/"files"/original.lstrip("/")
 assert not Path(original).exists() and not Path(original).is_symlink()
 if "symlink" in entry: assert os.readlink(file)==entry["symlink"]
 else:
  assert hashlib.file_digest(file.open("rb"),"sha256").hexdigest()==entry["sha256"]
  assert file.stat().st_size==entry["size"]
assert hashlib.file_digest((base/"package.rpm").open("rb"),"sha256").hexdigest()==sys.argv[2]
assert hashlib.file_digest((base/"public.asc").open("rb"),"sha256").hexdigest()==sys.argv[5]
assert Path("/usr/lib/nvbroadcast/.transaction").read_text()==sys.argv[3]+"\n"
print("preserved bytes verified:",len(report["preserved"]))
'''


def digest(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RPM_BYTES:
            raise ValueError("expected bounded regular input")
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value, length):
    if not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) or set(value) == {"0"}:
        raise ValueError(f"expected explicit nonzero {length}-digit identity")
    return value


def require_project_signature(text):
    if not re.search(r"\bSignature\b[^\n]*:\s*OK\s*$", text, re.MULTILINE | re.IGNORECASE):
        raise ValueError("no verified OpenPGP signature in project-only RPM keyring")
    if re.search(r"\b(?:NOKEY|BAD|NOTTRUSTED|UNTRUSTED)\b|\bNOT\s+OK\b", text, re.IGNORECASE):
        raise ValueError("RPM signature or integrity verification failed")


def check_bindings(record, plan):
    package = record["package"]
    if (package["schema_version"] != 1 or package["variant"] != "cpu"
            or package["version"] != plan["package_version_release"]
            or package["source"]["revision"] != plan["source_revision"]
            or package["adapter"]["revision"] != plan["source_revision"]
            or package["source"]["version"] != plan["package_version"]
            or package["runtime_manifest_sha256"] != plan["runtime_manifest_sha256"]
            or record["runtime_manifest_sha256"] != plan["runtime_manifest_sha256"]
            or record["helper_sha256"] != plan["helper_sha256"]):
        raise ValueError("signed payload does not match the approved hosted source/runtime/helper")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpm", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="root-reviewed signed RPM SHA-256")
    parser.add_argument("--source-revision", required=True, help="exact hosted application AND adapter commit")
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True, help="rendered helper payload SHA-256, not the template")
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--package-revision", required=True)
    parser.add_argument("--image", required=True, help="already provisioned sha256 fixture image ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="run only after review of these exact inputs")
    return parser.parse_args(argv)


def make_plan(args):
    package = args.rpm.absolute()
    output = args.output.absolute()
    identity(args.sha256, 64)
    identity(args.source_revision, 40)
    identity(args.runtime_manifest_sha256, 64)
    identity(args.helper_sha256, 64)
    if not args.image.startswith("sha256:"):
        raise ValueError("fixture must be an explicit image ID")
    identity(args.image.removeprefix("sha256:"), 64)
    if not re.fullmatch(r"[1-9][0-9]*", args.package_revision):
        raise ValueError("package revision must be a positive integer")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", args.package_version):
        raise ValueError("invalid package version")
    if output.exists() or output.is_symlink():
        raise ValueError("use a fresh proof output directory")
    if any(c in str(path) for path in (package, KEY, output) for c in ",\n\r"):
        raise ValueError("unsupported bind-mount path")
    if digest(package) != args.sha256 or digest(KEY) != PUBLIC_SHA:
        raise ValueError("local signed input or project public key identity mismatch")
    return {"status": "planned-not-executed", "scenario": "fresh-install-only",
            "image": args.image, "package": str(package), "package_sha256": args.sha256,
            "package_version": args.package_version,
            "package_version_release": args.package_version + "-" + args.package_revision + ".cpu",
            "source_revision": args.source_revision,
            "runtime_manifest_sha256": args.runtime_manifest_sha256,
            "helper_sha256": args.helper_sha256, "public_key_sha256": PUBLIC_SHA,
            "publisher_fingerprint": PRIMARY, "output": str(output),
            "driver_sha256": digest(Path(__file__)), "based_on_driver_sha256": BASE_DRIVER_SHA,
            "no_host_package_changes": True, "no_build_staging_or_source_mounted": True,
            "network": "none", "container_limits": {"memory": "4g", "memory_swap": "4g", "cpus": 2, "pids": 256},
            "maximum_step_log_bytes": MAX_LOG_BYTES, "authenticated": False, "steps": []}


def snapshot(source, destination, expected):
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as original, destination.open("xb") as copy:
        before = os.fstat(original.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_RPM_BYTES:
            raise ValueError("invalid snapshot input")
        copied = 0
        for block in iter(lambda: original.read(1024**2), b""):
            copied += len(block)
            if copied > MAX_RPM_BYTES:
                raise ValueError("snapshot input grew beyond bound")
            copy.write(block)
        copy.flush()
        os.fsync(copy.fileno())
        after = os.fstat(original.fileno())
    destination.chmod(0o600)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("snapshot input changed")
    if digest(destination) != expected:
        raise ValueError("snapshot differs from reviewed input")


def run_capped(command, *, timeout, limit=MAX_LOG_BYTES, log=None, cwd=None):
    """Bound combined output before retaining it in memory or a log file."""
    stream = log.open("xb") if log is not None else None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    except BaseException:
        if stream is not None:
            stream.close()
        raise
    buffers = {"stdout": [], "stderr": []}
    total = 0
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for label, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, label)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(min(0.1, remaining)):
                    block = os.read(key.fd, min(65536, limit - total + 1))
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    available = limit - total
                    if stream is not None:
                        stream.write(block[:available])
                    if len(block) > available:
                        raise RuntimeError("command exceeded bounded output limit")
                    buffers[key.data].append(block)
                    total += len(block)
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
        return subprocess.CompletedProcess(command, process.returncode,
            b"".join(buffers["stdout"]), b"".join(buffers["stderr"]))
    finally:
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        finally:
            process.stdout.close()
            process.stderr.close()
            if stream is not None:
                stream.close()


def inspect_image(image):
    completed = run_capped(["docker", "image", "inspect", image], timeout=30, limit=1024**2)
    completed.check_returncode()
    records = json.loads(completed.stdout)
    if len(records) != 1 or (records[0]["Id"], records[0]["Os"], records[0]["Architecture"]) != (image, "linux", "amd64"):
        raise ValueError("fixture image identity or platform mismatch")


def execute(plan):
    image_id = plan["image"]
    output = Path(plan["output"])
    package = Path(plan["package"])
    inspect_image(image_id)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir(mode=0o700)
    snapshot(package, inputs / "target.rpm", plan["package_sha256"])
    snapshot(KEY, inputs / "public.asc", PUBLIC_SHA)
    name = "nvb-hosted-cpu-fresh-" + uuid.uuid4().hex[:12]
    plan.update(status="running", container=name)
    bootstrap = "/var/lib/nvbroadcast/recovery-tools/recover-" + plan["helper_sha256"] + ".py"
    manager = None
    manager_reader = None
    manager_errors = []

    def save():
        (output / "result.json").write_text(json.dumps(plan, indent=2) + "\n")

    def step(label, command, *, failure=False, required=None, timeout=600):
        log = output / (str(len(plan["steps"])).zfill(2) + "-" + label + ".log")
        try:
            completed = run_capped(["docker", "exec", name, *command], timeout=timeout, log=log)
        except BaseException as error:
            plan["steps"].append({"name": label, "error": str(error), "log": log.name})
            save()
            raise
        plan["steps"].append({"name": label, "exit_code": completed.returncode,
            "expected_failure": failure, "log": log.name, "log_sha256": digest(log)})
        save()
        if completed.returncode != (1 if failure else 0):
            raise RuntimeError(f"unexpected result: {label}; see {log}")
        text = (completed.stdout + completed.stderr).decode()
        if required and required not in text:
            raise RuntimeError(f"expected diagnostic absent: {label}; see {log}")
        return completed.stdout.decode()

    try:
        save()
        subprocess.run(["docker", "run", "-d", "--init", "--pull=never", "--network=none", "--name", name,
            "--memory", "4g", "--memory-swap", "4g", "--cpus", "2", "--pids-limit", "256",
            "--mount", f"type=bind,source={inputs / 'target.rpm'},target=/inputs/target.rpm,readonly",
            "--mount", f"type=bind,source={inputs / 'public.asc'},target=/inputs/public.asc,readonly",
            image_id, "sleep", "infinity"], check=True, stdout=subprocess.DEVNULL, timeout=60)
        step("confirm-clean-fixture", ["rpm", "-q", "nvbroadcast"], failure=True,
             required="package nvbroadcast is not installed")
        step("clean-private-paths", ["sh", "-eu", "-c",
             "test ! -e /usr/lib/nvbroadcast; test ! -e /usr/bin/nvbroadcast; test ! -e /var/lib/nvbroadcast/recovery-tools"])
        step("system-python", ["/usr/bin/python3", "-IB", "-c", "import sys; assert sys.version_info >= (3,11); print(sys.version)"])
        step("private-verification-keyring", ["mkdir", "-m", "700", "/tmp/nvb-project-rpmdb"])
        step("import-only-project-public-key", ["rpmkeys", "--dbpath", "/tmp/nvb-project-rpmdb", "--import", "/inputs/public.asc"])
        signature = step("authenticate-signed-target", ["rpmkeys", "--dbpath", "/tmp/nvb-project-rpmdb", "--checksig", "--verbose", "/inputs/target.rpm"])
        require_project_signature(signature)
        metadata = step("signed-rpm-metadata", ["rpm", "-qp", "--qf", "%{NAME}\n%{VERSION}-%{RELEASE}\n%{ARCH}\n", "/inputs/target.rpm"])
        if metadata.splitlines() != ["nvbroadcast", plan["package_version_release"], "x86_64"]:
            raise ValueError("expected exact reviewed CPU package version")
        binding = json.loads(step("authenticated-payload-bindings", ["/usr/bin/python3", "-IB", "-c", PAYLOAD_BINDINGS]))
        check_bindings(binding, plan)
        plan["authenticated"] = True
        save()
        step("import-install-public-key", ["rpm", "--import", "/inputs/public.asc"])
        manager_log = output / "interrupted-manager.log"
        with manager_log.open("xb") as stream:
            manager = subprocess.Popen(["docker", "exec", name, "setsid", "--fork", "--wait", "sh", "-c",
                "echo $$ > /tmp/manager.pid; exec rpm --upgrade /inputs/target.rpm"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            def read_manager():
                total = 0
                try:
                    while block := manager.stdout.read(65536):
                        stream.write(block[:MAX_LOG_BYTES - total])
                        total += len(block)
                        if total > MAX_LOG_BYTES:
                            raise RuntimeError("manager exceeded bounded output limit")
                except BaseException as error:
                    manager_errors.append(str(error))
                    if manager.poll() is None:
                        manager.kill()

            manager_reader = threading.Thread(target=read_manager, daemon=True)
            manager_reader.start()
            deadline = time.monotonic() + 180
            observed = ""
            while time.monotonic() < deadline:
                if manager_errors:
                    raise RuntimeError(manager_errors[0])
                probe = run_capped(["docker", "exec", name, "sh", "-c",
                    'test -f /usr/lib/nvbroadcast/.transaction && find /usr/lib/nvbroadcast/runtime -type f -name "*;????????" -size +1M -print -quit'],
                    timeout=15, limit=16384)
                if probe.returncode == 0 and probe.stdout.strip():
                    observed = probe.stdout.decode().strip()
                    break
                if manager.poll() is not None:
                    raise RuntimeError("manager finished before genuine mid-unpack observation")
                time.sleep(0.025)
            if not re.fullmatch(r"/usr/lib/nvbroadcast/runtime/[^\n]+;[0-9a-f]{8}", observed):
                raise RuntimeError("no bounded transaction fragment was observed")
            killed = step("sigkill-observed-first-install", ["/usr/bin/python3", "-IB", "-c", KILL_MANAGER,
                observed, plan["package_version_release"]], timeout=30)
            manager.wait(timeout=30)
            manager_reader.join(timeout=10)
            if manager_reader.is_alive() or manager_errors:
                raise RuntimeError("manager log did not finish cleanly: " + str(manager_errors))
            if manager.returncode != 9:
                raise RuntimeError("interrupted manager did not report the observed SIGKILL")
            plan["interruption"] = json.loads(killed)
            stream.flush()
            plan["interruption"].update(manager_exit=manager.returncode, manager_log_sha256=digest(manager_log))
            save()
        step("fresh-launcher-not-committed", ["test", "!", "-e", "/usr/bin/nvbroadcast"])
        if step("bootstrap-identity", ["sha256sum", bootstrap]).split()[0] != plan["helper_sha256"]:
            raise ValueError("bootstrap differs from authenticated payload")
        helper = ["/usr/bin/python3", "-IB", bootstrap]
        step("bootstrap-uses-system-python", [*helper, "--help"])
        inspected = json.loads(step("inspect-authenticated-fragments", [*helper, "/inputs/target.rpm"]))
        if inspected["package_sha256"] != plan["package_sha256"] or observed not in [entry["original"] for entry in inspected["fragments"]]:
            raise ValueError("inspection does not identify the observed signed fragment")
        preserved = json.loads(step("preserve-authenticated-fragments", [*helper, "--preserve", "/inputs/target.rpm"]))
        (output / "preservation.json").write_text(json.dumps(preserved, indent=2) + "\n")
        plan["preservation_sha256"] = digest(output / "preservation.json")
        if preserved["package_sha256"] != plan["package_sha256"] or not preserved["preserved"]:
            raise ValueError("preservation does not retain the selected signed package")
        saved = preserved["package"]
        if not re.fullmatch(r"/usr/lib/nvbroadcast-recovery/[0-9a-f]{32}/package\.rpm", saved):
            raise ValueError("unexpected preserved package destination")
        step("audit-preserved-bytes-and-marker", ["/usr/bin/python3", "-IB", "-c", PRESERVED_BYTES,
             saved, plan["package_sha256"], plan["package_version_release"], PRIMARY, PUBLIC_SHA])
        step("import-preserved-public-key", ["rpm", "--import", str(Path(saved).with_name("public.asc"))])
        step("replay-exact-preserved-package", ["rpm", "--upgrade", "--replacepkgs", "--oldpackage", saved])
        step("configured-marker-and-bootstrap-cleared", ["sh", "-eu", "-c",
             "test ! -e /usr/lib/nvbroadcast/.transaction; test ! -e /var/lib/nvbroadcast/recovery-tools"])
        installed = json.loads(step("installed-payload-bindings", ["/usr/bin/python3", "-IB", "-c", INSTALLED_BINDINGS]))
        check_bindings(installed, plan)
        if installed != binding:
            raise ValueError("installed metadata differs from authenticated pre-install payload")
        step("native-package-state", ["/usr/lib/nvbroadcast/check-packages"])
        step("complete-installed-integrity", ["/usr/lib/nvbroadcast/runtime/bin/python", "-IB",
             "/usr/lib/nvbroadcast/validate-install.py", "--variant", "cpu"])
        step("launcher-after-repair", ["/usr/bin/nvbroadcast", "--help"])
        plan["status"] = "pass"
    except BaseException as error:
        plan.update(status="fail", error=str(error))
        raise
    finally:
        cleanup_errors = []
        try:
            if manager is not None and manager.poll() is None:
                manager.terminate()
                try:
                    manager.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    manager.kill()
                    manager.wait(timeout=10)
        except BaseException as error:
            cleanup_errors.append("manager shutdown: " + str(error))
        # Removal remains independent of local Docker-client shutdown failure.
        try:
            removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
            remaining = subprocess.check_output(["docker", "container", "ls", "--all", "--quiet", "--filter", f"name=^/{name}$"], text=True, timeout=30)
            plan["container_removed"] = removed.returncode == 0 and not remaining.strip()
            if not plan["container_removed"]:
                raise RuntimeError("owned fixture container remains")
        except BaseException as error:
            cleanup_errors.append("container removal: " + str(error))
        try:
            if manager_reader is not None:
                manager_reader.join(timeout=10)
                if manager_reader.is_alive():
                    raise RuntimeError("manager log reader did not stop")
            if manager is not None and manager.stdout is not None:
                manager.stdout.close()
            plan["original_package_unchanged"] = digest(package) == plan["package_sha256"]
            plan["snapshots_unchanged"] = digest(inputs / "target.rpm") == plan["package_sha256"] and digest(inputs / "public.asc") == PUBLIC_SHA
            if not plan["original_package_unchanged"] or not plan["snapshots_unchanged"]:
                raise RuntimeError("exact input identity verification failed")
        except BaseException as error:
            cleanup_errors.append("final evidence verification: " + str(error))
        if cleanup_errors:
            plan.update(status="fail", cleanup_errors=cleanup_errors)
        save()
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    return plan


def main(argv=None):
    args = parse_arguments(argv)
    plan = make_plan(args)
    if args.execute:
        plan = execute(plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

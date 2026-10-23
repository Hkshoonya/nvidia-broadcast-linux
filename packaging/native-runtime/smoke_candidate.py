#!/usr/bin/env python3
"""Bounded fresh-install acceptance for independently qualified hosted CPU bytes.

This driver does not establish CI provenance itself. The caller supplies the
exact source/artifact/manifest identities after the independent artifact review.
It downloads nothing, signs nothing, and mutates only a disposable offline fixture.
RPM lifecycle acceptance requires finalized signed bytes and the exact project
public key. Signature verification happens before any package installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tomllib
import uuid

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_candidate_recovery", Path(__file__).with_name("recover_candidate.py"))
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
PUBLIC_SHA = "bf9371c4fc83c41f8b1dbc3607c25ffa261b1e5de99fe799cf0a863db16cbd8a"
PRIMARY_FINGERPRINT = "A691DCA6C031FEFD2C34F5AAFE33D0C20C9451D1"
PROBES = {
    "desktop_probe.sh": "packaging/runtime-prototype/desktop_probe.sh",
    "probe.py": "packaging/runtime-prototype/probe.py",
    "recording_probe.py": "packaging/native-runtime/recording_probe.py",
}
# The DEB owns its interpreter and does not require system Python. Record a
# fixture's existing executable hash or its absence, then verify that state
# after removal. A dangling symlink is not an absent executable.
SYSTEM_PYTHON_CAPTURE = '''if [ -e "$1" ] || [ -L "$1" ]; then
 test -f "$1"
 sha256sum -- "$1" > "$2.sha256"
 printf present > "$2.state"
 cat -- "$2.sha256"
else
 printf absent > "$2.state"
 printf 'System Python absent\\n'
fi
'''
SYSTEM_PYTHON_VERIFY = '''case "$(cat -- "$2.state")" in
 present) test -f "$1"; sha256sum -c -- "$2.sha256" ;;
 absent) test ! -e "$1"; test ! -L "$1"; printf 'System Python remains absent\\n' ;;
 *) exit 1 ;;
esac
'''
INSTALLED_IDENTITY = '''from pathlib import Path
import hashlib,json,os,sys
expected=json.loads(Path('/probes/expected.json').read_text())
base=Path('/usr/lib/nvbroadcast')
actual=json.loads((base/'package.json').read_text())
assert actual['source']['revision']==expected['source']==actual['adapter']['revision']
assert actual['version']==expected['version'] and actual['variant']=='cpu'
assert actual['runtime_manifest_sha256']==expected['runtime_manifest_sha256']
assert hashlib.sha256((base/'runtime-manifest.json').read_bytes()).hexdigest()==expected['runtime_manifest_sha256']
for relative,digest in actual['adapter']['files'].items():
 assert expected['adapter_files'][relative]==digest,(relative,digest)
assert set(actual['adapter']['files'])==set(expected['adapter_files'])
assert not (base/'.transaction').exists()
assert not Path('/var/lib/nvbroadcast/recovery-tools').exists()
if expected['family']=='rpm':
 helper=Path('/usr/bin/nvbroadcast-recover-rpm')
 assert hashlib.sha256(helper.read_bytes()).hexdigest()==expected['recovery_helper_sha256']
 assert (helper.stat().st_uid,helper.stat().st_gid,helper.stat().st_nlink)==(0,0,1)
 assert (helper.stat().st_mode & 0o777)==0o755
lock=Path('/var/lib/nvbroadcast/runtime.lock').stat()
Path('/tmp/runtime-lock.identity').write_text(str(lock.st_dev)+':'+str(lock.st_ino))
print(json.dumps({'status':'pass','package':actual,'runtime_lock_identity':Path('/tmp/runtime-lock.identity').read_text()}))
'''


sha = recovery.digest


def git_bytes(source: str, path: str) -> bytes:
    result = recovery.run_capped(["git", "show", f"{source}:{path}"], cwd=ROOT, timeout=30)
    result.check_returncode()
    return result.stdout


snapshot = recovery.snapshot

def check_public_identity(key: bytes, fingerprint: str) -> None:
    if hashlib.sha256(key).hexdigest() != PUBLIC_SHA or fingerprint.strip() != PRIMARY_FINGERPRINT:
        raise ValueError("project public key or fingerprint differs from reviewed publisher")


def authenticate_rpm(execute) -> dict:
    # A new private verification database contains only the exact pinned project
    # key. Fixture/system trust cannot make an unrelated signature acceptable.
    execute("fresh-project-verification-keyring", ["mkdir", "-m", "700", "/tmp/nvb-project-rpmdb"])
    execute("import-project-verification-key", ["rpmkeys", "--dbpath", "/tmp/nvb-project-rpmdb",
        "--import", "/probes/public.asc"])
    verification = execute("verify-finalized-project-signature", ["rpmkeys", "--dbpath", "/tmp/nvb-project-rpmdb",
        "--checksig", "--verbose", "/inputs/package.rpm"])
    # rpmkeys may succeed for unsigned packages whose digests are intact.
    if not re.search(r"\bSignature\b[^\n]*:\s*OK\s*$", verification, re.MULTILINE | re.IGNORECASE):
        raise ValueError("no verified OpenPGP signature in project-only RPM keyring")
    if re.search(r"\b(?:NOKEY|BAD|NOTTRUSTED|UNTRUSTED)\b|\bNOT\s+OK\b", verification, re.IGNORECASE):
        raise ValueError("RPM verification contains a failed trust or integrity result")
    # The installation database is separate; import the same public key only
    # after independent authentication, inside this disposable fixture alone.
    execute("import-project-installation-key", ["rpm", "--import", "/probes/public.asc"])
    return {"status": "verified", "public_key_sha256": PUBLIC_SHA,
            "publisher_fingerprint": PRIMARY_FINGERPRINT,
            "verification_keyring": "fresh-project-only", "verification": verification}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--family", choices=("deb", "rpm"), required=True)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--package-revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--recovery-helper-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    recovery.identity(args.source, 40)
    recovery.identity(args.artifact_sha256, 64)
    recovery.identity(args.runtime_manifest_sha256, 64)
    if not re.fullmatch(r"[1-9][0-9]*", args.package_revision):
        parser.error("revision must be a positive integer")
    artifact = args.artifact.absolute()
    output = args.output.resolve()
    if any(c in str(path) for path in (artifact, output) for c in ",\n\r"):
        parser.error("unsupported Docker mount path")
    image = args.image
    if not image.startswith("sha256:"):
        raise ValueError("fixture must be an explicit image ID")
    recovery.identity(image.removeprefix("sha256:"), 64)
    recovery.identity(args.recovery_helper_sha256, 64)
    recovery.inspect_image(image)
    project = tomllib.loads(git_bytes(args.source, "pyproject.toml").decode())
    if project["project"]["version"] != args.package_version:
        raise ValueError("package version differs from approved source")
    version = args.package_version + "-" + args.package_revision + ".cpu"
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+-[1-9][0-9]*\.cpu", version):
        raise ValueError("unexpected native version")
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=False)
    package = output / ("package." + args.family)
    snapshot(artifact, package, args.artifact_sha256)
    probes = output / "probes"
    probes.mkdir(mode=0o755)
    # mkdir's mode is filtered by the private umask; the mounted probe root
    # must also be traversable when the container UID differs from the host.
    probes.chmod(0o755)
    # Execute qualification probes from the trusted finalizer checkout, while
    # comparing the installed application/adapter against the producer commit.
    tool_source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=30).strip()
    recovery.identity(tool_source, 40)
    source_probes = {}
    for name, path in PROBES.items():
        content = git_bytes(tool_source, path)
        (probes / name).write_bytes(content)
        (probes / name).chmod(0o644)
        source_probes[path] = hashlib.sha256(content).hexdigest()
    key = git_bytes(args.source, "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc")
    if hashlib.sha256(key).hexdigest() != PUBLIC_SHA:
        raise ValueError("source project key differs from reviewed publisher")
    if args.family == "rpm":
        fingerprint = git_bytes(args.source, "packaging/keys/RPM-GPG-KEY-nvbroadcast.fingerprint").decode("ascii")
        check_public_identity(key, fingerprint)
        (probes / "public.asc").write_bytes(key)
        (probes / "public.asc").chmod(0o444)
    recovery_source = git_bytes(args.source, "scripts/recover_native_rpm.py").decode()
    if recovery_source.count('PUBLIC_ARMOR = b""') != 1 or PUBLIC_SHA not in recovery_source:
        raise ValueError("unexpected rendered recovery source")
    rendered_helper = recovery_source.replace('PUBLIC_ARMOR = b""', 'PUBLIC_ARMOR = ' + repr(key)).encode()
    # This inventory matches build.py's explicit recipe_identity list. Refuse
    # newly added adapter inputs until the driver is reviewed for that change.
    adapter_paths = ["packaging/native-runtime/" + name for name in (
        "build.py", "builders.json", "package-hooks.sh", "validate_install.py", "recovery_bootstrap.py",
        "pylock.linux-x86_64-cp313-cpu.toml", "pylock.linux-x86_64-cp313-cuda.toml")]
    adapter_paths += ["packaging/runtime-prototype/" + name for name in (
        "prepare.py", "assemble.py", "inputs.json", "pylock.build.toml", "build-bindings.sh")]
    adapter_paths += ["packaging/native-prototype/build.py", "scripts/recover_native_rpm.py", "packaging/keys/RPM-GPG-KEY-nvbroadcast.asc"]
    if hashlib.sha256(rendered_helper).hexdigest() != args.recovery_helper_sha256:
        raise ValueError("rendered recovery helper differs from approved input")
    expected = {"source": args.source, "version": version, "family": args.family,
                "runtime_manifest_sha256": args.runtime_manifest_sha256,
                "recovery_helper_sha256": hashlib.sha256(rendered_helper).hexdigest(),
                "adapter_files": {path: hashlib.sha256(git_bytes(args.source, path)).hexdigest() for path in adapter_paths}}
    (probes / "expected.json").write_text(json.dumps(expected, sort_keys=True) + "\n")
    (probes / "expected.json").chmod(0o644)
    probe_snapshots = {path.name: sha(path) for path in probes.iterdir()}
    app_files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", args.source, "--", "src/nvbroadcast"], cwd=ROOT, text=True, timeout=30).splitlines()
    expected_app = {path.removeprefix("src/nvbroadcast/"): hashlib.sha256(git_bytes(args.source, path)).hexdigest()
                    for path in app_files if path.endswith(".py")}
    inputs = json.loads(git_bytes(args.source, "packaging/runtime-prototype/inputs.json"))
    python_version = inputs["python"]["version"]
    name = "nvb-hosted-smoke-" + args.family + "-" + uuid.uuid4().hex[:12]
    result = {"status": "running", "source": args.source, "family": args.family, "version": version,
              "qualification_tool_source": tool_source,
              "artifact_sha256": args.artifact_sha256, "runtime_manifest_sha256": args.runtime_manifest_sha256,
              "image": image, "driver_sha256": sha(Path(__file__)), "probe_sources": source_probes,
              "recovery_helper_sha256": args.recovery_helper_sha256,
              "shared_driver_sha256": sha(Path(__file__).with_name("recover_candidate.py")),
              "based_on_driver_sha256": "206c5c17fc327cef5fadf092a48435f64c73dc8251c999ad90699c6b89f651b7",
              "probe_snapshot_sha256": probe_snapshots,
              "steps": [], "network": "none", "host_devices_or_application_source_mounted": False}
    created = False

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    def execute(label, command, *, user=False, failure=False):
        prefix = ["docker", "exec"]
        if user:
            prefix += ["--user", "1000:1000", "-e", "HOME=/tmp/nvb-home", "-e", "XDG_CACHE_HOME=/tmp/nvb-home/.cache"]
        log = output / f"{len(result['steps']):02d}-{label}.log"
        run = recovery.run_capped([*prefix, name, *command], timeout=360, log=log)
        result["steps"].append({"name": label, "exit_code": run.returncode, "expected_failure": failure,
                                "log": log.name, "sha256": sha(log)})
        save()
        if run.returncode != (1 if failure else 0):
            raise RuntimeError(f"unexpected result: {label}; see {log}")
        return (run.stdout + run.stderr).decode()

    try:
        save()
        subprocess.run(["docker", "run", "-d", "--init", "--pull=never", "--network=none", "--name", name,
            "--memory", "4g", "--memory-swap", "4g", "--cpus", "2", "--pids-limit", "256",
            "--mount", f"type=bind,source={package},target=/inputs/package.{args.family},readonly",
            "--mount", f"type=bind,source={probes},target=/probes,readonly",
            image, "sleep", "infinity"], check=True, stdout=subprocess.DEVNULL, timeout=60)
        created = True
        execute("confirm-no-existing-package", ["dpkg-query", "-W", "nvbroadcast"] if args.family == "deb" else ["rpm", "-q", "nvbroadcast"], failure=True)
        if args.family == "rpm":
            result["rpm_authentication"] = authenticate_rpm(execute)
            save()
        execute("sentinels", ["sh", "-eu", "-c", "mkdir -p /tmp/nvb-home/.config/nvbroadcast /tmp/nvb-home/Recordings; "
            "printf preserved > /tmp/nvb-home/.config/nvbroadcast/sentinel; printf preserved > /tmp/nvb-home/Recordings/sentinel; "
            "chown -R 1000:1000 /tmp/nvb-home"])
        execute("record-system-python-state", ["sh", "-eu", "-c", SYSTEM_PYTHON_CAPTURE,
            "nvb-system-python", "/usr/bin/python3", "/tmp/system-python"])
        execute("install-exact-hosted-package", ["dpkg", "-i", "/inputs/package.deb"] if args.family == "deb" else ["rpm", "--upgrade", "/inputs/package.rpm"])
        identity_python = "/usr/lib/nvbroadcast/runtime/bin/python" if args.family == "deb" else "/usr/bin/python3"
        execute("installed-source-adapter-manifest-identity", [identity_python, "-IB", "-c", INSTALLED_IDENTITY])
        execute("complete-installed-integrity", ["/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B", "/usr/lib/nvbroadcast/validate-install.py", "--variant", "cpu"])
        execute("native-package-state", ["/usr/lib/nvbroadcast/check-packages"])
        execute("ordinary-user-launcher", ["/usr/bin/nvbroadcast", "--help"], user=True)
        if args.family == "rpm":
            execute("system-python-recovery-interface", ["/usr/bin/nvbroadcast-recover-rpm", "--help"])
        text = execute("ordinary-user-runtime-media-gtk", ["bash", "/probes/desktop_probe.sh", "/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B", "/probes/probe.py", "--runtime", "/usr/lib/nvbroadcast/runtime", "--python-version", python_version, "--window", "--variant", "cpu", "--lock", "/usr/lib/nvbroadcast/runtime/share/nvbroadcast-runtime-provenance/pylock.toml"], user=True)
        reports = [json.loads(line.removeprefix("RESULT=")) for line in text.splitlines() if line.startswith("RESULT=")]
        if len(reports) != 1 or reports[0]["status"] != "pass" or reports[0]["app_python_hashes"] != expected_app:
            raise ValueError("live runtime probe does not match exact CI application source")
        result["runtime_probe"] = reports[0]
        text = execute("ordinary-user-generated-recording", ["bash", "/probes/desktop_probe.sh", "/usr/lib/nvbroadcast/runtime/bin/python", "-I", "-B", "/probes/recording_probe.py"], user=True)
        reports = [json.loads(line.removeprefix("RESULT=")) for line in text.splitlines() if line.startswith("RESULT=")]
        if len(reports) != 1 or reports[0]["status"] != "pass":
            raise ValueError("generated recording result missing")
        result["recording"] = reports[0]
        execute("remove-native-package", ["dpkg", "--purge", "nvbroadcast"] if args.family == "deb" else ["rpm", "--erase", "nvbroadcast"])
        execute("package-uninstalled", ["dpkg-query", "-W", "nvbroadcast"] if args.family == "deb" else ["rpm", "-q", "nvbroadcast"], failure=True)
        execute("preserve-user-system-and-stable-lock", ["sh", "-eu", "-c", "test ! -e /usr/lib/nvbroadcast/runtime; test ! -e /usr/bin/nvbroadcast; test ! -e /usr/bin/nvbroadcast-recover-rpm; "
            "test ! -e /usr/lib/nvbroadcast/.transaction; test ! -e /var/lib/nvbroadcast/recovery-tools; "
            'test "$(cat /tmp/nvb-home/.config/nvbroadcast/sentinel)" = preserved; test "$(cat /tmp/nvb-home/Recordings/sentinel)" = preserved; '
            'test "$(cat /tmp/runtime-lock.identity)" = "$(stat -c %d:%i /var/lib/nvbroadcast/runtime.lock)"'])
        execute("preserve-system-python-state", ["sh", "-eu", "-c", SYSTEM_PYTHON_VERIFY,
            "nvb-system-python", "/usr/bin/python3", "/tmp/system-python"])
        result["status"] = "pass"
    except BaseException as error:
        result["status"], result["error"] = "fail", str(error)
        raise
    finally:
        try:
            removed = subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=60)
            remaining = subprocess.check_output(["docker", "container", "ls", "--all", "--quiet", "--filter", f"name=^/{name}$"], text=True, timeout=30)
            result["container_removed"] = not remaining.strip() and (not created or removed.returncode == 0)
            result["input_unchanged"] = sha(artifact) == args.artifact_sha256
            result["snapshot_unchanged"] = sha(package) == args.artifact_sha256
            result["probes_unchanged"] = {path.name: sha(path) for path in probes.iterdir()} == probe_snapshots
            if not all(result[key] for key in ("container_removed", "input_unchanged", "snapshot_unchanged", "probes_unchanged")):
                raise RuntimeError("cleanup or package identity recheck failed")
        except BaseException as error:
            result["status"], result["cleanup_error"] = "fail", str(error)
            raise
        finally:
            save()
    print(json.dumps({"status": result["status"], "steps": len(result["steps"]), "result": str(output / "result.json")}))


if __name__ == "__main__":
    main()

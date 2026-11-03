"""Bound hosted input preparation and CPU qualification; retain small evidence only."""

import json
import os
from pathlib import Path
from decimal import Decimal, ROUND_CEILING
import re
import shutil
import signal
import selectors
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/nix-cpu"
MAX_LOG = 2 * 1024**2
MIN_FREE = 3 * 1024**3
OFFICIAL_KEY = "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
PROFILES = {
    "standard": {"build_derivations": 200, "download_bytes": 8 * 1024**3,
                 "unpacked_bytes": 12 * 1024**3, "build_scratch_bytes": 6 * 1024**3,
                 "minimum_free_bytes": MIN_FREE, "job_minutes": 60,
                 "step_minutes": 50, "build_seconds": 2400},
    "full-rebuild": {"build_derivations": 900, "download_bytes": 8 * 1024**3,
                     "unpacked_bytes": 24 * 1024**3, "build_scratch_bytes": 30 * 1024**3,
                     "minimum_free_bytes": 10 * 1024**3, "initial_free_bytes": 75 * 1024**3,
                     "maximum_growth_bytes": 60 * 1024**3, "minimum_cpus": 4,
                     "minimum_memory_bytes": 14 * 1024**3, "job_minutes": 300,
                     "step_minutes": 280, "build_seconds": 270 * 60},
}


def profile_limits(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError("Unknown Nix qualification profile")
    return dict(PROFILES[profile])


def select_profile(event_name: str, event: dict) -> str:
    """A maintainer-controlled PR label or manual choice opts into the finite rebuild."""
    if event_name == "pull_request":
        labels = event.get("pull_request", {}).get("labels", [])
        return "full-rebuild" if any(label.get("name") == "nix-full-rebuild"
                                     for label in labels) else "standard"
    if event_name == "workflow_dispatch":
        inputs = event.get("inputs", {})
        profile = inputs.get("profile", "standard")
        profile_limits(profile)  # Reject unknown values even for preparation.
        return "standard" if inputs.get("operation") == "prepare-lock" else profile
    raise ValueError("Unsupported Nix qualification event")


def selected_profile() -> str:
    path = Path(os.environ["GITHUB_EVENT_PATH"])
    if path.stat().st_size > 1024**2:
        raise ValueError("Unexpectedly large GitHub event payload")
    return select_profile(os.environ["GITHUB_EVENT_NAME"], json.loads(path.read_text()))


def check_resources(profile: str, *, free_bytes: int, cpus: int, memory_bytes: int) -> dict:
    limits = profile_limits(profile)
    resources = {"profile": profile, "initial_free_bytes": free_bytes,
                 "cpus": cpus, "memory_bytes": memory_bytes, "limits": limits}
    if profile == "full-rebuild" and (free_bytes < limits["initial_free_bytes"] or
            cpus < limits["minimum_cpus"] or memory_bytes < limits["minimum_memory_bytes"]):
        raise ValueError("Hosted runner is too small for the full-rebuild profile: " + json.dumps(resources))
    return resources


def check_live_storage(profile: str, initial_free_bytes: int, free_bytes: int) -> str | None:
    limits = profile_limits(profile)
    if profile == "full-rebuild" and initial_free_bytes <= 0:
        raise ValueError("Full rebuild requires a fixed initial storage baseline")
    if free_bytes < limits["minimum_free_bytes"]:
        return "hosted filesystem free-space guard"
    if ("maximum_growth_bytes" in limits and
            initial_free_bytes - free_bytes > limits["maximum_growth_bytes"]):
        return "hosted filesystem observed-growth guard"
    return None


def check_plan(log: str, free_bytes: int, *, verified_present=False, profile="standard") -> dict:
    """Gate the pinned Nix version's transitive stderr forecast, not output paths."""
    builds, downloads, unpacked = 0, 0, 0
    seen = set()
    announced, paths, section = {}, {"built": set(), "fetched": set()}, None
    units = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}

    def size(text):
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)? (?:KiB|MiB|GiB|TiB)", text):
            raise ValueError("Unknown Nix forecast size")
        value, unit = text.split()
        # Nix displays one decimal. Add one displayed increment so rounding
        # cannot make a near-budget actual value look smaller than its bound.
        return int(((Decimal(value) + Decimal("0.1")) * units[unit]).to_integral_value(rounding=ROUND_CEILING))

    for line in log.splitlines():
        line = line.strip()
        if "don't know how to build these paths" in line:
            raise ValueError("Nix forecast contains unknown paths")
        built = re.fullmatch(r"(this derivation|these ([0-9]+) derivations) will be built:", line)
        fetched = re.fullmatch(r"(this path|these ([0-9]+) paths) will be fetched \(([^,]+) download, ([^,]+) unpacked\):", line)
        if built:
            if "built" in seen:
                raise ValueError("Ambiguous repeated build forecast")
            seen.add("built")
            builds = int(built.group(2) or "1")
            if builds <= 0:
                raise ValueError("Nix forecast count must be positive")
            announced["built"], section = builds, "built"
        elif fetched:
            if "fetched" in seen:
                raise ValueError("Ambiguous repeated fetch forecast")
            seen.add("fetched")
            count = int(fetched.group(2) or "1")
            if count <= 0:
                raise ValueError("Nix forecast count must be positive")
            announced["fetched"], section = count, "fetched"
            downloads, unpacked = size(fetched.group(3)), size(fetched.group(4))
        elif line.startswith("/nix/store/"):
            if (section is None or not re.fullmatch(r"/nix/store/[0-9a-z]{32}-[^/\s]+", line)
                    or line in paths[section] or (section == "built" and not line.endswith(".drv"))):
                raise ValueError("Malformed/duplicate Nix forecast path")
            paths[section].add(line)
        elif re.match(r"(?:this|these) .* (?:will be|are not yet|is not yet)", line):
            raise ValueError("Unrecognized or unknown-path Nix cost forecast")
        elif line and not line.startswith(("warning:", "trace:")):
            raise ValueError("Unrecognized Nix dry-run output")
    if any(len(paths[key]) != count for key, count in announced.items()):
        raise ValueError("Nix forecast path counts do not match its summaries")
    if not seen and not verified_present:
        raise ValueError("Missing Nix cost forecast without verified existing outputs")
    limits = profile_limits(profile)
    plan = {"profile": profile, "build_derivations": builds, "download_bytes_upper_bound": downloads,
            "unpacked_bytes_upper_bound": unpacked, "free_bytes": free_bytes,
            "limits": limits}
    if (builds > limits["build_derivations"] or downloads > limits["download_bytes"] or
            unpacked > limits["unpacked_bytes"] or
            unpacked + limits["build_scratch_bytes"] + limits["minimum_free_bytes"] > free_bytes):
        raise ValueError("Nix dry-run exceeds the hosted derivation/download/storage budget: " + json.dumps(plan))
    return plan


def run(name: str, command: list[str], seconds: int, records: list, *, json_output=True,
        profile="standard", initial_free_bytes=0) -> Path:
    profile_limits(profile)
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"GITHUB_TOKEN", "GH_TOKEN", "NIX_CONFIG", "NIX_PATH"}}
    environment["NIX_USER_CONF_FILES"] = "/dev/null"
    environment.update(NO_COLOR="1", LC_ALL="C", TERM="dumb")
    output, error = OUT / (name + (".json" if json_output else ".txt")), OUT / (name + ".log")
    started = time.monotonic()
    reason = None
    minimum_observed_free = None
    counts = {output.name: 0, error.name: 0}
    with output.open("xb") as stdout, error.open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, (stdout, output.name))
        selector.register(process.stderr, selectors.EVENT_READ, (stderr, error.name))
        terminated = None

        def stop_group(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        try:
            while selector.get_map() or process.poll() is None:
                if reason is None and time.monotonic() - started > seconds:
                    reason = "command timeout"
                elif reason is None:
                    free = shutil.disk_usage("/nix").free
                    minimum_observed_free = free if minimum_observed_free is None else min(minimum_observed_free, free)
                    reason = check_live_storage(profile, initial_free_bytes, free)
                if reason and terminated is None:
                    terminated = time.monotonic()
                    stop_group(signal.SIGTERM)
                if terminated is not None and time.monotonic() - terminated > 10:
                    stop_group(signal.SIGKILL)
                    break
                for key, _ in selector.select(timeout=0.2):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    target, filename = key.data
                    remaining = max(0, MAX_LOG - counts[filename])
                    target.write(data[:remaining])
                    counts[filename] += len(data)
                    if len(data) > remaining:
                        reason = "bounded log size exceeded"
            process.wait(timeout=10)
        finally:
            # Descendants can retain pipes after the group leader exits.
            stop_group(signal.SIGKILL)
            for key in list(selector.get_map().values()):
                key.fileobj.close()
            selector.close()
            process.wait(timeout=10)
    free_after = shutil.disk_usage("/nix").free
    minimum_observed_free = free_after if minimum_observed_free is None else min(minimum_observed_free, free_after)
    reason = reason or check_live_storage(profile, initial_free_bytes, free_after)
    records.append({"name": name, "command": command, "exit_code": process.returncode,
                    "seconds": round(time.monotonic() - started, 2), "stop_reason": reason,
                    "observed_log_bytes": counts,
                    "free_bytes_after": free_after,
                    "minimum_observed_free_bytes": minimum_observed_free,
                    "maximum_observed_growth_bytes": max(0, initial_free_bytes - minimum_observed_free),
                    "observed_growth_bytes_after": max(0, initial_free_bytes - free_after)})
    if reason or process.returncode:
        raise RuntimeError(f"{name} failed: {reason or process.returncode}; see bounded evidence")
    return output


def checked_copy(source: Path, name: str) -> None:
    if not source.is_file() or source.stat().st_size > 512 * 1024:
        raise ValueError("Expected a bounded evidence file: " + name)
    shutil.copyfile(source, OUT / name)


def stop_job_daemon() -> dict:
    """Stop only the fresh installation made by this disposable job."""
    marker = json.loads((OUT / "bootstrap.json").read_text())
    if marker["version"] != "2.35.4":
        raise ValueError("Cannot identify this job's Nix installation")
    result = subprocess.run(["sudo", "systemctl", "stop", "nix-daemon.socket", "nix-daemon.service"],
                            capture_output=True, timeout=40, check=True)
    states = {}
    for unit in ("nix-daemon.socket", "nix-daemon.service"):
        state = subprocess.run(["systemctl", "show", "--property=ActiveState", "--value", unit],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        if state not in {"inactive", "failed"}:
            raise RuntimeError("This job's Nix unit did not stop: " + unit)
        states[unit] = state
    return {"units": ["nix-daemon.socket", "nix-daemon.service"],
            "exit_code": result.returncode, "states": states,
            "kill_mode": "control-group (bootstrap verified)"}


def main() -> None:
    if (os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted" or
            sys.argv[1:] not in (["preflight"], ["prepare-lock"], ["qualify"])):
        raise ValueError("Run only on hosted CI with a fixed preparation or qualification operation")
    operation = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    profile = "standard" if operation == "prepare-lock" else selected_profile()
    limits = profile_limits(profile)
    if operation == "preflight":
        memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemTotal:"))
        cpus = len(os.sched_getaffinity(0))
        resources = check_resources(profile, free_bytes=shutil.disk_usage("/").free,
                                    cpus=cpus, memory_bytes=memory)
        resources["filesystem_device"] = os.stat("/").st_dev
        (OUT / "resource-preflight.json").write_text(json.dumps(resources, indent=2) + "\n")
        return
    initial_free = shutil.disk_usage("/nix").free
    def execute(*args, **kwargs):
        return run(*args, **kwargs, profile=profile, initial_free_bytes=initial_free)

    def interrupted(_signal, _frame):
        raise InterruptedError("Hosted workflow interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    pins = json.loads((ROOT / "nix/security-inputs.json").read_text())
    records = []
    receipt = {"operation": operation, "scope": "Input preparation only" if operation == "prepare-lock"
               else "Hermetic build and generated CPU/native checks; no physical NixOS acceptance",
               "expected_nixpkgs": pins["nixpkgsRevision"], "commands": records,
               "profile": profile, "initial_free_bytes": initial_free,
               "limits": {**limits, "per_log_bytes": MAX_LOG,
                          "job_minutes": 15 if operation == "prepare-lock" else limits["job_minutes"]}}
    try:
        if profile == "full-rebuild":
            preflight = json.loads((OUT / "resource-preflight.json").read_text())
            if preflight["profile"] != profile or preflight["filesystem_device"] != os.stat("/nix").st_dev:
                raise ValueError("Full rebuild requires a preflight baseline on the Nix filesystem")
            check_resources(profile, free_bytes=preflight["initial_free_bytes"],
                            cpus=preflight["cpus"], memory_bytes=preflight["memory_bytes"])
            initial_free = preflight["initial_free_bytes"]
            receipt["initial_free_bytes"] = initial_free
            receipt["runner_resources"] = preflight
        version = execute("nix-version", ["nix", "--version"], 20, records, json_output=False).read_text().strip()
        if version != "nix (Nix) 2.35.4":
            raise ValueError("Unexpected Nix bootstrap version")
        receipt["nix_version"] = version
        # Inspect configuration in memory. Never persist the raw response: an
        # unexpected ambient credential must not enter always-uploaded evidence.
        policy_environment = {key: value for key, value in os.environ.items()
                              if key not in {"GITHUB_TOKEN", "GH_TOKEN", "NIX_CONFIG", "NIX_PATH"}}
        policy_environment["NIX_USER_CONF_FILES"] = "/dev/null"
        policy = subprocess.run(["nix", "config", "show", "--json"], cwd=ROOT,
                                env=policy_environment, capture_output=True, text=True,
                                timeout=30, check=True)
        config = json.loads(policy.stdout)
        value = lambda key: config[key]["value"]
        words = lambda item: item.split() if isinstance(item, str) else item
        if (value("access-tokens") not in ("", {}, []) or
                str(value("sandbox")).lower() != "true" or str(value("require-sigs")).lower() != "true" or
                {x.rstrip("/") for x in words(value("substituters"))} != {"https://cache.nixos.org"} or
                set(words(value("trusted-public-keys"))) != {OFFICIAL_KEY} or
                words(value("trusted-users")) != ["root"] or value("builders") not in ("", []) or
                str(value("accept-flake-config")).lower() != "false" or
                str(value("allow-import-from-derivation")).lower() != "false" or
                str(value("max-jobs")) != "2" or str(value("cores")) != "2"):
            raise ValueError("Nix credential/sandbox/cache configuration differs from reviewed policy")
        # The selected config contains no credential; nevertheless retain only
        # this explicit policy subset, not arbitrary global configuration.
        (OUT / "nix-config.json").write_text(json.dumps({key: value(key) for key in
            ("sandbox", "require-sigs", "substituters", "trusted-public-keys", "trusted-users",
             "builders", "accept-flake-config", "allow-import-from-derivation", "max-jobs", "cores")}, indent=2) + "\n")
        if operation == "prepare-lock":
            execute("resolve-lock", ["nix", "flake", "lock", "--override-input", "nixpkgs",
                "github:NixOS/nixpkgs/" + pins["nixpkgsRevision"]], 240, records, json_output=False)
            checked_copy(ROOT / "flake.lock", "prepared-flake.lock")
        lock = json.loads((ROOT / "flake.lock").read_text())
        locked = lock["nodes"]["nixpkgs"]["locked"]
        if locked["rev"] != pins["nixpkgsRevision"] or not locked.get("narHash", "").startswith("sha256-"):
            raise ValueError("Reviewed committed Nixpkgs lock is required; use the separate preparation operation")
        receipt["locked_input"] = locked
        execute("flake-metadata", ["nix", "flake", "metadata", "--json", "--no-update-lock-file"], 90, records)
        if operation == "qualify":
            execute("flake-evaluation", ["nix", "flake", "check", "--no-build", "--no-update-lock-file"], 120, records, json_output=False)
            targets = [".#checks.x86_64-linux." + name for name in ("nvbroadcast", "help", "runtime", "module")]
            dry_run = execute("build-plan", ["nix", "build", "--dry-run", "--no-link", "--json", "--no-update-lock-file", *targets], 180, records)
            declared = [Path(item["outputs"]["out"]) for item in json.loads(dry_run.read_text())]
            present = len(declared) == len(targets) and all(
                path.resolve().is_relative_to(Path("/nix/store")) and path.exists() for path in declared)
            if present:
                execute("existing-outputs", ["nix", "path-info", "--offline", "--json", *map(str, declared)], 30, records)
            plan = check_plan((OUT / "build-plan.log").read_text(), shutil.disk_usage("/nix").free,
                              verified_present=present, profile=profile)
            (OUT / "build-budget.json").write_text(json.dumps(plan, indent=2) + "\n")
            result = execute("build-result", ["nix", "build", "--no-link", "--json", "--no-update-lock-file",
                         "--max-silent-time", "600", "--timeout", str(limits["build_seconds"]), *targets],
                         limits["build_seconds"], records)
            outputs = [Path(item["outputs"]["out"]) for item in json.loads(result.read_text())]
            for output in outputs:
                if not output.resolve().is_relative_to(Path("/nix/store")):
                    raise ValueError("Build output is outside the Nix store")
                if output.name.endswith("-finished-cpu-runtime-check"):
                    for name in ("runtime.json", "recording.json", "wrapper-help.txt"):
                        checked_copy(output / name, name)
                elif output.name.endswith("-module-evaluation.json"):
                    checked_copy(output, "module-evaluation.json")
            if not all((OUT / name).is_file() for name in ("runtime.json", "recording.json", "module-evaluation.json")):
                raise ValueError("Finished runtime/module evidence was not produced")
        receipt["status"] = "preparation-only" if operation == "prepare-lock" else "pass"
    except BaseException as error:
        receipt["status"], receipt["error"] = "failed", str(error)
        try:
            receipt["abort_daemon_cleanup"] = stop_job_daemon()
        except Exception as cleanup_error:
            receipt["abort_daemon_cleanup"] = {"failed": type(cleanup_error).__name__}
        raise
    finally:
        (OUT / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()

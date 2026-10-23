#!/usr/bin/env python3
"""Qualify reviewed CPU candidate bytes in explicitly provisioned offline fixtures.

Run finalize_candidate.py verify first to authenticate the bundle manifest.
This command checks the supplied package hash again; it neither signs packages
nor establishes their CI provenance. Without --execute it only prints a plan.
RPM execution includes both the installed smoke test and real first-install
SIGKILL recovery using the package's shipped bootstrap, in separate containers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def load(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load("recover_candidate.py")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("deb", "rpm"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--source", required=True, help="approved full application and adapter commit")
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--package-revision", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--recovery-helper-sha256", required=True, help="rendered RPM helper, not its template")
    parser.add_argument("--image", required=True, help="already provisioned sha256 image ID, never a mutable tag")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, help="fresh public evidence directory; excludes package/probe snapshots")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def make_plan(args):
    for value, size in ((args.source, 40), (args.artifact_sha256, 64),
                        (args.runtime_manifest_sha256, 64), (args.recovery_helper_sha256, 64)):
        recovery.identity(value, size)
    if not args.image.startswith("sha256:"):
        raise ValueError("fixture must be an explicit image ID")
    recovery.identity(args.image.removeprefix("sha256:"), 64)
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", args.package_version):
        raise ValueError("invalid package version")
    if not re.fullmatch(r"[1-9][0-9]*", args.package_revision):
        raise ValueError("package revision must be a positive integer")
    artifact, output = args.artifact.absolute(), args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("use a fresh qualification output directory")
    if any(c in str(path) for path in (artifact, output) for c in ",\n\r"):
        raise ValueError("unsupported Docker mount path")
    if recovery.digest(artifact) != args.artifact_sha256 or not artifact.stat().st_size:
        raise ValueError("artifact differs from the approved input identity")
    if args.evidence_output is not None and (args.evidence_output.exists() or args.evidence_output.is_symlink()):
        raise ValueError("use a fresh evidence output directory")
    return {"status": "planned-not-executed", "family": args.family,
            "source": args.source, "version": args.package_version + "-" + args.package_revision + ".cpu",
            "artifact": str(artifact), "artifact_sha256": args.artifact_sha256,
            "runtime_manifest_sha256": args.runtime_manifest_sha256,
            "recovery_helper_sha256": args.recovery_helper_sha256,
            "image": args.image, "output": str(output), "network": "none",
            "scenarios": ["smoke"] + (["recovery"] if args.family == "rpm" else []),
            "host_devices_or_application_source_mounted": False,
            "drivers": {name: recovery.digest(HERE / name) for name in
                        ("qualify_candidate.py", "smoke_candidate.py", "recover_candidate.py")}}


def export_evidence(plan, target):
    """Copy only explicitly bound public receipts/logs, never input snapshots."""
    source = Path(plan["output"])
    selected = {}
    for scenario in plan["scenarios"]:
        relative = scenario + "/result.json"
        expected = plan["receipts"][relative]
        if recovery.digest(source / relative) != expected:
            raise ValueError("qualification receipt changed before export")
        result = json.loads((source / relative).read_text())
        if result["status"] != "pass":
            raise ValueError("only completed qualification evidence can be exported")
        selected[relative] = expected
        for step in result["steps"]:
            name = step["log"]
            if not re.fullmatch(r"[0-9]{2}-[a-z0-9-]+\.log", name):
                raise ValueError("invalid qualification log path")
            selected[scenario + "/" + name] = step["sha256" if scenario == "smoke" else "log_sha256"]
        if scenario == "recovery":
            selected[scenario + "/preservation.json"] = result["preservation_sha256"]
            selected[scenario + "/interrupted-manager.log"] = result["interruption"]["manager_log_sha256"]
    for path, expected in selected.items():
        recovery.identity(expected, 64)
        if recovery.digest(source / path) != expected:
            raise ValueError("qualification evidence changed before export: " + path)
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    for relative, expected in selected.items():
        destination = target / relative
        destination.parent.mkdir(exist_ok=True)
        recovery.snapshot(source / relative, destination, expected)


def execute(args, plan):
    output = Path(plan["output"])
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    smoke = load("smoke_candidate.py")
    shared = ["--package-version", args.package_version, "--package-revision", args.package_revision,
              "--image", args.image, "--runtime-manifest-sha256", args.runtime_manifest_sha256]
    plan["status"] = "running"
    try:
        smoke.main([*shared, "--artifact", str(args.artifact.absolute()),
                    "--artifact-sha256", args.artifact_sha256, "--source", args.source,
                    "--family", args.family, "--recovery-helper-sha256", args.recovery_helper_sha256,
                    "--output", str(output / "smoke")])
        if args.family == "rpm":
            recovery.main([*shared, "--rpm", str(args.artifact.absolute()),
                           "--sha256", args.artifact_sha256, "--source-revision", args.source,
                           "--helper-sha256", args.recovery_helper_sha256,
                           "--output", str(output / "recovery"), "--execute"])
        plan["receipts"] = {name + "/result.json": recovery.digest(output / name / "result.json")
                            for name in plan["scenarios"]}
        if args.evidence_output is not None:
            export_evidence(plan, args.evidence_output.absolute())
            plan["evidence_output"] = str(args.evidence_output.absolute())
        plan["status"] = "pass"
    except BaseException as error:
        plan.update(status="fail", error=str(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main(argv=None):
    args = parse_arguments(argv)
    plan = make_plan(args)
    if args.execute:
        plan = execute(args, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

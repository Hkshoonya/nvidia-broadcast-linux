#!/usr/bin/env python3
"""Provision one disposable qualification image; do not install a candidate.

Only the base image is immutable. Native dependencies are acquired from the
distribution's current repositories. The result records the actual image and
OS package inventory; it does not claim reproduction of historical fixtures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("candidate_recovery", HERE.parent / "recover_candidate.py")
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def recipe(family):
    manifest = json.loads((HERE / "images.json").read_text())
    if manifest["schema_version"] != 1 or manifest["platform"] != "linux/amd64":
        raise ValueError("unsupported fixture manifest")
    selected = manifest[family]
    if not re.fullmatch(r"(?:ubuntu|fedora)@sha256:[a-f0-9]{64}", selected["base"]):
        raise ValueError("fixture base must be a pinned official distribution image")
    dockerfile = ROOT / selected["dockerfile"]
    if dockerfile != HERE / ("Dockerfile." + family):
        raise ValueError("unexpected fixture recipe path")
    text = dockerfile.read_text()
    if re.findall(r"^FROM (.+)$", text, re.MULTILINE) != [selected["base"]]:
        raise ValueError("Dockerfile base does not match the fixture manifest")
    return {**selected, "family": family, "platform": manifest["platform"],
            "dockerfile_sha256": common.digest(dockerfile),
            "manifest_sha256": common.digest(HERE / "images.json"),
            "driver_sha256": common.digest(Path(__file__)),
            "scope": manifest["scope"]}


def build(plan, output):
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    name = "nvb-fixture-inventory-" + uuid.uuid4().hex[:12]
    plan["status"] = "running"
    created = False
    try:
        # Send no repository, home, package, key or credentials to the builder.
        with tempfile.TemporaryDirectory(prefix="nvb-qualification-fixture-") as context:
            dockerfile = Path(context) / "Dockerfile"
            common.snapshot(ROOT / plan["dockerfile"], dockerfile, plan["dockerfile_sha256"])
            iidfile = output / "image.id"
            completed = common.run_capped(["docker", "build", "--platform", plan["platform"],
                "--iidfile", str(iidfile), context], timeout=1800, log=output / "build.log")
            completed.check_returncode()
        image = iidfile.read_text().strip()
        if not image.startswith("sha256:"):
            raise ValueError("builder returned an invalid image ID")
        common.identity(image.removeprefix("sha256:"), 64)
        plan["image"] = image
        inspected = common.run_capped(["docker", "image", "inspect", image], timeout=30)
        inspected.check_returncode()
        details = json.loads(inspected.stdout)
        if len(details) != 1 or (details[0]["Id"], details[0]["Os"], details[0]["Architecture"]) != (image, "linux", "amd64"):
            raise ValueError("unexpected fixture image identity or platform")
        made = common.run_capped(["docker", "create", "--pull=never", "--network=none",
            "--name", name, image, "true"], timeout=60)
        made.check_returncode()
        created = True
        for source, target in (("/usr/share/nvbroadcast-qualification/packages.tsv", "fixture-packages.txt"),
                               ("/etc/os-release", "os-release")):
            copied = common.run_capped(["docker", "cp", "-L", name + ":" + source,
                                       str(output / target)], timeout=60)
            copied.check_returncode()
        plan["inventory_sha256"] = common.digest(output / "fixture-packages.txt")
        plan["os_release"] = (output / "os-release").read_text()
        plan["files"] = {name: common.digest(output / name)
                         for name in ("build.log", "image.id", "fixture-packages.txt", "os-release")}
        plan["status"] = "pass"
    except BaseException as error:
        plan.update(status="fail", error=str(error))
        raise
    finally:
        try:
            removed = common.run_capped(["docker", "rm", "--force", name], timeout=60)
            remaining = common.run_capped(["docker", "container", "ls", "--all", "--quiet", "--filter", f"name=^/{name}$"], timeout=30)
            remaining.check_returncode()
            plan["inventory_container_removed"] = not remaining.stdout.strip() and (not created or removed.returncode == 0)
            if not plan["inventory_container_removed"]:
                raise RuntimeError("fixture inventory container remains")
        except BaseException as error:
            plan.update(status="fail", cleanup_error=str(error))
            raise
        finally:
            (output / "result.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            (output / "fixture.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("deb", "rpm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("use a fresh fixture output directory")
    plan = recipe(args.family)
    plan["status"] = "planned-not-executed"
    if args.execute:
        plan = build(plan, output)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

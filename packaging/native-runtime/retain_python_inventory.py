"""Attach partial Python SPDX evidence after exact native qualification succeeds.

The expected signed candidate manifest comes from the signing job's immutable
output, never from a self-consistent local document. Package bytes, signed bundle
allowlists and family qualification evidence remain unchanged. Only source-owned
tooling and hash-pinned host evidence wheels execute; projected runtime paths never
enter the host interpreter's import path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finalize = load("inventory_finalization", Path(__file__).with_name("finalize_candidate.py"))
projection_tool = load("inventory_projection", Path(__file__).with_name("python_inventory.py"))
host_tools = load("inventory_host_tools", Path(__file__).with_name("inventory_host_tools.py"))
require = finalize.require
SCHEMA = ROOT / "tests/data/spdx-2.3/schema.json"
SCHEMA_SHA256 = "239208b7ac287b3cf5d9a9af23f9d69863971102a5e1587a27a398b43490b89b"


def directory(path):
    result = Path(path).absolute()
    require(result.resolve(strict=True) == result and stat.S_ISDIR(result.lstat().st_mode),
            "expected a nonredirected evidence directory")
    return result


def source_epoch(source):
    value = subprocess.check_output(
        ["git", "--no-replace-objects", "show", "-s", "--format=%ct", source],
        cwd=ROOT, text=True, timeout=30).strip()
    require(value.isdecimal(), "source epoch is unavailable")
    return int(value)


def checked_qualification(qualified, expected, variant):
    qualified = directory(qualified)
    manifest = finalize.verify_bundle(qualified / "packages", expected, variant=variant)
    report_path = finalize.relative_file(qualified, "qualification.json")
    report = finalize.read_json(report_path)
    require(report.get("schema_version") == 1 and report.get("status") == "qualified-development-candidate"
            and report.get("candidate_manifest_sha256") == expected and report.get("variant") == variant
            and report.get("producer") == manifest["producer"] and report.get("packages") == manifest["packages"],
            "completed qualification identity differs from the signing-job anchor")
    deb, rpm = qualified / "evidence/deb", qualified / "evidence/rpm"
    results = {
        "deb": finalize.check_execution(deb / "smoke/result.json", manifest, "deb", variant=variant),
        "rpm": finalize.check_execution(rpm / "smoke/result.json", manifest, "rpm", variant=variant),
        "recovery": finalize.check_execution(rpm / "recovery/result.json", manifest, "rpm", recovery=True, variant=variant),
    }
    fixtures = {"deb": finalize.evidence_files(deb, "deb"), "rpm": finalize.evidence_files(rpm, "rpm")}
    require(report.get("qualification") == results and report.get("fixtures") == fixtures,
            "completed qualification evidence changed")
    require(all(fixtures[name]["image"] == results[name]["image"] for name in ("deb", "rpm"))
            and fixtures["rpm"]["image"] == results["recovery"]["image"], "qualified fixture identity differs")
    return manifest, finalize.sha(report_path)


def checked_projection(root, expected, manifest, epoch):
    root = directory(root)
    receipt_path = finalize.relative_file(root, "projection.json")
    require(isinstance(expected, dict) and set(expected) == {
        "receipt_sha256", "receipt_bytes", "metadata_count", "metadata_bytes", "driver_sha256"},
        "validation has no exact Python projection binding")
    require(finalize.sha(receipt_path) == finalize.digest(expected["receipt_sha256"])
            and receipt_path.stat().st_size == expected["receipt_bytes"]
            and expected["driver_sha256"] == finalize.sha(ROOT / "packaging/native-runtime/python_inventory.py"),
            "projection differs from authenticated validation")
    receipt = finalize.read_json(receipt_path)
    deb = manifest["packages"]["deb"]
    require(receipt.get("schema_version") == 1 and receipt.get("architecture") == "amd64"
            and receipt.get("source_commit") == manifest["producer"]["source"]
            and type(receipt.get("source_date_epoch")) is int and receipt["source_date_epoch"] == epoch
            and receipt.get("runtime_manifest_sha256") == manifest["runtime_manifest_sha256"]
            and receipt.get("archive") == {"name": PurePosixPath(deb["path"]).name,
                                            "bytes": deb["bytes"], "sha256": deb["sha256"]},
            "projection source/archive/runtime context differs")
    files = receipt.get("files")
    require(isinstance(files, dict) and 1 < len(files) <= projection_tool.MAX_DISTRIBUTIONS + 1
            and "context/assembled-pylock.toml" in files, "invalid projection file set")
    metadata = {}
    expected_directories = set()
    for name, record in files.items():
        require(isinstance(name, str), "invalid projection path")
        for parent in PurePosixPath(name).parents:
            if str(parent) != ".":
                expected_directories.add(str(parent))
        if name != "context/assembled-pylock.toml":
            require(name.startswith("runtime/"), "unexpected projection member")
            relative = name.removeprefix("runtime/")
            match = projection_tool.layout(projection_tool.PREFIX + relative)
            require(match is not None and match.group(2) is not None
                    and match.group(2).lower().endswith(".dist-info") and match.group(3) == "METADATA",
                    "unexpected projected metadata path")
            metadata[name] = record
    require(metadata and type(receipt.get("metadata_count")) is int
            and receipt["metadata_count"] == expected["metadata_count"] == len(metadata), "projection count differs")
    total = 0
    for record in metadata.values():
        require(isinstance(record, dict) and set(record) == {"bytes", "sha256"}
                and type(record["bytes"]) is int and 0 < record["bytes"] <= projection_tool.MAX_METADATA_BYTES,
                "invalid projected metadata size")
        total += record["bytes"]
    require(type(receipt.get("metadata_bytes")) is int
            and total == receipt["metadata_bytes"] == expected["metadata_bytes"]
            and total <= projection_tool.MAX_TOTAL_METADATA_BYTES, "projection aggregate differs")
    actual_files, actual_directories = set(), set()
    for count, path in enumerate(root.rglob("*"), 1):
        require(count <= 12 * projection_tool.MAX_DISTRIBUTIONS + 32, "excess projection entries")
        mode, relative = path.lstat().st_mode, path.relative_to(root).as_posix()
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "projection contains a link or special member")
        (actual_directories if stat.S_ISDIR(mode) else actual_files).add(relative)
    require(actual_files == set(files) | {"projection.json"} and actual_directories == expected_directories,
            "unexpected or missing projection entries")
    for name, record in files.items():
        finalize.verified_file(root, name, record, maximum=projection_tool.MAX_METADATA_BYTES)
    return receipt, root / "runtime", root / "context/assembled-pylock.toml"


def load_generator(host_wheel, schema_tools):
    """Import only authenticated source and the finite evidence-tool closure."""
    lock_path = ROOT / "packaging/runtime-prototype/pylock.build.toml"
    lock = tomllib.loads(lock_path.read_text())
    entries = [entry for entry in lock["packages"] if entry["name"] == "packaging"]
    require(len(entries) == 1 and entries[0]["version"] == "26.0" and len(entries[0]["wheels"]) == 1,
            "reviewed pure host packaging input changed")
    entry, host_wheel = entries[0]["wheels"][0], Path(host_wheel).absolute()
    require(host_wheel.resolve(strict=True) == host_wheel and stat.S_ISREG(host_wheel.lstat().st_mode)
            and host_wheel.stat().st_size == entry["size"] and finalize.sha(host_wheel) == entry["hashes"]["sha256"],
            "host tool differs from the source-pinned wheel")
    site, schema_receipt = host_tools.verified_site(schema_tools)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(host_wheel), str(site), str(ROOT / "src")]
    import packaging
    require(packaging.__version__ == "26.0" and packaging.__file__.startswith(str(host_wheel) + "/"),
            "ambient packaging import is forbidden")
    schema_receipt["installed"] = host_tools.validate_imports(site, host_tools.inputs())
    from jsonschema import Draft7Validator
    from referencing import Registry
    require(finalize.sha(SCHEMA) == SCHEMA_SHA256, "official SPDX schema identity differs")
    def no_remote_reference(uri):
        raise ValueError("remote SPDX schema retrieval is forbidden: " + uri)
    schema = json.loads(SCHEMA.read_text())
    Draft7Validator.check_schema(schema)
    validator = Draft7Validator(schema, registry=Registry(retrieve=no_remote_reference))
    generator = load("native_python_spdx_generator", ROOT / "scripts/generate_python_sbom.py")
    return generator, {"packaging_version": packaging.__version__, "packaging_wheel_sha256": finalize.sha(host_wheel),
                       "source_build_lock_sha256": finalize.sha(lock_path), "host_python": sys.version,
                       "schema_closure": schema_receipt, "spdx_schema_sha256": SCHEMA_SHA256}, validator


def validate_document(document, validator, installed, projected, archive, generator, lock):
    """Validate SPDX syntax plus the partial-scope metadata/archive semantics."""
    require(validator.is_valid(document), "SPDX official schema validation failed")
    packages, files = document["packages"], document["files"]
    require(len(packages) == len(installed) + 1 and len(files) == len(installed), "SPDX inventory count differs")
    artifact = packages[0]
    require(artifact["name"] == archive["name"] and artifact["checksums"] == [
        {"algorithm": "SHA256", "checksumValue": archive["sha256"]}], "SPDX archive identity differs")
    require(all(package["filesAnalyzed"] is False and package["downloadLocation"] == "NOASSERTION"
                and package["licenseConcluded"] == package["licenseDeclared"] == package["copyrightText"] == "NOASSERTION"
                for package in packages), "SPDX partial license/file scope differs")
    expected_context = {"source_commit": projected["source_commit"], "architecture": "x86_64",
                        "locks": [{"name": lock.name, "sha256": finalize.sha(lock)}],
                        "authentication": "caller-provided context; not verified"}
    require(artifact["sourceInfo"] == "Caller-provided source context: " + json.dumps(expected_context, sort_keys=True)
            and document["comment"] == artifact["comment"] == generator.SCOPE,
            "SPDX source or partial-scope context differs")
    created = datetime.fromtimestamp(projected["source_date_epoch"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    require(document["creationInfo"]["created"] == created, "SPDX source clock differs")
    by_name = {package["name"]: package for package in packages[1:]}
    by_path = {item["fileName"]: item for item in files}
    require(len(by_name) == len(installed) and len(by_path) == len(installed), "duplicate SPDX metadata identity")
    expected_edges = {("SPDXRef-DOCUMENT", "DESCRIBES", artifact["SPDXID"])}
    for record in installed:
        package, metadata_file = by_name.get(record["name"]), by_path.get("./" + record["path"])
        require(package is not None and metadata_file is not None and package["versionInfo"] == record["version"]
                and metadata_file["checksums"] == [{"algorithm": "SHA1", "checksumValue": record["sha1"]},
                                                    {"algorithm": "SHA256", "checksumValue": record["sha256"]}]
                and metadata_file["licenseConcluded"] == metadata_file["copyrightText"] == "NOASSERTION"
                and metadata_file["licenseInfoInFiles"] == ["NOASSERTION"], "SPDX installed metadata differs")
        expected_edges.add((artifact["SPDXID"], "CONTAINS", package["SPDXID"]))
        expected_edges.add((metadata_file["SPDXID"], "METAFILE_OF", package["SPDXID"]))
    ids = [document["SPDXID"], *(item["SPDXID"] for item in packages), *(item["SPDXID"] for item in files)]
    edges = [(item["spdxElementId"], item["relationshipType"], item["relatedSpdxElement"]) for item in document["relationships"]]
    require(len(ids) == len(set(ids)) and len(edges) == len(expected_edges) and set(edges) == expected_edges,
            "SPDX metadata relationships differ")
    body = {key: value for key, value in document.items() if key != "documentNamespace"}
    identity = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(document["documentNamespace"] == "https://github.com/Hkshoonya/nvidia-broadcast-linux/sbom/python-inventory/" + identity,
            "SPDX content identity differs")


def retain_inventory(qualified, expected, projection, host_wheel, schema_tools, variant="cpu"):
    qualified, projection = directory(qualified), directory(projection)
    require(not projection.is_relative_to(qualified) and not qualified.is_relative_to(projection),
            "projection must be separate from package and qualification evidence")
    manifest, qualification_sha = checked_qualification(qualified, expected, variant)
    validation = finalize.read_json(qualified / "packages/validation.json")
    projected, runtime_root, lock = checked_projection(
        projection, validation.get("python_inventory_projection"), manifest, source_epoch(manifest["producer"]["source"]))
    output = qualified / "python-inventory"
    require(not output.exists() and not output.is_symlink(), "inventory output already exists")
    generator, tools, validator = load_generator(host_wheel, schema_tools)
    installed = generator.installed_metadata(runtime_root, "amd64")
    output.mkdir(mode=0o700)
    inventories = {}
    for family in ("deb", "rpm"):
        package = finalize.relative_file(qualified / "packages", manifest["packages"][family]["path"])
        document = generator.build_document(runtime_root, "amd64", package, projected["source_commit"],
                                            projected["source_date_epoch"], (lock,))
        require(document["packages"][0]["checksums"] == [{"algorithm": "SHA256", "checksumValue": manifest["packages"][family]["sha256"]}],
                "inventory archive identity differs from finalized package")
        data = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
        require(len(data) <= 16 * 1024**2, "inventory document exceeds evidence bound")
        repeated = generator.build_document(runtime_root, "amd64", package, projected["source_commit"],
                                             projected["source_date_epoch"], (lock,))
        require(data == (json.dumps(repeated, sort_keys=True, indent=2) + "\n").encode(),
                "repeated inventory generation differs")
        validate_document(document, validator, installed, projected,
                          {"name": package.name, "sha256": manifest["packages"][family]["sha256"]}, generator, lock)
        target = output / (family + ".python-inventory.spdx.json")
        with target.open("xb") as stream:
            stream.write(data)
        inventories[target.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                    "archive_sha256": manifest["packages"][family]["sha256"],
                                    "repeat_identical": True, "official_schema_valid": True, "metadata_semantics_valid": True}
    # Repeat the existing exact package/evidence gates after read-only generation.
    after, after_qualification_sha = checked_qualification(qualified, expected, variant)
    require(after == manifest and after_qualification_sha == qualification_sha, "qualification/package bytes changed")
    checked_projection(projection, validation["python_inventory_projection"], manifest, projected["source_date_epoch"])
    source_paths = ("packaging/native-runtime/retain_python_inventory.py", "packaging/native-runtime/python_inventory.py",
                    "packaging/native-runtime/inventory_host_tools.py", "packaging/native-runtime/python-inventory-tools.json",
                    "packaging/native-runtime/finalize_candidate.py", "packaging/native-runtime/candidate_policy.py",
                    "packaging/native-runtime/validate_candidate.py", ".github/workflows/native-cpu-finalize.yml",
                    "packaging/runtime-prototype/prepare.py",
                    "tests/data/spdx-2.3/schema.json", "tests/data/spdx-2.3/UPSTREAM.json",
                    "scripts/generate_python_sbom.py", "src/nvbroadcast/__init__.py",
                    "src/nvbroadcast/runtime/__init__.py", "src/nvbroadcast/runtime/artifact.py", "src/nvbroadcast/runtime/variants.py")
    receipt = {"schema_version": 1, "status": "partial-python-inventory-evidence", "variant": variant,
               "candidate_manifest_sha256": expected, "qualification_sha256": qualification_sha,
               "validation_receipt_sha256": finalize.sha(qualified / "packages/validation.json"),
               "projection_receipt_sha256": finalize.sha(projection / "projection.json"),
               "source_commit": projected["source_commit"], "source_date_epoch": projected["source_date_epoch"],
               "runtime_manifest_sha256": manifest["runtime_manifest_sha256"], "assembled_lock_sha256": finalize.sha(lock),
               "tools": tools, "tool_source_sha256": {path: finalize.sha(ROOT / path) for path in source_paths},
               "inventories": inventories,
               "scope": "Partial app-owned Python METADATA inventory only. Existing native qualification remains separate; complete native/interpreter/codec/model/license inventory and release/device qualification are excluded."}
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--host-packaging-wheel", type=Path, required=True)
    parser.add_argument("--schema-tools", type=Path, required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:
        parser.error("run this evidence tool with the trusted host interpreter -I -S")
    try:
        result = retain_inventory(args.qualified, args.expected_manifest, args.projection,
                                  args.host_packaging_wheel, args.schema_tools, args.variant)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Python inventory evidence failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": result["status"], "inventories": result["inventories"]}, sort_keys=True))


if __name__ == "__main__":
    main()

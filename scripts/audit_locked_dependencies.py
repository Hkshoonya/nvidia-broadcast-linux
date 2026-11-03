#!/usr/bin/env python3
"""Audit exact active package pins through PyPI's release-specific JSON API.

No dependency resolution, package installation or package code execution occurs.
This complements the resolved-project pip-audit: it covers frozen packages for
other platforms too. API documentation: https://docs.pypi.org/api/json/
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version
import yaml


REQUIRED_LOCKS = (
    "packaging/macos-runtime-prototype/pylock.macos-arm64-cp313-cpu.toml",
    "packaging/native-runtime/pylock.linux-x86_64-cp313-cpu.toml",
    "packaging/native-runtime/pylock.linux-x86_64-cp313-cuda.toml",
    "packaging/runtime-prototype/pylock.build.toml",
    "packaging/runtime-prototype/pylock.linux-x86_64-cp313-cpu.toml",
    "packaging/runtime-prototype/pylock.linux-x86_64-cp313-cuda.toml",
)
REQUIRED_INPUTS = (
    "packaging/macos-runtime-prototype/inputs.json",
    "packaging/runtime-prototype/inputs.json",
    "packaging/flatpak/inputs.json",
)
REQUIRED_FLATPAKS = (
    "packaging/flatpak/com.nvbroadcast.NVBroadcast.yml",
    "packaging/flatpak/python3-flatpak-requirements.yaml",
    "packaging/flatpak/python3-private-runtime.yml",
)
MAX_BYTES = 4 * 1024 * 1024
MAX_PINS = 1000
MAX_ADVISORIES = 200
MAX_AUDIT_SECONDS = 300


def identity(name: str, version: str) -> tuple[str, str]:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Package pin has an invalid project name")
    if not isinstance(version, str) or not version or len(version) > 100:
        raise ValueError("Package pin requires an exact version")
    return canonicalize_name(name), str(Version(version))


def collect_pins(root: Path) -> dict:
    root = root.resolve()
    files, pins, skipped, external = {}, {}, [], []

    def read(path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError(f"Missing, redirected or oversized pin input: {path.relative_to(root)}")
        data = path.read_bytes()
        files[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")

    def add(name, version, path, kind):
        key = identity(name, version)
        pins.setdefault(key, set()).add((str(path.relative_to(root)), kind))
        if len(pins) > MAX_PINS:
            raise ValueError("Exact pin inventory exceeds the audit bound")

    lock_paths = set(root / path for path in REQUIRED_LOCKS)
    lock_paths.update(path for path in (root / "packaging").rglob("pylock*.toml")
                      if "qualification" not in path.relative_to(root).parts)
    for path in sorted(lock_paths):
        lock = tomllib.loads(read(path))
        if lock.get("lock-version") != "1.0" or not isinstance(lock.get("packages"), list) or not lock["packages"]:
            raise ValueError(f"Unsupported or empty lock: {path.relative_to(root)}")
        for package in lock["packages"]:
            name, version = identity(package["name"], package["version"])
            archive = package.get("archive", {})
            if name == "nvbroadcast" and archive.get("path", "").endswith(".whl") and "url" not in archive:
                skipped.append({"name": name, "version": version, "path": str(path.relative_to(root)),
                                "reason": "locally rebuilt project application"})
                continue
            # Locally built GI bindings still have public upstream versions.
            add(name, version, path, "pylock")

    input_paths = set(root / path for path in REQUIRED_INPUTS)
    input_paths.update(path for path in (root / "packaging").rglob("inputs.json")
                       if "qualification" not in path.relative_to(root).parts)
    for path in sorted(input_paths):
        data = json.loads(read(path))
        for field in ("bootstrap", "bindings"):
            if not isinstance(data.get(field, []), list):
                raise ValueError(f"Invalid {field} input inventory")
            for item in data.get(field, []):
                add(item["name"], item["version"], path, field)
        if "uv" in data:
            add("uv", data["uv"]["version"], path, "refresh-tool")
        if "python" in data:
            external.append({"path": str(path.relative_to(root)), "component": "CPython",
                             "version": data["python"]["version"], "reason": "interpreter is outside the PyPI advisory API"})

    flatpak_paths = set(root / path for path in REQUIRED_FLATPAKS)
    flatpak_paths.update((root / "packaging/flatpak").glob("*.yml"))
    flatpak_paths.update((root / "packaging/flatpak").glob("*.yaml"))
    captured_sources, local_modules = {}, {}
    for path in sorted(flatpak_paths):
        document = yaml.safe_load(read(path))
        local_modules[path] = []
        pending, seen, captured = [document], set(), 0
        while pending:
            node = pending.pop()
            if not isinstance(node, (dict, list)) or id(node) in seen:
                continue
            seen.add(id(node))
            if len(seen) > 10000:
                raise ValueError("Flatpak manifest exceeds the audit bound")
            if isinstance(node, list):
                pending.extend(node)
                continue
            for module in node.get("modules", []):
                if isinstance(module, str):
                    if Path(module).name != module or not module.endswith((".yml", ".yaml")):
                        raise ValueError("Flatpak module references must be local manifest filenames")
                    local_modules[path].append(path.parent / module)
            pending.extend(node.values())
            if "url" not in node:
                continue
            if not isinstance(node["url"], str):
                raise ValueError("Flatpak source URL must be a string")
            url = urllib.parse.urlsplit(node["url"])
            filename = urllib.parse.unquote(url.path.rsplit("/", 1)[-1])
            if filename.endswith(".whl"):
                name, version, _, _ = parse_wheel_filename(filename)
            elif url.hostname == "files.pythonhosted.org":
                name, version = parse_sdist_filename(filename)
            else:
                continue
            add(name, str(version), path, "flatpak-source")
            captured += 1
        captured_sources[path] = captured
    for path, includes in local_modules.items():
        if any(include not in captured_sources for include in includes):
            raise ValueError(f"Missing included Flatpak source manifest: {path.relative_to(root)}")
    for path in (root / name for name in REQUIRED_FLATPAKS):
        if not captured_sources[path] and not any(captured_sources[include] for include in local_modules[path]):
            raise ValueError(f"Required Flatpak manifest has no exact Python sources: {path.relative_to(root)}")
    return {"input_sha256": dict(sorted(files.items())),
            "pins": [{"name": name, "version": version,
                      "sources": [{"path": path, "kind": kind} for path, kind in sorted(sources)]}
                     for (name, version), sources in sorted(pins.items())],
            "skipped_local_application": skipped, "outside_pypi_scope": external}


def release_url(name: str, version: str) -> str:
    name, version = identity(name, version)
    return f"https://pypi.org/pypi/{urllib.parse.quote(name, safe='')}/{urllib.parse.quote(version, safe='')}/json"


def fetch_release(name: str, version: str) -> dict:
    url = release_url(name, version)
    deadline = time.monotonic() + 30
    for attempt in range(2):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PyPI audit request deadline exceeded")
            request = urllib.request.Request(url, headers={"Accept": "application/json",
                "User-Agent": "NVBroadcast-exact-pin-audit/1"})
            with urllib.request.urlopen(request, timeout=min(15, remaining)) as response:
                final = urllib.parse.urlsplit(response.geturl())
                if response.status != 200 or final.scheme != "https" or final.hostname != "pypi.org":
                    raise ValueError("Unexpected PyPI response status or origin")
                chunks, size = [], 0
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("PyPI audit response deadline exceeded")
                    chunk = response.read1(min(65536, MAX_BYTES + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("PyPI release response exceeds the size bound")
                return json.loads(b"".join(chunks))
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt:
                raise
        if time.monotonic() >= deadline:
            raise TimeoutError("PyPI audit retry deadline exceeded")
        time.sleep(0.5)
    raise RuntimeError("PyPI fetch did not produce a response")


def short_text(value, limit=300):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("Malformed or oversized PyPI advisory field")
    return value


def advisory_summary(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Malformed PyPI advisory")
    identifier = short_text(value["id"], 150)
    if not identifier:
        raise ValueError("PyPI advisory lacks an identifier")
    withdrawn = value.get("withdrawn")
    if withdrawn is not None:
        stamp = datetime.fromisoformat(short_text(withdrawn, 50).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Invalid withdrawn advisory timestamp")
    aliases = value.get("aliases", [])
    fixed = value.get("fixed_in", [])
    if not isinstance(aliases, list) or not isinstance(fixed, list) or len(aliases) > 30 or len(fixed) > 50:
        raise ValueError("Malformed advisory aliases or fixed versions")
    link = short_text(value.get("link") or "https://osv.dev/vulnerability/" + urllib.parse.quote(identifier, safe=""), 2048)
    if urllib.parse.urlsplit(link).scheme != "https":
        raise ValueError("Unexpected advisory link scheme")
    severity = value.get("severity")
    if severity is not None and not isinstance(severity, (str, list)):
        raise ValueError("Malformed advisory severity")
    if len(json.dumps(severity)) > 2000:
        raise ValueError("Oversized advisory severity")
    return {"id": identifier, "aliases": [short_text(item, 150) for item in aliases],
            "fixed_in": [short_text(item, 100) for item in fixed], "link": link,
            "withdrawn": withdrawn, "severity": severity}


def check_release(pin: dict, payload: dict) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        raise ValueError("PyPI release metadata is missing")
    if identity(payload["info"].get("name"), payload["info"].get("version")) != identity(pin["name"], pin["version"]):
        raise ValueError("PyPI name/version differs from the exact requested pin")
    entries = payload.get("vulnerabilities")
    if not isinstance(entries, list) or len(entries) > MAX_ADVISORIES:
        raise ValueError("PyPI vulnerability inventory is missing or exceeds its bound")
    advisories = [advisory_summary(value) for value in entries]
    active = [entry for entry in advisories if entry["withdrawn"] is None]
    return {**pin, "status": "vulnerable" if active else "pass", "query_url": release_url(pin["name"], pin["version"]),
            "active_advisories": active, "withdrawn_advisories": [entry for entry in advisories if entry["withdrawn"] is not None]}


def audit_inventory(inventory: dict, *, fetch=fetch_release) -> dict:
    if not inventory["pins"]:
        raise ValueError("Exact pin inventory is empty")
    deadline = time.monotonic() + MAX_AUDIT_SECONDS

    def inspect(pin):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("Overall exact-pin audit deadline exceeded")
            payload = fetch(pin["name"], pin["version"])
            if time.monotonic() >= deadline:
                raise TimeoutError("Overall exact-pin audit deadline exceeded")
            return check_release(pin, payload)
        except Exception as error:
            return {**pin, "status": "error", "query_url": release_url(pin["name"], pin["version"]),
                    "error": f"{type(error).__name__}: {str(error)[:400]}"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, inventory["pins"]))
    counts = {status: sum(result["status"] == status for result in results) for status in ("pass", "vulnerable", "error")}
    return {"schema_version": 1, "status": "pass" if counts["vulnerable"] == counts["error"] == 0 else "fail",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "api": "https://docs.pypi.org/api/json/",
            "counts": counts, "input_sha256": inventory["input_sha256"], "results": results,
            "skipped_local_application": inventory["skipped_local_application"],
            "outside_pypi_scope": inventory["outside_pypi_scope"],
            "limits": ["Known PyPI release advisories only; a pass is not a complete security assessment",
                       "Native/system libraries, interpreters, model weights and non-PyPI components require separate review"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_inventory(collect_pins(args.root))
    except Exception as error:
        report = {"schema_version": 1, "status": "fail", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "inventory_error": f"{type(error).__name__}: {str(error)[:400]}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Exact-pin advisory audit: " + report["status"] + " " + json.dumps(report.get("counts", {})))
    for result in report.get("results", []):
        if result["status"] != "pass":
            print(json.dumps({key: result[key] for key in ("name", "version", "status")}, sort_keys=True))
    if "inventory_error" in report:
        print(json.dumps({"inventory_error": report["inventory_error"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

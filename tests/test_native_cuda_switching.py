"""Package-selection and execution-boundary tests without Docker or installation."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_switching_test", ROOT / "packaging/native-runtime/qualify_switching.py")
switch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(switch)


class NativeCudaSwitchingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="native switching ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = "1" * 40
        self.data = {path: path.encode() for path in switch.ADAPTER_PATHS}
        self.data["pyproject.toml"] = b'[project]\nversion="2.7.11"\n'
        self.files = {path: hashlib.sha256(self.data[path]).hexdigest() for path in switch.ADAPTER_PATHS}
        self.records = {}
        self.arguments = ["--source", self.source, "--family", "rpm", "--image", "sha256:" + "4" * 64,
                          "--output", str(self.root / "result")]
        for variant in ("cpu", "cuda"):
            base = self.root / variant
            base.mkdir()
            (base / "package.rpm").write_bytes((variant + " fake package, never install").encode())
            (base / "content.json").write_text('{}\n')
            record = {"family": "rpm", "variant": variant, "version": "2.7.11-5." + variant,
                      "artifact": "package.rpm", "sha256": switch.digest(base / "package.rpm"),
                      "bytes": (base / "package.rpm").stat().st_size, "content": "content.json",
                      "content_sha256": switch.digest(base / "content.json"), "runtime_manifest_sha256": ("2" if variant == "cpu" else "3") * 64,
                      "source": {"revision": self.source, "version": "2.7.11", "source_date_epoch": 123},
                      "adapter": {"revision": self.source, "tree": "6" * 40, "files": self.files}}
            self.records[variant] = record
            (base / "packages.json").write_text(json.dumps([record]))
            self.arguments += ["--" + variant + "-packages", str(base), "--" + variant + "-manifest-sha256", switch.digest(base / "packages.json")]
        patch = mock.patch.object(switch.smoke, "git_bytes", side_effect=lambda revision, path: self.data[path])
        patch.start()
        self.addCleanup(patch.stop)

    def args(self):
        return switch.parse_arguments(self.arguments)

    def rewrite(self, args, variant, records):
        p = self.root / variant / "packages.json"
        p.write_text(json.dumps(records))
        setattr(args, variant + "_manifest_sha256", switch.digest(p))

    def test_default_plan_is_read_only_and_never_calls_docker(self):
        before = sorted(str(p) for p in self.root.rglob("*"))
        output = io.StringIO()
        with mock.patch.object(switch, "execute", side_effect=AssertionError("execution")), \
             mock.patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(switch.main(self.arguments), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["status"], "planned-not-executed")
        self.assertEqual(plan["sequence"], ["cpu", "cuda", "cpu"])
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*")), before)

    def test_interrupted_receipt_publication_retains_complete_last_state(self):
        path = self.root / "result.json"
        old = {"status": "running", "steps": [{"name": "installed-cpu"}]}
        path.write_text(json.dumps(old))
        before = set(self.root.iterdir())
        new = {"status": "pass", "steps": [{"name": "installed-cpu"}, {"name": "installed-cuda"}]}
        with mock.patch.object(switch.os, "replace", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            switch.save_result(path, new)
        self.assertEqual(json.loads(path.read_text()), old)
        self.assertEqual(set(self.root.iterdir()), before)
        original_replace = os.replace
        def inspect_then_replace(source, destination):
            self.assertEqual(json.loads(path.read_text()), old)
            self.assertEqual(json.loads(Path(source).read_text()), new)
            self.assertEqual(Path(source).parent, path.parent)
            original_replace(source, destination)
        with mock.patch.object(switch.os, "replace", side_effect=inspect_then_replace):
            switch.save_result(path, new)
        self.assertEqual(json.loads(path.read_text()), new)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_explicit_gpu_uuid_is_required(self):
        for value in ("all", "0", "device=0", "GPU-unknown", "GPU-a;true"):
            args = self.args()
            args.gpu_device = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                switch.make_plan(args)
        args.gpu_device = "GPU-1c6cff52-f9a0-d675-67ef-0474224bb868"
        self.assertEqual(switch.make_plan(args)["gpu_device"], args.gpu_device)

    def test_missing_bound_identities_and_existing_output_are_rejected(self):
        for field, value in (("source", "main"), ("source", "0" * 40), ("image", "fedora:44"),
                             ("cpu_manifest_sha256", "0" * 64), ("cuda_manifest_sha256", "f" * 64)):
            args = self.args()
            setattr(args, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                switch.make_plan(args)
        args = self.args()
        args.output.mkdir()
        (args.output / "sentinel").write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh"):
            switch.make_plan(args)
        self.assertEqual((args.output / "sentinel").read_text(), "keep")

    def test_manifest_variant_source_adapter_revision_and_version_are_bound(self):
        changes = [("variant", "cpu"), ("version", "2.7.11-4.cuda"), ("version", "2.7.12-5.cuda"),
                   ("source", {"revision": "8" * 40, "version": "2.7.11"}),
                   ("adapter", {"revision": self.source, "files": {}})]
        for key, value in changes:
            args = self.args()
            record = copy.deepcopy(self.records["cuda"])
            record[key] = value
            self.rewrite(args, "cuda", [record])
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                switch.make_plan(args)
        args = self.args()
        self.rewrite(args, "cuda", [self.records["cuda"], self.records["cuda"]])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            switch.make_plan(args)

    def test_matching_wrong_native_versions_cannot_override_approved_source(self):
        args = self.args()
        for variant in ("cpu", "cuda"):
            record = copy.deepcopy(self.records[variant])
            record["version"] = "9.9.9-5." + variant
            self.rewrite(args, variant, [record])
        with self.assertRaisesRegex(ValueError, "source adapter or version"):
            switch.make_plan(args)

    def test_artifact_and_native_content_must_match_trusted_manifest(self):
        for name in ("package.rpm", "content.json"):
            path = self.root / "cuda" / name
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "identity mismatch"):
                switch.make_plan(self.args())
            path.write_bytes(original)

    def test_duplicate_json_keys_are_rejected_even_with_matching_digest(self):
        path = self.root / "cuda" / "packages.json"
        path.write_text('[{"family":"rpm","family":"deb"}]')
        args = self.args()
        args.cuda_manifest_sha256 = switch.digest(path)
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            switch.make_plan(args)

    def test_links_traversal_and_oversized_files_fail_before_execution(self):
        artifact = self.root / "cpu" / "package.rpm"
        real = artifact.with_name("real.rpm")
        artifact.rename(real)
        artifact.symlink_to(real)
        with self.assertRaises(OSError):
            switch.make_plan(self.args())
        for relative in ("../cuda/package.rpm", "/tmp/package.rpm"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                switch.contained(self.root / "cpu", relative)
        huge = self.root / "sparse"
        with huge.open("wb") as stream:
            stream.truncate(switch.MAX_PACKAGE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "bounded regular"):
            switch.digest(huge)

    def test_hashing_rejects_a_file_that_changes_during_the_read(self):
        path = self.root / "changing"
        path.write_bytes(b"original")
        original_regular = switch.regular

        @contextlib.contextmanager
        def changing_regular(name, maximum):
            with original_regular(name, maximum) as stream:
                original_read = stream.read
                def read(size):
                    result = original_read(size)
                    if result:
                        with path.open("ab") as writer:
                            writer.write(b"changed")
                        stream.read = original_read
                    return result
                stream.read = read
                yield stream

        with mock.patch.object(switch, "regular", changing_regular), self.assertRaisesRegex(ValueError, "changed while hashing"):
            switch.digest(path)

    def test_snapshots_are_distinct_private_copies_and_refuse_overwrite(self):
        original = self.root / "cuda" / "package.rpm"
        target = self.root / "snapshot"
        switch.snapshot(original, target, switch.digest(original))
        self.assertEqual(target.read_bytes(), original.read_bytes())
        self.assertNotEqual(target.stat().st_ino, original.stat().st_ino)
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            switch.snapshot(original, target, switch.digest(original))
        with self.assertRaisesRegex(ValueError, "reviewed input"):
            switch.snapshot(original, self.root / "bad", "f" * 64)

    def test_install_and_probe_commands_select_the_current_variant(self):
        self.assertIn("--oldpackage", switch.install_command("rpm", "cpu"))
        for variant in ("cpu", "cuda", "cpu"):
            for gpu in (None, "GPU-1c6cff52-f9a0-d675-67ef-0474224bb868"):
                command = switch.probe_command(variant, gpu, "3.13.16")
                self.assertEqual(command[command.index("--variant") + 1], variant)
                self.assertEqual("--cuda-unavailable" in command, variant == "cuda" and not gpu)

    def test_container_exposes_only_explicit_gpu_and_readonly_inputs(self):
        plan = switch.make_plan(self.args())
        paths = {v: self.root / (v + ".rpm") for v in ("cpu", "cuda")}
        for gpu in (None, "GPU-1c6cff52-f9a0-d675-67ef-0474224bb868"):
            plan["gpu_device"] = gpu
            command = switch.container_command(plan, "task-container", paths, self.root / "probes")
            self.assertIn("--network=none", command)
            self.assertIn("--memory", command)
            self.assertEqual("--gpus" in command, bool(gpu))
            self.assertEqual("--runtime=runc" in command, not gpu)
            mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
            self.assertEqual(len(mounts), 3)
            self.assertTrue(all(value.endswith(",readonly") for value in mounts))
            self.assertFalse(any("/dev/" in value or "docker.sock" in value for value in mounts))

    def test_runtime_result_requires_actual_expected_provider_and_source(self):
        app = {"__init__.py": "f" * 64}
        report = {"status": "pass", "variant": "cuda", "app_python_hashes": app,
                  "cpu_execution": {"success": True}, "cuda_execution": {"success": False},
                  "cuda_unavailable_expected": True, "cupy_execution": None}
        render = lambda: "RESULT=" + json.dumps(report)
        switch.runtime_report(render(), "cuda", None, app)
        with self.assertRaisesRegex(ValueError, "CUDA execution"):
            switch.runtime_report(render(), "cuda", "GPU-uuid", app)
        report.update(cuda_execution={"success": True}, cuda_unavailable_expected=False)
        with self.assertRaisesRegex(ValueError, "CuPy"):
            switch.runtime_report(render(), "cuda", "GPU-uuid", app)
        report["cupy_execution"] = {"output": [float(i * i) for i in range(16)]}
        switch.runtime_report(render(), "cuda", "GPU-uuid", app)
        with self.assertRaisesRegex(ValueError, "identity"):
            switch.runtime_report(render(), "cpu", "GPU-uuid", app)
        with self.assertRaisesRegex(ValueError, "identity"):
            switch.runtime_report(render(), "cuda", "GPU-uuid", {"other.py": "e" * 64})
        with self.assertRaisesRegex(ValueError, "one runtime"):
            switch.runtime_report(render() + "\n" + render(), "cuda", "GPU-uuid", app)


if __name__ == "__main__":
    unittest.main()

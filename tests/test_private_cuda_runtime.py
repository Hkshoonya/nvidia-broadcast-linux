"""Variant ownership and scoped dependency-exclusion regressions."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from nvbroadcast.runtime.artifact import ArtifactEnvironment

ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "packaging/runtime-prototype"
sys.path.insert(0, str(PROTOTYPE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolve = load("cuda_lock_resolve", PROTOTYPE / "resolve.py")
probe = load("cuda_runtime_probe", PROTOTYPE / "probe.py")
build = load("cuda_native_build", ROOT / "packaging/native-prototype/build.py")


class PrivateCudaRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cuda runtime ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def wheel(self, requirements, version="1.2.1"):
        cache = self.root / "metadata-inputs"
        cache.mkdir()
        wheel = cache / "faster.whl"
        content = f"Metadata-Version: 2.1\nName: faster-whisper\nVersion: {version}\nRequires-Python: >=3.9\n"
        content += "Provides-Extra: dev\n" + "".join(f"Requires-Dist: {r}\n" for r in requirements)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("faster_whisper-1.2.1.dist-info/METADATA", content)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        (self.root / "pylock.linux-x86_64-cp313-cpu.toml").write_text(
            '[[packages]]\nname="faster-whisper"\nversion="1.2.1"\n'
            'wheels=[{url="https://example.invalid/faster.whl",hashes={sha256="' + digest + '"}}]\n')
        return wheel

    def test_scoped_override_preserves_other_edges_markers_and_wheel_bytes(self):
        other = ["ctranslate2<5,>=4", "av>=11", 'pytest==7.*; extra == "dev"',
                 'new-upstream-dependency>=2; python_version >= "3.11"']
        wheel = self.wheel(["onnxruntime<2,>=1.14", *other])
        before = wheel.read_bytes()
        with mock.patch.object(resolve.prepare, "HERE", self.root), mock.patch.object(resolve.prepare.urllib.request, "urlopen") as network:
            metadata = resolve.cuda_metadata(self.root, "1.2.1")
            network.assert_not_called()
        self.assertEqual(metadata["requires-dist"], other)
        self.assertEqual(metadata["provides-extra"], ["dev"])
        self.assertEqual(wheel.read_bytes(), before)
        record = json.loads((self.root / "cuda-metadata-override.json").read_text())
        self.assertEqual(record["omitted"], ["onnxruntime<2,>=1.14"])

    def test_override_rejects_tampered_wheel(self):
        wheel = self.wheel(["onnxruntime>=1.14,<2"])
        wheel.write_bytes(b"tampered")
        with mock.patch.object(resolve.prepare, "HERE", self.root), self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            resolve.cuda_metadata(self.root, "1.2.1")

    def test_override_rejects_unreviewed_conditional_edge(self):
        self.wheel(['onnxruntime>=1.14; python_version >= "3.11"'])
        with mock.patch.object(resolve.prepare, "HERE", self.root), self.assertRaisesRegex(ValueError, "unconditional"):
            resolve.cuda_metadata(self.root, "1.2.1")

    def distribution(self, name, version, requirements=()):
        directory = self.root / "lib/python3.13/site-packages" / f"{name.replace('-', '_')}-{version}.dist-info"
        directory.mkdir(parents=True)
        (directory / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n" +
                                           "".join(f"Requires-Dist: {r}\n" for r in requirements))

    def test_closure_checks_every_other_edge_and_rejects_unreviewed_ort_owner(self):
        self.distribution("faster-whisper", "1.2.1", ("onnxruntime>=1.14,<2", "backend>=2"))
        self.distribution("onnxruntime-gpu", "1.24.4")
        def environment():
            return ArtifactEnvironment.inspect(self.root, "amd64")

        self.assertTrue(any("missing package backend" in p for p in probe.closure_problems(environment(), "cuda")))
        self.distribution("backend", "2.1")
        self.assertEqual(probe.closure_problems(environment(), "cuda"), [])
        self.distribution("unreviewed-package", "1", ("onnxruntime>=1",))
        self.assertTrue(any("unreviewed CPU ORT" in p for p in probe.closure_problems(environment(), "cuda")))

    def test_package_label_must_match_unique_runtime_owner(self):
        self.distribution("onnxruntime-gpu", "1.24.4")
        build.verify_owner(self.root, "cuda")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build.verify_owner(self.root, "cpu")
        self.distribution("onnxruntime", "1.24.4")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build.verify_owner(self.root, "cuda")

    def test_cuda_split_launcher_rejects_cpu_runtime_and_old_app(self):
        version = build.versions("deb", 1, "cuda")
        script = self.root / "check-packages"
        script.write_text(build.version_check("deb", "app", version, "cuda"))
        script.chmod(0o755)
        fake = self.root / "dpkg-query"
        fake.write_text('#!/bin/sh\ncase "$*" in *nvbroadcast-runtime-cuda*) printf "%s" "$GPU_STATE";; '
                        '*) printf "%s" "$APP_STATE";; esac\n')
        fake.chmod(0o755)
        current = "install ok installed " + version
        for app, gpu, expected in ((current, current, 0), (current, "", 78),
                                   ("install ok installed " + build.versions("deb", 1), current, 78)):
            with self.subTest(app=app, gpu=gpu):
                env = {**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
                       "APP_STATE": app, "GPU_STATE": gpu}
                result = subprocess.run([str(script)], env=env, capture_output=True)
                self.assertEqual(result.returncode, expected)


if __name__ == "__main__":
    unittest.main()

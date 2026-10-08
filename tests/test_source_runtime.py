from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import venv
from types import ModuleType

from scripts import source_runtime as runtime
from nvbroadcast.runtime.artifact import ArtifactEnvironment
from nvbroadcast.runtime.variants import RuntimeVariant


class SourceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nvb source runtime ")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.store = runtime.SourceRuntimeStore(self.project)
        self.probe = mock.patch.object(
            runtime, "verify_runtime", return_value="cpu"
        ).start()
        self.addCleanup(mock.patch.stopall)

    def legacy(self):
        path = self.project / ".venv"
        (path / "bin").mkdir(parents=True)
        (path / "bin/python").write_bytes(b"previous interpreter")
        return path

    def activate(self, variant="cpu"):
        candidate = self.store.prepare()
        self.store.activate(candidate, variant, "none")
        return candidate

    def test_failed_candidate_probe_preserves_legacy_environment(self):
        previous = self.legacy()
        candidate = self.store.prepare()
        self.probe.side_effect = subprocess.CalledProcessError(1, ["python", "probe"])
        with self.assertRaises(subprocess.CalledProcessError):
            self.store.activate(candidate, "cuda", "none")
        self.assertEqual(self.store.active(), previous)
        self.assertEqual(
            (previous / "bin/python").read_bytes(), b"previous interpreter"
        )
        self.assertFalse(self.store.state_file.exists())
        self.store.discard(candidate)
        self.assertFalse(candidate.exists())

    def test_successful_upgrade_does_not_move_or_mutate_legacy(self):
        previous = self.legacy()
        candidate = self.activate()
        self.assertEqual(self.store.active(), candidate)
        self.assertEqual(self.store.state()["previous"]["path"], "legacy")
        self.assertEqual(
            (previous / "bin/python").read_bytes(), b"previous interpreter"
        )
        self.probe.assert_called_once_with(candidate, "cpu", "none")

    def test_same_variant_update_uses_new_path_and_retains_previous(self):
        first = self.activate()
        (first / "installed-file").write_text("original")
        second = self.activate()
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.state()["previous"]["path"], first.name)
        self.assertEqual((first / "installed-file").read_text(), "original")

    def test_failed_update_preserves_both_selected_generations(self):
        first = self.activate()
        second = self.activate()
        state = self.store.state_file.read_bytes()
        third = self.store.prepare()
        self.probe.side_effect = RuntimeError("missing core dependency")
        with self.assertRaisesRegex(RuntimeError, "missing core"):
            self.store.activate(third, "cuda", "all")
        self.assertEqual(self.store.state_file.read_bytes(), state)
        self.assertTrue(first.is_dir())
        self.assertEqual(self.store.active(), second)

    def test_rollback_reprobes_previous_and_can_return_to_new_generation(self):
        first = self.activate()
        second = self.activate("cuda")
        self.probe.reset_mock()
        self.store.rollback()
        self.probe.assert_called_once_with(first, "cpu", "none")
        self.assertEqual(self.store.active(), first)
        self.probe.return_value = "cuda"
        self.store.rollback()
        self.assertEqual(self.store.active(), second)

    def test_rollback_to_legacy_preserves_its_original_prefix(self):
        legacy = self.legacy()
        candidate = self.activate()
        self.store.rollback()
        self.assertEqual(self.store.active(), legacy)
        self.assertEqual(self.store.state()["previous"]["path"], candidate.name)
        self.probe.assert_called_with(legacy, None, "none")

    def test_failed_rollback_keeps_current_selection(self):
        self.activate()
        current = self.activate()
        self.probe.side_effect = RuntimeError("previous provider no longer works")
        with self.assertRaisesRegex(RuntimeError, "no longer works"):
            self.store.rollback()
        self.assertEqual(self.store.active(), current)

    def test_two_concurrent_candidates_cannot_overwrite_new_selection(self):
        first = self.store.prepare()
        stale = self.store.prepare()
        self.store.activate(first, "cpu", "none")
        self.probe.reset_mock()
        with self.assertRaisesRegex(runtime.SourceRuntimeError, "selection changed"):
            self.store.activate(stale, "cuda", "none")
        self.probe.assert_not_called()
        self.assertEqual(self.store.active(), first)

    def test_live_app_or_service_prevents_activation_and_rollback(self):
        self.activate()
        current = self.activate()
        candidate = self.store.prepare()
        with mock.patch.object(
            runtime, "find_source_processes", return_value=[object()]
        ):
            for operation in (
                lambda: self.store.activate(candidate, "cpu", "none"),
                self.store.rollback,
            ):
                with (
                    self.subTest(operation=operation),
                    self.assertRaisesRegex(
                        runtime.SourceRuntimeError, "Stop NVBroadcast"
                    ),
                ):
                    operation()
        self.assertEqual(self.store.active(), current)

    def test_replace_failure_does_not_change_selection(self):
        current = self.activate()
        candidate = self.store.prepare()
        with mock.patch.object(runtime.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.activate(candidate, "cpu", "none")
        self.assertEqual(self.store.active(), current)
        self.assertEqual(list(self.store.root.glob(".selection.json-*")), [])

    def test_directory_sync_failure_reports_already_changed_selection(self):
        self.activate()
        candidate = self.store.prepare()
        with mock.patch.object(
            runtime.os, "fsync", side_effect=[None, OSError("I/O error")]
        ):
            with self.assertRaisesRegex(
                runtime.SourceRuntimeError, "may already have changed"
            ):
                self.store.activate(candidate, "cpu", "none")
        self.assertEqual(self.store.active(), candidate)
        with self.assertRaisesRegex(runtime.SourceRuntimeError, "Cannot discard"):
            self.store.discard(candidate)

    def test_discard_never_removes_active_or_previous(self):
        previous = self.activate()
        current = self.activate()
        for candidate in (previous, current):
            with (
                self.subTest(candidate=candidate),
                self.assertRaisesRegex(runtime.SourceRuntimeError, "Cannot discard"),
            ):
                self.store.discard(candidate)
            self.assertTrue(candidate.exists())

    def test_invalid_state_does_not_fall_back_to_legacy(self):
        self.legacy()
        self.store.initialize()
        runtime.write_json(
            self.store.state_file, {"schema": 1, "active": {"path": "../../other"}}
        )
        with self.assertRaisesRegex(runtime.SourceRuntimeError, "selection path"):
            self.store.active()

    def test_store_symlink_and_world_writable_directory_are_rejected(self):
        external = self.project / "external"
        external.mkdir()
        self.store.root.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(
            runtime.SourceRuntimeError, "Unsafe runtime directory"
        ):
            self.store.prepare()
        self.store.root.unlink()
        self.store.root.mkdir(mode=0o777)
        self.store.root.chmod(0o777)
        with self.assertRaisesRegex(
            runtime.SourceRuntimeError, "Unsafe runtime directory"
        ):
            self.store.prepare()

    def test_metadata_symlink_is_rejected(self):
        current = self.activate()
        candidate = self.store.prepare()
        marker = candidate / runtime.MARKER
        data = marker.read_text()
        external = self.project / "external.json"
        external.write_text(data)
        marker.unlink()
        marker.symlink_to(external)
        with self.assertRaises(OSError):
            self.store.activate(candidate, "cpu", "none")
        self.assertEqual(self.store.active(), current)

    def test_candidate_from_another_store_is_rejected(self):
        external_project = self.project / "other"
        external_project.mkdir()
        other = runtime.SourceRuntimeStore(external_project)
        with self.assertRaisesRegex(runtime.SourceRuntimeError, "outside"):
            self.store.activate(other.prepare(), "cpu", "none")

    def test_parallel_mutation_fails_without_blocking(self):
        with self.store.locked():
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    runtime.__file__,
                    "--project",
                    str(self.project),
                    "prepare",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.store.state_file.exists())

    def test_inspects_previous_optional_features_without_importing_them(self):
        previous = self.project / ".venv"
        venv.EnvBuilder(with_pip=False).create(previous)
        purelib = Path(
            subprocess.check_output(
                [
                    str(previous / "bin/python"),
                    "-I",
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ],
                text=True,
            ).strip()
        )
        self.assertEqual(self.store.features(), {"meeting": "none", "tensorrt": False})
        for distribution, expected in (
            ("faster_whisper", "faster"),
            ("openai_whisper", "all"),
        ):
            metadata = purelib / f"{distribution}-1.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(f"Name: {distribution}\nVersion: 1.0\n")
            # No importable ML module exists; only installed metadata is needed.
            self.assertEqual(self.store.features()["meeting"], expected)
        metadata = purelib / "tensorrt_cu12_libs-10.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: tensorrt-cu12-libs\nVersion: 10.0\n")
        self.assertEqual(self.store.features(), {"meeting": "all", "tensorrt": True})

    def test_fresh_source_install_has_no_optional_features_to_preserve(self):
        self.assertEqual(self.store.features(), {"meeting": "none", "tensorrt": False})

    def test_generated_launchers_follow_selection_with_spaces_and_quoted_arguments(
        self,
    ):
        # Real venv prefixes and real launcher processes; the tiny test module
        # reports exactly which generation, arguments and user-site policy ran.
        for path in (self.project / ".venv", self.store.prepare()):
            venv.EnvBuilder(with_pip=False).create(path)
            purelib = subprocess.check_output(
                [
                    str(path / "bin/python"),
                    "-I",
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ],
                text=True,
            ).strip()
            package = Path(purelib) / "nvbroadcast"
            package.mkdir()
            (package / "__init__.py").touch()
            code = "import json,sys,site; print(json.dumps([sys.prefix,sys.argv[1:],site.ENABLE_USER_SITE]))"
            (package / "__main__.py").write_text(code)
            (package / "vcam_service.py").write_text(code)
            if path.name != ".venv":
                candidate = path
        prefix = self.project / 'launch prefix $dollar "quote"'
        self.store.launchers(prefix)
        arguments = ["a b", "$(touch never)", '"quoted"']
        for expected in (self.project / ".venv", candidate):
            if expected == candidate:
                self.store.activate(candidate, "cpu", "none")
            for launcher in ("nvbroadcast", "nvbroadcast-vcam"):
                output = subprocess.check_output(
                    [str(prefix / "bin" / launcher), *arguments], text=True
                )
                self.assertEqual(json.loads(output), [str(expected), arguments, False])
        self.store.rollback()
        output = subprocess.check_output([str(prefix / "bin/nvbroadcast")], text=True)
        self.assertEqual(json.loads(output)[0], str(self.project / ".venv"))


class SourceRuntimeVerificationExtrasTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / "lib/python3.12/site-packages"
        self.site.mkdir(parents=True)
        self.gi = ModuleType("gi")
        self.gi.require_version = mock.Mock()
        effects = ModuleType("nvbroadcast.video.effects")
        effects.VideoEffects = mock.Mock()
        self.modules = {"gi": self.gi, "nvbroadcast.video.effects": effects,
                        "nvbroadcast.app": ModuleType("nvbroadcast.app"),
                        "faster_whisper": ModuleType("faster_whisper"),
                        "whisper": ModuleType("whisper")}

    def distribution(self, name, version="1.0", requirements=()):
        directory = self.site / f"{name.replace('-', '_')}-{version}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n" +
            "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
        )

    def run_verification(self, variant, meeting):
        environment = ArtifactEnvironment.inspect(self.root, "amd64")

        def run(arguments, **_kwargs):
            if arguments[2] == "-c":
                # Execute the exact candidate verification program against
                # real fixture metadata. Native imports and the later device
                # execution probe are replaced; no ML runtime is loaded.
                output = io.StringIO()
                with mock.patch.object(sys, "argv", ["-c", *arguments[4:]]), redirect_stdout(output):
                    exec(arguments[3], {})
                return subprocess.CompletedProcess(arguments, 0, output.getvalue(), "")
            return subprocess.CompletedProcess(arguments, 0)

        with (
            mock.patch.dict(sys.modules, self.modules),
            mock.patch("nvbroadcast.runtime.variants.detect_runtime_variant",
                       return_value=RuntimeVariant(variant)),
            mock.patch.object(importlib.metadata, "distributions",
                              return_value=environment.distributions),
            mock.patch.object(runtime.subprocess, "run", side_effect=run),
            mock.patch.object(importlib, "import_module", return_value=ModuleType("fixture")),
        ):
            return runtime.verify_runtime(self.root, variant, meeting)

    def test_candidate_verification_rejects_missing_selected_cuda_extra(self):
        self.distribution("nvbroadcast", requirements=(
            'cupy-cuda12x>=14.1.1,<15; extra == "cuda"',
        ))
        self.distribution("onnxruntime-gpu", "1.24.4")
        with self.assertRaisesRegex(RuntimeError, "missing package cupy-cuda12x"):
            self.run_verification("cuda", "none")
        self.distribution("cupy-cuda12x", "14.2.0")
        self.assertEqual(self.run_verification("cuda", "none"), "cuda")

    def test_candidate_verification_selects_only_requested_meeting_extras(self):
        self.distribution("nvbroadcast", requirements=(
            'support-leaf; extra == "meeting-support"',
            'compatibility-leaf; extra == "meeting"',
        ))
        self.distribution("faster-whisper", "1.2.1")
        self.distribution("openai-whisper", "1.0")
        self.distribution("onnxruntime", "1.24.4")
        self.assertEqual(self.run_verification("cpu", "none"), "cpu")
        with self.assertRaisesRegex(RuntimeError, "missing package support-leaf"):
            self.run_verification("cpu", "faster")
        self.distribution("support-leaf")
        self.assertEqual(self.run_verification("cpu", "faster"), "cpu")
        with self.assertRaisesRegex(RuntimeError, "missing package compatibility-leaf"):
            self.run_verification("cpu", "all")
        self.distribution("compatibility-leaf")
        self.assertEqual(self.run_verification("cpu", "all"), "cpu")

    def test_older_generation_uses_current_checker_without_importing_checkout_app(self):
        self.distribution("nvbroadcast", requirements=(
            'required-leaf; extra == "cpu"',
        ))
        self.distribution("onnxruntime", "1.24.4")
        old = ModuleType("nvbroadcast.runtime.artifact")
        old.ArtifactEnvironment = mock.Mock()
        old.ArtifactEnvironment.current.side_effect = TypeError("old checker API")
        self.modules[old.__name__] = old
        with self.assertRaisesRegex(RuntimeError, "missing package required-leaf"):
            self.run_verification("cpu", "none")
        self.distribution("required-leaf")
        previous_paths = list(sys.path)
        self.assertEqual(self.run_verification("cpu", "none"), "cpu")
        self.assertEqual(sys.path, previous_paths)
        old.ArtifactEnvironment.current.assert_not_called()


if __name__ == "__main__":
    unittest.main()

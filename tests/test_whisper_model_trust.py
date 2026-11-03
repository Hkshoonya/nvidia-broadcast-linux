"""Model downloads may only be loaded from pinned, checked snapshots."""

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from multiprocessing import get_context
from unittest import mock

from nvbroadcast.ai import model_trust, transcriber


REPO = "Systran/faster-whisper-tiny"
REVISION = "a" * 40


class WhisperModelTrustTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        files = {}
        for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
            data = f"sample {name}".encode()
            (self.snapshot / name).write_bytes(data)
            files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.manifest = self.root / "whisper-models.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "upstream": "faster-whisper==1.2.1",
                    "models": {REPO: {"revision": REVISION, "aliases": ["tiny"], "files": files}},
                }
            ),
            encoding="utf-8",
        )
        patcher = mock.patch.object(model_trust, "_MANIFEST", self.manifest)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_alias_download_uses_commit_and_checks_files(self):
        download = mock.Mock(return_value=str(self.snapshot))

        path = model_trust.verified_model_path("tiny", "1.2.1", download)

        self.assertEqual(path, self.snapshot)
        download.assert_called_once_with(REPO, revision=REVISION, use_auth_token=False)

    def test_repo_id_is_also_pinned(self):
        download = mock.Mock(return_value=str(self.snapshot))
        self.assertEqual(
            model_trust.verified_model_path(REPO, "1.2.1", download), self.snapshot
        )
        download.assert_called_once_with(REPO, revision=REVISION, use_auth_token=False)

    def test_modified_weight_is_rejected(self):
        weight = self.snapshot / "model.bin"
        weight.write_bytes(b"x" * weight.stat().st_size)
        with self.assertRaisesRegex(model_trust.ModelTrustError, "SHA-256 mismatch"):
            model_trust.verified_model_path(
                "tiny", "1.2.1", mock.Mock(return_value=str(self.snapshot))
            )

    def test_missing_tokenizer_is_rejected(self):
        (self.snapshot / "tokenizer.json").unlink()
        with self.assertRaisesRegex(model_trust.ModelTrustError, "Missing or wrong-size"):
            model_trust.verified_model_path(
                "tiny", "1.2.1", mock.Mock(return_value=str(self.snapshot))
            )

    def test_unpinned_optional_model_input_is_rejected(self):
        for filename in ("vocabulary.json", "preprocessor_config.json"):
            path = self.snapshot / filename
            path.write_text("{}", encoding="utf-8")
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(
                    model_trust.ModelTrustError, "Unpinned faster-whisper file"
                ):
                    model_trust.verified_model_path(
                        "tiny", "1.2.1", mock.Mock(return_value=str(self.snapshot))
                    )
            path.unlink()

    def test_unknown_remote_model_never_downloads(self):
        download = mock.Mock()
        with self.assertRaisesRegex(model_trust.ModelTrustError, "Unpinned"):
            model_trust.verified_model_path("someone/custom-whisper", "1.2.1", download)
        download.assert_not_called()

    def test_wrong_faster_whisper_version_never_downloads(self):
        download = mock.Mock()
        with self.assertRaisesRegex(model_trust.ModelTrustError, "version"):
            model_trust.verified_model_path("tiny", "1.3.0", download)
        download.assert_not_called()

    def test_pinned_revision_download_failure_is_a_trust_error(self):
        download = mock.Mock(side_effect=OSError("offline or missing revision"))
        with self.assertRaisesRegex(model_trust.ModelTrustError, "Cannot resolve pinned"):
            model_trust.verified_model_path("tiny", "1.2.1", download)

    def test_explicit_local_directory_never_downloads(self):
        download = mock.Mock()
        self.assertEqual(
            model_trust.verified_model_path(str(self.snapshot), "1.2.1", download),
            self.snapshot,
        )
        download.assert_not_called()

    def test_existing_relative_directory_takes_precedence_over_alias(self):
        original_directory = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original_directory)
        Path("tiny").mkdir()
        Path("custom-model").mkdir()
        download = mock.Mock()

        self.assertEqual(
            model_trust.verified_model_path("tiny", "1.2.1", download), Path("tiny")
        )
        self.assertEqual(
            model_trust.verified_model_path("custom-model", "1.2.1", download),
            Path("custom-model"),
        )
        download.assert_not_called()

    def test_cold_large_model_budget_includes_download_and_hashing(self):
        with mock.patch.object(transcriber, "expected_model_bytes", return_value=3_090_835_702):
            timeout_s = transcriber._model_load_timeout_s("large", "faster-whisper")
        self.assertGreaterEqual(timeout_s, 3000)
        self.assertLessEqual(timeout_s, 3600)

    def test_timeout_terminates_real_worker(self):
        executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
        future = executor.submit(time.sleep, 30)
        workers = tuple(executor._processes.values())
        try:
            with self.assertRaisesRegex(
                transcriber.WorkerDeadlineExceeded, "worker was terminated"
            ):
                transcriber._await_worker_result(executor, future, 0.1, "Test model")
            self.assertTrue(workers)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=2)
            executor.shutdown(wait=False, cancel_futures=True)

    def test_initialize_reports_trust_error_from_spawned_worker(self):
        # A real spawn, with a tiny fake faster-whisper distribution, proves
        # that initialize receives the trust error instead of BrokenProcessPool.
        fake_package = self.root / "fake-package"
        fake_package.mkdir()
        (fake_package / "faster_whisper.py").write_text(
            '__version__ = "1.2.1"\n'
            'def download_model(*args, **kwargs):\n'
            '    raise AssertionError("unexpected model download")\n'
            'class WhisperModel:\n'
            '    pass\n',
            encoding="utf-8",
        )
        output = io.StringIO()
        meeting = transcriber.MeetingTranscriber("unlisted-model")
        meeting._backend_preference = "faster-whisper"
        with mock.patch.object(sys, "path", [str(fake_package), *sys.path]), \
             mock.patch.object(transcriber, "_has_supported_backend", return_value=True), \
             redirect_stdout(output):
            self.assertFalse(meeting.initialize())

        self.assertIn("Unpinned faster-whisper model", output.getvalue())
        self.assertNotIn("process pool was terminated abruptly", output.getvalue())
        self.assertIsNone(meeting._executor)

    def test_auto_trust_failure_cannot_fall_back_to_openai(self):
        weight = self.snapshot / "model.bin"
        weight.write_bytes(b"x" * weight.stat().st_size)
        faster = types.ModuleType("faster_whisper")
        faster.__version__ = "1.2.1"
        faster.download_model = mock.Mock(return_value=str(self.snapshot))
        faster.WhisperModel = mock.Mock()
        openai = types.ModuleType("whisper")
        openai.load_model = mock.Mock()

        with mock.patch.dict(sys.modules, {"faster_whisper": faster, "whisper": openai}):
            with self.assertRaisesRegex(model_trust.ModelTrustError, "SHA-256 mismatch"):
                transcriber._init_transcriber_worker("tiny", "cpu", "auto")

        faster.WhisperModel.assert_not_called()
        openai.load_model.assert_not_called()

    def test_verified_local_path_is_passed_to_loader(self):
        faster = types.ModuleType("faster_whisper")
        faster.__version__ = "1.2.1"
        faster.download_model = mock.Mock(return_value=str(self.snapshot))
        faster.WhisperModel = mock.Mock()

        with mock.patch.dict(sys.modules, {"faster_whisper": faster}):
            transcriber._init_transcriber_worker("tiny", "cpu", "faster-whisper")

        faster.WhisperModel.assert_called_once_with(
            str(self.snapshot), device="cpu", compute_type="int8"
        )

    def test_auto_runtime_failure_keeps_openai_fallback(self):
        faster = types.ModuleType("faster_whisper")
        faster.__version__ = "1.2.1"
        faster.download_model = mock.Mock(return_value=str(self.snapshot))
        faster.WhisperModel = mock.Mock(side_effect=RuntimeError("runtime unavailable"))
        openai = types.ModuleType("whisper")
        openai.load_model = mock.Mock()

        with mock.patch.dict(sys.modules, {"faster_whisper": faster, "whisper": openai}), \
             mock.patch.object(transcriber, "supports_openai_whisper_python", return_value=True):
            transcriber._init_transcriber_worker("tiny", "cpu", "auto")

        faster.WhisperModel.assert_called_once()
        openai.load_model.assert_called_once_with("tiny", device="cpu")

    def test_explicit_openai_whisper_backend_remains_available(self):
        openai = types.ModuleType("whisper")
        openai.load_model = mock.Mock()
        with mock.patch.dict(sys.modules, {"whisper": openai}):
            transcriber._init_transcriber_worker("tiny", "cpu", "whisper")
        openai.load_model.assert_called_once_with("tiny", device="cpu")


if __name__ == "__main__":
    unittest.main()

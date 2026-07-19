import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nvbroadcast.core import model_download


def _response(payload: bytes):
    chunks = [payload, b""]
    response = mock.MagicMock()
    response.read.side_effect = lambda _size: chunks.pop(0)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class VerifiedModelDownloadTests(unittest.TestCase):
    def test_rejects_path_traversal_and_non_https_urls(self):
        with self.assertRaisesRegex(ValueError, "filename"):
            model_download.download_verified_model(
                "../model.onnx",
                "https://example.invalid/model.onnx",
                "0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            model_download.download_verified_model(
                "model.onnx",
                "http://example.invalid/model.onnx",
                "0" * 64,
            )

    def test_valid_bundled_model_is_used_without_network_access(self):
        payload = b"bundled model"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundled = root / "bundled"
            bundled.mkdir()
            expected = bundled / "model.onnx"
            expected.write_bytes(payload)

            with mock.patch("urllib.request.urlopen") as urlopen:
                result = model_download.download_verified_model(
                    "model.onnx",
                    "https://example.invalid/model.onnx",
                    digest,
                    bundled_dir=bundled,
                    cache_dir=root / "cache",
                )

            self.assertEqual(result, expected)
            urlopen.assert_not_called()

    def test_download_uses_writable_cache_and_atomic_verified_file(self):
        payload = b"downloaded model"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            with mock.patch(
                "urllib.request.urlopen",
                return_value=_response(payload),
            ) as urlopen:
                result = model_download.download_verified_model(
                    "model.onnx",
                    "https://example.invalid/model.onnx",
                    digest,
                    bundled_dir=root / "read-only-install-models",
                    cache_dir=cache,
                )

            self.assertEqual(result, cache / "model.onnx")
            self.assertEqual(result.read_bytes(), payload)
            self.assertFalse(list(cache.glob("*.part")))
            urlopen.assert_called_once_with(
                "https://example.invalid/model.onnx",
                timeout=model_download.DOWNLOAD_TIMEOUT_S,
            )
            if os.name == "posix":
                self.assertEqual(result.stat().st_mode & 0o777, 0o600)

    def test_checksum_mismatch_preserves_previous_cache_and_cleans_partial(self):
        payload = b"tampered model"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            target = cache / "model.onnx"
            target.write_bytes(b"previous invalid cache")

            with mock.patch(
                "urllib.request.urlopen",
                return_value=_response(payload),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    model_download.download_verified_model(
                        "model.onnx",
                        "https://example.invalid/model.onnx",
                        "0" * 64,
                        bundled_dir=root / "bundled",
                        cache_dir=cache,
                    )

            self.assertEqual(target.read_bytes(), b"previous invalid cache")
            self.assertEqual(list(cache.iterdir()), [target])

    def test_snap_cache_uses_snap_user_common(self):
        with mock.patch.dict(
            os.environ,
            {"SNAP_USER_COMMON": "/tmp/nvb-snap-user"},
            clear=True,
        ):
            self.assertEqual(
                model_download.model_cache_dir(),
                Path("/tmp/nvb-snap-user/models"),
            )


if __name__ == "__main__":
    unittest.main()

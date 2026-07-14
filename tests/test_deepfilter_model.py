import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nvbroadcast.audio import deepfilter


def _fake_response(payload: bytes):
    """Context-manager response mock whose read() streams payload once."""
    chunks = [payload, b""]
    response = mock.MagicMock()
    response.read.side_effect = lambda _n: chunks.pop(0)
    cm = mock.MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return cm


def _write_fused_model(path: Path):
    """Tiny graph with the one node shape _prepare_cuda_compatible patches."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node("FusedConv", ["x", "w"], ["y"],
                            name="conv0", domain="com.microsoft")
    node.attribute.append(helper.make_attribute("activation", "Sigmoid"))
    graph = helper.make_graph([node], "tiny", [x, w], [y])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13),
        helper.make_opsetid("com.microsoft", 1),
    ])
    onnx.save(model, str(path))


class ModelDownloadTests(unittest.TestCase):
    def test_download_is_timeout_bounded_and_verified(self):
        payload = b"fake model bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(deepfilter, "_MODELS_DIR", Path(tmp)), \
                 mock.patch.object(deepfilter, "MODEL_SHA256", digest), \
                 mock.patch("urllib.request.urlopen",
                            return_value=_fake_response(payload)) as urlopen:
                path = deepfilter._download_model()

            urlopen.assert_called_once_with(
                deepfilter.MODEL_URL, timeout=deepfilter.DOWNLOAD_TIMEOUT_S)
            self.assertEqual(path.read_bytes(), payload)

    def test_checksum_mismatch_deletes_partial_and_raises(self):
        payload = b"tampered bytes"
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(deepfilter, "_MODELS_DIR", Path(tmp)), \
                 mock.patch.object(deepfilter, "MODEL_SHA256", "0" * 64), \
                 mock.patch("urllib.request.urlopen",
                            return_value=_fake_response(payload)):
                with self.assertRaises(RuntimeError):
                    deepfilter._download_model()
            self.assertEqual(list(Path(tmp).iterdir()), [],
                             "partial download must not be left behind")


class PatchedGraphCacheTests(unittest.TestCase):
    """The cached _unfused.onnx must be verified, never trusted blindly."""

    def _prepare(self, tmp: Path):
        source = tmp / "model.onnx"
        if not source.exists():
            _write_fused_model(source)
        with mock.patch.object(deepfilter, "MODEL_SHA256",
                               deepfilter._sha256(source)):
            return deepfilter._prepare_cuda_compatible(source)

    def test_generates_patched_file_with_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            patched = self._prepare(Path(tmp))
            self.assertTrue(patched.name.endswith("_unfused.onnx"))
            stamp = patched.with_name(patched.name + ".sha256")
            recorded_source, recorded_patched = stamp.read_text().split()
            self.assertEqual(recorded_patched, deepfilter._sha256(patched))

    def test_valid_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            patched = self._prepare(Path(tmp))
            first_mtime = patched.stat().st_mtime_ns
            self.assertEqual(self._prepare(Path(tmp)), patched)
            self.assertEqual(patched.stat().st_mtime_ns, first_mtime,
                             "valid cached graph should not be regenerated")

    def test_corrupted_cache_is_regenerated(self):
        import onnx
        with tempfile.TemporaryDirectory() as tmp:
            patched = self._prepare(Path(tmp))
            good = patched.read_bytes()
            patched.write_bytes(good + b"garbage")
            regenerated = self._prepare(Path(tmp))
            self.assertEqual(regenerated.read_bytes(), good)
            onnx.load(str(regenerated))  # must be a loadable graph again

    def test_missing_stamp_forces_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            patched = self._prepare(Path(tmp))
            stamp = patched.with_name(patched.name + ".sha256")
            stamp.unlink()
            # A patched file of unknown provenance must not be trusted even
            # if its bytes happen to be fine.
            patched.write_bytes(b"unknown provenance")
            regenerated = self._prepare(Path(tmp))
            self.assertTrue(stamp.exists())
            self.assertNotEqual(regenerated.read_bytes(), b"unknown provenance")


if __name__ == "__main__":
    unittest.main()

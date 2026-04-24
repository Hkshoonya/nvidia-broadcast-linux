import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from nvbroadcast.video.effects import _RVMBackend, _prepare_rvm_tensorrt_model


def _tensor_dims(value_info) -> list[int | str]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


class TensorrtRvmTests(unittest.TestCase):
    def test_prepare_rvm_tensorrt_model_decouples_recurrent_symbols(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "dummy_rvm_trt.onnx"
            config_dir = root / "config"

            graph = helper.make_graph(
                nodes=[],
                name="dummy-rvm",
                inputs=[
                    helper.make_tensor_value_info("src", TensorProto.FLOAT, ["batch_size", 3, "height", "width"]),
                    helper.make_tensor_value_info("r1i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("r2i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("r3i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("r4i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("downsample_ratio", TensorProto.FLOAT, [1]),
                ],
                outputs=[
                    helper.make_tensor_value_info("fgr", TensorProto.FLOAT, ["batch_size", 3, "height", "width"]),
                    helper.make_tensor_value_info("pha", TensorProto.FLOAT, ["batch_size", 1, "height", "width"]),
                    helper.make_tensor_value_info("r1o", TensorProto.FLOAT, ["batch_size", 16, "height", "width"]),
                    helper.make_tensor_value_info("r2o", TensorProto.FLOAT, ["batch_size", 32, "height", "width"]),
                    helper.make_tensor_value_info("r3o", TensorProto.FLOAT, ["batch_size", 64, "height", "width"]),
                    helper.make_tensor_value_info("r4o", TensorProto.FLOAT, ["batch_size", 128, "height", "width"]),
                ],
            )
            model = helper.make_model(graph)
            onnx.save(model, model_path)

            with mock.patch("nvbroadcast.video.effects.CONFIG_DIR", config_dir):
                compat_path = _prepare_rvm_tensorrt_model(model_path)

            patched = onnx.load(str(compat_path))
            dims_by_name = {
                value_info.name: _tensor_dims(value_info)
                for value_info in list(patched.graph.input) + list(patched.graph.output)
            }

            self.assertEqual(dims_by_name["src"], ["batch_size", 3, "height", "width"])
            self.assertEqual(dims_by_name["r1i"], [1, 16, "r1i_h", "r1i_w"])
            self.assertEqual(dims_by_name["r4i"], [1, 128, "r4i_h", "r4i_w"])
            self.assertEqual(dims_by_name["r2o"], [1, 32, "r2o_h", "r2o_w"])
            self.assertEqual(dims_by_name["r4o"], [1, 128, "r4o_h", "r4o_w"])

    def test_prepare_rvm_tensorrt_model_staticizes_resize_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "dummy_rvm_trt.onnx"
            config_dir = root / "config"

            roi = np.array([], dtype=np.float32)
            scales = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
            rgb_sizes = np.array([1, 3, 480, 640], dtype=np.int64)
            alpha_sizes = np.array([1, 1, 480, 640], dtype=np.int64)

            nodes = [
                helper.make_node("Resize", ["src", "roi", "scales"], ["low"], name="Resize_3"),
                helper.make_node("Identity", ["r1i"], ["r1o"], name="R1"),
                helper.make_node("Identity", ["r2i"], ["r2o"], name="R2"),
                helper.make_node("Identity", ["r3i"], ["r3o"], name="R3"),
                helper.make_node("Identity", ["r4i"], ["r4o"], name="R4"),
                helper.make_node("Resize", ["low", "roi", "scales", "rgb_sizes"], ["fgr"], name="Resize_292"),
                helper.make_node("Resize", ["low", "roi", "scales", "alpha_sizes"], ["pha"], name="Resize_306"),
            ]
            graph = helper.make_graph(
                nodes=nodes,
                name="dummy-rvm-static",
                inputs=[
                    helper.make_tensor_value_info("src", TensorProto.FLOAT, ["batch_size", 3, "height", "width"]),
                    helper.make_tensor_value_info("r1i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("r2i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("r3i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("r4i", TensorProto.FLOAT, ["batch_size", "channels", "height", "width"]),
                    helper.make_tensor_value_info("downsample_ratio", TensorProto.FLOAT, [1]),
                ],
                outputs=[
                    helper.make_tensor_value_info("fgr", TensorProto.FLOAT, ["batch_size", 3, "height", "width"]),
                    helper.make_tensor_value_info("pha", TensorProto.FLOAT, ["batch_size", 1, "height", "width"]),
                    helper.make_tensor_value_info("r1o", TensorProto.FLOAT, ["batch_size", 16, "height", "width"]),
                    helper.make_tensor_value_info("r2o", TensorProto.FLOAT, ["batch_size", 32, "height", "width"]),
                    helper.make_tensor_value_info("r3o", TensorProto.FLOAT, ["batch_size", 64, "height", "width"]),
                    helper.make_tensor_value_info("r4o", TensorProto.FLOAT, ["batch_size", 128, "height", "width"]),
                ],
                initializer=[
                    numpy_helper.from_array(roi, name="roi"),
                    numpy_helper.from_array(scales, name="scales"),
                    numpy_helper.from_array(rgb_sizes, name="rgb_sizes"),
                    numpy_helper.from_array(alpha_sizes, name="alpha_sizes"),
                ],
            )
            model = helper.make_model(graph)
            onnx.save(model, model_path)

            recurrent_shapes = {
                "r1": (1, 16, 90, 120),
                "r2": (1, 32, 45, 60),
                "r3": (1, 64, 23, 30),
                "r4": (1, 128, 12, 15),
            }
            with mock.patch("nvbroadcast.video.effects.CONFIG_DIR", config_dir):
                compat_path = _prepare_rvm_tensorrt_model(
                    model_path,
                    infer_shape=(640, 480),
                    downsample_ratio=0.375,
                    recurrent_shapes=recurrent_shapes,
                )

            patched = onnx.load(str(compat_path))
            dims_by_name = {
                value_info.name: _tensor_dims(value_info)
                for value_info in list(patched.graph.input) + list(patched.graph.output)
            }
            self.assertEqual(dims_by_name["src"], [1, 3, 480, 640])
            self.assertEqual(dims_by_name["fgr"], [1, 3, 480, 640])
            self.assertEqual(dims_by_name["pha"], [1, 1, 480, 640])
            self.assertEqual(dims_by_name["r1i"], [1, 16, 90, 120])
            self.assertEqual(dims_by_name["r4o"], [1, 128, 12, 15])

            nodes_by_name = {node.name: node for node in patched.graph.node}
            self.assertEqual(nodes_by_name["Resize_3"].input[2], "nvb_static_resize3_scales")
            self.assertEqual(nodes_by_name["Resize_292"].input[3], "nvb_static_rgb_sizes")
            self.assertEqual(nodes_by_name["Resize_306"].input[3], "nvb_static_alpha_sizes")
            initializer_names = {init.name for init in patched.graph.initializer}
            self.assertIn("nvb_static_resize3_scales", initializer_names)
            self.assertIn("nvb_static_rgb_sizes", initializer_names)
            self.assertIn("nvb_static_alpha_sizes", initializer_names)

    def test_load_defers_trt_build_until_first_frame(self):
        backend = _RVMBackend(1)
        fake_session = mock.Mock()
        fake_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        base_model = Path("/tmp/rvm.onnx")

        with mock.patch("nvbroadcast.video.effects._download_model", return_value=base_model), \
             mock.patch("nvbroadcast.video.effects._create_session", return_value=fake_session) as create_session, \
             mock.patch("pathlib.Path.exists", return_value=True):
            msg = backend.load("quality", use_tensorrt=True)

        self.assertIn("first frame", msg)
        self.assertTrue(backend._trt_requested)
        self.assertFalse(backend._active_trt)
        self.assertEqual(backend._trt_model_path, base_model.with_name("rvm_trt.onnx"))
        create_session.assert_called_once_with(base_model, 1, use_tensorrt=False)

    def test_rvm_backend_promotes_trt_once_per_resolution(self):
        backend = _RVMBackend(1)
        backend._base_model_path = Path("/tmp/base.onnx")
        backend._trt_model_path = Path("/tmp/base_trt.onnx")
        backend._trt_cache_path = "/tmp/trt-cache"
        backend._trt_requested = True
        backend._downsample_ratio = np.array([0.375], dtype=np.float32)

        warmup_outputs = [
            np.zeros((1, 3, 480, 640), dtype=np.float32),
            np.zeros((1, 1, 480, 640), dtype=np.float32),
            np.zeros((1, 16, 90, 120), dtype=np.float32),
            np.zeros((1, 32, 45, 60), dtype=np.float32),
            np.zeros((1, 64, 23, 30), dtype=np.float32),
            np.zeros((1, 128, 12, 15), dtype=np.float32),
        ]
        cuda_session = mock.Mock()
        cuda_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        cuda_session.run.return_value = warmup_outputs
        trt_session = mock.Mock()
        trt_session.get_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        backend.session = cuda_session

        with mock.patch("nvbroadcast.video.effects._prepare_rvm_tensorrt_model", return_value=Path("/tmp/static_trt.onnx")) as prep_model, \
             mock.patch("nvbroadcast.video.effects._create_session", return_value=trt_session) as create_session, \
             mock.patch("nvbroadcast.video.effects._release_session") as release_session:
            src = np.zeros((1, 3, 480, 640), dtype=np.float32)
            backend._ensure_trt_state(src, 640, 480)
            backend._ensure_trt_state(src, 640, 480)

        prep_model.assert_called_once()
        create_session.assert_called_once_with(
            Path("/tmp/static_trt.onnx"),
            1,
            use_tensorrt=True,
            trt_cache_path="/tmp/trt-cache",
        )
        self.assertEqual(release_session.call_count, 1)
        self.assertIs(backend.session, trt_session)
        self.assertTrue(backend._active_trt)
        self.assertEqual(backend._trt_seed_shape, (640, 480))
        self.assertEqual(backend._trt_session_shape, (640, 480))
        self.assertEqual(backend._r1.shape, (1, 16, 90, 120))
        self.assertEqual(backend._r4.shape, (1, 128, 12, 15))

    def test_set_tensorrt_requested_false_reloads_cuda_session(self):
        backend = _RVMBackend(1)
        backend._base_model_path = Path("/tmp/base.onnx")
        backend._trt_requested = True
        backend._active_trt = True
        trt_session = mock.Mock()
        trt_session.get_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        cuda_session = mock.Mock()
        cuda_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        backend.session = trt_session

        with mock.patch("nvbroadcast.video.effects._create_session", return_value=cuda_session) as create_session, \
             mock.patch("nvbroadcast.video.effects._release_session") as release_session:
            backend.set_tensorrt_requested(False)

        self.assertFalse(backend._trt_requested)
        self.assertTrue(backend._trt_disabled)
        self.assertFalse(backend._active_trt)
        self.assertEqual(backend._trt_session_shape, None)
        self.assertIs(backend.session, cuda_session)
        create_session.assert_called_once_with(Path("/tmp/base.onnx"), 1, use_tensorrt=False)
        release_session.assert_called_once_with(trt_session)

    def test_sync_runtime_provider_state_marks_runtime_demote(self):
        backend = _RVMBackend(1)
        backend._active_trt = True
        backend.session = mock.Mock()
        backend.session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with mock.patch("builtins.print") as print_mock:
            backend._sync_runtime_provider_state()

        self.assertFalse(backend._active_trt)
        self.assertTrue(backend._trt_disabled)
        self.assertIsNone(backend._trt_session_shape)
        print_mock.assert_called_once()

    def test_infer_generic_runtime_error_does_not_reset_state(self):
        backend = _RVMBackend(1)
        backend.session = mock.Mock()
        backend.session.run.side_effect = RuntimeError("cuda kernel launch failed")
        backend._downsample_ratio = np.array([0.375], dtype=np.float32)
        backend._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        frame = np.zeros((360, 640, 4), dtype=np.uint8)

        with mock.patch.object(backend, "reset_state", wraps=backend.reset_state) as reset_state:
            with self.assertRaises(RuntimeError):
                backend.infer(frame, 640, 360)

        reset_state.assert_not_called()

    @mock.patch("nvbroadcast.video.effects._release_session")
    @mock.patch("nvbroadcast.video.effects._create_session")
    def test_infer_cuda_runtime_error_rebuilds_session_once(self, create_session, release_session):
        backend = _RVMBackend(1)
        original_session = mock.Mock()
        rebuilt_session = mock.Mock()
        outputs = [
            np.zeros((1, 3, 360, 640), dtype=np.float32),
            np.zeros((1, 1, 360, 640), dtype=np.float32),
            np.zeros((1, 16, 68, 120), dtype=np.float32),
            np.zeros((1, 32, 34, 60), dtype=np.float32),
            np.zeros((1, 64, 17, 30), dtype=np.float32),
            np.zeros((1, 128, 9, 15), dtype=np.float32),
        ]
        original_session.run.side_effect = RuntimeError(
            "CUDA failure 400: invalid resource handle"
        )
        rebuilt_session.run.return_value = outputs
        create_session.return_value = rebuilt_session

        backend.session = original_session
        backend._base_model_path = Path("/tmp/base.onnx")
        backend._downsample_ratio = np.array([0.375], dtype=np.float32)
        backend._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        frame = np.zeros((360, 640, 4), dtype=np.uint8)

        with mock.patch.object(backend, "reset_state", wraps=backend.reset_state) as reset_state:
            alpha = backend.infer(frame, 640, 360)

        self.assertEqual(alpha.shape, (360, 640))
        self.assertIs(backend.session, rebuilt_session)
        create_session.assert_called_once_with(Path("/tmp/base.onnx"), 1, use_tensorrt=False)
        release_session.assert_called_once_with(original_session)
        reset_state.assert_called_once()

    def test_infer_shape_transition_error_resets_once_and_recovers(self):
        backend = _RVMBackend(1)
        outputs = [
            np.zeros((1, 3, 360, 640), dtype=np.float32),
            np.zeros((1, 1, 360, 640), dtype=np.float32),
            np.zeros((1, 16, 68, 120), dtype=np.float32),
            np.zeros((1, 32, 34, 60), dtype=np.float32),
            np.zeros((1, 64, 17, 30), dtype=np.float32),
            np.zeros((1, 128, 9, 15), dtype=np.float32),
        ]
        backend.session = mock.Mock()
        backend.session.run.side_effect = [
            RuntimeError("Got invalid dimensions for input: r1i"),
            outputs,
        ]
        backend._downsample_ratio = np.array([0.375], dtype=np.float32)
        backend._r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        backend._r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        frame = np.zeros((360, 640, 4), dtype=np.uint8)

        with mock.patch.object(backend, "reset_state", wraps=backend.reset_state) as reset_state:
            alpha = backend.infer(frame, 640, 360)

        self.assertEqual(alpha.shape, (360, 640))
        reset_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()

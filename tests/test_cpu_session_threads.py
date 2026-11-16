import unittest
from unittest import mock

import nvbroadcast.video.effects as effects


class CpuSessionThreadTests(unittest.TestCase):
    def _session_options(self, **kwargs):
        with mock.patch.object(effects.ort, "InferenceSession") as session:
            effects._create_session("model.onnx", 0, **kwargs)
        return session.call_args.args[1]

    def test_cpu_session_caps_intra_op_pool(self):
        # The ORT default is one thread per core: ~7 cores at 30 fps for RVM on
        # a 32-thread host. CPU sessions must not inherit it.
        with mock.patch.dict(effects.os.environ, {}, clear=True), \
             mock.patch.object(effects.os, "cpu_count", return_value=32):
            opts = self._session_options(cpu_only=True)
        self.assertEqual(opts.intra_op_num_threads, 4)
        self.assertEqual(opts.inter_op_num_threads, 1)

    def test_gpu_session_keeps_small_pool(self):
        with mock.patch(
            "nvbroadcast.core.platform.get_onnx_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            opts = self._session_options()
        self.assertEqual(opts.intra_op_num_threads, 2)

    def test_env_override_is_clamped_to_cpu_count(self):
        with mock.patch.object(effects.os, "cpu_count", return_value=6):
            with mock.patch.dict(effects.os.environ, {"NVBROADCAST_CPU_THREADS": "12"}):
                self.assertEqual(effects._cpu_intra_op_threads(), 6)
            with mock.patch.dict(effects.os.environ, {"NVBROADCAST_CPU_THREADS": "3"}):
                self.assertEqual(effects._cpu_intra_op_threads(), 3)

    def test_invalid_env_override_falls_back_to_default(self):
        with mock.patch.object(effects.os, "cpu_count", return_value=32):
            for value in ("", "zero", "0", "-2"):
                with mock.patch.dict(effects.os.environ, {"NVBROADCAST_CPU_THREADS": value}):
                    self.assertEqual(effects._cpu_intra_op_threads(), 4, value)

    def test_default_never_exceeds_small_machines(self):
        with mock.patch.dict(effects.os.environ, {}, clear=True), \
             mock.patch.object(effects.os, "cpu_count", return_value=2):
            self.assertEqual(effects._cpu_intra_op_threads(), 2)


if __name__ == "__main__":
    unittest.main()

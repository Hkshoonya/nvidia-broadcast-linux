import ctypes
import unittest
from unittest import mock

from nvbroadcast.core import gpu


class _FakeCudaDriver:
    @staticmethod
    def cuInit(_flags):
        return 0

    @staticmethod
    def cuDeviceGetCount(count):
        count._obj.value = 1
        return 0

    @staticmethod
    def cuDriverGetVersion(version):
        version._obj.value = 13020
        return 0

    @staticmethod
    def cuDeviceGet(device, ordinal):
        device._obj.value = ordinal
        return 0

    @staticmethod
    def cuDeviceGetName(buffer, _size, _device):
        buffer.value = b"NVIDIA Sandbox GPU"
        return 0

    @staticmethod
    def cuDeviceTotalMem_v2(total_memory, _device):
        total_memory._obj.value = 12 * 1024 * 1024 * 1024
        return 0

    @staticmethod
    def cuDeviceComputeCapability(major, minor, _device):
        major._obj.value = 8
        minor._obj.value = 9
        return 0


class GpuDetectionTests(unittest.TestCase):
    def test_cuda_driver_fallback_detects_gpu_without_nvidia_smi(self):
        with mock.patch.object(
            gpu, "_detect_gpus_with_nvidia_smi", return_value=[]
        ), mock.patch.object(ctypes, "CDLL", return_value=_FakeCudaDriver()):
            detected = gpu.detect_gpus()

        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].index, 0)
        self.assertEqual(detected[0].name, "NVIDIA Sandbox GPU")
        self.assertEqual(detected[0].memory_total_mb, 12 * 1024)
        self.assertEqual(detected[0].compute_capability, "8.9")
        self.assertEqual(detected[0].driver_version, "CUDA driver API 13.2")

    def test_nvidia_smi_result_remains_preferred(self):
        expected = [gpu.GpuInfo(0, "Host GPU", 8192, "8.6", "600.1")]
        with mock.patch.object(
            gpu, "_detect_gpus_with_nvidia_smi", return_value=expected
        ), mock.patch.object(gpu, "_detect_gpus_with_cuda_driver") as fallback:
            detected = gpu.detect_gpus()

        self.assertEqual(detected, expected)
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()

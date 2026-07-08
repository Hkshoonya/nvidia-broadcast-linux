import unittest

import numpy as np

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        _HAS_CUDA = True
    except Exception:
        _HAS_CUDA = False
except ImportError:
    cp = None
    _HAS_CUDA = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class _StubEffects:
    """Effects stand-in: never eligible, so frames pass through."""
    last_output_mirrored = False

    def process_frame_gpu(self, *_a, **_k):
        return None

    def composite_only_gpu(self, *_a, **_k):
        return None


@unittest.skipUnless(_HAS_CUDA and _HAS_CV2, "requires CUDA + OpenCV")
class ColorKernelParityTests(unittest.TestCase):
    """GPU colorspace kernels must match OpenCV's BT.601 conversions
    within fixed-point rounding (max ±3/255)."""

    W, H = 128, 96

    @classmethod
    def setUpClass(cls):
        from nvbroadcast.video.gpu_frame_path import GpuFramePath
        cls.path = GpuFramePath.create(_StubEffects())
        assert cls.path is not None

    def _random_yuv_planes(self, seed=7):
        rng = np.random.default_rng(seed)
        # Stay inside studio swing so both implementations clamp identically.
        y = rng.integers(16, 236, (self.H, self.W), dtype=np.uint8)
        u = rng.integers(16, 241, (self.H // 2, self.W // 2), dtype=np.uint8)
        v = rng.integers(16, 241, (self.H // 2, self.W // 2), dtype=np.uint8)
        return y, u, v

    def _run_ingest(self, payload, fmt):
        self.path.configure(self.W, self.H, fmt)
        self.path.ingest(memoryview(np.ascontiguousarray(payload)))
        return self.path.download_bgra(source=True)

    def test_i420_to_bgra_matches_opencv(self):
        y, u, v = self._random_yuv_planes()
        i420 = np.concatenate([y.ravel(), u.ravel(), v.ravel()])
        got = self._run_ingest(i420, "I420")
        want = cv2.cvtColor(i420.reshape(self.H * 3 // 2, self.W),
                            cv2.COLOR_YUV2BGRA_I420)
        diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
        self.assertLessEqual(diff.max(), 3)
        self.assertLess(diff.mean(), 1.0)

    def test_nv12_to_bgra_matches_opencv(self):
        y, u, v = self._random_yuv_planes(seed=11)
        uv = np.empty((self.H // 2, self.W), dtype=np.uint8)
        uv[:, 0::2] = u
        uv[:, 1::2] = v
        nv12 = np.concatenate([y.ravel(), uv.ravel()])
        got = self._run_ingest(nv12, "NV12")
        want = cv2.cvtColor(nv12.reshape(self.H * 3 // 2, self.W),
                            cv2.COLOR_YUV2BGRA_NV12)
        diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
        self.assertLessEqual(diff.max(), 3)

    def test_yuy2_to_bgra_matches_opencv(self):
        y, u, v = self._random_yuv_planes(seed=13)
        # Pack YUYV: per pixel-pair [y0 u y1 v]; 4:2:2 chroma has full
        # vertical resolution, so repeat the 4:2:0-shaped planes rowwise.
        yuy2 = np.empty(self.H * self.W * 2, dtype=np.uint8)
        yuy2[0::4] = y[:, 0::2].ravel()
        yuy2[2::4] = y[:, 1::2].ravel()
        u_rows = u.repeat(2, axis=0)   # chroma rows are full height in 4:2:2
        v_rows = v.repeat(2, axis=0)
        yuy2[1::4] = u_rows.ravel()
        yuy2[3::4] = v_rows.ravel()
        got = self._run_ingest(yuy2, "YUY2")
        want = cv2.cvtColor(yuy2.reshape(self.H, self.W, 2),
                            cv2.COLOR_YUV2BGRA_YUY2)
        diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
        self.assertLessEqual(diff.max(), 3)

    def test_bgra_to_yuy2_roundtrip_on_smooth_image(self):
        # A smooth gradient loses almost nothing to 4:2:2 subsampling, so
        # encode->decode must come back close to the original.
        xx, yy = np.meshgrid(np.linspace(0, 255, self.W),
                             np.linspace(0, 255, self.H))
        bgra = np.stack([
            xx, yy, (xx + yy) / 2, np.full_like(xx, 255)],
            axis=-1).astype(np.uint8)
        self.path.configure(self.W, self.H, "BGRA")
        self.path.ingest(memoryview(np.ascontiguousarray(bgra)))
        out = np.empty(self.W * self.H * 2, dtype=np.uint8)
        self.path.download_yuy2_into(memoryview(out))
        decoded = cv2.cvtColor(out.reshape(self.H, self.W, 2),
                               cv2.COLOR_YUV2BGRA_YUY2)
        diff = np.abs(decoded[:, :, :3].astype(np.int16)
                      - bgra[:, :, :3].astype(np.int16))
        self.assertLessEqual(diff.max(), 6)
        self.assertLess(diff.mean(), 2.0)

    def test_passthrough_i420_to_yuy2_end_to_end(self):
        # Smooth planes: random chroma would legitimately diverge through
        # the 4:2:0 -> BGRA -> 4:2:2 re-subsampling this path performs.
        xx, yy = np.meshgrid(np.linspace(0, 1, self.W),
                             np.linspace(0, 1, self.H))
        y = (16 + 200 * (xx + yy) / 2).astype(np.uint8)
        cxx, cyy = np.meshgrid(np.linspace(0, 1, self.W // 2),
                               np.linspace(0, 1, self.H // 2))
        u = (16 + 200 * cxx).astype(np.uint8)
        v = (16 + 200 * cyy).astype(np.uint8)
        i420 = np.concatenate([y.ravel(), u.ravel(), v.ravel()])
        self.path.configure(self.W, self.H, "I420")
        self.path.ingest(memoryview(i420))
        # Stub effects always decline -> passthrough
        self.assertFalse(self.path.composite(mirror=False,
                                             inline_inference=False))
        out = np.empty(self.W * self.H * 2, dtype=np.uint8)
        self.path.download_yuy2_into(memoryview(out))
        got = cv2.cvtColor(out.reshape(self.H, self.W, 2),
                           cv2.COLOR_YUV2BGR_YUY2)
        want = cv2.cvtColor(i420.reshape(self.H * 3 // 2, self.W),
                            cv2.COLOR_YUV2BGR_I420)
        diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
        # Two fixed-point conversions + 4:2:0->4:2:2 resampling stack up.
        self.assertLess(diff.mean(), 3.0)

    def test_jpeg_gpu_decode_roundtrip(self):
        if not self.path.supports_jpeg:
            self.skipTest("nvidia-nvimgcodec not installed")
        xx, yy = np.meshgrid(np.linspace(0, 255, self.W),
                             np.linspace(0, 255, self.H))
        bgr = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.uint8)
        ok, jpeg = cv2.imencode(".jpg", bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertTrue(ok)
        self.path.configure(self.W, self.H, "JPEG")
        self.path.ingest_jpeg(memoryview(jpeg.tobytes()))
        got = self.path.download_bgra(source=True)
        diff = np.abs(got[:, :, :3].astype(np.int16) - bgr.astype(np.int16))
        # JPEG is lossy; a q95 smooth gradient should stay very close.
        self.assertLess(diff.mean(), 3.0)
        self.assertEqual(int(got[0, 0, 3]), 255)

    def test_mixed_mode_output_upload(self):
        rng = np.random.default_rng(23)
        bgra = rng.integers(0, 256, (self.H, self.W, 4), dtype=np.uint8)
        self.path.configure(self.W, self.H, "BGRA")
        self.path.ingest(memoryview(np.ascontiguousarray(bgra)))
        processed = np.ascontiguousarray(bgra[:, ::-1, :])  # fake CPU effect
        self.path.set_output_bgra(memoryview(processed))
        got = self.path.download_bgra()
        np.testing.assert_array_equal(got, processed)


if __name__ == "__main__":
    unittest.main()

# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""Device-resident per-frame path.

The legacy effects path moves every frame CPU->GPU->CPU as full BGRA
(3.7MB at 720p) three times per frame across the capture-convert leg,
the compositor, and the vcam convert leg. This module keeps the frame
on the GPU end to end instead:

    appsink (camera-native I420/NV12/YUY2, no convert element)
      -> pinned staging -> one H2D -> GPU colorspace kernel -> BGRA
      -> effects (device-resident) -> BGRA->YUY2 kernel
      -> one D2H straight into the outgoing Gst buffer -> v4l2sink

Host copies per frame drop to one small ingest copy (1.4MB I420) and
one YUY2 download (1.8MB), both through pinned memory so the PCIe
transfers are true DMA instead of pageable staging memcpys.

CPU consumers (preview, recording, the async alpha worker, CPU face
effects) request a BGRA download only on frames that need one.
"""

import numpy as np

_BT601_KERNELS = r"""
// BT.601 limited-range (studio swing) fixed-point conversions, matching
// GStreamer videoconvert/cudaconvert defaults for SD/HD camera JPEG.

extern "C" __global__ void i420_to_bgra(
        const unsigned char* __restrict__ src,
        unsigned char* __restrict__ dst,
        const int width, const int height) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = width * height;
    if (i >= total) return;
    const int x = i % width;
    const int y = i / width;
    const unsigned char* uplane = src + total;
    const unsigned char* vplane = uplane + (total >> 2);
    const int ci = (y >> 1) * (width >> 1) + (x >> 1);
    const int c = (int)src[i] - 16;
    const int d = (int)uplane[ci] - 128;
    const int e = (int)vplane[ci] - 128;
    int r = (298 * c + 409 * e + 128) >> 8;
    int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int b = (298 * c + 516 * d + 128) >> 8;
    unsigned char* o = dst + (size_t)i * 4;
    o[0] = (unsigned char)min(max(b, 0), 255);
    o[1] = (unsigned char)min(max(g, 0), 255);
    o[2] = (unsigned char)min(max(r, 0), 255);
    o[3] = 255;
}

extern "C" __global__ void nv12_to_bgra(
        const unsigned char* __restrict__ src,
        unsigned char* __restrict__ dst,
        const int width, const int height) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = width * height;
    if (i >= total) return;
    const int x = i % width;
    const int y = i / width;
    const unsigned char* uv = src + total;
    const int ci = ((y >> 1) * (width >> 1) + (x >> 1)) * 2;
    const int c = (int)src[i] - 16;
    const int d = (int)uv[ci] - 128;
    const int e = (int)uv[ci + 1] - 128;
    int r = (298 * c + 409 * e + 128) >> 8;
    int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int b = (298 * c + 516 * d + 128) >> 8;
    unsigned char* o = dst + (size_t)i * 4;
    o[0] = (unsigned char)min(max(b, 0), 255);
    o[1] = (unsigned char)min(max(g, 0), 255);
    o[2] = (unsigned char)min(max(r, 0), 255);
    o[3] = 255;
}

extern "C" __global__ void yuy2_to_bgra(
        const unsigned char* __restrict__ src,
        unsigned char* __restrict__ dst,
        const int width, const int height) {
    const int p = blockDim.x * blockIdx.x + threadIdx.x;   // pixel pair
    const int pairs = (width * height) >> 1;
    if (p >= pairs) return;
    const unsigned char* s = src + (size_t)p * 4;
    const int d = (int)s[1] - 128;
    const int e = (int)s[3] - 128;
    unsigned char* o = dst + (size_t)p * 8;
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        const int c = (int)s[k * 2] - 16;
        int r = (298 * c + 409 * e + 128) >> 8;
        int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        int b = (298 * c + 516 * d + 128) >> 8;
        o[k * 4 + 0] = (unsigned char)min(max(b, 0), 255);
        o[k * 4 + 1] = (unsigned char)min(max(g, 0), 255);
        o[k * 4 + 2] = (unsigned char)min(max(r, 0), 255);
        o[k * 4 + 3] = 255;
    }
}

extern "C" __global__ void rgb_to_bgra(
        const unsigned char* __restrict__ src,
        unsigned char* __restrict__ dst,
        const int total) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    const unsigned char* s = src + (size_t)i * 3;
    unsigned char* o = dst + (size_t)i * 4;
    o[0] = s[2];
    o[1] = s[1];
    o[2] = s[0];
    o[3] = 255;
}

extern "C" __global__ void bgra_to_yuy2(
        const unsigned char* __restrict__ src,
        unsigned char* __restrict__ dst,
        const int width, const int height) {
    const int p = blockDim.x * blockIdx.x + threadIdx.x;   // pixel pair
    const int pairs = (width * height) >> 1;
    if (p >= pairs) return;
    const unsigned char* s = src + (size_t)p * 8;
    const int b0 = s[0], g0 = s[1], r0 = s[2];
    const int b1 = s[4], g1 = s[5], r1 = s[6];
    const int y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16;
    const int y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
    // 4:2:2 chroma from the averaged pixel pair.
    const int ra = (r0 + r1 + 1) >> 1;
    const int ga = (g0 + g1 + 1) >> 1;
    const int ba = (b0 + b1 + 1) >> 1;
    const int u = ((-38 * ra - 74 * ga + 112 * ba + 128) >> 8) + 128;
    const int v = ((112 * ra - 94 * ga - 18 * ba + 128) >> 8) + 128;
    unsigned char* o = dst + (size_t)p * 4;
    o[0] = (unsigned char)min(max(y0, 0), 255);
    o[1] = (unsigned char)min(max(u, 0), 255);
    o[2] = (unsigned char)min(max(y1, 0), 255);
    o[3] = (unsigned char)min(max(v, 0), 255);
}
"""

# Bytes per pixel of each supported appsink input format. JPEG frames are
# variable-size and go straight to the GPU decoder without pinned staging.
_INPUT_FORMATS = {
    "I420": 1.5,
    "NV12": 1.5,
    "YUY2": 2.0,
    "BGRA": 4.0,
    "JPEG": 0.0,
}


class GpuFramePath:
    """Owns all pinned/GPU buffers for the device-resident frame path.

    Buffer ownership: everything allocated here is reused frame to frame
    and never handed to GStreamer — downloads copy into caller-provided
    mapped Gst memory or into small pooled host arrays.
    """

    def __init__(self, cp, effects, gpu_index: int = 0):
        self._cp = cp
        self._effects = effects
        self._gpu_index = gpu_index
        self._module = cp.RawModule(code=_BT601_KERNELS)
        self._kernels = {
            "I420": self._module.get_function("i420_to_bgra"),
            "NV12": self._module.get_function("nv12_to_bgra"),
            "YUY2": self._module.get_function("yuy2_to_bgra"),
            "to_yuy2": self._module.get_function("bgra_to_yuy2"),
            "rgb": self._module.get_function("rgb_to_bgra"),
        }
        # Optional GPU JPEG decode (nvidia-nvimgcodec-cu12 + nvjpeg wheel).
        # When present the capture leg skips jpegdec entirely: only the
        # ~100-300KB compressed frame crosses into Python. The Decoder
        # constructs even without libnvjpeg, so probe an actual decode.
        self._jpeg_decoder = None
        try:
            import cv2
            from nvidia import nvimgcodec
            decoder = nvimgcodec.Decoder()
            probe = np.zeros((16, 16, 3), dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", probe)
            image = decoder.decode(jpeg.tobytes()) if ok else None
            if image is not None and cp.asarray(image).shape == (16, 16, 3):
                self._jpeg_decoder = decoder
        except Exception:
            self._jpeg_decoder = None
        self._width = 0
        self._height = 0
        self._in_format = ""
        self._in_nbytes = 0
        self._pinned_in = None       # pinned host staging for ingest
        self._pinned_in_np = None
        self._in_gpu = None          # raw camera-format bytes on device
        self._src_bgra_gpu = None    # converted source frame
        self._out_bgra_gpu = None    # post-effects output (may alias src)
        self._yuy2_gpu = None
        self._pinned_bgra = None     # pinned host landing zone for downloads
        self._pinned_bgra_np = None
        self._pinned_up = None       # pinned host staging for mixed-mode upload
        self._pinned_up_np = None
        # 2-deep pool so a preview/recording consumer on another thread never
        # aliases the landing buffer the next frame overwrites.
        self._host_pool = [None, None]
        self._host_pool_idx = 0
        self._jpeg_image = None

    @classmethod
    def create(cls, effects, gpu_index: int = 0):
        """Build the path, compiling kernels; None when CUDA is unusable."""
        try:
            import cupy as cp
            with cp.cuda.Device(gpu_index):
                path = cls(cp, effects, gpu_index)
                # Force kernel compilation now so failures demote at build
                # time instead of mid-stream.
                path.configure(64, 64, "I420")
                probe = np.zeros(int(64 * 64 * 1.5), dtype=np.uint8)
                path.ingest(memoryview(probe))
                out = np.empty(64 * 64 * 2, dtype=np.uint8)
                path.download_yuy2_into(memoryview(out))
                path._width = 0  # force reconfigure on first real frame
            return path
        except Exception as e:
            print(f"[NV Broadcast] GPU frame path unavailable: {e}", flush=True)
            return None

    @staticmethod
    def supports_format(fmt: str) -> bool:
        return fmt in _INPUT_FORMATS

    def configure(self, width: int, height: int, in_format: str) -> bool:
        """(Re)allocate buffers for a new negotiated size/format."""
        if in_format not in _INPUT_FORMATS:
            return False
        if (width, height, in_format) == (self._width, self._height,
                                          self._in_format):
            return True
        if width % 2 or height % 2:
            return False   # 4:2:x kernels assume even dimensions
        cp = self._cp
        with cp.cuda.Device(self._gpu_index):
            self._in_nbytes = int(width * height * _INPUT_FORMATS[in_format])
            if self._in_nbytes:
                self._pinned_in = cp.cuda.alloc_pinned_memory(self._in_nbytes)
                self._pinned_in_np = np.frombuffer(
                    self._pinned_in, dtype=np.uint8, count=self._in_nbytes)
                self._in_gpu = cp.empty(self._in_nbytes, dtype=cp.uint8)
            else:
                self._pinned_in = None
                self._pinned_in_np = None
                self._in_gpu = None
            self._src_bgra_gpu = cp.empty((height, width, 4), dtype=cp.uint8)
            self._out_bgra_gpu = self._src_bgra_gpu
            self._yuy2_gpu = cp.empty(width * height * 2, dtype=cp.uint8)
            self._pinned_yuy2 = cp.cuda.alloc_pinned_memory(width * height * 2)
            self._pinned_yuy2_np = np.frombuffer(
                self._pinned_yuy2, dtype=np.uint8, count=width * height * 2)
            bgra_nbytes = width * height * 4
            self._pinned_bgra = cp.cuda.alloc_pinned_memory(bgra_nbytes)
            self._pinned_bgra_np = np.frombuffer(
                self._pinned_bgra, dtype=np.uint8,
                count=bgra_nbytes).reshape(height, width, 4)
            self._pinned_up = cp.cuda.alloc_pinned_memory(bgra_nbytes)
            self._pinned_up_np = np.frombuffer(
                self._pinned_up, dtype=np.uint8, count=bgra_nbytes)
            self._host_pool = [None, None]
        self._width = width
        self._height = height
        self._in_format = in_format
        return True

    # ─── Per-frame stages ────────────────────────────────────────────────

    def ingest(self, data) -> None:
        """Copy one camera-format frame to the GPU and convert to BGRA.

        ``data`` is any buffer over the mapped appsink bytes; it is fully
        consumed before return, so the caller can unmap immediately after.
        """
        cp = self._cp
        src = np.frombuffer(data, dtype=np.uint8, count=self._in_nbytes)
        np.copyto(self._pinned_in_np, src)
        with cp.cuda.Device(self._gpu_index):
            self._in_gpu.data.copy_from_host(
                self._pinned_in_np.ctypes.data, self._in_nbytes)
            if self._in_format == "BGRA":
                self._src_bgra_gpu = self._in_gpu.reshape(
                    self._height, self._width, 4)
            else:
                kernel = self._kernels[self._in_format]
                if self._in_format == "YUY2":
                    work = (self._width * self._height) // 2
                else:
                    work = self._width * self._height
                threads = 256
                blocks = (work + threads - 1) // threads
                kernel((blocks,), (threads,), (
                    self._in_gpu, self._src_bgra_gpu,
                    np.int32(self._width), np.int32(self._height)))
        self._out_bgra_gpu = self._src_bgra_gpu

    @property
    def supports_jpeg(self) -> bool:
        return self._jpeg_decoder is not None

    def ingest_jpeg(self, data) -> None:
        """Decode one MJPEG frame on the GPU and convert to BGRA.

        Raises on decode failure or size mismatch; the pipeline's error
        counter demotes the path after repeated failures.
        """
        cp = self._cp
        with cp.cuda.Device(self._gpu_index):
            image = self._jpeg_decoder.decode(bytes(data))
            if image is None:
                raise RuntimeError("nvimgcodec returned no image")
            rgb = cp.asarray(image)
            if rgb.shape != (self._height, self._width, 3):
                raise RuntimeError(f"decoded JPEG is {rgb.shape}, expected "
                                   f"{(self._height, self._width, 3)}")
            # Keep the decoder image alive until this frame's synchronous
            # download completes — cp.asarray is a zero-copy view of it.
            self._jpeg_image = image
            total = self._width * self._height
            threads = 256
            blocks = (total + threads - 1) // threads
            self._kernels["rgb"]((blocks,), (threads,), (
                cp.ascontiguousarray(rgb), self._src_bgra_gpu,
                np.int32(total)))
        self._out_bgra_gpu = self._src_bgra_gpu

    def composite(self, mirror: bool, inline_inference: bool) -> bool:
        """Run device-resident effects on the ingested frame.

        Returns False when effects couldn't run on-device this frame (matte
        not ready, mode not eligible) — the source frame then passes through
        unprocessed, matching the legacy callback's behavior.
        """
        if inline_inference:
            out = self._effects.process_frame_gpu(
                self._src_bgra_gpu, self._width, self._height, mirror=mirror)
        else:
            out = self._effects.composite_only_gpu(
                self._src_bgra_gpu, self._width, self._height, mirror=mirror)
        if out is None:
            # Pass through — but keep the mirror contract the legacy CPU
            # callback honored even while the matte wasn't ready yet.
            if mirror:
                cp = self._cp
                with cp.cuda.Device(self._gpu_index):
                    self._out_bgra_gpu = cp.ascontiguousarray(
                        self._src_bgra_gpu[:, ::-1, :])
            else:
                self._out_bgra_gpu = self._src_bgra_gpu
            return False
        self._out_bgra_gpu = out
        return True

    def set_output_bgra(self, data) -> None:
        """Mixed mode: adopt a CPU-processed BGRA frame as the output."""
        cp = self._cp
        nbytes = self._width * self._height * 4
        src = np.frombuffer(data, dtype=np.uint8, count=nbytes)
        np.copyto(self._pinned_up_np, src)
        with cp.cuda.Device(self._gpu_index):
            if self._out_bgra_gpu is self._src_bgra_gpu:
                self._out_bgra_gpu = cp.empty_like(self._src_bgra_gpu)
            self._out_bgra_gpu.data.copy_from_host(
                self._pinned_up_np.ctypes.data, nbytes)

    def download_yuy2(self) -> np.ndarray:
        """Convert the output to YUY2 and download it to pinned host memory.

        Returns a view of the pinned staging buffer, valid until the next
        call. Synchronous: on return the GPU work for this frame is
        complete and every staging buffer is reusable. (PyGObject exposes
        WRITE-mapped Gst buffers as a bytes COPY, so downloading directly
        into a mapped buffer silently produces black frames — the caller
        must copy this result in via Gst.Buffer.fill instead.)
        """
        cp = self._cp
        nbytes = self._width * self._height * 2
        with cp.cuda.Device(self._gpu_index):
            work = (self._width * self._height) // 2
            threads = 256
            blocks = (work + threads - 1) // threads
            self._kernels["to_yuy2"]((blocks,), (threads,), (
                cp.ascontiguousarray(self._out_bgra_gpu), self._yuy2_gpu,
                np.int32(self._width), np.int32(self._height)))
            self._yuy2_gpu.data.copy_to_host(
                self._pinned_yuy2_np.ctypes.data, nbytes)
        return self._pinned_yuy2_np

    def download_yuy2_into(self, dst) -> None:
        """Test helper: download YUY2 into a writable numpy-backed buffer."""
        np.copyto(np.frombuffer(dst, dtype=np.uint8,
                                count=self._width * self._height * 2),
                  self.download_yuy2())

    def download_bgra(self, source: bool = False) -> np.ndarray:
        """Download output (or source) BGRA into a pooled host array.

        The two-deep pool means the returned array stays valid while the
        next frame downloads; consumers that hold frames longer must copy.
        """
        cp = self._cp
        gpu = self._src_bgra_gpu if source else self._out_bgra_gpu
        with cp.cuda.Device(self._gpu_index):
            contiguous = cp.ascontiguousarray(gpu)
            contiguous.data.copy_to_host(
                self._pinned_bgra_np.ctypes.data,
                self._pinned_bgra_np.nbytes)
        slot = self._host_pool_idx
        self._host_pool_idx = (slot + 1) % len(self._host_pool)
        buf = self._host_pool[slot]
        if buf is None or buf.shape != self._pinned_bgra_np.shape:
            buf = np.empty_like(self._pinned_bgra_np)
            self._host_pool[slot] = buf
        np.copyto(buf, self._pinned_bgra_np)
        return buf

    @property
    def output_mirrored(self) -> bool:
        return bool(getattr(self._effects, "last_output_mirrored", False))

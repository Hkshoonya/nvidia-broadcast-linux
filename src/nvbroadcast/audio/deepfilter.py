# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
# Original author: doczeus | AI Powered
#
"""DeepFilterNet3 neural noise removal (streaming ONNX).

A modern full-band (48kHz) speech-enhancement network, the same class of
model behind NVIDIA Broadcast's noise removal — far stronger than RNNoise
on keyboards, fans, and non-stationary noise.

The model is the torchDF streaming export of DeepFilterNet3 (MIT/Apache-2.0):
the whole feature pipeline (STFT, ERB, deep filtering, ISTFT) is inside the
graph, so inference is one session.run per 480-sample (10ms) hop with an
explicit recurrent state tensor. Output is delayed by exactly one hop.

Provider choice is measured, not assumed: the network is tiny, so CPU
inference costs ~0.6ms per 10ms hop while CUDA pays ~1.2ms in launch
overhead plus a dedicated context in the audio helper process. CPU is the
default; set NVBROADCAST_DFN_PROVIDER=cuda to force the GPU.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"

MODEL_FILENAME = "deepfilternet3_streaming.onnx"
# Pinned to a commit so the artifact is immutable.
MODEL_URL = (
    "https://raw.githubusercontent.com/yuyun2000/SpeechDenoiser/"
    "9a1f069f562089c2f56008a8ecd897fa981fbc50/48k/denoiser_model.onnx"
)
MODEL_SHA256 = "fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9"

HOP_SIZE = 480          # 10ms at 48kHz
STATE_SIZE = 45304
SAMPLE_RATE = 48000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


DOWNLOAD_TIMEOUT_S = 30       # Per connect/read; a dead server must not hang init
DOWNLOAD_DEADLINE_S = 600     # Whole transfer; a trickling server must not either


def _download_model() -> Path:
    """Download and verify the streaming model if not already present.

    Every failure mode raises (and the caller falls back to RNNoise):
    timeouts are bounded both per read and for the whole transfer, and a
    checksum mismatch deletes the partial file.
    """
    model_path = _MODELS_DIR / MODEL_FILENAME
    if model_path.exists() and _sha256(model_path) == MODEL_SHA256:
        return model_path
    import time
    import urllib.request
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".part")
    print(f"[NV Broadcast] Downloading {MODEL_FILENAME}...", flush=True)
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    try:
        with urllib.request.urlopen(MODEL_URL,
                                    timeout=DOWNLOAD_TIMEOUT_S) as response, \
                open(tmp_path, "wb") as out:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"model download exceeded {DOWNLOAD_DEADLINE_S}s")
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    if _sha256(tmp_path) != MODEL_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("DeepFilterNet model checksum mismatch")
    tmp_path.replace(model_path)
    print(f"[NV Broadcast] Downloaded {MODEL_FILENAME}", flush=True)
    return model_path


def _prepare_cuda_compatible(model_path: Path) -> Path:
    """Unfuse FusedConv(Sigmoid) so the graph loads on the CUDA EP.

    The export contains com.microsoft.FusedConv nodes; the CUDA EP only
    implements the Relu activation, so the single Sigmoid-fused conv makes
    session creation fail on CUDA. Splitting it into Conv + Sigmoid is
    bit-exact (verified) and also loads fine on CPU, so the patched file is
    used everywhere.

    The patched file is never trusted just because it exists: a sidecar
    records the source model hash it was generated from and its own hash,
    and any mismatch (corruption, tampering, or a new source model)
    regenerates it from the already checksum-verified source.
    """
    patched_path = model_path.with_name(model_path.stem + "_unfused.onnx")
    stamp_path = patched_path.with_name(patched_path.name + ".sha256")
    if patched_path.exists():
        try:
            recorded_source, recorded_patched = stamp_path.read_text().split()
            if (recorded_source == MODEL_SHA256
                    and _sha256(patched_path) == recorded_patched):
                return patched_path
        except (OSError, ValueError):
            pass
        print("[NV Broadcast] Cached DeepFilterNet graph failed verification; "
              "regenerating", flush=True)
    import onnx
    from onnx import helper

    model = onnx.load(str(model_path))
    new_nodes = []
    patched = 0
    for node in model.graph.node:
        activation = None
        for attr in node.attribute:
            if attr.name == "activation" and attr.type == onnx.AttributeProto.STRING:
                activation = attr.s.decode()
        if node.op_type == "FusedConv" and activation == "Sigmoid":
            conv_out = node.output[0] + "_prefused"
            conv_attrs = [a for a in node.attribute
                          if a.name not in ("activation", "activation_params")]
            conv = helper.make_node("Conv", list(node.input), [conv_out],
                                    name=node.name + "_unfused")
            conv.attribute.extend(conv_attrs)
            sigmoid = helper.make_node("Sigmoid", [conv_out], list(node.output),
                                       name=node.name + "_sigmoid")
            new_nodes.extend([conv, sigmoid])
            patched += 1
        else:
            new_nodes.append(node)
    if not patched:
        return model_path
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    # Write-then-rename so a crash mid-save cannot leave a truncated graph;
    # the stamp is written last, so a stale stamp only ever causes a
    # harmless regeneration, never acceptance of a bad file.
    tmp_path = patched_path.with_suffix(".part")
    onnx.save(model, str(tmp_path))
    tmp_path.replace(patched_path)
    stamp_path.write_text(f"{MODEL_SHA256} {_sha256(patched_path)}\n")
    return patched_path


class DeepFilterDenoiser:
    """Streaming DeepFilterNet3 denoiser with RNNoise-compatible framing.

    process_chunk accepts arbitrary block sizes (the live pipeline delivers
    ~1024 samples) and internally re-frames to 480-sample hops with a ring
    buffer, so every sample is denoised — no pass-through remainders.
    Total added latency: one 10ms model hop plus up to 10ms of framing.
    """

    def __init__(self):
        self.session = None
        self._state = None
        self._atten = np.zeros(1, dtype=np.float32)
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._out_buf = np.zeros(0, dtype=np.float32)
        self._prev_dry = np.zeros(HOP_SIZE, dtype=np.float32)
        self._initialized = False

    @property
    def available(self) -> bool:
        return self._initialized

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            model_path = _prepare_cuda_compatible(_download_model())

            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3
            # Single thread: the model runs ~0.6ms/hop; a second intra-op
            # worker costs a cross-thread wake 100x/sec for no latency win.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            try:
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
            except Exception:
                pass

            # CPU is measured faster than CUDA for this model size (0.6ms
            # vs 1.2ms per hop) and avoids a CUDA context in the helper.
            if os.getenv("NVBROADCAST_DFN_PROVIDER", "").lower() == "cuda":
                providers = [("CUDAExecutionProvider", {"device_id": 0}),
                             "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(model_path), opts,
                                                providers=providers)
            self.reset()
            self._initialized = True
            active = self.session.get_providers()[0]
            device = "GPU" if "CUDA" in active else "CPU"
            print(f"[NVIDIA Broadcast] Audio denoiser initialized "
                  f"(DeepFilterNet3 on {device})", flush=True)
            return True
        except Exception as e:
            print(f"[NVIDIA Broadcast] DeepFilterNet init failed: {e}", flush=True)
            self.session = None
            return False

    def reset(self):
        """Reset streaming state (call on stream start/discontinuity)."""
        self._state = np.zeros(STATE_SIZE, dtype=np.float32)
        # Flat pre-sized buffers instead of per-hop np.concatenate churn.
        # 32 hops (320ms) comfortably covers any real capture block size;
        # process_chunk grows them if a larger block ever arrives.
        cap = HOP_SIZE * 32
        self._in_buf = np.empty(cap, dtype=np.float32)
        self._in_len = 0
        self._out_buf = np.empty(cap, dtype=np.float32)
        # Prime one hop of silence: with this fixed 10ms of extra latency
        # the output buffer can never run dry mid-stream, because the
        # deficit is bounded by (samples_in mod HOP_SIZE) < HOP_SIZE.
        self._out_buf[:HOP_SIZE] = 0.0
        self._out_len = HOP_SIZE
        self._prev_dry = np.zeros(HOP_SIZE, dtype=np.float32)
        self._frame_scratch = np.empty(HOP_SIZE, dtype=np.float32)

    @staticmethod
    def _ensure_capacity(buf: np.ndarray, used: int, extra: int) -> np.ndarray:
        needed = used + extra
        if needed <= len(buf):
            return buf
        grown = np.empty(max(needed, len(buf) * 2), dtype=np.float32)
        grown[:used] = buf[:used]
        return grown

    def process_chunk(self, audio_data: np.ndarray, sample_rate: int = 48000,
                      intensity: float = 1.0) -> np.ndarray:
        """Denoise a float32 mono block, preserving its length.

        The enhanced signal lags the dry signal by one hop, so the
        intensity blend uses a matching one-hop-delayed dry frame.
        Returns an array owned by the caller (never a view of state).
        """
        if not self._initialized or sample_rate != SAMPLE_RATE:
            return audio_data

        n = len(audio_data)
        self._in_buf = self._ensure_capacity(self._in_buf, self._in_len, n)
        self._in_buf[self._in_len:self._in_len + n] = audio_data
        self._in_len += n

        pos = 0
        while self._in_len - pos >= HOP_SIZE:
            frame = self._frame_scratch
            np.copyto(frame, self._in_buf[pos:pos + HOP_SIZE])
            pos += HOP_SIZE
            enhanced, self._state, _lsnr = self.session.run(
                None, {"input_frame": frame, "states": self._state,
                       "atten_lim_db": self._atten})
            if intensity < 1.0:
                np.multiply(enhanced, intensity, out=enhanced)
                enhanced += (1.0 - intensity) * self._prev_dry
            np.copyto(self._prev_dry, frame)
            self._out_buf = self._ensure_capacity(
                self._out_buf, self._out_len, HOP_SIZE)
            self._out_buf[self._out_len:self._out_len + HOP_SIZE] = enhanced
            self._out_len += HOP_SIZE
        if pos:
            remaining = self._in_len - pos
            if remaining:
                self._in_buf[:remaining] = self._in_buf[pos:self._in_len]
            self._in_len = remaining

        if self._out_len < n:
            # Startup only: pad the first block with leading silence while
            # the first hops are still buffering.
            deficit = n - self._out_len
            self._out_buf = self._ensure_capacity(
                self._out_buf, self._out_len, deficit)
            self._out_buf[deficit:deficit + self._out_len] = \
                self._out_buf[:self._out_len]
            self._out_buf[:deficit] = 0.0
            self._out_len = n
        out = self._out_buf[:n].copy()
        remaining = self._out_len - n
        if remaining:
            self._out_buf[:remaining] = self._out_buf[n:self._out_len]
        self._out_len = remaining
        return out

    def cleanup(self):
        self.session = None
        self._initialized = False

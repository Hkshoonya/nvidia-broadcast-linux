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


def _download_model() -> Path:
    """Download and verify the streaming model if not already present."""
    model_path = _MODELS_DIR / MODEL_FILENAME
    if model_path.exists() and _sha256(model_path) == MODEL_SHA256:
        return model_path
    import urllib.request
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".part")
    print(f"[NV Broadcast] Downloading {MODEL_FILENAME}...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, str(tmp_path))
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
    """
    patched_path = model_path.with_name(model_path.stem + "_unfused.onnx")
    if patched_path.exists():
        return patched_path
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
    onnx.save(model, str(patched_path))
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
            opts.intra_op_num_threads = 2
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
        self._in_buf = np.zeros(0, dtype=np.float32)
        # Prime one hop of silence: with this fixed 10ms of extra latency
        # the output buffer can never run dry mid-stream, because the
        # deficit is bounded by (samples_in mod HOP_SIZE) < HOP_SIZE.
        self._out_buf = np.zeros(HOP_SIZE, dtype=np.float32)
        self._prev_dry = np.zeros(HOP_SIZE, dtype=np.float32)

    def process_chunk(self, audio_data: np.ndarray, sample_rate: int = 48000,
                      intensity: float = 1.0) -> np.ndarray:
        """Denoise a float32 mono block, preserving its length.

        The enhanced signal lags the dry signal by one hop, so the
        intensity blend uses a matching one-hop-delayed dry frame.
        """
        if not self._initialized or sample_rate != SAMPLE_RATE:
            return audio_data

        self._in_buf = np.concatenate((self._in_buf, audio_data.astype(np.float32)))

        while len(self._in_buf) >= HOP_SIZE:
            frame = self._in_buf[:HOP_SIZE]
            self._in_buf = self._in_buf[HOP_SIZE:]
            enhanced, self._state, _lsnr = self.session.run(
                None, {"input_frame": frame, "states": self._state,
                       "atten_lim_db": self._atten})
            if intensity < 1.0:
                enhanced = (intensity * enhanced
                            + (1.0 - intensity) * self._prev_dry)
            self._prev_dry = frame
            self._out_buf = np.concatenate((self._out_buf, enhanced))

        n = len(audio_data)
        if len(self._out_buf) < n:
            # Startup only: pad the first block with leading silence while
            # the first hops are still buffering.
            pad = np.zeros(n - len(self._out_buf), dtype=np.float32)
            self._out_buf = np.concatenate((pad, self._out_buf))
        out = self._out_buf[:n]
        self._out_buf = self._out_buf[n:]
        return out

    def cleanup(self):
        self.session = None
        self._initialized = False

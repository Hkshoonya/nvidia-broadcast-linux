"""Exercise the actual subprocess protocol without requiring NVIDIA hardware."""
from unittest import mock

import numpy as np
import pytest

from nvbroadcast.audio.hush import HushDenoiser
from nvbroadcast.audio.effects import AudioEffects


@pytest.fixture
def worker(tmp_path, monkeypatch):
    script = tmp_path / "hush-worker"
    script.write_text('''#!/usr/bin/env python3
import struct, sys
r, w = sys.stdin.buffer, sys.stdout.buffer
w.write(b'HSH1' + struct.pack('<I', 480)); w.flush()
while True:
    data = r.read(1924)
    if not data: break
    strength = struct.unpack('<f', data[:4])[0]
    samples = struct.unpack('<480f', data[4:])
    w.write(struct.pack('<480f', *(x * (1 - strength / 2) for x in samples)))
    w.flush()
''')
    script.chmod(0o755)
    model = tmp_path / "model.trtpkg"
    model.touch()
    monkeypatch.setenv("NVBROADCAST_HUSH_WORKER", str(script))
    monkeypatch.setenv("NVAFX_MODEL", str(model))
    denoiser = HushDenoiser()
    assert denoiser.initialize()
    yield denoiser
    denoiser.cleanup()


def test_stream_reframes_and_keeps_one_frame_latency(worker):
    rng = np.random.default_rng(12)
    sizes = [128, 960, 17, 480, 333, 4800]
    audio = rng.uniform(-1, 1, sum(sizes)).astype(np.float32)
    results, offset = [], 0
    for size in sizes:
        results.append(worker.process_chunk(audio[offset:offset + size], intensity=0.6))
        assert len(results[-1]) == size
        offset += size
    expected = np.concatenate((np.zeros(480), audio * 0.7))[:len(audio)]
    np.testing.assert_allclose(np.concatenate(results), expected, atol=1e-7)


def test_dead_worker_and_wrong_sample_rate_are_rejected(worker):
    with pytest.raises(ValueError):
        worker.process_chunk(np.zeros(480), sample_rate=44100)
    worker._process.kill()
    worker._process.wait()
    with pytest.raises(RuntimeError):
        worker.process_chunk(np.zeros(480))


def test_cleanup_reaps_worker(worker):
    process = worker._process
    worker.cleanup()
    assert process.poll() is not None
    assert not worker.available
    worker.cleanup()


def test_worker_timeout_is_bounded(worker):
    worker._process.send_signal(__import__('signal').SIGSTOP)
    try:
        with pytest.raises(TimeoutError):
            worker.process_chunk(np.zeros(480))
    finally:
        worker._process.send_signal(__import__('signal').SIGCONT)


def test_runtime_failure_demotes_and_preserves_audio():
    effects = AudioEffects()
    effects._engine_pref = "hush"
    effects._enabled = effects._initialized = True
    effects._hush = mock.Mock(available=True)
    effects._hush.process_chunk.side_effect = RuntimeError("GPU lost")
    hush = effects._hush
    audio = np.ones(480, dtype=np.float32)
    np.testing.assert_array_equal(effects.process_chunk(audio), audio)
    hush.cleanup.assert_called_once()
    assert effects.active_engine == "rnnoise"


def test_selecting_another_engine_releases_hush():
    effects = AudioEffects()
    effects._engine_pref = "hush"
    effects._hush = mock.Mock()
    hush = effects._hush
    effects.engine = "rnnoise"
    hush.cleanup.assert_called_once()
    assert effects._hush is None


def test_missing_worker_fails_without_starting_process(monkeypatch):
    monkeypatch.setenv("NVBROADCAST_HUSH_WORKER", "/does/not/exist/hush-worker")
    worker = HushDenoiser()
    assert not worker.initialize()
    assert not worker.available


def test_hush_preference_and_fallback_initialization():
    effects = AudioEffects()
    effects.engine = "hush"
    with mock.patch("nvbroadcast.audio.hush.HushDenoiser") as worker, \
            mock.patch.object(effects, "_init_deepfilter") as fallback:
        worker.return_value.initialize.return_value = False
        effects.initialize()
    fallback.assert_called_once()
    assert effects.engine == "hush"
    effects.cleanup()


def test_self_configured_worker_does_not_require_parent_model_env(worker, monkeypatch):
    worker.cleanup()
    monkeypatch.delenv("NVAFX_MODEL")
    assert worker.initialize()

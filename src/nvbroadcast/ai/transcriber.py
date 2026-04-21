"""Meeting transcription -- local Whisper-based speech-to-text.

Uses OpenAI Whisper (open-source) running locally.
No data leaves the machine. Transcribes audio chunks in real-time
and saves complete transcript at meeting end.
"""

import os
import re
import time
import threading
import warnings
import wave
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np


_WORKER_BACKEND = ""
_WORKER_MODEL = None


def _coerce_language(value: str | None) -> str | None:
    """Treat blank/auto language values as model auto-detect."""
    if value is None:
        return None
    value = value.strip().lower()
    if not value or value in {"auto", "detect"}:
        return None
    return value


def _init_transcriber_worker(model_size: str, device: str):
    """Load Whisper once in a dedicated worker process."""
    global _WORKER_BACKEND, _WORKER_MODEL

    warnings.filterwarnings(
        "ignore", message="Performing inference on CPU when CUDA is available"
    )

    try:
        from faster_whisper import WhisperModel

        compute_type = "int8"
        if device == "cuda":
            compute_type = "float16"
        elif device == "mps":
            compute_type = "float32"

        _WORKER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
        _WORKER_BACKEND = "faster-whisper"
        return
    except Exception:
        pass

    import whisper

    _WORKER_MODEL = whisper.load_model(model_size, device=device)
    _WORKER_BACKEND = "openai-whisper"


def _worker_ping() -> str:
    return _WORKER_BACKEND or "unknown"


def _decode_with_worker(
    source: np.ndarray | str,
    chunk_start: float,
    use_fp16: bool,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
) -> list[dict]:
    """Run one chunk decode inside the worker process."""
    global _WORKER_BACKEND, _WORKER_MODEL

    if _WORKER_MODEL is None:
        raise RuntimeError("Transcriber worker model not initialized")

    segments = []
    beam_size = max(1, int(beam_size))
    if _WORKER_BACKEND == "faster-whisper":
        result_segments, _info = _WORKER_MODEL.transcribe(
            source,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=beam_size,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 250,
                "speech_pad_ms": 120,
            },
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.45,
        )
        for seg in result_segments:
            text = seg.text.strip()
            if text:
                segments.append(
                    {
                        "text": text,
                        "start_time": chunk_start + seg.start,
                        "end_time": chunk_start + seg.end,
                        "confidence": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                    }
                )
        return segments

    result = _WORKER_MODEL.transcribe(
        source,
        language=language,
        fp16=use_fp16,
        no_speech_threshold=0.45,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        temperature=0.0,
        beam_size=beam_size,
        best_of=beam_size,
        verbose=None,
    )
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if text:
            segments.append(
                {
                    "text": text,
                    "start_time": chunk_start + seg["start"],
                    "end_time": chunk_start + seg["end"],
                    "confidence": seg.get("avg_logprob", 0.0),
                }
            )
    return segments


@dataclass
class TranscriptSegment:
    """A single transcribed segment."""
    text: str
    start_time: float  # seconds from meeting start
    end_time: float
    confidence: float = 0.0


class MeetingTranscriber:
    """Real-time meeting transcription using Whisper."""

    def __init__(self, model_size: str = "base", final_model_size: str | None = None):
        """
        Args:
            model_size: Whisper model size -- "tiny", "base", "small", "medium"
                        tiny=39MB (fastest), base=74MB (good balance),
                        small=244MB (better accuracy), medium=769MB (best)
        """
        self._model_size = model_size
        self._final_model_size = final_model_size or model_size
        self._model = None
        self._initialized = False
        self._recording = False
        self._segments: list[TranscriptSegment] = []
        self._audio_buffer = []
        self._start_time = 0.0
        self._buffer_duration = 0.0
        self._lock = threading.Lock()
        self._processing_lock = threading.Lock()
        # A slightly longer chunk plus overlap improves word-boundary accuracy
        # enough to matter for meeting notes without making the UI unusably slow.
        self._chunk_duration = float(os.getenv("NVBROADCAST_TRANSCRIBER_CHUNK_SECONDS", "3.2"))
        self._chunk_overlap = float(os.getenv("NVBROADCAST_TRANSCRIBER_CHUNK_OVERLAP", "0.45"))
        self._sample_rate = 16000  # Whisper expects 16kHz
        self._thread = None
        self._segment_callback = None
        self._device = "cpu"
        self._use_fp16 = False
        self._backend_name = ""
        self._executor: ProcessPoolExecutor | None = None
        self._futures = set()
        # Keep meeting transcription off the GPU by default. In live testing,
        # sharing the GPU with the video stack produced garbage punctuation
        # transcripts even though offline decoding of the same audio was fine.
        self._preferred_device = os.getenv(
            "NVBROADCAST_TRANSCRIBER_DEVICE", "cpu"
        ).strip().lower()
        self._final_device = os.getenv(
            "NVBROADCAST_TRANSCRIBER_FINAL_DEVICE", "cpu"
        ).strip().lower()
        self._language = _coerce_language(os.getenv("NVBROADCAST_TRANSCRIBER_LANGUAGE"))
        try:
            self._beam_size = max(
                1, min(8, int(os.getenv("NVBROADCAST_TRANSCRIBER_BEAM_SIZE", "5")))
            )
        except ValueError:
            self._beam_size = 5
        self._vad_filter = os.getenv("NVBROADCAST_TRANSCRIBER_VAD", "1").strip() != "0"
        self._target_rms = 0.12

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def segments(self) -> list[TranscriptSegment]:
        with self._lock:
            return list(self._segments)

    def initialize(self) -> bool:
        """Load Whisper model. Downloads on first use (~74MB for base)."""
        if self._initialized:
            return True
        try:
            backend_ok = False
            for module_name in ("faster_whisper", "whisper"):
                try:
                    __import__(module_name)
                    backend_ok = True
                    break
                except ImportError:
                    continue
            if not backend_ok:
                raise ImportError("No local transcription backend installed")
            try:
                import torch
                preferred = self._preferred_device
                if preferred == "cuda" and torch.cuda.is_available():
                    self._device = "cuda"
                elif preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
                # Streaming decode is more stable with fp32 across devices.
                self._use_fp16 = False
            except Exception:
                self._device = "cpu"
                self._use_fp16 = False
            print(f"[Transcriber] Loading Whisper {self._model_size} model on {self._device}...")

            ctx = get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_init_transcriber_worker,
                initargs=(self._model_size, self._device),
            )
            # Force worker startup now so meeting start fails fast instead of
            # crashing later in the middle of a live capture.
            self._backend_name = self._executor.submit(_worker_ping).result(timeout=120)
            self._initialized = True
            print(
                f"[Transcriber] Model loaded ({self._model_size}, {self._device}, {self._backend_name})"
            )
            return True
        except ImportError:
            print(
                "[Transcriber] No meeting transcription backend installed. "
                "Run: pip install faster-whisper"
            )
            return False
        except Exception as e:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            print(f"[Transcriber] Failed to load model: {e}")
            return False

    def set_segment_callback(self, callback):
        """Receive live transcript segments as they are produced."""
        self._segment_callback = callback

    def start(self) -> bool:
        """Start recording a meeting transcript."""
        if not self._initialized:
            if not self.initialize():
                return False
        self._recording = True
        self._start_time = time.monotonic()
        self._segments = []
        self._audio_buffer = []
        self._buffer_duration = 0.0
        print("[Transcriber] Meeting recording started")
        return True

    def stop(self) -> list[TranscriptSegment]:
        """Stop recording and process any remaining audio."""
        self._recording = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        self._thread = None
        # Process remaining buffer
        self._process_buffer()
        pending = ()
        with self._lock:
            pending = tuple(self._futures)
        if pending:
            wait(pending, timeout=10)
        print(f"[Transcriber] Meeting ended. {len(self._segments)} segments transcribed.")
        return self.segments

    def cleanup(self):
        """Release worker resources explicitly during app shutdown."""
        self._recording = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        self._thread = None
        with self._lock:
            self._audio_buffer = []
            self._buffer_duration = 0.0
            pending = tuple(self._futures)
            self._futures.clear()
        if pending:
            wait(pending, timeout=5)
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._initialized = False

    def replace_segments(self, segments: list[TranscriptSegment]):
        with self._lock:
            self._segments = list(segments)

    def transcribe_file(self, audio_path: str) -> list[TranscriptSegment]:
        """Run a higher-accuracy pass on the complete meeting audio."""
        if not audio_path or not os.path.exists(audio_path):
            return []

        final_model = self._select_final_model(audio_path)
        final_device = self._coerce_final_device(self._final_device)
        attempts = [(final_model, final_device)]
        if final_device != "cpu":
            attempts.append((final_model, "cpu"))
        if final_model != "small":
            attempts.append(("small", "cpu"))
        attempts.append(("base", "cpu"))

        seen: set[tuple[str, str]] = set()
        last_error: Exception | None = None
        for model_name, device_name in attempts:
            key = (model_name, device_name)
            if key in seen:
                continue
            seen.add(key)
            try:
                return self._transcribe_file_once(audio_path, model_name, device_name)
            except Exception as exc:
                last_error = exc
                print(
                    f"[Transcriber] Final pass fallback after {model_name}/{device_name}: {exc}"
                )

        if last_error is not None:
            raise last_error
        return []

    def _coerce_final_device(self, value: str | None) -> str:
        value = (value or "cpu").strip().lower()
        if value not in {"cpu", "cuda", "mps"}:
            return "cpu"
        return value

    def _estimate_audio_duration(self, audio_path: str) -> float:
        try:
            with wave.open(audio_path, "rb") as wav_file:
                rate = wav_file.getframerate() or 16000
                frames = wav_file.getnframes()
                return float(frames) / float(rate)
        except Exception:
            return 0.0

    def _select_final_model(self, audio_path: str) -> str:
        override = os.getenv("NVBROADCAST_TRANSCRIBER_FINAL_MODEL")
        if override and override.strip():
            return override.strip()

        duration = self._estimate_audio_duration(audio_path)
        if duration and duration <= 20 * 60:
            return "medium"
        return self._final_model_size

    def _transcribe_file_once(
        self,
        audio_path: str,
        model_name: str,
        device_name: str,
    ) -> list[TranscriptSegment]:
        print(
            f"[Transcriber] Finalizing transcript with {model_name} "
            f"on {device_name}..."
        )

        ctx = get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=_init_transcriber_worker,
            initargs=(model_name, device_name),
        )
        try:
            backend = executor.submit(_worker_ping).result(timeout=180)
            print(
                f"[Transcriber] Final pass backend: {backend} "
                f"({model_name}, {device_name})"
            )
            source = self._load_audio_source(audio_path)
            future = executor.submit(
                _decode_with_worker,
                source,
                0.0,
                False,
                self._language,
                max(self._beam_size, 5),
                True,
            )
            timeout = max(3600, int(self._estimate_audio_duration(audio_path) * 6))
            segments = future.result(timeout=timeout)
            return [TranscriptSegment(**seg) for seg in segments]
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _load_audio_source(self, audio_path: str) -> np.ndarray | str:
        """Load meeting WAV files directly so final passes do not depend on ffmpeg."""
        if audio_path.lower().endswith(".wav"):
            audio, sample_rate = self._read_wav_file(audio_path)
            return self._prepare_audio(audio, sample_rate)
        return audio_path

    def _read_wav_file(self, audio_path: str) -> tuple[np.ndarray, int]:
        with wave.open(audio_path, "rb") as wav_file:
            channels = max(1, wav_file.getnchannels())
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate() or self._sample_rate
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio.astype(np.float32, copy=False), sample_rate

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample, de-bias, and normalize speech before decode."""
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.nan_to_num(
            audio.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if audio.size:
            audio = audio - float(audio.mean())

        np.clip(audio, -1.0, 1.0, out=audio)

        if sample_rate != self._sample_rate:
            import scipy.signal

            audio = scipy.signal.resample_poly(audio, self._sample_rate, sample_rate)
            audio = audio.astype(np.float32, copy=False)

        if audio.size:
            rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-8))
            if rms > 0.0:
                gain = min(4.0, max(0.7, self._target_rms / rms))
                audio = audio * gain
        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    def _normalized_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()

    def _store_segment(self, segment: TranscriptSegment) -> bool:
        """Append decoded output while suppressing overlap duplicates."""
        with self._lock:
            if self._segments:
                last = self._segments[-1]
                overlap_window = segment.start_time <= (last.end_time + 0.8)
                if overlap_window:
                    last_text = self._normalized_text(last.text)
                    new_text = self._normalized_text(segment.text)
                    if not new_text:
                        return False
                    if new_text == last_text or new_text in last_text:
                        return False
                    if last_text and last_text in new_text:
                        self._segments[-1] = segment
                        return False
            self._segments.append(segment)
        return True

    def feed_audio(self, audio: np.ndarray, sample_rate: int = 48000):
        """Feed audio chunk from the pipeline.

        Args:
            audio: float32 numpy array
            sample_rate: input sample rate (will be resampled to 16kHz)
        """
        if not self._recording:
            return

        audio = self._prepare_audio(audio, sample_rate)

        with self._lock:
            self._audio_buffer.append(audio.astype(np.float32))
            self._buffer_duration += len(audio) / self._sample_rate

        # Process when we have enough audio
        if self._buffer_duration >= self._chunk_duration:
            # Process in background thread to not block audio pipeline
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._process_buffer, daemon=True)
                self._thread.start()

    def _process_buffer(self):
        """Transcribe accumulated audio buffer."""
        with self._processing_lock:
            with self._lock:
                if not self._audio_buffer:
                    return
                buffered_duration = self._buffer_duration
                audio = np.concatenate(self._audio_buffer)
                chunk_start = time.monotonic() - self._start_time - buffered_duration
                overlap_samples = min(
                    int(self._chunk_overlap * self._sample_rate),
                    max(0, len(audio) - 1600),
                )
                if overlap_samples > 0:
                    self._audio_buffer = [audio[-overlap_samples:].copy()]
                    self._buffer_duration = overlap_samples / self._sample_rate
                else:
                    self._audio_buffer = []
                    self._buffer_duration = 0.0

            if self._executor is None or len(audio) < 1600:  # <0.1s
                return

            try:
                future = self._executor.submit(
                    _decode_with_worker,
                    audio,
                    chunk_start,
                    self._use_fp16,
                    self._language,
                    self._beam_size,
                    self._vad_filter,
                )
                with self._lock:
                    self._futures.add(future)
                future.add_done_callback(self._on_future_done)
            except Exception as e:
                print(f"[Transcriber] Error: {e}")

    def _on_future_done(self, future):
        with self._lock:
            self._futures.discard(future)

        try:
            segments = future.result()
        except Exception as e:
            print(f"[Transcriber] Error: {e}")
            return

        for seg in segments:
            segment = TranscriptSegment(**seg)
            appended = self._store_segment(segment)
            if appended and self._segment_callback is not None:
                try:
                    self._segment_callback(segment)
                except Exception:
                    pass

    def get_full_transcript(self) -> str:
        """Get the complete transcript as plain text."""
        return "\n".join(seg.text for seg in self.segments)

    def get_timestamped_transcript(self) -> str:
        """Get transcript with timestamps."""
        lines = []
        for seg in self.segments:
            m1, s1 = divmod(int(seg.start_time), 60)
            m2, s2 = divmod(int(seg.end_time), 60)
            lines.append(f"[{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] {seg.text}")
        return "\n".join(lines)


def save_transcript(segments: list[TranscriptSegment], filepath: str,
                    format: str = "txt") -> str:
    """Save transcript to file.

    Args:
        segments: list of TranscriptSegment
        filepath: output file path (without extension)
        format: "txt", "srt", or "json"
    """
    from pathlib import Path

    if format == "srt":
        path = Path(filepath).with_suffix(".srt")
        lines = []
        for i, seg in enumerate(segments, 1):
            s1 = _format_srt_time(seg.start_time)
            s2 = _format_srt_time(seg.end_time)
            lines.append(f"{i}\n{s1} --> {s2}\n{seg.text}\n")
        path.write_text("\n".join(lines))

    elif format == "json":
        import json
        path = Path(filepath).with_suffix(".json")
        data = [{
            "text": seg.text,
            "start": seg.start_time,
            "end": seg.end_time,
        } for seg in segments]
        path.write_text(json.dumps(data, indent=2))

    else:
        path = Path(filepath).with_suffix(".txt")
        lines = []
        for seg in segments:
            m, s = divmod(int(seg.start_time), 60)
            lines.append(f"[{m:02d}:{s:02d}] {seg.text}")
        path.write_text("\n".join(lines))

    return str(path)


def _format_srt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

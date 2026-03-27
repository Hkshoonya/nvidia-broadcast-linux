"""Meeting transcription -- local Whisper-based speech-to-text.

Uses OpenAI Whisper (open-source) running locally on GPU/CPU.
No data leaves the machine. Transcribes audio chunks in real-time
and saves complete transcript at meeting end.
"""

import time
import threading
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TranscriptSegment:
    """A single transcribed segment."""
    text: str
    start_time: float  # seconds from meeting start
    end_time: float
    confidence: float = 0.0


class MeetingTranscriber:
    """Real-time meeting transcription using Whisper."""

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: Whisper model size -- "tiny", "base", "small", "medium"
                        tiny=39MB (fastest), base=74MB (good balance),
                        small=244MB (better accuracy), medium=769MB (best)
        """
        self._model_size = model_size
        self._model = None
        self._initialized = False
        self._recording = False
        self._segments: list[TranscriptSegment] = []
        self._audio_buffer = []
        self._start_time = 0.0
        self._buffer_duration = 0.0
        self._lock = threading.Lock()
        # Process audio every 5 seconds of accumulated audio
        self._chunk_duration = 5.0
        self._sample_rate = 16000  # Whisper expects 16kHz
        self._thread = None
        self._segment_callback = None

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
            import whisper
            print(f"[Transcriber] Loading Whisper {self._model_size} model...")
            self._model = whisper.load_model(self._model_size)
            self._initialized = True
            print(f"[Transcriber] Model loaded ({self._model_size})")
            return True
        except ImportError:
            print("[Transcriber] openai-whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            print(f"[Transcriber] Failed to load model: {e}")
            return False

    def set_segment_callback(self, callback):
        """Receive live transcript segments as they are produced."""
        self._segment_callback = callback

    def start(self):
        """Start recording a meeting transcript."""
        if not self._initialized:
            if not self.initialize():
                return
        self._recording = True
        self._start_time = time.monotonic()
        self._segments = []
        self._audio_buffer = []
        self._buffer_duration = 0.0
        print("[Transcriber] Meeting recording started")

    def stop(self) -> list[TranscriptSegment]:
        """Stop recording and process any remaining audio."""
        self._recording = False
        # Process remaining buffer
        self._process_buffer()
        print(f"[Transcriber] Meeting ended. {len(self._segments)} segments transcribed.")
        return self.segments

    def feed_audio(self, audio: np.ndarray, sample_rate: int = 48000):
        """Feed audio chunk from the pipeline.

        Args:
            audio: float32 numpy array
            sample_rate: input sample rate (will be resampled to 16kHz)
        """
        if not self._recording:
            return

        # Resample to 16kHz if needed
        if sample_rate != self._sample_rate:
            import scipy.signal
            audio = scipy.signal.resample(
                audio, int(len(audio) * self._sample_rate / sample_rate)
            )

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
        with self._lock:
            if not self._audio_buffer:
                return
            audio = np.concatenate(self._audio_buffer)
            chunk_start = time.monotonic() - self._start_time - self._buffer_duration
            self._audio_buffer = []
            self._buffer_duration = 0.0

        if self._model is None or len(audio) < 1600:  # <0.1s
            return

        try:
            result = self._model.transcribe(
                audio,
                language="en",
                fp16=True,  # GPU acceleration
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
            )

            for seg in result.get("segments", []):
                text = seg["text"].strip()
                if text:
                    segment = TranscriptSegment(
                        text=text,
                        start_time=chunk_start + seg["start"],
                        end_time=chunk_start + seg["end"],
                        confidence=seg.get("avg_logprob", 0.0),
                    )
                    with self._lock:
                        self._segments.append(segment)
                    if self._segment_callback is not None:
                        try:
                            self._segment_callback(segment)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Transcriber] Error: {e}")

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

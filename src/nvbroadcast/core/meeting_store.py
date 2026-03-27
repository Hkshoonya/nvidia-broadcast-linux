# NVIDIA Broadcast for Linux
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0 - see LICENSE file
#
"""On-device meeting session storage with retention cleanup."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from nvbroadcast.core.constants import CONFIG_DIR


MEETINGS_DIR = CONFIG_DIR / "meetings"
RETENTION_DAYS = 7


@dataclass
class MeetingSession:
    session_id: str
    created_at: int
    title: str
    summary: str
    transcript_preview: str
    duration_seconds: float
    notes_path: str
    transcript_path: str
    transcript_srt_path: str
    audio_path: str
    video_path: str


def _session_dir(session_id: str) -> Path:
    return MEETINGS_DIR / session_id


def cleanup_old_sessions(retention_days: int = RETENTION_DAYS) -> None:
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - (retention_days * 24 * 60 * 60)
    for session_dir in MEETINGS_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            stat = session_dir.stat()
        except OSError:
            continue
        if stat.st_mtime < cutoff:
            shutil.rmtree(session_dir, ignore_errors=True)


def create_session() -> tuple[str, Path]:
    cleanup_old_sessions()
    session_id = time.strftime("%Y%m%d-%H%M%S")
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def save_session(session: MeetingSession) -> Path:
    session_dir = _session_dir(session.session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = session_dir / "metadata.json"
    metadata_path.write_text(json.dumps(asdict(session), indent=2))
    return metadata_path


def list_sessions(limit: int = 25) -> list[MeetingSession]:
    cleanup_old_sessions()
    sessions: list[MeetingSession] = []
    if not MEETINGS_DIR.exists():
        return sessions
    for session_dir in sorted(MEETINGS_DIR.iterdir(), reverse=True):
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            data = json.loads(metadata_path.read_text())
            sessions.append(MeetingSession(**data))
        except Exception:
            continue
        if len(sessions) >= limit:
            break
    return sessions


def read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    try:
        return file_path.read_text()
    except Exception:
        return ""

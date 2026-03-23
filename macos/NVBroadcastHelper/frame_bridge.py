#!/usr/bin/env python3
# NV Broadcast - macOS Frame Bridge
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Proprietary license — see macos/LICENSE
#
# Bridges processed frames from the Python NV Broadcast app to the
# CoreMediaIO Camera Extension via a shared temp file.
#
# The Python app calls write_frame() with a BGRA numpy array.
# The Swift extension polls the same file and delivers frames
# to Zoom, Chrome, FaceTime, etc.

import os
import struct
import tempfile

# Frame file path — must match StreamSource.swift _frameFilePath
FRAME_FILE = os.path.join(tempfile.gettempdir(), "nvbroadcast_frame.raw")

# Frame header: width(4) + height(4) + sequence(8) = 16 bytes
HEADER_SIZE = 16


class FrameBridge:
    """File-based frame bridge from Python to CoreMediaIO extension."""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.frame_size = width * height * 4  # BGRA
        self._sequence = 0
        self._fd = None
        self._setup()

    def _setup(self):
        """Create/open the shared frame file."""
        # Pre-allocate file to avoid size changes during read
        self._fd = os.open(FRAME_FILE, os.O_CREAT | os.O_RDWR, 0o666)
        total = HEADER_SIZE + self.frame_size
        os.ftruncate(self._fd, total)

    def write_frame(self, frame_bgra: bytes | memoryview):
        """Write a BGRA frame to the shared file.

        Args:
            frame_bgra: Raw BGRA pixel data (width * height * 4 bytes)
        """
        if self._fd is None:
            return

        self._sequence += 1
        header = struct.pack("<IIQ", self.width, self.height, self._sequence)

        # Atomic-ish write: seek to start, write header + frame
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, header)
        if isinstance(frame_bgra, memoryview):
            os.write(self._fd, bytes(frame_bgra[:self.frame_size]))
        else:
            os.write(self._fd, frame_bgra[:self.frame_size])

    def write_numpy_frame(self, frame):
        """Write a numpy BGRA array to the shared file.

        Args:
            frame: numpy array of shape (height, width, 4) dtype uint8
        """
        self.write_frame(frame.tobytes())

    def close(self):
        """Clean up."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            os.unlink(FRAME_FILE)
        except OSError:
            pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

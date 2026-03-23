#!/usr/bin/env python3
# NV Broadcast - macOS Frame Bridge
# Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
# Licensed under GPL-3.0
#
# Bridges processed frames from the Python NV Broadcast app to the
# CoreMediaIO Camera Extension via POSIX shared memory + Darwin notifications.
#
# The Python app calls write_frame() with a BGRA numpy array.
# The Swift extension reads from the same shared memory segment
# and delivers frames to Zoom, Chrome, FaceTime, etc.

import ctypes
import ctypes.util
import mmap
import os
import struct

# Shared memory name must match NVBroadcastConstants.sharedMemoryName
SHM_NAME = "/nvbroadcast_frame"
NOTIFY_NAME = "com.doczeus.nvbroadcast.newframe"

# Frame header: width(4) + height(4) + sequence(8) = 16 bytes
HEADER_SIZE = 16


class FrameBridge:
    """Zero-copy frame bridge from Python to CoreMediaIO extension."""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.frame_size = width * height * 4  # BGRA
        self.total_size = HEADER_SIZE + self.frame_size
        self._sequence = 0
        self._shm_fd = -1
        self._mmap = None
        self._notify_center = None
        self._setup_shm()
        self._setup_notify()

    def _setup_shm(self):
        """Create POSIX shared memory segment for frame data."""
        # Load C runtime
        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        # shm_open flags
        O_CREAT = 0x0200
        O_RDWR = 0x0002
        O_TRUNC = 0x0400

        # Create shared memory
        shm_name = SHM_NAME.encode()
        self._shm_fd = libc.shm_open(shm_name, O_CREAT | O_RDWR | O_TRUNC, 0o666)
        if self._shm_fd < 0:
            raise RuntimeError(f"Failed to create shared memory: {SHM_NAME}")

        # Set size
        libc.ftruncate(self._shm_fd, self.total_size)

        # mmap
        self._mmap = mmap.mmap(self._shm_fd, self.total_size)

    def _setup_notify(self):
        """Set up Darwin notification posting (CFNotificationCenter)."""
        try:
            self._cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            # Get Darwin notify center
            self._cf.CFNotificationCenterGetDarwinNotifyCenter.restype = ctypes.c_void_p
            self._notify_center = self._cf.CFNotificationCenterGetDarwinNotifyCenter()

            # Create CFString for notification name
            self._cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            self._cf.CFStringCreateWithCString.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32
            ]
            self._notify_name = self._cf.CFStringCreateWithCString(
                None, NOTIFY_NAME.encode(), 0x08000100  # kCFStringEncodingUTF8
            )
        except Exception:
            self._notify_center = None

    def write_frame(self, frame_bgra: bytes | memoryview):
        """Write a BGRA frame to shared memory and notify the extension.

        Args:
            frame_bgra: Raw BGRA pixel data (width * height * 4 bytes)
        """
        if self._mmap is None:
            return

        self._sequence += 1

        # Write header + frame data
        self._mmap.seek(0)
        self._mmap.write(struct.pack("<IIQ", self.width, self.height, self._sequence))
        self._mmap.write(frame_bgra[:self.frame_size])

        # Post Darwin notification to wake the extension
        if self._notify_center:
            self._cf.CFNotificationCenterPostNotification(
                self._notify_center,
                self._notify_name,
                None,
                None,
                1,  # deliverImmediately
            )

    def write_numpy_frame(self, frame):
        """Write a numpy BGRA array to shared memory.

        Args:
            frame: numpy array of shape (height, width, 4) dtype uint8
        """
        self.write_frame(frame.tobytes())

    def close(self):
        """Clean up shared memory."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._shm_fd >= 0:
            os.close(self._shm_fd)
            self._shm_fd = -1
            # Unlink shared memory
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"))
                libc.shm_unlink(SHM_NAME.encode())
            except Exception:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

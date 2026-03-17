"""Integration tests for NVIDIA Broadcast."""

import subprocess
import time
import os
import numpy as np


def test_gpu_detection():
    from nvbroadcast.core.gpu import detect_gpus, select_compute_gpu
    gpus = detect_gpus()
    assert len(gpus) >= 1, "No GPUs detected"
    compute = select_compute_gpu(gpus)
    assert compute is not None


def test_camera_detection():
    from nvbroadcast.video.virtual_camera import list_camera_devices
    cameras = list_camera_devices()
    assert len(cameras) >= 1, "No cameras detected"
    assert cameras[0]["device"].startswith("/dev/video")


def test_virtual_camera_exists():
    from nvbroadcast.video.virtual_camera import get_virtual_camera_device
    device = get_virtual_camera_device()
    assert device is not None, "Virtual camera not found"
    assert os.path.exists(device)


def test_config_roundtrip():
    from nvbroadcast.core.config import AppConfig, save_config, load_config
    config = AppConfig()
    config.video.camera_device = "/dev/video99"
    config.compute_gpu = 1
    save_config(config)
    loaded = load_config()
    assert loaded.video.camera_device == "/dev/video99"
    config.video.camera_device = "/dev/video0"
    save_config(config)


def test_video_effects_blur():
    """Test background blur effect."""
    from nvbroadcast.video.effects import VideoEffects
    vfx = VideoEffects()
    assert vfx.initialize()
    vfx.enabled = True
    vfx.mode = "blur"
    vfx.intensity = 0.7
    frame = np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8).tobytes()
    result = vfx.process_frame(frame, 1280, 720)
    assert len(result) == len(frame)
    vfx.cleanup()


def test_video_effects_replace():
    """Test background replacement with custom image."""
    from nvbroadcast.video.effects import VideoEffects
    import cv2
    # Create test background
    bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg[:, :] = [50, 100, 200]
    os.makedirs("data/backgrounds", exist_ok=True)
    cv2.imwrite("data/backgrounds/test_bg.png", bg)

    vfx = VideoEffects()
    assert vfx.initialize()
    vfx.enabled = True
    vfx.mode = "replace"
    assert vfx.set_background_image("data/backgrounds/test_bg.png")
    frame = np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8).tobytes()
    result = vfx.process_frame(frame, 1280, 720)
    assert len(result) == len(frame)
    vfx.cleanup()


def test_autoframe():
    """Test auto-frame face tracking."""
    from nvbroadcast.video.autoframe import AutoFrame
    af = AutoFrame()
    assert af.initialize()
    af.enabled = True
    af.zoom_level = 1.5
    frame = np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8).tobytes()
    result = af.process_frame(frame, 1280, 720)
    assert len(result) == len(frame)
    af.cleanup()


def test_audio_denoise():
    """Test audio noise removal."""
    from nvbroadcast.audio.effects import AudioEffects
    afx = AudioEffects()
    assert afx.initialize()
    afx.enabled = True
    afx.intensity = 1.0
    audio = np.random.randn(4800).astype(np.float32) * 0.1
    result = afx.process_chunk(audio, 48000)
    assert len(result) == len(audio)
    # Denoised output should be quieter than noise input
    assert np.std(result) <= np.std(audio) + 0.01
    afx.cleanup()


def test_vcam_pipeline():
    """Test virtual camera pipeline streams successfully."""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    from nvbroadcast.vcam_service import build_pipeline
    pipeline = build_pipeline("/dev/video0", "/dev/video10", 1280, 720, 30, "yuy2")
    pipeline.set_state(Gst.State.PLAYING)
    time.sleep(2)
    ret, state, _ = pipeline.get_state(5 * Gst.SECOND)
    assert state == Gst.State.PLAYING, f"Pipeline not playing: {state}"
    pipeline.set_state(Gst.State.NULL)


def test_vcam_capture_mode():
    """Test virtual camera shows as capture device while streaming."""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    from nvbroadcast.vcam_service import build_pipeline
    pipeline = build_pipeline("/dev/video0", "/dev/video10", 1280, 720, 30, "yuy2")
    pipeline.set_state(Gst.State.PLAYING)
    time.sleep(2)
    result = subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video10", "--info"],
        capture_output=True, text=True
    )
    assert "Video Capture" in result.stdout, "Not showing as capture device"
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    tests = [
        test_gpu_detection,
        test_camera_detection,
        test_virtual_camera_exists,
        test_config_roundtrip,
        test_video_effects_blur,
        test_video_effects_replace,
        test_autoframe,
        test_audio_denoise,
        test_vcam_pipeline,
        test_vcam_capture_mode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")

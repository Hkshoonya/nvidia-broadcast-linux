import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nvbroadcast.video.perf_monitor import PerfMonitor


class PerfMonitorTests(unittest.TestCase):
    def test_format_status_reports_selected_gpu(self):
        monitor = PerfMonitor(gpu_index=2)
        monitor._fps = 17.4
        monitor._gpu_util = 55
        monitor._vram_used = 2048
        monitor._vram_total = 8192
        monitor._gpu_temp = 64

        status = monitor.format_status()

        self.assertIn("17 fps", status)
        self.assertIn("GPU 2 55%", status)
        self.assertIn("VRAM 2048MB/8192MB", status)
        self.assertIn("64°C", status)

    def test_poll_gpu_uses_selected_gpu_index(self):
        monitor = PerfMonitor(gpu_index=3)
        monitor._running = True
        commands = []

        def fake_run(cmd, capture_output, text, timeout):
            commands.append(cmd)
            return SimpleNamespace(stdout="12, 345, 6789, 55\n")

        def fake_sleep(_seconds):
            monitor._running = False

        with patch("nvbroadcast.video.perf_monitor.subprocess.run", side_effect=fake_run):
            with patch("nvbroadcast.video.perf_monitor.time.sleep", side_effect=fake_sleep):
                monitor._poll_gpu()

        self.assertEqual(len(commands), 1)
        self.assertIn("--id=3", commands[0])
        self.assertEqual(monitor.gpu_util, 12)
        self.assertEqual(monitor.vram_used_mb, 345)
        self.assertEqual(monitor.vram_total_mb, 6789)
        self.assertEqual(monitor.gpu_temp, 55)


if __name__ == "__main__":
    unittest.main()

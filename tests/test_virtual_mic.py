import signal
import unittest
from unittest import mock

from nvbroadcast.audio import virtual_mic


class VirtualMicTests(unittest.TestCase):
    def setUp(self):
        virtual_mic._pw_loopback_process = None
        virtual_mic._pulse_sink_module_id = None
        virtual_mic._pulse_source_module_id = None

    def tearDown(self):
        virtual_mic._pw_loopback_process = None
        virtual_mic._pulse_sink_module_id = None
        virtual_mic._pulse_source_module_id = None

    @mock.patch("nvbroadcast.audio.virtual_mic.shutil.which")
    def test_virtual_mic_backend_prefers_pactl(self, which):
        which.side_effect = lambda name: "/usr/bin/pactl" if name == "pactl" else "/usr/bin/pw-loopback"
        self.assertEqual(virtual_mic.virtual_mic_backend(), "pulse")

    @mock.patch("nvbroadcast.audio.virtual_mic._run_pactl")
    @mock.patch("nvbroadcast.audio.virtual_mic.shutil.which", return_value="/usr/bin/pactl")
    def test_create_virtual_mic_uses_pulse_modules_when_pactl_available(self, _which, run_pactl):
        run_pactl.side_effect = [
            mock.Mock(returncode=0, stdout="536870913\n", stderr=""),
            mock.Mock(returncode=0, stdout="536870914\n", stderr=""),
            mock.Mock(returncode=0, stdout=f"1\t{virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n", stderr=""),
        ]

        self.assertTrue(virtual_mic.create_virtual_mic())

        sink_call = run_pactl.call_args_list[0].args[0]
        source_call = run_pactl.call_args_list[1].args[0]
        self.assertIn("module-null-sink", sink_call)
        self.assertIn(f"sink_name={virtual_mic.VIRTUAL_MIC_SINK_NAME}", sink_call)
        self.assertIn("module-remap-source", source_call)
        self.assertIn(f"source_name={virtual_mic.VIRTUAL_MIC_SOURCE_NAME}", source_call)

    @mock.patch("nvbroadcast.audio.virtual_mic.subprocess.Popen")
    @mock.patch("nvbroadcast.audio.virtual_mic.shutil.which")
    def test_create_virtual_mic_falls_back_to_pw_loopback(self, which, popen):
        which.side_effect = lambda name: "/usr/bin/pw-loopback" if name == "pw-loopback" else None
        proc = mock.Mock()
        proc.poll.return_value = None
        popen.return_value = proc

        self.assertTrue(virtual_mic.create_virtual_mic())

        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[0], "pw-loopback")
        self.assertIn("--capture-props", cmd)
        self.assertIn("--playback-props", cmd)

    @mock.patch("nvbroadcast.audio.virtual_mic._run_pactl")
    def test_destroy_virtual_mic_unloads_pulse_modules(self, run_pactl):
        virtual_mic._pulse_sink_module_id = 11
        virtual_mic._pulse_source_module_id = 12

        virtual_mic.destroy_virtual_mic()

        unload_calls = [call.args[0] for call in run_pactl.call_args_list]
        self.assertEqual(unload_calls, [["unload-module", "12"], ["unload-module", "11"]])
        self.assertIsNone(virtual_mic._pulse_sink_module_id)
        self.assertIsNone(virtual_mic._pulse_source_module_id)

    def test_destroy_virtual_mic_stops_running_pw_loopback(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        virtual_mic._pw_loopback_process = proc

        virtual_mic.destroy_virtual_mic()

        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(virtual_mic._pw_loopback_process)

    def test_virtual_mic_sink_name_is_stable(self):
        self.assertEqual(virtual_mic.virtual_mic_sink_name(), "nvbroadcast_sink")


if __name__ == "__main__":
    unittest.main()

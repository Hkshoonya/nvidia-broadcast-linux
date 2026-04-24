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
        def _side_effect(args):
            if args[:3] == ["list", "short", "modules"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            if args[:3] == ["list", "sources", "short"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            if args[:3] == ["list", "sinks", "short"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            if args[:2] == ["load-module", "module-null-sink"]:
                return mock.Mock(returncode=0, stdout="536870913\n", stderr="")
            if args[:2] == ["load-module", "module-remap-source"]:
                return mock.Mock(returncode=0, stdout="536870914\n", stderr="")
            raise AssertionError(f"Unexpected pactl args: {args}")

        run_pactl.side_effect = _side_effect

        self.assertTrue(virtual_mic.create_virtual_mic())

        calls = [call.args[0] for call in run_pactl.call_args_list]
        sink_call = next(call for call in calls if call[:2] == ["load-module", "module-null-sink"])
        source_call = next(call for call in calls if call[:2] == ["load-module", "module-remap-source"])
        self.assertIn("module-null-sink", sink_call)
        self.assertIn(f"sink_name={virtual_mic.VIRTUAL_MIC_SINK_NAME}", sink_call)
        self.assertIn("module-remap-source", source_call)
        self.assertIn(f"source_name={virtual_mic.VIRTUAL_MIC_SOURCE_NAME}", source_call)

    @mock.patch("nvbroadcast.audio.virtual_mic._run_pactl")
    @mock.patch("nvbroadcast.audio.virtual_mic.shutil.which", return_value="/usr/bin/pactl")
    def test_create_virtual_mic_reuses_existing_single_pulse_pair(self, _which, run_pactl):
        def _side_effect(args):
            if args[:3] == ["list", "short", "modules"]:
                return mock.Mock(
                    returncode=0,
                    stdout=(
                        f"11\tmodule-null-sink\tsink_name={virtual_mic.VIRTUAL_MIC_SINK_NAME}\n"
                        f"12\tmodule-remap-source\tsource_name={virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\n"
                    ),
                    stderr="",
                )
            if args[:3] == ["list", "sources", "short"]:
                return mock.Mock(
                    returncode=0,
                    stdout=f"1\t{virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n",
                    stderr="",
                )
            if args[:3] == ["list", "sinks", "short"]:
                return mock.Mock(
                    returncode=0,
                    stdout=f"2\t{virtual_mic.VIRTUAL_MIC_SINK_NAME}\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n",
                    stderr="",
                )
            raise AssertionError(f"Unexpected pactl args: {args}")

        run_pactl.side_effect = _side_effect

        self.assertTrue(virtual_mic.create_virtual_mic())
        self.assertEqual(run_pactl.call_count, 3)
        self.assertEqual(virtual_mic._pulse_sink_module_id, 11)
        self.assertEqual(virtual_mic._pulse_source_module_id, 12)

    @mock.patch("nvbroadcast.audio.virtual_mic._run_pactl")
    @mock.patch("nvbroadcast.audio.virtual_mic.shutil.which", return_value="/usr/bin/pactl")
    def test_create_virtual_mic_cleans_duplicates_before_recreating(self, _which, run_pactl):
        def _side_effect(args):
            if args[:3] == ["list", "short", "modules"]:
                return mock.Mock(
                    returncode=0,
                    stdout=(
                        f"11\tmodule-null-sink\tsink_name={virtual_mic.VIRTUAL_MIC_SINK_NAME}\n"
                        f"12\tmodule-null-sink\tsink_name={virtual_mic.VIRTUAL_MIC_SINK_NAME}\n"
                        f"21\tmodule-remap-source\tsource_name={virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\n"
                        f"22\tmodule-remap-source\tsource_name={virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\n"
                    ),
                    stderr="",
                )
            if args[:3] == ["list", "sources", "short"]:
                return mock.Mock(
                    returncode=0,
                    stdout=f"1\t{virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n",
                    stderr="",
                )
            if args[:3] == ["list", "sinks", "short"]:
                return mock.Mock(
                    returncode=0,
                    stdout=f"2\t{virtual_mic.VIRTUAL_MIC_SINK_NAME}\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n",
                    stderr="",
                )
            if args[:1] == ["unload-module"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            if args[:2] == ["load-module", "module-null-sink"]:
                return mock.Mock(returncode=0, stdout="31\n", stderr="")
            if args[:2] == ["load-module", "module-remap-source"]:
                return mock.Mock(returncode=0, stdout="32\n", stderr="")
            raise AssertionError(f"Unexpected pactl args: {args}")

        run_pactl.side_effect = _side_effect

        self.assertTrue(virtual_mic.create_virtual_mic())

        unload_calls = [call.args[0] for call in run_pactl.call_args_list if call.args[0][0] == "unload-module"]
        self.assertEqual(
            unload_calls,
            [["unload-module", "21"], ["unload-module", "22"], ["unload-module", "11"], ["unload-module", "12"]],
        )
        self.assertEqual(virtual_mic._pulse_sink_module_id, 31)
        self.assertEqual(virtual_mic._pulse_source_module_id, 32)

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
        run_pactl.side_effect = lambda args: (
            mock.Mock(
                returncode=0,
                stdout=(
                    f"11\tmodule-null-sink\tsink_name={virtual_mic.VIRTUAL_MIC_SINK_NAME}\n"
                    f"12\tmodule-remap-source\tsource_name={virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\n"
                ),
                stderr="",
            )
            if args[:3] == ["list", "short", "modules"]
            else mock.Mock(returncode=0, stdout="", stderr="")
        )

        virtual_mic.destroy_virtual_mic()

        unload_calls = [call.args[0] for call in run_pactl.call_args_list]
        self.assertEqual(
            unload_calls,
            [["list", "short", "modules"], ["list", "short", "modules"], ["unload-module", "12"], ["unload-module", "11"]],
        )
        self.assertIsNone(virtual_mic._pulse_sink_module_id)
        self.assertIsNone(virtual_mic._pulse_source_module_id)

    @mock.patch("nvbroadcast.audio.virtual_mic._run_pactl")
    def test_destroy_virtual_mic_unloads_matching_modules_even_without_globals(self, run_pactl):
        run_pactl.side_effect = lambda args: (
            mock.Mock(
                returncode=0,
                stdout=(
                    f"15\tmodule-null-sink\tsink_name={virtual_mic.VIRTUAL_MIC_SINK_NAME}\n"
                    f"16\tmodule-remap-source\tsource_name={virtual_mic.VIRTUAL_MIC_SOURCE_NAME}\n"
                ),
                stderr="",
            )
            if args[:3] == ["list", "short", "modules"]
            else mock.Mock(returncode=0, stdout="", stderr="")
        )

        virtual_mic.destroy_virtual_mic()

        unload_calls = [call.args[0] for call in run_pactl.call_args_list]
        self.assertEqual(
            unload_calls,
            [["list", "short", "modules"], ["list", "short", "modules"], ["unload-module", "16"], ["unload-module", "15"]],
        )

    @mock.patch("nvbroadcast.audio.virtual_mic.virtual_mic_backend", return_value="")
    def test_destroy_virtual_mic_stops_running_pw_loopback(self, _backend):
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

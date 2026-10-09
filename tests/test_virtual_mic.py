import signal
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
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

    @mock.patch("nvbroadcast.audio.virtual_mic.virtual_mic_backend", return_value="pulse")
    @mock.patch("nvbroadcast.audio.virtual_mic._run_pactl")
    def test_destroy_virtual_mic_unloads_pulse_modules(self, run_pactl, _backend):
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

    @mock.patch("nvbroadcast.audio.virtual_mic.virtual_mic_backend", return_value="pulse")
    @mock.patch("nvbroadcast.audio.virtual_mic._run_pactl")
    def test_destroy_virtual_mic_unloads_matching_modules_even_without_globals(self, run_pactl, _backend):
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


@unittest.skipUnless(
    shutil.which("pulseaudio") and shutil.which("pactl"),
    "A private PulseAudio server and pactl are required",
)
class PulseVirtualMicIntegrationTests(unittest.TestCase):
    def test_real_pulse_module_parser_preserves_description_and_reconnects(self):
        # A private server with no hardware modules cannot change the user's
        # microphone, speakers, default sources, or running audio session.
        with tempfile.TemporaryDirectory(prefix="nvb-pulse-") as directory:
            root = Path(directory)
            env = {
                **os.environ,
                "XDG_RUNTIME_DIR": str(root),
                "XDG_CONFIG_HOME": str(root / "config"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "PULSE_RUNTIME_PATH": str(root),
                "PULSE_SERVER": f"unix:{root}/native",
            }
            with (root / "server.log").open("w+") as log:
                server = subprocess.Popen(
                    ["pulseaudio", "-n", "--daemonize=no", "--exit-idle-time=-1",
                     "--use-pid-file=no", "--disable-shm=yes", "-L",
                     f"module-native-protocol-unix socket={root}/native auth-anonymous=1"],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 5
                    while not (root / "native").exists():
                        if server.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            self.fail(f"Private PulseAudio failed to start: {log.read()}")
                        time.sleep(.02)
                    with mock.patch.dict(os.environ, env, clear=True):
                        for _ in range(2):
                            self.assertTrue(virtual_mic.create_virtual_mic())
                            sinks = virtual_mic._run_pactl(["list", "sinks"])
                            self.assertEqual(sinks.returncode, 0, sinks.stderr)
                            self.assertIn(
                                f"Description: {virtual_mic.VIRTUAL_MIC_INPUT_DESCRIPTION}",
                                sinks.stdout,
                            )
                            sources, _ = virtual_mic._pulse_named_nodes()
                            self.assertEqual(sources, [virtual_mic.VIRTUAL_MIC_SOURCE_NAME])
                            self.assertTrue(virtual_mic.create_virtual_mic())
                            self.assertEqual(
                                tuple(map(len, virtual_mic._list_pulse_virtual_modules())),
                                (1, 1),
                            )
                            virtual_mic.destroy_virtual_mic()
                            self.assertEqual(virtual_mic._pulse_named_nodes(), ([], []))
                finally:
                    server.terminate()
                    server.wait(timeout=5)
                    virtual_mic._pulse_sink_module_id = None
                    virtual_mic._pulse_source_module_id = None


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

from nvbroadcast.audio import devices


class AudioDeviceResolverTests(unittest.TestCase):
    def test_resolve_pipewire_target_maps_numeric_id_to_node_name(self):
        fake_nodes = [
            {
                "type": "PipeWire:Interface:Node",
                "id": 33,
                "info": {"props": {"node.name": "alsa_input.demo", "media.class": "Audio/Source"}},
            }
        ]
        with mock.patch.object(devices, "_pw_nodes", return_value=fake_nodes):
            self.assertEqual(devices.resolve_pipewire_target("33"), "alsa_input.demo")

    def test_resolve_pulse_source_name_maps_saved_numeric_id(self):
        with mock.patch.object(devices, "_pactl_short", return_value=[
            ("59", "alsa_input.usb-demo"),
        ]):
            self.assertEqual(
                devices.resolve_pulse_source_name("59"), "alsa_input.usb-demo"
            )
            self.assertEqual(
                devices.resolve_pulse_source_name("alsa_input.usb-demo"),
                "alsa_input.usb-demo",
            )
            self.assertEqual(devices.resolve_pulse_source_name("123"), "")

    def test_stale_numeric_pulse_sink_omits_monitor_branch(self):
        with mock.patch.object(devices, "_pactl_short", return_value=[
            ("7", "alsa_output.usb-demo"),
        ]):
            self.assertEqual(
                devices.resolve_pulse_sink_name("123"), ""
            )

    def test_resolve_speaker_monitor_name_maps_numeric_sink_without_pipewire(self):
        with mock.patch.object(devices, "_pw_nodes", return_value=[]), \
             mock.patch.object(devices, "_pactl_short", side_effect=lambda kind: {
                 "sinks": [("7", "alsa_output.usb-demo")],
                 "sources": [("8", "alsa_output.usb-demo.monitor")],
             }[kind]):
            self.assertEqual(
                devices.resolve_speaker_monitor_name("7"),
                "alsa_output.usb-demo.monitor",
            )

    def test_resolve_speaker_monitor_returns_monitor_source_id(self):
        fake_sources = "228\talsa_output.demo.monitor\tPipeWire\ts16le 2ch 48000Hz\tRUNNING\n"
        with mock.patch.object(devices, "resolve_pipewire_target", return_value="alsa_output.demo"):
            with mock.patch("subprocess.run") as run:
                run.return_value.stdout = fake_sources
                self.assertEqual(devices.resolve_speaker_monitor("alsa_output.demo"), "228")

    def test_resolve_speaker_monitor_name_returns_monitor_source_name(self):
        fake_sources = "228\talsa_output.demo.monitor\tPipeWire\ts16le 2ch 48000Hz\tRUNNING\n"
        with mock.patch.object(devices, "resolve_pipewire_target", return_value="alsa_output.demo"):
            with mock.patch("subprocess.run") as run:
                run.return_value.stdout = fake_sources
                self.assertEqual(
                    devices.resolve_speaker_monitor_name("alsa_output.demo"),
                    "alsa_output.demo.monitor",
                )

    def test_resolve_speaker_sink_uses_default_when_device_missing(self):
        with mock.patch.object(devices, "default_speaker_device", return_value="alsa_output.default"):
            with mock.patch.object(devices, "resolve_pipewire_target", return_value="alsa_output.default"):
                self.assertEqual(devices.resolve_speaker_sink(""), "alsa_output.default")

    def test_list_speakers_skips_internal_virtual_mic_sink(self):
        fake_nodes = [
            {
                "type": "PipeWire:Interface:Node",
                "id": 45,
                "info": {"props": {"node.name": "alsa_output.demo", "node.description": "Demo Speaker", "media.class": "Audio/Sink"}},
            },
            {
                "type": "PipeWire:Interface:Node",
                "id": 46,
                "info": {"props": {"node.name": "nvbroadcast_sink", "node.description": "nvbroadcast input", "media.class": "Audio/Sink/Virtual"}},
            },
        ]
        with mock.patch.object(devices, "_pw_nodes", return_value=fake_nodes):
            speakers = devices.list_speakers()

        self.assertEqual(speakers, [{"name": "Demo Speaker", "device": "alsa_output.demo"}])

    def test_list_speakers_dedupes_duplicate_entries(self):
        fake_nodes = [
            {
                "type": "PipeWire:Interface:Node",
                "id": 45,
                "info": {"props": {"node.name": "alsa_output.demo", "node.description": "Demo Speaker", "media.class": "Audio/Sink"}},
            },
            {
                "type": "PipeWire:Interface:Node",
                "id": 46,
                "info": {"props": {"node.name": "alsa_output.demo", "node.description": "Demo Speaker", "media.class": "Audio/Sink"}},
            },
        ]
        with mock.patch.object(devices, "_pw_nodes", return_value=fake_nodes):
            speakers = devices.list_speakers()

        self.assertEqual(speakers, [{"name": "Demo Speaker", "device": "alsa_output.demo"}])

    def test_list_microphones_skips_virtual_mic_and_dedupes(self):
        fake_nodes = [
            {
                "type": "PipeWire:Interface:Node",
                "id": 33,
                "info": {"props": {"node.name": "alsa_input.demo", "node.description": "Demo Mic", "media.class": "Audio/Source"}},
            },
            {
                "type": "PipeWire:Interface:Node",
                "id": 34,
                "info": {"props": {"node.name": "alsa_input.demo", "node.description": "Demo Mic", "media.class": "Audio/Source"}},
            },
            {
                "type": "PipeWire:Interface:Node",
                "id": 35,
                "info": {"props": {"node.name": "nvbroadcast_mic", "node.description": "nvbroadcast", "media.class": "Audio/Source"}},
            },
        ]
        with mock.patch.object(devices, "_pw_nodes", return_value=fake_nodes):
            mics = devices.list_microphones()

        self.assertEqual(mics, [{"name": "Demo Mic", "device": "alsa_input.demo"}])


if __name__ == "__main__":
    unittest.main()

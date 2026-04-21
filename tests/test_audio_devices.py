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


if __name__ == "__main__":
    unittest.main()

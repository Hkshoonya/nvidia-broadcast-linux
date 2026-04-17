import unittest
from unittest import mock

from nvbroadcast.audio.monitor import SpeakerMonitor


class SpeakerMonitorRoutingTests(unittest.TestCase):
    @mock.patch("nvbroadcast.audio.monitor.resolve_speaker_sink", return_value="alsa_output.demo")
    @mock.patch("nvbroadcast.audio.monitor.resolve_speaker_monitor_name",
                return_value="alsa_output.demo.monitor")
    @mock.patch("nvbroadcast.audio.monitor.Gst.ElementFactory.find")
    def test_prefers_pulse_backends_when_available(self, element_find, _monitor_name, _sink):
        element_find.side_effect = lambda name: object() if name in {"pulsesrc", "pulsesink"} else None

        monitor = SpeakerMonitor()
        monitor.configure(speaker_device="alsa_output.demo")

        self.assertEqual(
            monitor._select_capture_backend(),
            ("pulsesrc", "alsa_output.demo.monitor"),
        )
        self.assertEqual(
            monitor._select_output_backend(),
            ("pulsesink", "alsa_output.demo"),
        )

    @mock.patch("nvbroadcast.audio.monitor.resolve_speaker_sink", return_value="alsa_output.demo")
    @mock.patch("nvbroadcast.audio.monitor.resolve_speaker_monitor", return_value="228")
    @mock.patch("nvbroadcast.audio.monitor.Gst.ElementFactory.find")
    def test_falls_back_to_pipewire_when_pulse_missing(self, element_find, _monitor_id, _sink):
        element_find.side_effect = lambda name: object() if name in {"pipewiresrc", "pipewiresink"} else None

        monitor = SpeakerMonitor()
        monitor.configure(speaker_device="alsa_output.demo")

        self.assertEqual(
            monitor._select_capture_backend(),
            ("pipewiresrc", "228"),
        )
        self.assertEqual(
            monitor._select_output_backend(),
            ("pipewiresink", "alsa_output.demo"),
        )


if __name__ == "__main__":
    unittest.main()

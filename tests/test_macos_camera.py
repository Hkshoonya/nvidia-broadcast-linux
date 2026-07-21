import unittest
from unittest import mock

from nvbroadcast.core import platform
from nvbroadcast.video import virtual_camera


class _FakeStructure:
    def __init__(self, **values):
        self._values = values

    def has_field(self, field):
        return field in self._values

    def get_value(self, field):
        return self._values[field]


class _FakeDevice:
    def __init__(self, name, *, index=0, **properties):
        self._name = name
        self._index = index
        self._properties = _FakeStructure(**properties)

    def get_display_name(self):
        return self._name

    def get_properties(self):
        return self._properties

    def get_property(self, name):
        if name != "device-index":
            raise AttributeError(name)
        return self._index


class MacOSCameraTests(unittest.TestCase):
    def setUp(self):
        platform._avfvideosrc_supports_unique_id.cache_clear()

    def tearDown(self):
        platform._avfvideosrc_supports_unique_id.cache_clear()

    def test_avf_probe_uses_provider_indexes_and_skips_non_camera_sources(self):
        devices = [
            _FakeDevice(
                "OBS Virtual Camera",
                index=0,
                **{"device.api": "avf", "avf.unique_id": "obs-output"},
            ),
            _FakeDevice(
                "FaceTime HD Camera",
                index=1,
                **{"device.api": "avf", "avf.unique_id": "built-in"},
            ),
            _FakeDevice(
                "Desktop",
                index=-1,
                **{
                    "device.api": "avf",
                    "avf.unique_id": "screen-1",
                    "avf.capture_screen": True,
                },
            ),
            _FakeDevice(
                "PipeWire Camera",
                index=2,
                **{"device.api": "pipewire", "avf.unique_id": "other"},
            ),
        ]

        self.assertEqual(
            platform._avf_camera_details(devices),
            [
                {"name": "OBS Virtual Camera", "unique_id": "obs-output", "index": 0},
                {"name": "FaceTime HD Camera", "unique_id": "built-in", "index": 1},
            ],
        )

    def test_camera_listing_assigns_distinct_stable_identifiers(self):
        with mock.patch.object(
            platform,
            "_probe_avf_cameras",
            return_value=[
                {"name": "OBS Virtual Camera", "unique_id": "obs-output", "index": 0},
                {"name": "FaceTime HD Camera", "unique_id": "built-in", "index": 1},
            ],
        ):
            cameras = platform.list_cameras_macos()

        self.assertEqual(cameras[0]["device"], "avf:obs-output")
        self.assertEqual(cameras[0]["legacy_device"], "0")
        self.assertEqual(cameras[1]["device"], "avf:built-in")
        self.assertEqual(cameras[1]["legacy_device"], "1")
        self.assertNotEqual(cameras[0]["device"], cameras[1]["device"])

    def test_macos_input_listing_excludes_obs_virtual_camera(self):
        cameras = [
            {"name": "OBS Virtual Camera", "device": "avf:obs-output", "legacy_device": "0"},
            {"name": "FaceTime HD Camera", "device": "avf:built-in", "legacy_device": "1"},
        ]
        with mock.patch.object(virtual_camera, "IS_MACOS", True), mock.patch(
            "nvbroadcast.core.platform.list_cameras_macos",
            return_value=cameras,
        ):
            self.assertEqual(
                virtual_camera.list_camera_devices(),
                [cameras[1]],
            )

    def test_saved_obs_index_migrates_to_first_physical_camera(self):
        physical = {
            "name": "FaceTime HD Camera",
            "device": "avf:built-in",
            "legacy_device": "1",
        }
        with mock.patch.object(virtual_camera, "IS_MACOS", True), mock.patch.object(
            virtual_camera,
            "list_camera_devices",
            return_value=[physical],
        ):
            self.assertEqual(
                virtual_camera.resolve_camera_device("0"),
                "avf:built-in",
            )
            self.assertEqual(
                virtual_camera.resolve_camera_device("1"),
                "avf:built-in",
            )

    def test_current_gstreamer_maps_stable_id_to_avfoundation_index(self):
        device = platform._encode_macos_camera_device("built-in")
        with mock.patch.object(platform, "IS_MACOS", True), mock.patch.object(
            platform,
            "_avfvideosrc_supports_unique_id",
            return_value=False,
        ), mock.patch.object(
            platform,
            "_probe_avf_cameras",
            return_value=[
                {"name": "FaceTime HD Camera", "unique_id": "built-in", "index": 1},
            ],
        ):
            source = platform.get_gst_camera_caps(device, 1280, 720, 30, "raw")

        self.assertEqual(
            source,
            "avfvideosrc device-index=1 ! videoconvert ! videoscale ! "
            "videorate ! video/x-raw,width=1280,height=720,framerate=30/1",
        )

    def test_new_gstreamer_uses_escaped_stable_id(self):
        device = platform._encode_macos_camera_device('camera:"built-in"')
        with mock.patch.object(platform, "IS_MACOS", True), mock.patch.object(
            platform,
            "_avfvideosrc_supports_unique_id",
            return_value=True,
        ):
            source = platform.get_gst_camera_caps(device, 640, 480, 30, "raw")

        self.assertIn('avfvideosrc unique-id="camera:\\"built-in\\""', source)

    def test_missing_stable_camera_does_not_fall_back_to_default_source(self):
        device = platform._encode_macos_camera_device("unplugged")
        with mock.patch.object(platform, "IS_MACOS", True), mock.patch.object(
            platform,
            "_avfvideosrc_supports_unique_id",
            return_value=False,
        ), mock.patch.object(platform, "_probe_avf_cameras", return_value=[]):
            with self.assertRaisesRegex(ValueError, "no longer available"):
                platform.get_gst_camera_caps(device, 1280, 720, 30, "raw")


if __name__ == "__main__":
    unittest.main()

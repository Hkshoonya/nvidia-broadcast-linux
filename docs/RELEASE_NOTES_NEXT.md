# Next Release Notes

Use this file as the working checklist for the next version update.

## Included Changes

- Configurable virtual camera output device for Issue #22:
  - Persisted `video.vcam_device` in the user config.
  - Added a Camera section output-device field in the GTK app.
  - Made `nvbroadcast-vcam --vcam` and the config value feed the headless service.
  - Updated virtual-camera setup and recovery messages to use the selected `/dev/videoN`.
  - Added source/setup installer overrides with `NVBROADCAST_VCAM_DEVICE_NUM` or `NVBROADCAST_VCAM_DEVICE`.
  - Added unit coverage for config persistence, reset preservation, device selection, reset commands, and headless selection rules.

## Verification To Repeat Before Release

- Run the config and virtual-camera tests.
- Smoke-test the GUI with the default `/dev/video10`.
- Smoke-test a non-default output device such as `/dev/video11` when available.
- Recheck README and package notes before cutting the version changelog.

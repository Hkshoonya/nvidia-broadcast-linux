# NV Broadcast v1.2.1

This patch makes Eye Contact visibly useful at normal settings and carries the
corrected macOS OBS Virtual Camera output from v1.2.0 to users upgrading from
older builds.

## Eye contact correction

- Moderate off-center gaze now moves visibly toward the eye center at the
  persisted default intensity instead of often becoming a sub-pixel no-op.
- Horizontal and vertical correction are both applied within the detected
  eyelid contour.
- The eye region is feathered at its boundary so eyelids, glasses, and nearby
  skin remain anchored.
- Gaze movement is smoothed between frames to reduce landmark jitter.
- Blinks, very small detections, and unstable iris offsets are rejected before
  correction.
- The frame alpha channel is preserved for background removal and virtual
  camera output.

## macOS OBS output

The reporter was using v1.1.13, whose experimental fallback could pass an
undeclared, non-contiguous BGRA slice to OBS. The supported path introduced in
v1.2.0 and included here opens OBS Virtual Camera explicitly in BGR mode and
sends contiguous BGR frames in both passthrough and effects modes.

macOS release qualification remains Apple Silicon, macOS 13 or newer, Python
3.11-3.13, and OBS Studio. Open OBS once after installing or upgrading it,
start and stop **Virtual Camera**, close OBS, and then select **OBS Virtual
Camera** in the meeting application. Intel macOS remains excluded because no
secure current MediaPipe wheel is available for that architecture.

## Reporter credit

Thanks to [@13v13reddy](https://github.com/13v13reddy) for reporting the Apple
M3 Pro behavior in [Issue #31](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/31).

# NV Broadcast v1.2.2

This patch follows the Apple M3 Pro validation of v1.2.1. Eye Contact is now
visibly effective at normal intensity; v1.2.2 corrects the remaining inward
convergence and tracking-latency behavior reported during that validation.

## Shared binocular targeting

- Both eyes now derive correction from one normalized, camera-facing gaze
  direction instead of centering and smoothing each iris independently.
- Separate target positions add a small outward bias and preserve most of the
  person's measured binocular disparity, avoiding a cross-eyed result.
- Conservative head-yaw compensation adjusts the shared target when the face
  is not perfectly frontal.
- Extreme side gaze fades through a smooth falloff before correction stops,
  avoiding an abrupt boundary or an over-corrected eye warp.
- If either eye is blinking, too small, or unstable, neither eye is corrected
  for that frame.

## Lower tracking latency

- Coordinated eye movement follows the new correction faster than v1.2.1.
- A changing left/right disparity is treated as landmark instability and keeps
  stronger smoothing so one noisy iris cannot pull the pair.
- When Eye Contact is the only active face effect, the live pipeline requests
  fresh landmarks every frame instead of deliberately reusing each result.
- The frame alpha channel remains unchanged for background effects and virtual
  camera output.

## Validation

- Deterministic regressions cover natural iris spacing, shared side gaze,
  head yaw, coordinated-motion latency, one-eye landmark jumps, blinks,
  extreme gaze, zero intensity, and alpha preservation.
- The full local suite passes on the release tree. Linux amd64/arm64 and the
  supported Apple Silicon Python 3.11-3.13 matrix are release gates.
- The supported macOS output remains OBS Virtual Camera using contiguous BGR
  frames. Intel macOS remains excluded because no secure current MediaPipe
  wheel is available for that architecture.

## Reporter credit

Thanks to [@13v13reddy](https://github.com/13v13reddy) for testing v1.2.1 on an
Apple M3 Pro, confirming the visibility fix, and providing the shared-target
test case in [Issue #31](https://github.com/Hkshoonya/nvidia-broadcast-linux/issues/31).

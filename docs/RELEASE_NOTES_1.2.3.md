# NV Broadcast v1.2.3

This patch improves background blur and replacement behavior in strongly
backlit scenes. The change is deliberately limited to segmentation: it helps
the RVM model recover foreground detail from deep shadow without altering the
camera image shown to the user.

## Adaptive backlight matting

- Quality and Ultra modes now classify scenes using shadow, median, and
  highlight levels sampled from the incoming frame.
- Adaptation activates only when strong highlights, deep shadows, and enough
  scene contrast are present together. Uniformly dark, uniformly bright, and
  normally balanced scenes remain neutral.
- A bounded gamma lift is applied to a private RGB copy used for RVM inference.
  The original camera pixels, replacement image, and final output colors are
  not modified by the adaptation.
- Attack and recovery are smoothed across frames to avoid abrupt matte changes
  when the subject or sunlight moves.
- Performance mode bypasses the additional work; its existing speed-oriented
  behavior is unchanged.

## Validation

- Fresh webcam captures were compared frame by frame with and without the
  change under difficult window backlight.
- The tested sequence showed about 5% fewer small edge-fragment changes and
  about 5% less foreground-area variation, with slightly lower frame-to-frame
  matte movement overall.
- The added estimator and lookup-table work measured under 0.6 ms per frame at
  1280x720 on the validation system.
- Deterministic regressions cover triggering, dark/mid/bright bypass,
  smoothed recovery, performance-mode bypass, and preservation of source
  pixels.
- Integration tests now use temporary config and image paths so release checks
  cannot overwrite a user's persisted settings or background selection.

## Scope

This release does not change resolution selection, model resolution, or the
existing full-pipeline performance cost at higher output resolutions. It also
does not enable TensorRT on Python 3.14: the currently available ONNX Runtime
wheel still links against TensorRT 10 while Python 3.14 TensorRT packages use
TensorRT 11, so exposing that mode would currently allow a silent CUDA
fallback rather than verified TensorRT execution.

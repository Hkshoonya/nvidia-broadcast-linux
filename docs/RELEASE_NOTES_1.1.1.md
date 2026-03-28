# NV Broadcast 1.1.1

Release date: 2026-03-28

## Highlights

- Stabilized the Linux virtual-camera sink path so loopback startup is more reliable.
- Reduced live video lag in heavier face-effect stacks by reusing landmarks and restricting relighting to the face ROI.
- Improved replace-mode edges around shoulders, hair, ears, and under-arm gaps.
- Made local meeting transcription start more reliably with shorter chunks and correct saved meeting audio.
- Changed live resolution switching to a safe save-and-apply-after-restart flow instead of risking a stream hang.

## Patch posture

- `1.1.1` is a patch release on top of `1.1.0`.
- It focuses on runtime stability, lower perceived lag, and better live composite quality.
- `DocZeus` remains the recommended default quality mode.
- The learned neural matte refiner remains experimental and disabled by default.

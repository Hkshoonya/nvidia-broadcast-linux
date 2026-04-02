# NV Broadcast 1.1.2

Release date: 2026-04-02

## Fixes First

- Improved final meeting transcript quality with better chunk cleanup and a stronger full-audio final pass.
- Moved meeting transcript, notes, and summary finalization off the UI thread so ending a meeting no longer hangs the app.
- Persisted speaker selection and active profile state across restarts.
- Added `Reset to Defaults` so users can get back to a known-good setup quickly.
- Fixed microphone test recording and playback reliability.
- Expanded microphone test capture choices to `30s`, `45s`, and `60s`, with early stop support.

## Upgrade Recommendation

- `1.1.2` is the recommended stable update on top of `1.1.1`.
- Users still on `1.1.0` or `1.1.1` should upgrade to `1.1.2` for the current meeting-quality and settings-persistence fixes.
- Resolution changes still save immediately and should be applied after stopping and starting the app again.

## Why It Is Better

- Meeting notes now start from a cleaner final transcript instead of depending only on low-latency chunk output.
- Finishing a meeting is more reliable because transcript/note generation no longer blocks the GTK thread.
- Users no longer lose speaker routing or active profile selection across launches.
- Mic test is now usable for real setup checks instead of only a short fragile recording.

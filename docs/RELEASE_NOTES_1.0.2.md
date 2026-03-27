# NV Broadcast 1.0.2

Release date: 2026-03-27

## Highlights

- Improved background replacement and green-screen quality with tighter matte handling.
- Fixed premium mode persistence and startup restore behavior.
- Packaged CSS and desktop assets are now included in built wheels.
- Standard installer now performs a regular install instead of an editable install.
- Added a repeatable local release smoke check for compile, tests, and wheel asset validation.
- Package upgrades are now treated as in-place updates instead of uninstall-like removals.
- Added startup release checks so installed users see when a newer GitHub release is available.

## Release posture

- `DocZeus` remains the recommended default quality mode.
- The learned neural matte refiner remains experimental and is still disabled by default.

# Flatpak controls screenshots

Captured 9 October 2026 from the actual `NVBroadcastWindow` and project CSS on an
isolated Xvfb Linux display using GTK's `WidgetPaintable` and GSK rendering.
The desktop size requests were 1000×700 and 640×700; the actual GTK window
content dimensions are 990×690 and 630×690. The PNGs are direct, unedited
window renders including their header and decorations.

Only UI construction was exercised. Camera/microphone discovery, capability
probes, saved profiles, and periodic hardware work were disabled in the capture
harness. Empty device selectors are intentional. The app is idle with preview
hidden; no personal camera frames, speech, desktop, or fabricated camera output
were captured. The default controls screenshot shows Camera and Audio side by
side; the second shows Audio selected in the compact layout.

These screenshots document the current controls. They do not establish a
working camera, audio device, AI effect, or Wayland desktop session. Final
hardware acceptance remains separately recorded in issue #95.

AppStream references immutable GitHub raw URLs at source commit
`3e7bb7aa20b18a698a9a60630689abb0a8fd5b9f`. This lets PR checks fetch the exact
reviewed PNGs before the metadata change deploys to the website, and prevents a
later website edit from silently changing the listed screenshots. The same files
are available through GitHub Pages after merge; publishing screenshots does not
publish or qualify the Flatpak itself.

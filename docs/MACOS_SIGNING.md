# macOS package signing and acceptance

The installer declares Apple Silicon and macOS 13 as its minimum platform.
The current Homebrew runtime setup is supported on Apple Silicon macOS
15 or newer; the installed-runtime CI qualification uses macOS 15.7.9.
Homebrew classifies macOS 13 and 14 as unsupported Tier 3 configurations, so
the package's minimum declaration does not qualify current dependency setup
on those systems. See Homebrew's [installation requirements](https://docs.brew.sh/Installation)
and [support tiers](https://docs.brew.sh/Support-Tiers).

The package contains Python application source, resources and launch/setup scripts. The
distribution identity is **Developer ID Installer**. The separate proprietary
Camera Extension prototype is outside this package; supported virtual-camera
output continues through OBS.

## Create the installer identity

Use your Mac to keep the private key in your own Keychain:

1. Open Keychain Access and choose **Certificate Assistant → Request a
   Certificate From a Certificate Authority**. Enter your Apple Developer
   email and a descriptive common name. Save the CSR to disk.
2. As the Apple Developer account holder, open
   [Certificates](https://developer.apple.com/account/resources/certificates/list),
   create a **Developer ID Installer** certificate, and upload that CSR.
   Download the issued certificate and open it on the same Mac.
3. In Keychain Access → **My Certificates**, confirm the Installer certificate
   has its private key beneath it. Export that identity as a password-protected
   `.p12`. Keep the file and password outside the repository and chat.
4. Create an app-specific password at [Apple Account](https://account.apple.com).
   This is the notarization credential. Record your ten-character Team ID from
   Apple Developer membership details.

Apple's current instructions cover
[CSR creation](https://developer.apple.com/help/account/certificates/create-a-certificate-signing-request),
[Developer ID certificates](https://developer.apple.com/help/account/certificates/create-developer-id-certificates)
and [notarization](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow).

## Configure GitHub environment secrets

Open the repository's
[environments settings](https://github.com/Hkshoonya/nvidia-broadcast-linux/settings/environments)
and select **macos-signing**. Add these as environment secrets:

| Name | Value |
| --- | --- |
| `MACOS_INSTALLER_CERTIFICATE_P12_BASE64` | Base64 of the exported Installer certificate and private key |
| `MACOS_INSTALLER_CERTIFICATE_PASSWORD` | Password protecting that `.p12` |
| `MACOS_NOTARY_APPLE_ID` | Apple Account email used for notarization |
| `MACOS_NOTARY_APP_PASSWORD` | Its Apple app-specific password |
| `MACOS_TEAM_ID` | Ten-character Apple Developer Team ID |

To prepare the first value on your Mac without printing it:

```bash
base64 -i "$HOME/Downloads/NVBroadcast-Developer-ID-Installer.p12" | pbcopy
```

Paste the clipboard only into the corresponding GitHub environment secret.
Clear the clipboard afterward. If you use GitHub CLI, pipe the encoded file
directly into `gh secret set` instead:

```bash
base64 -i "$HOME/Downloads/NVBroadcast-Developer-ID-Installer.p12" |
  gh secret set MACOS_INSTALLER_CERTIFICATE_P12_BASE64 \
    --repo Hkshoonya/nvidia-broadcast-linux --env macos-signing
```

Set deployment branch/tag policies before adding secrets. For the current
candidate work, allowed branches are `main`, `release/1.5.3-preparation` and
`fix/macos-signing-20261007`; allowed tags are `v*`. Remove the temporary fix
branch policy after integration. Do not broaden this to all branches or expose
signing secrets to pull-request jobs. GitHub documents
[environment restrictions](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)
and [certificate handling on runners](https://docs.github.com/en/actions/how-tos/deploy/deploy-to-third-party-platforms/sign-xcode-applications).

## Run and verify

Run **Build Packages** on the reviewed candidate with `sign_macos=true` for a
manual signing test. An ordinary manual run leaves signing disabled. A `v*`
tag requires signing; missing credentials or failed gates stop package
attestation and draft release creation.

The signing job uses this run's exact unsigned package and an isolated
temporary keychain. It verifies the certificate's Installer common name and
Team ID before using the identity. The orchestration then requires:

1. Timestamped signing and a trusted Installer signature from the expected team.
2. Identical expanded payload contents, modes and symlink targets.
3. Accepted notarization, a matching submission log and archive identity.
4. Ticket stapling, ticket validation and Gatekeeper's Notarized Developer ID acceptance.
5. Repeated signature/payload verification and a checksum of the final stapled bytes.

The final output is created only after all gates pass. Failure preserves
diagnostics without creating a release package. Cleanup runs even after an
import or keychain-deletion failure. Secret material stays in private runner
temporary storage and is excluded from artifacts.

Download **macos-signed-packages** and **macos-signing-evidence** from the same
successful run. `summary.json` and `evidence-manifest.json` bind the final `.pkg`
SHA-256. Only signed package artifacts enter tag checksums, attestations and
the draft release. Account approval, mocked signing tests and unsigned builds
do not establish completed signing/notarization.

## Test the signed installer on your Mac

Use a regular user on an Apple Silicon Mac with macOS 15+ and the supported
Homebrew runtime stack. macOS 13/14 runtime compatibility remains unqualified.
Verify the downloaded
package against the final evidence checksum, then run:

```bash
pkgutil --check-signature "$HOME/Downloads/NVBroadcast-1.5.3-1.pkg"
xcrun stapler validate "$HOME/Downloads/NVBroadcast-1.5.3-1.pkg"
spctl --assess --type install --verbose=4 "$HOME/Downloads/NVBroadcast-1.5.3-1.pkg"
```

Gatekeeper should accept the notarized Developer ID package. Open the package
normally in Installer. Keep Gatekeeper enabled. The package installs its
admin-owned source; runtime preparation is a separate user step:

```bash
/opt/homebrew/bin/brew install python@3.13 pygobject3 gtk4 libadwaita gstreamer
/opt/nvbroadcast/scripts/setup_macos_runtime.sh
/usr/local/bin/nvbroadcast
```

Run those commands without sudo. Setup checks Python/GI ABI and required native
plugins, provisions a user runtime tied to the exact package source, and marks
it ready only after validation. Old runtime generations are retained. Unsafe
or redirected package destinations require administrator review; Installer
does not change ownership/ACLs or execute Homebrew/Python as root.

Use `/usr/local/bin/nvbroadcast` explicitly during acceptance so a previous
source-install launcher does not shadow this package. Check:

- Camera permission, physical camera discovery, Start/Stop and compact preview access.
- Blur/Remove/Replace and motion at a supported Mac processing level; record quality/FPS.
- OBS Virtual Camera and a consumer application, including stop/reconnect.
- Microphone permission, built-in and selected external microphones, and Mic Test playback.
- Rec: speak, stop, locate the MP4, and play its complete video and audible speech.
- Meeting: selected microphone capture and completion. macOS Meeting captures the microphone;
  speaker/system loopback capture and exported processed virtual microphone are unavailable.
- Quit/relaunch and an upgrade. Identify the exact source/runtime generation and retain prior
  user recordings/configuration.

The installed-runtime CI job verifies real package installation, GI/native
imports, sole CPU ONNX ownership, numerical inference and a fully decoded
generated audiovisual recording. GitHub's disposable runner has a user-owned,
group-writable `/usr/local/bin`; the job prepares that directory as root-owned
and mode 755 before testing the package's supported destination profile. It
retains access metadata before and after preparation. This is a CI fixture;
the production installer continues to reject unsafe existing destinations.
Its recording sources are `videotestsrc` and
`audiotestsrc`; it does not request physical camera/microphone access. Physical
permissions, real speech, OBS output, effects and long-session behavior remain
the Mac acceptance above.

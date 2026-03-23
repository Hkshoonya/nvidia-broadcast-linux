// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Proprietary license — see macos/LICENSE
//
// Entry point for the NV Broadcast virtual camera system extension.
// macOS discovers this extension and presents it as a camera source
// in Zoom, FaceTime, Chrome, OBS, Discord, etc.

import Foundation
import CoreMediaIO

let providerSource = NVBroadcastProvider()
CMIOExtensionProvider.startService(provider: providerSource.provider)
CFRunLoopRun()

// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Licensed under GPL-3.0
//
// Entry point for the NV Broadcast virtual camera system extension.
// macOS discovers this extension and presents it as a camera source
// in Zoom, FaceTime, Chrome, OBS, Discord, etc.

import Foundation
import CoreMediaIO

let provider = NVBroadcastProvider()
CMIOExtensionProvider.startService(provider: provider)
RunLoop.current.run()

// NV Broadcast - Proprietary macOS Virtual Camera
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Licensed under GPL-3.0

import Foundation

enum NVBroadcastConstants {
    static let extensionBundleID = "com.doczeus.nvbroadcast.camera"
    static let appGroupID = "group.com.doczeus.nvbroadcast"
    static let deviceName = "NV Broadcast"
    static let deviceModel = "NV Broadcast Virtual Camera"
    static let manufacturer = "Doczeus"

    // Shared memory / IPC
    static let frameNotificationName = "com.doczeus.nvbroadcast.newframe"
    static let controlNotificationName = "com.doczeus.nvbroadcast.control"
    static let sharedMemoryName = "nvbroadcast_frame"

    // Frame defaults
    static let defaultWidth: Int32 = 1920
    static let defaultHeight: Int32 = 1080
    static let defaultFPS: Int32 = 30
    static let pixelFormat: UInt32 = 0x42475241  // 'BGRA'
}

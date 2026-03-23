// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Licensed under GPL-3.0
//
// CMIOExtensionProvider — top-level provider that manages all NV Broadcast
// virtual camera devices. macOS calls into this to discover available cameras.

import Foundation
import CoreMediaIO

class NVBroadcastProvider: NSObject, CMIOExtensionProviderSource {

    private var _device: NVBroadcastDevice!
    private var _cmioDevice: CMIOExtensionDevice!

    override init() {
        super.init()
        let deviceID = UUID()
        _device = NVBroadcastDevice()
        _cmioDevice = CMIOExtensionDevice(
            localizedName: NVBroadcastConstants.deviceName,
            deviceID: deviceID,
            legacyDeviceID: nil,
            source: _device
        )
    }

    // MARK: - CMIOExtensionProviderSource

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.providerManufacturer]
    }

    func providerProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionProviderProperties {
        let properties = CMIOExtensionProviderProperties(dictionary: [:])
        properties.setPropertyState(
            CMIOExtensionPropertyState(value: NVBroadcastConstants.manufacturer as NSString),
            forProperty: .providerManufacturer
        )
        return properties
    }

    func setProviderProperties(
        _ providerProperties: CMIOExtensionProviderProperties
    ) throws {
        // Read-only provider
    }

    func connect(to client: CMIOExtensionClient) throws -> CMIOExtensionDevice {
        return _cmioDevice
    }

    func disconnect(from client: CMIOExtensionClient) {
        // Cleanup if needed
    }
}

// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Proprietary license — see macos/LICENSE
//
// CMIOExtensionProvider — top-level provider that manages all NV Broadcast
// virtual camera devices. macOS calls into this to discover available cameras.

import Foundation
import CoreMediaIO

class NVBroadcastProvider: NSObject, CMIOExtensionProviderSource {

    private(set) var provider: CMIOExtensionProvider!
    private var _device: NVBroadcastDevice!
    private var _cmioDevice: CMIOExtensionDevice!

    override init() {
        super.init()

        provider = CMIOExtensionProvider(source: self, clientQueue: nil)

        let deviceID = UUID()
        _device = NVBroadcastDevice(provider: provider)
        _cmioDevice = CMIOExtensionDevice(
            localizedName: NVBroadcastConstants.deviceName,
            deviceID: deviceID,
            legacyDeviceID: nil,
            source: _device
        )

        do {
            try provider.addDevice(_cmioDevice)
        } catch {
            fatalError("Failed to add device: \(error)")
        }
    }

    // MARK: - CMIOExtensionProviderSource

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.providerManufacturer]
    }

    func providerProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionProviderProperties {
        let providerProps = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) {
            providerProps.setPropertyState(
                CMIOExtensionPropertyState(value: NVBroadcastConstants.manufacturer as NSString),
                forProperty: .providerManufacturer
            )
        }
        return providerProps
    }

    func setProviderProperties(
        _ providerProperties: CMIOExtensionProviderProperties
    ) throws {
        // Read-only provider
    }
}

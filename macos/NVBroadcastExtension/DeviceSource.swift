// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Proprietary license — see macos/LICENSE
//
// CMIOExtensionDevice — represents the "NV Broadcast" virtual camera device.
// Manages the video stream and device properties visible to apps.

import Foundation
import CoreMediaIO

class NVBroadcastDevice: NSObject, CMIOExtensionDeviceSource {

    private(set) var device: CMIOExtensionDevice!
    private var _streamSource: NVBroadcastStream!
    private var _cmioStream: CMIOExtensionStream!
    private weak var _provider: CMIOExtensionProvider?

    init(provider: CMIOExtensionProvider) {
        _provider = provider
        super.init()
    }

    func setupStream(for device: CMIOExtensionDevice) {
        self.device = device
        let streamID = UUID()
        _streamSource = NVBroadcastStream()
        _cmioStream = CMIOExtensionStream(
            localizedName: "NV Broadcast Video",
            streamID: streamID,
            direction: .source,
            clockType: .hostTimeClock,
            source: _streamSource
        )

        do {
            try device.addStream(_cmioStream)
        } catch {
            print("[NV Broadcast] Failed to add stream: \(error)")
        }
    }

    // MARK: - CMIOExtensionDeviceSource

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.deviceTransportType, .deviceModel]
    }

    func deviceProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionDeviceProperties {
        let props = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            props.setPropertyState(
                CMIOExtensionPropertyState(value: NSNumber(value: 2)),
                forProperty: .deviceTransportType
            )
        }
        if properties.contains(.deviceModel) {
            props.setPropertyState(
                CMIOExtensionPropertyState(value: NVBroadcastConstants.deviceModel as NSString),
                forProperty: .deviceModel
            )
        }
        return props
    }

    func setDeviceProperties(
        _ deviceProperties: CMIOExtensionDeviceProperties
    ) throws {
        // Read-only device
    }
}

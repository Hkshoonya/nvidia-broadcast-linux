// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Licensed under GPL-3.0
//
// CMIOExtensionDevice — represents the "NV Broadcast" virtual camera device.
// Manages the video stream and device properties visible to apps.

import Foundation
import CoreMediaIO

class NVBroadcastDevice: NSObject, CMIOExtensionDeviceSource {

    private var _streamSource: NVBroadcastStream!
    private var _cmioStream: CMIOExtensionStream!

    override init() {
        super.init()
        let streamID = UUID()
        _streamSource = NVBroadcastStream()
        _cmioStream = CMIOExtensionStream(
            localizedName: "NV Broadcast",
            streamID: streamID,
            direction: .source,
            clockType: .hostTimeClock,
            source: _streamSource
        )
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
            // Virtual device transport type
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

    func streams() -> [CMIOExtensionStream] {
        return [_cmioStream]
    }
}

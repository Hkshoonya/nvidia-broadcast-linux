// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Proprietary license — see macos/LICENSE
//
// CMIOExtensionStream — delivers processed frames to video apps.
// Reads frames from a shared file written by the Python app.

import Foundation
import CoreMediaIO
import CoreVideo

class NVBroadcastStream: NSObject, CMIOExtensionStreamSource {

    private var _isStreaming = false
    private var _sequenceNumber: UInt64 = 0
    private var _timer: DispatchSourceTimer?

    private var _pixelBufferPool: CVPixelBufferPool?
    private var _currentPixelBuffer: CVPixelBuffer?
    private let _bufferLock = NSLock()

    private let _width = Int(NVBroadcastConstants.defaultWidth)
    private let _height = Int(NVBroadcastConstants.defaultHeight)
    private let _fps = Int(NVBroadcastConstants.defaultFPS)

    private var _streamFormat: CMIOExtensionStreamFormat!
    private var _lastSequence: UInt64 = 0

    // Shared frame file path (written by Python, read by extension)
    private let _frameFilePath: String = {
        let tmpDir = NSTemporaryDirectory()
        return (tmpDir as NSString).appendingPathComponent("nvbroadcast_frame.raw")
    }()

    override init() {
        super.init()
        _setupPixelBufferPool()
        _createStreamFormat()
    }

    // MARK: - Setup

    private func _setupPixelBufferPool() {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: _width,
            kCVPixelBufferHeightKey as String: _height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &_pixelBufferPool)
    }

    private func _createStreamFormat() {
        var formatDesc: CMFormatDescription?
        CMVideoFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            codecType: kCVPixelFormatType_32BGRA,
            width: Int32(_width),
            height: Int32(_height),
            extensions: nil,
            formatDescriptionOut: &formatDesc
        )
        guard let fmt = formatDesc else { fatalError("Failed to create format description") }
        _streamFormat = CMIOExtensionStreamFormat(
            formatDescription: fmt,
            maxFrameDuration: CMTime(value: 1, timescale: CMTimeScale(_fps)),
            minFrameDuration: CMTime(value: 1, timescale: CMTimeScale(_fps)),
            validFrameDurations: nil
        )
    }

    // MARK: - Frame reading from shared file

    private func _readFrameFromFile() {
        let headerSize = 16  // width(4) + height(4) + sequence(8)
        let frameSize = _width * _height * 4
        let totalSize = headerSize + frameSize

        guard let data = try? Data(contentsOf: URL(fileURLWithPath: _frameFilePath)),
              data.count >= totalSize else { return }

        // Read sequence number from header (bytes 8-16, little-endian UInt64)
        let sequence = data.withUnsafeBytes { ptr -> UInt64 in
            ptr.load(fromByteOffset: 8, as: UInt64.self)
        }

        // Skip if we already processed this frame
        guard sequence > _lastSequence else { return }
        _lastSequence = sequence

        _bufferLock.lock()
        defer { _bufferLock.unlock() }

        guard let pool = _pixelBufferPool else { return }
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
        guard let pb = pixelBuffer else { return }

        CVPixelBufferLockBaseAddress(pb, [])
        if let dest = CVPixelBufferGetBaseAddress(pb) {
            data.withUnsafeBytes { ptr in
                _ = memcpy(dest, ptr.baseAddress!.advanced(by: headerSize), frameSize)
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        _currentPixelBuffer = pb
    }

    // MARK: - Frame delivery

    private func _startFrameDelivery() {
        let interval = 1.0 / Double(_fps)
        _timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInteractive))
        _timer?.schedule(deadline: .now(), repeating: interval)
        _timer?.setEventHandler { [weak self] in self?._deliverFrame() }
        _timer?.resume()
    }

    private func _stopFrameDelivery() {
        _timer?.cancel()
        _timer = nil
    }

    private func _deliverFrame() {
        guard _isStreaming else { return }

        // Poll for new frames from the Python app
        _readFrameFromFile()

        _bufferLock.lock()
        let pb = _currentPixelBuffer ?? _makeBlackFrame()
        _bufferLock.unlock()

        guard let pixelBuffer = pb else { return }

        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: CMTimeScale(_fps)),
            presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
            decodeTimeStamp: .invalid
        )
        var formatDescription: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDescription
        )
        guard let fmt = formatDescription else { return }

        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: fmt,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        )
        guard sampleBuffer != nil else { return }
        _sequenceNumber += 1
    }

    private func _makeBlackFrame() -> CVPixelBuffer? {
        guard let pool = _pixelBufferPool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        return pb
    }

    // MARK: - CMIOExtensionStreamSource

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.streamActiveFormatIndex]
    }

    func streamProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionStreamProperties {
        let props = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) {
            props.setPropertyState(
                CMIOExtensionPropertyState(value: NSNumber(value: 0)),
                forProperty: .streamActiveFormatIndex
            )
        }
        return props
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}
    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { return true }

    func startStream() throws {
        _isStreaming = true
        _sequenceNumber = 0
        _lastSequence = 0
        _startFrameDelivery()
    }

    func stopStream() throws {
        _isStreaming = false
        _stopFrameDelivery()
    }

    var formats: [CMIOExtensionStreamFormat] { return [_streamFormat] }
}

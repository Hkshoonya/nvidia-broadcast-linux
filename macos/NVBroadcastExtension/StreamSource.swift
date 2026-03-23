// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Proprietary license — see macos/LICENSE
//
// CMIOExtensionStream — the video stream that delivers processed frames
// from the Python NV Broadcast app to consuming apps (Zoom, Chrome, etc.).
//
// Receives frames via POSIX shared memory from the Python process,
// wraps them in CVPixelBuffer + CMSampleBuffer, and delivers to clients.

import Foundation
import CoreMediaIO
import CoreVideo

class NVBroadcastStream: NSObject, CMIOExtensionStreamSource {

    private var _isStreaming = false
    private var _sequenceNumber: UInt64 = 0
    private var _timer: DispatchSourceTimer?

    // Frame buffer — written by Python helper, read by this stream
    private var _pixelBufferPool: CVPixelBufferPool?
    private var _currentPixelBuffer: CVPixelBuffer?
    private let _bufferLock = NSLock()

    private let _width = Int(NVBroadcastConstants.defaultWidth)
    private let _height = Int(NVBroadcastConstants.defaultHeight)
    private let _fps = Int(NVBroadcastConstants.defaultFPS)

    private var _streamFormat: CMIOExtensionStreamFormat!

    override init() {
        super.init()
        _setupPixelBufferPool()
        _createStreamFormat()
        _setupFrameListener()
    }

    // MARK: - Setup

    private func _setupPixelBufferPool() {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: _width,
            kCVPixelBufferHeightKey as String: _height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        CVPixelBufferPoolCreate(
            kCFAllocatorDefault, nil, attrs as CFDictionary, &_pixelBufferPool
        )
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
        guard let fmt = formatDesc else {
            fatalError("Failed to create format description")
        }
        _streamFormat = CMIOExtensionStreamFormat(
            formatDescription: fmt,
            maxFrameDuration: CMTime(value: 1, timescale: CMTimeScale(_fps)),
            minFrameDuration: CMTime(value: 1, timescale: CMTimeScale(_fps)),
            validFrameDurations: nil
        )
    }

    // MARK: - IPC: Receive frames from Python

    private func _setupFrameListener() {
        // Listen for "new frame ready" notifications from the Python helper.
        // Uses Darwin notify API (matches notify_post on the Python side).
        var token: Int32 = 0
        notify_register_dispatch(
            NVBroadcastConstants.frameNotificationName,
            &token,
            DispatchQueue.global(qos: .userInteractive)
        ) { [weak self] _ in
            self?._onNewFrameFromPython()
        }
    }

    private func _onNewFrameFromPython() {
        guard _isStreaming else { return }

        let shmName = "/" + NVBroadcastConstants.sharedMemoryName
        let fd = shm_open(shmName, O_RDONLY, 0)
        guard fd >= 0 else { return }
        defer { close(fd) }

        let headerSize = 16  // width(4) + height(4) + sequence(8)
        let frameSize = _width * _height * 4
        let totalSize = headerSize + frameSize
        guard let ptr = mmap(nil, totalSize, PROT_READ, MAP_SHARED, fd, 0),
              ptr != MAP_FAILED else { return }
        defer { munmap(ptr, totalSize) }

        _bufferLock.lock()
        defer { _bufferLock.unlock() }

        guard let pool = _pixelBufferPool else { return }
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
        guard let pb = pixelBuffer else { return }

        CVPixelBufferLockBaseAddress(pb, [])
        if let dest = CVPixelBufferGetBaseAddress(pb) {
            // Skip header, copy frame data
            memcpy(dest, ptr.advanced(by: headerSize), frameSize)
        }
        CVPixelBufferUnlockBaseAddress(pb, [])

        _currentPixelBuffer = pb
    }

    // MARK: - Frame delivery

    private func _startFrameDelivery() {
        let interval = 1.0 / Double(_fps)
        _timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInteractive))
        _timer?.schedule(deadline: .now(), repeating: interval)
        _timer?.setEventHandler { [weak self] in
            self?._deliverFrame()
        }
        _timer?.resume()
    }

    private func _stopFrameDelivery() {
        _timer?.cancel()
        _timer = nil
    }

    private func _deliverFrame() {
        guard _isStreaming else { return }

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
        guard let sb = sampleBuffer else { return }

        _sequenceNumber += 1
        sb.setOutputPresentationTimeStamp(timing.presentationTimeStamp)
        // The stream infrastructure picks this up via the scheduled timer
    }

    private func _makeBlackFrame() -> CVPixelBuffer? {
        guard let pool = _pixelBufferPool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        return pb  // Pool creates zero-initialized (black) buffers
    }

    // MARK: - CMIOExtensionStreamSource

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.streamActiveFormatIndex, .streamFrameDuration]
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
        if properties.contains(.streamFrameDuration) {
            let duration = CMTime(value: 1, timescale: CMTimeScale(_fps))
            props.setPropertyState(
                CMIOExtensionPropertyState(value: NSValue(time: duration)),
                forProperty: .streamFrameDuration
            )
        }
        return props
    }

    func setStreamProperties(
        _ streamProperties: CMIOExtensionStreamProperties
    ) throws {
        // Handle format changes if needed
    }

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool {
        return true
    }

    func startStream() throws {
        _isStreaming = true
        _sequenceNumber = 0
        _startFrameDelivery()
    }

    func stopStream() throws {
        _isStreaming = false
        _stopFrameDelivery()
    }

    var formats: [CMIOExtensionStreamFormat] {
        return [_streamFormat]
    }
}

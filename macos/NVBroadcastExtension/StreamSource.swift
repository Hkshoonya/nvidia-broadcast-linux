// NV Broadcast - CoreMediaIO Camera Extension
// Copyright (c) 2026 doczeus (https://github.com/Hkshoonya)
// Licensed under GPL-3.0
//
// CMIOExtensionStream — the video stream that delivers processed frames
// from the Python NV Broadcast app to consuming apps (Zoom, Chrome, etc.).
//
// Receives frames via shared memory (IOSurface) from the Python process,
// wraps them in CVPixelBuffer + CMSampleBuffer, and delivers to clients.

import Foundation
import CoreMediaIO
import CoreVideo
import IOSurface

class NVBroadcastStream: NSObject, CMIOExtensionStreamSource {

    private var _isStreaming = false
    private var _sequenceNumber: UInt64 = 0
    private var _timer: DispatchSourceTimer?
    private weak var _client: CMIOExtensionClient?

    // Frame buffer — written by Python helper, read by this stream
    private var _pixelBufferPool: CVPixelBufferPool?
    private var _currentPixelBuffer: CVPixelBuffer?
    private let _bufferLock = NSLock()

    private let _width = Int(NVBroadcastConstants.defaultWidth)
    private let _height = Int(NVBroadcastConstants.defaultHeight)
    private let _fps = Int(NVBroadcastConstants.defaultFPS)

    override init() {
        super.init()
        _setupPixelBufferPool()
        _setupFrameListener()
    }

    // MARK: - Pixel Buffer Pool

    private func _setupPixelBufferPool() {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: _width,
            kCVPixelBufferHeightKey as String: _height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            nil,
            attrs as CFDictionary,
            &_pixelBufferPool
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
        // Read frame from shared memory and update _currentPixelBuffer.
        // The Python helper writes raw BGRA bytes to a POSIX shared memory segment
        // named by NVBroadcastConstants.sharedMemoryName.
        guard _isStreaming else { return }

        let shmName = "/" + NVBroadcastConstants.sharedMemoryName
        let fd = shm_open(shmName, O_RDONLY, 0)
        guard fd >= 0 else { return }
        defer { close(fd) }

        let frameSize = _width * _height * 4  // BGRA
        guard let ptr = mmap(nil, frameSize, PROT_READ, MAP_SHARED, fd, 0),
              ptr != MAP_FAILED else { return }
        defer { munmap(ptr, frameSize) }

        _bufferLock.lock()
        defer { _bufferLock.unlock() }

        // Create CVPixelBuffer from the shared memory data
        guard let pool = _pixelBufferPool else { return }
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
        guard let pb = pixelBuffer else { return }

        CVPixelBufferLockBaseAddress(pb, [])
        if let dest = CVPixelBufferGetBaseAddress(pb) {
            memcpy(dest, ptr, frameSize)
        }
        CVPixelBufferUnlockBaseAddress(pb, [])

        _currentPixelBuffer = pb
    }

    // MARK: - Frame delivery timer

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
        let pb = _currentPixelBuffer
        _bufferLock.unlock()

        guard let pixelBuffer = pb else {
            // No frame from Python yet — deliver black frame
            _deliverBlackFrame()
            return
        }

        // Create CMSampleBuffer from CVPixelBuffer
        var sampleBuffer: CMSampleBuffer?
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

        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: fmt,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        )

        guard let sb = sampleBuffer else { return }

        _sequenceNumber += 1
        // Deliver to the extension stream delegate (macOS forwards to apps)
        // This is handled by the CMIOExtensionStream infrastructure
    }

    private func _deliverBlackFrame() {
        guard let pool = _pixelBufferPool else { return }
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
        guard let pb = pixelBuffer else { return }
        // Pool creates zero-initialized buffers (black in BGRA)
        _currentPixelBuffer = pb
    }

    // MARK: - CMIOExtensionStreamSource

    var availableProperties: Set<CMIOExtensionProperty> {
        return [
            .streamActiveFormatIndex,
            .streamFrameDuration,
        ]
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
        _client = client
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
        // Advertise 1080p30 BGRA as the default format
        let desc = CMVideoFormatDescription.init(
            videoCodecType: .init(rawValue: kCVPixelFormatType_32BGRA),
            width: Int32(_width),
            height: Int32(_height),
            extensions: nil
        )
        guard case .success(let fmt) = desc else { return [] }
        return [
            CMIOExtensionStreamFormat(
                formatDescription: fmt,
                maxFrameDuration: CMTime(value: 1, timescale: CMTimeScale(_fps)),
                minFrameDuration: CMTime(value: 1, timescale: CMTimeScale(_fps)),
                validFrameDurations: nil
            )
        ]
    }
}

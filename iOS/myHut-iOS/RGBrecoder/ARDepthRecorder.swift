import Foundation
import SwiftUI
import Combine
import ARKit
import CoreImage
import CoreGraphics
import UIKit
import simd

final class ARDepthRecorder: NSObject, ObservableObject, ARSessionDelegate {

    // MARK: - Network Configuration

    // Configure your Mac's local IP address in LocalConfig.swift.
    private let serverIP = LocalConfig.serverIP
    private let serverPort = LocalConfig.serverPort

    private var serverURL: String {
        "http://\(serverIP):\(serverPort)/upload"
    }

    private var commandURL: String {
        "http://\(serverIP):\(serverPort)/command"
    }

    private let urlSession = URLSession(configuration: .ephemeral)
   
    // MARK: - AR / Rendering

    let session = ARSession()
    private let ciContext = CIContext()

    private let sendingQueue = DispatchQueue(label: "com.rgbd.recorder.networkQueue")

    // ARFrame.capturedImage and sceneDepth.depthMap are saved in raw ARKit pixel-buffer orientation.
    // Do not rotate only RGB, otherwise RGB/depth/intrinsics/pose will no longer match.
    private let captureOrientation: UIInterfaceOrientation = .landscapeRight

    // MARK: - UI State

    @Published var selectedCaptureMode: CaptureMode? = nil

    @Published var isReviewing: Bool = false
    @Published var isRecording: Bool = false
    @Published var savedFrameCount: Int = 0
    @Published var currentSessionName: String = "No active session"
    @Published var lastSavedFrameName: String = "None"
    @Published var captureMessage: String = "Choose capture mode"
    @Published var rgbResolutionText: String = "RGB: —"
    @Published var depthResolutionText: String = "Depth: —"

    // MARK: - Capture Counters

    private var frameIndex: Int = 0
    private var internalFrameCounter: Int = 0

    // iPhone ARKit often runs at about 60 FPS.
    // Capturing every 3rd frame gives about 20 FPS.
    // Increase to 4–6 if network upload drops frames.
    private let captureEveryNFrames = 3

    // MARK: - Public Controls
    
    /// Starts the camera feed so the user can see the environment before recording.
    func startPreview() {
        let config = ARWorldTrackingConfiguration()
        
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
            captureMessage = "Preview: Smoothed Scene Depth"
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
            captureMessage = "Preview: Scene Depth"
        } else {
            captureMessage = "Device lacks ARKit scene depth support"
            return
        }
        
        config.worldAlignment = .gravity
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func start() {
        guard let selectedCaptureMode else {
            DispatchQueue.main.async {
                self.captureMessage = "Please choose a capture mode first."
            }
            return
        }

        frameIndex = 0
        savedFrameCount = 0
        internalFrameCounter = 0
        lastSavedFrameName = "None"
        isReviewing = false

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"

        let timestamp = formatter.string(from: Date())

        // Examples:
        // object_2026-06-29_15-55-10
        // measure_room_2026-06-29_15-55-10
        // room_full_2026-06-29_15-55-10
        currentSessionName = "\(selectedCaptureMode.rawValue)_\(timestamp)"

        sendSessionMetadata()

        DispatchQueue.main.async {
            self.isRecording = true
            self.captureMessage = "Recording \(selectedCaptureMode.title)"
        }
    }

    func stop() {
        DispatchQueue.main.async {
            self.isRecording = false
            self.isReviewing = true
            self.captureMessage = "Recording stopped. Review this capture."
        }
    }

    func submitDecision(keepData: Bool) {
        guard let url = URL(string: commandURL) else { return }

        let action = keepData ? "save" : "discard"

        let payload: [String: String] = [
            "session_name": currentSessionName,
            "action": action,
            "capture_mode": selectedCaptureMode?.rawValue ?? "unknown",
            "recommended_python_mode": selectedCaptureMode?.recommendedPythonMode ?? "unknown"
        ]

        guard let jsonData = try? JSONSerialization.data(withJSONObject: payload) else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        let task = urlSession.dataTask(with: request) { _, _, error in
            if let error = error {
                print("Command error: \(error.localizedDescription)")
            }
        }
        task.resume()

        DispatchQueue.main.async {
            self.isReviewing = false
            self.isRecording = false
            self.captureMessage = keepData ? "Data saved to Mac." : "Data discarded."
            self.savedFrameCount = 0
            self.rgbResolutionText = "RGB: —"
            self.depthResolutionText = "Depth: —"
            self.lastSavedFrameName = "None"
            self.selectedCaptureMode = nil
            self.currentSessionName = "No active session"
            
            // Pause the camera to save battery when returning to the main menu
            self.session.pause()
        }
    }

    // MARK: - Network Sender Helper

    private func sendData(_ data: Data, type: String, indexString: String, fileExtension: String) {
        guard let url = URL(string: "\(serverURL)/\(type)") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(currentSessionName, forHTTPHeaderField: "Session-Name")
        request.setValue(indexString, forHTTPHeaderField: "Frame-Index")
        request.setValue(fileExtension, forHTTPHeaderField: "File-Extension")
        request.setValue(selectedCaptureMode?.rawValue ?? "unknown", forHTTPHeaderField: "Capture-Mode")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")

        let task = urlSession.uploadTask(with: request, from: data) { _, _, error in
            if let error = error {
                print("Network error sending \(type) \(indexString): \(error.localizedDescription)")
            }
        }

        task.resume()
    }

    private func sendSessionMetadata() {
        let mode = selectedCaptureMode?.rawValue ?? "unknown"

        let metadata: [String: Any] = [
            "description": "iPhone ARKit RGB-D capture streamed from ARDepthRecorder",
            "capture_mode": mode,
            "recommended_python_mode": mode,
            "mode_description": selectedCaptureMode?.title ?? "Unknown",
            "mode_guidance": selectedCaptureMode?.guidance ?? "",
            "device_hint": "iPhone LiDAR / ARKit sceneDepth or smoothedSceneDepth",
            "rgb_format": "jpg, raw ARFrame.capturedImage orientation",
            "depth_format": "float32_binary_row_major",
            "depth_units": "meters",
            "confidence_format": "uint8_binary_row_major, ARKit confidence values 0=low 1=medium 2=high, when available",
            "pose_format": "4x4 ARKit camera_to_world matrix, column-major matrix written as text rows",
            "intrinsics_format": "3x3 ARKit camera intrinsics for capturedImage resolution; scale to depth resolution in Python",
            "timestamp_format": "one timestamp text file per frame in timestamp/",
            "coordinate_note": "ARKit camera uses x right, y up, -z forward. Convert to Open3D with diag(1,-1,-1,1).",
            "capture_orientation": "landscapeRight",
            "capture_every_n_frames": captureEveryNFrames,
            "session_name": currentSessionName,
            "created_at": ISO8601DateFormatter().string(from: Date())
        ]

        guard JSONSerialization.isValidJSONObject(metadata),
              let data = try? JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted]) else {
            return
        }

        sendData(data, type: "metadata", indexString: "000000", fileExtension: ".json")
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRecording else { return }

        internalFrameCounter += 1
        guard internalFrameCounter % captureEveryNFrames == 0 else { return }

        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }

        let indexString = String(format: "%06d", frameIndex)

        let rgbBuffer = frame.capturedImage
        let depthBuffer = depthData.depthMap
        let confidenceBuffer = depthData.confidenceMap

        let cameraTransform = frame.camera.transform
        let cameraIntrinsics = frame.camera.intrinsics
        let cameraImageResolution = frame.camera.imageResolution
        let timestamp = frame.timestamp
        let trackingStateDescription = String(describing: frame.camera.trackingState)
        let eulerAngles = frame.camera.eulerAngles

        let rgbWidth = CVPixelBufferGetWidth(rgbBuffer)
        let rgbHeight = CVPixelBufferGetHeight(rgbBuffer)
        let depthWidth = CVPixelBufferGetWidth(depthBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthBuffer)

        let displayTransform = frame.displayTransform(
            for: captureOrientation,
            viewportSize: CGSize(width: depthWidth, height: depthHeight)
        )

        sendingQueue.async { [weak self] in
            guard let self = self else { return }

            self.streamRGB(pixelBuffer: rgbBuffer, indexString: indexString)
            self.streamFloat32PixelBuffer(depthBuffer, type: "depth", indexString: indexString)

            if let confidenceBuffer {
                self.streamUInt8PixelBuffer(confidenceBuffer, type: "confidence", indexString: indexString)
            }

            self.streamPose(transform: cameraTransform, indexString: indexString)
            self.streamIntrinsics(intrinsics: cameraIntrinsics, indexString: indexString)
            self.streamTimestamp(timestamp: timestamp, indexString: indexString)

            self.streamFrameMetadata(
                indexString: indexString,
                timestamp: timestamp,
                rgbWidth: rgbWidth,
                rgbHeight: rgbHeight,
                depthWidth: depthWidth,
                depthHeight: depthHeight,
                cameraImageResolution: cameraImageResolution,
                displayTransform: displayTransform,
                trackingStateDescription: trackingStateDescription,
                eulerAngles: eulerAngles,
                hasConfidence: confidenceBuffer != nil
            )

            DispatchQueue.main.async {
                self.savedFrameCount += 1
                self.lastSavedFrameName = "Streamed: \(indexString)"
                self.rgbResolutionText = "RGB: \(rgbWidth) × \(rgbHeight)"
                self.depthResolutionText = "Depth: \(depthWidth) × \(depthHeight)"
            }
        }

        frameIndex += 1
    }

    // MARK: - Streaming Functions

    private func streamRGB(pixelBuffer: CVPixelBuffer, indexString: String) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        if let jpegData = ciContext.jpegRepresentation(
            of: ciImage,
            colorSpace: colorSpace,
            options: [:]
        ) {
            sendData(jpegData, type: "rgb", indexString: indexString, fileExtension: ".jpg")
        }
    }

    private func streamFloat32PixelBuffer(_ pixelBuffer: CVPixelBuffer, type: String, indexString: String) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }

        var values = [Float32]()
        values.reserveCapacity(width * height)

        for y in 0..<height {
            let rowPointer = baseAddress.advanced(by: y * bytesPerRow)
            let floatPointer = rowPointer.assumingMemoryBound(to: Float32.self)

            for x in 0..<width {
                values.append(floatPointer[x])
            }
        }

        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }

        sendData(data, type: type, indexString: indexString, fileExtension: ".bin")
        sendShape(height: height, width: width, type: type, indexString: indexString)
    }

    private func streamUInt8PixelBuffer(_ pixelBuffer: CVPixelBuffer, type: String, indexString: String) {
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)

        guard pixelFormat == kCVPixelFormatType_OneComponent8 else {
            print("Unsupported UInt8 pixel buffer format for \(type): \(pixelFormat)")
            return
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }

        var values = [UInt8]()
        values.reserveCapacity(width * height)

        for y in 0..<height {
            let rowPointer = baseAddress.advanced(by: y * bytesPerRow)
            let uint8Pointer = rowPointer.assumingMemoryBound(to: UInt8.self)

            for x in 0..<width {
                values.append(uint8Pointer[x])
            }
        }

        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }

        sendData(data, type: type, indexString: indexString, fileExtension: ".bin")
        sendShape(height: height, width: width, type: type, indexString: indexString)
    }

    private func sendShape(height: Int, width: Int, type: String, indexString: String) {
        let shapeText = "\(height) \(width)\n"

        if let shapeData = shapeText.data(using: .utf8) {
            sendData(
                shapeData,
                type: type,
                indexString: "\(indexString)_shape",
                fileExtension: ".txt"
            )
        }
    }

    private func streamPose(transform: simd_float4x4, indexString: String) {
        let text = matrix4x4ToText(transform)

        if let data = text.data(using: .utf8) {
            sendData(data, type: "pose", indexString: indexString, fileExtension: ".txt")
        }
    }

    private func streamIntrinsics(intrinsics: simd_float3x3, indexString: String) {
        let text = matrix3x3ToText(intrinsics)

        if let data = text.data(using: .utf8) {
            sendData(data, type: "intrinsics", indexString: indexString, fileExtension: ".txt")
        }
    }

    private func streamTimestamp(timestamp: TimeInterval, indexString: String) {
        let line = "\(timestamp)\n"

        if let data = line.data(using: .utf8) {
            sendData(data, type: "timestamp", indexString: indexString, fileExtension: ".txt")
        }
    }

    private func streamFrameMetadata(
        indexString: String,
        timestamp: TimeInterval,
        rgbWidth: Int,
        rgbHeight: Int,
        depthWidth: Int,
        depthHeight: Int,
        cameraImageResolution: CGSize,
        displayTransform: CGAffineTransform,
        trackingStateDescription: String,
        eulerAngles: simd_float3,
        hasConfidence: Bool
    ) {
        let mode = selectedCaptureMode?.rawValue ?? "unknown"

        let metadata: [String: Any] = [
            "frame_index": indexString,
            "capture_mode": mode,
            "recommended_python_mode": mode,
            "timestamp": timestamp,
            "rgb_width": rgbWidth,
            "rgb_height": rgbHeight,
            "depth_width": depthWidth,
            "depth_height": depthHeight,
            "camera_image_resolution_width": Double(cameraImageResolution.width),
            "camera_image_resolution_height": Double(cameraImageResolution.height),
            "capture_orientation": "landscapeRight",
            "display_transform_3x3": [
                [
                    Double(displayTransform.a),
                    Double(displayTransform.c),
                    Double(displayTransform.tx)
                ],
                [
                    Double(displayTransform.b),
                    Double(displayTransform.d),
                    Double(displayTransform.ty)
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "tracking_state": trackingStateDescription,
            "camera_euler_angles_xyz_radians": [
                Double(eulerAngles.x),
                Double(eulerAngles.y),
                Double(eulerAngles.z)
            ],
            "has_confidence": hasConfidence
        ]

        guard JSONSerialization.isValidJSONObject(metadata),
              let data = try? JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted]) else {
            return
        }

        sendData(data, type: "frame_metadata", indexString: indexString, fileExtension: ".json")
    }

    // MARK: - Matrix Formatting Helpers

    private func matrix4x4ToText(_ m: simd_float4x4) -> String {
        var lines: [String] = []

        for r in 0..<4 {
            let row = [
                m.columns.0[r],
                m.columns.1[r],
                m.columns.2[r],
                m.columns.3[r]
            ]

            lines.append(
                row
                    .map { String(format: "%.8f", $0) }
                    .joined(separator: " ")
            )
        }

        return lines.joined(separator: "\n") + "\n"
    }

    private func matrix3x3ToText(_ m: simd_float3x3) -> String {
        var lines: [String] = []

        for r in 0..<3 {
            let row = [
                m.columns.0[r],
                m.columns.1[r],
                m.columns.2[r]
            ]

            lines.append(
                row
                    .map { String(format: "%.8f", $0) }
                    .joined(separator: " ")
            )
        }

        return lines.joined(separator: "\n") + "\n"
    }
}

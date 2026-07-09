import SwiftUI
import ARKit
import SceneKit

// MARK: - Capture Mode

enum CaptureMode: String, CaseIterable, Identifiable, Codable {
    case object = "object"
    case measureRoom = "measure_room"
    case roomFull = "room_full"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .object:
            return "Object Scan"
        case .measureRoom:
            return "Room Measurement"
        case .roomFull:
            return "Full Room Scan"
        }
    }

    var technicalTitle: String {
        switch self {
        case .object:
            return "Object Reconstruction"
        case .measureRoom:
            return "Room Size Measurement"
        case .roomFull:
            return "Full Room Reconstruction"
        }
    }

    var subtitle: String {
        switch self {
        case .object:
            return "Capture one focused object for a clean 3D model."
        case .measureRoom:
            return "Scan walls, floor, and ceiling to estimate room size."
        case .roomFull:
            return "Capture the complete room with furniture and layout."
        }
    }

    var guidance: String {
        switch self {
        case .object:
            return "Move slowly around the object. Keep it centered and avoid background clutter."
        case .measureRoom:
            return "Scan all walls, corners, floor, and ceiling edges for the best measurements."
        case .roomFull:
            return "Walk slowly through the room and cover furniture, walls, floor, and corners."
        }
    }

    var icon: String {
        switch self {
        case .object:
            return "cube.transparent"
        case .measureRoom:
            return "ruler"
        case .roomFull:
            return "house.and.flag"
        }
    }

    var accent: Color {
        switch self {
        case .object:
            return .cyan
        case .measureRoom:
            return .purple
        case .roomFull:
            return .green
        }
    }

    var recommendedPythonMode: String {
        rawValue
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var recorder = ARDepthRecorder()

    @State private var animateBackground = false
    @State private var showHomeContent = false

    private var showCamera: Bool {
        recorder.isRecording || recorder.isReviewing || recorder.selectedCaptureMode != nil
    }

    var body: some View {
        ZStack {
            if showCamera {
                ARViewContainer(session: recorder.session)
                    .ignoresSafeArea()
                    .transition(.opacity)

                CameraDarkOverlay()
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            } else {
                MyHutAnimatedBackground(animate: animateBackground)
                    .ignoresSafeArea()
            }

            VStack(spacing: 0) {
                TopStatusBar(recorder: recorder)

                Spacer(minLength: 20)

                if recorder.selectedCaptureMode == nil && !recorder.isRecording && !recorder.isReviewing {
                    HomeModePicker(
                        recorder: recorder,
                        showHomeContent: showHomeContent
                    )
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                } else {
                    CapturePanel(recorder: recorder)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .padding(.horizontal, 18)
            .padding(.top, 14)
            .padding(.bottom, 20)
        }
        .onAppear {
            animateBackground = true

            withAnimation(.spring(response: 0.7, dampingFraction: 0.85).delay(0.1)) {
                showHomeContent = true
            }
        }
        .animation(.spring(response: 0.38, dampingFraction: 0.86), value: recorder.selectedCaptureMode)
        .animation(.spring(response: 0.38, dampingFraction: 0.86), value: recorder.isRecording)
        .animation(.spring(response: 0.38, dampingFraction: 0.86), value: recorder.isReviewing)
    }
}

// MARK: - Home Mode Picker

struct HomeModePicker: View {
    @ObservedObject var recorder: ARDepthRecorder
    let showHomeContent: Bool

    var body: some View {
        VStack(spacing: 22) {
            heroHeader
                .opacity(showHomeContent ? 1 : 0)
                .offset(y: showHomeContent ? 0 : 24)

            GlassPanel {
                VStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Choose Capture Type")
                            .font(.title2)
                            .bold()
                            .foregroundColor(.white)

                        Text("Select what you want myHut to capture before recording.")
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.72))
                    }

                    VStack(spacing: 12) {
                        ForEach(Array(CaptureMode.allCases.enumerated()), id: \.element.id) { index, mode in
                            ModeCard(mode: mode) {
                                UIImpactFeedbackGenerator(style: .medium).impactOccurred()

                                recorder.selectedCaptureMode = mode
                                recorder.captureMessage = mode.guidance
                            }
                            .opacity(showHomeContent ? 1 : 0)
                            .offset(y: showHomeContent ? 0 : 18)
                            .animation(
                                .spring(response: 0.55, dampingFraction: 0.82)
                                .delay(0.12 + Double(index) * 0.08),
                                value: showHomeContent
                            )
                        }
                    }
                }
            }
            .opacity(showHomeContent ? 1 : 0)
            .offset(y: showHomeContent ? 0 : 28)
        }
    }

    private var heroHeader: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.08))
                    .frame(width: 120, height: 120)
                    .blur(radius: 2)

                Circle()
                    .stroke(
                        LinearGradient(
                            colors: [
                                Color.cyan.opacity(0.65),
                                Color.green.opacity(0.25),
                                Color.white.opacity(0.12)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1.2
                    )
                    .frame(width: 112, height: 112)

                Image("myhut_icon")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 74, height: 74)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .shadow(color: Color.cyan.opacity(0.22), radius: 20, x: 0, y: 10)
            }

            VStack(spacing: 6) {
                Text("myHut")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                Text("Scan, measure, and rebuild your space")
                    .font(.headline)
                    .foregroundColor(.white.opacity(0.72))
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 10)
    }
}

// MARK: - Top Status Bar

struct TopStatusBar: View {
    @ObservedObject var recorder: ARDepthRecorder

    var body: some View {
        HStack(spacing: 12) {
            Image("myhut_icon")
                .resizable()
                .scaledToFit()
                .frame(width: 38, height: 38)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color.white.opacity(0.18), lineWidth: 1)
                )

            VStack(alignment: .leading, spacing: 3) {
                Text("myHut")
                    .font(.headline)
                    .foregroundColor(.white)

                Text(statusSubtitle)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.72))
                    .lineLimit(1)
            }

            Spacer()

            HStack(spacing: 7) {
                Circle()
                    .fill(recorder.isRecording ? Color.green : Color.gray)
                    .frame(width: 9, height: 9)
                    .shadow(
                        color: recorder.isRecording ? Color.green.opacity(0.75) : Color.clear,
                        radius: 7,
                        x: 0,
                        y: 0
                    )

                Text(recorder.isRecording ? "LIVE" : "IDLE")
                    .font(.caption2)
                    .bold()
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial)
            .clipShape(Capsule())
            .overlay(
                Capsule()
                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
            )
        }
    }

    private var statusSubtitle: String {
        if recorder.isRecording {
            return "Recording RGB-D stream"
        }

        if recorder.isReviewing {
            return "Review capture"
        }

        if let mode = recorder.selectedCaptureMode {
            return mode.title
        }

        return "Scan, measure, and rebuild"
    }
}

// MARK: - Capture Panel

struct CapturePanel: View {
    @ObservedObject var recorder: ARDepthRecorder

    var body: some View {
        GlassPanel {
            VStack(spacing: 17) {
                selectedModeHeader

                Divider()
                    .background(Color.white.opacity(0.18))

                Text(recorder.captureMessage)
                    .font(.subheadline)
                    .foregroundColor(recorder.isRecording ? .green : .white.opacity(0.78))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)

                captureStats

                if recorder.isReviewing {
                    ReviewDialog(recorder: recorder)
                } else {
                    recordButton
                }
            }
        }
    }

    private var selectedModeHeader: some View {
        HStack(spacing: 13) {
            if let mode = recorder.selectedCaptureMode {
                ZStack {
                    RoundedRectangle(cornerRadius: 17, style: .continuous)
                        .fill(mode.accent.opacity(0.20))
                        .frame(width: 54, height: 54)

                    Image(systemName: mode.icon)
                        .font(.title2)
                        .foregroundColor(mode.accent)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(mode.title)
                        .font(.headline)
                        .foregroundColor(.white)

                    Text(mode.guidance)
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.66))
                        .lineLimit(2)
                }

                Spacer()

                if !recorder.isRecording && !recorder.isReviewing {
                    Button {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()

                        recorder.selectedCaptureMode = nil
                        recorder.captureMessage = "Choose capture mode"
                    } label: {
                        Text("Change")
                            .font(.caption)
                            .bold()
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(Color.white.opacity(0.14))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var captureStats: some View {
        HStack(spacing: 10) {
            StatPill(
                title: "Frames",
                value: "\(recorder.savedFrameCount)"
            )

            StatPill(
                title: "RGB",
                value: cleanResolution(recorder.rgbResolutionText, prefix: "RGB: ")
            )

            StatPill(
                title: "Depth",
                value: cleanResolution(recorder.depthResolutionText, prefix: "Depth: ")
            )
        }
    }

    private func cleanResolution(_ text: String, prefix: String) -> String {
        let value = text.replacingOccurrences(of: prefix, with: "")
        return value == "—" ? "—" : value
    }

    private var recordButton: some View {
        Button {
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred()

            if recorder.isRecording {
                recorder.stop()
            } else {
                recorder.start()
            }
        } label: {
            HStack(spacing: 10) {
                Image(systemName: recorder.isRecording ? "stop.fill" : "record.circle")
                    .font(.title3)

                Text(recorder.isRecording ? "Stop Recording" : "Start Recording")
                    .font(.headline)
            }
            .foregroundColor(.white)
            .padding()
            .frame(maxWidth: .infinity)
            .background(
                LinearGradient(
                    colors: recorder.isRecording
                    ? [Color.orange, Color.red]
                    : [Color.blue, Color.cyan],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .clipShape(RoundedRectangle(cornerRadius: 17, style: .continuous))
            .shadow(
                color: (recorder.isRecording ? Color.red : Color.blue).opacity(0.36),
                radius: 15,
                x: 0,
                y: 9
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Review Dialog

struct ReviewDialog: View {
    @ObservedObject var recorder: ARDepthRecorder

    var body: some View {
        VStack(spacing: 15) {
            VStack(spacing: 5) {
                Text("Save this capture?")
                    .font(.headline)
                    .foregroundColor(.white)

                Text("Send the final decision to your Mac.")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.65))
            }

            HStack(spacing: 12) {
                Button {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    recorder.submitDecision(keepData: false)
                } label: {
                    Label("Discard", systemImage: "trash")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.red.opacity(0.88))
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
                .buttonStyle(.plain)

                Button {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    recorder.submitDecision(keepData: true)
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.green.opacity(0.88))
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Mode Card

struct ModeCard: View {
    let mode: CaptureMode
    let action: () -> Void

    @State private var isPressed = false

    var body: some View {
        Button {
            action()
        } label: {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(mode.accent.opacity(0.22))
                        .frame(width: 58, height: 58)

                    Image(systemName: mode.icon)
                        .font(.title2)
                        .foregroundColor(mode.accent)
                }

                VStack(alignment: .leading, spacing: 5) {
                    Text(mode.title)
                        .font(.headline)
                        .foregroundColor(.white)

                    Text(mode.subtitle)
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.70))
                        .multilineTextAlignment(.leading)
                        .lineLimit(2)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption)
                    .bold()
                    .foregroundColor(.white.opacity(0.65))
            }
            .padding(15)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(Color.white.opacity(0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(
                        LinearGradient(
                            colors: [
                                mode.accent.opacity(0.45),
                                Color.white.opacity(0.10)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1
                    )
            )
            .scaleEffect(isPressed ? 0.985 : 1.0)
            .shadow(color: mode.accent.opacity(0.12), radius: 14, x: 0, y: 8)
        }
        .buttonStyle(.plain)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    withAnimation(.easeOut(duration: 0.12)) {
                        isPressed = true
                    }
                }
                .onEnded { _ in
                    withAnimation(.easeOut(duration: 0.12)) {
                        isPressed = false
                    }
                }
        )
    }
}

// MARK: - Glass Panel

struct GlassPanel<Content: View>: View {
    let content: () -> Content

    var body: some View {
        content()
            .padding(18)
            .background(.ultraThinMaterial)
            .background(
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(Color.black.opacity(0.36))
            )
            .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .stroke(Color.white.opacity(0.17), lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.42), radius: 26, x: 0, y: 16)
    }
}

// MARK: - Stat Pill

struct StatPill: View {
    let title: String
    let value: String

    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.58))

            Text(value.isEmpty ? "—" : value)
                .font(.caption)
                .bold()
                .foregroundColor(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.65)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .background(Color.white.opacity(0.105))
        .clipShape(RoundedRectangle(cornerRadius: 15, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
    }
}

// MARK: - Backgrounds

struct MyHutAnimatedBackground: View {
    let animate: Bool

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.055, green: 0.065, blue: 0.075),
                    Color(red: 0.095, green: 0.100, blue: 0.115),
                    Color(red: 0.030, green: 0.032, blue: 0.038)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(Color.cyan.opacity(0.14))
                .frame(width: 300, height: 300)
                .blur(radius: 95)
                .offset(x: animate ? -150 : -210, y: animate ? -250 : -310)

            Circle()
                .fill(Color.green.opacity(0.10))
                .frame(width: 340, height: 340)
                .blur(radius: 110)
                .offset(x: animate ? 190 : 250, y: animate ? 260 : 320)

            Circle()
                .fill(Color.purple.opacity(0.09))
                .frame(width: 260, height: 260)
                .blur(radius: 105)
                .offset(x: animate ? 160 : 110, y: animate ? -120 : -170)

            WallTexture()
                .opacity(0.18)

            VStack {
                Spacer()

                Image(systemName: "house.and.flag.fill")
                    .font(.system(size: 180, weight: .bold))
                    .foregroundColor(.white.opacity(0.025))
                    .padding(.bottom, 160)
            }
        }
        .animation(
            .easeInOut(duration: 6.0).repeatForever(autoreverses: true),
            value: animate
        )
    }
}

struct CameraDarkOverlay: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color.black.opacity(0.18),
                    Color.black.opacity(0.40),
                    Color.black.opacity(0.78)
                ],
                startPoint: .top,
                endPoint: .bottom
            )

            RadialGradient(
                colors: [
                    Color.cyan.opacity(0.08),
                    Color.clear
                ],
                center: .topLeading,
                startRadius: 20,
                endRadius: 460
            )

            RadialGradient(
                colors: [
                    Color.green.opacity(0.08),
                    Color.clear
                ],
                center: .bottomTrailing,
                startRadius: 30,
                endRadius: 520
            )
        }
    }
}

struct WallTexture: View {
    var body: some View {
        Canvas { context, size in
            let spacing: CGFloat = 22

            for x in stride(from: CGFloat(0), through: size.width, by: spacing) {
                var path = Path()
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: size.height))

                context.stroke(
                    path,
                    with: .color(.white.opacity(0.025)),
                    lineWidth: 0.7
                )
            }

            for y in stride(from: CGFloat(0), through: size.height, by: spacing) {
                var path = Path()
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))

                context.stroke(
                    path,
                    with: .color(.white.opacity(0.020)),
                    lineWidth: 0.7
                )
            }

            for i in 0..<70 {
                let x = CGFloat((i * 37) % Int(max(size.width, 1)))
                let y = CGFloat((i * 91) % Int(max(size.height, 1)))

                let rect = CGRect(x: x, y: y, width: 1.2, height: 1.2)

                context.fill(
                    Path(ellipseIn: rect),
                    with: .color(.white.opacity(0.045))
                )
            }
        }
    }
}

// MARK: - AR Camera View Wrapper

struct ARViewContainer: UIViewRepresentable {
    var session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = true
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

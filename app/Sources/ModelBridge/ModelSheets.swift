import SwiftUI
import AppKit

struct InspectModelSheet: View {
    let sourcePath: String
    let rawJSON: String
    let info: InspectedModel?

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model Inspection")
                        .font(.title3.weight(.bold))
                    Text(sourcePath)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Button("Copy JSON") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(rawJSON, forType: .string)
                }
                .disabled(rawJSON.isEmpty)

                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }

            if let info {
                HStack(spacing: 8) {
                    if let fmt = info.format { BadgeView(text: fmt, color: .blue) }
                    if let arch = info.architecture { InfoPill(label: "Arch", value: arch) }
                    if let sz = info.file_size_gb { InfoPill(label: "Size", value: String(format: "%.2f GB", sz)) }
                    if let shards = info.num_weight_shards, shards > 0 { InfoPill(label: "Shards", value: "\(shards)") }
                    if let tensors = info.n_tensors, tensors > 0 { InfoPill(label: "Tensors", value: "\(tensors)") }
                }
            }

            ScrollView {
                Text(rawJSON.isEmpty ? "No inspection details available." : rawJSON)
                    .font(.system(size: 11.5, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2)))
        }
        .padding(18)
        .frame(minWidth: 580, minHeight: 420)
    }
}

struct ModelCardEditorSheet: View {
    let sourcePath: String
    @Binding var card: EditableModelCard
    var onSave: (EditableModelCard) -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var nameText: String = ""
    @State private var rendererText: String = ""
    @State private var parserText: String = ""
    @State private var capabilitiesText: String = ""
    @State private var numCtx: Int = 32768
    @State private var tempText: String = ""
    @State private var topPText: String = ""
    @State private var topKText: String = ""
    @State private var systemText: String = ""

    private var isOllama: Bool { card.kind == "ollama" }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(isOllama ? "Edit Ollama Model Card" : "Edit Model Generation Config")
                        .font(.title3.weight(.bold))
                    Text(isOllama ? "Zero-copy update via FROM \(card.name ?? "") — no weight conversion." : (card.path ?? sourcePath))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
            }

            Divider()

            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                if isOllama {
                    GridRow {
                        Text("Model Name")
                            .frame(width: 110, alignment: .trailing)
                            .foregroundStyle(.secondary)
                        TextField("Ollama model name", text: $nameText)
                            .textFieldStyle(.roundedBorder)
                    }
                    GridRow {
                        Text("Renderer / Parser")
                            .frame(width: 110, alignment: .trailing)
                            .foregroundStyle(.secondary)
                        HStack(spacing: 8) {
                            TextField("Renderer (e.g. gemma4, qwen3.5)", text: $rendererText)
                                .textFieldStyle(.roundedBorder)
                            TextField("Parser (optional)", text: $parserText)
                                .textFieldStyle(.roundedBorder)
                        }
                    }
                    GridRow {
                        Text("Capabilities")
                            .frame(width: 110, alignment: .trailing)
                            .foregroundStyle(.secondary)
                        TextField("completion, tools, thinking", text: $capabilitiesText)
                            .textFieldStyle(.roundedBorder)
                    }
                    GridRow {
                        Text("Context (num_ctx)")
                            .frame(width: 110, alignment: .trailing)
                            .foregroundStyle(.secondary)
                        Picker("Context", selection: $numCtx) {
                            Text("8K").tag(8192)
                            Text("32K").tag(32768)
                            Text("64K").tag(65536)
                            Text("96K").tag(98304)
                            Text("128K").tag(131072)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                    }
                }

                GridRow {
                    Text("Sampling")
                        .frame(width: 110, alignment: .trailing)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        TextField("temp (e.g. 0.6)", text: $tempText)
                            .textFieldStyle(.roundedBorder)
                        TextField("top_p (e.g. 0.95)", text: $topPText)
                            .textFieldStyle(.roundedBorder)
                        TextField("top_k (e.g. 30)", text: $topKText)
                            .textFieldStyle(.roundedBorder)
                    }
                }

                if isOllama {
                    GridRow {
                        Text("System Prompt")
                            .frame(width: 110, alignment: .topTrailing)
                            .foregroundStyle(.secondary)
                        TextEditor(text: $systemText)
                            .font(.system(.callout, design: .monospaced))
                            .frame(minHeight: 110, maxHeight: 180)
                            .padding(4)
                            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.25)))
                    }
                }
            }

            Spacer()

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save Card") {
                    var updated = card
                    updated.name = nameText
                    updated.renderer = rendererText
                    updated.parser = parserText
                    updated.capabilities = capabilitiesText
                    updated.num_ctx = numCtx
                    updated.temperature = Double(tempText.trimmingCharacters(in: .whitespaces))
                    updated.top_p = Double(topPText.trimmingCharacters(in: .whitespaces))
                    updated.top_k = Int(topKText.trimmingCharacters(in: .whitespaces))
                    updated.system = systemText
                    onSave(updated)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 560, minHeight: isOllama ? 440 : 230)
        .onAppear {
            nameText = card.name ?? ""
            rendererText = card.renderer ?? ""
            parserText = card.parser ?? ""
            capabilitiesText = card.capabilities ?? "completion"
            numCtx = card.num_ctx ?? 32768
            tempText = card.temperature.map { String($0) } ?? ""
            topPText = card.top_p.map { String($0) } ?? ""
            topKText = card.top_k.map { String($0) } ?? ""
            systemText = card.system ?? ""
        }
    }
}

struct TestModelSheet: View {
    @Bindable var state: BridgeState
    @Environment(\.dismiss) private var dismiss

    @State private var prompt: String = "Say hello and confirm you are working."
    @State private var maxTokens: Int = 256

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Test Model")
                        .font(.title3.weight(.bold))
                    Text(state.sourcePath)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                if let fmt = state.inspectedModel?.format {
                    BadgeView(text: fmt, color: .blue)
                }
                Button("Done") {
                    state.stopModelTest()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
            }

            HStack(spacing: 8) {
                TextField("Enter a test message…", text: $prompt)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        if !state.isTestingModel {
                            state.runModelTest(source: state.sourcePath, prompt: prompt, maxTokens: maxTokens)
                        }
                    }

                Picker("Max Tokens", selection: $maxTokens) {
                    Text("64").tag(64)
                    Text("128").tag(128)
                    Text("256").tag(256)
                    Text("512").tag(512)
                }
                .frame(width: 85)

                if state.isTestingModel {
                    Button(role: .destructive) {
                        state.stopModelTest()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                            .foregroundStyle(.white)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                } else {
                    Button {
                        state.runModelTest(source: state.sourcePath, prompt: prompt, maxTokens: maxTokens)
                    } label: {
                        Label("Send", systemImage: "paperplane.fill")
                            .foregroundStyle(.white)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.accentColor)
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .keyboardShortcut(.defaultAction)
                }
            }

            ScrollViewReader { proxy in
                ScrollView {
                    Text(state.testOutput.isEmpty ? (state.isTestingModel ? "Waiting for tokens…" : "Send a message to test inference.") : state.testOutput)
                        .font(.system(size: 12.5, design: .monospaced))
                        .foregroundStyle(state.testOutput.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .id("testEnd")
                }
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2)))
                .onChange(of: state.testOutput) {
                    proxy.scrollTo("testEnd", anchor: .bottom)
                }
            }

            HStack {
                if state.isTestingModel {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(state.testStatusNote)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Copy Output") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(state.testOutput, forType: .string)
                }
                .controlSize(.small)
                .disabled(state.testOutput.isEmpty)
            }
        }
        .padding(18)
        .frame(minWidth: 600, minHeight: 420)
    }
}

import SwiftUI
import UniformTypeIdentifiers
import AppKit

struct ContentView: View {
    @State private var state = BridgeState()
    @State private var activeTab: Int = 0
    @State private var isDropTargeted = false
    @State private var pendingDeleteOllama: String?
    @State private var pendingDeleteLMStudio: LMStudioModel?
    @State private var showingInspectSheet = false
    @State private var showingCardEditor = false
    @State private var showingTestSheet = false
    @State private var editingCard = EditableModelCard()

    var body: some View {
        Group {
            if activeTab == 0 {
                NavigationSplitView {
                    sidebarView
                        .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 350)
                } detail: {
                    detailView
                }
            } else {
                ExploreView(bridgeState: state) { hfSource in
                    state.selectSource(hfSource)
                    activeTab = 0
                }
            }
        }
        .frame(minWidth: 900, minHeight: 650)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("Mode", selection: $activeTab) {
                    Text("Bridge & Convert").tag(0)
                    Text("Explore Hub").tag(1)
                }
                .pickerStyle(.segmented)
                .frame(width: 240)
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await state.refreshAll() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(state.isRefreshing)
                .help("Rescan LM Studio and Ollama models")
            }
        }
        .task {
            await state.refreshAll()
        }
        .alert(
            "Delete Ollama Model?",
            isPresented: Binding(
                get: { pendingDeleteOllama != nil },
                set: { if !$0 { pendingDeleteOllama = nil } }
            )
        ) {
            Button("Delete", role: .destructive) {
                if let name = pendingDeleteOllama {
                    Task { await state.deleteOllamaModel(name) }
                }
                pendingDeleteOllama = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDeleteOllama = nil
            }
        } message: {
            Text("This will permanently remove '\(pendingDeleteOllama ?? "")' from Ollama (`ollama rm`).")
        }
        .alert(
            "Delete LM Studio Model?",
            isPresented: Binding(
                get: { pendingDeleteLMStudio != nil },
                set: { if !$0 { pendingDeleteLMStudio = nil } }
            )
        ) {
            Button("Delete", role: .destructive) {
                if let m = pendingDeleteLMStudio {
                    Task { await state.deleteLMStudioModel(m.path) }
                }
                pendingDeleteLMStudio = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDeleteLMStudio = nil
            }
        } message: {
            Text("This will permanently delete '\(pendingDeleteLMStudio?.displayName ?? "")' from disk.")
        }
        .sheet(isPresented: $showingInspectSheet) {
            InspectModelSheet(
                sourcePath: state.sourcePath,
                rawJSON: state.inspectedRawJSON,
                info: state.inspectedModel
            )
        }
        .sheet(isPresented: $showingCardEditor) {
            ModelCardEditorSheet(
                sourcePath: state.sourcePath,
                card: $editingCard,
                onSave: { updated in
                    state.saveModelCard(
                        source: state.sourcePath,
                        name: updated.name,
                        renderer: updated.renderer,
                        parser: updated.parser,
                        capabilities: updated.capabilities,
                        numCtx: updated.num_ctx,
                        temperature: updated.temperature,
                        topP: updated.top_p,
                        topK: updated.top_k,
                        systemPrompt: updated.system
                    )
                }
            )
        }
        .sheet(isPresented: $showingTestSheet) {
            TestModelSheet(state: state)
        }
    }

    // MARK: - Sidebar

    private var sidebarView: some View {
        List {
            Section {
                if state.lmStudioModels.isEmpty {
                    Text("No models in ~/.lmstudio/models")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(state.lmStudioModels) { model in
                        Button {
                            state.selectSource(model.path)
                        } label: {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(model.displayName)
                                    .font(.system(.body, weight: .medium))
                                    .lineLimit(1)
                                HStack(spacing: 6) {
                                    BadgeView(text: model.format ?? "Unknown", color: formatColor(model.format))
                                    if let sz = model.file_size_gb {
                                        Text(String(format: "%.1f GB", sz))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .padding(.vertical, 2)
                        .contextMenu {
                            Button("Select as Source") {
                                state.selectSource(model.path)
                            }
                            Button("Archive / Backup…") {
                                state.selectSource(model.path)
                                state.targetFormat = .archive
                            }
                            Divider()
                            Button("Delete from Disk…", role: .destructive) {
                                pendingDeleteLMStudio = model
                            }
                        }
                    }
                }
            } header: {
                Label("LM Studio", systemImage: "folder.badge.gearshape")
            }

            Section {
                if state.ollamaModels.isEmpty {
                    Text("No Ollama models found")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(state.ollamaModels) { model in
                        Button {
                            state.selectSource("ollama:\(model.name)")
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(model.name)
                                    .font(.system(.callout, weight: .medium))
                                    .lineLimit(1)
                                HStack {
                                    Text(model.size)
                                    Text("·")
                                    Text(model.modified)
                                }
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .padding(.vertical, 2)
                        .contextMenu {
                            Button("Select as Source") {
                                state.selectSource("ollama:\(model.name)")
                            }
                            Button("Export to LM Studio") {
                                state.selectSource("ollama:\(model.name)")
                                state.targetFormat = .lmstudio
                            }
                            Button("Archive / Backup…") {
                                state.selectSource("ollama:\(model.name)")
                                state.targetFormat = .archive
                            }
                            Divider()
                            Button("Delete from Ollama…", role: .destructive) {
                                pendingDeleteOllama = model.name
                            }
                        }
                    }
                }
            } header: {
                Label("Ollama Registry", systemImage: "terminal")
            }
        }
        .listStyle(.sidebar)
    }

    // MARK: - Detail View

    private var detailView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                sourceCard
                targetCard
                consoleCard
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 14)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var sourceCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    TextField("Drop .gguf / folder, browse, or enter HuggingFace ID (e.g. LiquidAI/LFM2.5-1.2B-Instruct)", text: $state.sourcePath)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            state.selectSource(state.sourcePath)
                        }

                    Button("Browse…") {
                        pickSourcePath()
                    }

                    Button {
                        Task {
                            await state.inspectSource(state.sourcePath)
                            showingInspectSheet = true
                        }
                    } label: {
                        if state.isInspecting {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Text("Inspect")
                        }
                    }
                    .disabled(state.sourcePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || state.isInspecting)
                }

                if let info = state.inspectedModel {
                    if let err = info.error {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.red)
                    } else {
                        HStack(spacing: 12) {
                            BadgeView(text: info.format ?? "Unknown", color: formatColor(info.format))

                            if let arch = info.architecture {
                                InfoPill(label: "Arch", value: arch)
                            }
                            if let sz = info.file_size_gb {
                                InfoPill(label: "Size", value: String(format: "%.2f GB", sz))
                            }
                            if let q = info.quantization, let bits = q.bits {
                                InfoPill(label: "Quant", value: "\(bits)-bit \(q.mode ?? "")")
                            } else if let shards = info.num_weight_shards, shards > 0 {
                                InfoPill(label: "Shards", value: "\(shards)")
                            }

                            Spacer()

                            if state.sourcePath.hasPrefix("ollama:") || FileManager.default.fileExists(atPath: state.sourcePath) {
                                Button {
                                    state.testOutput = ""
                                    state.testStatusNote = ""
                                    showingTestSheet = true
                                } label: {
                                    Label("Test…", systemImage: "play.circle")
                                }
                                .controlSize(.small)
                                .buttonStyle(.bordered)
                                .help("Send a quick test message to this model")

                                Button {
                                    Task {
                                        if let loaded = await state.fetchModelCard(for: state.sourcePath), loaded.error == nil {
                                            editingCard = loaded
                                            showingCardEditor = true
                                        }
                                    }
                                } label: {
                                    Label("Edit Card…", systemImage: "square.and.pencil")
                                }
                                .controlSize(.small)
                                .buttonStyle(.bordered)
                                .help("Edit Model Card / parameters without converting weights")
                            }

                            if state.sourcePath.hasPrefix("ollama:") {
                                Button(role: .destructive) {
                                    pendingDeleteOllama = String(state.sourcePath.dropFirst("ollama:".count))
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                                .controlSize(.small)
                                .buttonStyle(.bordered)
                                .tint(.red)
                            } else if let lmMatch = state.lmStudioModels.first(where: { $0.path == state.sourcePath }) {
                                Button(role: .destructive) {
                                    pendingDeleteLMStudio = lmMatch
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                                .controlSize(.small)
                                .buttonStyle(.bordered)
                                .tint(.red)
                            }
                        }
                    }
                }
            }
            .padding(6)
        } label: {
            Label("Source Model", systemImage: "cube.box")
                .font(.headline)
        }
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isDropTargeted ? Color.accentColor : Color.clear, lineWidth: 2)
        )
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    private var targetCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Picker("Target", selection: $state.targetFormat) {
                    ForEach(TargetFormat.allCases) { target in
                        Text(target.displayName).tag(target)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(state.targetFormat.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Divider()

                // Dynamic target options
                switch state.targetFormat {
                case .ollama, .ollamaGGUF, .ollamaMLX:
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        if state.targetFormat == .ollamaMLX {
                            GridRow {
                                Text("MLX Precision")
                                    .frame(width: 100, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                                Picker("Bits", selection: $state.mlxBits) {
                                    Text("4-bit (Fastest)").tag(4)
                                    Text("8-bit (Balanced)").tag(8)
                                    Text("16-bit (BF16)").tag(16)
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                            }
                        } else if state.targetFormat == .ollamaGGUF {
                            GridRow {
                                Text("GGUF Quant")
                                    .frame(width: 100, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                                Picker("GGUF Quant", selection: $state.ggufQuant) {
                                    Text("Q4_K_M").tag("Q4_K_M")
                                    Text("Q5_K_M").tag("Q5_K_M")
                                    Text("Q8_0").tag("Q8_0")
                                    Text("IQ4_NL").tag("IQ4_NL")
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                            }
                        }
                        GridRow {
                            Text("Model Name")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            TextField("e.g. my-local-coder", text: $state.ollamaModelName)
                                .textFieldStyle(.roundedBorder)
                        }
                        GridRow {
                            Text("Renderer / Parser")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("Renderer (e.g. qwen3.5, gemma4)", text: $state.ollamaRenderer)
                                    .textFieldStyle(.roundedBorder)
                                TextField("Parser (optional)", text: $state.ollamaParser)
                                    .textFieldStyle(.roundedBorder)
                            }
                        }
                        GridRow {
                            Text("Capabilities")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            TextField("completion, tools, thinking", text: $state.ollamaCapabilities)
                                .textFieldStyle(.roundedBorder)
                        }
                        GridRow {
                            Text("Context (num_ctx)")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            Picker("Context", selection: $state.ollamaNumCtx) {
                                Text("8K").tag(8192)
                                Text("32K").tag(32768)
                                Text("64K").tag(65536)
                                Text("96K").tag(98304)
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }
                        GridRow {
                            Text("System Prompt")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            TextField("Optional system instruction", text: $state.ollamaSystemPrompt)
                                .textFieldStyle(.roundedBorder)
                        }
                        if state.targetFormat == .ollama {
                            GridRow {
                                Text("Quantize")
                                    .frame(width: 100, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                                Picker("Ollama Quant", selection: $state.ollamaQuant) {
                                    Text("Original (BF16/F16)").tag("")
                                    Text("INT4").tag("int4")
                                    Text("INT8").tag("int8")
                                    Text("MXFP4").tag("mxfp4")
                                    Text("NVFP4").tag("nvfp4")
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                            }
                        }
                        GridRow {
                            Text("")
                                .frame(width: 100)
                            Toggle("Move to Ollama (remove original files after import to save disk space)", isOn: $state.moveToOllama)
                                .font(.caption)
                        }
                    }
                case .mlx:
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        GridRow {
                            Text("Precision")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            Picker("Bits", selection: $state.mlxBits) {
                                Text("4-bit (Fastest)").tag(4)
                                Text("8-bit (Balanced)").tag(8)
                                Text("16-bit (BF16)").tag(16)
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }
                        GridRow {
                            Text("Output Dir")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("Default: <source>-mlx-<bits>bit", text: $state.customOutputPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("Choose…") {
                                    pickOutputFolder()
                                }
                            }
                        }
                    }
                case .gguf:
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        GridRow {
                            Text("Quantization")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            Picker("Quant", selection: $state.ggufQuant) {
                                Text("Q4_K_M").tag("Q4_K_M")
                                Text("Q5_K_M").tag("Q5_K_M")
                                Text("Q8_0").tag("Q8_0")
                                Text("IQ4_NL").tag("IQ4_NL")
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }
                        GridRow {
                            Text("Output Dir / File")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("Default: <source>_<QUANT>.gguf", text: $state.customOutputPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("Choose…") {
                                    pickOutputFolder()
                                }
                            }
                        }
                    }
                case .safetensors:
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        GridRow {
                            Text("Precision")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            Picker("Dtype", selection: $state.safetensorsDtype) {
                                Text("BF16").tag("bfloat16")
                                Text("FP16").tag("float16")
                                Text("FP32").tag("float32")
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }
                        GridRow {
                            Text("Output Dir")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("Default: ~/ModelExports/<name>-safetensors", text: $state.customOutputPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("Choose…") {
                                    pickOutputFolder()
                                }
                            }
                        }
                    }
                case .onnx:
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        GridRow {
                            Text("Quantization")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            Picker("ONNX Quant", selection: $state.onnxQuant) {
                                Text("4-bit (q4f16)").tag("q4f16")
                                Text("INT4").tag("int4")
                                Text("INT8").tag("int8")
                                Text("FP16").tag("fp16")
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }
                        GridRow {
                            Text("Output Dir")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("Default: ~/ModelExports/<name>-ONNX-<quant>", text: $state.customOutputPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("Choose…") {
                                    pickOutputFolder()
                                }
                            }
                        }
                    }
                case .lmstudio:
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        GridRow {
                            Text("Format")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            Picker("LM Studio Format", selection: $state.lmsTargetFormat) {
                                Text("MLX").tag("mlx")
                                Text("Direct / Auto").tag("auto")
                                Text("GGUF").tag("gguf")
                                Text("Safetensors").tag("safetensors")
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }
                        if state.lmsTargetFormat == "mlx" {
                            GridRow {
                                Text("Precision")
                                    .frame(width: 100, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                                Picker("Bits", selection: $state.mlxBits) {
                                    Text("4-bit (Fastest)").tag(4)
                                    Text("8-bit (Balanced)").tag(8)
                                    Text("16-bit (BF16)").tag(16)
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                            }
                        } else if state.lmsTargetFormat == "gguf" {
                            GridRow {
                                Text("Quantization")
                                    .frame(width: 100, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                                Picker("Quant", selection: $state.ggufQuant) {
                                    Text("Q4_K_M").tag("Q4_K_M")
                                    Text("Q5_K_M").tag("Q5_K_M")
                                    Text("Q8_0").tag("Q8_0")
                                    Text("F16").tag("F16")
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                            }
                        } else if state.lmsTargetFormat == "safetensors" {
                            GridRow {
                                Text("Dtype")
                                    .frame(width: 100, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                                Picker("Dtype", selection: $state.safetensorsDtype) {
                                    Text("BF16").tag("bfloat16")
                                    Text("FP16").tag("float16")
                                    Text("FP32").tag("float32")
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                            }
                        }
                        GridRow {
                            Text("Destination")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("Default: ~/.lmstudio/models/<owner>/<model>", text: $state.customOutputPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("Choose…") {
                                    pickOutputFolder()
                                }
                            }
                        }
                        GridRow {
                            Text("")
                                .frame(width: 100)
                            Toggle("Move source (remove original after importing to LM Studio)", isOn: $state.moveToArchive)
                                .font(.caption)
                        }
                    }
                case .archive:
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        GridRow {
                            Text("Archive Folder")
                                .frame(width: 100, alignment: .trailing)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                TextField("/Volumes/ExternalDrive/ModelArchive", text: $state.archiveDir)
                                    .textFieldStyle(.roundedBorder)
                                Button("Choose…") {
                                    pickArchiveFolder()
                                }
                            }
                        }
                        GridRow {
                            Text("")
                                .frame(width: 100)
                            Toggle("Move to Archive (remove original from registry/disk after backup)", isOn: $state.moveToArchive)
                                .font(.caption)
                        }
                    }
                }

                if let warning = state.compatibilityWarning {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(warning)
                            .font(.caption)
                            .foregroundStyle(.primary)
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                }

                if state.status == .running {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(state.progressMessage.isEmpty ? "Running conversion…" : state.progressMessage)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                            Spacer()
                            if let frac = state.progressFraction {
                                Text("\(Int(frac * 100))%")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                        }
                        if let frac = state.progressFraction {
                            ProgressView(value: frac)
                                .progressViewStyle(.linear)
                        } else {
                            ProgressView()
                                .progressViewStyle(.linear)
                        }
                    }
                    .padding(10)
                    .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                }

                HStack {
                    if state.status == .running {
                        Button(role: .destructive) {
                            state.stopConversion()
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)

                        ProgressView()
                            .controlSize(.small)
                            .padding(.leading, 4)
                    } else {
                        Button {
                            state.startConversion()
                        } label: {
                            Label(
                                state.targetFormat == .archive
                                    ? (state.moveToArchive ? "Move to Archive" : "Copy to Archive")
                                    : "Convert Model",
                                systemImage: state.targetFormat == .archive ? "archivebox" : "arrow.triangle.2.circlepath"
                            )
                            .foregroundStyle(.white)
                            .padding(.horizontal, 6)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(state.sourcePath.isEmpty ? .gray : .accentColor)
                        .disabled(state.sourcePath.isEmpty)
                    }

                    Spacer()

                    StatusBadge(status: state.status)
                }
            }
            .padding(6)
        } label: {
            Label("Target Format", systemImage: "arrow.right.circle")
                .font(.headline)
        }
    }

    private var consoleCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Spacer()
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(state.logOutput, forType: .string)
                    }
                    .controlSize(.small)
                    .disabled(state.logOutput.isEmpty)

                    Button("Clear") {
                        state.logOutput = ""
                        if state.status != .running {
                            state.status = .idle
                        }
                    }
                    .controlSize(.small)
                    .disabled(state.logOutput.isEmpty)
                }

                ScrollViewReader { proxy in
                    ScrollView {
                        Text(state.logOutput.isEmpty ? "Select a model and click Convert Model to begin." : state.logOutput)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(state.logOutput.isEmpty ? .secondary : .primary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                            .id("logEnd")
                    }
                    .frame(minHeight: 160, maxHeight: 240)
                    .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                    )
                    .onChange(of: state.logOutput) {
                        proxy.scrollTo("logEnd", anchor: .bottom)
                    }
                }
            }
            .padding(6)
        } label: {
            Label("Conversion Log", systemImage: "apple.terminal")
                .font(.headline)
        }
    }

    // MARK: - Helpers

    private func pickSourcePath() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select a .gguf file or a Hugging Face / MLX model directory"
        if panel.runModal() == .OK, let url = panel.url {
            state.selectSource(url.path)
        }
    }

    private func pickArchiveFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select an Archive / Backup folder (e.g. on an external drive)"
        if panel.runModal() == .OK, let url = panel.url {
            state.archiveDir = url.path
        }
    }

    private func pickOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select an output directory"
        if panel.runModal() == .OK, let url = panel.url {
            state.customOutputPath = url.path
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) else {
            return false
        }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
            Task { @MainActor in
                state.selectSource(url.path)
            }
        }
        return true
    }

    private func formatColor(_ fmt: String?) -> Color {
        switch fmt {
        case "GGUF", "Ollama (GGUF)": return .blue
        case "MLX", "Ollama (MLX/Safetensors)", "Ollama (Custom/MLX)": return .purple
        case "HuggingFace Safetensors", "HuggingFace Hub": return .orange
        default: return .gray
        }
    }
}

struct BadgeView: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

struct InfoPill: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 4) {
            Text(label + ":")
                .foregroundStyle(.secondary)
            Text(value)
                .fontWeight(.medium)
        }
        .font(.caption)
    }
}

struct StatusBadge: View {
    let status: ConversionStatus

    var color: Color {
        switch status {
        case .idle: return .secondary
        case .running: return .blue
        case .success: return .green
        case .failed: return .red
        }
    }

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text(status.rawValue)
                .font(.caption.weight(.medium))
                .foregroundStyle(color)
        }
    }
}

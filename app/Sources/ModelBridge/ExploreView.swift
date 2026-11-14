import SwiftUI
import Observation
import AppKit

struct HFSibling: Codable, Hashable, Identifiable {
    let rfilename: String
    let size: Int64?

    var id: String { rfilename }
}

struct HFModelSummary: Codable, Identifiable, Hashable {
    let id: String
    let author: String?
    let likes: Int?
    let downloads: Int?
    let tags: [String]?
    let pipeline_tag: String?
    let createdAt: String?
    let lastModified: String?
    let siblings: [HFSibling]?

    var shortName: String {
        id.split(separator: "/").last.map(String.init) ?? id
    }

    var formatBadges: [String] {
        guard let tags else { return [] }
        var out: [String] = []
        for t in ["mlx", "gguf", "safetensors", "onnx", "4-bit", "8-bit"] where tags.contains(t) {
            out.append(t.uppercased())
        }
        return Array(out.prefix(4))
    }
}

@Observable
@MainActor
final class HFExplorerState {
    var query: String = ""
    var filterFormat: String = "all"
    var sortBy: String = "downloads"

    var results: [HFModelSummary] = []
    var selectedModel: HFModelSummary?
    var selectedDetails: HFModelSummary?
    var readmeText: String = ""

    var isSearching: Bool = false
    var isLoadingDetails: Bool = false
    var errorMessage: String?

    func search() async {
        isSearching = true
        errorMessage = nil
        defer { isSearching = false }

        var components = URLComponents(string: "https://huggingface.co/api/models")!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "sort", value: sortBy),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "35"),
            URLQueryItem(name: "full", value: "true")
        ]

        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            items.append(URLQueryItem(name: "search", value: trimmed))
        } else if filterFormat == "all" {
            // Default curated/popular MLX & GGUF models when empty
            items.append(URLQueryItem(name: "search", value: "mlx"))
        }

        if filterFormat != "all" {
            items.append(URLQueryItem(name: "filter", value: filterFormat))
        }

        components.queryItems = items
        guard let url = components.url else { return }

        do {
            var req = URLRequest(url: url)
            req.timeoutInterval = 12
            let (data, response) = try await URLSession.shared.data(for: req)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                errorMessage = "Hugging Face API returned HTTP \(http.statusCode)"
                return
            }
            let decoded = try JSONDecoder().decode([HFModelSummary].self, from: data)
            self.results = decoded
            if let first = decoded.first, selectedModel == nil {
                await select(first)
            }
        } catch {
            self.errorMessage = error.localizedDescription
        }
    }

    func select(_ model: HFModelSummary) async {
        selectedModel = model
        selectedDetails = model
        readmeText = ""
        isLoadingDetails = true
        defer { isLoadingDetails = false }

        async let detailTask: HFModelSummary? = fetchModelDetails(id: model.id)
        async let readmeTask: String = fetchReadme(id: model.id)

        let (details, readme) = await (detailTask, readmeTask)
        if selectedModel?.id == model.id {
            if let details {
                self.selectedDetails = details
            }
            self.readmeText = readme
        }
    }

    private func fetchModelDetails(id: String) async -> HFModelSummary? {
        guard let url = URL(string: "https://huggingface.co/api/models/\(id)?blobs=true") else { return nil }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            return try JSONDecoder().decode(HFModelSummary.self, from: data)
        } catch {
            return nil
        }
    }

    private func fetchReadme(id: String) async -> String {
        guard let url = URL(string: "https://huggingface.co/\(id)/raw/main/README.md") else { return "" }
        do {
            let (data, resp) = try await URLSession.shared.data(from: url)
            if let http = resp as? HTTPURLResponse, http.statusCode != 200 { return "" }
            let raw = String(data: data, encoding: .utf8) ?? ""
            return stripYAMLFrontmatter(raw)
        } catch {
            return ""
        }
    }

    private func stripYAMLFrontmatter(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("---") else { return trimmed }
        let lines = trimmed.components(separatedBy: .newlines)
        if let endIdx = lines.dropFirst().firstIndex(where: { $0.trimmingCharacters(in: .whitespaces) == "---" }) {
            return lines.suffix(from: endIdx + 1).joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return trimmed
    }
}

struct ExploreView: View {
    @Bindable var bridgeState: BridgeState
    var onSelectForBridge: (String) -> Void

    @State private var explorer = HFExplorerState()
    @State private var readmeHeight: CGFloat = 420

    var body: some View {
        HSplitView {
            leftSearchPane
                .frame(minWidth: 300, idealWidth: 340, maxWidth: 420)

            rightDetailPane
                .frame(minWidth: 480, maxWidth: .infinity, maxHeight: .infinity)
        }
        .task {
            if explorer.results.isEmpty {
                await explorer.search()
            }
        }
    }

    // MARK: - Left Search Pane

    private var leftSearchPane: some View {
        VStack(spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search Hugging Face Hub…", text: $explorer.query)
                    .textFieldStyle(.plain)
                    .onSubmit {
                        Task { await explorer.search() }
                    }
                if !explorer.query.isEmpty {
                    Button {
                        explorer.query = ""
                        Task { await explorer.search() }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
            )

            HStack(spacing: 8) {
                Picker("Format", selection: $explorer.filterFormat) {
                    Text("All").tag("all")
                    Text("MLX").tag("mlx")
                    Text("GGUF").tag("gguf")
                    Text("Safetensors").tag("safetensors")
                    Text("ONNX").tag("onnx")
                }
                .labelsHidden()
                .onChange(of: explorer.filterFormat) {
                    Task { await explorer.search() }
                }

                Picker("Sort", selection: $explorer.sortBy) {
                    Text("Downloads").tag("downloads")
                    Text("Likes").tag("likes")
                    Text("Updated").tag("lastModified")
                    Text("Newest").tag("createdAt")
                }
                .labelsHidden()
                .onChange(of: explorer.sortBy) {
                    Task { await explorer.search() }
                }

                if explorer.isSearching {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            if let err = explorer.errorMessage {
                Label(err, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            List(explorer.results, selection: Binding(
                get: { explorer.selectedModel?.id },
                set: { newId in
                    if let newId, let match = explorer.results.first(where: { $0.id == newId }) {
                        Task { await explorer.select(match) }
                    }
                }
            )) { model in
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.shortName)
                        .font(.system(.callout, weight: .semibold))
                        .lineLimit(1)

                    Text(model.id)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        ForEach(model.formatBadges, id: \.self) { badge in
                            BadgeView(text: badge, color: badgeColor(badge))
                        }
                        Spacer()
                        if let rel = relativeDate(model.lastModified ?? model.createdAt) {
                            Text(rel)
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                        if let dl = model.downloads {
                            Label(formatCount(dl), systemImage: "arrow.down.circle")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        if let likes = model.likes {
                            Label(formatCount(likes), systemImage: "heart")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.vertical, 4)
                .help(dateTooltip(created: model.createdAt, updated: model.lastModified))
                .tag(model.id)
            }
            .listStyle(.inset)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Right Detail Pane

    @ViewBuilder
    private var rightDetailPane: some View {
        if let model = explorer.selectedDetails ?? explorer.selectedModel {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(model.shortName)
                                .font(.title2.weight(.bold))
                            Text(model.id)
                                .font(.system(.callout, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                        }

                        Spacer()

                        if let url = URL(string: "https://huggingface.co/\(model.id)") {
                            Link(destination: url) {
                                Label("Open on Web", systemImage: "arrow.up.right.square")
                            }
                            .buttonStyle(.bordered)
                        }
                    }

                    // Stats & Badges
                    HStack(spacing: 10) {
                        if let dl = model.downloads {
                            InfoPill(label: "Downloads", value: formatCount(dl))
                        }
                        if let likes = model.likes {
                            InfoPill(label: "Likes", value: formatCount(likes))
                        }
                        if let pipe = model.pipeline_tag {
                            InfoPill(label: "Task", value: pipe)
                        }
                        if let added = relativeDate(model.createdAt) {
                            InfoPill(label: "Added", value: added)
                                .help(formattedDate(model.createdAt) ?? "")
                        }
                        if let updated = relativeDate(model.lastModified) {
                            InfoPill(label: "Updated", value: updated)
                                .help(formattedDate(model.lastModified) ?? "")
                        }
                        ForEach(model.formatBadges, id: \.self) { badge in
                            BadgeView(text: badge, color: badgeColor(badge))
                        }
                    }

                    // Actions
                    HStack(spacing: 10) {
                        Button {
                            onSelectForBridge("hf:\(model.id)")
                        } label: {
                            Label("Load in Bridge & Convert", systemImage: "arrow.triangle.2.circlepath")
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.accentColor)

                        Button {
                            bridgeState.downloadFromHF(repoId: model.id, toLMStudio: true)
                            onSelectForBridge("hf:\(model.id)")
                        } label: {
                            Label("Download to LM Studio", systemImage: "arrow.down.to.line")
                        }
                        .buttonStyle(.bordered)
                        .disabled(bridgeState.status == .running)

                        if explorer.isLoadingDetails {
                            ProgressView()
                                .controlSize(.small)
                        }
                    }

                    // Weight Files
                    let weightFiles = (model.siblings ?? []).filter {
                        let n = $0.rfilename.lowercased()
                        return n.hasSuffix(".gguf") || n.hasSuffix(".safetensors") || n.hasSuffix(".onnx") || n.hasSuffix(".onnx_data")
                    }
                    if !weightFiles.isEmpty {
                        GroupBox {
                            VStack(alignment: .leading, spacing: 6) {
                                ForEach(weightFiles.prefix(20)) { file in
                                    HStack {
                                        Text(file.rfilename)
                                            .font(.system(.caption, design: .monospaced))
                                            .lineLimit(1)
                                        Spacer()
                                        if let sz = file.size {
                                            Text(formatBytes(sz))
                                                .font(.caption.monospacedDigit())
                                                .foregroundStyle(.secondary)
                                        }
                                        if file.rfilename.lowercased().hasSuffix(".gguf") {
                                            Button("Use File") {
                                                onSelectForBridge("hf:\(model.id)#\(file.rfilename)")
                                            }
                                            .controlSize(.mini)
                                            .buttonStyle(.bordered)
                                            .help("Load just this GGUF file in Bridge & Convert")
                                        }
                                    }
                                    if file.id != weightFiles.prefix(20).last?.id {
                                        Divider()
                                    }
                                }
                                if weightFiles.count > 20 {
                                    Text("+ \(weightFiles.count - 20) more weight files")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(4)
                        } label: {
                            Label("Model Files (\(weightFiles.count))", systemImage: "doc.on.doc")
                                .font(.headline)
                        }
                    }

                    // README
                    GroupBox {
                        if explorer.readmeText.isEmpty {
                            Text(explorer.isLoadingDetails ? "Loading README…" : "No README available.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(6)
                        } else {
                            ReadmeWebView(
                                markdown: explorer.readmeText,
                                repoId: model.id,
                                contentHeight: $readmeHeight
                            )
                            .frame(height: readmeHeight)
                        }
                    } label: {
                        Label("README", systemImage: "doc.text")
                            .font(.headline)
                    }
                }
                .padding(20)
            }
        } else {
            ContentUnavailableView(
                "Explore Hugging Face Hub",
                systemImage: "globe",
                description: Text("Search for MLX, GGUF, Safetensors, or ONNX models to inspect, download, or convert.")
            )
        }
    }

    // MARK: - Formatting

    private func badgeColor(_ badge: String) -> Color {
        switch badge {
        case "MLX": return .purple
        case "GGUF": return .blue
        case "SAFETENSORS": return .orange
        case "ONNX": return .teal
        default: return .secondary
        }
    }

    private func formatCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        if gb >= 0.1 { return String(format: "%.2f GB", gb) }
        let mb = Double(bytes) / 1_048_576.0
        return String(format: "%.1f MB", mb)
    }

    private func parseISODate(_ iso: String?) -> Date? {
        guard let iso, !iso.isEmpty else { return nil }
        let withFrac = ISO8601DateFormatter()
        withFrac.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = withFrac.date(from: iso) { return d }
        let basic = ISO8601DateFormatter()
        basic.formatOptions = [.withInternetDateTime]
        return basic.date(from: iso)
    }

    private func relativeDate(_ iso: String?) -> String? {
        guard let date = parseISODate(iso) else { return nil }
        let fmt = RelativeDateTimeFormatter()
        fmt.unitsStyle = .abbreviated
        return fmt.localizedString(for: date, relativeTo: Date())
    }

    private func formattedDate(_ iso: String?) -> String? {
        guard let date = parseISODate(iso) else { return nil }
        let fmt = DateFormatter()
        fmt.dateStyle = .medium
        fmt.timeStyle = .short
        return fmt.string(from: date)
    }

    private func dateTooltip(created: String?, updated: String?) -> String {
        var parts: [String] = []
        if let c = formattedDate(created) { parts.append("Added: \(c)") }
        if let u = formattedDate(updated) { parts.append("Updated: \(u)") }
        return parts.joined(separator: "\n")
    }
}

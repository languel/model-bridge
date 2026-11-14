import Foundation
import Observation

enum ConversionStatus: String {
    case idle = "Ready"
    case running = "Converting…"
    case success = "Completed"
    case failed = "Failed"
}

@Observable
@MainActor
final class BridgeState {
    var lmStudioModels: [LMStudioModel] = []
    var ollamaModels: [OllamaModel] = []
    var inspectedModel: InspectedModel?

    var sourcePath: String = ""
    var targetFormat: TargetFormat = .ollama

    // Target-specific options
    var ollamaModelName: String = ""
    var ollamaSystemPrompt: String = ""
    var ollamaNumCtx: Int = 32768
    var ollamaRenderer: String = ""
    var ollamaParser: String = ""
    var ollamaCapabilities: String = "completion"
    var moveToOllama: Bool = false
    var moveToArchive: Bool = false
    var archiveDir: String = UserDefaults.standard.string(forKey: "archiveDir") ?? (NSHomeDirectory() + "/ModelArchive") {
        didSet { UserDefaults.standard.set(archiveDir, forKey: "archiveDir") }
    }
    var mlxBits: Int = 4
    var ggufQuant: String = "Q4_K_M"
    var safetensorsDtype: String = "bfloat16"
    var onnxQuant: String = "q4f16"
    var customOutputPath: String = ""

    var status: ConversionStatus = .idle
    var logOutput: String = ""
    var progressFraction: Double? = nil
    var progressMessage: String = ""
    var isRefreshing: Bool = false
    var isInspecting: Bool = false

    private var activeProcess: Process?
    private var createdOutputPathOnStart: String?

    /// Resolves the path to `cli/model_converter.py`.
    var cliScriptPath: String {
        let candidates = [
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("cli/model_converter.py").path,
            Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("cli/model_converter.py").path,
            NSHomeDirectory() + "/dev/model-bridge/cli/model_converter.py"
        ]
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            return path
        }
        return NSHomeDirectory() + "/dev/model-bridge/cli/model_converter.py"
    }

    var pythonPath: String {
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            return path
        }
        return "/usr/bin/python3"
    }

    var compatibilityWarning: String? {
        guard let fmt = inspectedModel?.format else { return nil }
        if sourcePath.hasPrefix("ollama:") {
            if targetFormat == .ollama {
                return "Source is already in Ollama. Select Ollama MLX to convert to MLX, or export to Safetensors, GGUF, ONNX, LM Studio, or Archive."
            }
            if targetFormat == .lmstudio && inspectedModel?.has_single_gguf == false {
                return "This Ollama model uses split MLX/Safetensors layers. Exporting will create a portable bundle rather than a single GGUF."
            }
            return nil
        }
        if fmt == "MLX" && targetFormat == .gguf {
            return "MLX weights are already quantized. Export to Safetensors first, or use an unquantized Hugging Face Safetensors folder."
        }
        return nil
    }

    func refreshAll() async {
        isRefreshing = true
        defer { isRefreshing = false }

        async let lmTask: [LMStudioModel] = fetchJSON(args: ["list-lmstudio", "--json"])
        async let olTask: [OllamaModel] = fetchJSON(args: ["list-ollama", "--json"])

        let (lm, ol) = await (lmTask, olTask)
        self.lmStudioModels = lm
        self.ollamaModels = ol
    }

    func selectSource(_ path: String) {
        self.sourcePath = path
        if path.hasPrefix("ollama:") {
            if targetFormat == .ollama {
                targetFormat = .ollamaMLX
            }
        } else {
            let base = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
                .lowercased()
                .replacingOccurrences(of: "_", with: "-")
                .replacingOccurrences(of: " ", with: "-")
            self.ollamaModelName = base
            if targetFormat == .lmstudio {
                targetFormat = .ollama
            }
        }
        Task {
            await inspectSource(path)
        }
    }

    func inspectSource(_ path: String) async {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            self.inspectedModel = nil
            return
        }
        isInspecting = true
        defer { isInspecting = false }

        let (data, errStr) = await runCaptureWithErr(args: ["inspect", trimmed])
        guard let data else {
            let msg = errStr.isEmpty ? "Inspect command failed." : errStr
            self.inspectedModel = InspectedModel(
                format: nil, backend: nil, path: nil, name: nil,
                architecture: nil, model_type: nil, file_size_gb: nil,
                num_weight_shards: nil, n_tensors: nil, has_single_gguf: nil,
                quantization: nil, error: msg
            )
            return
        }

        do {
            let result = try JSONDecoder().decode(InspectedModel.self, from: data)
            self.inspectedModel = result

            // Auto-populate Ollama card defaults from architecture
            let combined = "\(result.architecture ?? "") \(result.model_type ?? "") \(result.name ?? "")".lowercased()
            if combined.contains("qwen3_5") || combined.contains("qwen3.5") {
                ollamaRenderer = "qwen3.5"
                ollamaParser = "qwen3.5"
                ollamaCapabilities = "completion, tools, thinking"
            } else if combined.contains("qwen3.8") || combined.contains("qwen3_8") {
                ollamaRenderer = "qwen3.8"
                ollamaParser = "qwen3.5"
                ollamaCapabilities = "completion, tools, thinking"
            } else if combined.contains("gemma4") {
                ollamaRenderer = "gemma4"
                ollamaParser = ""
                ollamaCapabilities = "completion"
            } else {
                ollamaRenderer = ""
                ollamaParser = ""
                ollamaCapabilities = "completion"
            }
        } catch {
            let raw = String(data: data, encoding: .utf8) ?? ""
            self.inspectedModel = InspectedModel(
                format: nil, backend: nil, path: nil, name: nil,
                architecture: nil, model_type: nil, file_size_gb: nil,
                num_weight_shards: nil, n_tensors: nil, has_single_gguf: nil,
                quantization: nil, error: "Decode error: \(error.localizedDescription) (\(raw.prefix(120)))"
            )
        }
    }

    func startConversion() {
        guard status != .running, !sourcePath.isEmpty else { return }

        var args: [String] = [cliScriptPath]

        if targetFormat == .archive {
            args.append(contentsOf: ["archive", sourcePath, "--dest", archiveDir])
            if moveToArchive {
                args.append("--move")
            }
        } else {
            args.append(contentsOf: ["convert", sourcePath, "--to", targetFormat.rawValue])

            switch targetFormat {
            case .ollama, .ollamaMLX:
                if targetFormat == .ollamaMLX {
                    args.append(contentsOf: ["--bits", String(mlxBits)])
                    let out = customOutputPath.trimmingCharacters(in: .whitespaces)
                    if !out.isEmpty {
                        args.append(contentsOf: ["--output", out])
                    }
                }
                if !ollamaModelName.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--name", ollamaModelName.trimmingCharacters(in: .whitespaces)])
                }
                if !ollamaSystemPrompt.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--system", ollamaSystemPrompt])
                }
                args.append(contentsOf: ["--ctx", String(ollamaNumCtx)])
                if !ollamaRenderer.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--renderer", ollamaRenderer.trimmingCharacters(in: .whitespaces)])
                }
                if !ollamaParser.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--parser", ollamaParser.trimmingCharacters(in: .whitespaces)])
                }
                if !ollamaCapabilities.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--capabilities", ollamaCapabilities.trimmingCharacters(in: .whitespaces)])
                }
                if moveToOllama {
                    args.append("--move")
                }
            case .mlx:
                args.append(contentsOf: ["--bits", String(mlxBits)])
                if !customOutputPath.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--output", customOutputPath.trimmingCharacters(in: .whitespaces)])
                }
            case .gguf:
                args.append(contentsOf: ["--quant", ggufQuant])
                if !customOutputPath.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--output", customOutputPath.trimmingCharacters(in: .whitespaces)])
                }
                if sourcePath.hasPrefix("ollama:") && moveToArchive {
                    args.append("--move")
                }
            case .safetensors:
                args.append(contentsOf: ["--dtype", safetensorsDtype])
                if !customOutputPath.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--output", customOutputPath.trimmingCharacters(in: .whitespaces)])
                }
            case .onnx:
                args.append(contentsOf: ["--onnx-quant", onnxQuant])
                if !customOutputPath.trimmingCharacters(in: .whitespaces).isEmpty {
                    args.append(contentsOf: ["--output", customOutputPath.trimmingCharacters(in: .whitespaces)])
                }
            case .lmstudio:
                if moveToArchive {
                    args.append("--move")
                }
            case .archive:
                break
            }
        }

        runStreaming(args: args)
    }

    func downloadFromHF(repoId: String, toLMStudio: Bool = true) {
        guard status != .running else { return }
        var args = [cliScriptPath, "download", repoId]
        if toLMStudio {
            args.append("--to-lmstudio")
        }
        runStreaming(args: args)
    }

    private func runStreaming(args: [String]) {
        status = .running
        progressFraction = nil
        progressMessage = "Starting…"
        let outTrimmed = customOutputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !outTrimmed.isEmpty && !FileManager.default.fileExists(atPath: outTrimmed) {
            createdOutputPathOnStart = outTrimmed
        } else {
            createdOutputPathOnStart = nil
        }
        logOutput = "$ python3 \(args.dropFirst().joined(separator: " "))\n\n"

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = args

        var env = ProcessInfo.processInfo.environment
        let existingPath = env["PATH"] ?? "/usr/bin:/bin"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:\(existingPath)"
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.logOutput.append(chunk)
                self.parseProgress(from: chunk)
            }
        }

        process.terminationHandler = { [weak self] proc in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.activeProcess = nil
                self.status = proc.terminationStatus == 0 ? .success : .failed
                self.progressFraction = proc.terminationStatus == 0 ? 1.0 : nil
                if proc.terminationStatus == 0 {
                    self.progressMessage = "Completed"
                }
                self.createdOutputPathOnStart = nil
                self.logOutput.append("\n[Process exited with code \(proc.terminationStatus)]\n")
                if proc.terminationStatus == 0 {
                    await self.refreshAll()
                }
            }
        }

        do {
            self.activeProcess = process
            try process.run()
        } catch {
            self.activeProcess = nil
            self.status = .failed
            self.progressFraction = nil
            self.progressMessage = "Failed to launch"
            self.logOutput.append("Failed to launch process: \(error.localizedDescription)\n")
        }
    }

    private func parseProgress(from chunk: String) {
        let lines = chunk.components(separatedBy: CharacterSet.newlines)
        for raw in lines {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }

            // Match percentage like "45%|" or " 72%"
            if let range = line.range(of: #"\b(\d{1,3})%"#, options: .regularExpression),
               let pct = Int(line[range].dropLast()), (0...100).contains(pct) {
                self.progressFraction = Double(pct) / 100.0
            }

            // Match Step X/Y
            if let stepRange = line.range(of: #"Step\s+(\d+)/(\d+)"#, options: .regularExpression) {
                let sub = String(line[stepRange])
                let nums = sub.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap(Int.init)
                if nums.count == 2, nums[1] > 0 {
                    self.progressFraction = max(0.05, Double(nums[0] - 1) / Double(nums[1]))
                }
            }

            if line.hasPrefix("[*]") {
                self.progressMessage = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("[+]") {
                self.progressMessage = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            }
        }
    }

    func stopConversion() {
        if let proc = activeProcess {
            let pid = proc.processIdentifier
            proc.terminate()
            // Also signal child process group if present
            kill(-pid, SIGTERM)
        }
        activeProcess = nil
        status = .failed
        progressFraction = nil
        progressMessage = "Cancelled & cleaned up"
        cleanupUnfinishedArtifacts()
        logOutput.append("\n[Cancelled by user — unfinished temporary/partial outputs cleaned up]\n")
    }

    private func cleanupUnfinishedArtifacts() {
        let fm = FileManager.default
        if let outPath = createdOutputPathOnStart, fm.fileExists(atPath: outPath) {
            try? fm.removeItem(atPath: outPath)
        }
        createdOutputPathOnStart = nil

        if let tmpItems = try? fm.contentsOfDirectory(atPath: "/tmp") {
            let prefixes = ["hf_from_gguf_", "hf_dequant_", "ollama_mlx_", "Modelfile_"]
            for item in tmpItems where prefixes.contains(where: { item.hasPrefix($0) }) {
                try? fm.removeItem(atPath: "/tmp/\(item)")
            }
        }
    }

    func deleteOllamaModel(_ name: String) async {
        logOutput.append("\n[*] Deleting Ollama model '\(name)'...\n")
        let ok = await runSimple(args: ["delete-ollama", name])
        logOutput.append(ok ? "[+] Deleted '\(name)' from Ollama.\n" : "[-] Failed to delete '\(name)'.\n")
        if sourcePath == "ollama:\(name)" {
            sourcePath = ""
            inspectedModel = nil
        }
        await refreshAll()
    }

    func deleteLMStudioModel(_ path: String) async {
        logOutput.append("\n[*] Deleting LM Studio model '\(path)'...\n")
        let ok = await runSimple(args: ["delete-lmstudio", path])
        logOutput.append(ok ? "[+] Deleted from LM Studio.\n" : "[-] Failed to delete from LM Studio.\n")
        if sourcePath == path {
            sourcePath = ""
            inspectedModel = nil
        }
        await refreshAll()
    }

    private func runSimple(args: [String]) async -> Bool {
        let py = pythonPath
        let script = cliScriptPath
        return await Task.detached(priority: .userInitiated) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: py)
            proc.arguments = [script] + args
            var env = ProcessInfo.processInfo.environment
            let existingPath = env["PATH"] ?? "/usr/bin:/bin"
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:\(existingPath)"
            proc.environment = env
            do {
                try proc.run()
                proc.waitUntilExit()
                return proc.terminationStatus == 0
            } catch {
                return false
            }
        }.value
    }

    // MARK: - Subprocess JSON Helpers

    private func fetchJSON<T: Decodable & Sendable>(args: [String]) async -> [T] {
        guard let data = await runCapture(args: args) else { return [] }
        return (try? JSONDecoder().decode([T].self, from: data)) ?? []
    }

    private func fetchSingleJSON<T: Decodable & Sendable>(args: [String]) async -> T? {
        guard let data = await runCapture(args: args) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }

    private func runCapture(args: [String]) async -> Data? {
        let (data, _) = await runCaptureWithErr(args: args)
        return data
    }

    private func runCaptureWithErr(args: [String]) async -> (Data?, String) {
        let py = pythonPath
        let script = cliScriptPath
        return await Task.detached(priority: .userInitiated) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: py)
            proc.arguments = [script] + args

            var env = ProcessInfo.processInfo.environment
            let existingPath = env["PATH"] ?? "/usr/bin:/bin"
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:\(existingPath)"
            proc.environment = env

            let outPipe = Pipe()
            let errPipe = Pipe()
            proc.standardOutput = outPipe
            proc.standardError = errPipe

            do {
                try proc.run()
                let data = outPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                let errStr = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                return (proc.terminationStatus == 0 ? data : nil, errStr)
            } catch {
                return (nil, error.localizedDescription)
            }
        }.value
    }
}

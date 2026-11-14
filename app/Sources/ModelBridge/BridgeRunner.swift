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
    var isRefreshing: Bool = false

    private var activeProcess: Process?

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
            if targetFormat == .ollama || targetFormat == .mlx || targetFormat == .onnx {
                return "To export an Ollama model, select Safetensors, GGUF, LM Studio, or Archive."
            }
            if targetFormat == .lmstudio && inspectedModel?.has_single_gguf == false {
                return "This Ollama model uses split MLX/Safetensors layers. Exporting will create a portable bundle rather than a single GGUF."
            }
            return nil
        }
        if fmt == "GGUF" && (targetFormat == .mlx || targetFormat == .onnx) {
            return "Export to Safetensors first (supported for BF16/F16/Q8 GGUF), or use a Hugging Face Safetensors folder."
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
            if targetFormat == .ollama || targetFormat == .mlx || targetFormat == .onnx {
                targetFormat = .archive
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
        guard !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            self.inspectedModel = nil
            return
        }
        let result: InspectedModel? = await fetchSingleJSON(args: ["inspect", path])
        self.inspectedModel = result

        // Auto-populate Ollama card defaults from architecture
        if let info = result {
            let combined = "\(info.architecture ?? "") \(info.model_type ?? "") \(info.name ?? "")".lowercased()
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
            case .ollama:
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

        status = .running
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
                self?.logOutput.append(chunk)
            }
        }

        process.terminationHandler = { [weak self] proc in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.activeProcess = nil
                self.status = proc.terminationStatus == 0 ? .success : .failed
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
            self.logOutput.append("Failed to launch process: \(error.localizedDescription)\n")
        }
    }

    func stopConversion() {
        activeProcess?.terminate()
        activeProcess = nil
        status = .failed
        logOutput.append("\n[Cancelled by user]\n")
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
            proc.standardOutput = outPipe
            proc.standardError = Pipe()

            do {
                try proc.run()
                let data = outPipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                return proc.terminationStatus == 0 ? data : nil
            } catch {
                return nil
            }
        }.value
    }
}

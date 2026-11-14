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
    var mlxBits: Int = 4
    var ggufQuant: String = "Q4_K_M"
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
        if fmt == "GGUF" && (targetFormat == .mlx || targetFormat == .onnx) {
            return "Quantized GGUF → \(targetFormat.displayName) is lossy/unsupported. Select a Hugging Face Safetensors directory instead."
        }
        if fmt == "MLX" && (targetFormat == .ollama || targetFormat == .gguf || targetFormat == .onnx) {
            return "MLX weights are already Apple Silicon quantized. Use an unquantized Hugging Face Safetensors folder for \(targetFormat.displayName) export."
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
        if ollamaModelName.isEmpty {
            let base = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
                .lowercased()
                .replacingOccurrences(of: "_", with: "-")
                .replacingOccurrences(of: " ", with: "-")
            self.ollamaModelName = base
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
    }

    func startConversion() {
        guard status != .running, !sourcePath.isEmpty else { return }

        var args = [cliScriptPath, "convert", sourcePath, "--to", targetFormat.rawValue]

        switch targetFormat {
        case .ollama:
            if !ollamaModelName.trimmingCharacters(in: .whitespaces).isEmpty {
                args.append(contentsOf: ["--name", ollamaModelName.trimmingCharacters(in: .whitespaces)])
            }
            if !ollamaSystemPrompt.trimmingCharacters(in: .whitespaces).isEmpty {
                args.append(contentsOf: ["--system", ollamaSystemPrompt])
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
        case .onnx:
            if !customOutputPath.trimmingCharacters(in: .whitespaces).isEmpty {
                args.append(contentsOf: ["--output", customOutputPath.trimmingCharacters(in: .whitespaces)])
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

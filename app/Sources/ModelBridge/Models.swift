import Foundation

enum TargetFormat: String, CaseIterable, Identifiable {
    case ollama = "ollama"
    case mlx = "mlx"
    case gguf = "gguf"
    case onnx = "onnx"
    case lmstudio = "lmstudio"
    case archive = "archive"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .ollama: return "Ollama"
        case .mlx: return "MLX"
        case .gguf: return "GGUF"
        case .onnx: return "ONNX"
        case .lmstudio: return "LM Studio"
        case .archive: return "Archive"
        }
    }

    var subtitle: String {
        switch self {
        case .ollama: return "Import GGUF, MLX, or HF Safetensors into Ollama with a full Model Card"
        case .mlx: return "Apple Silicon Metal native weights (4/8/16-bit)"
        case .gguf: return "Quantize GGUF via llama-quantize, or extract GGUF from Ollama"
        case .onnx: return "Browser WebGPU causal LM (q4f16)"
        case .lmstudio: return "Export Ollama model directly into LM Studio (~/.lmstudio/models)"
        case .archive: return "Copy or move model (with Modelfile/weights) to a backup or external drive"
        }
    }
}

struct LMStudioModel: Codable, Identifiable, Hashable {
    let lmstudio_id: String?
    let name: String?
    let format: String?
    let architecture: String?
    let file_size_gb: Double?
    let path: String

    var id: String { path }

    var displayName: String {
        lmstudio_id ?? name ?? URL(fileURLWithPath: path).lastPathComponent
    }
}

struct OllamaModel: Codable, Identifiable, Hashable {
    let name: String
    let id: String
    let size: String
    let modified: String
}

struct QuantizationInfo: Codable, Hashable {
    let group_size: Int?
    let bits: Int?
    let mode: String?
}

struct InspectedModel: Codable, Hashable {
    let format: String?
    let backend: String?
    let path: String?
    let name: String?
    let architecture: String?
    let model_type: String?
    let file_size_gb: Double?
    let num_weight_shards: Int?
    let n_tensors: Int?
    let has_single_gguf: Bool?
    let quantization: QuantizationInfo?
    let error: String?
}

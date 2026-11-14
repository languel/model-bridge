import Foundation

enum TargetFormat: String, CaseIterable, Identifiable {
    case ollama = "ollama"
    case mlx = "mlx"
    case gguf = "gguf"
    case safetensors = "safetensors"
    case onnx = "onnx"
    case lmstudio = "lmstudio"
    case archive = "archive"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .ollama: return "Ollama"
        case .mlx: return "MLX"
        case .gguf: return "GGUF"
        case .safetensors: return "Safetensors"
        case .onnx: return "ONNX"
        case .lmstudio: return "LM Studio"
        case .archive: return "Archive"
        }
    }

    var subtitle: String {
        switch self {
        case .ollama: return "Register or convert any model (GGUF, MLX, or Hugging Face Safetensors) into Ollama"
        case .mlx: return "Apple Silicon Metal native weights (4/8/16-bit) — chains HF, Safetensors, GGUF, or Ollama"
        case .gguf: return "Quantize GGUF via llama-quantize, or convert HF / Safetensors / MLX to GGUF"
        case .safetensors: return "Export MLX or BF16/F16/Q8 GGUF to Hugging Face Safetensors"
        case .onnx: return "Browser WebGPU causal LM (q4f16) — chains HF, Safetensors, MLX, GGUF, or Ollama"
        case .lmstudio: return "Convert/import any source (HF Hub, Safetensors, GGUF, Ollama) into LM Studio (~/.lmstudio/models)"
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

struct EditableModelCard: Codable {
    var kind: String?
    var name: String?
    var path: String?
    var renderer: String?
    var parser: String?
    var capabilities: String?
    var num_ctx: Int?
    var temperature: Double?
    var top_p: Double?
    var top_k: Int?
    var system: String?
    var modelfile: String?
    var error: String?
}

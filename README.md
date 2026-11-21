# Model Bridge

Inspect, plan, and convert local LLM formats across **GGUF**, **Safetensors**, **MLX (Apple Silicon)**, **Ollama**, **LM Studio**, and **ONNX (WebGPU)**.

- **Web App (zero-install, 100% in-browser)**: [languel.github.io/model-bridge](https://languel.github.io/model-bridge/)
- **CLI**: `./bin/model-convert` (Python 3, stdlib-only for inspection & Ollama management)
- **macOS App**: Native SwiftUI wrapper (`ModelBridge.app`)

## Quick Start

### 1. Browser Web App
Open **[languel.github.io/model-bridge](https://languel.github.io/model-bridge/)** — everything runs locally in your browser with zero uploads:
- **Zero-copy local inspection**: Drop `.gguf`, `.safetensors`, `config.json`, or a model folder to parse binary headers & tensor tables without loading weights into RAM.
- **Hugging Face Hub Explorer**: Search repos, inspect configs/files, and check quantization variants.
- **Memory & Speed Estimator**: Estimate Unified Memory / VRAM usage (weights + KV cache) and tok/s ceiling on Apple Silicon (M1–M5) and GPUs.
- **Modelfile & CLI Builder**: Generate Ollama `Modelfile`s and copy-paste conversion commands.

### 2. CLI
```bash
git clone https://github.com/languel/model-bridge.git
cd model-bridge

# Inventory local models
./bin/model-convert list-lmstudio
./bin/model-convert list-ollama

# Inspect any local GGUF, HF/MLX folder, Ollama model, or HF repo
./bin/model-convert inspect /path/to/model.gguf
./bin/model-convert inspect ollama:qwen3:8b
./bin/model-convert inspect hf:mlx-community/Qwen3-8B-4bit

# Register GGUF into Ollama (zero-copy Modelfile)
./bin/model-convert convert /path/to/model.gguf --to ollama --name my-model

# Convert Hugging Face / Safetensors to 4-bit MLX
./bin/model-convert convert owner/repo --to mlx --bits 4
```

### 3. macOS SwiftUI App
```bash
./build_app.sh
open ModelBridge.app
```

## Documentation
- [Setup & New Machine Guide](docs/setup.md) — Homebrew dependencies, optional `llama.cpp` converter venv, and building the macOS app.
- [CLI Reference & Conversion Matrix](docs/cli-reference.md) — Complete subcommand reference, flags, and supported source → target routes.
- [Technical Architecture](docs/architecture.md) — GGUF/Safetensors binary parsing, Ollama v1/v2 manifests, MLX pipelines, and Web App internals.

## License
[MIT](LICENSE)

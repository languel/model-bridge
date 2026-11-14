# Model Bridge (Apple Silicon)

A unified CLI and native macOS SwiftUI app for inspecting and converting LLM formats on Apple Silicon (M1–M5).

## Features
- **Auto-Discovery**: Scans local **LM Studio** (`~/.lmstudio/models`) and **Ollama** registries.
- **Zero-Copy Inspection**: Parses GGUF v2/v3 headers and Hugging Face / MLX `config.json` metadata without loading weights into RAM.
- **Format Conversions**:
  - **GGUF $\rightarrow$ Ollama**: Instant zero-copy registration via Modelfile.
  - **Hugging Face $\rightarrow$ MLX**: Apple Silicon Metal native 4-bit, 8-bit, or BF16 via `mlx_lm.convert`.
  - **GGUF $\rightarrow$ Quantized GGUF**: `Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ4_NL` via `llama-quantize`.
  - **Hugging Face $\rightarrow$ ONNX (WebGPU)**: Causal LM `q4f16` export.

## Quick Start

### macOS SwiftUI App
```bash
./build_app.sh
open ModelBridge.app
```

### CLI
```bash
# List LM Studio and Ollama models
./bin/model-convert list-lmstudio
./bin/model-convert list-ollama

# Inspect any model
./bin/model-convert inspect ~/.lmstudio/models/npario/humanizer-12b-mlx-8bit

# Convert / Register GGUF into Ollama
./bin/model-convert convert /path/to/model.gguf --to ollama --name my-model

# Convert Hugging Face model to 4-bit MLX
./bin/model-convert convert /path/to/hf-model --to mlx --bits 4
```

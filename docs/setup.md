# Setup & New Machine Guide

This repository tracks **only tools, scripts, docs, and the web app** — never model weights or local registries.

## Prerequisites (macOS / Apple Silicon)

| Tool | Required For | Install |
| --- | --- | --- |
| Python 3.10+ | CLI (`./bin/model-convert`) | Preinstalled or `brew install python` |
| Ollama | `list-ollama`, `convert --to ollama`, `export-ollama`, `edit-card` | `brew install ollama` or [ollama.com](https://ollama.com) |
| `mlx-lm` | HF/Safetensors ↔ MLX conversions, HF Hub downloads | `brew install mlx-lm` |
| `llama.cpp` | `llama-quantize` (GGUF quantization) & `llama-cli` (testing) | `brew install llama.cpp` |
| Xcode / Swift 5.9+ | Building native `ModelBridge.app` | `xcode-select --install` |

### Quick Install via Homebrew
```bash
brew install ollama mlx-lm llama.cpp
```

## Optional: HF/Safetensors → GGUF Converter (`convert_hf_to_gguf.py`)

Direct `Safetensors/MLX → GGUF` and `HF → Ollama GGUF` pipelines use `llama.cpp`'s `convert_hf_to_gguf.py`. Because `tools/` and `.gguf-venv/` are intentionally gitignored, recreate them once on a new machine:

```bash
# 1. Fetch llama.cpp conversion scripts (sparse checkout)
mkdir -p tools/llama_cpp_convert
git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp.git /tmp/llama-cpp-sparse
cd /tmp/llama-cpp-sparse
git sparse-checkout set convert_hf_to_gguf.py gguf-py requirements conversion
cp -R convert_hf_to_gguf.py gguf-py requirements conversion "$OLDPWD/tools/llama_cpp_convert/"
cd "$OLDPWD" && rm -rf /tmp/llama-cpp-sparse

# 2. Create isolated Python venv
python3 -m venv .gguf-venv
.gguf-venv/bin/pip install --upgrade pip
.gguf-venv/bin/pip install -e tools/llama_cpp_convert/gguf-py torch safetensors transformers sentencepiece protobuf numpy
```

`cli/model_converter.py` automatically detects `.gguf-venv/bin/python` and `tools/llama_cpp_convert/convert_hf_to_gguf.py` relative to the repo root.

## Optional: ONNX (WebGPU) Export

For `--to onnx` exports (Transformers.js / WebGPU):
```bash
pip install optimum[onnxruntime] onnxruntime transformers
```

## Building the Native macOS App

```bash
./build_app.sh
open ModelBridge.app
```

`build_app.sh` runs `swift build -c release` in `app/`, packages `ModelBridge.app` with `AppIcon.icns` and `Info.plist`, and keeps all build artifacts gitignored.

## Running the Web App Locally

The web app in `web/` is pure static HTML/CSS/JS with zero build step:
```bash
python3 -m http.server 8080 --directory web
# Open http://localhost:8080
```

To connect the web app to your local Ollama daemon from the browser, start Ollama with CORS allowed:
```bash
OLLAMA_ORIGINS="*" ollama serve
```

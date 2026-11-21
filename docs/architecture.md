# Technical Architecture

Model Bridge has three surfaces sharing the same metadata conventions and conversion logic:

```
model-bridge/
├── cli/model_converter.py   # Core Python engine & CLI (bin/model-convert)
├── app/                     # Native macOS SwiftUI app (SwiftPM -> ModelBridge.app)
├── web/                     # 100% client-side static Web App (GitHub Pages)
└── docs/                    # Technical documentation
```

## 1. Zero-Copy Binary Inspection

Both the Python CLI and the Browser Web App inspect multi-gigabyte model files without reading weight tensors into memory.

### GGUF (v2 / v3)
- Reads the 24-byte header (`magic = "GGUF"`, `uint32 version`, `uint64 tensor_count`, `uint64 metadata_kv_count`).
- Streams only the metadata KV block (and optionally tensor info headers) using bounded slices (`File.slice()` in the browser, `f.read()` in Python).
- Extracts architecture (`general.architecture`), model name (`general.name`), file type / quantization (`general.file_type`), context length (`<arch>.context_length`), layer/head counts (`<arch>.block_count`, `<arch>.attention.head_count`, `<arch>.attention.head_count_kv`), embedding dimension, and Jinja chat templates (`tokenizer.chat_template`).

### Safetensors
- Reads the 8-byte little-endian `uint64` header length `N` at byte offset `0`.
- Reads bytes `[8, 8 + N)` as UTF-8 JSON containing every tensor's `dtype`, `shape`, and `data_offsets`, plus `__metadata__`.
- Computes exact parameter count, byte breakdown by dtype (`BF16`, `F16`, `F32`, `F8_E4M3`, `U8`, etc.), and tensor layout in milliseconds — even for 20+ GB shards.

### Hugging Face / MLX Directories
- Reads `config.json` to classify format (`MLX` if `quantization` block is present, otherwise `HuggingFace Safetensors`), architecture, `model_type`, `torch_dtype`, `vocab_size`, and `max_position_embeddings`.

## 2. Ollama Registry & Manifest Internals

Ollama stores models under `$OLLAMA_MODELS` (default `~/.ollama/models`):
- **Manifests**: Supports both legacy `manifests/registry.ollama.ai/...` and newer `manifests-v2/ollama.com/...`, plus `db.sqlite`-backed custom models resolved via `ollama show --modelfile`.
- **Blobs**: Content-addressed `blobs/sha256-<digest>` files.
  - `application/vnd.ollama.image.model`: Single GGUF blob (can be exported directly to `.gguf`).
  - `application/vnd.ollama.image.tensor` / `runner: "mlx"`: Split MLX/Safetensors layers (exported as a self-contained portable bundle).
- **Custom Templates & Renderer Stripping**: When creating a custom `TEMPLATE` model without an explicit `RENDERER`, Ollama may auto-inject a default renderer/stop tokens for certain architectures (e.g. `gemma4`). `_strip_ollama_manifest_renderer()` rewrites the config/params blobs and updates content-addressed SHA-256 digests cleanly.

## 3. Apple Silicon MLX & Gemma 4 Pipeline

- Dynamically locates the Homebrew `mlx-lm` Python environment (`_find_mlx_python()`) rather than pinning a Cellar version.
- For `gemma4_unified` multimodal checkpoints, stages a sanitized view that strips `vision_embedder` / `audio_embedder` keys during `mlx_lm.convert` so text-only MLX quantization succeeds cleanly.
- Registers all temporary staging/dequantization directories with `atexit` and `SIGINT`/`SIGTERM` handlers so interrupted conversions never leave orphaned multi-GB temp folders.

## 4. Browser Web App (`web/`)

Designed to run 100% client-side on GitHub Pages with **zero build tooling** and **zero server dependency**:
- **`web/index.html` + `web/styles.css`**: Single-page responsive dark/light-aware studio UI.
- **`web/parsers.js`**: Pure-JS `DataView` GGUF v2/v3 parser (with full KV type decoding + tensor table), Safetensors header parser, and `config.json` / folder inspector.
- **`web/estimator.js`**: Analytical memory & bandwidth estimator computing model weight footprint, GQA KV-cache size (`2 * layers * kv_heads * head_dim * ctx * bytes_per_kv_elem`), overhead, and memory-bandwidth tok/s ceiling across Apple Silicon (M1–M5 Base/Pro/Max/Ultra) and discrete GPUs.
- **`web/app.js`**: Connects local file drop, Hugging Face Hub API explorer (`huggingface.co/api/models`), Modelfile & CLI command generator, and optional `localhost:11434` Ollama browser bridge.

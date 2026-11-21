# CLI Reference & Conversion Matrix

All commands are invoked via `./bin/model-convert` (symlinked to `cli/model_converter.py`).

## Conversion Matrix

| Source | Target (`--to`) | Mechanism | Notes |
| --- | --- | --- | --- |
| Local `.gguf` | `ollama` | Zero-copy `Modelfile` + `ollama create` | Auto-infers `RENDERER`, `PARSER`, `CAPABILITY`, `num_ctx` |
| Local `.gguf` | `gguf` | `llama-quantize` | Supports `Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ4_NL`, etc. |
| Local `.gguf` | `safetensors` | Pure-Python tensor unpacker | Supports `F32`, `F16`, `BF16`, `Q8_0` |
| HF Repo / Local Safetensors | `mlx` | `mlx_lm.convert` | `--bits 4`, `8`, or `16` (BF16); handles `gemma4_unified` sanitization |
| HF Repo / Local Safetensors | `ollama` / `ollama-gguf` | `convert_hf_to_gguf.py` + `llama-quantize` + `ollama create` | Optional `--quant` |
| HF Repo / Local Safetensors | `ollama-mlx` | `mlx_lm.convert` + Ollama MLX Modelfile | Strips orphan multimodal configs when needed |
| HF Repo / Local Safetensors | `onnx` | `optimum-cli` / Transformers.js export | `--onnx-quant q4f16 \| int4 \| int8 \| fp16` |
| HF Repo / Local Safetensors | `lmstudio` | Copies/converts into `~/.lmstudio/models/` | `--lms-format auto \| mlx \| gguf \| safetensors` |
| MLX Folder | `safetensors` | `mlx_lm.convert --dequantize` | `--dtype bfloat16 \| float16 \| float32` |
| MLX Folder | `gguf` / `ollama-gguf` | Dequantize → `convert_hf_to_gguf.py` → `llama-quantize` | Temporary intermediates cleaned up on exit/SIGTERM |
| `ollama:<name>` | `gguf` | Blob extraction (+ optional `llama-quantize`) | Works for both manifests-v1, manifests-v2, and `db.sqlite` models |
| `ollama:<name>` | `ollama` | Zero-copy Model Card update | Updates parameters/system/renderer without touching weights |

---

## Subcommands

### `inspect <target>`
Zero-copy metadata inspection (prints JSON).
- `<target>` can be:
  - Path to `.gguf` file or directory containing a single `.gguf`
  - Path to Hugging Face Safetensors or MLX directory (`config.json`)
  - LM Studio model ID relative to `~/.lmstudio/models/`
  - `ollama:<model[:tag]>`
  - `hf:<owner/repo>` or `hf:<owner/repo#filename.gguf>` or `https://huggingface.co/owner/repo`

### `list-lmstudio [--json]`
Scans `~/.lmstudio/models/<author>/<repo>/` and reports GGUF, MLX, and Safetensors models with format, architecture, and disk size.

### `list-ollama [--json]`
Parses `ollama list` output into a structured table or JSON array.

### `convert <source> --to <target> [options]`
Convert or register a model.
- `--to`: `ollama`, `ollama-gguf`, `ollama-mlx`, `mlx`, `onnx`, `gguf`, `safetensors`, `lmstudio`
- `--name <name>`: Target Ollama model name
- `--bits {4,8,16}`: MLX quantization bit-width (default: `4`)
- `--quant <type>`: GGUF quantization type (e.g. `Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ4_NL`) or Ollama Safetensors quant (`int4`, `int8`, `mxfp4`, `nvfp4`)
- `--dtype {bfloat16,float16,float32}`: Target dtype for Safetensors export
- `--onnx-quant {q4f16,int4,int8,fp16}`: Target precision for ONNX export
- `--lms-format {auto,mlx,gguf,safetensors}`: Sub-format when `--to lmstudio`
- `--output <path>`: Custom output file or directory
- `--system <prompt>`: System prompt for Ollama Modelfile
- `--ctx <int>`: Context length (`PARAMETER num_ctx`)
- `--temp <float>`: Sampling temperature (`PARAMETER temperature`)
- `--renderer <name>`: Ollama `RENDERER` (e.g. `qwen3.5`, `qwen3.8`, `gemma4`)
- `--parser <name>`: Ollama `PARSER` (e.g. `qwen3.5`)
- `--capabilities <csv>`: Comma-separated Ollama `CAPABILITY` list (e.g. `completion,tools,thinking,vision`)
- `--move`: Delete source after a verified conversion/import

### `get-card <source>` / `edit-card <source> [options]`
Read or update an Ollama Model Card (`Modelfile`) or local `generation_config.json` **without** copying or reconverting weights.
- Options for `edit-card`: `--name`, `--system`, `--ctx`, `--temp`, `--top-p`, `--top-k`, `--renderer`, `--parser`, `--capabilities`.

### `export-ollama <name> [--output <path>] [--to-lmstudio] [--move]`
Extracts the underlying GGUF blob (plus rewritten relative `Modelfile`) or a self-contained multi-layer bundle from `~/.ollama/models`.

### `archive <source> --dest <dir> [--move]`
Copies or moves an Ollama model (`ollama:<name>`), LM Studio model, or local directory into an external/archive directory preserving structure (`ollama/`, `lmstudio/`, `local/`).

### `download <repo> [--to-lmstudio] [--output <dir>]`
Downloads a Hugging Face repository via `huggingface_hub.snapshot_download` into `~/.lmstudio/models/<owner>/<repo>` or a custom folder.

### `delete-ollama <name>` / `delete-lmstudio <path>`
Deletes an Ollama model (`ollama rm`) or safely removes a model inside `~/.lmstudio/models/` (with root-containment guard and empty author folder cleanup).

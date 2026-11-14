#!/usr/bin/env python3
"""
M5 Model Bridge (scripts/model_converter.py)
Universal Apple Silicon (M-Series) Model Format Converter & Hub.

Unified interface bridging:
  Inputs:
    - GGUF (.gguf files)
    - HuggingFace Safetensors (local folders or HuggingFace IDs)
    - LM Studio models (~/.lmstudio/models/ or custom paths)
    - MLX models
  Targets:
    - Ollama (instant zero-copy registration via GGUF Modelfile)
    - MLX (Apple Silicon Metal native 4-bit, 8-bit, bf16 via mlx-lm)
    - ONNX (WebGPU/Transformers.js causal LM q4f16 format)
    - GGUF / Quantized GGUF (via llama-quantize: Q4_K_M, Q8_0, etc.)
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LM_STUDIO_MODELS_DIR = Path.home() / ".lmstudio" / "models"
GGUF_MAGIC = b"GGUF"


# =====================================================================
# GGUF Parser (Pure Python, zero-dependency)
# =====================================================================
def inspect_gguf(file_path: Path) -> Dict[str, Any]:
    """Reads GGUF v2/v3 header and extracts architecture & metadata."""
    info: Dict[str, Any] = {
        "format": "GGUF",
        "file_size_gb": round(file_path.stat().st_size / (1024**3), 2),
        "path": str(file_path.resolve()),
        "architecture": "unknown",
        "name": file_path.stem,
        "n_tensors": 0,
        "n_kv": 0,
        "metadata": {},
    }
    try:
        with open(file_path, "rb") as f:
            header = f.read(24)
            if len(header) < 24:
                return info
            magic, version, n_tensors, n_kv = struct.unpack("<4sIQI", header[:20])
            if magic != GGUF_MAGIC:
                # Try v2 header structure (<4sIQQ)
                magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", header[:24])
            if magic != GGUF_MAGIC:
                return info

            info["version"] = version
            info["n_tensors"] = n_tensors
            info["n_kv"] = n_kv

            # Extract basic KV pairs up to 64KB
            data = f.read(65536)
            pos = 0

            def read_str(p: int) -> Tuple[Optional[str], int]:
                if p + 8 > len(data):
                    return None, p
                (length,) = struct.unpack_from("<Q", data, p)
                p += 8
                if p + length > len(data):
                    return None, p
                s = data[p : p + length].decode("utf-8", errors="replace")
                return s, p + length

            for _ in range(min(n_kv, 100)):
                key, pos = read_str(pos)
                if key is None or pos + 4 > len(data):
                    break
                (vtype,) = struct.unpack_from("<I", data, pos)
                pos += 4
                if vtype == 8:  # STRING
                    val, pos = read_str(pos)
                    if val is not None:
                        info["metadata"][key] = val
                        if key.endswith(".architecture"):
                            info["architecture"] = val
                        elif key == "general.name":
                            info["name"] = val
                elif vtype in (4, 5, 6):  # INT32/UINT32/FLOAT32
                    if pos + 4 <= len(data):
                        (val,) = struct.unpack_from("<I", data, pos)
                        info["metadata"][key] = val
                        pos += 4
                elif vtype in (10, 11, 12):  # INT64/UINT64/FLOAT64
                    if pos + 8 <= len(data):
                        (val,) = struct.unpack_from("<Q", data, pos)
                        info["metadata"][key] = val
                        pos += 8
                else:
                    break
    except Exception as e:
        info["error"] = str(e)
    return info


# =====================================================================
# HuggingFace & MLX Directory Inspector
# =====================================================================
def inspect_hf_or_mlx(dir_path: Path) -> Dict[str, Any]:
    """Inspects a HuggingFace safetensors or MLX directory."""
    config_path = dir_path / "config.json"
    if not config_path.exists():
        return {"format": "unknown", "path": str(dir_path)}

    info: Dict[str, Any] = {
        "path": str(dir_path.resolve()),
        "name": dir_path.name,
    }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        info["architecture"] = cfg.get("architectures", [cfg.get("model_type", "unknown")])[0]
        info["model_type"] = cfg.get("model_type", "unknown")
        info["vocab_size"] = cfg.get("vocab_size", "unknown")
        info["torch_dtype"] = cfg.get("torch_dtype", "unknown")
        info["context_length"] = cfg.get("max_position_embeddings", cfg.get("context_length", "unknown"))

        # Check for MLX quantization config
        if "quantization" in cfg:
            info["format"] = "MLX"
            info["quantization"] = cfg["quantization"]
        else:
            info["format"] = "HuggingFace Safetensors"

        # Calculate total weight size
        st_files = list(dir_path.glob("*.safetensors")) + list(dir_path.glob("*.bin"))
        total_bytes = sum(f.stat().st_size for f in st_files)
        info["file_size_gb"] = round(total_bytes / (1024**3), 2)
        info["num_weight_shards"] = len(st_files)
    except Exception as e:
        info["format"] = "error"
        info["error"] = str(e)

    return info


def inspect_model(target: str) -> Dict[str, Any]:
    """Inspect any model path, file, or LM Studio identifier."""
    p = Path(target).expanduser()
    if p.is_file() and p.suffix.lower() == ".gguf":
        return inspect_gguf(p)
    elif p.is_dir():
        # Check if directory contains a single GGUF
        ggufs = list(p.glob("*.gguf"))
        if len(ggufs) == 1 and not (p / "config.json").exists():
            return inspect_gguf(ggufs[0])
        return inspect_hf_or_mlx(p)

    # Check LM Studio models path
    lm_p = LM_STUDIO_MODELS_DIR / target
    if lm_p.exists():
        if lm_p.is_file() and lm_p.suffix.lower() == ".gguf":
            return inspect_gguf(lm_p)
        return inspect_hf_or_mlx(lm_p)

    return {"error": f"Path '{target}' not found."}


# =====================================================================
# Scanners: LM Studio & Ollama
# =====================================================================
def list_lm_studio_models() -> List[Dict[str, Any]]:
    """Scans ~/.lmstudio/models and returns structured inventory."""
    if not LM_STUDIO_MODELS_DIR.exists():
        return []

    models = []
    # LM Studio structures directories as ~/.lmstudio/models/<author>/<repo>/...
    for author_dir in LM_STUDIO_MODELS_DIR.iterdir():
        if not author_dir.is_dir() or author_dir.name.startswith("."):
            continue
        for repo_dir in author_dir.iterdir():
            if not repo_dir.is_dir() or repo_dir.name.startswith("."):
                continue

            # Check if this repo contains GGUF files or safetensors
            ggufs = list(repo_dir.glob("*.gguf"))
            if ggufs:
                for g in ggufs:
                    info = inspect_gguf(g)
                    info["lmstudio_id"] = f"{author_dir.name}/{repo_dir.name}/{g.name}"
                    models.append(info)
            elif (repo_dir / "config.json").exists():
                info = inspect_hf_or_mlx(repo_dir)
                info["lmstudio_id"] = f"{author_dir.name}/{repo_dir.name}"
                models.append(info)

    return models


def list_ollama_models() -> List[Dict[str, Any]]:
    """Runs `ollama list` and parses outputs."""
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if res.returncode != 0:
        return []

    lines = res.stdout.strip().splitlines()
    if len(lines) <= 1:
        return []

    models = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            models.append({
                "name": parts[0],
                "id": parts[1],
                "size": " ".join(parts[2:4]),
                "modified": " ".join(parts[4:]) if len(parts) > 4 else "unknown",
            })
    return models


# =====================================================================
# Converters & Bridges
# =====================================================================
def infer_ollama_card_defaults(info: Dict[str, Any]) -> Dict[str, Any]:
    """Infers Ollama RENDERER, PARSER, and capabilities from inspected model metadata."""
    arch = (info.get("architecture") or "").lower()
    model_type = (info.get("model_type") or "").lower()
    name = (info.get("name") or "").lower()
    combined = f"{arch} {model_type} {name}"

    defaults: Dict[str, Any] = {
        "renderer": None,
        "parser": None,
        "capabilities": ["completion"],
        "num_ctx": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 30,
    }

    if "qwen3_5" in combined or "qwen3.5" in combined:
        defaults["renderer"] = "qwen3.5"
        defaults["parser"] = "qwen3.5"
        defaults["capabilities"] = ["completion", "tools", "thinking"]
    elif "qwen3.8" in combined or "qwen3_8" in combined:
        defaults["renderer"] = "qwen3.8"
        defaults["parser"] = "qwen3.5"
        defaults["capabilities"] = ["completion", "tools", "thinking"]
    elif "gemma4" in combined:
        defaults["renderer"] = "gemma4"
        defaults["capabilities"] = ["completion"]
    elif "lfm" in combined:
        defaults["capabilities"] = ["completion"]

    if "vision" in combined or "vl" in combined or "conditionalgeneration" in arch:
        if "vision" not in defaults["capabilities"] and ("qwen3" in combined or "gemma4" in combined):
            defaults["capabilities"].append("vision")

    return defaults


def build_modelfile(
    src_path: Path,
    info: Dict[str, Any],
    system_prompt: Optional[str] = None,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    renderer: Optional[str] = None,
    parser_name: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
) -> str:
    """Builds a complete Ollama Model Card (Modelfile)."""
    defaults = infer_ollama_card_defaults(info)
    lines = [f"FROM {src_path.resolve()}"]

    eff_renderer = renderer if renderer is not None else defaults["renderer"]
    eff_parser = parser_name if parser_name is not None else defaults["parser"]
    eff_caps = capabilities if capabilities is not None else defaults["capabilities"]
    eff_ctx = num_ctx if num_ctx is not None else defaults["num_ctx"]
    eff_temp = temperature if temperature is not None else defaults["temperature"]

    if eff_renderer:
        lines.append(f"RENDERER {eff_renderer}")
    if eff_parser:
        lines.append(f"PARSER {eff_parser}")

    for cap in eff_caps:
        cap_clean = cap.strip()
        if cap_clean:
            lines.append(f"CAPABILITY {cap_clean}")

    lines.append(f"PARAMETER num_ctx {eff_ctx}")
    lines.append(f"PARAMETER temperature {eff_temp}")
    lines.append(f"PARAMETER top_p {defaults['top_p']}")
    lines.append(f"PARAMETER top_k {defaults['top_k']}")
    lines.append("PARAMETER repeat_penalty 1")

    if system_prompt and system_prompt.strip():
        lines.append(f'SYSTEM """{system_prompt.strip()}"""')

    # Embed LICENSE if present in source directory
    if src_path.is_dir():
        for lic_name in ["LICENSE", "LICENSE.txt", "LICENSE.md"]:
            lic_path = src_path / lic_name
            if lic_path.is_file():
                try:
                    lic_text = lic_path.read_text(encoding="utf-8", errors="replace").strip()
                    if lic_text:
                        lines.append(f'LICENSE """\n{lic_text}\n"""')
                except Exception:
                    pass
                break

    return "\n".join(lines) + "\n"


def convert_to_ollama(
    source: str,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    renderer: Optional[str] = None,
    parser_name: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    move_source: bool = False,
) -> bool:
    """Imports GGUF, MLX, or HF Safetensors model into Ollama with a full Model Card."""
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    src_path = Path(info["path"])
    fmt = info.get("format", "unknown")

    if fmt not in ("GGUF", "MLX", "HuggingFace Safetensors"):
        print(f"[-] Unsupported source format '{fmt}' for Ollama import.", file=sys.stderr)
        return False

    if not model_name:
        clean_name = src_path.stem.lower().replace("_", "-").replace(".", "-")
        clean_name = clean_name.replace("-q4-k-m", ":q4").replace("-q8-0", ":q8").replace("-bf16", ":bf16")
        if fmt == "MLX" and "-mlx" not in clean_name:
            clean_name += "-mlx"
        model_name = clean_name

    print(f"[*] Preparing Ollama model '{model_name}' from {fmt}: {src_path.name} ({info.get('file_size_gb', 0)} GB)")

    modelfile_content = build_modelfile(
        src_path=src_path,
        info=info,
        system_prompt=system_prompt,
        num_ctx=num_ctx,
        temperature=temperature,
        renderer=renderer,
        parser_name=parser_name,
        capabilities=capabilities,
    )

    print("[*] Generated Ollama Model Card (Modelfile):")
    print("-" * 50)
    for line in modelfile_content.splitlines()[:25]:
        print(f"    {line}")
    print("-" * 50)

    modelfile_path = Path(f"/tmp/Modelfile_{model_name.replace(':', '_').replace('/', '_')}")
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    try:
        print(f"[*] Executing: ollama create {model_name} -f {modelfile_path}")
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        subprocess.run(cmd, check=True)
        print(f"[+] Successfully registered '{model_name}' in Ollama!")

        if move_source:
            print(f"[*] --move enabled: removing original source at '{src_path}' (blobs are now stored in Ollama)...")
            if src_path.is_dir():
                shutil.rmtree(src_path)
            elif src_path.is_file():
                src_path.unlink()
            print("[+] Original source moved/cleaned up.")

        print(f"    Run via: ollama run {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Failed to create Ollama model: {e}", file=sys.stderr)
        return False
    finally:
        if modelfile_path.exists():
            modelfile_path.unlink()


def convert_to_mlx(source: str, output_dir: Optional[str] = None, q_bits: int = 4, q_group_size: int = 64) -> bool:
    """Converts HuggingFace Safetensors model to Apple Silicon Metal MLX format."""
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    if info.get("format") == "GGUF":
        print("[-] GGUF -> MLX direct conversion is lossy and unsupported for quantized GGUF.", file=sys.stderr)
        print("    Apple Silicon best practice: convert from the source HuggingFace repository or unquantized safetensors.", file=sys.stderr)
        return False

    src_path = Path(info["path"])
    if not output_dir:
        output_dir = str(src_path.parent / f"{src_path.name}-mlx-{q_bits}bit")

    out_p = Path(output_dir)
    print(f"[*] Converting HuggingFace model -> MLX ({q_bits}-bit, group_size={q_group_size})")
    print(f"    Source: {src_path}")
    print(f"    Destination: {out_p}")

    # Use homebrew mlx_lm.convert binary if available
    mlx_convert_bin = shutil.which("mlx_lm.convert") or "/opt/homebrew/bin/mlx_lm.convert"
    if not Path(mlx_convert_bin).exists():
        print(f"[-] mlx_lm.convert executable not found at '{mlx_convert_bin}'. Install via: brew install mlx-lm", file=sys.stderr)
        return False

    cmd = [
        mlx_convert_bin,
        "--hf-path", str(src_path),
        "--mlx-path", str(out_p),
        "-q",
        "--q-bits", str(q_bits),
        "--q-group-size", str(q_group_size),
    ]

    try:
        print(f"[*] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[+] Successfully converted to MLX at '{out_p}'!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] MLX conversion failed: {e}", file=sys.stderr)
        return False


def quantize_gguf(source: str, target_quant: str = "Q4_K_M", output_file: Optional[str] = None) -> bool:
    """Quantizes a GGUF file using Apple Silicon accelerated llama-quantize."""
    info = inspect_model(source)
    if info.get("format") != "GGUF":
        print(f"[-] Source must be a GGUF file (got {info.get('format')}).", file=sys.stderr)
        return False

    src_path = Path(info["path"])
    if not output_file:
        output_file = str(src_path.parent / f"{src_path.stem}_{target_quant.upper()}.gguf")

    out_p = Path(output_file)
    llama_quantize_bin = shutil.which("llama-quantize") or "/opt/homebrew/bin/llama-quantize"
    if not Path(llama_quantize_bin).exists():
        print("[-] llama-quantize not found. Install via: brew install llama.cpp", file=sys.stderr)
        return False

    print(f"[*] Quantizing GGUF: {src_path.name} -> {target_quant.upper()}")
    cmd = [llama_quantize_bin, str(src_path), str(out_p), target_quant.upper()]

    try:
        print(f"[*] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[+] Quantized GGUF saved to '{out_p}' ({round(out_p.stat().st_size / (1024**3), 2)} GB)!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] llama-quantize failed: {e}", file=sys.stderr)
        return False


def convert_to_onnx(source: str, output_dir: Optional[str] = None) -> bool:
    """Exports HuggingFace model to ONNX causal LM (q4f16 format) for WebGPU/Transformers.js."""
    info = inspect_model(source)
    if info.get("format") != "HuggingFace Safetensors":
        print(f"[-] ONNX WebGPU export requires HuggingFace safetensors source (got {info.get('format')}).", file=sys.stderr)
        return False

    src_path = Path(info["path"])
    model_type = info.get("model_type", "").lower()

    if not output_dir:
        output_dir = str(src_path.parent / f"{src_path.name}-ONNX-q4f16")
    out_p = Path(output_dir)

    print(f"[*] Exporting HuggingFace ({model_type}) -> WebGPU ONNX q4f16 format at '{out_p}'")

    # Check if we have dedicated export scripts in threecoder/scripts
    script_dir = Path(__file__).resolve().parent
    if "lfm" in model_type or "liquid" in str(src_path).lower():
        export_script = script_dir / "export_lfm3_onnx_q4f16.py"
    elif "qwen" in model_type or "qwen" in str(src_path).lower():
        export_script = script_dir / "export_qwen35_onnx_q4f16.py"
    else:
        export_script = None

    if export_script and export_script.exists():
        cmd = [sys.executable, str(export_script), "--model-dir", str(src_path), "--output-dir", str(out_p)]
        print(f"[*] Running architecture-specific WebGPU exporter: {export_script.name}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[+] Successfully exported WebGPU ONNX model to '{out_p}'!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[-] ONNX export failed: {e}", file=sys.stderr)
            return False
    else:
        print(f"[-] Universal ONNX exporter: use optimum-cli or vendor exporter for architecture '{model_type}'.", file=sys.stderr)
        return False


# =====================================================================
# CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="M5 Model Bridge: Universal Apple Silicon Model Format Converter & Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect any model or GGUF
  ./scripts/model_converter.py inspect /path/to/model.gguf
  ./scripts/model_converter.py inspect ~/.lmstudio/models/npario/humanizer-12b-mlx-8bit

  # List models downloaded in LM Studio
  ./scripts/model_converter.py list-lmstudio

  # List registered Ollama models
  ./scripts/model_converter.py list-ollama

  # Convert / Register LM Studio GGUF directly into Ollama
  ./scripts/model_converter.py convert /path/to/model.gguf --to ollama --name my-model

  # Convert HuggingFace model to MLX (4-bit Metal)
  ./scripts/model_converter.py convert /path/to/hf-model --to mlx --bits 4

  # Quantize GGUF (e.g. FP16/Q8 to Q4_K_M)
  ./scripts/model_converter.py convert /path/to/model.gguf --to gguf --quant Q4_K_M

  # Convert HuggingFace model to WebGPU ONNX
  ./scripts/model_converter.py convert /path/to/hf-model --to onnx
"""
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a model file or directory")
    inspect_parser.add_argument("target", help="Path to .gguf, HF folder, or LM Studio identifier")

    # List LM Studio
    lm_parser = subparsers.add_parser("list-lmstudio", help="List models stored in LM Studio (~/.lmstudio/models)")
    lm_parser.add_argument("--json", action="store_true", help="Output JSON")

    # List Ollama
    ol_parser = subparsers.add_parser("list-ollama", help="List models registered in Ollama")
    ol_parser.add_argument("--json", action="store_true", help="Output JSON")

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Convert model between formats")
    convert_parser.add_argument("source", help="Source model file, HF directory, or LM Studio identifier")
    convert_parser.add_argument("--to", required=True, choices=["ollama", "mlx", "onnx", "gguf"], help="Target format")
    convert_parser.add_argument("--name", help="Model name for Ollama")
    convert_parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16], help="Quantization bits for MLX (default: 4)")
    convert_parser.add_argument("--quant", default="Q4_K_M", help="Target quantization for GGUF (e.g. Q4_K_M, Q8_0)")
    convert_parser.add_argument("--output", help="Output path/directory")
    convert_parser.add_argument("--system", help="System prompt to embed (for Ollama)")
    convert_parser.add_argument("--ctx", type=int, help="Context length num_ctx (for Ollama)")
    convert_parser.add_argument("--temp", type=float, help="Temperature parameter (for Ollama)")
    convert_parser.add_argument("--renderer", help="Ollama RENDERER (e.g. qwen3.5, qwen3.8, gemma4)")
    convert_parser.add_argument("--parser", dest="parser_name", help="Ollama PARSER (e.g. qwen3.5)")
    convert_parser.add_argument("--capabilities", help="Comma-separated Ollama CAPABILITY values (e.g. completion,tools,thinking)")
    convert_parser.add_argument("--move", action="store_true", help="Remove original source after successful Ollama import to save disk space")

    args = parser.parse_args()

    if args.command == "inspect":
        info = inspect_model(args.target)
        print(json.dumps(info, indent=2))

    elif args.command == "list-lmstudio":
        models = list_lm_studio_models()
        if args.json:
            print(json.dumps(models, indent=2))
            return
        if not models:
            print("No models found in ~/.lmstudio/models.")
            return
        print(f"\n{'LM STUDIO ID / MODEL':<55} {'FORMAT':<12} {'ARCH':<12} {'SIZE'}")
        print("-" * 90)
        for m in models:
            ident = m.get("lmstudio_id", m.get("name", "unknown"))
            fmt = m.get("format", "unknown")
            arch = m.get("architecture", "unknown")
            sz = f"{m.get('file_size_gb', 0)} GB"
            print(f"{ident:<55} {fmt:<12} {arch:<12} {sz}")
        print()

    elif args.command == "list-ollama":
        models = list_ollama_models()
        if args.json:
            print(json.dumps(models, indent=2))
            return
        if not models:
            print("No models registered in Ollama.")
            return
        print(f"\n{'OLLAMA MODEL NAME':<35} {'ID':<16} {'SIZE':<12} {'MODIFIED'}")
        print("-" * 80)
        for m in models:
            print(f"{m['name']:<35} {m['id']:<16} {m['size']:<12} {m['modified']}")
        print()

    elif args.command == "convert":
        ok = False
        if args.to == "ollama":
            caps = [c.strip() for c in args.capabilities.split(",")] if args.capabilities else None
            ok = convert_to_ollama(
                args.source,
                model_name=args.name,
                system_prompt=args.system,
                num_ctx=args.ctx,
                temperature=args.temp,
                renderer=args.renderer,
                parser_name=args.parser_name,
                capabilities=caps,
                move_source=args.move,
            )
        elif args.to == "mlx":
            ok = convert_to_mlx(args.source, output_dir=args.output, q_bits=args.bits)
        elif args.to == "gguf":
            ok = quantize_gguf(args.source, target_quant=args.quant, output_file=args.output)
        elif args.to == "onnx":
            ok = convert_to_onnx(args.source, output_dir=args.output)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

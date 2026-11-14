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
import atexit
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TEMP_CLEANUP_PATHS: List[Path] = []


def _register_temp_path(p: Path) -> Path:
    if p not in _TEMP_CLEANUP_PATHS:
        _TEMP_CLEANUP_PATHS.append(p)
    return p


def _cleanup_temp_paths() -> None:
    for p in list(_TEMP_CLEANUP_PATHS):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
        except Exception:
            pass
    _TEMP_CLEANUP_PATHS.clear()


def _handle_sigterm(signum, frame):
    _cleanup_temp_paths()
    raise KeyboardInterrupt(f"Terminated by signal {signum}")


atexit.register(_cleanup_temp_paths)
signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

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


def is_hf_repo_id(target: str) -> bool:
    """Returns True if target looks like a Hugging Face repo ID ('owner/repo' or 'hf:owner/repo')."""
    clean = target.replace("hf:", "", 1).strip()
    if clean.startswith("/") or clean.startswith("~") or clean.startswith("."):
        return False
    parts = clean.split("/")
    return len(parts) == 2 and all(bool(p) and not p.endswith(".gguf") for p in parts)


def inspect_hf_hub(repo_id: str) -> Dict[str, Any]:
    """Fetches config.json from Hugging Face Hub without downloading weights."""
    import urllib.request
    clean_id = repo_id.replace("hf:", "", 1).strip()
    url = f"https://huggingface.co/{clean_id}/resolve/main/config.json"
    info: Dict[str, Any] = {
        "format": "HuggingFace Hub",
        "path": f"hf:{clean_id}",
        "repo_id": clean_id,
        "name": clean_id.split("/")[-1],
    }
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "model-bridge/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            cfg = json.loads(resp.read().decode("utf-8"))
        info["architecture"] = cfg.get("architectures", [cfg.get("model_type", "unknown")])[0]
        info["model_type"] = cfg.get("model_type", "unknown")
        info["torch_dtype"] = cfg.get("torch_dtype", "unknown")
        info["vocab_size"] = cfg.get("vocab_size", "unknown")
        if "quantization" in cfg or "quantization_config" in cfg:
            qcfg = cfg.get("quantization") or cfg.get("quantization_config")
            info["quantization"] = qcfg if isinstance(qcfg, dict) else {}
    except Exception as e:
        info["error"] = f"Could not fetch HuggingFace config for '{clean_id}': {e}"
    return info


def resolve_hf_hub_snapshot(repo_id: str) -> Optional[Path]:
    """Downloads (or resolves cached) Hugging Face repo snapshot via huggingface_hub."""
    clean_id = repo_id.replace("hf:", "", 1).strip()
    mlx_py = "/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python"
    py_exec = mlx_py if Path(mlx_py).exists() else sys.executable
    print(f"[*] Resolving Hugging Face Hub repo '{clean_id}' (downloading/caching snapshot)...")
    code = (
        "import sys; "
        "from huggingface_hub import snapshot_download; "
        "print(snapshot_download(repo_id=sys.argv[1]))"
    )
    res = subprocess.run([py_exec, "-c", code, clean_id], capture_output=True, text=True)
    if res.returncode == 0:
        out_path = res.stdout.strip().splitlines()[-1].strip()
        p = Path(out_path)
        if p.is_dir():
            print(f"[+] Snapshot ready at: {p}")
            return p
    print(f"[-] Failed to download HF repo '{clean_id}': {res.stderr}", file=sys.stderr)
    return None


def inspect_model(target: str) -> Dict[str, Any]:
    """Inspect any model path, file, Ollama model ('ollama:<name>'), HF repo ('owner/repo'), or LM Studio ID."""
    target = target.strip()
    if target.startswith("https://huggingface.co/"):
        parts = target.replace("https://huggingface.co/", "", 1).strip("/").split("/")
        if len(parts) >= 2:
            target = f"hf:{parts[0]}/{parts[1]}"

    if target.startswith("ollama:"):
        return inspect_ollama_model(target.replace("ollama:", "", 1))

    if target.startswith("hf:"):
        return inspect_hf_hub(target)

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

    # Check HuggingFace Hub ID (owner/repo)
    if is_hf_repo_id(target):
        return inspect_hf_hub(target)

    return {"error": f"Path or repo '{target}' not found."}


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


OLLAMA_MODELS_DIR = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def find_ollama_manifest(model_name: str) -> Optional[Path]:
    """Resolves an Ollama model name (e.g. 'lfm3:1.7b' or 'LisyNeko/qwen3.8-9b-coder:latest') to its manifest file."""
    clean = model_name.strip()
    if ":" in clean:
        repo_part, tag = clean.rsplit(":", 1)
    else:
        repo_part, tag = clean, "latest"

    if "/" in repo_part:
        rel = Path("registry.ollama.ai") / repo_part / tag
    else:
        rel = Path("registry.ollama.ai") / "library" / repo_part / tag

    manifest_path = OLLAMA_MODELS_DIR / "manifests" / rel
    if manifest_path.is_file():
        return manifest_path

    # Fallback: search manifests directory for matching repo/tag
    manifests_root = OLLAMA_MODELS_DIR / "manifests"
    if manifests_root.exists():
        for p in manifests_root.rglob(tag):
            if p.is_file() and p.parent.name == repo_part.split("/")[-1]:
                return p
    return None


def get_ollama_modelfile_blob(model_name: str) -> Tuple[Optional[Path], str]:
    """Runs `ollama show <name> --modelfile` and extracts the FROM blob path if present."""
    res = subprocess.run(["ollama", "show", model_name, "--modelfile"], capture_output=True, text=True)
    if res.returncode != 0:
        return None, ""
    mf_text = res.stdout
    for line in mf_text.splitlines():
        if line.startswith("FROM "):
            target = line[5:].strip()
            p = Path(target)
            if p.is_file():
                return p, mf_text
    return None, mf_text


def get_ollama_show_metadata(model_name: str) -> Dict[str, Any]:
    """Parses architecture, parameters, quantization from `ollama show <name>`."""
    res = subprocess.run(["ollama", "show", model_name], capture_output=True, text=True)
    meta: Dict[str, Any] = {}
    if res.returncode != 0:
        return meta
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            k, v = parts[0].lower(), parts[1].strip()
            if k in ("architecture", "parameters", "quantization", "renderer", "parser"):
                meta[k] = v
    return meta


def inspect_ollama_model(model_name: str) -> Dict[str, Any]:
    """Inspects an Ollama model via its manifest file or `ollama show` (for db.sqlite custom models)."""
    manifest_path = find_ollama_manifest(model_name)
    if manifest_path:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            layers = manifest.get("layers", [])
            runner = manifest.get("runner", "")
            fmt = manifest.get("format", "")
            model_blobs = [l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.model"]
            tensor_blobs = [l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.tensor"]

            total_bytes = sum(l.get("size", 0) for l in layers)
            backend = "GGUF" if model_blobs else ("MLX/Safetensors" if (runner == "mlx" or fmt == "safetensors" or tensor_blobs) else "Unknown")

            return {
                "name": model_name,
                "format": f"Ollama ({backend})",
                "backend": backend,
                "manifest_path": str(manifest_path),
                "file_size_gb": round(total_bytes / (1024**3), 2),
                "num_layers": len(layers),
                "has_single_gguf": len(model_blobs) == 1,
            }
        except Exception as e:
            return {"error": f"Failed to parse manifest: {e}"}

    # Fallback for Ollama db.sqlite custom models (`ollama create`)
    blob_path, mf_text = get_ollama_modelfile_blob(model_name)
    meta = get_ollama_show_metadata(model_name)
    if blob_path and blob_path.exists():
        is_gguf = False
        try:
            with open(blob_path, "rb") as f:
                is_gguf = f.read(4) == GGUF_MAGIC
        except Exception:
            pass
        backend = "GGUF" if is_gguf else "Blob"
        return {
            "name": model_name,
            "format": f"Ollama ({backend})",
            "backend": backend,
            "architecture": meta.get("architecture"),
            "path": str(blob_path),
            "file_size_gb": round(blob_path.stat().st_size / (1024**3), 2),
            "has_single_gguf": is_gguf,
        }
    elif mf_text:
        return {
            "name": model_name,
            "format": "Ollama (Custom/MLX)",
            "backend": "Custom/MLX",
            "architecture": meta.get("architecture"),
            "has_single_gguf": False,
        }

    return {"error": f"Manifest or model data not found for Ollama model '{model_name}'."}


def export_ollama_model(
    model_name: str,
    output_path: Optional[str] = None,
    to_lmstudio: bool = False,
    move: bool = False,
) -> bool:
    """Exports an Ollama model to GGUF (or self-contained bundle) and optionally imports to LM Studio."""
    manifest_path = find_ollama_manifest(model_name)
    safe_name = model_name.replace(":", "-").replace("/", "_")

    if to_lmstudio:
        dest_dir = LM_STUDIO_MODELS_DIR / "ollama-export" / safe_name
    elif output_path:
        p = Path(output_path).expanduser()
        dest_dir = p.parent if p.suffix.lower() == ".gguf" else p
    else:
        dest_dir = Path.home() / "ModelExports" / safe_name

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Save Modelfile
    blob_path, mf_text = get_ollama_modelfile_blob(model_name)
    if mf_text:
        (dest_dir / "Modelfile").write_text(mf_text, encoding="utf-8")

    # Resolve single GGUF blob either from manifest or from `ollama show --modelfile`
    single_gguf_blob: Optional[Path] = None
    manifest = None
    layers = []

    if manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layers = manifest.get("layers", [])
        blobs_dir = OLLAMA_MODELS_DIR / "blobs"
        model_layers = [l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.model"]
        if len(model_layers) == 1:
            candidate = blobs_dir / model_layers[0]["digest"].replace(":", "-")
            if candidate.exists():
                single_gguf_blob = candidate
    elif blob_path and blob_path.exists():
        single_gguf_blob = blob_path

    if single_gguf_blob:
        if output_path and Path(output_path).suffix.lower() == ".gguf" and not to_lmstudio:
            gguf_dest = Path(output_path).expanduser()
        else:
            gguf_dest = dest_dir / f"{safe_name}.gguf"

        print(f"[*] Extracting GGUF blob ({round(single_gguf_blob.stat().st_size / (1024**3), 2)} GB) -> {gguf_dest}")
        shutil.copy2(single_gguf_blob, gguf_dest)
        if (dest_dir / "Modelfile").exists():
            mf_lines = (dest_dir / "Modelfile").read_text(encoding="utf-8").splitlines()
            new_mf = [f"FROM ./{gguf_dest.name}" if l.startswith("FROM ") else l for l in mf_lines]
            (dest_dir / "Modelfile").write_text("\n".join(new_mf) + "\n", encoding="utf-8")
        print(f"[+] Exported GGUF to: {gguf_dest}")
    elif manifest and layers:
        blobs_dir = OLLAMA_MODELS_DIR / "blobs"
        if to_lmstudio:
            print("[-] Note: This Ollama model uses split safetensors/MLX layers; exporting self-contained bundle instead of single GGUF.")
        print(f"[*] Exporting Ollama bundle ({len(layers)} layers) -> {dest_dir}")
        (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out_blobs = dest_dir / "blobs"
        out_blobs.mkdir(exist_ok=True)
        for l in layers:
            digest = l.get("digest", "").replace(":", "-")
            if not digest:
                continue
            src_blob = blobs_dir / digest
            if src_blob.exists():
                shutil.copy2(src_blob, out_blobs / digest)
                if l.get("mediaType") == "application/vnd.ollama.image.json" and l.get("name"):
                    shutil.copy2(src_blob, dest_dir / l["name"])
        print(f"[+] Exported Ollama bundle to: {dest_dir}")
    else:
        print(f"[-] Could not resolve exportable weights for '{model_name}'.", file=sys.stderr)
        return False

    if move:
        print(f"[*] --move enabled: removing '{model_name}' from Ollama registry...")
        subprocess.run(["ollama", "rm", model_name], check=True)
        print(f"[+] Removed '{model_name}' from Ollama.")

    return True


def archive_model(source: str, archive_dir: str, move: bool = False) -> bool:
    """Archives an Ollama model or LM Studio / local model directory to a backup/external drive folder."""
    dest_root = Path(archive_dir).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    # Check if source is prefixed with 'ollama:' or matches an Ollama manifest
    if source.startswith("ollama:") or (not Path(source).expanduser().exists() and find_ollama_manifest(source) is not None):
        ol_name = source.replace("ollama:", "", 1)
        safe_name = ol_name.replace(":", "-").replace("/", "_")
        target_dir = dest_root / "ollama" / safe_name
        print(f"[*] Archiving Ollama model '{ol_name}' -> {target_dir}")
        return export_ollama_model(ol_name, output_path=str(target_dir), to_lmstudio=False, move=move)

    # Otherwise treat as local path / LM Studio model
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    src_path = Path(info["path"]).resolve()
    try:
        rel = src_path.relative_to(LM_STUDIO_MODELS_DIR.resolve())
        target_path = dest_root / "lmstudio" / rel
    except ValueError:
        target_path = dest_root / "local" / src_path.name

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] {'Moving' if move else 'Copying'} ({info.get('file_size_gb', 0)} GB) -> {target_path}")

    if src_path.is_dir():
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(src_path, target_path)
        if move:
            shutil.rmtree(src_path)
    else:
        shutil.copy2(src_path, target_path)
        if move:
            src_path.unlink()

    print(f"[+] Archived successfully to: {target_path}")
    return True


def delete_ollama_model(model_name: str) -> bool:
    """Deletes a model from the Ollama registry."""
    print(f"[*] Removing Ollama model: {model_name}")
    res = subprocess.run(["ollama", "rm", model_name])
    if res.returncode == 0:
        print(f"[+] Deleted '{model_name}' from Ollama.")
        return True
    print(f"[-] Failed to delete '{model_name}' from Ollama.", file=sys.stderr)
    return False


def delete_lmstudio_model(path_str: str) -> bool:
    """Safely deletes a model inside ~/.lmstudio/models/."""
    target = Path(path_str).expanduser().resolve()
    lm_root = LM_STUDIO_MODELS_DIR.resolve()
    try:
        target.relative_to(lm_root)
    except ValueError:
        print(f"[-] Safety check: '{target}' is not inside '{lm_root}'. Refusing to delete.", file=sys.stderr)
        return False

    if not target.exists():
        print(f"[-] Path not found: {target}", file=sys.stderr)
        return False

    print(f"[*] Deleting LM Studio model: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()

    # Clean up empty parent author folder if applicable
    if target.parent != lm_root and target.parent.is_dir() and not any(target.parent.iterdir()):
        target.parent.rmdir()

    print(f"[+] Deleted '{target.name}' from LM Studio.")
    return True


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
    quant: Optional[str] = None,
) -> bool:
    """Imports GGUF, MLX, or HF Safetensors model into Ollama with a full Model Card."""
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    fmt = info.get("format", "unknown")
    if fmt == "HuggingFace Hub":
        snap = resolve_hf_hub_snapshot(info["repo_id"])
        if not snap:
            return False
        info = inspect_model(str(snap))
        fmt = info.get("format", "unknown")

    src_path = Path(info["path"])

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

    staged_ollama_dir: Optional[Path] = None
    modelfile_src_path = src_path
    if fmt == "MLX" and src_path.is_dir():
        staged_ollama_dir = _stage_ollama_mlx_dir(src_path)
        if staged_ollama_dir:
            modelfile_src_path = staged_ollama_dir

    modelfile_content = build_modelfile(
        src_path=modelfile_src_path,
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

    modelfile_path = _register_temp_path(Path(f"/tmp/Modelfile_{model_name.replace(':', '_').replace('/', '_')}"))
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    try:
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        if quant and quant.lower() not in ("original", "none", "as-is", "") and fmt == "HuggingFace Safetensors":
            cmd.extend(["-q", quant.upper()])
        print(f"[*] Executing: {' '.join(cmd)}")
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
        if staged_ollama_dir and staged_ollama_dir.exists():
            shutil.rmtree(staged_ollama_dir, ignore_errors=True)


MLX_MODEL_TYPE_ALIASES = {
    "gemma4_unified": "gemma4",
    "gemma4_unified_text": "gemma4_text",
}


def _stage_mlx_compatible_dir(src_dir: Path) -> Optional[Path]:
    """Creates a symlinked temp dir with patched config.json if model_type needs MLX aliasing."""
    cfg_path = src_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    mtype = cfg.get("model_type")
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    text_mtype = text_cfg.get("model_type") if text_cfg else None

    needs_patch = (mtype in MLX_MODEL_TYPE_ALIASES) or (text_mtype in MLX_MODEL_TYPE_ALIASES)
    if not needs_patch:
        return None

    staged_dir = _register_temp_path(Path(f"/tmp/mlx_staged_{src_dir.name}"))
    if staged_dir.exists():
        shutil.rmtree(staged_dir, ignore_errors=True)
    staged_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        if item.name == "config.json":
            continue
        (staged_dir / item.name).symlink_to(item)

    if mtype in MLX_MODEL_TYPE_ALIASES:
        cfg["model_type"] = MLX_MODEL_TYPE_ALIASES[mtype]
    if text_cfg and text_mtype in MLX_MODEL_TYPE_ALIASES:
        text_cfg["model_type"] = MLX_MODEL_TYPE_ALIASES[text_mtype]
    if "vocab_size" not in cfg and text_cfg and "vocab_size" in text_cfg:
        cfg["vocab_size"] = text_cfg["vocab_size"]

    # Strip multimodal config blocks when converting via text-only mlx_lm.convert
    for mm_key in (
        "audio_config", "vision_config",
        "audio_token_id", "image_token_id", "video_token_id",
        "boa_token_id", "boi_token_id", "eoa_token_index", "eoi_token_id",
    ):
        cfg.pop(mm_key, None)
    if cfg.get("architectures") == ["Gemma4UnifiedForConditionalGeneration"]:
        cfg["architectures"] = ["Gemma4ForCausalLM"]

    (staged_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[*] Patched config.json model_type '{mtype}' -> '{cfg.get('model_type')}' (text-only) for mlx-lm compatibility.")
    return staged_dir


def _stage_ollama_mlx_dir(src_dir: Path) -> Optional[Path]:
    """If an MLX folder's config.json declares multimodal embedders missing from its weights, stage a cleaned config.json."""
    cfg_path = src_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    has_mm_cfg = ("audio_config" in cfg) or ("vision_config" in cfg)
    mtype = cfg.get("model_type")
    needs_alias = mtype in MLX_MODEL_TYPE_ALIASES
    if not has_mm_cfg and not needs_alias:
        return None

    # Check if multimodal weights actually exist in index
    idx_path = src_dir / "model.safetensors.index.json"
    has_mm_weights = False
    if idx_path.exists():
        try:
            wmap = json.loads(idx_path.read_text(encoding="utf-8")).get("weight_map", {})
            has_mm_weights = any(("embed_audio" in k or "embed_vision" in k or "audio_tower" in k or "vision_tower" in k) for k in wmap.keys())
        except Exception:
            pass

    if has_mm_weights and not needs_alias:
        return None

    staged_dir = _register_temp_path(Path(f"/tmp/ollama_mlx_staged_{src_dir.name}"))
    if staged_dir.exists():
        shutil.rmtree(staged_dir, ignore_errors=True)
    staged_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        if item.name == "config.json":
            continue
        (staged_dir / item.name).symlink_to(item)

    if needs_alias:
        cfg["model_type"] = MLX_MODEL_TYPE_ALIASES[mtype]
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    if text_cfg and text_cfg.get("model_type") in MLX_MODEL_TYPE_ALIASES:
        text_cfg["model_type"] = MLX_MODEL_TYPE_ALIASES[text_cfg["model_type"]]
    if not has_mm_weights:
        for mm_key in (
            "audio_config", "vision_config",
            "audio_token_id", "image_token_id", "video_token_id",
            "boa_token_id", "boi_token_id", "eoa_token_index", "eoi_token_id",
        ):
            cfg.pop(mm_key, None)
        if cfg.get("architectures") == ["Gemma4UnifiedForConditionalGeneration"]:
            cfg["architectures"] = ["Gemma4ForCausalLM"]

    (staged_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("[*] Cleaned orphan multimodal config keys from MLX config.json for Ollama MLX runner.")
    return staged_dir


def convert_to_mlx(source: str, output_dir: Optional[str] = None, q_bits: int = 4, q_group_size: int = 64) -> bool:
    """Converts HuggingFace Hub, Safetensors, or GGUF/Ollama (via Safetensors) to Apple Silicon Metal MLX format."""
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    fmt = info.get("format", "")
    temp_hf_dir: Optional[Path] = None

    if fmt in ("GGUF", "Ollama (GGUF)"):
        base_name = (info.get("name") or Path(info.get("path", "model")).stem).replace(":", "-").replace("/", "_")
        temp_hf_dir = _register_temp_path(Path(f"/tmp/hf_from_gguf_{base_name}"))
        print(f"[*] Step 1/2: Extracting {fmt} -> temporary Safetensors at '{temp_hf_dir}'...")
        if not convert_to_safetensors(source, output_dir=str(temp_hf_dir), dtype="bfloat16"):
            return False
        info = inspect_model(str(temp_hf_dir))
        fmt = info.get("format", "")

    if fmt == "HuggingFace Hub":
        hf_arg = info["repo_id"]
        default_out = str(Path.home() / "ModelExports" / f"{info['name']}-mlx-{q_bits}bit")
    else:
        src_path = Path(info["path"])
        hf_arg = str(src_path)
        clean_stem = src_path.stem.replace(":", "-")
        default_out = str(Path.home() / "ModelExports" / f"{clean_stem}-mlx-{q_bits}bit") if temp_hf_dir else str(src_path.parent / f"{src_path.name}-mlx-{q_bits}bit")

    if not output_dir:
        output_dir = default_out

    staged_dir: Optional[Path] = None
    if Path(hf_arg).is_dir():
        staged_dir = _stage_mlx_compatible_dir(Path(hf_arg))
        if staged_dir:
            hf_arg = str(staged_dir)

    out_p = Path(output_dir).expanduser()
    step_prefix = "[*] Step 2/2:" if temp_hf_dir else "[*]"
    print(f"{step_prefix} Converting model -> MLX ({q_bits}-bit, group_size={q_group_size})")
    print(f"    Source: {hf_arg}")
    print(f"    Destination: {out_p}")

    # Use homebrew mlx_lm.convert binary if available
    mlx_convert_bin = shutil.which("mlx_lm.convert") or "/opt/homebrew/bin/mlx_lm.convert"
    if not Path(mlx_convert_bin).exists():
        print(f"[-] mlx_lm.convert executable not found at '{mlx_convert_bin}'. Install via: brew install mlx-lm", file=sys.stderr)
        if temp_hf_dir and temp_hf_dir.exists():
            shutil.rmtree(temp_hf_dir, ignore_errors=True)
        return False

    cmd = [
        mlx_convert_bin,
        "--hf-path", hf_arg,
        "--mlx-path", str(out_p),
    ]
    if q_bits in (4, 8):
        cmd.extend(["-q", "--q-bits", str(q_bits), "--q-group-size", str(q_group_size)])
    else:
        cmd.extend(["--dtype", "bfloat16"])

    if staged_dir is not None:
        # Run mlx_lm.convert with patched gemma4.Model.sanitize for unified multimodal keys
        mlx_py = "/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python"
        if not Path(mlx_py).exists():
            # Fallback: read shebang from mlx_lm.convert
            try:
                first_line = Path(mlx_convert_bin).read_text(encoding="utf-8", errors="ignore").splitlines()[0]
                if first_line.startswith("#!"):
                    mlx_py = first_line[2:].strip().split()[0]
            except Exception:
                mlx_py = sys.executable
        wrapper_code = (
            "import sys\n"
            "import mlx_lm.models.gemma4 as g4\n"
            "from mlx_lm.convert import main\n"
            "_orig = g4.Model.sanitize\n"
            "def _patched(self, w):\n"
            "    f = {k: v for k, v in w.items() if not k.removeprefix('model.').startswith(('vision_embedder', 'audio_embedder'))}\n"
            "    return _orig(self, f)\n"
            "g4.Model.sanitize = _patched\n"
            "sys.argv = ['mlx_lm.convert'] + sys.argv[1:]\n"
            "main()\n"
        )
        cmd = [mlx_py, "-c", wrapper_code] + cmd[1:]

    try:
        print(f"[*] Running: {' '.join(cmd if staged_dir is None else [mlx_convert_bin] + cmd[3:])}")
        subprocess.run(cmd, check=True)
        print(f"[+] Successfully converted to MLX at '{out_p}'!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] MLX conversion failed: {e}", file=sys.stderr)
        return False
    finally:
        if temp_hf_dir and temp_hf_dir.exists():
            shutil.rmtree(temp_hf_dir, ignore_errors=True)
        if staged_dir and staged_dir.exists():
            shutil.rmtree(staged_dir, ignore_errors=True)


def convert_to_ollama_mlx(
    source: str,
    model_name: Optional[str] = None,
    q_bits: int = 4,
    output_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    renderer: Optional[str] = None,
    parser_name: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    move_source: bool = False,
    quant: Optional[str] = None,
) -> bool:
    """Chains any source (HF Hub, Safetensors, GGUF, Ollama, MLX) -> MLX -> Ollama MLX."""
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    fmt = info.get("format", "unknown")
    resolved_source = source

    if fmt == "HuggingFace Hub":
        print(f"[*] Step 1/3: Resolving HuggingFace Hub snapshot for '{info['repo_id']}'...")
        snap = resolve_hf_hub_snapshot(info["repo_id"])
        if not snap:
            return False
        resolved_source = str(snap)
        info = inspect_model(resolved_source)
        fmt = info.get("format", "unknown")

    if fmt == "MLX":
        print("[*] Source is already MLX; registering directly in Ollama...")
        return convert_to_ollama(
            resolved_source,
            model_name=model_name,
            system_prompt=system_prompt,
            num_ctx=num_ctx,
            temperature=temperature,
            renderer=renderer,
            parser_name=parser_name,
            capabilities=capabilities,
            move_source=move_source,
        )

    raw_name = (model_name or info.get("name") or Path(info.get("path", "model")).stem)
    safe_dir_name = raw_name.replace(":", "-").replace("/", "_")
    mlx_dir = Path(output_dir).expanduser() if output_dir else _register_temp_path(Path(f"/tmp/ollama_mlx_{safe_dir_name}-{q_bits}bit"))
    temp_mlx = output_dir is None

    print(f"[*] Step 2/3: Converting {fmt} -> Apple Silicon MLX ({q_bits}-bit)...")
    if not convert_to_mlx(resolved_source, output_dir=str(mlx_dir), q_bits=q_bits):
        return False

    target_name = model_name or f"{safe_dir_name.lower()}-mlx"
    print(f"[*] Step 3/3: Registering MLX model '{target_name}' in Ollama...")
    try:
        return convert_to_ollama(
            str(mlx_dir),
            model_name=target_name,
            system_prompt=system_prompt,
            num_ctx=num_ctx,
            temperature=temperature,
            renderer=renderer,
            parser_name=parser_name,
            capabilities=capabilities,
            move_source=move_source and not temp_mlx,
        )
    finally:
        if temp_mlx and mlx_dir.exists():
            shutil.rmtree(mlx_dir, ignore_errors=True)


def quantize_gguf(source: str, target_quant: str = "Q4_K_M", output_file: Optional[str] = None) -> bool:
    """Quantizes a GGUF file, or chains HF Hub / Safetensors / MLX -> quantized GGUF."""
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    fmt = info.get("format", "")
    q_upper = (target_quant or "Q4_K_M").upper()

    if fmt in ("HuggingFace Hub", "HuggingFace Safetensors", "MLX"):
        if fmt == "HuggingFace Hub":
            print(f"[*] Step 1/3: Resolving HuggingFace Hub snapshot for '{info['repo_id']}'...")
            snap = resolve_hf_hub_snapshot(info["repo_id"])
            if not snap:
                return False
            info = inspect_model(str(snap))
            fmt = info.get("format", "")

        src_path = Path(info["path"])
        clean_stem = (info.get("name") or src_path.stem).replace(":", "-").replace("/", "_")
        temp_hf_dir: Optional[Path] = None

        if fmt == "MLX":
            temp_hf_dir = _register_temp_path(Path(f"/tmp/hf_dequant_{clean_stem}"))
            print(f"[*] Step 1/3: Dequantizing MLX -> temporary Safetensors at '{temp_hf_dir}'...")
            if not convert_to_safetensors(str(src_path), output_dir=str(temp_hf_dir), dtype="bfloat16"):
                return False
            src_path = temp_hf_dir

        if not output_file:
            out_p = Path.home() / "ModelExports" / f"{clean_stem}_{q_upper}.gguf"
        else:
            cand = Path(output_file).expanduser()
            if cand.is_dir() or cand.suffix.lower() != ".gguf":
                cand.mkdir(parents=True, exist_ok=True)
                out_p = cand / f"{clean_stem}_{q_upper}.gguf"
            else:
                out_p = cand
        out_p.parent.mkdir(parents=True, exist_ok=True)

        tmp_ol_name = f"_tmp_gguf_{clean_stem.lower()}_{os.getpid()}"
        print(f"[*] Step 2/3: Importing & quantizing Safetensors -> {q_upper} GGUF via Ollama...")
        try:
            ok = convert_to_ollama(str(src_path), model_name=tmp_ol_name, quant=q_upper)
            if not ok:
                return False
            print(f"[*] Step 3/3: Exporting quantized GGUF to '{out_p}'...")
            return export_ollama_model(tmp_ol_name, output_path=str(out_p), to_lmstudio=False, move=False)
        finally:
            delete_ollama_model(tmp_ol_name)
            if temp_hf_dir and temp_hf_dir.exists():
                shutil.rmtree(temp_hf_dir, ignore_errors=True)

    if fmt != "GGUF":
        print(f"[-] Source must be GGUF, HuggingFace Hub, Safetensors, or MLX (got {fmt}).", file=sys.stderr)
        return False

    src_path = Path(info["path"])
    if not output_file:
        out_p = src_path.parent / f"{src_path.stem}_{q_upper}.gguf"
    else:
        cand = Path(output_file).expanduser()
        if cand.is_dir() or cand.suffix.lower() != ".gguf":
            cand.mkdir(parents=True, exist_ok=True)
            out_p = cand / f"{src_path.stem}_{q_upper}.gguf"
        else:
            out_p = cand

    llama_quantize_bin = shutil.which("llama-quantize") or "/opt/homebrew/bin/llama-quantize"
    if not Path(llama_quantize_bin).exists():
        print("[-] llama-quantize not found. Install via: brew install llama.cpp", file=sys.stderr)
        return False

    print(f"[*] Quantizing GGUF: {src_path.name} -> {q_upper}")
    cmd = [llama_quantize_bin, str(src_path), str(out_p), q_upper]

    try:
        print(f"[*] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[+] Quantized GGUF saved to '{out_p}' ({round(out_p.stat().st_size / (1024**3), 2)} GB)!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] llama-quantize failed: {e}", file=sys.stderr)
        return False


def convert_to_safetensors(source: str, output_dir: Optional[str] = None, dtype: str = "bfloat16") -> bool:
    """Exports MLX, F16/BF16/Q8_0 GGUF, or Ollama GGUF to Hugging Face Safetensors."""
    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    fmt = info.get("format", "")
    if fmt == "HuggingFace Hub":
        snap = resolve_hf_hub_snapshot(info["repo_id"])
        if not snap:
            return False
        info = inspect_model(str(snap))
        fmt = info.get("format", "")

    src_path_str = info.get("path")
    if not src_path_str:
        print(f"[-] Cannot resolve file path for source '{source}'.", file=sys.stderr)
        return False

    src_path = Path(src_path_str)
    if not output_dir:
        base_name = (info.get("name") or src_path.stem).replace(":", "-").replace("/", "_")
        output_dir = str(Path.home() / "ModelExports" / f"{base_name}-safetensors")

    out_p = Path(output_dir).expanduser()
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. MLX -> Safetensors via `mlx_lm.convert --dequantize`
    if fmt == "MLX":
        mlx_convert_bin = shutil.which("mlx_lm.convert") or "/opt/homebrew/bin/mlx_lm.convert"
        if not Path(mlx_convert_bin).exists():
            print(f"[-] mlx_lm.convert not found at '{mlx_convert_bin}'.", file=sys.stderr)
            return False

        # mlx_lm.convert expects destination not to exist or overwrites inside it
        if out_p.exists() and not any(out_p.iterdir()):
            out_p.rmdir()

        print(f"[*] Dequantizing MLX -> Hugging Face Safetensors ({dtype}) at '{out_p}'...")
        cmd = [
            mlx_convert_bin,
            "--hf-path", str(src_path),
            "--mlx-path", str(out_p),
            "--dequantize",
            "--dtype", dtype,
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[+] Successfully exported Safetensors to '{out_p}'!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[-] MLX dequantize failed: {e}", file=sys.stderr)
            return False

    # 2. HuggingFace Safetensors -> copy
    if fmt == "HuggingFace Safetensors":
        print(f"[*] Copying HuggingFace Safetensors -> '{out_p}'...")
        shutil.copytree(src_path, out_p, dirs_exist_ok=True)
        print(f"[+] Copied Safetensors to '{out_p}'!")
        return True

    # 3. GGUF / Ollama (GGUF) -> Safetensors (supports F32, F16, BF16, Q8_0)
    if fmt in ("GGUF", "Ollama (GGUF)") and src_path.is_file():
        mlx_py = "/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python"
        py_exec = mlx_py if Path(mlx_py).exists() else sys.executable
        print(f"[*] Extracting GGUF tensors -> Safetensors at '{out_p}'...")

        helper_code = r'''
import json, struct, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

gguf_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

GGUF_SCALAR_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
TYPE_INFO = {
    0: ("F32", 4, 1),
    1: ("F16", 2, 1),
    8: ("Q8_0", 34, 32),
    30: ("BF16", 2, 1),
}

with open(gguf_path, "rb") as f:
    header = f.read(24)
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", header)
    assert magic == b"GGUF", "Not a GGUF file"
    buf = f.read(1024 * 1024 * 32)  # up to 32MB header

pos = 0
alignment = 32
kv_meta = {}

def read_str(p):
    (ln,) = struct.unpack_from("<Q", buf, p)
    return buf[p+8:p+8+ln].decode("utf-8", errors="replace"), p+8+ln

def read_val(p, vtype):
    if vtype == 8:
        return read_str(p)
    if vtype == 9:
        atype, alen = struct.unpack_from("<IQ", buf, p)
        p += 12
        if atype in GGUF_SCALAR_SIZES:
            return None, p + alen * GGUF_SCALAR_SIZES[atype]
        for _ in range(alen):
            _, p = read_val(p, atype)
        return None, p
    fmt_map = {4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if vtype in fmt_map:
        val = struct.unpack_from(fmt_map[vtype], buf, p)[0]
        return val, p + GGUF_SCALAR_SIZES[vtype]
    return None, p + GGUF_SCALAR_SIZES[vtype]

for _ in range(n_kv):
    k, pos = read_str(pos)
    (vtype,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    val, pos = read_val(pos, vtype)
    if k == "general.alignment" and isinstance(val, int):
        alignment = val
    if val is not None and not k.startswith("tokenizer."):
        kv_meta[k] = val

tensors_meta = []
for _ in range(n_tensors):
    tname, pos = read_str(pos)
    (ndims,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    dims = list(struct.unpack_from(f"<{ndims}Q", buf, pos))
    pos += 8 * ndims
    ggml_type, rel_offset = struct.unpack_from("<IQ", buf, pos)
    pos += 12
    tensors_meta.append((tname, dims, ggml_type, rel_offset))

data_offset = 24 + ((pos + alignment - 1) // alignment * alignment)

unsupported = set(t[2] for t in tensors_meta if t[2] not in TYPE_INFO)
if unsupported:
    print(f"ERROR: Unsupported quantized GGUF tensor types {unsupported}. Only F32, F16, BF16, and Q8_0 can be exported cleanly to Safetensors.", file=sys.stderr)
    sys.exit(2)

out_tensors = {}
with open(gguf_path, "rb") as f:
    for tname, dims, gtype, rel_offset in tensors_meta:
        shape = tuple(reversed(dims))
        numel = int(np.prod(shape))
        f.seek(data_offset + rel_offset)
        t_label, bsize, bcount = TYPE_INFO[gtype]
        nbytes = (numel // bcount) * bsize
        raw = f.read(nbytes)
        if gtype == 0:
            arr = np.frombuffer(raw, dtype="<f4").reshape(shape)
        elif gtype == 1:
            arr = np.frombuffer(raw, dtype="<f2").reshape(shape)
        elif gtype == 30:
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            arr = u16.view(np.float32).astype(np.float16).reshape(shape)
        elif gtype == 8:
            qdtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
            blocks = np.frombuffer(raw, dtype=qdtype)
            arr = (blocks["qs"].astype(np.float32) * blocks["d"][:, None].astype(np.float32)).astype(np.float16).reshape(shape)
        out_tensors[tname] = np.ascontiguousarray(arr)

save_file(out_tensors, str(out_dir / "model.safetensors"))
(out_dir / "gguf_metadata.json").write_text(json.dumps(kv_meta, indent=2))
print(f"Saved {len(out_tensors)} tensors to {out_dir / 'model.safetensors'}")
'''
        res = subprocess.run([py_exec, "-c", helper_code, str(src_path), str(out_p)])
        if res.returncode == 0:
            print(f"[+] Successfully exported Safetensors to '{out_p}'!")
            return True
        return False

    print(f"[-] Safetensors export from '{fmt}' is not supported.", file=sys.stderr)
    return False


def convert_to_onnx(source: str, output_dir: Optional[str] = None, onnx_quant: str = "q4f16") -> bool:
    """Exports HuggingFace Safetensors, HuggingFace Hub, or MLX (auto-dequantized) to ONNX with optional quantization."""
    info = inspect_model(source)
    fmt = info.get("format", "")
    temp_hf_dir: Optional[Path] = None

    if fmt == "HuggingFace Hub":
        snap = resolve_hf_hub_snapshot(info["repo_id"])
        if not snap:
            return False
        info = inspect_model(str(snap))
        fmt = info.get("format", "")

    if fmt in ("MLX", "GGUF", "Ollama (GGUF)"):
        base_name = (info.get("name") or Path(info.get("path", "model")).stem).replace(":", "-").replace("/", "_")
        temp_hf_dir = _register_temp_path(Path(f"/tmp/hf_dequant_{base_name}"))
        print(f"[*] Step 1/2: Extracting {fmt} -> temporary Safetensors at '{temp_hf_dir}'...")
        if not convert_to_safetensors(source, output_dir=str(temp_hf_dir), dtype="float16"):
            return False
        info = inspect_model(str(temp_hf_dir))
        fmt = info.get("format", "")

    if fmt != "HuggingFace Safetensors":
        print(f"[-] ONNX export requires HuggingFace Hub, Safetensors, MLX, or GGUF/Ollama source (got {fmt}).", file=sys.stderr)
        return False

    src_path = Path(info["path"])
    model_type = info.get("model_type", "").lower()

    if not output_dir:
        base_name = (info.get("name") or src_path.name).replace(":", "-")
        output_dir = str(Path.home() / "ModelExports" / f"{base_name}-ONNX-{onnx_quant}")
    out_p = Path(output_dir).expanduser()

    print(f"[*] Exporting HuggingFace ({model_type}) -> ONNX ({onnx_quant}) at '{out_p}'")

    script_dir = Path(__file__).resolve().parent
    threecoder_scripts = Path.home() / "dev" / "threecoder" / "scripts"
    search_dirs = [script_dir, threecoder_scripts]

    export_name = None
    if "lfm" in model_type or "liquid" in str(src_path).lower():
        export_name = "export_lfm3_onnx_q4f16.py"
    elif "qwen" in model_type or "qwen" in str(src_path).lower():
        export_name = "export_qwen35_onnx_q4f16.py"

    export_script = None
    if export_name and onnx_quant in ("q4f16", "int4"):
        for d in search_dirs:
            if (d / export_name).exists():
                export_script = d / export_name
                break

    if export_script:
        cmd = [sys.executable, str(export_script), "--model-dir", str(src_path), "--output-dir", str(out_p)]
        print(f"[*] Running architecture-specific WebGPU ({onnx_quant}) exporter: {export_script}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[+] Successfully exported WebGPU ONNX ({onnx_quant}) model to '{out_p}'!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[-] ONNX export failed: {e}", file=sys.stderr)
            return False
    elif shutil.which("optimum-cli"):
        cmd = ["optimum-cli", "export", "onnx", "--model", str(src_path), "--task", "text-generation-with-past", str(out_p)]
        if onnx_quant == "fp16":
            cmd.extend(["--dtype", "fp16"])
        print(f"[*] Step 1/2: Exporting ONNX graph via optimum-cli...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[-] optimum-cli export failed: {e}", file=sys.stderr)
            return False

        # Step 2: Quantize ONNX weights if 4-bit or 8-bit requested
        if onnx_quant in ("q4f16", "int4", "int8"):
            print(f"[*] Step 2/2: Quantizing ONNX weights ({onnx_quant})...")
            onnx_files = list(out_p.glob("*.onnx"))
            if onnx_files:
                q_code = r'''
import sys
from pathlib import Path
onnx_file = Path(sys.argv[1])
q_mode = sys.argv[2]
if q_mode in ("q4f16", "int4"):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    import onnx
    model = onnx.load(str(onnx_file), load_external_data=True)
    quant = MatMulNBitsQuantizer(model=model, block_size=32, is_symmetric=True, bits=4)
    quant.process()
    out_file = onnx_file.with_name(onnx_file.stem + f"_{q_mode}.onnx")
    quant.model.save_model_to_file(str(out_file), use_external_data_format=True)
    print(f"Quantized 4-bit ONNX saved to {out_file}")
elif q_mode == "int8":
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out_file = onnx_file.with_name(onnx_file.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_file), str(out_file), weight_type=QuantType.QInt8)
    print(f"Quantized INT8 ONNX saved to {out_file}")
'''
                res = subprocess.run([sys.executable, "-c", q_code, str(onnx_files[0]), onnx_quant])
                if res.returncode != 0:
                    print("[-] Warning: ONNX quantization step failed (is onnxruntime installed?). Unquantized ONNX is available in output dir.", file=sys.stderr)
        print(f"[+] Successfully exported ONNX ({onnx_quant}) model to '{out_p}'!")
        return True
    else:
        print(f"[-] No ONNX exporter found for architecture '{model_type}' (install optimum or provide an exporter script).", file=sys.stderr)
        return False


def convert_to_lmstudio(
    source: str,
    lms_format: str = "auto",
    q_bits: int = 4,
    quant: str = "Q4_K_M",
    dtype: str = "bfloat16",
    output_dir: Optional[str] = None,
    move_source: bool = False,
) -> bool:
    """Converts/downloads any source directly into ~/.lmstudio/models/<owner>/<model>."""
    if source.startswith("ollama:") and lms_format in ("auto", "gguf"):
        ol_name = source.replace("ollama:", "", 1)
        return export_ollama_model(ol_name, output_path=output_dir, to_lmstudio=(output_dir is None), move=move_source)

    info = inspect_model(source)
    if "error" in info:
        print(f"[-] Error inspecting source: {info['error']}", file=sys.stderr)
        return False

    fmt = info.get("format", "")
    # Determine publisher/model folder inside ~/.lmstudio/models
    if fmt == "HuggingFace Hub":
        repo_id = info["repo_id"]
        parts = repo_id.split("/", 1)
        owner = parts[0] if len(parts) == 2 else "hf-imports"
        repo_name = parts[1] if len(parts) == 2 else parts[0]
    else:
        owner = "local-imports"
        raw_name = info.get("name") or Path(info.get("path", "model")).stem
        repo_name = raw_name.replace(":", "-").replace("/", "_")

    if output_dir:
        dest_dir = Path(output_dir).expanduser()
    else:
        suffix = ""
        if lms_format == "mlx":
            suffix = f"-mlx-{q_bits}bit" if "mlx" not in repo_name.lower() else ""
        elif lms_format == "safetensors":
            suffix = "-safetensors"
        elif lms_format == "gguf":
            suffix = f"-gguf-{quant.lower()}"
        dest_dir = LM_STUDIO_MODELS_DIR / owner / f"{repo_name}{suffix}"

    if lms_format == "auto":
        if fmt == "HuggingFace Hub":
            return download_hf_repo(info["repo_id"], output_dir=str(dest_dir), to_lmstudio=False)
        src_p = Path(info["path"])
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Importing {fmt} -> LM Studio at '{dest_dir}'...")
        if src_p.is_dir():
            if move_source:
                shutil.move(str(src_p), str(dest_dir))
            else:
                shutil.copytree(src_p, dest_dir, dirs_exist_ok=True)
        elif src_p.is_file():
            dest_dir.mkdir(parents=True, exist_ok=True)
            target_file = dest_dir / src_p.name
            if move_source:
                shutil.move(str(src_p), str(target_file))
            else:
                shutil.copy2(src_p, target_file)
        print(f"[+] Imported into LM Studio at '{dest_dir}'!")
        return True

    if lms_format == "mlx":
        if fmt == "MLX":
            src_p = Path(info["path"])
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"[*] Copying MLX model -> LM Studio at '{dest_dir}'...")
            if move_source:
                shutil.move(str(src_p), str(dest_dir))
            else:
                shutil.copytree(src_p, dest_dir, dirs_exist_ok=True)
            print(f"[+] MLX model ready in LM Studio at '{dest_dir}'!")
            return True
        ok = convert_to_mlx(source, output_dir=str(dest_dir), q_bits=q_bits)
        if ok and move_source and not source.startswith(("hf:", "ollama:")):
            src_p = Path(info["path"])
            if src_p.is_dir():
                shutil.rmtree(src_p, ignore_errors=True)
            elif src_p.is_file():
                src_p.unlink(missing_ok=True)
        return ok

    if lms_format == "safetensors":
        return convert_to_safetensors(source, output_dir=str(dest_dir), dtype=dtype)

    if lms_format == "gguf":
        if fmt == "GGUF":
            dest_dir.mkdir(parents=True, exist_ok=True)
            src_p = Path(info["path"])
            out_gguf = dest_dir / f"{src_p.stem}_{quant.upper()}.gguf"
            return quantize_gguf(source, target_quant=quant, output_file=str(out_gguf))
        print(f"[-] Direct {fmt} -> GGUF for LM Studio requires a GGUF or Ollama source.", file=sys.stderr)
        return False

    print(f"[-] Unsupported LM Studio format '{lms_format}'.", file=sys.stderr)
    return False


def get_model_card(source: str) -> Dict[str, Any]:
    """Reads editable Model Card / metadata for Ollama or local LM Studio/HF models."""
    if source.startswith("ollama:"):
        ol_name = source.replace("ollama:", "", 1)
        _, mf_text = get_ollama_modelfile_blob(ol_name)
        meta = get_ollama_show_metadata(ol_name)
        caps: List[str] = []
        renderer = meta.get("renderer", "")
        parser_name = meta.get("parser", "")
        system_lines: List[str] = []
        num_ctx: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None

        in_system = False
        for line in mf_text.splitlines():
            stripped = line.strip()
            if in_system:
                if stripped.endswith('"""'):
                    system_lines.append(line[: line.rfind('"""')])
                    in_system = False
                else:
                    system_lines.append(line)
                continue
            upper = stripped.upper()
            if upper.startswith("RENDERER "):
                renderer = stripped[9:].strip()
            elif upper.startswith("PARSER "):
                parser_name = stripped[7:].strip()
            elif upper.startswith("CAPABILITY "):
                caps.append(stripped[11:].strip())
            elif upper.startswith("SYSTEM "):
                rest = stripped[7:].strip()
                if rest.startswith('"""'):
                    inner = rest[3:]
                    if inner.endswith('"""') and len(inner) >= 3:
                        system_lines.append(inner[:-3])
                    else:
                        system_lines.append(inner)
                        in_system = True
                else:
                    system_lines.append(rest.strip('"'))
            elif upper.startswith("PARAMETER "):
                parts = stripped[10:].strip().split(None, 1)
                if len(parts) == 2:
                    pk, pv = parts[0].lower(), parts[1].strip()
                    try:
                        if pk == "num_ctx":
                            num_ctx = int(pv)
                        elif pk == "temperature":
                            temperature = float(pv)
                        elif pk == "top_p":
                            top_p = float(pv)
                        elif pk == "top_k":
                            top_k = int(pv)
                    except ValueError:
                        pass

        return {
            "kind": "ollama",
            "name": ol_name,
            "renderer": renderer,
            "parser": parser_name,
            "capabilities": ", ".join(caps) if caps else "completion",
            "num_ctx": num_ctx or 32768,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "system": "\n".join(system_lines).strip(),
            "modelfile": mf_text,
        }

    p = Path(source).expanduser()
    if p.is_dir():
        gen_cfg_path = p / "generation_config.json"
        gen_cfg: Dict[str, Any] = {}
        if gen_cfg_path.exists():
            try:
                gen_cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "kind": "local",
            "name": p.name,
            "path": str(p),
            "temperature": gen_cfg.get("temperature"),
            "top_p": gen_cfg.get("top_p"),
            "top_k": gen_cfg.get("top_k"),
            "max_new_tokens": gen_cfg.get("max_new_tokens"),
        }

    return {"error": "Model Card editing is supported for Ollama models and local model folders."}


def edit_model_card(
    source: str,
    new_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    renderer: Optional[str] = None,
    parser_name: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
) -> bool:
    """Updates an Ollama Model Card (zero-copy FROM <model>) or local generation_config.json without converting weights."""
    if source.startswith("ollama:"):
        ol_name = source.replace("ollama:", "", 1)
        target_name = (new_name or ol_name).strip()

        lines = [f"FROM {ol_name}"]
        if renderer:
            lines.append(f"RENDERER {renderer}")
        if parser_name:
            lines.append(f"PARSER {parser_name}")
        if capabilities:
            for cap in capabilities:
                if cap:
                    lines.append(f"CAPABILITY {cap}")
        if system_prompt is not None and system_prompt.strip():
            lines.append(f'SYSTEM """{system_prompt}"""')
        if num_ctx:
            lines.append(f"PARAMETER num_ctx {num_ctx}")
        if temperature is not None:
            lines.append(f"PARAMETER temperature {temperature}")
        if top_p is not None:
            lines.append(f"PARAMETER top_p {top_p}")
        if top_k is not None:
            lines.append(f"PARAMETER top_k {top_k}")

        modelfile_content = "\n".join(lines) + "\n"
        modelfile_path = _register_temp_path(Path(f"/tmp/Modelfile_edit_{target_name.replace(':', '_').replace('/', '_')}"))
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        print(f"[*] Updating Ollama Model Card '{target_name}' (zero-copy, no weight conversion):")
        print("-" * 50)
        for l in lines:
            print(f"    {l}")
        print("-" * 50)
        try:
            subprocess.run(["ollama", "create", target_name, "-f", str(modelfile_path)], check=True)
            print(f"[+] Updated Ollama Model Card for '{target_name}'!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[-] Failed to update Ollama Model Card: {e}", file=sys.stderr)
            return False
        finally:
            if modelfile_path.exists():
                modelfile_path.unlink(missing_ok=True)

    p = Path(source).expanduser()
    if p.is_dir():
        gen_cfg_path = p / "generation_config.json"
        cfg: Dict[str, Any] = {}
        if gen_cfg_path.exists():
            try:
                cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if temperature is not None:
            cfg["temperature"] = temperature
        if top_p is not None:
            cfg["top_p"] = top_p
        if top_k is not None:
            cfg["top_k"] = top_k
        gen_cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"[+] Updated generation_config.json at '{gen_cfg_path}'!")
        return True

    print("[-] Card editing is only supported for Ollama models and local model directories.", file=sys.stderr)
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
    convert_parser.add_argument("source", help="Source model file, HF directory, LM Studio ID, or 'ollama:<name>'")
    convert_parser.add_argument("--to", required=True, choices=["ollama", "ollama-mlx", "mlx", "onnx", "gguf", "safetensors", "lmstudio"], help="Target format")
    convert_parser.add_argument("--name", help="Model name for Ollama")
    convert_parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16], help="Quantization bits for MLX (default: 4)")
    convert_parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Target dtype for Safetensors export")
    convert_parser.add_argument("--onnx-quant", default="q4f16", choices=["q4f16", "int4", "int8", "fp16"], help="Target quantization/precision for ONNX export")
    convert_parser.add_argument("--quant", default=None, help="Target quantization for GGUF / Ollama Safetensors import (e.g. Q4_K_M, Q8_0)")
    convert_parser.add_argument("--lms-format", default="auto", choices=["auto", "mlx", "gguf", "safetensors"], help="Target sub-format when --to lmstudio")
    convert_parser.add_argument("--output", help="Output path/directory")
    convert_parser.add_argument("--system", help="System prompt to embed (for Ollama)")
    convert_parser.add_argument("--ctx", type=int, help="Context length num_ctx (for Ollama)")
    convert_parser.add_argument("--temp", type=float, help="Temperature parameter (for Ollama)")
    convert_parser.add_argument("--renderer", help="Ollama RENDERER (e.g. qwen3.5, qwen3.8, gemma4)")
    convert_parser.add_argument("--parser", dest="parser_name", help="Ollama PARSER (e.g. qwen3.5)")
    convert_parser.add_argument("--capabilities", help="Comma-separated Ollama CAPABILITY values (e.g. completion,tools,thinking)")
    convert_parser.add_argument("--move", action="store_true", help="Remove original source after successful import/export to save disk space")

    # Export from Ollama
    export_ol_parser = subparsers.add_parser("export-ollama", help="Export an Ollama model to GGUF or portable bundle")
    export_ol_parser.add_argument("name", help="Ollama model name (e.g. lfm3:1.7b)")
    export_ol_parser.add_argument("--output", help="Output directory or .gguf path")
    export_ol_parser.add_argument("--to-lmstudio", action="store_true", help="Export directly into ~/.lmstudio/models/ollama-export/")
    export_ol_parser.add_argument("--move", action="store_true", help="Remove from Ollama after export")

    # Archive
    archive_parser = subparsers.add_parser("archive", help="Copy or move a model (Ollama or LM Studio) to an archive/external drive folder")
    archive_parser.add_argument("source", help="Local path or 'ollama:<name>'")
    archive_parser.add_argument("--dest", required=True, help="Archive destination folder (e.g. /Volumes/External/ModelArchive)")
    archive_parser.add_argument("--move", action="store_true", help="Delete original after archiving")

    # Delete Ollama
    del_ol_parser = subparsers.add_parser("delete-ollama", help="Delete a model from Ollama registry")
    del_ol_parser.add_argument("name", help="Ollama model name")

    # Delete LM Studio
    del_lm_parser = subparsers.add_parser("delete-lmstudio", help="Delete a model inside ~/.lmstudio/models")
    del_lm_parser.add_argument("path", help="Path to LM Studio model file or folder")

    # Download from Hugging Face
    dl_parser = subparsers.add_parser("download", help="Download a Hugging Face repo to LM Studio or a custom folder")
    dl_parser.add_argument("repo", help="Hugging Face repo ID (e.g. mlx-community/Qwen3.8-27B-4bit)")
    dl_parser.add_argument("--to-lmstudio", action="store_true", help="Download into ~/.lmstudio/models/<owner>/<repo>")
    dl_parser.add_argument("--output", help="Custom destination directory")

    # Get / Edit Model Card without weight conversion
    get_card_parser = subparsers.add_parser("get-card", help="Read Model Card / generation_config for Ollama or local model")
    get_card_parser.add_argument("source", help="'ollama:<name>' or local model folder")

    edit_card_parser = subparsers.add_parser("edit-card", help="Update Ollama Model Card or local generation_config without weight conversion")
    edit_card_parser.add_argument("source", help="'ollama:<name>' or local model folder")
    edit_card_parser.add_argument("--name", help="New/target Ollama model name")
    edit_card_parser.add_argument("--system", help="System prompt")
    edit_card_parser.add_argument("--ctx", type=int, help="Context length num_ctx")
    edit_card_parser.add_argument("--temp", type=float, help="Temperature")
    edit_card_parser.add_argument("--top-p", type=float, help="Top-p")
    edit_card_parser.add_argument("--top-k", type=int, help="Top-k")
    edit_card_parser.add_argument("--renderer", help="Ollama RENDERER")
    edit_card_parser.add_argument("--parser", dest="parser_name", help="Ollama PARSER")
    edit_card_parser.add_argument("--capabilities", help="Comma-separated Ollama CAPABILITY values")

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
        if args.source.startswith("ollama:") and args.to in ("gguf", "safetensors", "ollama"):
            if args.to == "gguf":
                ol_name = args.source.replace("ollama:", "", 1)
                if args.quant:
                    tmp_raw = _register_temp_path(Path(f"/tmp/ollama_raw_{os.getpid()}.gguf"))
                    if export_ollama_model(ol_name, output_path=str(tmp_raw), to_lmstudio=False, move=False):
                        ok = quantize_gguf(str(tmp_raw), target_quant=args.quant, output_file=args.output)
                        if ok and args.move:
                            delete_ollama_model(ol_name)
                    if tmp_raw.exists():
                        tmp_raw.unlink(missing_ok=True)
                else:
                    ok = export_ollama_model(ol_name, output_path=args.output, to_lmstudio=False, move=args.move)
            elif args.to == "safetensors":
                ok = convert_to_safetensors(args.source, output_dir=args.output, dtype=args.dtype)
            elif args.to == "ollama":
                caps = [c.strip() for c in args.capabilities.split(",")] if args.capabilities else None
                ok = edit_model_card(
                    args.source,
                    new_name=args.name,
                    system_prompt=args.system,
                    num_ctx=args.ctx,
                    temperature=args.temp,
                    renderer=args.renderer,
                    parser_name=args.parser_name,
                    capabilities=caps,
                )
        elif args.to == "ollama":
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
                quant=args.quant,
            )
        elif args.to == "ollama-mlx":
            caps = [c.strip() for c in args.capabilities.split(",")] if args.capabilities else None
            ok = convert_to_ollama_mlx(
                args.source,
                model_name=args.name,
                q_bits=args.bits,
                output_dir=args.output,
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
            ok = quantize_gguf(args.source, target_quant=args.quant or "Q4_K_M", output_file=args.output)
        elif args.to == "safetensors":
            ok = convert_to_safetensors(args.source, output_dir=args.output, dtype=args.dtype)
        elif args.to == "onnx":
            ok = convert_to_onnx(args.source, output_dir=args.output, onnx_quant=args.onnx_quant)
        elif args.to == "lmstudio":
            ok = convert_to_lmstudio(
                args.source,
                lms_format=args.lms_format,
                q_bits=args.bits,
                quant=args.quant or "Q4_K_M",
                dtype=args.dtype,
                output_dir=args.output,
                move_source=args.move,
            )
        sys.exit(0 if ok else 1)

    elif args.command == "get-card":
        print(json.dumps(get_model_card(args.source), indent=2))

    elif args.command == "edit-card":
        caps = [c.strip() for c in args.capabilities.split(",")] if args.capabilities else None
        ok = edit_model_card(
            args.source,
            new_name=args.name,
            system_prompt=args.system,
            num_ctx=args.ctx,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            renderer=args.renderer,
            parser_name=args.parser_name,
            capabilities=caps,
        )
        sys.exit(0 if ok else 1)

    elif args.command == "export-ollama":
        ok = export_ollama_model(args.name, output_path=args.output, to_lmstudio=args.to_lmstudio, move=args.move)
        sys.exit(0 if ok else 1)

    elif args.command == "archive":
        ok = archive_model(args.source, archive_dir=args.dest, move=args.move)
        sys.exit(0 if ok else 1)

    elif args.command == "delete-ollama":
        ok = delete_ollama_model(args.name)
        sys.exit(0 if ok else 1)

    elif args.command == "delete-lmstudio":
        ok = delete_lmstudio_model(args.path)
        sys.exit(0 if ok else 1)

    elif args.command == "download":
        clean_id = args.repo.replace("hf:", "", 1).strip()
        if args.to_lmstudio and "/" in clean_id:
            owner, repo = clean_id.split("/", 1)
            dest = LM_STUDIO_MODELS_DIR / owner / repo
        elif args.output:
            dest = Path(args.output).expanduser()
        else:
            dest = Path.home() / "ModelExports" / clean_id.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)
        mlx_py = "/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python"
        py_exec = mlx_py if Path(mlx_py).exists() else sys.executable
        print(f"[*] Downloading '{clean_id}' -> {dest} ...")
        code = (
            "import sys; "
            "from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])"
        )
        res = subprocess.run([py_exec, "-c", code, clean_id, str(dest)])
        if res.returncode == 0:
            print(f"[+] Downloaded '{clean_id}' to '{dest}'!")
        sys.exit(res.returncode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

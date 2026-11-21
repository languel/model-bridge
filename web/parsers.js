// Pure-JS Zero-Copy Model Parsers (GGUF v2/v3, Safetensors, HF/MLX config.json, Modelfile)
// Reads only file headers via File.slice() — safe for 100GB+ local models in the browser.

export const GGML_TYPES = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
  10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
  16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
  21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
  27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
  39: "MXFP4",
};

export const GGUF_FILE_TYPES = {
  0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
  7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1",
  10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L",
  14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M",
  18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS", 21: "MOSTLY_Q2_K_S",
  22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS", 24: "MOSTLY_IQ1_S", 25: "MOSTLY_IQ4_NL",
  26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S", 29: "MOSTLY_IQ2_M",
  30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16",
};

const DTYPE_BYTES = {
  F64: 8, I64: 8, U64: 8,
  F32: 4, I32: 4, U32: 4,
  F16: 2, BF16: 2, I16: 2, U16: 2,
  F8_E4M3: 1, F8_E5M2: 1, I8: 1, U8: 1, BOOL: 1,
};

const textDecoder = new TextDecoder("utf-8", { fatal: false });

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, i)).toFixed(i >= 3 ? 2 : 1)} ${units[i]}`;
}

export function formatParams(n) {
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

/**
 * Zero-copy GGUF v2/v3 header parser using File.slice().
 */
export async function parseGGUF(file, maxHeaderBytes = 24 * 1024 * 1024) {
  const sliceLen = Math.min(file.size, maxHeaderBytes);
  const buf = await file.slice(0, sliceLen).arrayBuffer();
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);

  if (buf.byteLength < 24) {
    throw new Error("File is too small to be a valid GGUF file.");
  }

  const magic = textDecoder.decode(bytes.subarray(0, 4));
  if (magic !== "GGUF") {
    throw new Error(`Invalid GGUF magic header: "${magic}" (expected "GGUF").`);
  }

  const version = view.getUint32(4, true);
  let nTensors, nKv, pos;
  if (version === 1) {
    nTensors = view.getUint32(8, true);
    nKv = view.getUint32(12, true);
    pos = 16;
  } else {
    nTensors = Number(view.getBigUint64(8, true));
    nKv = Number(view.getBigUint64(16, true));
    pos = 24;
  }

  function readStr() {
    if (pos + 8 > buf.byteLength) return null;
    const len = Number(view.getBigUint64(pos, true));
    pos += 8;
    if (len < 0 || pos + len > buf.byteLength) return null;
    const s = textDecoder.decode(bytes.subarray(pos, pos + len));
    pos += len;
    return s;
  }

  function skipStr() {
    if (pos + 8 > buf.byteLength) return false;
    const len = Number(view.getBigUint64(pos, true));
    pos += 8;
    if (len < 0 || pos + len > buf.byteLength) return false;
    pos += len;
    return true;
  }

  const SCALAR_SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

  function readScalar(t) {
    const sz = SCALAR_SIZE[t];
    if (!sz || pos + sz > buf.byteLength) return undefined;
    let v;
    switch (t) {
      case 0: v = view.getUint8(pos); break;
      case 1: v = view.getInt8(pos); break;
      case 2: v = view.getUint16(pos, true); break;
      case 3: v = view.getInt16(pos, true); break;
      case 4: v = view.getUint32(pos, true); break;
      case 5: v = view.getInt32(pos, true); break;
      case 6: v = Number(view.getFloat32(pos, true).toPrecision(6)); break;
      case 7: v = view.getUint8(pos) !== 0; break;
      case 10: v = Number(view.getBigUint64(pos, true)); break;
      case 11: v = Number(view.getBigInt64(pos, true)); break;
      case 12: v = view.getFloat64(pos, true); break;
    }
    pos += sz;
    return v;
  }

  function readValue(t) {
    if (t === 8) return readStr();
    if (t === 9) {
      if (pos + 12 > buf.byteLength) return undefined;
      const elemType = view.getUint32(pos, true);
      const count = Number(view.getBigUint64(pos + 4, true));
      pos += 12;
      // Fast-skip large arrays (e.g., 150k tokenizer vocab)
      if (elemType !== 8 && elemType !== 9 && SCALAR_SIZE[elemType]) {
        const total = count * SCALAR_SIZE[elemType];
        if (pos + total > buf.byteLength) return undefined;
        if (count <= 16) {
          const arr = [];
          for (let i = 0; i < count; i++) arr.push(readScalar(elemType));
          return arr;
        }
        pos += total;
        return `[Array(${count})]`;
      }
      if (elemType === 8) {
        if (count <= 16) {
          const arr = [];
          for (let i = 0; i < count; i++) {
            const s = readStr();
            if (s === null) return undefined;
            arr.push(s);
          }
          return arr;
        }
        for (let i = 0; i < count; i++) {
          if (!skipStr()) return undefined;
        }
        return `[StringArray(${count})]`;
      }
      return undefined;
    }
    return readScalar(t);
  }

  const metadata = {};
  let kvParsed = 0;
  let truncated = false;

  for (let i = 0; i < nKv; i++) {
    const key = readStr();
    if (key === null || pos + 4 > buf.byteLength) {
      truncated = true;
      break;
    }
    const vtype = view.getUint32(pos, true);
    pos += 4;
    const val = readValue(vtype);
    if (val === undefined || val === null) {
      truncated = true;
      break;
    }
    metadata[key] = val;
    kvParsed++;
  }

  // Parse tensor info headers if KV section completed within buffer
  const tensors = [];
  const dtypeCounts = {};
  let totalParams = 0;
  if (!truncated) {
    for (let i = 0; i < nTensors; i++) {
      const name = readStr();
      if (name === null || pos + 4 > buf.byteLength) break;
      const nDims = view.getUint32(pos, true);
      pos += 4;
      if (pos + nDims * 8 + 12 > buf.byteLength) break;
      const dims = [];
      let elems = 1;
      for (let d = 0; d < nDims; d++) {
        const dim = Number(view.getBigUint64(pos, true));
        pos += 8;
        dims.push(dim);
        elems *= dim;
      }
      const ggmlType = view.getUint32(pos, true);
      pos += 4;
      const offset = Number(view.getBigUint64(pos, true));
      pos += 8;
      const dtype = GGML_TYPES[ggmlType] || `TYPE_${ggmlType}`;
      dtypeCounts[dtype] = (dtypeCounts[dtype] || 0) + 1;
      totalParams += elems;
      if (tensors.length < 300) {
        tensors.push({ name, shape: dims, dtype, params: elems, offset });
      }
    }
  }

  const arch = metadata["general.architecture"] || "unknown";
  const fileTypeId = metadata["general.file_type"];
  const quantLabel = fileTypeId !== undefined ? (GGUF_FILE_TYPES[fileTypeId] || `type_${fileTypeId}`) : "unknown";

  return {
    format: "GGUF",
    name: metadata["general.name"] || file.name.replace(/\.gguf$/i, ""),
    filename: file.name,
    fileSizeBytes: file.size,
    version,
    architecture: arch,
    quantization: quantLabel,
    nTensors,
    nKv,
    kvParsed,
    truncated,
    totalParams: totalParams || null,
    contextLength: metadata[`${arch}.context_length`] || null,
    blockCount: metadata[`${arch}.block_count`] || null,
    embeddingLength: metadata[`${arch}.embedding_length`] || null,
    headCount: metadata[`${arch}.attention.head_count`] || null,
    headCountKv: metadata[`${arch}.attention.head_count_kv`] || metadata[`${arch}.attention.head_count`] || null,
    vocabSize: metadata[`${arch}.vocab_size`] || null,
    chatTemplate: metadata["tokenizer.chat_template"] || null,
    dtypeCounts,
    tensors,
    metadata,
  };
}

/**
 * Zero-copy Safetensors header parser using File.slice().
 */
export async function parseSafetensors(file) {
  if (file.size < 8) {
    throw new Error("File is too small to be a valid Safetensors file.");
  }
  const lenBuf = await file.slice(0, 8).arrayBuffer();
  const headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
  if (headerLen <= 0 || headerLen > 100 * 1024 * 1024 || 8 + headerLen > file.size) {
    throw new Error(`Invalid Safetensors header length: ${headerLen}`);
  }
  const headerBuf = await file.slice(8, 8 + headerLen).arrayBuffer();
  const headerJson = JSON.parse(textDecoder.decode(new Uint8Array(headerBuf)));

  const metadata = headerJson.__metadata__ || {};
  const tensors = [];
  const dtypeCounts = {};
  const dtypeBytes = {};
  let totalParams = 0;
  let totalTensorCount = 0;

  for (const [name, info] of Object.entries(headerJson)) {
    if (name === "__metadata__") continue;
    totalTensorCount++;
    const shape = info.shape || [];
    const dtype = info.dtype || "UNKNOWN";
    const params = shape.reduce((a, b) => a * b, 1);
    const bytes = info.data_offsets
      ? info.data_offsets[1] - info.data_offsets[0]
      : params * (DTYPE_BYTES[dtype] || 2);
    totalParams += params;
    dtypeCounts[dtype] = (dtypeCounts[dtype] || 0) + 1;
    dtypeBytes[dtype] = (dtypeBytes[dtype] || 0) + bytes;
    if (tensors.length < 300) {
      tensors.push({ name, shape, dtype, params, bytes });
    }
  }

  const dominantDtype = Object.entries(dtypeBytes).sort((a, b) => b[1] - a[1])[0]?.[0] || "unknown";

  return {
    format: "Safetensors",
    name: file.name.replace(/\.safetensors$/i, ""),
    filename: file.name,
    fileSizeBytes: file.size,
    architecture: metadata.format || "safetensors",
    quantization: dominantDtype,
    nTensors: totalTensorCount,
    totalParams,
    dtypeCounts,
    dtypeBytes,
    tensors,
    metadata,
  };
}

/**
 * Inspects a Hugging Face / MLX config.json (plus optional sibling files if a folder was dropped).
 */
export async function parseConfigJson(configFile, allFiles = []) {
  const text = await configFile.text();
  const cfg = JSON.parse(text);
  const textCfg = cfg.text_config || cfg;

  const isMlx = Boolean(cfg.quantization);
  const qCfg = cfg.quantization || cfg.quantization_config || null;

  const weightFiles = allFiles.filter((f) => /\.(safetensors|bin|gguf)$/i.test(f.name));
  const totalBytes = weightFiles.reduce((sum, f) => sum + f.size, 0);

  let quantLabel = textCfg.torch_dtype || cfg.torch_dtype || "unknown";
  if (qCfg && typeof qCfg === "object") {
    if (qCfg.bits) quantLabel = `${isMlx ? "MLX " : ""}${qCfg.bits}-bit (g${qCfg.group_size || 64})`;
    else if (qCfg.quant_method) quantLabel = qCfg.quant_method.toUpperCase();
  }

  const folderName = configFile.webkitRelativePath
    ? configFile.webkitRelativePath.split("/")[0]
    : (cfg._name_or_path || cfg.model_type || "model");

  return {
    format: isMlx ? "MLX" : "HuggingFace Safetensors",
    name: folderName,
    filename: configFile.name,
    fileSizeBytes: totalBytes || configFile.size,
    shardCount: weightFiles.length || null,
    architecture: (cfg.architectures && cfg.architectures[0]) || cfg.model_type || "unknown",
    modelType: cfg.model_type || "unknown",
    quantization: quantLabel,
    contextLength: textCfg.max_position_embeddings || cfg.max_position_embeddings || cfg.context_length || null,
    blockCount: textCfg.num_hidden_layers || null,
    embeddingLength: textCfg.hidden_size || null,
    headCount: textCfg.num_attention_heads || null,
    headCountKv: textCfg.num_key_value_heads || textCfg.num_attention_heads || null,
    vocabSize: textCfg.vocab_size || cfg.vocab_size || null,
    metadata: cfg,
    tensors: [],
    dtypeCounts: {},
  };
}

/**
 * Mirrors cli/model_converter.py's infer_ollama_card_defaults().
 */
export function inferOllamaDefaults(info = {}) {
  const combined = `${info.architecture || ""} ${info.modelType || ""} ${info.name || ""}`.toLowerCase();
  const defaults = {
    renderer: "",
    parser: "",
    capabilities: ["completion"],
    num_ctx: Math.min(info.contextLength || 32768, 32768),
    temperature: 0.6,
    top_p: 0.95,
    top_k: 30,
  };

  if (combined.includes("qwen3_5") || combined.includes("qwen3.5")) {
    defaults.renderer = "qwen3.5";
    defaults.parser = "qwen3.5";
    defaults.capabilities = ["completion", "tools", "thinking"];
  } else if (combined.includes("qwen3.8") || combined.includes("qwen3_8")) {
    defaults.renderer = "qwen3.8";
    defaults.parser = "qwen3.5";
    defaults.capabilities = ["completion", "tools", "thinking"];
  } else if (combined.includes("qwen3")) {
    defaults.capabilities = ["completion", "tools", "thinking"];
  } else if (combined.includes("gemma4")) {
    defaults.renderer = "gemma4";
    defaults.capabilities = ["completion"];
  } else if (combined.includes("llama") || combined.includes("mistral")) {
    defaults.capabilities = ["completion", "tools"];
  }

  if (combined.includes("vision") || combined.includes("vl") || combined.includes("conditionalgeneration")) {
    if (!defaults.capabilities.includes("vision")) {
      defaults.capabilities.push("vision");
    }
  }
  return defaults;
}

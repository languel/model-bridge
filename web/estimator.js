// Analytical VRAM / Apple Silicon Unified Memory & Bandwidth Estimator

export const HARDWARE_PRESETS = [
  { id: "m1-16", label: "Apple M1 (16 GB)", ramGb: 16, usableRatio: 0.72, bwGbS: 68 },
  { id: "m2-24", label: "Apple M2 (24 GB)", ramGb: 24, usableRatio: 0.75, bwGbS: 100 },
  { id: "m3-pro-36", label: "Apple M3 Pro (36 GB)", ramGb: 36, usableRatio: 0.78, bwGbS: 150 },
  { id: "m3-max-64", label: "Apple M3 Max (64 GB)", ramGb: 64, usableRatio: 0.80, bwGbS: 400 },
  { id: "m4-pro-48", label: "Apple M4 Pro (48 GB)", ramGb: 48, usableRatio: 0.80, bwGbS: 273 },
  { id: "m4-max-128", label: "Apple M4 Max (128 GB)", ramGb: 128, usableRatio: 0.82, bwGbS: 546 },
  { id: "m3-ultra-256", label: "Apple M3 Ultra (256 GB)", ramGb: 256, usableRatio: 0.85, bwGbS: 819 },
  { id: "m5-max-128", label: "Apple M5 Max (128 GB)", ramGb: 128, usableRatio: 0.82, bwGbS: 614 },
  { id: "rtx-4090", label: "NVIDIA RTX 4090 (24 GB)", ramGb: 24, usableRatio: 0.92, bwGbS: 1008 },
  { id: "rtx-5090", label: "NVIDIA RTX 5090 (32 GB)", ramGb: 32, usableRatio: 0.92, bwGbS: 1792 },
  { id: "h100-80", label: "NVIDIA H100 (80 GB)", ramGb: 80, usableRatio: 0.94, bwGbS: 3350 },
];

export const QUANT_PRESETS = [
  { id: "FP16", label: "FP16 / BF16 (16 bpw)", bpw: 16.0 },
  { id: "Q8_0", label: "Q8_0 / MLX 8-bit (~8.5 bpw)", bpw: 8.5 },
  { id: "Q6_K", label: "Q6_K (~6.56 bpw)", bpw: 6.56 },
  { id: "Q5_K_M", label: "Q5_K_M (~5.68 bpw)", bpw: 5.68 },
  { id: "Q4_K_M", label: "Q4_K_M (~4.85 bpw)", bpw: 4.85 },
  { id: "MLX_4BIT", label: "MLX 4-bit g64 (~4.5 bpw)", bpw: 4.5 },
  { id: "IQ4_XS", label: "IQ4_XS (~4.25 bpw)", bpw: 4.25 },
  { id: "Q3_K_M", label: "Q3_K_M (~3.91 bpw)", bpw: 3.91 },
  { id: "Q2_K", label: "Q2_K (~3.0 bpw)", bpw: 3.0 },
];

export const KV_DTYPE_BYTES = {
  fp16: 2.0,
  q8_0: 1.06,
  q4_0: 0.56,
};

/**
 * Estimates memory footprint and theoretical decode speed.
 * @param {object} opts
 * @param {number} opts.paramsB - Parameter count in billions
 * @param {number} opts.bpw - Bits per weight
 * @param {number} opts.contextLength - Context window in tokens
 * @param {number} opts.layers - Number of transformer layers
 * @param {number} opts.kvHeads - Number of KV attention heads (GQA)
 * @param {number} opts.headDim - Dimension per attention head
 * @param {string} opts.kvDtype - 'fp16' | 'q8_0' | 'q4_0'
 * @param {object} opts.hardware - Hardware preset
 */
export function estimateMemory({
  paramsB = 8,
  bpw = 4.85,
  contextLength = 32768,
  layers = 32,
  kvHeads = 8,
  headDim = 128,
  kvDtype = "fp16",
  hardware = HARDWARE_PRESETS[4],
}) {
  const weightsGb = (paramsB * 1e9 * (bpw / 8)) / 1024 ** 3;
  const kvBytesPerElem = KV_DTYPE_BYTES[kvDtype] || 2.0;
  // 2 (K and V) * layers * kvHeads * headDim * contextLength * bytes
  const kvCacheGb = (2 * layers * kvHeads * headDim * contextLength * kvBytesPerElem) / 1024 ** 3;
  // Compute / activation / runtime overhead (~0.45 GB base + 3% of weights)
  const overheadGb = 0.45 + weightsGb * 0.03;
  const totalGb = weightsGb + kvCacheGb + overheadGb;

  const usableGb = hardware.ramGb * hardware.usableRatio;
  const utilization = totalGb / usableGb;

  let status = "fits";
  let statusLabel = "Fits Comfortably";
  if (utilization > 1.0) {
    status = "oom";
    statusLabel = "Exceeds Memory";
  } else if (utilization > 0.85) {
    status = "tight";
    statusLabel = "Tight Fit";
  }

  // Memory-bandwidth bound decode estimate (~75% sustained efficiency)
  const activeReadGb = weightsGb + kvCacheGb * 0.25;
  const estTokPerSec = activeReadGb > 0 ? (hardware.bwGbS * 0.75) / activeReadGb : 0;

  return {
    weightsGb,
    kvCacheGb,
    overheadGb,
    totalGb,
    usableGb,
    ramGb: hardware.ramGb,
    utilization,
    status,
    statusLabel,
    estTokPerSec,
  };
}

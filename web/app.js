import {
  parseGGUF,
  parseSafetensors,
  parseConfigJson,
  inferOllamaDefaults,
  formatBytes,
  formatParams,
} from "./parsers.js";
import {
  HARDWARE_PRESETS,
  QUANT_PRESETS,
  estimateMemory,
} from "./estimator.js";

let currentModel = null;

// --- Helpers ---
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => t.classList.remove("show"), 2000);
}

async function copyText(text, label = "Copied to clipboard!") {
  try {
    await navigator.clipboard.writeText(text);
    showToast(label);
  } catch {
    showToast("Could not copy automatically.");
  }
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// --- Navigation Tabs ---
function switchTab(tabId) {
  $$(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tabId}`));
}

$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// =====================================================================
// TAB 1: LOCAL FILE INSPECTOR
// =====================================================================
const dropzone = $("#dropzone");
const inputFile = $("#input-file");
const inputFolder = $("#input-folder");

$("#btn-pick-file").addEventListener("click", (e) => {
  e.stopPropagation();
  inputFile.click();
});
$("#btn-pick-folder").addEventListener("click", (e) => {
  e.stopPropagation();
  inputFolder.click();
});
dropzone.addEventListener("click", () => inputFile.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const files = Array.from(e.dataTransfer.files || []);
  if (files.length) await handleFiles(files);
});

inputFile.addEventListener("change", async () => {
  if (inputFile.files.length) await handleFiles(Array.from(inputFile.files));
});
inputFolder.addEventListener("change", async () => {
  if (inputFolder.files.length) await handleFiles(Array.from(inputFolder.files));
});

async function handleFiles(files) {
  const errBox = $("#inspect-error");
  errBox.classList.add("hidden");

  try {
    const configFile = files.find((f) => f.name.toLowerCase() === "config.json");
    const ggufFile = files.find((f) => f.name.toLowerCase().endsWith(".gguf"));
    const stFile = files.find((f) => f.name.toLowerCase().endsWith(".safetensors"));

    let parsed;
    if (files.length === 1 && ggufFile) {
      parsed = await parseGGUF(ggufFile);
    } else if (files.length === 1 && stFile) {
      parsed = await parseSafetensors(stFile);
    } else if (configFile) {
      parsed = await parseConfigJson(configFile, files);
      // If a single safetensors shard is also present, enrich with its tensor preview
      if (stFile) {
        try {
          const stInfo = await parseSafetensors(stFile);
          parsed.tensors = stInfo.tensors;
          parsed.dtypeCounts = stInfo.dtypeCounts;
          parsed.nTensors = stInfo.nTensors;
          if (!parsed.totalParams && files.filter((f) => f.name.endsWith(".safetensors")).length === 1) {
            parsed.totalParams = stInfo.totalParams;
          }
        } catch { /* ignore shard parse error */ }
      }
    } else if (ggufFile) {
      parsed = await parseGGUF(ggufFile);
    } else if (stFile) {
      parsed = await parseSafetensors(stFile);
    } else {
      throw new Error("Unsupported file(s). Drop a .gguf, .safetensors, config.json, or a model folder.");
    }

    currentModel = parsed;
    renderInspection(parsed);
  } catch (err) {
    errBox.textContent = `Inspection error: ${err.message}`;
    errBox.classList.remove("hidden");
  }
}

function renderInspection(m) {
  $("#inspect-results").classList.remove("hidden");
  $("#insp-title").textContent = m.name || m.filename || "Model";
  $("#insp-subtitle").textContent = `${m.format}${m.version ? ` v${m.version}` : ""} • ${m.filename || ""}${m.shardCount ? ` (${m.shardCount} weight shard${m.shardCount > 1 ? "s" : ""})` : ""}`;

  const metrics = [
    ["Format", m.format],
    ["Architecture", m.architecture || "unknown"],
    ["Quant / Dtype", m.quantization || "unknown"],
    ["File Size", formatBytes(m.fileSizeBytes)],
    ["Parameters", m.totalParams ? formatParams(m.totalParams) : "—"],
    ["Context Length", m.contextLength ? Number(m.contextLength).toLocaleString() : "—"],
    ["Layers", m.blockCount ?? "—"],
    ["Heads / KV Heads", m.headCount ? `${m.headCount} / ${m.headCountKv ?? m.headCount}` : "—"],
  ];

  $("#insp-metrics").innerHTML = metrics
    .map(([k, v]) => `<div class="metric"><div class="metric-label">${esc(k)}</div><div class="metric-value">${esc(v)}</div></div>`)
    .join("");

  const dtypesEl = $("#insp-dtypes");
  const dtypes = Object.entries(m.dtypeCounts || {});
  dtypesEl.innerHTML = dtypes.length
    ? dtypes.map(([dt, cnt]) => `<span class="badge badge-blue">${esc(dt)}: ${cnt} tensors</span>`).join("")
    : "";

  renderKvTable(m.metadata || {}, "");
  renderTensorTable(m.tensors || [], m.nTensors || 0, "");

  const tmplCard = $("#chat-template-card");
  if (m.chatTemplate) {
    tmplCard.classList.remove("hidden");
    $("#chat-template-pre").textContent = m.chatTemplate;
  } else {
    tmplCard.classList.add("hidden");
  }
}

function renderKvTable(meta, filter) {
  const q = filter.toLowerCase();
  const entries = Object.entries(meta).filter(
    ([k, v]) => !q || k.toLowerCase().includes(q) || String(v).toLowerCase().includes(q)
  );
  $("#kv-tbody").innerHTML = entries.length
    ? entries
        .slice(0, 400)
        .map(([k, v]) => {
          const valStr = typeof v === "object" && v !== null ? JSON.stringify(v) : String(v);
          const display = valStr.length > 220 ? `${valStr.slice(0, 220)}…` : valStr;
          return `<tr><td class="mono">${esc(k)}</td><td class="mono">${esc(display)}</td></tr>`;
        })
        .join("")
    : `<tr><td colspan="2" style="color:var(--muted)">No metadata keys match.</td></tr>`;
}

function renderTensorTable(tensors, totalCount, filter) {
  $("#tensor-title").textContent = totalCount
    ? `Tensors (${Math.min(tensors.length, totalCount)} shown of ${totalCount})`
    : "Tensors";
  const q = filter.toLowerCase();
  const list = tensors.filter((t) => !q || t.name.toLowerCase().includes(q) || t.dtype.toLowerCase().includes(q));
  $("#tensor-tbody").innerHTML = list.length
    ? list
        .map(
          (t) =>
            `<tr><td class="mono">${esc(t.name)}</td><td class="mono"><span class="badge badge-blue">${esc(t.dtype)}</span></td><td class="mono">[${t.shape.join(", ")}]</td></tr>`
        )
        .join("")
    : `<tr><td colspan="3" style="color:var(--muted)">No tensor headers available for this file type.</td></tr>`;
}

$("#kv-filter").addEventListener("input", (e) => {
  if (currentModel) renderKvTable(currentModel.metadata || {}, e.target.value);
});
$("#tensor-filter").addEventListener("input", (e) => {
  if (currentModel) renderTensorTable(currentModel.tensors || [], currentModel.nTensors || 0, e.target.value);
});

$("#btn-copy-template").addEventListener("click", () => {
  if (currentModel?.chatTemplate) copyText(currentModel.chatTemplate, "Chat template copied!");
});

$("#btn-export-json").addEventListener("click", () => {
  if (!currentModel) return;
  const exportObj = { ...currentModel, tensors: currentModel.tensors?.slice(0, 50) };
  downloadText(`${(currentModel.name || "model").replace(/[^a-zA-Z0-9._-]/g, "_")}_inspect.json`, JSON.stringify(exportObj, null, 2));
});

$("#btn-send-estimator").addEventListener("click", () => {
  if (!currentModel) return;
  populateEstimatorFromModel(currentModel);
  switchTab("estimator");
});

$("#btn-send-builder").addEventListener("click", () => {
  if (!currentModel) return;
  populateBuilderFromModel(currentModel);
  switchTab("builder");
});

// =====================================================================
// TAB 2: HUGGING FACE HUB EXPLORER
// =====================================================================
async function searchHF(query) {
  const errEl = $("#hf-error");
  const grid = $("#hf-results");
  errEl.classList.add("hidden");
  grid.innerHTML = `<p style="color:var(--muted)">Searching Hugging Face Hub…</p>`;

  try {
    const url = `https://huggingface.co/api/models?search=${encodeURIComponent(query)}&sort=downloads&direction=-1&limit=18`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const models = await res.json();

    if (!models.length) {
      grid.innerHTML = `<p style="color:var(--muted)">No repositories found for "${esc(query)}".</p>`;
      return;
    }

    grid.innerHTML = models
      .map((m) => {
        const tags = (m.tags || []).filter((t) => ["gguf", "mlx", "safetensors", "onnx", "transformers"].includes(t));
        return `
          <div class="hf-item">
            <div>
              <div class="hf-id"><a href="https://huggingface.co/${esc(m.id)}" target="_blank" rel="noopener">${esc(m.id)}</a></div>
              <div class="hf-meta" style="margin-top:6px;">
                <span>↓ ${(m.downloads || 0).toLocaleString()}</span>
                <span>♥ ${(m.likes || 0).toLocaleString()}</span>
                ${tags.map((t) => `<span class="badge badge-blue">${esc(t)}</span>`).join("")}
              </div>
            </div>
            <div class="btn-row" style="justify-content:flex-start;">
              <button class="btn btn-sm btn-primary hf-inspect-btn" data-repo="${esc(m.id)}">Inspect Repo</button>
              <button class="btn btn-sm hf-use-btn" data-repo="${esc(m.id)}">Use in CLI</button>
            </div>
          </div>`;
      })
      .join("");

    grid.querySelectorAll(".hf-inspect-btn").forEach((b) =>
      b.addEventListener("click", () => inspectHFRepo(b.dataset.repo))
    );
    grid.querySelectorAll(".hf-use-btn").forEach((b) =>
      b.addEventListener("click", () => {
        $("#mf-source").value = `hf:${b.dataset.repo}`;
        $("#mf-name").value = b.dataset.repo.split("/").pop().toLowerCase().replace(/[^a-z0-9._-]/g, "-");
        updateBuilder();
        switchTab("builder");
      })
    );
  } catch (err) {
    errEl.textContent = `Hugging Face API error: ${err.message}`;
    errEl.classList.remove("hidden");
    grid.innerHTML = "";
  }
}

async function inspectHFRepo(rawId) {
  const repoId = rawId.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/^hf:/, "").replace(/\/$/, "");
  const errEl = $("#hf-error");
  const detailEl = $("#hf-repo-detail");
  errEl.classList.add("hidden");
  detailEl.classList.remove("hidden");
  detailEl.innerHTML = `<p style="color:var(--muted)">Loading <code>${esc(repoId)}</code>…</p>`;

  try {
    const [metaRes, treeRes] = await Promise.all([
      fetch(`https://huggingface.co/api/models/${encodeURIComponent(repoId).replace(/%2F/g, "/")}`),
      fetch(`https://huggingface.co/api/models/${encodeURIComponent(repoId).replace(/%2F/g, "/")}/tree/main`),
    ]);
    if (!metaRes.ok) throw new Error(`Repo "${repoId}" not found (HTTP ${metaRes.status}).`);
    const meta = await metaRes.json();
    const tree = treeRes.ok ? await treeRes.json() : [];

    // Try fetching config.json if present in tree
    let cfg = null;
    if (tree.some((f) => f.path === "config.json")) {
      try {
        const cfgRes = await fetch(`https://huggingface.co/${repoId}/resolve/main/config.json`);
        if (cfgRes.ok) cfg = await cfgRes.json();
      } catch { /* CORS or gated */ }
    }

    const weightFiles = tree.filter((f) => /\.(gguf|safetensors|onnx|bin)$/i.test(f.path));
    const totalSize = weightFiles.reduce((s, f) => s + (f.size || 0), 0);
    const ggufMeta = meta.gguf || null;
    const textCfg = cfg?.text_config || cfg || {};

    const inferredModel = {
      format: meta.tags?.includes("mlx") ? "MLX (HF Hub)" : meta.tags?.includes("gguf") ? "GGUF (HF Hub)" : "HuggingFace Hub",
      name: repoId.split("/").pop(),
      sourcePath: `hf:${repoId}`,
      architecture: (cfg?.architectures && cfg.architectures[0]) || ggufMeta?.architecture || cfg?.model_type || "unknown",
      modelType: cfg?.model_type || ggufMeta?.architecture || "",
      totalParams: meta.safetensors?.total || ggufMeta?.total || null,
      contextLength: textCfg.max_position_embeddings || ggufMeta?.context_length || null,
      blockCount: textCfg.num_hidden_layers || null,
      embeddingLength: textCfg.hidden_size || null,
      headCount: textCfg.num_attention_heads || null,
      headCountKv: textCfg.num_key_value_heads || textCfg.num_attention_heads || null,
      fileSizeBytes: totalSize,
    };

    detailEl.innerHTML = `
      <div class="card-header">
        <div>
          <h3 class="card-title"><a href="https://huggingface.co/${esc(repoId)}" target="_blank" rel="noopener">${esc(repoId)}</a></h3>
          <p class="subtitle">${esc(inferredModel.format)} • Architecture: <code>${esc(inferredModel.architecture)}</code></p>
        </div>
        <div class="btn-row">
          <button id="hf-detail-est" class="btn btn-sm btn-primary">Estimate Memory</button>
          <button id="hf-detail-cli" class="btn btn-sm">Build CLI / Modelfile</button>
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric"><div class="metric-label">Parameters</div><div class="metric-value">${formatParams(inferredModel.totalParams)}</div></div>
        <div class="metric"><div class="metric-label">Weight Files</div><div class="metric-value">${weightFiles.length}</div></div>
        <div class="metric"><div class="metric-label">Total Weight Size</div><div class="metric-value">${formatBytes(totalSize)}</div></div>
        <div class="metric"><div class="metric-label">Context Length</div><div class="metric-value">${inferredModel.contextLength ? Number(inferredModel.contextLength).toLocaleString() : "—"}</div></div>
      </div>
      ${weightFiles.length ? `
        <div class="table-wrap" style="max-height:220px;">
          <table>
            <thead><tr><th>Weight File</th><th>Size</th><th>Action</th></tr></thead>
            <tbody>
              ${weightFiles.slice(0, 40).map((f) => `
                <tr>
                  <td class="mono">${esc(f.path)}</td>
                  <td class="mono">${formatBytes(f.size)}</td>
                  <td><button class="btn btn-sm hf-file-cli" data-spec="hf:${esc(repoId)}#${esc(f.path)}">Use File</button></td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>` : ""}
    `;

    $("#hf-detail-est").addEventListener("click", () => {
      populateEstimatorFromModel(inferredModel);
      switchTab("estimator");
    });
    $("#hf-detail-cli").addEventListener("click", () => {
      populateBuilderFromModel(inferredModel);
      switchTab("builder");
    });
    detailEl.querySelectorAll(".hf-file-cli").forEach((b) =>
      b.addEventListener("click", () => {
        $("#mf-source").value = b.dataset.spec;
        updateBuilder();
        switchTab("builder");
      })
    );
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
    detailEl.classList.add("hidden");
  }
}

$("#btn-hf-search").addEventListener("click", () => {
  const q = $("#hf-query").value.trim();
  if (q) searchHF(q);
});
$("#btn-hf-inspect").addEventListener("click", () => {
  const q = $("#hf-query").value.trim();
  if (q) inspectHFRepo(q);
});
$("#hf-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const q = e.target.value.trim();
    if (q.includes("/") && !q.includes(" ")) inspectHFRepo(q);
    else if (q) searchHF(q);
  }
});
$$(".hf-preset").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("#hf-query").value = chip.dataset.q;
    searchHF(chip.dataset.q);
  });
});

// =====================================================================
// TAB 3: MEMORY & SPEED ESTIMATOR
// =====================================================================
function initEstimator() {
  const hwSel = $("#est-hw");
  hwSel.innerHTML = HARDWARE_PRESETS.map((h, i) => `<option value="${i}" ${h.id === "m4-pro-48" ? "selected" : ""}>${esc(h.label)} — ${h.bwGbS} GB/s</option>`).join("");

  const qSel = $("#est-quant");
  qSel.innerHTML = QUANT_PRESETS.map((q, i) => `<option value="${i}" ${q.id === "Q4_K_M" ? "selected" : ""}>${esc(q.label)}</option>`).join("");

  ["#est-hw", "#est-quant", "#est-params", "#est-ctx", "#est-layers", "#est-kv-heads", "#est-head-dim", "#est-kv-dtype"].forEach((sel) => {
    $(sel).addEventListener("input", updateEstimator);
    $(sel).addEventListener("change", updateEstimator);
  });
  updateEstimator();
}

function populateEstimatorFromModel(m) {
  if (m.totalParams) {
    $("#est-params").value = (m.totalParams / 1e9).toFixed(2);
  } else if (m.fileSizeBytes > 1e8) {
    // Rough fallback if totalParams unknown
    $("#est-params").value = Math.max(0.5, m.fileSizeBytes / 0.6e9).toFixed(1);
  }
  if (m.blockCount) $("#est-layers").value = m.blockCount;
  if (m.headCountKv) $("#est-kv-heads").value = m.headCountKv;
  if (m.embeddingLength && m.headCount) {
    $("#est-head-dim").value = Math.round(m.embeddingLength / m.headCount);
  }
  if (m.contextLength) {
    const opts = Array.from($("#est-ctx").options).map((o) => Number(o.value));
    const closest = opts.reduce((a, b) => (Math.abs(b - m.contextLength) < Math.abs(a - m.contextLength) ? b : a));
    $("#est-ctx").value = String(closest);
  }
  updateEstimator();
}

function updateEstimator() {
  const hardware = HARDWARE_PRESETS[Number($("#est-hw").value)] || HARDWARE_PRESETS[4];
  const quant = QUANT_PRESETS[Number($("#est-quant").value)] || QUANT_PRESETS[4];
  const res = estimateMemory({
    paramsB: Math.max(0.1, Number($("#est-params").value) || 8),
    bpw: quant.bpw,
    contextLength: Number($("#est-ctx").value) || 32768,
    layers: Math.max(1, Number($("#est-layers").value) || 32),
    kvHeads: Math.max(1, Number($("#est-kv-heads").value) || 8),
    headDim: Math.max(16, Number($("#est-head-dim").value) || 128),
    kvDtype: $("#est-kv-dtype").value,
    hardware,
  });

  const badge = $("#est-status-badge");
  badge.textContent = res.statusLabel;
  badge.className = `badge ${res.status === "fits" ? "badge-green" : res.status === "tight" ? "badge-amber" : "badge-red"}`;

  $("#est-metrics").innerHTML = [
    ["Total Needed", `${res.totalGb.toFixed(2)} GB`],
    ["Usable Budget", `${res.usableGb.toFixed(1)} / ${res.ramGb} GB`],
    ["Weights", `${res.weightsGb.toFixed(2)} GB`],
    ["KV Cache", `${res.kvCacheGb.toFixed(2)} GB`],
    ["Est. Decode Speed", res.status === "oom" ? "Swapping / OOM" : `~${res.estTokPerSec.toFixed(1)} tok/s`],
    ["Memory Bandwidth", `${hardware.bwGbS} GB/s`],
  ]
    .map(([k, v]) => `<div class="metric"><div class="metric-label">${esc(k)}</div><div class="metric-value">${esc(v)}</div></div>`)
    .join("");

  const denom = Math.max(res.usableGb, res.totalGb);
  $("#bar-weights").style.width = `${(res.weightsGb / denom) * 100}%`;
  $("#bar-kv").style.width = `${(res.kvCacheGb / denom) * 100}%`;
  $("#bar-overhead").style.width = `${(res.overheadGb / denom) * 100}%`;

  $("#est-note").textContent =
    res.status === "oom"
      ? `Exceeds usable memory budget (${res.usableGb.toFixed(1)} GB) by ${(res.totalGb - res.usableGb).toFixed(1)} GB. Try a lower quantization (e.g. Q4_K_M / MLX 4-bit), Q8_0 KV cache, or shorter context.`
      : `Uses ${(res.utilization * 100).toFixed(0)}% of usable memory (${res.usableGb.toFixed(1)} GB). Decode speed assumes ~75% sustained memory-bandwidth efficiency.`;
}

// =====================================================================
// TAB 4: MODELFILE & CLI BUILDER
// =====================================================================
function populateBuilderFromModel(m) {
  const defaults = inferOllamaDefaults(m);
  $("#mf-source").value = m.sourcePath || m.filename || "./model.gguf";
  $("#mf-name").value = (m.name || "my-model").toLowerCase().replace(/[^a-z0-9._:-]/g, "-");
  $("#mf-ctx").value = defaults.num_ctx;
  $("#mf-temp").value = defaults.temperature;
  $("#mf-renderer").value = defaults.renderer;
  $("#mf-parser").value = defaults.parser;
  $$(".mf-cap").forEach((cb) => {
    cb.checked = defaults.capabilities.includes(cb.value);
  });
  updateBuilder();
}

function updateBuilder() {
  const source = $("#mf-source").value.trim() || "./model.gguf";
  const name = $("#mf-name").value.trim() || "my-model:latest";
  const target = $("#mf-target").value;
  const quant = $("#mf-quant").value;
  const ctx = $("#mf-ctx").value.trim();
  const temp = $("#mf-temp").value.trim();
  const renderer = $("#mf-renderer").value.trim();
  const parser = $("#mf-parser").value.trim();
  const system = $("#mf-system").value.trim();
  const caps = $$(".mf-cap").filter((c) => c.checked).map((c) => c.value);

  const mfLines = [`FROM ${source}`];
  if (renderer) mfLines.push(`RENDERER ${renderer}`);
  if (parser) mfLines.push(`PARSER ${parser}`);
  for (const cap of caps) mfLines.push(`CAPABILITY ${cap}`);
  if (ctx) mfLines.push(`PARAMETER num_ctx ${ctx}`);
  if (temp) mfLines.push(`PARAMETER temperature ${temp}`);
  if (system) mfLines.push(`SYSTEM """${system}"""`);
  $("#out-modelfile").textContent = mfLines.join("\n") + "\n";

  // Build CLI command
  const args = ["./bin/model-convert", "convert", `"${source}"`, "--to", target];
  if (["ollama", "ollama-gguf", "ollama-mlx"].includes(target)) {
    args.push("--name", `"${name}"`);
    if (ctx) args.push("--ctx", ctx);
    if (temp) args.push("--temp", temp);
    if (renderer) args.push("--renderer", renderer);
    if (parser) args.push("--parser", parser);
    if (caps.length) args.push("--capabilities", caps.join(","));
  }
  if (["mlx", "ollama-mlx"].includes(target)) {
    const bits = ["4", "8", "16"].includes(quant) ? quant : "4";
    args.push("--bits", bits);
  } else if (["gguf", "ollama-gguf"].includes(target)) {
    const q = ["4", "8", "16"].includes(quant) ? "Q4_K_M" : quant;
    args.push("--quant", q);
  }

  const cliText = [
    "# Using Model Bridge CLI:",
    args.join(" "),
    "",
    "# Or register the downloaded Modelfile directly with Ollama:",
    `ollama create "${name}" -f Modelfile`,
  ].join("\n");

  $("#out-cli").textContent = cliText;
}

["#mf-source", "#mf-name", "#mf-target", "#mf-quant", "#mf-ctx", "#mf-temp", "#mf-renderer", "#mf-parser", "#mf-system"].forEach((sel) => {
  $(sel).addEventListener("input", updateBuilder);
  $(sel).addEventListener("change", updateBuilder);
});
$$(".mf-cap").forEach((cb) => cb.addEventListener("change", updateBuilder));

$("#btn-copy-mf").addEventListener("click", () => copyText($("#out-modelfile").textContent, "Modelfile copied!"));
$("#btn-dl-mf").addEventListener("click", () => downloadText("Modelfile", $("#out-modelfile").textContent));
$("#btn-copy-cli").addEventListener("click", () => copyText($("#out-cli").textContent, "CLI command copied!"));

// =====================================================================
// TAB 5: LOCAL OLLAMA BRIDGE
// =====================================================================
async function connectOllama() {
  const baseUrl = $("#ollama-url").value.trim().replace(/\/$/, "");
  const errEl = $("#ollama-error");
  const listEl = $("#ollama-models");
  const detailEl = $("#ollama-detail");
  errEl.classList.add("hidden");
  detailEl.classList.add("hidden");
  listEl.innerHTML = `<p style="color:var(--muted)">Connecting to ${esc(baseUrl)}…</p>`;

  try {
    const res = await fetch(`${baseUrl}/api/tags`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const models = data.models || [];
    if (!models.length) {
      listEl.innerHTML = `<p style="color:var(--muted)">Connected, but no Ollama models found.</p>`;
      return;
    }

    listEl.innerHTML = models
      .map(
        (m) => `
      <div class="hf-item">
        <div>
          <div class="hf-id">${esc(m.name)}</div>
          <div class="hf-meta" style="margin-top:6px;">
            <span>${formatBytes(m.size)}</span>
            ${m.details?.family ? `<span class="badge badge-blue">${esc(m.details.family)}</span>` : ""}
            ${m.details?.parameter_size ? `<span class="badge badge-green">${esc(m.details.parameter_size)}</span>` : ""}
            ${m.details?.quantization_level ? `<span class="badge badge-amber">${esc(m.details.quantization_level)}</span>` : ""}
          </div>
        </div>
        <div class="btn-row" style="justify-content:flex-start;">
          <button class="btn btn-sm btn-primary ol-show-btn" data-name="${esc(m.name)}">Inspect Modelfile</button>
        </div>
      </div>`
      )
      .join("");

    listEl.querySelectorAll(".ol-show-btn").forEach((b) =>
      b.addEventListener("click", () => inspectLocalOllama(baseUrl, b.dataset.name))
    );
  } catch (err) {
    errEl.innerHTML = `Could not reach <code>${esc(baseUrl)}</code> (${esc(err.message)}). If opening from GitHub Pages, start Ollama with <code>OLLAMA_ORIGINS="*" ollama serve</code>.`;
    errEl.classList.remove("hidden");
    listEl.innerHTML = "";
  }
}

async function inspectLocalOllama(baseUrl, name) {
  const detailEl = $("#ollama-detail");
  detailEl.classList.remove("hidden");
  detailEl.innerHTML = `<p style="color:var(--muted)">Fetching <code>${esc(name)}</code>…</p>`;

  try {
    const res = await fetch(`${baseUrl}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    detailEl.innerHTML = `
      <div class="card-header">
        <h3 class="card-title">${esc(name)}</h3>
        <button id="btn-copy-ol-mf" class="btn btn-sm">Copy Modelfile</button>
      </div>
      <pre class="code-box">${esc(data.modelfile || "(No Modelfile returned)")}</pre>
    `;
    $("#btn-copy-ol-mf").addEventListener("click", () => copyText(data.modelfile || "", "Modelfile copied!"));
  } catch (err) {
    detailEl.innerHTML = `<p style="color:var(--red)">Failed to inspect ${esc(name)}: ${esc(err.message)}</p>`;
  }
}

$("#btn-ollama-connect").addEventListener("click", connectOllama);

// --- Boot ---
initEstimator();
updateBuilder();
searchHF("mlx-community");

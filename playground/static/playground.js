/**
 * Laya Playground — interactive layer.
 *
 * Vanilla JS, no dependencies, no persistence, no auto-run.
 * All user-provided text is rendered via textContent (never innerHTML).
 * API contract:
 *   GET  /api/health  → upstream /health JSON or gateway error
 *   POST /api/query   → upstream /system_one JSON or gateway error
 */
"use strict";

// ── helpers ────────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "style") node.setAttribute("style", v);
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    if (typeof child === "string") node.appendChild(document.createTextNode(child));
    else if (child instanceof Node) node.appendChild(child);
  }
  return node;
}

function fmtProb(p) {
  return Number.isFinite(p) ? p.toFixed(4) : "—";
}

function fmtMs(ms) {
  if (ms < 1000) return Math.round(ms) + " ms";
  return (ms / 1000).toFixed(2) + " s";
}

function pct(v) {
  return (v * 100).toFixed(1) + "%";
}

// ── state ──────────────────────────────────────────────────────────────────

const state = {
  format: "text", // "text" | "json"
  text: "",
  json: "",
};

function getStateValue() {
  if (state.format === "text") return state.text;
  const trimmed = state.json.trim();
  if (!trimmed) return "";
  try {
    return JSON.parse(trimmed);
  } catch (e) {
    return null; // signals parse error
  }
}

// ── health strip ───────────────────────────────────────────────────────────

async function refreshHealth() {
  const pills = { service: $("#connState"), model: $("#modelState"), provider: $("#providerState") };
  try {
    const r = await fetch("/api/health");
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      pills.service.textContent = "service: unreachable (" + r.status + ")";
      pills.service.dataset.state = "bad";
      pills.model.textContent = "model: unknown";
      pills.model.dataset.state = "bad";
      pills.provider.textContent = "provider: unknown";
      pills.provider.dataset.state = "bad";
      return;
    }
    const loaded = body.model_loaded === true;
    const providers = Array.isArray(body.providers) ? body.providers : [];
    const primary = providers[0] || null;

    pills.service.textContent = "service: reachable";
    pills.service.dataset.state = "ok";

    pills.model.textContent = loaded ? "model: loaded" : "model: unloaded (will reload on first query)";
    pills.model.dataset.state = loaded ? "ok" : "warn";

    if (!loaded) {
      pills.provider.textContent = "provider: not yet verified (model unloaded)";
      pills.provider.dataset.state = "warn";
    } else if (primary && primary.startsWith("CUDA")) {
      pills.provider.textContent = "provider: " + primary;
      pills.provider.dataset.state = "ok";
    } else if (primary && primary.startsWith("CPU")) {
      pills.provider.textContent = "provider: CPU ONLY (expected CUDA)";
      pills.provider.dataset.state = "bad";
    } else {
      pills.provider.textContent = "provider: unknown";
      pills.provider.dataset.state = "warn";
    }
  } catch (e) {
    pills.service.textContent = "service: unreachable";
    pills.service.dataset.state = "bad";
    pills.model.textContent = "model: unknown";
    pills.model.dataset.state = "bad";
    pills.provider.textContent = "provider: unknown";
    pills.provider.dataset.state = "bad";
  }
}

// ── tabs ───────────────────────────────────────────────────────────────────

function initTabs() {
  const tabs = $$(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const type = tab.dataset.type;
      tabs.forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      $$(".panel").forEach((p) => p.classList.add("hidden"));
      const panel = $("#panel-" + type);
      if (panel) panel.classList.remove("hidden");
      updateRunHint();
    });
  });
}

function activeTabType() {
  const active = $(".tab.active");
  return active ? active.dataset.type : "choice";
}

function updateRunHint() {
  const hint = $("#runHint");
  if (!hint) return;
  const t = activeTabType();
  const labels = {
    choice: "One choice question. Nothing runs until you press Run.",
    score: "One score question. Nothing runs until you press Run.",
    noul: "One noul question. Nothing runs until you press Run.",
    batch: "Several questions in one request. Nothing runs until you press Run.",
    raw: "Raw /system_one body, posted verbatim. Nothing runs until you press Run.",
  };
  hint.textContent = labels[t] || "Nothing runs until you press Run.";
}

// ── state editor ───────────────────────────────────────────────────────────

function initStateEditor() {
  const txtBtn = $("#stateModeText");
  const jsBtn = $("#stateModeJson");
  const txtArea = $("#stateText");
  const jsArea = $("#stateJson");
  const note = $("#stateModeNote");
  const err = $("#stateJsonError");

  function setMode(mode) {
    state.format = mode;
    txtBtn.classList.toggle("active", mode === "text");
    jsBtn.classList.toggle("active", mode === "json");
    txtBtn.setAttribute("aria-pressed", mode === "text");
    jsBtn.setAttribute("aria-pressed", mode === "json");
    txtArea.classList.toggle("hidden", mode !== "text");
    jsArea.classList.toggle("hidden", mode !== "json");
    err.classList.add("hidden");
    if (mode === "text") {
      note.textContent =
        "Sent to the model as a JSON string exactly as typed. Laya sees the state as input content — not as an instruction to generate text.";
    } else {
      note.textContent =
        "Parsed as JSON locally and sent as a JSON value. Laya sees the state as input content — not as an instruction to generate text.";
    }
  }

  txtBtn.addEventListener("click", () => setMode("text"));
  jsBtn.addEventListener("click", () => setMode("json"));
  txtArea.addEventListener("input", () => (state.text = txtArea.value));
  jsArea.addEventListener("input", () => {
    state.json = jsArea.value;
    err.classList.add("hidden");
  });
}

function validateState() {
  if (state.format !== "json") return true;
  const trimmed = state.json.trim();
  if (!trimmed) return true;
  try {
    JSON.parse(trimmed);
    return true;
  } catch (e) {
    const err = $("#stateJsonError");
    err.textContent = "JSON parse error: " + e.message;
    err.classList.remove("hidden");
    return false;
  }
}

// ── option / level rows ────────────────────────────────────────────────────

function addChoiceRow(key, desc, container) {
  container = container || $("#choiceRows");
  const idx = container.children.length + 1;
  const row = el("div", { class: "opt-row" });
  const idxSpan = el("span", { class: "idx" }, String(idx));
  const keyInput = el("input", {
    class: "field",
    type: "text",
    placeholder: "key (e.g. billing)",
    value: key || "",
    "aria-label": "option key " + idx,
  });
  const descInput = el("input", {
    class: "field",
    type: "text",
    placeholder: "optional description",
    value: desc || "",
    "aria-label": "option description " + idx,
  });
  const rmBtn = el("button", {
    class: "rm",
    type: "button",
    "aria-label": "remove option " + idx,
  }, "×");
  rmBtn.addEventListener("click", () => {
    if (container.children.length > 2) row.remove();
    renumberRows(container);
  });
  row.append(idxSpan, keyInput, descInput, rmBtn);
  container.appendChild(row);
}

function addScoreRow(label, container) {
  container = container || $("#scoreRows");
  const idx = container.children.length;
  const row = el("div", { class: "opt-row" });
  const idxSpan = el("span", { class: "idx" }, String(idx));
  const labelInput = el("input", {
    class: "field",
    type: "text",
    placeholder: "level description",
    value: label || "",
    "aria-label": "level " + idx,
  });
  const rmBtn = el("button", {
    class: "rm",
    type: "button",
    "aria-label": "remove level " + idx,
  }, "×");
  rmBtn.addEventListener("click", () => {
    if (container.children.length > 2) row.remove();
    renumberRows(container);
  });
  row.append(idxSpan, labelInput, rmBtn);
  // score rows have no description column
  row.style.gridTemplateColumns = "34px 1fr 34px";
  container.appendChild(row);
}

function renumberRows(container) {
  Array.from(container.children).forEach((row, i) => {
    const idxSpan = row.querySelector(".idx");
    if (idxSpan) idxSpan.textContent = String(i + 1);
  });
}

function readChoiceRows() {
  const rows = $$("#choiceRows .opt-row");
  const criteria = {};
  const dupes = [];
  for (const row of rows) {
    const inputs = row.querySelectorAll("input.field");
    const key = inputs[0].value.trim();
    const desc = inputs[1].value.trim();
    if (!key) continue;
    if (criteria[key] !== undefined) {
      dupes.push(key);
      continue;
    }
    criteria[key] = desc || null;
  }
  return { criteria, dupes };
}

function readScoreRows() {
  const rows = $$("#scoreRows .opt-row");
  const levels = [];
  for (const row of rows) {
    const input = row.querySelector("input.field");
    const val = input.value.trim();
    if (val) levels.push(val);
  }
  return levels;
}

// ── batch questions ────────────────────────────────────────────────────────

let batchIdCounter = 0;

function addBatchQuestion(type, instructions) {
  const container = $("#batchQuestions");
  const id = "q" + ++batchIdCounter;
  const card = el("div", { class: "batch-card", "data-batch-id": id });

  const head = el("h3");
  const qtypeSpan = el("span", { class: "qtype" }, type);
  const idLabel = el("span", {}, id);
  head.append(qtypeSpan, " ", idLabel);
  card.appendChild(head);

  const typeSel = el("select", {
    class: "field",
    "aria-label": "question type for " + id,
  });
  ["choice", "score", "noul"].forEach((t) => {
    const opt = el("option", { value: t }, t);
    if (t === type) opt.selected = true;
    typeSel.appendChild(opt);
  });
  card.appendChild(typeSel);

  const insArea = el("textarea", {
    class: "field",
    rows: "2",
    "aria-label": "instructions for " + id,
    placeholder: "Question / instructions",
  });
  insArea.value = instructions || "";
  card.appendChild(insArea);

  // type-specific sub-controls
  const subDiv = el("div", { class: "batch-sub" });
  function rebuildSub() {
    subDiv.innerHTML = "";
    const t = typeSel.value;
    if (t === "choice") {
      const hint = el("p", { class: "fine" }, "Option key → optional description (one per line: key=description)");
      subDiv.appendChild(hint);
      const ta = el("textarea", {
        class: "field",
        rows: "3",
        "aria-label": "choice options for " + id,
        placeholder: "billing=Payment issues\nsupport=Technical help",
      });
      subDiv.appendChild(ta);
    } else if (t === "score") {
      const hint = el("p", { class: "fine" }, "Ordered levels, one per line (index 0 = lowest)");
      subDiv.appendChild(hint);
      const ta = el("textarea", {
        class: "field",
        rows: "4",
        "aria-label": "score levels for " + id,
        placeholder: "low\nmedium\nhigh",
      });
      subDiv.appendChild(ta);
    }
  }
  rebuildSub();
  typeSel.addEventListener("change", rebuildSub);
  card.appendChild(subDiv);

  const rmBtn = el("button", {
    class: "btn ghost small",
    type: "button",
    "aria-label": "remove question " + id,
  }, "Remove");
  rmBtn.addEventListener("click", () => card.remove());
  card.appendChild(rmBtn);

  container.appendChild(card);
  return id;
}

function readBatchQuestions() {
  const cards = $$("#batchQuestions .batch-card");
  const questions = {};
  for (const card of cards) {
    const id = card.dataset.batchId;
    if (!id) continue;
    const typeSel = card.querySelector("select");
    const insArea = card.querySelector("textarea.field");
    const subDiv = card.querySelector(".batch-sub");
    const type = typeSel ? typeSel.value : "noul";
    const instructions = insArea ? insArea.value.trim() : "";
    const subTa = subDiv ? subDiv.querySelector("textarea") : null;
    const subText = subTa ? subTa.value : "";

    if (type === "choice") {
      const criteria = {};
      for (const line of subText.split("\n")) {
        const sepIdx = line.indexOf("=");
        if (sepIdx === -1) continue;
        const k = line.slice(0, sepIdx).trim();
        const v = line.slice(sepIdx + 1).trim();
        if (k) criteria[k] = v || null;
      }
      questions[id] = { type, instructions, criteria };
    } else if (type === "score") {
      const levels = subText
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
      questions[id] = { type, instructions, criteria: levels };
    } else {
      questions[id] = { type, instructions };
    }
  }
  return questions;
}

// ── request assembly ───────────────────────────────────────────────────────

function buildRequest() {
  const st = getStateValue();
  if (st === null) return null; // JSON parse error (already shown)

  const t = activeTabType();
  let questions;

  if (t === "choice") {
    const { criteria, dupes } = readChoiceRows();
    const err = $("#choiceError");
    if (dupes.length) {
      err.textContent = "Duplicate option keys: " + dupes.join(", ");
      err.classList.remove("hidden");
      return null;
    }
    if (Object.keys(criteria).length < 2) {
      err.textContent = "At least 2 options required.";
      err.classList.remove("hidden");
      return null;
    }
    err.classList.add("hidden");
    const ins = $("#choice-ins").value.trim();
    questions = { q: { type: "choice", instructions: ins, criteria } };
  } else if (t === "score") {
    const levels = readScoreRows();
    const err = $("#scoreError");
    if (levels.length < 2) {
      err.textContent = "At least 2 levels required (ordered, lowest first).";
      err.classList.remove("hidden");
      return null;
    }
    err.classList.add("hidden");
    const ins = $("#score-ins").value.trim();
    questions = { q: { type: "score", instructions: ins, criteria: levels } };
  } else if (t === "noul") {
    const ins = $("#noul-ins").value.trim();
    const trueDesc = $("#noul-true").value.trim();
    const falseDesc = $("#noul-false").value.trim();
    const q = { type: "noul", instructions: ins };
    const criteria = {};
    if (trueDesc) criteria.true = trueDesc;
    if (falseDesc) criteria.false = falseDesc;
    if (Object.keys(criteria).length) q.criteria = criteria;
    questions = { q };
  } else if (t === "batch") {
    questions = readBatchQuestions();
    const err = $("#batchError");
    if (Object.keys(questions).length === 0) {
      err.textContent = "Add at least one question.";
      err.classList.remove("hidden");
      return null;
    }
    err.classList.add("hidden");
  } else if (t === "raw") {
    const raw = $("#rawBody").value.trim();
    const err = $("#rawError");
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      err.textContent = "Invalid JSON: " + e.message;
      err.classList.remove("hidden");
      return null;
    }
    err.classList.add("hidden");
    // raw mode: the entire body is the request
    return parsed;
  }

  return { state: st, questions };
}

// ── run ────────────────────────────────────────────────────────────────────

async function doRun() {
  if (!validateState()) return;

  const body = buildRequest();
  if (!body) return;

  const resultsCol = $("#resultsCol");
  // disable run while in flight
  const btn = $("#btnRun");
  btn.disabled = true;
  btn.textContent = "Running…";

  // pre-fetch health for cold/warm label
  let preHealth = null;
  try {
    const hr = await fetch("/api/health");
    preHealth = await hr.json().catch(() => ({}));
  } catch (_) {
    /* advisory only */
  }
  const wasLoaded = preHealth ? preHealth.model_loaded === true : null;

  const t0 = performance.now();
  let resp;
  let gwErr = null;
  try {
    resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    gwErr = { status: 0, message: "Network error: " + e.message, origin: "browser" };
  }
  const elapsed = performance.now() - t0;

  btn.disabled = false;
  btn.textContent = "Run";

  if (gwErr) {
    renderError(gwErr, elapsed, body);
    refreshHealth();
    return;
  }

  const rawText = await resp.text();
  let parsed;
  try {
    parsed = JSON.parse(rawText);
  } catch (_) {
    parsed = null;
  }

  if (!resp.ok) {
    renderError(
      {
        status: resp.status,
        body: parsed,
        rawText: rawText,
        message: extractErrorMessage(parsed, rawText),
        origin: resp.headers.get("Content-Type") || "",
      },
      elapsed,
      body
    );
    refreshHealth();
    return;
  }

  renderSuccess(parsed, elapsed, body, wasLoaded, rawText);
  refreshHealth();
}

function extractErrorMessage(parsed, rawText) {
  if (parsed) {
    if (parsed.error && parsed.error.message) return parsed.error.message;
    if (parsed.detail) {
      if (typeof parsed.detail === "string") return parsed.detail;
      if (Array.isArray(parsed.detail))
        return parsed.detail.map((d) => (d.msg || JSON.stringify(d))).join("; ");
    }
  }
  return rawText.slice(0, 500);
}

// ── rendering ──────────────────────────────────────────────────────────────

function clearResults() {
  const col = $("#resultsCol");
  col.innerHTML = "";
}

function renderSuccess(data, elapsedMs, requestBody, wasLoaded, rawText) {
  clearResults();
  const col = $("#resultsCol");

  const usage = data.usage || {};
  const cached = data.cached === true;
  const inputTokens = usage.input_tokens;
  const outputTokens = usage.output_tokens;

  // meta strip
  const meta = el("div", { class: "meta-strip" });
  const latencyChip = el("span", { class: "meta-chip" },
    "wall time ", el("strong", {}, fmtMs(elapsedMs)), " (browser→service, not GPU kernel time)");
  if (inputTokens != null)
    meta.appendChild(el("span", { class: "meta-chip" },
      "input tokens ", el("strong", {}, String(inputTokens)), " (after any truncation)"));
  if (outputTokens != null)
    meta.appendChild(el("span", { class: "meta-chip" }, "output tokens ", el("strong", {}, String(outputTokens))));
  meta.appendChild(el("span", { class: "meta-chip", dataset: { state: cached ? "cache" : "" } },
    cached ? "cached: true" : "cached: false"));
  if (wasLoaded === false)
    meta.appendChild(el("span", { class: "meta-chip", dataset: { state: "cold" } },
      "model was unloaded before this request (reload likely)"));
  else if (wasLoaded === true)
    meta.appendChild(el("span", { class: "meta-chip" }, "model was loaded before this request"));
  else
    meta.appendChild(el("span", { class: "meta-chip" }, "cold/warm: unknown"));
  col.appendChild(meta);

  const answers = data.answers || {};
  const qids = Object.keys(answers);

  if (qids.length === 1) {
    const qid = qids[0];
    const ans = answers[qid];
    const card = el("div", { class: "card result-card" });
    renderAnswerCard(card, qid, ans);
    col.appendChild(card);
  } else {
    // multiple answers → batch view
    const header = el("h2", { class: "result-topline" }, "Batch — " + qids.length + " answers");
    col.appendChild(header);
    const batchCards = el("div", { class: "batch-cards" });
    for (const qid of qids) {
      const bc = el("div", { class: "batch-card" });
      renderAnswerCard(bc, qid, answers[qid]);
      batchCards.appendChild(bc);
    }
    col.appendChild(batchCards);
  }

  appendRawDetails(col, requestBody, rawText);
}

function renderAnswerCard(container, qid, ans) {
  const type = ans.type;

  // qid header
  const h = el("h3");
  if (type) h.appendChild(el("span", { class: "qtype" }, type + " "));
  h.appendChild(document.createTextNode(qid));
  container.appendChild(h);

  if (type === "choice") {
    renderChoice(container, ans);
  } else if (type === "score") {
    renderScore(container, ans);
  } else if (type === "noul") {
    renderNoul(container, ans);
  } else {
    container.appendChild(el("p", { class: "fine" }, "Unrecognised answer type: " + String(type)));
  }

  appendRouting(container, ans.rl_agent);
}

function renderChoice(container, ans) {
  const probs = ans.probabilities || {};
  const keys = Object.keys(probs);
  const winner = ans.choice;

  // compute margin (winner − runner-up) and ambiguous flag
  const sortedDesc = [...keys].sort((a, b) => probs[b] - probs[a]);
  const topP = sortedDesc.length > 0 ? probs[sortedDesc[0]] : 0;
  const secondP = sortedDesc.length > 1 ? probs[sortedDesc[1]] : 0;
  const margin = sortedDesc.length > 1 ? topP - secondP : (sortedDesc.length === 1 ? 1 : 0);
  const AMBIGUOUS_THRESHOLD = 0.05;
  const isAmbiguous = sortedDesc.length > 1 && margin < AMBIGUOUS_THRESHOLD;

  // top-line — add ambiguous cue when close
  const topline = el("p", { class: "result-topline" + (isAmbiguous ? " ambiguous" : "") });
  if (isAmbiguous) {
    topline.appendChild(el("span", { class: "ambig-badge" }, "close decision · ambiguous"));
    topline.append("  ");
  }
  topline.append(
    "Winning option: ",
    el("span", { class: "win" + (isAmbiguous ? " dim" : "") }, winner || "(none)"),
  );
  container.appendChild(topline);

  if (isAmbiguous) {
    container.appendChild(el("p", { class: "fine ambig-note" },
      "The model cannot meaningfully distinguish the top two options — the distribution is nearly flat. This is a useful signal about the taxonomy, not a verdict."));
  }

  // bars sorted descending
  const chart = el("div", { class: "chart" + (isAmbiguous ? " ambiguous" : "") });
  for (const k of sortedDesc) {
    const p = probs[k];
    const row = el("div", { class: "bar-row" + (k === winner ? " win" : "") + (isAmbiguous && sortedDesc.indexOf(k) < 2 ? " near-tie" : "") });
    const label = el("span", { class: "bar-label" },
      k === winner ? el("span", { class: "win-tag" }, k + " ★") : k);
    const track = el("div", { class: "bar-track" });
    const fill = el("div", {
      class: "bar-fill",
      style: "width:" + Math.max(0, Math.min(100, p * 100)).toFixed(2) + "%",
    });
    track.appendChild(fill);
    const val = el("span", { class: "bar-val" }, fmtProb(p));
    row.append(label, track, val);
    chart.appendChild(row);
  }
  container.appendChild(chart);

  // confidence + margin line
  if (ans.confidence != null) {
    const confLine = el("p", { class: "conf-line" },
      "Confidence (distribution sharpness, not correctness): ",
      el("span", { class: "val" }, fmtProb(ans.confidence)));
    if (sortedDesc.length > 1) {
      confLine.append(
        "  ·  Margin: ",
        el("span", { class: "val margin-val" }, fmtProb(margin)),
      );
    }
    container.appendChild(confLine);
  }
}

function renderScore(container, ans) {
  const probs = ans.probabilities || {};
  const legend = ans.legend || {};
  const indices = Object.keys(probs).map(Number).sort((a, b) => a - b);
  const n = indices.length;

  // distribution bars
  const chart = el("div", { class: "chart" });
  for (const i of indices) {
    const p = probs[String(i)];
    const label = legend[String(i)] != null ? legend[String(i)] : "level " + i;
    const row = el("div", { class: "bar-row" });
    const lab = el("span", { class: "bar-label" }, "[" + i + "] " + label);
    const track = el("div", { class: "bar-track" });
    const fill = el("div", {
      class: "bar-fill",
      style: "width:" + Math.max(0, Math.min(100, p * 100)).toFixed(2) + "%",
    });
    track.appendChild(fill);
    const val = el("span", { class: "bar-val" }, fmtProb(p));
    row.append(lab, track, val);
    chart.appendChild(row);
  }
  container.appendChild(chart);

  // expected score axis
  if (ans.score != null && n > 0) {
    const axis = el("div", { class: "score-axis" });
    const line = el("div", { class: "axis-line" });
    axis.appendChild(line);
    for (const i of indices) {
      const tick = el("span", {
        class: "tick",
        style: "left:" + (i / Math.max(1, n - 1)) * 100 + "%",
      }, String(i));
      axis.appendChild(tick);
    }
    const score = ans.score;
    const clamped = Math.max(0, Math.min(n - 1, score));
    const marker = el("div", {
      class: "marker",
      style: "left:" + (clamped / Math.max(1, n - 1)) * 100 + "%",
    });
    const mlabel = el("span", {
      class: "marker-label",
      style: "left:" + (clamped / Math.max(1, n - 1)) * 100 + "%",
    }, "E = " + fmtProb(score));
    axis.append(marker, mlabel);
    container.appendChild(axis);
    container.appendChild(el("p", { class: "fine" },
      "Expected score is a probability-weighted mean over 0…" + (n - 1) +
      " — it can fall between levels. It is not a selected level and not a 0–1 probability."));
  }

  if (ans.confidence != null) {
    container.appendChild(el("p", { class: "conf-line" },
      "Confidence (distribution sharpness, not correctness): ",
      el("span", { class: "val" }, fmtProb(ans.confidence))));
  }
}

function renderNoul(container, ans) {
  const pTrue = ans.noul != null ? ans.noul : NaN;
  const pFalse = Number.isFinite(pTrue) ? 1 - pTrue : NaN;

  const big = el("div", { class: "noul-big" },
    Number.isFinite(pTrue) ? fmtProb(pTrue) : "—");
  container.appendChild(big);
  container.appendChild(el("p", { class: "noul-true-label" }, "P(true) — as returned by the model"));

  const split = el("div", { class: "split-bar" });
  const segT = el("div", {
    class: "split-seg-true",
    style: "width:" + (Number.isFinite(pTrue) ? Math.max(0, Math.min(100, pTrue * 100)) : 0) + "%",
  });
  const segF = el("div", {
    class: "split-seg-false",
    style: "width:" + (Number.isFinite(pFalse) ? Math.max(0, Math.min(100, pFalse * 100)) : 0) + "%",
  });
  split.append(segT, segF);
  container.appendChild(split);

  const legend = el("div", { class: "split-legend" });
  legend.appendChild(el("span", {},
    el("span", { class: "sw", style: "background:var(--accent)" }),
    "P(true) " + (Number.isFinite(pTrue) ? pct(pTrue) : "—")));
  legend.appendChild(el("span", {},
    el("span", { class: "sw", style: "background:#4a4234" }),
    "P(false) " + (Number.isFinite(pFalse) ? pct(pFalse) : "—") + " (derived: 1 − P(true))"));
  container.appendChild(legend);
}

function appendRouting(container, rlAgent) {
  if (!rlAgent) return;
  const details = el("details", { class: "routing-panel" });
  const summary = el("summary", {},
    "Experimental routing signal",
    el("span", { class: "experimental-tag" }, "experimental — untrusted"));
  details.appendChild(summary);

  const body = el("div", { class: "routing-body" });
  if (rlAgent.act_probability != null)
    body.appendChild(el("span", {},
      "act_probability: ", el("span", { class: "val" }, fmtProb(rlAgent.act_probability))));
  if (rlAgent.escalate_probability != null)
    body.appendChild(el("span", {},
      "escalate_probability: ", el("span", { class: "val" }, fmtProb(rlAgent.escalate_probability))));
  body.appendChild(el("span", { class: "fine" },
    "These come from a separate RL-trained head. Do NOT use them for operational routing decisions without domain-specific calibration."));
  details.appendChild(body);
  container.appendChild(details);
}

function renderError(errInfo, elapsedMs, requestBody) {
  clearResults();
  const col = $("#resultsCol");
  const card = el("div", { class: "card result-card error-result" });

  const h = el("h2", {}, "Request failed" + (errInfo.status ? " — HTTP " + errInfo.status : ""));
  card.appendChild(h);

  const msg = el("p", {}, errInfo.message || "Unknown error");
  card.appendChild(msg);

  if (errInfo.body) {
    const pre = el("pre", { class: "raw-pre" });
    pre.textContent = JSON.stringify(errInfo.body, null, 2);
    card.appendChild(pre);
  } else if (errInfo.rawText) {
    const pre = el("pre", { class: "raw-pre" });
    pre.textContent = errInfo.rawText.slice(0, 4000);
    card.appendChild(pre);
  }

  const meta = el("div", { class: "meta-strip" });
  meta.appendChild(el("span", { class: "meta-chip" },
    "wall time ", el("strong", {}, fmtMs(elapsedMs))));
  card.appendChild(meta);

  appendRawDetails(card, requestBody, errInfo.rawText || "");
  col.appendChild(card);
}

function appendRawDetails(container, requestBody, rawResponseText) {
  const details = el("details", { class: "rawdetails" });
  const summary = el("summary", {}, "Raw request & response (JSON)");
  details.appendChild(summary);

  const reqPre = el("pre", { class: "raw-pre" });
  reqPre.textContent = "── REQUEST ──\n" + JSON.stringify(requestBody, null, 2);
  const respPre = el("pre", { class: "raw-pre" });
  respPre.textContent = "\n── RESPONSE (verbatim) ──\n" + (rawResponseText || "(none)");
  details.append(reqPre, respPre);
  container.appendChild(details);
}

// ── examples (populate only — never auto-run) ─────────────────────────────

const EXAMPLES = {
  supportTriage: {
    label: "Support triage (choice)",
    state: "Ticket #4821: Customer says the nightly backup job has been failing for three nights after the storage array upgrade. Error log shows 'insufficient space on /backup'. Customer is on the annual enterprise plan and is requesting a refund if not resolved by Friday.",
    tab: "choice",
    instructions: "Which team should handle this ticket?",
    options: [
      ["technical_support", "Faults, outages, configuration, storage and backup failures"],
      ["billing", "Refunds, invoices, subscriptions, payment disputes"],
    ],
  },
  urgencyScore: {
    label: "Urgency (score)",
    state: "Production database primary node has been degraded for 6 hours. Replication lag is growing. Two customer-facing APIs are timing out intermittently. No data loss reported yet but the on-call SRE estimates the node will fail within 24 hours if not replaced.",
    tab: "score",
    instructions: "How urgent is this incident on the following scale?",
    levels: ["routine / no action needed", "monitor / low priority", "investigate today", "respond within hours", "critical / active outage risk"],
  },
  factualNoul: {
    label: "Factual (noul)",
    state: "Server restart log from Monday: at 09:14 UTC the ops team ran 'systemctl restart laya-service'. The service came back healthy at 09:15. There is no other restart entry in the log for that day.",
    tab: "noul",
    instructions: "The laya-service was restarted at 14:00 on Monday.",
    trueDesc: "",
    falseDesc: "",
  },
  mixedBatch: {
    label: "Mixed batch (one request)",
    state: "Ticket #7210: User reports that after enabling two-factor authentication on their account, they can no longer receive the weekly CSV export email. They say the account is in good standing and they have paid all invoices. The user is asking whether this is a bug or an expected restriction.",
    tab: "batch",
    questions: [
      { type: "choice", instructions: "Is this most likely a product bug, an expected restriction, or a billing issue?", options: [["bug", "Product defect"], ["expected", "By-design behaviour"], ["billing", "Account in arrears"]] },
      { type: "score", instructions: "How likely is it that this is a genuine product bug?", levels: ["not at all", "unlikely", "plausible", "likely", "certain"] },
      { type: "noul", instructions: "The user has confirmed their account is in good standing." },
    ],
  },
};

function loadExample(name) {
  const ex = EXAMPLES[name];
  if (!ex) return;

  // set state
  state.format = "text";
  $("#stateModeText").click();
  $("#stateText").value = ex.state;
  state.text = ex.state;

  // switch tab
  const tab = $("#tab-" + ex.tab);
  if (tab) tab.click();

  if (ex.tab === "choice") {
    $("#choice-ins").value = ex.instructions;
    const rows = $("#choiceRows");
    rows.innerHTML = "";
    for (const [k, d] of ex.options) addChoiceRow(k, d, rows);
  } else if (ex.tab === "score") {
    $("#score-ins").value = ex.instructions;
    const rows = $("#scoreRows");
    rows.innerHTML = "";
    for (const lvl of ex.levels) addScoreRow(lvl, rows);
  } else if (ex.tab === "noul") {
    $("#noul-ins").value = ex.instructions;
    $("#noul-true").value = ex.trueDesc || "";
    $("#noul-false").value = ex.falseDesc || "";
  } else if (ex.tab === "batch") {
    const container = $("#batchQuestions");
    container.innerHTML = "";
    batchIdCounter = 0;
    for (const q of ex.questions) {
      const id = addBatchQuestion(q.type, q.instructions);
      const card = $$("#batchQuestions .batch-card").pop();
      const subTa = card.querySelector(".batch-sub textarea");
      if (subTa && q.type === "choice") {
        subTa.value = q.options.map(([k, d]) => k + "=" + (d || "")).join("\n");
      } else if (subTa && q.type === "score") {
        subTa.value = q.levels.join("\n");
      }
    }
  }
}

// ── break-it probes (populate only — never auto-run) ──────────────────────

const PROBES = {
  "empty-state": {
    tab: "noul",
    state: "",
    instructions: "The system is operational.",
  },
  "huge-state": {
    tab: "noul",
    state: (function () {
      const filler =
        "The infrastructure team performed a routine maintenance window on the storage array. " +
        "Firmware was updated across all twelve nodes. Bandwidth checks passed. Thermal sensors nominal. " +
        "The backup subsystem was re-registered with the scheduler. Log rotation was reconfigured. " +
        "The monitoring dashboards were refreshed. Alert thresholds were reviewed. On-call handover completed. ";
      return filler.repeat(60) +
        "CRITICAL FACT NEAR THE END: the nightly backup job WAS DISABLED on Friday at 17:00 by an admin, and this has never been re-enabled.";
    })(),
    instructions: "The nightly backup job is still enabled and running.",
  },
  "many-options": {
    tab: "choice",
    state: "A ticket with no clear category. Which of these applies?",
    instructions: "Pick the single best category.",
    options: Array.from({ length: 128 }, (_, i) => ["opt" + String(i).padStart(3, "0"), "option " + i]),
  },
  "score-10": {
    tab: "score",
    state: "Assess the urgency of this incident.",
    instructions: "Rate on the 10-level scale below.",
    levels: ["0 none", "1 negligible", "2 minor", "3 low", "4 moderate-low", "5 moderate", "6 moderate-high", "7 high", "8 very high", "9 critical"],
  },
  "score-11": {
    tab: "score",
    state: "Assess the urgency of this incident.",
    instructions: "Rate on the 11-level scale below (documented range is 2–10; 11 was observed to be accepted).",
    levels: ["0 none", "1 negligible", "2 minor", "3 low", "4 moderate-low", "5 moderate", "6 moderate-high", "7 high", "8 very high", "9 critical", "10 existential"],
  },
  "score-1": {
    tab: "raw",
    rawBody: JSON.stringify(
      {
        state: "Test state for 1-level score.",
        questions: {
          q: {
            type: "score",
            instructions: "Rate it.",
            criteria: ["only one level"],
          },
        },
      },
      null,
      2
    ),
  },
  negation: {
    tab: "noul",
    state: "Server restart log from Monday: at 09:14 UTC the ops team ran 'systemctl restart laya-service'. The service came back healthy at 09:15. No other restart entries that day.",
    instructions: "Was the service NOT restarted at 14:00 on Monday?",
  },
  "bad-type": {
    tab: "raw",
    rawBody: JSON.stringify(
      {
        state: "Test state.",
        questions: {
          q: { type: "banana", instructions: "What is this?" },
        },
      },
      null,
      2
    ),
  },
};

function loadProbe(key) {
  const probe = PROBES[key];
  if (!probe) return;

  if (probe.rawBody) {
    const tab = $("#tab-raw");
    if (tab) tab.click();
    $("#rawBody").value = probe.rawBody;
    return;
  }

  state.format = "text";
  $("#stateModeText").click();
  $("#stateText").value = probe.state || "";
  state.text = probe.state || "";

  const tab = $("#tab-" + probe.tab);
  if (tab) tab.click();

  if (probe.tab === "choice") {
    $("#choice-ins").value = probe.instructions;
    const rows = $("#choiceRows");
    rows.innerHTML = "";
    for (const [k, d] of probe.options) addChoiceRow(k, d, rows);
  } else if (probe.tab === "score") {
    $("#score-ins").value = probe.instructions;
    const rows = $("#scoreRows");
    rows.innerHTML = "";
    for (const lvl of probe.levels) addScoreRow(lvl, rows);
  } else if (probe.tab === "noul") {
    $("#noul-ins").value = probe.instructions;
    $("#noul-true").value = "";
    $("#noul-false").value = "";
  }
}

// ── init ───────────────────────────────────────────────────────────────────

function init() {
  initStateEditor();
  initTabs();

  // seed default choice rows
  addChoiceRow("billing", "Payment or plan issues");
  addChoiceRow("support", "General help");

  // seed default score rows
  addScoreRow("low");
  addScoreRow("high");

  // seed one batch question
  addBatchQuestion("choice", "Which team?");

  // seed raw body
  $("#rawBody").value = JSON.stringify(
    {
      state: { subject: "Test ticket", body: "Something happened." },
      questions: {
        q: {
          type: "noul",
          instructions: "Something happened, right?",
        },
      },
    },
    null,
    2
  );

  // run button
  $("#btnRun").addEventListener("click", doRun);

  // health refresh
  $("#btnRefreshHealth").addEventListener("click", refreshHealth);
  refreshHealth(); // initial load

  // add-option / add-level / add-batch buttons
  $("#btnAddChoice").addEventListener("click", () => addChoiceRow("", ""));
  $("#btnAddScore").addEventListener("click", () => addScoreRow(""));
  $("#btnAddBatchQ").addEventListener("click", () => addBatchQuestion("choice", ""));

  // break-it probe buttons
  $$(".btn.probe").forEach((btn) => {
    btn.addEventListener("click", () => loadProbe(btn.dataset.probe));
  });
}

document.addEventListener("DOMContentLoaded", init);

# Laya GPU Inference Service — Architecture & Service Spec

This document describes *how* the service is built: components, data flow, the HTTP contract in
detail, tuning knobs, and the daemon/lifecycle model. For deploy/ops details see [LINK.md](LINK.md);
for the original design rationale see [`../laya/docs/LAYA-GPU-SERVICE.md`](../laya/docs/LAYA-GPU-SERVICE.md).

---

## Design goal

A **stable, reliable, on-demand** GPU inference endpoint for the Laya decision model. The user's
constraint (2026-09-28): the model holds ~2.3 GB VRAM, which is not worth keeping resident 24/7.
So the service is started/stopped on demand (like ComfyUI / vLLM on bobai) **and** auto-unloads the
model from VRAM after an idle timeout, re-loading it transparently on the next request.

The topology mirrors `/MASTERFOLDER/Services/embed-server` (lean, well-tuned, reliable GPU HTTP):
single process, resident-while-busy model, request coalescing, inference lock, non-blocking event
loop, two-tier cache, and a clean start/stop lifecycle.

---

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| HTTP layer | `src/laya_server/server.py` | FastAPI app factory (`create_app`) + runner (`run_server`). Endpoints, lifespan lifecycle, idle reaper. Exposed as the `laya-server` console command. |
| Config | `src/laya_server/_config.py` | Typed Pydantic models (`AppConfig` and sub-models) with validation. Strict resolution: `LAYA_CONFIG_PATH` → CWD-relative → raise. |
| Worker | `src/laya_server/worker.py` | `LayaWorker` — owns the `onnxruntime-gpu` `InferenceSession` + tokenizer; builds inputs; runs; applies per-cardinality temperature + softmax; shapes answers exactly as the TS `laya.ts`. |
| Sequence logic | `src/laya_server/sequence.py` | Pure, model-free port of the request rendering (sequence layout, option rendering, `pyJsonDumps`, temperature buckets, confidence, softmax, `round4`). Unit-testable without weights. |
| Batcher | `src/laya_server/batcher.py` | `RequestBatcher` — asyncio queue + time-window coalescing; merges concurrent requests into one forward pass; dispatches the blocking call to the thread executor. |
| Cache | `src/laya_server/result_cache.py` | Two-tier LRU (in-memory) + SQLite; keyed on `sha256(state, questions)` (deterministic results). |

### Data flow

```
HTTP :11448
  └─ FastAPI (uvicorn, async event loop)
       ├─ POST /system_one
       │     └─ RequestBatcher.submit(payload)
       │           └─ [queue + 5ms window → merge N payloads]
       │                 └─ run_in_executor(_process_batch)
       │                       └─ per payload: _process_payload
       │                             ├─ cache.get(key)? → return (≈6 ms)
       │                             └─ cache miss → worker.system_one (infer_lock, CUDA:0)
       │                                   └─ buildSequence + tokenize → ONNX run →
       │                                      temperature/softmax → shape answer
       ├─ GET /health          (providers, model_loaded, idle_s, uptime)
       ├─ GET /v1/stats        (requests, latency p50/p99, cache hit/miss, providers)
       └─ POST /control/unload (explicitly free VRAM now)
       └─ _idle_reaper task: if idle ≥ unload_after_s → unload_worker() (frees VRAM)
```

Single process. The ONNX session loads once (pre-warmed at startup), is held while busy, and is
**unloaded when idle** (auto, or via `/control/unload`). On the next request it **re-loads lazily**
(~0.5 s) — that reload cost is the price of the idle-unload and is acceptable for on-demand use.

---

## Lifecycle

Process lifecycle is managed by systemd (production) or the `laya-server` foreground command (dev).
The server manages VRAM independently. Two levels:

| Level | Mechanism | Effect |
|-------|-----------|--------|
| **Process** | `systemctl start/stop/restart laya-service` (production) or `laya-server` in foreground (dev) | Starts/stops the whole HTTP service. Stop = SIGTERM, uvicorn drains in-flight requests. Frees all VRAM. |
| **VRAM (model)** | Auto idle-unload (`idle.unload_after_s`, default 300 s) **or** `POST /control/unload` | Releases the ONNX session's ~2.3 GB VRAM *without* stopping the process. Model **re-loads lazily** on the next `/system_one` request (~0.7 s cold start). `GET /health` reports `"status": "degraded"` and `"model_loaded": false` while unloaded. |

**Why two levels:** the user wants to start/stop the *service* on demand (ComfyUI/vLLM-style). The
VRAM level additionally means that even while the process is up, an idle service returns its VRAM
to the GPU after 5 min — so a "leave it running" session doesn't hoard 2.3 GB. Set
`idle.unload_after_s` to `0`/`idle.enabled=false` to keep it permanently resident.

**Idle tracking:** every `/system_one` updates a last-activity timestamp. The `_idle_reaper` task
checks every 60 s and unloads when idle ≥ threshold. `GET /health` reports `model_loaded` so
operators can distinguish "ready" from "will-be-ready-after-cold-start".

**SIGKILL:** in-flight requests are not checkpointed. If the process is killed without SIGTERM
(e.g. OOM kill), in-flight requests are lost. This is acceptable for an inference service —
requests are idempotent and can be retried.

This is the same TTL/idle-shutdown pattern as FSS Chatterbox's `model_daemon.py` (`ttl`,
`touch()`, auto-shutdown on idle), adapted from a file-queue daemon to an HTTP server.

---

## HTTP contract

Base URL: `http://<host>:11448`

### `POST /system_one`

**Request body** (`SystemOneRequest`):
```json
{
  "state": <any JSON value>,
  "questions": { "<qid>": <Question>, ... }
}
```
`state` may be a string (used verbatim) or a JSON object/array (serialized with Python
`json.dumps(ensure_ascii=False)` to match the reference). `questions` is a non-empty map of
qid → question.

**Question types** (mirrors the TS library / Jev's `system_one`):

| type | `criteria` | Answer fields |
|------|-----------|---------------|
| `choice` | `Record<option, description\|null>` **or** `string[]` of option names | `choice` (selected key), `probabilities` per option, `confidence` |
| `score` | `string[]` of ordered levels (index 0 = lowest) | `score` (expected level, float), `legend`, `probabilities` per level, `confidence` |
| `noul` | optional `{true?, false?}` descriptions | `noul` (P(true)) |

`instructions` is a string (or an object, which is JSON-serialized).

**Response** (200):
```json
{
  "model": "laya",
  "answers": {
    "<qid>": {
      "type": "choice|score|noul",
      "...per-type fields above...",
      "rl_agent": { "act_probability": <float> }
    }
  },
  "usage": { "input_tokens": <int>, "output_tokens": 0 }
}
```
Response headers: `X-Request-Id` (short id), `X-Latency-Ms`.

**Errors** (OpenAI-style envelope): `{"error": {"message": "...", "type": "...", "param": null, "code": null}}`
with status 422 (validation) or 500 (server). Error `type` ∈ `invalid_request_error`, `server_error`.

### `GET /health`
```json
{"status":"ok","model":"laya","providers":["CUDAExecutionProvider","CPUExecutionProvider"],
 "model_loaded":true,"idle_s":1.2,"idle_unload_after_s":300,"uptime_s":42.0}
```
`providers` empty / `model_loaded:false` ⇒ model currently unloaded (idle); next request reloads it.

### `GET /v1/stats`
```json
{"requests_total":N,"errors_total":N,"cache":{"hits":N,"misses":N,"enabled":true},
 "batch_coalesced":N,"latency_ms":{"p50":N,"p99":N,"last":N},
 "providers":[...],"model_loaded":true,"idle_s":N,"uptime_s":N}
```

### `POST /control/unload`
Explicitly free VRAM now (idempotent). Returns `{"status":"unloaded","model_loaded":false}`. The model
re-loads lazily on the next request (~0.7 s cold start).

---

## Tuning knobs (`config/laya_config.json`)

| Key | Default | Meaning |
|-----|---------|---------|
| `model_dir` | `/MASTERFOLDER/Services/laya/onnx` | Bundle location (read-only). |
| `host` / `port` | `0.0.0.0` / `11448` | Bind address. |
| `gpu.visible_devices` | `0` | Card to pin (set via `CUDA_VISIBLE_DEVICES`; see daemon). |
| `gpu.providers` | `[CUDA, CPU]` | Provider preference list (CUDA first, CPU fallback). |
| `idle.enabled` | `true` | Auto-unload when idle. |
| `idle.unload_after_s` | `300` | Idle seconds before VRAM is freed. `0`/disabled = always resident. |
| `cache.enabled` | `true` | Enable result cache. |
| `cache.ttl_days` / `max_memory_entries` | `30` / `10000` | Cache TTL + LRU size. |
| `cache.db_path` | `~/.laya-service/result_cache.db` | SQLite cache path. |
| `batcher.max_batch_size` / `max_wait_ms` | `512` / `5` | Coalescing cap + drain window. |

**Model-level knobs** (`max_len`, `head_max_len`, `temperature`, `temperature_by_options`) live in
the bundle's `laya_config.json` and are calibration constants from the checkpoint — change only
deliberately, and re-run `test_parity.py` (they break parity with the reference).

---

## Correctness notes

- **Parity:** GPU outputs differ from CPU by ~1–4e-4 per probability (~6–7e-4 on the nonlinear
  `confidence`/`score` aggregates) — standard CPU-vs-CUDA FP32 non-determinism. The decision is
  invariant. Acceptance bar in `test_parity.py`: `input_tokens` exact, decision identical, numerics
  within `1e-3` abs. Do **not** assert exact 4-decimal equality across backends.
- **Determinism + cache:** results are deterministic for a given `(state, questions)`, so the
  content-addressed cache is safe. Repeat requests are served from memory (~6 ms).
- **CUDA lib path:** the CUDA EP silently falls back to CPU unless
  `LD_LIBRARY_PATH` includes `/usr/local/lib/ollama/cuda_v13` (see LINK.md). Verify via
  `/health` → `providers` must list `CUDAExecutionProvider` first.

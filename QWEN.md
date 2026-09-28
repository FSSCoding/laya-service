# Laya GPU Inference Service — QWEN.md

## What this is

A Python FastAPI service that runs the **Laya 421M-param ONNX model** on an RTX 3090 via `onnxruntime-gpu` (CUDA EP) and exposes it over HTTP on **port 11448**. Laya is a System-1 decision model — it takes a state (ticket, email, JSON) plus typed questions (`choice`, `score`, `noul`) and returns calibrated probabilities in a single forward pass. It does **not** generate text.

The service exists because the Node.js Laya library (`/MASTERFOLDER/Services/laya`) has no CUDA build — `onnxruntime-node` doesn't support GPU. This service wraps the same ONNX bundle in a Python runtime that does, mirroring the topology of `embed-server`.

**Performance:** ~15 ms warm single call (~20× faster than CPU at 297 ms). Model is resident (always loaded, ~2.3 GB VRAM idle). Batching coalesces up to 512 concurrent requests into one forward pass.

## Architecture

```
laya_server.py          FastAPI app, HTTP endpoints, stats, caching layer
src/worker.py           LayaWorker — holds resident ORT session + tokenizer, runs inference
src/sequence.py         Tokenizer/sequence building, softmax, question-type shaping
src/batcher.py          Request coalescing — merges concurrent calls into one forward pass
src/result_cache.py     Two-tier cache: in-memory LRU + SQLite disk (~/.laya-service/result_cache.db)
config/laya_config.json Runtime config (port, GPU providers, batcher, cache, idle-unload)
```

**Request flow:** HTTP → batcher (5 ms window, max 512) → LayaWorker (single forward pass) → result_cache (check/store) → HTTP response.

## Critical: CUDA library path

`onnxruntime-gpu==1.30.0` needs CUDA 13 libraries that are **not** on the default path. Every invocation requires:

```bash
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu
CUDA_VISIBLE_DEVICES=0
```

Without this, CUDA silently falls back to CPU. Verify with `GET /health` — `providers` must list `CUDAExecutionProvider` first.

## Commands

```bash
# Parity test (GPU vs CPU reference — must be decision-identical, not numerically exact)
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python test_parity.py

# Benchmark (single + batched)
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python bench.py

# Run server (foreground)
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python laya_server.py
```

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/system_one` | `{state, questions}` → `{model, answers, usage}` |
| `GET` | `/health` | Liveness + active providers + uptime |
| `GET` | `/v1/stats` | Request/latency/cache/provider stats |

## Key conventions

- **Parity bar:** GPU vs CPU probabilities differ by ~1–4e-4 (FP32 non-determinism). The **decision** (argmax/choice) must be invariant. Accept within `1e-3` abs — do NOT assert exact decimal equality.
- **Model bundle is read-only** from `/MASTERFOLDER/Services/laya/onnx/` — this service never writes to the laya repo.
- **Dedicated venv** at `.venv/` — Python 3.12, `onnxruntime-gpu==1.30.0`, `tokenizers`, `fastapi`, `uvicorn`.
- **Config-driven** — all tunables (port, providers, batcher size/wait, cache TTL) in `config/laya_config.json`.
- **systemd unit** in `systemd/` — not yet installed (requires sudo-gui). Resource caps: 8G RAM, 300% CPU, Nice -5.

## Documentation map

| File | Purpose |
|------|---------|
| `README.md` | Entry point, quick start, API reference, resource footprint |
| `LINK.md` | Deploy paths, env vars, CUDA gotcha, systemd install, ops notes |
| `SERVICE.md` | Architecture, components, HTTP contract, data flow, tuning knobs |
| `ROADMAP.md` | Planned work (CLI, client libs, primitives) |
| `CHANGELOG.md` | Dated change log |

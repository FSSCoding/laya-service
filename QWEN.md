# Laya GPU Inference Service — QWEN.md

## What this is

A Python FastAPI service that runs the Laya 421M-parameter ONNX model with `onnxruntime-gpu` on an RTX 3090 and serves it over HTTP on port 11448. Laya is a System-1 decision model: it takes a state (ticket, email, JSON) and typed questions (`choice`, `score`, `noul`), returning calibrated probabilities in one forward pass. It does not generate text.

The service is expected to grow into a multi-component project (including CLIs and MCPs). Its implementation belongs in the named Python package `laya_server` under `src/`; keep new application code inside this package rather than adding more root-level modules. The distribution name (`laya-service`) and console command (`laya-server`) use hyphens; Python imports use `laya_server` with an underscore.

## Architecture

```
src/laya_server/
  __init__.py     Package metadata and lazy public exports
  server.py       FastAPI application, HTTP endpoints, lifecycle, stats, cache wiring
  worker.py       Resident ONNX Runtime session + tokenizer and inference
  sequence.py     Tokenization/sequence building, softmax, question-type shaping
  batcher.py      Concurrent request coalescing into a forward pass
  result_cache.py In-memory LRU + SQLite result cache
config/laya_config.json Runtime configuration
```

Request flow: HTTP → batcher (5 ms window, up to 512 requests) → worker → result cache → HTTP response.

## Packaging and execution

- Setuptools discovers `laya_server*` from `src/` (`where = ["src"]`).
- `laya-server` is the installed console entry point for `laya_server.server:run_server`.
- Install with `.venv/bin/python -m pip install -e .` (all runtime deps included); add `,[dev]` for test/lint tools.
- `bench.py` at the repo root is a standalone benchmark script; package imports must use `laya_server.*`.
- Keep the model bundle read-only at `/MASTERFOLDER/Services/laya/onnx/`.

## Critical: CUDA library path

`onnxruntime-gpu==1.30.0` requires CUDA 13 libraries not on the default path. Run the service and GPU checks with:

```bash
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu
CUDA_VISIBLE_DEVICES=1
```

Without this, CUDA can silently fall back to CPU. Verify `GET /health`: `providers` must list `CUDAExecutionProvider` first.

## Commands

```bash
# Install server and development extras
.venv/bin/python -m pip install -e ".[server,dev]"

# Parity test
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pytest tests/

# Benchmark
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python bench.py

# Run the HTTP service
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/laya-server
```

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/system_one` | `{state, questions}` → `{model, answers, usage}` |
| `GET` | `/health` | Liveness, active providers, model state, uptime |
| `GET` | `/v1/stats` | Request, latency, cache, and provider statistics |
| `POST` | `/control/unload` | Unload the model from GPU memory |

## Key conventions

- **Parity:** GPU/CPU probabilities can differ by ~1–4e-4 due to FP32 nondeterminism. Decisions must match; numeric tolerance is `1e-3`. Do not assert exact decimal equality.
- **Configuration:** Runtime tunables live in `config/laya_config.json`; `LAYA_CONFIG_PATH` can override its path.
- **Virtual environment:** Use the project `.venv`; do not install dependencies globally.
- **Service deployment:** Unit source is under `systemd/`. Deployment/host-specific operations are documented in `LINK.md`.

## Documentation map

| File | Purpose |
|------|---------|
| `README.md` | Overview, quick start, API, and resource footprint |
| `LINK.md` | Paths, environment, CUDA details, service operation/deployment |
| `SERVICE.md` | Components, API contract, data flow, tuning |
| `ROADMAP.md` | Planned work |
| `CHANGELOG.md` | Dated change history |
| `QWEN.md` | This agent context |
| `pyproject.toml` | Package metadata, dependencies, console entry point, test/lint configuration |
| `tests/` | Pytest suite (parity, version drift) + shared fixtures |
| `systemd/` | systemd unit and restart timer source |
| `bench.py` | Standalone benchmark script |

## Runtime state and footprint

The ONNX model is loaded on demand and may be unloaded after configured idle time. The cache database is stored under `~/.laya-service/result_cache.db`. See README.md and LINK.md for measured resource use and operational details.

## Do not

- Write to the model bundle in `/MASTERFOLDER/Services/laya/onnx/`.
- Reintroduce `src` as the import package or add service modules at repository root.
- Claim CUDA is active without checking `/health` providers.
- Compare GPU/CPU floating-point results using exact equality.
- Commit structural changes without an explicit user instruction to commit.
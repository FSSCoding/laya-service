# Laya GPU Inference Service — Agent Context

Canonical how-it-runs document for agents working in this repo. Covers architecture, deploy
paths, environment, CUDA-13 gotcha, systemd ops, verification, and conventions. Read this first
when operating or modifying the service on bobai.

## What this is

A Python FastAPI service that runs the Laya 421M-parameter ONNX model with `onnxruntime-gpu` on
an RTX 3090 and serves it over HTTP on port 11448. Laya is a System-1 decision model: it takes a
state (ticket, email, JSON) and typed questions (`choice`, `score`, `noul`), returning calibrated
probabilities in one forward pass. It does not generate text.

The service is expected to grow into a multi-component project (including CLIs and MCPs). Its
implementation belongs in the named Python package `laya_server` under `src/`; keep new
application code inside this package rather than adding more root-level modules. The distribution
name (`laya-service`) and console command (`laya-server`) use hyphens; Python imports use
`laya_server` with an underscore.

## Identity

| Field | Value |
|-------|-------|
| Service | `laya-service` |
| Host | **bobai** (10.23.88.5) — for now; may move to another FSS server later |
| Port | **11448** (user-assigned 2026-09-28; verified free) |
| Interface | HTTP only (no Unix socket — not needed yet) |
| Model | Laya 421M-param ONNX (English), from `/MASTERFOLDER/Services/laya/onnx/` |
| Runtime | Python 3.12 + `onnxruntime-gpu==1.30.0` (CUDA EP) |
| GPU | One RTX 3090, pinned via `CUDA_VISIBLE_DEVICES=1` (card 1; card 0 hosts other tenants) |
| Venv | **Dedicated**: `/MASTERFOLDER/Services/laya-service/.venv` |

## Paths

| Thing | Path |
|-------|------|
| Service root | `/MASTERFOLDER/Services/laya-service/` |
| Python package | `/MASTERFOLDER/Services/laya-service/src/laya_server/` (`server.py`, `_config.py`, `worker.py`, `sequence.py`, `batcher.py`, `result_cache.py`, `primitives.py`, `client.py`, `cli.py`, `_state.py`, `__main__.py`) |
| Server entry point | `/MASTERFOLDER/Services/laya-service/.venv/bin/laya-server` (`laya_server.server:run_server`) |
| CLI entry point | `/MASTERFOLDER/Services/laya-service/.venv/bin/laya` (`laya_server.cli:cli`) |
| Config | `/MASTERFOLDER/Services/laya-service/config/laya_config.json` |
| Venv | `/MASTERFOLDER/Services/laya-service/.venv` |
| Model bundle (read-only) | `/MASTERFOLDER/Services/laya/onnx/` (`laya.onnx`, `laya.onnx.data`, `laya_config.json`, `tokenizer/`) |
| Result cache DB | `~/.laya-service/result_cache.db` (SQLite; created at runtime) |
| systemd unit (source) | `/MASTERFOLDER/Services/laya-service/systemd/laya-service.service` |
| systemd timer (source) | `/MASTERFOLDER/Services/laya-service/systemd/laya-service-restart.timer` |
| systemd unit (**installed**) | `/etc/systemd/system/laya-service.service` (system unit, `User=bob`) |
| systemd timer (**installed**) | `/etc/systemd/system/laya-service-restart.timer` |
| systemd drop-in override | `/etc/systemd/system/laya-service.service.d/override.conf` (re-pins `CUDA_VISIBLE_DEVICES=1`) |
| MCP server | `~/.qwen/mcp-servers/laya.js` (registered in `~/.qwen/settings.json`) |
| Gitea remote | `https://gitea.bobai.com.au/foxadmin/laya-service.git` |
| Tests | `/MASTERFOLDER/Services/laya-service/tests/` (pytest) |
| Benchmark | `/MASTERFOLDER/Services/laya-service/bench.py` |
| Research/design doc | `/MASTERFOLDER/Services/laya/docs/LAYA-GPU-SERVICE.md` |

> The model bundle is **not** owned by this repo — it lives in the `laya` library repo and is
> downloaded there. This service reads it read-only. If you move the service to another machine,
> the bundle must exist at `model_dir` (config) or be downloaded first.

## Architecture

```
src/laya_server/
  __init__.py       Package metadata + re-exports primitives and result dataclasses
  __main__.py       `python -m laya_server` → delegates to CLI
  server.py         FastAPI app factory (`create_app(cfg)`) + runner (`run_server()`)
  _config.py        Typed Pydantic config models with validation
  _state.py         `initialize_state(app, cfg)` — single source of truth for app state
  worker.py         `LayaWorker` — ONNX session + tokenizer; builds inputs; runs; shapes answers
  sequence.py       Pure model-free port: sequence layout, option rendering, softmax, rounding
  batcher.py        `RequestBatcher` — asyncio queue + 5 ms window coalescing
  result_cache.py   Two-tier cache (in-memory LRU + SQLite); keyed on `sha256(state, questions)`
  primitives.py     Typed primitives: laya_choice/score/noul/batch + classify/multi_select/rank
  client.py         Lightweight HTTP client (httpx) with explicit endpoint resolution
  cli.py            Click CLI with 14 subcommands; sole owner of systemd lifecycle
config/laya_config.json Runtime configuration
```

Request flow: HTTP → batcher (5 ms window, up to 512 requests) → worker → result cache → HTTP
response. Single process. The ONNX session loads once (pre-warmed at startup), is held while
busy, and is **unloaded when idle** (auto after `unload_after_s` or via `/control/unload`). On
the next request it re-loads lazily (~0.7 s cold start).

## Packaging and execution

- Setuptools discovers `laya_server*` from `src/` (`where = ["src"]`).
- `laya-server` is the installed console entry point for `laya_server.server:run_server`.
- `laya` is the installed console entry point for `laya_server.cli:cli`.
- Install with `.venv/bin/python -m pip install -e .` (all runtime deps included); add `,[dev]`
  for test/lint tools.
- `bench.py` at the repo root is a standalone benchmark script; package imports must use
  `laya_server.*`.
- Keep the model bundle read-only at `/MASTERFOLDER/Services/laya/onnx/`.

## Environment

### The CUDA 13 library path (CRITICAL)

`onnxruntime-gpu==1.30.0` is built against **CUDA 13.0** (ORT ≥1.27 defaults to CUDA 13). Its
cuBLAS / cuBLAS-Lt **13** libraries are **not** on the default `LD_LIBRARY_PATH`. On bobai they
live in Ollama's CUDA-13 stack:

- cuBLAS/cuBLAS-Lt 13 + cuDART 13: `/usr/local/lib/ollama/cuda_v13/`
- cuDNN 9.25: `/usr/lib/x86_64-linux-gnu`

**Without this, the CUDA EP silently falls back to CPU** with:
```
Failed to load library .../libonnxruntime_providers_cuda.so
  libcublasLt.so.13: cannot open shared object file: No such file or directory
```

So every invocation (and the systemd unit) sets:
```
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu
CUDA_VISIBLE_DEVICES=1
```

To verify CUDA is actually active after a change: `GET /health` → `providers` must list
`"CUDAExecutionProvider"` first. If it only shows `CPUExecutionProvider`, the lib path is wrong.

> If a future ORT upgrade moves to a CUDA 12.x build, point `LD_LIBRARY_PATH` at the matching
> cuBLAS 12 path instead (`ldconfig -p | grep cublasLt`). Don't guess.

### Other env (set by the systemd unit)

| Var | Value | Why |
|-----|-------|-----|
| `CUDA_VISIBLE_DEVICES` | `1` | Pin to one 3090. Card 1 is the pin as of 2026-09-28 (card 0 hosts Fss-RAG, fss-tts, etc.); switch to `0` if card 1 gets saturated. |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `2` | Prevent thread oversubscription (model is small; 2 threads is plenty). |
| `TOKENIZERS_PARALLELISM` | `false` | Avoid tokenizers warning + redundant threads. |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Inherited from embed-server; harmless (ORT doesn't use torch) — keep for consistency. |

### Application config resolution

The server reads **one** env var: `LAYA_CONFIG_PATH`. All other configuration comes from the JSON
config file. Resolution order (strict — no silent fallback):

1. `LAYA_CONFIG_PATH` env var, if set
2. `config/laya_config.json` relative to CWD, if it exists
3. Raise `FileNotFoundError`

The systemd unit sets `WorkingDirectory=/MASTERFOLDER/Services/laya-service`, so the CWD-relative
resolution finds `config/laya_config.json` without needing `LAYA_CONFIG_PATH`.

Do NOT set env vars like `LAYA_IDLE_UNLOAD_S` or `LAYA_PORT` — the server does not read them.
All runtime tunables are in the JSON config file.

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

# Run the HTTP service (foreground)
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/laya-server

# Or via `python -m`
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m laya_server
```

### systemd (production)

**Installed & active on bobai (2026-09-28).** The unit is a **system** unit at
`/etc/systemd/system/laya-service.service` (`User=bob`), with the weekly restart timer at
`/etc/systemd/system/laya-service-restart.timer` and a drop-in override at
`/etc/systemd/system/laya-service.service.d/override.conf` (re-pins `CUDA_VISIBLE_DEVICES=1`).
Journal confirms `systemd[1]` owns the process. Verify with `sudo-gui systemctl status
laya-service.service` and `curl -s http://127.0.0.1:11448/health`.

The unit runs the installed `laya-server` console command. The package is editable-installed in
the service venv (`.venv/bin/python -m pip install -e .`), so the running service tracks the
source tree.

**Re-install / update from source** (needs sudo — route through `sudo-gui`):

```bash
sudo-gui cp /MASTERFOLDER/Services/laya-service/systemd/laya-service.service /etc/systemd/system/laya-service.service
sudo-gui cp /MASTERFOLDER/Services/laya-service/systemd/laya-service-restart.timer /etc/systemd/system/laya-service-restart.timer
sudo-gui systemctl daemon-reload
sudo-gui systemctl enable --now laya-service.service
sudo-gui systemctl enable --now laya-service-restart.timer
```

Resource caps in the unit: `MemoryMax=8G`, `CPUQuota=300%`, `Nice=-5`, `Restart=on-failure`,
`RestartSec=5`. The weekly timer reclaims fragmented CUDA memory (mirrors embed-server).

**Single uvicorn worker.** The unit does not set `workers=` (defaults to 1). Each worker process
loads its own ONNX session (~2.3 GB VRAM). Running multiple workers would duplicate the VRAM
footprint. This is intentional for a single-card GPU service.

> **Port conflict:** if a foreground instance is already bound to 11448, stop it before enabling
> the unit (two processes can't share the port). Check with `ss -tlnp | grep 11448`.

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/system_one` | `{state, questions}` → `{model, answers, usage}` |
| `GET` | `/health` | Liveness, active providers, model state, uptime |
| `GET` | `/v1/stats` | Request, latency, cache, and provider statistics |
| `POST` | `/control/unload` | Unload the model from GPU memory |

Every answer's `rl_agent` field carries a routing signal envelope:
`{act_probability, escalate_probability, experimental: true}`. The `escalate_probability`
and `experimental` flag are new in v0.3.0. The `act_probability` key is backwards-compatible.

## Verification

```bash
# liveness + provider (must show CUDAExecutionProvider first)
curl -s http://127.0.0.1:11448/health

# stats (cache hits/misses, latency p50/p99, providers, uptime)
curl -s http://127.0.0.1:11448/v1/stats

# parity gate (decision-identical to CPU reference; run from the service dir)
cd /MASTERFOLDER/Services/laya-service
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pytest tests/ -v

# benchmark
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python bench.py
```

### Parity bar (correct across backends)

GPU vs CPU differ by ~1–4e-4 per probability (and ~6–7e-4 on the nonlinear `confidence`/`score`
aggregates) — standard CPU-vs-CUDA FP32 non-determinism. The **decision is invariant**.
Acceptance: `input_tokens` exact (267 on the fixture), same `choice`/argmax, numerics within
`1e-3` abs. Do NOT assert exact 4-decimal equality between CPU and GPU — that's not achievable
and isn't the goal.

## Key conventions

- **Parity:** GPU/CPU probabilities can differ by ~1–4e-4 due to FP32 nondeterminism. Decisions
  must match; numeric tolerance is `1e-3`. Do not assert exact decimal equality.
- **Configuration:** Runtime tunables live in `config/laya_config.json`; `LAYA_CONFIG_PATH` can
  override its path.
- **Virtual environment:** Use the project `.venv`; do not install dependencies globally.
- **Service deployment:** Unit source is under `systemd/`; installed as a system unit on bobai.
- **Adoption gate:** No operational Laya decision is adopted solely from general benchmark
  performance. Each production decision requires a representative domain-specific evaluation
  slice with pre-agreed acceptance criteria.

## Evaluation & benchmarking

**Current mission (Phase 3.5): Service Hardening & Operating Envelope.** Characterise and harden
the three native primitives (`choice`, `score`, `noul`), plus batching, coalescing, token limits,
concurrency, and error handling. Establish the mechanical operating envelope before operational use.

- Spec: `benchmarks/SPEC-HARDENING.md`
- Harness: `benchmarks/harness/` (runner, metrics, systems, report)
- Smoke corpus: `benchmarks/v1/cases.jsonl` (33 cases, frozen regression baseline)
- Deferred research: `benchmarks/FUTURE-METHODS.md` (300-case semantic corpus, inversion pairs,
  consistency perturbation suite, OOD taxonomy, 27B reference comparison — preserved but not
  current build requirements)

**What is NOT the current mission:** large-scale semantic evaluation, inversion experiments,
consistency perturbation suites, OOD taxonomies. Those are deferred. The immediate goal is to
know the operating limits, failure behaviour, and safe usage envelope of the actual primitives.

**After hardening:** Stop. Use it. Run actual inferencing. Form intuition through interaction.
Domain-specific adoption slices only when real use cases emerge.

## Documentation map

| File | Purpose |
|------|---------|
| `README.md` | Overview, quick start, API, and resource footprint |
| `SERVICE.md` | Components, API contract, data flow, tuning |
| `ROADMAP.md` | Planned work (Phase 3.5 = Service Hardening & Operating Envelope is current) |
| `CHANGELOG.md` | Dated change history |
| `QWEN.md` | This agent context (architecture, paths, environment, ops, conventions) |
| `pyproject.toml` | Package metadata, dependencies, console entry point, test/lint configuration |
| `tests/` | Pytest suite (parity, server contract, integration, CLI, primitives, config, version) + shared fixtures |
| `systemd/` | systemd unit and restart timer source |
| `bench.py` | Standalone benchmark script (CPU vs GPU latency) |
| `benchmarks/` | Evaluation harness: smoke corpus, quality metrics, systems benchmark, hardening spec |
| `benchmarks/SPEC-HARDENING.md` | Current mission: service hardening & operating envelope |
| `benchmarks/FUTURE-METHODS.md` | Deferred: large-scale semantic evaluation methodology |
| `benchmarks/README.md` | How to run the harness, adoption-gate policy |
| `MODEL-DOCTRINE.md` | Operating doctrine: mental model, safe-usage rules per primitive, question-formulation checklist |

## Runtime state and footprint

The ONNX model is loaded on demand and may be unloaded after configured idle time. The cache
database is stored under `~/.laya-service/result_cache.db`. See README.md for measured resource
use.

### Model lifecycle

With the current config (`idle.unload_after_s=300`, `idle.enabled=true`) the model **auto-unloads
after 300 s idle** and **lazily reloads (~0.7 s) on the next `/system_one`**. Set
`unload_after_s=0` / `idle.enabled=false` to keep it permanently resident. `/health` reports
`model_loaded` and `status: "degraded"` while unloaded (normal after idle — not a fault). Idle
cost while loaded is ~2.3 GB VRAM + ~736 MB RAM + ~0% GPU util.

### Cache

Keyed on `sha256(state, questions)` — deterministic, so repeat requests are served from memory
(~6 ms) instead of the GPU. Cache lives at `~/.laya-service/result_cache.db` + in-memory LRU.
To clear: stop the service and delete the DB (or set `cache.enabled=false` in config).

## Do not

- Write to the model bundle in `/MASTERFOLDER/Services/laya/onnx/`.
- Reintroduce `src` as the import package or add service modules at repository root.
- Claim CUDA is active without checking `/health` providers.
- Compare GPU/CPU floating-point results using exact equality.
- Commit structural changes without an explicit user instruction to commit.

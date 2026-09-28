# Changelog

All notable changes to the Laya GPU Inference Service. Dates are absolute (UTC, bobai).
Format follows the embed-server changelog convention.

---

## [Unreleased] — 2026-09-28 (architecture refactor)

### Added
- **`src/laya_server/_config.py`** — typed Pydantic config models with validation (port range,
  positive TTL, path expansion, model_dir existence check). Strict resolution: `LAYA_CONFIG_PATH`
  → CWD-relative `config/laya_config.json` → raise `FileNotFoundError`.
- **`src/laya_server/__main__.py`** — enables `python -m laya_server`.
- **`tests/fixtures.py`** — shared STATE/QUESTIONS test fixtures (single source of truth).
- **`tests/test_version.py`** — asserts `importlib.metadata.version` matches `__version__`.

### Changed
- **`server.py` rewritten** as an app factory (`create_app(cfg)`) + runner (`run_server()`).
  No module-level globals, no import-time side effects. All state on `app.state`.
- **`pyproject.toml`** — moved `fastapi`/`uvicorn`/`pydantic` from `[server]` extra to core
  dependencies. Entry point now calls `run_server()`. This is a service package: all runtime
  deps are installed by default.
- **`__init__.py`** — removed dead `__getattr__` lazy-export layer.
- **`bench.py`** — imports STATE/QUESTIONS from `tests/fixtures.py`.
- **LINK.md** — documented config resolution policy, single uvicorn worker, removed
  `LAYA_IDLE_UNLOAD_S` (was a daemon-only env var that didn't work).

### Removed
- **`laya-daemon`** bash script — replaced by systemd unit (production) and `laya-server`
  foreground command (dev). The `LAYA_IDLE_UNLOAD_S` env var bug is eliminated.
- **`test_parity.py`** (root script) — replaced by `pytest tests/` which covers the same checks.
- **`requirements.txt`** — superseded by `pyproject.toml` dependencies.

---

## [Initial build] — 2026-09-28

### Added
- **Core service** — FastAPI/uvicorn HTTP server on port **11448** wrapping Laya ONNX inference
  (`onnxruntime-gpu==1.30.0`, CUDA EP) in a single resident process.
- **`src/worker.py`** — `LayaWorker`: loads the ONNX session + tokenizer, builds inputs
  (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`), runs a single forward pass,
  applies per-cardinality temperature + softmax, and shapes answers to match the TS library exactly.
- **`src/sequence.py`** — pure Python port of the request rendering (port of `sequence.ts`): sequence
  layout, option rendering, `pyJsonDumps`, temperature buckets, confidence, softmax, `round4`.
- **`src/batcher.py`** — `RequestBatcher`: coalesces concurrent requests into one forward pass
  (asyncio queue + 5 ms window), dispatching the blocking call to the thread executor.
- **`src/result_cache.py`** — two-tier result cache (in-memory LRU + SQLite) keyed on
  `sha256(state, questions)`; repeat requests served from memory (~6 ms vs ~128 ms GPU).
- **`laya-daemon`** — start/stop/status/unload/restart CLI (PID file + idle-unload), mirroring the
  FSS Chatterbox `model_daemon.py` pattern.
- **Idle-unload** — the model auto-unloads from VRAM after `idle.unload_after_s` (default 300 s) and
  re-loads lazily on the next request; `POST /control/unload` + `laya-daemon unload` for explicit
  VRAM release. This makes the ~2.3 GB VRAM footprint on-demand rather than 24/7.
- **Endpoints** — `POST /system_one`, `GET /health` (providers, `model_loaded`, `idle_s`, uptime),
  `GET /v1/stats` (latency p50/p99, cache hit/miss, providers), `POST /control/unload`.
- **`test_parity.py`** — parity gate vs the CPU reference fixture (decision-invariant + `1e-3` tol).
- **`bench.py`** — CPU vs GPU latency benchmark (single + batched).
- **systemd** — `systemd/laya-service.service` (resource caps, CUDA-13 lib path) +
  `systemd/laya-service-restart.timer` (weekly CUDA-memory reclaim).
- **Docs** — README, LINK, SERVICE, CHANGELOG, ROADMAP.

### Verified (2026-09-28, RTX 3090, CUDA:0)
- Parity: decision-identical to the CPU reference; `input_tokens` exact (267); numerics within 1e-3.
- CUDA active: `/health` → `providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]`.
- **~20× speedup**: single warm call 297 ms (CPU) → 15 ms (GPU); 16-concurrent batch ~15 ms/request.
- Idle-unload/reload cycle proven: unload frees 2.3 GB VRAM; next request reloads (~0.5 s) and
  resumes CUDA inference.
- Daemon lifecycle: start/stop/status/unload all verified (port freed, VRAM released on stop).

### Not yet done
- systemd unit **not installed** (awaiting user's sudo; exact commands in LINK.md).
- CLI client + interface primitives (see ROADMAP).
- Git repo initialised locally; remote push pending user review.

## 2026-09-28 — Package layout

- Moved application modules into `src/laya_server/`, a named importable Python package.
- Moved the FastAPI application to `laya_server.server` and added the `laya-server` console entry point.
- Updated the daemon and systemd launchers, tests, operational scripts, and current documentation to use the package and entry point.
- Runtime configuration continues to resolve from the repository-level `config/` directory.

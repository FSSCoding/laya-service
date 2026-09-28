# Changelog

All notable changes to the Laya GPU Inference Service. Dates are absolute (UTC, bobai).
Format follows the embed-server changelog convention.

---

## [0.3.0] — 2026-09-29 (typed primitives layer)

### Added
- **`src/laya_server/primitives.py`** — typed primitives module: native wrappers
  (`laya_choice`, `laya_score`, `laya_noul`), heterogeneous batch (`laya_batch`), and
  composed helpers (`classify`, `multi_select`, `rank`). All return frozen dataclasses
  (`ChoiceResult`, `ScoreResult`, `NoulResult`, `BatchResult`, `RoutingSignal`).
- **Routing signal** — every answer now carries `rl_agent` with `act_probability`,
  `escalate_probability`, and `experimental: True`. The escalate head (P(Escalate)) is
  exposed under an unmistakable experimental flag.
- **CLI verbs** — `laya choice`, `laya score`, `laya noul`, `laya batch` with `--json`
  flag for structured output. Composed helpers are Python-only for now.
- **`tests/test_primitives.py`** — 22 new tests: native wrappers, batch, composed helpers,
  inversion-comparison pattern.
- **`rank` semantics** — independent score per item against a **shared caller-supplied
  rubric**. The item name is embedded into per-item instructions so every candidate is
  evaluated against an identical scale. Full per-level distributions preserved.

### Changed
- **`worker.py`** — `rl_agent` field changed from `{act_probability}` to a `routing_signal`
  envelope `{act_probability, escalate_probability, experimental}`. Backwards-compatible
  for consumers reading `answer["rl_agent"]["act_probability"]`.
- **`__init__.py`** — re-exports the public primitives and result dataclasses.
- **`cli.py`** — `_print_answers` updated for the new routing signal shape; text output
  now shows routing: act/escalate probabilities and (experimental) marker.
- **`pyproject.toml`** — version 0.2.0 → 0.3.0.

### Deferred (documented, not built)
- OpenAI/chat-shaped adapter with pseudo-model names — may be explored later for
  low-friction remote experimentation.
- Raw logits exposure.
- Pairwise ranking as alternative strategy for `rank`.
- Inversion helper (the pattern is documented and testable via the primitives).

---

## [0.2.0] — 2026-09-28 (next pass: client, CLI, MCP, tests)

### Added
- **`src/laya_server/_state.py`** — `initialize_state(app, cfg)` helper. Single source of
  truth for app state shape, used by both production lifespan and test fixtures.
- **`src/laya_server/client.py`** — lightweight HTTP client (httpx). Explicit endpoint
  resolution (`--endpoint` → `LAYA_ENDPOINT` → default). Retry on `ReadError`/`ReadTimeout`
  only; no retry on connection refused.
- **`src/laya_server/cli.py`** — Click CLI with 10 subcommands: serve, query, health, stats,
  unload, version, start, stop, restart, status. CLI is the sole owner of systemd lifecycle.
  `status --json` emits a frozen schema (tested). Exit codes: 0=healthy, 1=degraded,
  2=unreachable, 3=stopped.
- **`tests/test_server.py`** — 10 HTTP contract tests (mocked worker, no GPU, no lifespan).
  Verifies routing, validation, status codes, response shape.
- **`tests/test_integration.py`** — 2 near-end-to-end tests (real lifespan, real batcher,
  mocked worker). Catches "tests pass but app wiring broken."
- **`tests/test_cli.py`** — 7 CLI contract tests (version, status --json schema, health exit
  codes). Verifies the frozen JSON API.
- **`~/.qwen/mcp-servers/laya.js`** — JS MCP server (stdio). Direct HTTP via fetch. 4 tools:
  laya_system_one, laya_health, laya_stats, laya_unload. No lifecycle management. No Python
  dependency. Registered in `~/.qwen/settings.json`.
- **`httpx>=0.27`, `click>=8.0`** added to core dependencies.
- **`laya`** console entry point (`laya_server.cli:cli`).

### Changed
- **`server.py`** — lifespan now calls `initialize_state()` from `_state.py` instead of
  inline state construction. Removed duplicate `_make_process_payload` (moved to `_state.py`).
- **`__main__.py`** — now delegates to CLI (`click` group) instead of auto-starting the server.
  `python -m laya_server` shows help; use `python -m laya_server serve` to start.
- **`pyproject.toml`** — added httpx, click to deps; added `laya` entry point.

### Architecture notes
- The HTTP API is the single source of truth. CLI and MCP are independent frontends.
- MCP does NOT manage service lifecycle. CLI owns start/stop/restart/status.
- Text CLI output is unstable (human-friendly). JSON output (`--json`) is a frozen API.
- Batcher coalescing/concurrency is NOT covered by test_server.py (documented gap).

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

### Verified (2026-09-28, RTX 3090, GPU card 1)
- Parity: decision-identical to the CPU reference; `input_tokens` exact (267); numerics within 1e-3.
- CUDA active: `/health` → `providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]`.
- **~20× speedup**: single warm call 297 ms (CPU) → 15 ms (GPU); 16-concurrent batch ~15 ms/request.
- Idle-unload/reload cycle proven: unload frees 2.3 GB VRAM; next request reloads (~0.5 s) and
  resumes CUDA inference.
- Daemon lifecycle: start/stop/status/unload all verified (port freed, VRAM released on stop).

### Done (2026-09-28)
- **systemd unit installed & active** on bobai: `/etc/systemd/system/laya-service.service` +
  `laya-service-restart.timer`, with drop-in override at `laya-service.service.d/override.conf`
  re-pinning `CUDA_VISIBLE_DEVICES=1`. Journal confirms `systemd[1]` owns the process.
- **Repo pushed** to Gitea: `https://gitea.bobai.com.au/foxadmin/laya-service.git`
  (`master`, clean tree).

### Not yet done
- Interface primitives (typed `laya_choice`/`laya_score`/`laya_noul`, batch primitive) — ROADMAP.
- Remote deployment + ops hardening (alerting, log rotation, cache-DB disk) — ROADMAP.

## 2026-09-28 — Package layout

- Moved application modules into `src/laya_server/`, a named importable Python package.
- Moved the FastAPI application to `laya_server.server` and added the `laya-server` console entry point.
- Updated the daemon and systemd launchers, tests, operational scripts, and current documentation to use the package and entry point.
- Runtime configuration continues to resolve from the repository-level `config/` directory.
